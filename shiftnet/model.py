"""ShiftNet architecture: shift-estimator stem + MoE body + shift calibrator."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        # GroupNorm, not BatchNorm: batch statistics are meaningless under
        # per-sample random corruptions (known failure mode).
        g = max(1, c_out // 4)
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(g, c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ShiftEstimator(nn.Module):
    """Lightweight stem that estimates the input's shift signature.

    Outputs:
      s            (B, sig_dim)  disentangled shift signature embedding
      family_logits (B, n_families)  incl. 'clean' class (auxiliary supervision)
      severity     (B,) in [0, 1]    (auxiliary supervision)
    """

    def __init__(self, n_families=8, sig_dim=64, width=1.0):
        super().__init__()
        c = [int(v * width) for v in (16, 32, 64)]
        self.stem = nn.Sequential(
            ConvBlock(3, c[0], stride=2),   # 28 -> 14
            ConvBlock(c[0], c[1], stride=2),  # 14 -> 7
            ConvBlock(c[1], c[2], stride=2),  # 7 -> 4 (ceil)
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.sig = nn.Sequential(
            nn.Linear(c[2], sig_dim), nn.ReLU(inplace=True),
        )
        self.family_head = nn.Linear(sig_dim, n_families)
        self.severity_head = nn.Sequential(
            nn.Linear(sig_dim, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.stem(x)
        s = self.sig(h)
        return s, self.family_head(s), self.severity_head(s).squeeze(-1)


class ShiftMoEBody(nn.Module):
    """Shared backbone + K experts with shift-conditioned top-k routing.

    router_mode:
      'shift'      gate = W [s; f]   (the thesis)
      'feat'       gate = W f         (feature-only routing baseline)
      'shift_only' gate = W s         (shift-only ablation)
      'oracle'     gate = W [onehot(family); f]  (upper bound: perfect shift
                   knowledge; isolates whether shift-routing CAN help)
      'random'     uniform random top-2 (sanity floor)
    """

    def __init__(self, num_classes, n_experts=4, sig_dim=64,
                 feat_dim=128, width=1.0, router_mode="shift", top_k=2,
                 n_families=8):
        super().__init__()
        assert router_mode in ("shift", "feat", "shift_only", "oracle",
                               "random")
        self.router_mode = router_mode
        self.top_k = top_k
        self.n_experts = n_experts
        c = [int(v * width) for v in (32, 64, feat_dim)]
        feat_dim = c[2]  # actual dim after width scaling
        self.backbone = nn.Sequential(
            ConvBlock(3, c[0], stride=1),
            ConvBlock(c[0], c[1], stride=2),
            ConvBlock(c[1], c[2], stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        gate_in = {"shift": sig_dim + feat_dim, "feat": feat_dim,
                   "shift_only": sig_dim, "oracle": n_families + feat_dim,
                   "random": 0}[router_mode]
        self.gate = nn.Linear(gate_in, n_experts) if gate_in else None
        self.n_families = n_families
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(feat_dim, 64), nn.ReLU(inplace=True),
                          nn.Linear(64, num_classes))
            for _ in range(n_experts)
        ])

    def forward(self, x, s, fam_onehot=None):
        f = self.backbone(x)
        B = x.size(0)
        if self.router_mode == "random":
            idx = torch.stack([torch.randperm(self.n_experts, device=x.device)[:self.top_k]
                               for _ in range(B)])
            gates = torch.full((B, self.n_experts), 1.0 / self.top_k,
                               device=x.device)
            mask = torch.zeros_like(gates).scatter_(1, idx, 1.0)
            gates = gates * mask
        else:
            if self.router_mode == "oracle":
                assert fam_onehot is not None, "oracle needs family labels"
                rin = torch.cat([fam_onehot, f], dim=1)
            else:
                rin = {"shift": torch.cat([s, f], dim=1), "feat": f,
                       "shift_only": s}[self.router_mode]
            logits = self.gate(rin)
            gates = F.softmax(logits, dim=1)
            topv, topi = torch.topk(gates, self.top_k, dim=1)
            mask = torch.zeros_like(gates).scatter_(1, topi, 1.0)
            gates = (gates * mask)
            gates = gates / gates.sum(dim=1, keepdim=True).clamp_min(1e-9)
        out = torch.zeros(B, self.experts[0][-1].out_features, device=x.device)
        for k, expert in enumerate(self.experts):
            g = gates[:, k:k + 1]
            if g.abs().sum() > 0:
                out = out + g * expert(f)
        # Switch-style load-balancing aux loss
        with torch.no_grad():
            importance = gates.sum(0)
        aux = (importance / B)
        aux_loss = self.n_experts * (aux * aux).sum()
        return out, gates, aux_loss


class ShiftCalibrator(nn.Module):
    """Per-input temperature predicted from the shift signature."""

    def __init__(self, sig_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sig_dim, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, s):
        # temperature in ~[0.15, ~4]; 1.0 when net outputs 0
        return 0.15 + F.softplus(self.net(s)).squeeze(-1)


class ShiftNet(nn.Module):
    """Full model: estimator stem + MoE body + (optional) shift calibrator."""

    def __init__(self, num_classes, n_families=8, n_experts=4,
                 sig_dim=64, width=1.0, router_mode="shift",
                 use_calibrator=True, n_oracle_families=11):
        super().__init__()
        self.router_mode = router_mode
        self.n_oracle_families = n_oracle_families
        self.estimator = ShiftEstimator(n_families, sig_dim, width)
        self.body = ShiftMoEBody(num_classes, n_experts, sig_dim,
                                 width=width, router_mode=router_mode,
                                 n_families=n_oracle_families)
        self.calibrator = ShiftCalibrator(sig_dim) if use_calibrator else None

    def forward(self, x, fam=None):
        s, family_logits, severity = self.estimator(x)
        fam_onehot = None
        if self.router_mode == "oracle":
            assert fam is not None, "oracle_shift needs family labels"
            fam_onehot = F.one_hot(fam, self.n_oracle_families).float()
        logits, gates, aux_loss = self.body(x, s, fam_onehot)
        if self.calibrator is not None:
            T = self.calibrator(s).unsqueeze(-1)
            logits = logits / T
        else:
            T = None
        return {
            "logits": logits,
            "gates": gates,
            "aux_loss": aux_loss,
            "family_logits": family_logits,
            "severity": severity,
            "signature": s,
            "temperature": T,
        }
