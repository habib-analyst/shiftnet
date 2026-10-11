"""Joint training for ShiftNet. Resumable: checkpoints + --resume + DONE marker."""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from shiftnet.model import ShiftNet
from shiftnet.data import load_medmnist_npz, ShiftDataset, eval_cells
from shiftnet.corrupt import N_TRAIN_FAMILIES, N_ALL_FAMILIES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--dataset", default="dermamnist")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--width", type=float, default=1.0)
    p.add_argument("--n-experts", type=int, default=4)
    p.add_argument("--router-mode", default="shift",
                   choices=["shift", "feat", "shift_only", "oracle", "random"])
    p.add_argument("--no-calibrator", action="store_true")
    p.add_argument("--w-fam", type=float, default=0.5)
    p.add_argument("--w-sev", type=float, default=0.2)
    p.add_argument("--w-aux", type=float, default=0.01)
    p.add_argument("--max-n", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--eval-every", type=int, default=2)
    return p.parse_args()


def ece(probs, labels, n_bins=15):
    conf, pred = probs.max(1)
    acc = (pred == labels)
    edges = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    e = torch.zeros((), device=probs.device)
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() > 0:
            e = e + (m.float().mean()
                     * (acc[m].float().mean() - conf[m].mean()).abs())
    return e.item()


@torch.no_grad()
def evaluate(model, loader, device, n_classes):
    model.eval()
    tot, correct, loss_sum, fam_correct = 0, 0, 0.0, 0
    all_probs, all_labels = [], []
    class_correct = torch.zeros(n_classes)
    class_tot = torch.zeros(n_classes)
    for x, y, fam, sev in loader:
        x, y, fam = x.to(device), y.to(device), fam.to(device)
        out = model(x, fam)
        loss_sum += F.cross_entropy(out["logits"], y,
                                    reduction="sum").item()
        pred = out["logits"].argmax(1)
        correct += (pred == y).sum().item()
        fam_correct += (out["family_logits"].argmax(1) == fam).sum().item()
        for c in range(n_classes):
            mc = (y == c)
            class_correct[c] += (pred[mc] == c).sum().item()
            class_tot[c] += mc.sum().item()
        tot += y.size(0)
        all_probs.append(F.softmax(out["logits"], 1).cpu())
        all_labels.append(y.cpu())
    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    bal_acc = (class_correct / class_tot.clamp_min(1)).mean().item()
    return {"acc": correct / tot, "bal_acc": bal_acc,
            "loss": loss_sum / tot,
            "fam_acc": fam_correct / tot, "ece": ece(probs, labels)}


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, "ckpt.pt")
    done_marker = os.path.join(args.out, "DONE")
    log_path = os.path.join(args.out, "train_log.jsonl")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")

    (tr_i, tr_l), (va_i, va_l), _ = load_medmnist_npz(args.dataset)
    n_classes = int(tr_l.max()) + 1
    # class weights: derma is 67% majority class; without this the model
    # collapses to majority prediction and never learns (verified 2026-10-11)
    cnt = np.bincount(tr_l, minlength=n_classes).astype(float)
    class_w = torch.tensor(cnt.sum() / (n_classes * cnt),
                           dtype=torch.float32)
    train_ds = ShiftDataset(tr_i, tr_l, mode="train", seed=args.seed,
                            max_n=args.max_n or None)
    val_ds = ShiftDataset(va_i, va_l, mode="val", seed=999,
                          max_n=args.max_n or None)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=512, shuffle=False,
                        num_workers=0)

    model = ShiftNet(n_classes, n_families=N_TRAIN_FAMILIES,
                     n_oracle_families=N_ALL_FAMILIES,
                     n_experts=args.n_experts, width=args.width,
                     router_mode=args.router_mode,
                     use_calibrator=not args.no_calibrator).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_epoch = 0
    if args.resume and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_epoch = state["epoch"] + 1
        print(f"resumed from epoch {start_epoch}", flush=True)

    logf = open(log_path, "a" if args.resume else "w")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        tloss = 0.0
        n = 0
        for x, y, fam, sev in train_ld:
            x, y = x.to(device), y.to(device)
            fam, sev = fam.to(device), sev.to(device)
            out = model(x, fam)
            loss = (F.cross_entropy(out["logits"], y, weight=class_w)
                    + args.w_fam * F.cross_entropy(out["family_logits"], fam)
                    + args.w_sev * F.mse_loss(out["severity"], sev)
                    + args.w_aux * out["aux_loss"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tloss += loss.item() * y.size(0)
            n += y.size(0)
        rec = {"epoch": epoch, "train_loss": tloss / n,
               "time": time.time()}
        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            rec.update({"val_" + k: v for k, v in
                        evaluate(model, val_ld, device,
                                 n_classes).items()})
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch, "args": vars(args)}, ckpt)
        print(json.dumps(rec), flush=True)
    logf.close()
    torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
    open(done_marker, "w").write("done\n")
    print("TRAINING DONE", flush=True)


if __name__ == "__main__":
    main()
