"""Full corruption-matrix evaluation for ShiftNet (robocorrupt-style).

Evaluates every (family, severity) cell + clean on the test split,
reporting accuracy, ECE, and estimator family accuracy per cell.
Held-out families (occlusion, hue_shift, camera_jitter) are the
generalization test. Resumable: skips cells already in results.jsonl.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from shiftnet.model import ShiftNet
from shiftnet.data import load_medmnist_npz, ShiftDataset, eval_cells
from shiftnet.corrupt import N_TRAIN_FAMILIES, N_ALL_FAMILIES, HELDOUT_FAMILIES
from train import ece


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True,
                   help="training run dir containing model.pt")
    p.add_argument("--dataset", default="dermamnist")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=float, default=1.0)
    p.add_argument("--n-experts", type=int, default=4)
    p.add_argument("--router-mode", default="shift")
    p.add_argument("--no-calibrator", action="store_true")
    return p.parse_args()


@torch.no_grad()
def eval_cell(model, loader, device):
    model.eval()
    tot, correct, fam_correct = 0, 0, 0
    probs, labels = [], []
    for x, y, fam, sev in loader:
        x, y, fam = x.to(device), y.to(device), fam.to(device)
        out = model(x, fam)
        pred = out["logits"].argmax(1)
        correct += (pred == y).sum().item()
        fam_correct += (out["family_logits"].argmax(1) == fam).sum().item()
        tot += y.size(0)
        probs.append(F.softmax(out["logits"], 1).cpu())
        labels.append(y.cpu())
    probs = torch.cat(probs)
    labels = torch.cat(labels)
    return {"acc": correct / tot, "fam_acc": fam_correct / tot,
            "ece": ece(probs, labels)}


def main():
    args = parse_args()
    device = torch.device("cpu")
    torch.manual_seed(args.seed)
    _, _, (te_i, te_l) = load_medmnist_npz(args.dataset)
    n_classes = int(te_l.max()) + 1
    model = ShiftNet(n_classes, n_families=N_TRAIN_FAMILIES,
                     n_oracle_families=N_ALL_FAMILIES,
                     n_experts=args.n_experts, width=args.width,
                     router_mode=args.router_mode,
                     use_calibrator=not args.no_calibrator).to(device)
    state = torch.load(os.path.join(args.run_dir, "model.pt"),
                       map_location=device, weights_only=True)
    model.load_state_dict(state)

    out_path = os.path.join(args.run_dir, "matrix_results.jsonl")
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path):
            r = json.loads(l)
            done.add((r["family"], r["severity"]))

    logf = open(out_path, "a")
    for fam, sev in eval_cells():
        if (fam, sev) in done:
            continue
        ds = ShiftDataset(te_i, te_l, mode="eval", family=fam,
                          severity=sev, seed=args.seed)
        ld = DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)
        m = eval_cell(model, ld, device)
        rec = {"family": fam, "severity": sev,
               "held_out": fam in HELDOUT_FAMILIES, **m}
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        print(json.dumps(rec), flush=True)
    logf.close()

    # summary
    recs = [json.loads(l) for l in open(out_path)]
    for group, filt in [("all", lambda r: True),
                        ("train_fam", lambda r: not r["held_out"]
                         and r["family"] != "clean"),
                        ("held_out", lambda r: r["held_out"])]:
        sub = [r for r in recs if filt(r)]
        acc = float(np.mean([r["acc"] for r in sub]))
        ec = float(np.mean([r["ece"] for r in sub]))
        print(f"{group}: mean_acc={acc:.4f} mean_ece={ec:.4f} "
              f"n={len(sub)}", flush=True)
    print("MATRIX DONE", flush=True)


if __name__ == "__main__":
    main()
