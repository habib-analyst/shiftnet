"""Tiny CPU smoke run: verifies the PIPELINE, not learning.

Checks: forward shapes, loss decrease, no NaNs, multiple experts get used,
checkpoint + DONE + --resume work. Learning is verified by the pilot.
"""
import json
import os
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs", "smoke")


def run(args):
    cmd = [sys.executable, os.path.join(HERE, "train.py")] + args
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
    if r.returncode != 0:
        print("STDOUT:", r.stdout[-2000:])
        print("STDERR:", r.stderr[-3000:])
    return r.returncode


def main():
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    # 1 epoch fresh
    assert run(["--out", OUT, "--epochs", "1", "--batch-size", "64",
                "--max-n", "800", "--width", "0.5", "--eval-every", "1",
                "--seed", "0"]) == 0, "smoke training failed"
    assert os.path.exists(os.path.join(OUT, "ckpt.pt")), "no checkpoint"
    # resume for 1 more epoch -> tests --resume path
    assert run(["--out", OUT, "--epochs", "2", "--batch-size", "64",
                "--max-n", "800", "--width", "0.5", "--eval-every", "1",
                "--seed", "0", "--resume"]) == 0, "resume failed"
    assert os.path.exists(os.path.join(OUT, "DONE")), "no DONE marker"
    assert os.path.exists(os.path.join(OUT, "model.pt")), "no model.pt"

    recs = [json.loads(l) for l in
            open(os.path.join(OUT, "train_log.jsonl"))]
    assert len(recs) == 2, f"expected 2 epochs logged, got {len(recs)}"
    assert recs[1]["epoch"] == 1, "resume did not continue epoch count"
    assert recs[1]["train_loss"] < recs[0]["train_loss"], \
        "loss did not decrease"
    assert recs[1]["train_loss"] == recs[1]["train_loss"], "NaN loss"

    # expert usage: load model, check >1 expert receives gate mass
    sys.path.insert(0, HERE)
    from shiftnet.model import ShiftNet
    from shiftnet.corrupt import N_TRAIN_FAMILIES
    m = ShiftNet(7, n_families=N_TRAIN_FAMILIES, width=0.5)
    m.load_state_dict(torch.load(os.path.join(OUT, "model.pt"),
                                 map_location="cpu", weights_only=True))
    m.eval()
    with torch.no_grad():
        out = m(torch.rand(32, 3, 28, 28))
    assert out["logits"].shape == (32, 7), out["logits"].shape
    assert out["family_logits"].shape == (32, N_TRAIN_FAMILIES)
    assert ((out["severity"] >= 0) & (out["severity"] <= 1)).all()
    used = (out["gates"].sum(0) > 0.5).sum().item()
    print(f"experts with gate mass: {used}/{out['gates'].size(1)}")
    assert used >= 2, "router collapsed to a single expert"
    print(f"SMOKE OK: pipeline works "
          f"(loss {recs[0]['train_loss']:.3f} -> {recs[1]['train_loss']:.3f}, "
          f"resume OK, {used} experts used)")


if __name__ == "__main__":
    main()
