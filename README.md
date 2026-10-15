[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](pyproject.toml)
[![GitHub stars](https://img.shields.io/github/stars/habib-analyst/shiftnet.svg)](https://github.com/habib-analyst/shiftnet)

**NEGATIVE RESULT, kept public on purpose: I built the first architecture with shift estimation as a first-class component — a lightweight estimator stem that tells the expert router *what* corruption it sees — then killed it when the decisive experiment showed even *perfect* shift information doesn't help routing.**

## The decisive experiment

TinyCNN backbone (verified: 0.54 balanced accuracy on clean derma) with a 4-expert MoE head, trained on corrupted derma (8 corruption families). Two conditions, identical backbone, data, and training — only the router input differed:

| Router input | Corrupted-val balanced accuracy |
|---|---|
| Oracle shift (ground-truth corruption family as one-hot) | 0.262 |
| Features only | **0.279** |
| Delta | −0.017 |

The pre-registered kill-gate required a clear margin for the architecture to survive. There is none. Even perfect shift information does not improve routing — **the features already encode the corruption**, so the router extracts whatever shift signal it needs from them directly. The estimator stem adds parameters and optimization difficulty without adding information.

**What was tried before killing:** BatchNorm→GroupNorm (BN statistics meaningless under per-sample corruptions); frozen validation assignments (eval-set drift bug); inverse-frequency class weights (derma's 67% majority class collapses without them — verified even a plain CNN); uniform family sampling (estimator collapsed to "clean" at p_clean=0.35); the oracle upper-bound test above.

**Not tested:** shift-calibrator alone for ECE improvement (the full architecture couldn't learn well enough to test this fairly).

Full writeup: `papers/shiftnet/PAPER.md`.

## Quickstart

The code still works and is worth reusing — the MoE body with 5 router modes (`shift`, `feat`, `shift_only`, `oracle`, `random`) is a clean thesis-isolating harness for routing research:

```bash
git clone https://github.com/habib-analyst/shiftnet.git
cd shiftnet
pip install -r requirements.txt   # torch, medmnist
```

```python
import torch
from shiftnet import ShiftNet, ShiftEstimator

m = ShiftNet(num_classes=7, n_families=8, n_experts=4, router_mode="feat")
out = m(torch.rand(16, 3, 28, 28))
print(out["logits"].shape, out["gates"].shape)
# -> torch.Size([16, 7]) torch.Size([16, 4])
```

Smoke run (verifies the pipeline, not learning — data downloads from MedMNIST):

```bash
python smoke.py
# -> SMOKE OK: pipeline works (loss decreases, resume OK, >=2 experts used)
```

Train one router condition:

```bash
python train.py --out runs/feat --epochs 8 --batch-size 64 --max-n 3000 --width 0.75 --router-mode feat --seed 0
```

## How it worked

`x → [ShiftEstimator] → s` (64-d shift signature, family logits, severity) → `[Backbone] → f` (128-d task features) → `gate = top2-softmax(W·[s; f])` → expert mixture; then `T = calibrator(s)` and `output = logits / T` (per-shift temperature). The thesis: an explicitly estimated shift signature should beat feature-only routing on held-out corruptions, and per-shift calibration should cut ECE. The 31-cell robocorrupt-style matrix (10 families × 3 severities + clean) was the arena. The oracle ablation showed the arena was lost before it started — features subsume the shift signal.

Full writeup: `papers/shiftnet/PAPER.md` (working paper, in preparation).

---
Built by [Habib Ur Rehman](https://github.com/habib-analyst) · ML/AI Engineer & healthcare-AI researcher ·
[LinkedIn](https://www.linkedin.com/in/hur-dev) · [habib.top](https://habib.top)
