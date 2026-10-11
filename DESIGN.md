# ShiftNet — design decisions and debugging log

## Architecture (final)
- **ShiftEstimator**: 3x ConvBlock(stride 2) + GAP -> 64-d signature `s`;
  family head (8 classes), severity head (sigmoid). Auxiliary-supervised.
- **Backbone**: 3x ConvBlock -> 128-d features `f` (width-scaled).
- **Router**: top-2 softmax over `W_r [s; f]` (shift), `W_r f` (feat),
  `W_r s` (shift_only), or uniform random. Switch-style balance aux loss.
- **Calibrator**: `T(s) = 0.15 + softplus(MLP(s))`; logits/T.
- **Loss**: class-weighted CE_task + 0.5*CE_fam + 0.2*MSE_sev + 0.01*L_bal.

## Hard-won fixes (2026-10-11)
1. **BatchNorm -> GroupNorm.** Under per-sample random corruptions, batch
   statistics are meaningless; BN running stats oscillated wildly and the
   estimator's family accuracy swung 0.11-0.56 between epochs on a FIXED
   eval set. GroupNorm (batch-independent) stabilized it. (robocorrupt-bench
   also uses no BN — plain Conv+ReLU.)
2. **Eval-set drift bug.** First `ShiftDataset` sampled fresh corruptions on
   every `__getitem__`, so the validation set changed between epochs.
   Added `mode="val"` with frozen per-index (family, severity) assignments.
3. **Class imbalance collapse.** Derma is 67% class 5. Without class
   weights, even a plain CNN collapses to majority prediction and NEVER
   escapes (verified: 5 epochs stuck at exactly 0.669). Inverse-frequency
   class weights in the task CE fix it. Primary metric: balanced accuracy
   (majority baseline = 1/7 = 0.143).
4. **Width-scaling bug.** Gate input dim used nominal feat_dim=128 but the
   backbone outputs int(128*width) channels. Fixed by deriving feat_dim
   from the actual backbone width.

## Corruption split
- Train (estimator supervised): clean, gaussian_noise, blur, brightness,
  contrast, fog, speckle, pixelate.
- Held-out (never seen): occlusion, hue_shift, camera_jitter.

## Kill-gate (pre-registered)
shift-routing must beat feature-only routing by a clear margin on
HELD-OUT corruption families. Else: KILLED, reported with evidence.

5. **Estimator majority-collapse (2026-10-11).** With p_clean=0.35, the
   estimator collapsed to always predicting "clean" (fam_acc 0.897 on the
   clean cell, 0.000 on every corrupted cell) — the same majority-trap as
   the task head. A collapsed estimator emits a constant signature, making
   shift-routing degenerate. Fix: uniform family sampling over all 8 train
   families. First pilot run invalidated by this; re-running.
