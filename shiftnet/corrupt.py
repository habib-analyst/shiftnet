"""Corruption families: reuses robocorrupt-bench's parameterized corruptions.

Train families (estimator gets auxiliary supervision on these):
    gaussian_noise, blur, brightness, contrast, fog, speckle, pixelate + clean
Held-out families (NEVER seen during training — the generalization test):
    occlusion, hue_shift, camera_jitter
"""
import sys
import numpy as np

sys.path.insert(0, "/home/hatch/workspace/goals/github-portfolio-linkedin-presence-buildout/build/robocorrupt-bench")
from robocorrupt.corruptions import CORRUPTIONS  # noqa: E402

TRAIN_FAMILIES = ["clean", "gaussian_noise", "blur", "brightness",
                  "contrast", "fog", "speckle", "pixelate"]
HELDOUT_FAMILIES = ["occlusion", "hue_shift", "camera_jitter"]
ALL_FAMILIES = TRAIN_FAMILIES + HELDOUT_FAMILIES

FAMILY_TO_IDX = {name: i for i, name in enumerate(TRAIN_FAMILIES)}
N_TRAIN_FAMILIES = len(TRAIN_FAMILIES)
ALL_FAMILY_TO_IDX = {name: i for i, name in enumerate(ALL_FAMILIES)}
N_ALL_FAMILIES = len(ALL_FAMILIES)

SEVERITY_LEVELS = [0.33, 0.66, 1.0]


def corrupt_image(img, family, severity, rng):
    """img: uint8 HWC. family in ALL_FAMILIES. severity in [0, 1]."""
    if family == "clean":
        return img
    return CORRUPTIONS[family](img, float(severity), rng)


def sample_train_corruption(rng):
    """Sample (family, severity) for training, UNIFORM over families.

    Uniform (not p_clean=0.35): the estimator collapses to predicting
    "clean" (the plurality class) otherwise — verified 2026-10-11.
    severity ~ U[0.2, 1.0]; clean has severity 0.
    """
    fam = TRAIN_FAMILIES[rng.integers(0, len(TRAIN_FAMILIES))]
    sev = 0.0 if fam == "clean" else float(rng.uniform(0.2, 1.0))
    return fam, sev
