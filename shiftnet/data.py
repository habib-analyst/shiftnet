"""MedMNIST loading + on-the-fly corruption sampling for shiftnet."""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from .corrupt import (corrupt_image, sample_train_corruption,
                      ALL_FAMILY_TO_IDX,
                      ALL_FAMILIES, SEVERITY_LEVELS)


def load_medmnist_npz(name="dermamnist"):
    path = os.path.expanduser(f"~/.medmnist/{name}.npz")
    d = np.load(path)
    return ((d["train_images"], d["train_labels"].ravel()),
            (d["val_images"], d["val_labels"].ravel()),
            (d["test_images"], d["test_labels"].ravel()))


class ShiftDataset(Dataset):
    """Applies sampled corruptions on the fly; returns shift targets too.

    mode 'train': random train-family corruption per sample (aux supervision).
    mode 'val':   FIXED per-index (family, severity) assignments drawn once
                  with the seed — the eval set does not drift between epochs.
    mode 'eval':  fixed (family, severity) cell — pass family/severity.
    """

    def __init__(self, images, labels, mode="train", family=None,
                 severity=None, seed=0, max_n=None):
        self.images = images[:max_n] if max_n else images
        self.labels = labels[:max_n] if max_n else labels
        self.mode = mode
        self.family = family
        self.severity = severity
        self.rng = np.random.default_rng(seed)
        if mode == "val":
            # freeze one corruption assignment per index
            self._assign = [sample_train_corruption(self.rng)
                            for _ in range(len(self.images))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = self.images[i]
        if self.mode == "train":
            fam, sev = sample_train_corruption(self.rng)
        elif self.mode == "val":
            fam, sev = self._assign[i]
        else:
            fam, sev = self.family, self.severity
        cimg = corrupt_image(img, fam, sev, self.rng)
        x = torch.from_numpy(cimg).permute(2, 0, 1).float() / 255.0
        return (x,
                torch.tensor(int(self.labels[i])),
                torch.tensor(ALL_FAMILY_TO_IDX[fam]),
                torch.tensor(float(sev), dtype=torch.float32))


def eval_cells():
    """Full robocorrupt-style matrix: (family, severity) cells + clean."""
    cells = [("clean", 0.0)]
    for fam in ALL_FAMILIES:
        if fam == "clean":
            continue
        for sev in SEVERITY_LEVELS:
            cells.append((fam, sev))
    return cells
