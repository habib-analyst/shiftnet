"""shiftnet: Shift-Routed Networks.

The first architecture with shift estimation as a first-class component.
A lightweight shift-estimator stem predicts each input's corruption/shift
signature (family + severity); that signature drives BOTH an expert router
and a per-shift calibrator.

    x -> [ShiftEstimator] -> s (shift signature, 64-d)
                            |-- family logits, severity (auxiliary outputs)
         [Backbone] -> f (task features, 128-d)
         gate = top2-softmax(W_r [s; f])      # routing conditioned on shift
         y_logits = sum_k gate_k * Expert_k(f)
         T = calibrator(s)                    # per-shift temperature
         output = y_logits / T
"""
from .model import ShiftEstimator, ShiftMoEBody, ShiftCalibrator, ShiftNet

__all__ = ["ShiftEstimator", "ShiftMoEBody", "ShiftCalibrator", "ShiftNet"]
__version__ = "0.1.0"
