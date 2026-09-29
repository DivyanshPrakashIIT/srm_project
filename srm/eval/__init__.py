"""
Evaluation and Verification Package for Sentinel-2 Super-Resolution Mapping.
"""

from srm.eval.downstream import DownstreamEvaluator
from srm.eval.evaluator import SRMEvaluator
from srm.eval.metrics import (
    compute_all_metrics,
    compute_ergas,
    compute_psnr,
    compute_sam,
    compute_ssim,
)
from srm.eval.spectral_verifier import SpectralConservationVerifier

__all__ = [
    "compute_all_metrics",
    "compute_psnr",
    "compute_ssim",
    "compute_sam",
    "compute_ergas",
    "SRMEvaluator",
    "DownstreamEvaluator",
    "SpectralConservationVerifier",
]