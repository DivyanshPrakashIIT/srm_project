"""Code-level mirror of CONTRACTS.md (v1.0.0). Owner: A0. Change only via ADR."""
from __future__ import annotations

import numpy as np
import torch

CONTRACT_VERSION = "1.0.0"
BAND_ORDER = ("B2", "B3", "B4", "B8")  # Blue, Green, Red, NIR; index 0..3
NUM_BANDS = len(BAND_ORDER)
SCALE = 4
REFLECTANCE_SCALE = 10000.0
RHO_MIN, RHO_MAX = 0.0, 1.0

TRAIN_LR_SIZE, TRAIN_HR_SIZE = 32, 128
INFER_LR_SIZE, INFER_SR_SIZE = 64, 256
PRODUCT_BANDS = 5
PRODUCT_BAND_DESCRIPTIONS = BAND_ORDER + ("sigma_bar",)
MODEL_OUTPUT_KEYS = frozenset({"sr", "logvar"})


def dn_to_reflectance(dn: torch.Tensor) -> torch.Tensor:
    """C2: DN -> float32 reflectance, clipped to [0, 1]."""
    return (dn.float() / REFLECTANCE_SCALE).clamp(RHO_MIN, RHO_MAX)


def validate_reflectance(x: torch.Tensor, name: str = "tensor") -> None:
    if x.dtype != torch.float32:
        raise AssertionError(f"{name}: dtype {x.dtype} != float32")
    if x.ndim != 4 or x.shape[1] != NUM_BANDS:
        raise AssertionError(f"{name}: shape {tuple(x.shape)} is not (B, {NUM_BANDS}, H, W)")
    if not torch.isfinite(x).all():
        raise AssertionError(f"{name}: contains non-finite values")
    if x.min() < RHO_MIN or x.max() > RHO_MAX:
        raise AssertionError(f"{name}: values outside [{RHO_MIN}, {RHO_MAX}]")


def validate_pair(lr: torch.Tensor, hr: torch.Tensor) -> None:
    """C3: LR/HR pair must differ by exactly SCALE."""
    validate_reflectance(lr, "lr")
    validate_reflectance(hr, "hr")
    if lr.shape[0] != hr.shape[0]:
        raise AssertionError("batch size mismatch")
    if hr.shape[-2:] != (lr.shape[-2] * SCALE, lr.shape[-1] * SCALE):
        raise AssertionError(f"hr {tuple(hr.shape[-2:])} is not {SCALE}x lr {tuple(lr.shape[-2:])}")


def validate_model_output(out: dict, lr: torch.Tensor) -> None:
    """C4: model(lr) -> dict(sr=..., logvar=...)."""
    if not isinstance(out, dict) or set(out) != MODEL_OUTPUT_KEYS:
        raise AssertionError(f"model output must be dict with keys {sorted(MODEL_OUTPUT_KEYS)}")
    validate_pair(lr, out["sr"])
    lv = out["logvar"]
    if lv.shape != out["sr"].shape or lv.dtype != torch.float32:
        raise AssertionError("logvar must match sr in shape and be float32")
    if not torch.isfinite(lv).all():
        raise AssertionError("logvar contains non-finite values")


def sigma_bar(logvar: torch.Tensor) -> torch.Tensor:
    """C4: (B,4,H,W) log-variance -> (B,1,H,W) mean std over bands."""
    return torch.exp(0.5 * logvar).mean(dim=1, keepdim=True)


def assemble_product_array(sr: torch.Tensor, logvar: torch.Tensor) -> np.ndarray:
    """C5: single item (1,4,H,W) -> (5,H,W) float32 array [B2,B3,B4,B8,sigma_bar]."""
    if sr.shape[0] != 1:
        raise AssertionError("assemble_product_array expects batch size 1")
    prod = torch.cat([sr.clamp(RHO_MIN, RHO_MAX), sigma_bar(logvar)], dim=1)
    return prod[0].detach().cpu().numpy().astype(np.float32)
