import math
from typing import Dict, Union
import torch
import torch.nn.functional as F


def _gaussian_window(window_size: int = 11, sigma: float = 1.5, channels: int = 4) -> torch.Tensor:
    """Generates a 2D Gaussian kernel for multi-channel SSIM computation."""
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window_2d = g.unsqueeze(1) @ g.unsqueeze(0)
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


def compute_psnr(
    sr: torch.Tensor, hr: torch.Tensor, max_val: float = 1.0, max_clip: float = 100.0
) -> Dict[str, float]:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) overall and per-band [B2, B3, B4, B8].

    Args:
        sr: Super-resolved tensor (B, 4, H, W) in range [0, 1].
        hr: High-resolution ground truth tensor (B, 4, H, W) in range [0, 1].
        max_val: Peak value range of input images (default 1.0).
        max_clip: Cap value for perfect reconstruction (MSE = 0).

    Returns:
        Dict with "psnr" (mean across bands) and per-band keys ("psnr_b2", "psnr_b3", etc.)
    """
    band_names = ["psnr_b2", "psnr_b3", "psnr_b4", "psnr_b8"]
    band_psnrs = []
    results = {}

    for c, name in enumerate(band_names):
        mse = F.mse_loss(sr[:, c], hr[:, c]).item()
        if mse < 1e-10:
            psnr_val = max_clip
        else:
            psnr_val = 10.0 * math.log10((max_val**2) / mse)
        results[name] = float(psnr_val)
        band_psnrs.append(psnr_val)

    results["psnr"] = float(sum(band_psnrs) / len(band_psnrs))
    return results


def compute_ssim(
    sr: torch.Tensor,
    hr: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    max_val: float = 1.0,
) -> float:
    """
    Computes Structural Similarity Index Measure (SSIM) across channels and batch.

    Args:
        sr: Super-resolved tensor (B, 4, H, W) in range [0, 1].
        hr: High-resolution ground truth tensor (B, 4, H, W) in range [0, 1].

    Returns:
        Mean SSIM scalar float value in range [-1.0, 1.0].
    """
    channels = sr.shape[1]
    window = _gaussian_window(window_size, sigma, channels).to(device=sr.device, dtype=sr.dtype)

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    mu1 = F.conv2d(sr, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(hr, window, padding=window_size // 2, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(sr * sr, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(hr * hr, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(sr * hr, window, padding=window_size // 2, groups=channels) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    return float(ssim_map.mean().item())


def compute_sam(sr: torch.Tensor, hr: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Computes Spectral Angle Mapper (SAM) in degrees between spectral vectors.

    Args:
        sr: Super-resolved tensor (B, 4, H, W) in range [0, 1].
        hr: High-resolution ground truth tensor (B, 4, H, W) in range [0, 1].
        eps: Small epsilon for denominator clamping.

    Returns:
        Mean SAM angle in degrees.
    """
    # Upcast to float64 to prevent float32 dot/norm rounding discrepancies
    sr_64 = sr.double()
    hr_64 = hr.double()

    dot_product = torch.sum(sr_64 * hr_64, dim=1)
    sr_norm = torch.norm(sr_64, p=2, dim=1)
    hr_norm = torch.norm(hr_64, p=2, dim=1)

    # Use clamp on denominator rather than additive offset
    denom = (sr_norm * hr_norm).clamp(min=eps)
    cos_theta = torch.clamp(dot_product / denom, -1.0, 1.0)

    angle_rad = torch.acos(cos_theta)
    angle_deg = torch.rad2deg(angle_rad)

    return float(angle_deg.mean().item())


def compute_ergas(
    sr: torch.Tensor, hr: torch.Tensor, scale_factor: float = 4.0, eps: float = 1e-8
) -> float:
    """
    Computes Relative Global-Dimensional Synthesis Error (ERGAS).

    Args:
        sr: Super-resolved tensor (B, 4, H, W).
        hr: High-resolution ground truth tensor (B, 4, H, W).
        scale_factor: Spatial resolution ratio S = 4.
        eps: Small threshold to prevent division by zero.

    Returns:
        ERGAS score scalar float.
    """
    channels = sr.shape[1]
    sum_ratio_sq = 0.0

    for c in range(channels):
        mse_c = F.mse_loss(sr[:, c], hr[:, c])
        rmse_c = torch.sqrt(torch.clamp(mse_c, min=0.0)).item()
        mean_hr_c = torch.mean(hr[:, c]).item()
        denom = mean_hr_c if abs(mean_hr_c) > eps else eps
        sum_ratio_sq += (rmse_c / denom) ** 2

    ergas = 100.0 * (1.0 / scale_factor) * math.sqrt(sum_ratio_sq / channels)
    return float(ergas)

def compute_all_metrics(
    sr: torch.Tensor, hr: torch.Tensor, scale_factor: float = 4.0
) -> Dict[str, float]:
    """
    Evaluates full suite of spatial and spectral metrics across output patches.

    Args:
        sr: Super-resolved tensor (B, 4, H, W) in range [0, 1].
        hr: Ground truth HR tensor (B, 4, H, W) in range [0, 1].
        scale_factor: Spatial super-resolution scale factor (default 4.0).

    Returns:
        Dictionary containing PSNR (overall & per-band), SSIM, SAM, and ERGAS.
    """
    if sr.shape != hr.shape:
        raise ValueError(f"Shape mismatch: sr shape {sr.shape} != hr shape {hr.shape}")

    metrics = compute_psnr(sr, hr)
    metrics["ssim"] = compute_ssim(sr, hr)
    metrics["sam"] = compute_sam(sr, hr)
    metrics["ergas"] = compute_ergas(sr, hr, scale_factor=scale_factor)

    return metrics