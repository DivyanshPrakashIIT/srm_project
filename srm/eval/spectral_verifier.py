"""
Spectral Verification Module for Sentinel-2 Super-Resolution Mapping (SRM).

Verifies physical spectral conservation and index stability (NDVI/NDWI) when
spatially aggregating 2.5m super-resolved imagery back to 10m low-resolution grids.
"""

from typing import Dict, List, Union
import torch
import torch.nn.functional as F


class SpectralConservationVerifier:
    """
    Verifies physical band reflectance conservation and NDVI/NDWI accuracy.

    Band Index Mapping for 4-channel input (B2, B3, B4, B8):
        - Index 0: B2 (Blue)
        - Index 1: B3 (Green)
        - Index 2: B4 (Red)
        - Index 3: B8 (NIR)
    """

    def __init__(
        self,
        red_idx: int = 2,
        nir_idx: int = 3,
        green_idx: int = 1,
        eps: float = 1e-8,
    ) -> None:
        self.red_idx = red_idx
        self.nir_idx = nir_idx
        self.green_idx = green_idx
        self.eps = eps

    def compute_ndvi(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Computes Normalized Difference Vegetation Index (NDVI):
            NDVI = (B8 - B4) / (B8 + B4 + 1e-8)

        Args:
            tensor: Image tensor of shape (B, 4, H, W) or (4, H, W).

        Returns:
            NDVI map of shape (B, H, W) bounded strictly in [-1.0, 1.0].
        """
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        red = tensor[:, self.red_idx, :, :]
        nir = tensor[:, self.nir_idx, :, :]

        ndvi = (nir - red) / (nir + red + self.eps)
        return torch.clamp(ndvi, min=-1.0, max=1.0)

    def compute_ndwi(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Computes Normalized Difference Water Index (NDWI):
            NDWI = (B3 - B8) / (B3 + B8 + 1e-8)

        Args:
            tensor: Image tensor of shape (B, 4, H, W) or (4, H, W).

        Returns:
            NDWI map of shape (B, H, W) bounded strictly in [-1.0, 1.0].
        """
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        green = tensor[:, self.green_idx, :, :]
        nir = tensor[:, self.nir_idx, :, :]

        ndwi = (green - nir) / (green + nir + self.eps)
        return torch.clamp(ndwi, min=-1.0, max=1.0)

    def downsample_sr(self, sr_tensor: torch.Tensor, scale_factor: int = 4) -> torch.Tensor:
        """Spatially aggregates SR output back to LR spatial grid via average pooling."""
        return F.avg_pool2d(sr_tensor, kernel_size=scale_factor, stride=scale_factor)

    def verify_conservation(
        self,
        sr_tensor: torch.Tensor,
        lr_tensor: torch.Tensor,
        scale_factor: int = 4,
        ndvi_threshold: float = 0.02,
    ) -> Dict[str, Union[float, bool, List[float]]]:
        """
        Verifies spectral conservation between spatial aggregation of SR output and LR input.

        Args:
            sr_tensor: High-resolution SR output tensor (B, 4, H_sr, W_sr).
            lr_tensor: Low-resolution input patch tensor (B, 4, H_lr, W_lr).
            scale_factor: Spatial super-resolution scale factor (default: 4).
            ndvi_threshold: Contract limit for mean absolute delta NDVI (default: 0.02).

        Returns:
            Dict containing delta_ndvi, is_ndvi_conserved, mean_spectral_mae, band_maes, and passed.
        """
        if sr_tensor.ndim == 3:
            sr_tensor = sr_tensor.unsqueeze(0)
        if lr_tensor.ndim == 3:
            lr_tensor = lr_tensor.unsqueeze(0)

        sr_downsampled = self.downsample_sr(sr_tensor, scale_factor=scale_factor)

        if sr_downsampled.shape != lr_tensor.shape:
            raise ValueError(
                f"Shape mismatch after downsampling SR {sr_tensor.shape} with factor {scale_factor}: "
                f"got {sr_downsampled.shape}, expected {lr_tensor.shape}"
            )

        # Compute NDVI values on aggregated SR vs LR
        ndvi_sr_down = self.compute_ndvi(sr_downsampled)
        ndvi_lr = self.compute_ndvi(lr_tensor)

        delta_ndvi = torch.mean(torch.abs(ndvi_sr_down - ndvi_lr)).item()
        is_ndvi_conserved = delta_ndvi < ndvi_threshold

        # Reflectance conservation metrics
        band_diffs = torch.abs(sr_downsampled - lr_tensor)
        band_maes = torch.mean(band_diffs, dim=(0, 2, 3)).tolist()
        mean_spectral_mae = torch.mean(band_diffs).item()

        return {
            "delta_ndvi": float(delta_ndvi),
            "is_ndvi_conserved": bool(is_ndvi_conserved),
            "mean_spectral_mae": float(mean_spectral_mae),
            "band_maes": [float(b) for b in band_maes],
            "passed": bool(is_ndvi_conserved),
        }