"""
Sentinel-2 Super-Resolution Mapping (SRM) — Phase 1
Data Engineering & Preprocessing Pipeline

Implements:
  - DatasetConfig
  - RandomDegrade (randomized isotropic blur -> area-based downsample -> sensor noise)
  - Sentinel2SISRDataset (tiling, reflectance normalization, geospatial augmentation)
  - build_dataloader

Bands: B2 (Blue), B3 (Green), B4 (Red), B8 (NIR) — Sentinel-2 L2A, 10m GSD.
Reflectance scaling: rho = DN / 10000.0
Scale factor: 5x (50m synthetic LR -> 10m real HR during pre-training;
               10m real Sentinel-2 -> 2m SR product at deployment).
Tile size: 256x256 (HR / 10m grid).

Requires: torch, numpy, rasterio
"""

import os
import random
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Global constants
# ----------------------------------------------------------------------------
REFLECTANCE_SCALE = 10000.0   # Sentinel-2 L2A UINT16 -> reflectance
TILE_SIZE = 256                # HR (10m) tile size in pixels
DEGRADE_FACTOR = 5             # 10m -> 50m during pretraining; 10m -> 2m at deployment
SPECTRAL_ORDER = ("B2", "B3", "B4", "B8")  # Blue, Green, Red, NIR


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
@dataclass
class DatasetConfig:
    hr_dir: str                                    # directory of 10m, 4-band GeoTIFF tiles
    tile_size: int = TILE_SIZE
    degrade_factor: int = DEGRADE_FACTOR
    augment: bool = True
    spectral_shuffle_prob: float = 0.15
    blur_sigma_range: Tuple[float, float] = (0.3, 1.6)
    noise_std_range: Tuple[float, float] = (0.0, 0.01)
    scl_mask_dir: Optional[str] = None              # optional Scene Classification Layer masks
    valid_extensions: Tuple[str, ...] = field(default_factory=lambda: (".tif", ".tiff"))


# ----------------------------------------------------------------------------
# Degradation simulator
# ----------------------------------------------------------------------------
def _gaussian_kernel1d(sigma: float, device) -> torch.Tensor:
    """Odd-sized separable Gaussian kernel, radius sized to 3*sigma (min radius 1)."""
    radius = max(1, int(round(sigma * 3)))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


class RandomDegrade:
    """High-order degradation simulator: randomized blur -> area downsample -> noise.

    Mirrors the Real-ESRGAN "second-order degradation" philosophy, adapted for
    4-band reflectance data rather than 8-bit RGB, so the network sees more than
    a single idealized box-filter kernel and generalizes better to the true
    Sentinel-2 optical MTF / aliasing gap encountered at deployment time.

    Sequence:
      1) Randomized isotropic Gaussian blur (simulates sensor PSF variability)
      2) Area-based (average pooling) downsample by `degrade_factor`
      3) Additive Gaussian sensor-noise jitter
    """

    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg

    def __call__(self, hr: torch.Tensor) -> torch.Tensor:
        # hr: (C, H, W) float32 in [0, 1]
        c, h, w = hr.shape
        device = hr.device

        # 1) Randomized isotropic blur
        sigma = random.uniform(*self.cfg.blur_sigma_range)
        kernel_1d = _gaussian_kernel1d(sigma, device)
        k = kernel_1d.numel()
        pad = k // 2
        kernel_x = kernel_1d.view(1, 1, 1, k).repeat(c, 1, 1, 1)
        kernel_y = kernel_1d.view(1, 1, k, 1).repeat(c, 1, 1, 1)

        blurred = hr.unsqueeze(0)
        blurred = F.pad(blurred, (pad, pad, 0, 0), mode="reflect")
        blurred = F.conv2d(blurred, kernel_x, groups=c)
        blurred = F.pad(blurred, (0, 0, pad, pad), mode="reflect")
        blurred = F.conv2d(blurred, kernel_y, groups=c)

        # 2) Area-based (average pooling) downsample by degrade_factor
        lr = F.avg_pool2d(blurred, kernel_size=self.cfg.degrade_factor,
                           stride=self.cfg.degrade_factor)

        # 3) Additive sensor-noise jitter
        noise_std = random.uniform(*self.cfg.noise_std_range)
        if noise_std > 0:
            lr = lr + torch.randn_like(lr) * noise_std

        return lr.squeeze(0).clamp(0.0, 1.0)


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
class Sentinel2SISRDataset(Dataset):
    """4-band Sentinel-2 dataset for simulated-degradation super-resolution training.

    Each item returns a (LR, HR) pair:
        LR: (4, tile_size / degrade_factor, tile_size / degrade_factor)
        HR: (4, tile_size, tile_size)
    both scaled to [0, 1] reflectance, band order [B2, B3, B4, B8].

    Pipeline per __getitem__:
        1. Read the 4-band GeoTIFF tile via rasterio.
        2. Random-crop (with reflect-padding if the source tile is smaller
           than tile_size) to a `tile_size` x `tile_size` HR patch.
        3. Normalize DN -> reflectance via rho = DN / REFLECTANCE_SCALE, clip to [0, 1].
        4. Apply geospatial augmentation (optional): H/V flips, 90-degree-multiple
           rotations (safe for north-up rasters), and sparse spectral channel shuffle
           (spectral-invariance regularizer).
        5. Synthesize the LR input via RandomDegrade.
    """

    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self.tiles: List[str] = sorted(
            f for f in os.listdir(cfg.hr_dir)
            if f.lower().endswith(cfg.valid_extensions)
        )
        if len(self.tiles) == 0:
            raise RuntimeError(f"No GeoTIFF tiles found in {cfg.hr_dir}")
        self.degrader = RandomDegrade(cfg)

    def __len__(self) -> int:
        return len(self.tiles)

    def _read_tile(self, path: str) -> np.ndarray:
        with rasterio.open(path) as src:
            arr = src.read(out_dtype="float32")  # (4, H, W), raw DN, band order per source file
        return arr

    def _random_crop(self, arr: np.ndarray) -> np.ndarray:
        c, h, w = arr.shape
        ts = self.cfg.tile_size
        if h < ts or w < ts:
            pad_h, pad_w = max(0, ts - h), max(0, ts - w)
            arr = np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
            h, w = arr.shape[1], arr.shape[2]
        top = random.randint(0, h - ts)
        left = random.randint(0, w - ts)
        return arr[:, top:top + ts, left:left + ts]

    def _augment(self, hr: torch.Tensor) -> torch.Tensor:
        # Horizontal / vertical flip
        if random.random() < 0.5:
            hr = torch.flip(hr, dims=[2])
        if random.random() < 0.5:
            hr = torch.flip(hr, dims=[1])
        # 90-degree-multiple rotation (safe for north-up raster geometry)
        k = random.randint(0, 3)
        if k > 0:
            hr = torch.rot90(hr, k=k, dims=[1, 2])
        # Spectral channel shuffle: regularizes against band-order overfitting;
        # applied sparingly since true band semantics (B2/B3/B4/B8) matter downstream.
        if random.random() < self.cfg.spectral_shuffle_prob:
            perm = torch.randperm(hr.shape[0])
            hr = hr[perm]
        return hr

    def __getitem__(self, idx: int):
        path = os.path.join(self.cfg.hr_dir, self.tiles[idx])
        arr = self._read_tile(path)
        arr = self._random_crop(arr)

        hr = torch.from_numpy(arr).float() / REFLECTANCE_SCALE
        hr = hr.clamp(0.0, 1.0)

        if self.cfg.augment:
            hr = self._augment(hr)

        lr = self.degrader(hr)
        return {"lr": lr, "hr": hr, "tile_name": self.tiles[idx]}


# ----------------------------------------------------------------------------
# DataLoader builder
# ----------------------------------------------------------------------------
def build_dataloader(cfg: DatasetConfig, batch_size: int = 16,
                      shuffle: bool = True, num_workers: int = 8) -> DataLoader:
    ds = Sentinel2SISRDataset(cfg)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )


# ----------------------------------------------------------------------------
# Smoke test / sanity check (run directly: python dataset.py)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sanity-check the SRM Phase 1 dataset pipeline")
    parser.add_argument("hr_dir", type=str, help="Directory of 10m, 4-band GeoTIFF tiles")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    cfg = DatasetConfig(hr_dir=args.hr_dir)
    loader = build_dataloader(cfg, batch_size=args.batch_size, num_workers=args.num_workers)

    batch = next(iter(loader))
    lr, hr = batch["lr"], batch["hr"]
    print(f"Dataset size: {len(loader.dataset)} tiles")
    print(f"LR batch shape: {tuple(lr.shape)}  (expected: [{args.batch_size}, 4, "
          f"{TILE_SIZE // DEGRADE_FACTOR}, {TILE_SIZE // DEGRADE_FACTOR}])")
    print(f"HR batch shape: {tuple(hr.shape)}  (expected: [{args.batch_size}, 4, "
          f"{TILE_SIZE}, {TILE_SIZE}])")
    print(f"LR value range: [{lr.min():.4f}, {lr.max():.4f}]")
    print(f"HR value range: [{hr.min():.4f}, {hr.max():.4f}]")
