import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from srm.data import SRMDataset, SRMDegradation, create_dataloaders


def test_tensor_shapes_and_batching():
    """Verify that HR patches are (4, 128, 128) and LR patches are (4, 32, 32) under standard DataLoader batching."""
    patch_count = 6
    batch_size = 2
    raw_dn_data = torch.rand((patch_count, 4, 128, 128)) * 10000.0

    dataset = SRMDataset(data=raw_dn_data, scale_factor=4, patch_size=128)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for lr, hr in loader:
        assert hr.shape == (batch_size, 4, 128, 128)
        assert lr.shape == (batch_size, 4, 32, 32)
        assert hr.dtype == torch.float32
        assert lr.dtype == torch.float32


def test_reflectance_value_range():
    """Confirm all values in HR and degraded LR outputs strictly lie within [0.0, 1.0]."""
    raw_dn_data = torch.rand((4, 4, 128, 128)) * 15000.0  # Includes out-of-bounds DN values (>10000)
    degradation = SRMDegradation(noise_std=0.05)  # Degradation with additive noise

    dataset = SRMDataset(data=raw_dn_data, degradation=degradation)

    for lr, hr in dataset:
        assert torch.all(hr >= 0.0) and torch.all(hr <= 1.0)
        assert torch.all(lr >= 0.0) and torch.all(lr <= 1.0)


def test_deterministic_generation():
    """Validate that fixed random seed produces identical degradation outputs."""
    torch.manual_seed(1234)
    dedegradation = SRMDegradation(noise_std=0.1)

    hr_patch = torch.rand((4, 128, 128))

    torch.manual_seed(42)
    lr1 = dedegradation(hr_patch)

    torch.manual_seed(42)
    lr2 = dedegradation(hr_patch)

    assert torch.equal(lr1, lr2)


def test_scl_quality_filtering():
    """Ensure patches exceeding the 10% invalid pixel threshold are filtered out."""
    clean_patch = torch.ones((4, 128, 128)) * 2000.0
    cloudy_patch = torch.ones((4, 128, 128)) * 2000.0

    # Clean mask (0% invalid class 9)
    clean_scl = torch.full((128, 128), fill_value=4, dtype=torch.uint8)  # 4 = Vegetation

    # Invalid mask (20% invalid class 9 = High Probability Cloud)
    invalid_scl = torch.full((128, 128), fill_value=4, dtype=torch.uint8)
    invalid_scl[:40, :40] = 9  # ~1024 / 16384 = 15.6% invalid > 10% threshold

    dataset = SRMDataset(
        data=[clean_patch, cloudy_patch],
        scl_masks=[clean_scl, invalid_scl],
        max_invalid_ratio=0.10,
    )

    assert len(dataset) == 1


def test_create_dataloaders(tmp_path):
    """Test helper function loading configuration."""
    cfg_path = tmp_path / "data.yaml"
    cfg_path.write_text(
        """
data:
  batch_size: 2
  num_workers: 0
  scale_factor: 4
  patch_size: 128
  max_invalid_ratio: 0.10
  seed: 42
"""
    )

    train_loader, val_loader = create_dataloaders(cfg_path)
    lr_train, hr_train = next(iter(train_loader))

    assert hr_train.shape == (2, 4, 128, 128)
    assert lr_train.shape == (2, 4, 32, 32)