"""
Unit tests for Downstream Utility & Spectral Verification Module (Account A5).
"""

import pytest
import torch

from srm.eval.downstream import DownstreamEvaluator
from srm.eval.spectral_verifier import SpectralConservationVerifier


def test_downstream_evaluator_miou_perfect_match():
    evaluator = DownstreamEvaluator(num_classes=3)
    target = torch.tensor([[[0, 1], [2, 0]]])
    pred = torch.tensor([[[0, 1], [2, 0]]])

    miou = evaluator.evaluate_segmentation_miou(pred, target)
    assert pytest.approx(miou, abs=1e-5) == 1.0


def test_downstream_evaluator_miou_partial_match():
    evaluator = DownstreamEvaluator(num_classes=2)
    target = torch.tensor([[[0, 0], [1, 1]]])
    pred = torch.tensor([[[0, 0], [0, 1]]])

    miou = evaluator.evaluate_segmentation_miou(pred, target)
    expected = (2 / 3 + 1 / 2) / 2
    assert pytest.approx(miou, abs=1e-4) == expected


def test_downstream_evaluator_f1():
    evaluator = DownstreamEvaluator(num_classes=3)
    target = torch.tensor([0, 1, 2, 0, 1])
    pred = torch.tensor([0, 1, 2, 0, 0])

    f1 = evaluator.evaluate_classification_f1(pred, target)
    assert 0.0 <= f1 <= 1.0
    assert f1 < 1.0


def test_spectral_verifier_ndvi_bounds():
    verifier = SpectralConservationVerifier()
    sr_tensor = torch.rand(2, 4, 128, 128)
    ndvi = verifier.compute_ndvi(sr_tensor)

    assert ndvi.shape == (2, 128, 128)
    assert torch.all(ndvi >= -1.0)
    assert torch.all(ndvi <= 1.0)


def test_spectral_verifier_ndwi_bounds():
    verifier = SpectralConservationVerifier()
    sr_tensor = torch.rand(2, 4, 128, 128)
    ndwi = verifier.compute_ndwi(sr_tensor)

    assert ndwi.shape == (2, 128, 128)
    assert torch.all(ndwi >= -1.0)
    assert torch.all(ndwi <= 1.0)


def test_spectral_verifier_downsampled_aggregation():
    verifier = SpectralConservationVerifier()
    sr_tensor = torch.ones(1, 4, 128, 128) * 0.5
    lr_downscaled = verifier.downsample_sr(sr_tensor, scale_factor=4)

    assert lr_downscaled.shape == (1, 4, 32, 32)
    assert torch.allclose(lr_downscaled, torch.ones(1, 4, 32, 32) * 0.5)


def test_spectral_verifier_conservation_threshold():
    verifier = SpectralConservationVerifier()

    # Case 1: Perfectly matching SR (downsampled) vs LR -> Delta NDVI = 0 < 0.02 (PASS)
    lr_tensor = torch.rand(1, 4, 32, 32) * 0.8 + 0.1
    sr_tensor = lr_tensor.repeat_interleave(4, dim=2).repeat_interleave(4, dim=3)

    results_pass = verifier.verify_conservation(
        sr_tensor, lr_tensor, scale_factor=4, ndvi_threshold=0.02
    )
    assert results_pass["is_ndvi_conserved"] is True
    assert results_pass["delta_ndvi"] < 0.02
    assert results_pass["passed"] is True

    # Case 2: Perturbed SR NIR band -> Delta NDVI > 0.02 (FAIL)
    perturbed_sr = sr_tensor.clone()
    perturbed_sr[:, 3, :, :] += 0.5
    results_fail = verifier.verify_conservation(
        perturbed_sr, lr_tensor, scale_factor=4, ndvi_threshold=0.02
    )
    assert results_fail["is_ndvi_conserved"] is False
    assert results_fail["delta_ndvi"] > 0.02
    assert results_fail["passed"] is False