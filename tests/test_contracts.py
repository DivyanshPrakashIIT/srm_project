"""Contract tests for CONTRACTS.md v1.0.0. Run: make test"""
import numpy as np
import pytest
import torch

from srm import contracts as C


def rand(b, h, w, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(b, 4, h, w, generator=g)


# ---- C1 band ordering ----
def test_band_order_frozen():
    assert C.BAND_ORDER == ("B2", "B3", "B4", "B8")
    assert C.BAND_ORDER.index("B4") == 2  # red
    assert C.BAND_ORDER.index("B8") == 3  # NIR
    assert C.NUM_BANDS == 4


# ---- C2 radiometry / data range ----
def test_dn_to_reflectance_scaling_and_clip():
    dn = torch.tensor([0.0, 5000.0, 10000.0, 15000.0]).view(1, 1, 2, 2).repeat(1, 4, 1, 1)
    rho = C.dn_to_reflectance(dn)
    assert rho.dtype == torch.float32
    assert torch.allclose(rho[0, 0].flatten(), torch.tensor([0.0, 0.5, 1.0, 1.0]))


def test_reflectance_range_accepts_valid():
    C.validate_reflectance(rand(2, 32, 32))


@pytest.mark.parametrize("bad", [-0.01, 1.01, float("nan"), float("inf")])
def test_reflectance_range_rejects_bad_values(bad):
    x = rand(1, 32, 32)
    x[0, 0, 0, 0] = bad
    with pytest.raises(AssertionError):
        C.validate_reflectance(x)


def test_reflectance_rejects_wrong_dtype_and_bands():
    with pytest.raises(AssertionError):
        C.validate_reflectance(rand(1, 32, 32).double())
    with pytest.raises(AssertionError):
        C.validate_reflectance(torch.rand(1, 3, 32, 32))


# ---- C3 tensor shapes ----
def test_scale_and_sizes_are_consistent():
    assert C.SCALE == 4
    assert C.TRAIN_HR_SIZE == C.TRAIN_LR_SIZE * C.SCALE == 128
    assert C.INFER_SR_SIZE == C.INFER_LR_SIZE * C.SCALE == 256


@pytest.mark.parametrize("lr_size", [C.TRAIN_LR_SIZE, C.INFER_LR_SIZE])
def test_valid_pair_shapes(lr_size):
    C.validate_pair(rand(2, lr_size, lr_size), rand(2, lr_size * 4, lr_size * 4, 1))


@pytest.mark.parametrize("hr_size", [160, 130, 127, 255])
def test_wrong_scale_rejected(hr_size):  # 5x (=160), pad hacks (130) and odd crops
    with pytest.raises(AssertionError):
        C.validate_pair(rand(1, 32, 32), rand(1, hr_size, hr_size))


# ---- C4 model output API ----
@pytest.mark.parametrize("size", [C.TRAIN_LR_SIZE, C.INFER_LR_SIZE])
def test_reference_model_conforms(ref_model, size):
    lr = rand(2, size, size)
    with torch.no_grad():
        out = ref_model(lr)
    C.validate_model_output(out, lr)
    assert out["sr"].shape == (2, 4, size * 4, size * 4)


def test_model_output_rejects_non_dict_and_extra_keys():
    lr = rand(1, 32, 32)
    sr = rand(1, 128, 128)
    with pytest.raises(AssertionError):
        C.validate_model_output(sr, lr)  # bare tensor
    with pytest.raises(AssertionError):
        C.validate_model_output({"sr": sr}, lr)  # missing logvar
    with pytest.raises(AssertionError):
        C.validate_model_output({"sr": sr, "logvar": torch.zeros_like(sr), "x": 1}, lr)


def test_model_output_rejects_nonfinite_logvar():
    lr = rand(1, 32, 32)
    sr = rand(1, 128, 128)
    lv = torch.zeros_like(sr)
    lv[0, 0, 0, 0] = float("nan")
    with pytest.raises(AssertionError):
        C.validate_model_output({"sr": sr, "logvar": lv}, lr)


# ---- C4/C5 uncertainty and product array ----
def test_sigma_bar_math():
    lv = torch.full((1, 4, 8, 8), 2 * np.log(0.1), dtype=torch.float32)  # sigma = 0.1
    sb = C.sigma_bar(lv)
    assert sb.shape == (1, 1, 8, 8)
    assert torch.allclose(sb, torch.full_like(sb, 0.1), atol=1e-6)


def test_product_array_layout():
    sr = rand(1, 256, 256)
    prod = C.assemble_product_array(sr, torch.zeros_like(sr))
    assert prod.shape == (5, 256, 256) and prod.dtype == np.float32
    assert np.allclose(prod[:4], sr[0].numpy())
    assert np.allclose(prod[4], 1.0)  # exp(0) = 1
    assert C.PRODUCT_BAND_DESCRIPTIONS == ("B2", "B3", "B4", "B8", "sigma_bar")


def test_product_array_clips_sr_and_uncertainty_nonnegative():
    sr = rand(1, 64, 64) * 3 - 1  # out of range on purpose
    prod = C.assemble_product_array(sr, torch.randn_like(sr))
    assert prod[:4].min() >= 0.0 and prod[:4].max() <= 1.0
    assert prod[4].min() >= 0.0
