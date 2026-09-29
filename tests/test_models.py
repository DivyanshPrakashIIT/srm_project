import pytest
import torch
import torch.nn as nn

from srm.models import BicubicModel, RRDBNet, build_model


@pytest.fixture
def dummy_lr_batch():
    # Input LR Tensor shape: (B, C, H, W) = (2, 4, 32, 32)
    return torch.rand((2, 4, 32, 32), dtype=torch.float32)


@pytest.fixture
def dummy_hr_batch():
    # Target HR Tensor shape: (B, C, 4*H, 4*W) = (2, 4, 128, 128)
    return torch.rand((2, 4, 128, 128), dtype=torch.float32)


def test_m0_bicubic_forward_shape_and_keys(dummy_lr_batch):
    model = BicubicModel(scale_factor=4, in_nc=4, out_nc=4)
    output = model(dummy_lr_batch)

    assert isinstance(output, dict)
    assert "sr" in output and "logvar" in output
    assert output["logvar"] is None

    sr = output["sr"]
    assert sr.shape == (2, 4, 128, 128)
    assert sr.dtype == torch.float32


def test_m1_rrdb_forward_shape_and_keys(dummy_lr_batch):
    model = RRDBNet(in_nc=4, out_nc=4, num_filters=16, num_blocks=2, scale_factor=4)
    output = model(dummy_lr_batch)

    assert isinstance(output, dict)
    assert "sr" in output and "logvar" in output
    assert output["logvar"] is None

    sr = output["sr"]
    assert sr.shape == (2, 4, 128, 128)
    assert sr.dtype == torch.float32


def test_m1_rrdb_backpropagation(dummy_lr_batch, dummy_hr_batch):
    model = RRDBNet(in_nc=4, out_nc=4, num_filters=16, num_blocks=2, scale_factor=4)
    output = model(dummy_lr_batch)
    sr = output["sr"]

    criterion = nn.L1Loss()
    loss = criterion(sr, dummy_hr_batch)
    loss.backward()

    # Verify parameters receive non-zero gradients
    grad_exists = any(
        p.grad is not None and torch.abs(p.grad).sum() > 0
        for p in model.parameters()
    )
    assert grad_exists, "Model parameters failed to receive gradients during backprop."


def test_m1_rrdb_loss_reduction(dummy_lr_batch, dummy_hr_batch):
    model = RRDBNet(in_nc=4, out_nc=4, num_filters=16, num_blocks=2, scale_factor=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()

    out_initial = model(dummy_lr_batch)
    loss_initial = criterion(out_initial["sr"], dummy_hr_batch).item()

    # Optimize for 5 iterations
    for _ in range(5):
        optimizer.zero_grad()
        out = model(dummy_lr_batch)
        loss = criterion(out["sr"], dummy_hr_batch)
        loss.backward()
        optimizer.step()

    out_final = model(dummy_lr_batch)
    loss_final = criterion(out_final["sr"], dummy_hr_batch).item()

    assert loss_final < loss_initial, f"Loss did not decrease: {loss_initial:.6f} -> {loss_final:.6f}"


def test_build_model_factory():
    cfg_bicubic = {"model": {"name": "m0_bicubic", "scale_factor": 4}}
    model_m0 = build_model(cfg_bicubic)
    assert isinstance(model_m0, BicubicModel)

    cfg_rrdb = {"model": {"name": "m1_rrdb", "num_filters": 32, "num_blocks": 2}}
    model_m1 = build_model(cfg_rrdb)
    assert isinstance(model_m1, RRDBNet)

    with pytest.raises(ValueError):
        build_model({"model": {"name": "invalid_model_type"}})