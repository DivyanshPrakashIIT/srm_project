"""
Unit tests for Account A4 (Advanced Architectures & Physics Losses).
Tests SwinIR forward pass shapes, UncertaintyHead, SAMLoss, HeteroscedasticLoss, and SRMCompoundLoss.
"""

import pytest
import torch

from srm.models import SRMSwinIR, SwinIRNet, UncertaintyHead, build_model
from srm.train.losses import HeteroscedasticLoss, SAMLoss, SRMCompoundLoss


def test_uncertainty_head_shape():
    head = UncertaintyHead(in_channels=64, out_channels=4)
    x = torch.randn(2, 64, 128, 128)
    out = head(x)
    assert out.shape == (2, 4, 128, 128)


def test_swinir_forward_and_shapes():
    model = SwinIRNet(
        in_chans=4,
        out_chans=4,
        upscale=4,
        embed_dim=64,
        depths=[2, 2],
        num_heads=[2, 2],
        window_size=8,
    )
    lr_input = torch.randn(2, 4, 32, 32)
    output = model(lr_input)

    assert isinstance(output, dict)
    assert "sr" in output
    assert "logvar" in output
    assert output["sr"].shape == (2, 4, 128, 128)
    assert output["logvar"].shape == (2, 4, 128, 128)


def test_build_model_factory():
    model = build_model("m2_swin")
    assert isinstance(model, SwinIRNet)

    model_alias = build_model("swinir")
    assert isinstance(model_alias, SRMSwinIR)


def test_sam_loss_nonnegativity_and_backward():
    sam_fn = SAMLoss()
    pred = torch.rand(2, 4, 32, 32, requires_grad=True)
    target = torch.rand(2, 4, 32, 32)

    loss = sam_fn(pred, target)
    assert loss.item() >= 0.0

    loss.backward()
    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any()


def test_heteroscedastic_loss_backward():
    unc_fn = HeteroscedasticLoss()
    pred = torch.rand(2, 4, 32, 32, requires_grad=True)
    logvar = torch.zeros(2, 4, 32, 32, requires_grad=True)
    target = torch.rand(2, 4, 32, 32)

    loss = unc_fn(pred, target, logvar)
    loss.backward()

    assert pred.grad is not None
    assert logvar.grad is not None
    assert not torch.isnan(pred.grad).any()
    assert not torch.isnan(logvar.grad).any()


def test_srm_compound_loss():
    compound_fn = SRMCompoundLoss(w_l1=1.0, w_sam=0.1, w_unc=1.0)
    preds = {
        "sr": torch.rand(2, 4, 128, 128, requires_grad=True),
        "logvar": torch.zeros(2, 4, 128, 128, requires_grad=True),
    }
    target = torch.rand(2, 4, 128, 128)

    loss = compound_fn(preds, target)
    assert loss.item() > 0.0

    loss.backward()
    assert preds["sr"].grad is not None
    assert preds["logvar"].grad is not None

    components = compound_fn.get_components(preds, target)
    assert "total" in components
    assert "l1" in components
    assert "sam" in components
    assert "unc" in components