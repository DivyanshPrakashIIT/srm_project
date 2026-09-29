import pytest
import torch
import torch.nn.functional as F

from srm.contracts import SCALE


class ReferenceUpsampler(torch.nn.Module):
    """Contract-conforming stand-in model (bilinear 4x, logvar=0)."""

    def forward(self, lr):
        sr = F.interpolate(lr, scale_factor=SCALE, mode="bilinear", align_corners=False)
        return {"sr": sr.clamp(0, 1), "logvar": torch.zeros_like(sr)}


@pytest.fixture
def ref_model():
    return ReferenceUpsampler().eval()


@pytest.fixture
def synthetic_lr():
    g = torch.Generator().manual_seed(0)
    return torch.rand(2, 4, 64, 64, generator=g)
