import torch
import torch.nn as nn
import torch.nn.functional as F


class BicubicModel(nn.Module):
    """
    Non-trainable baseline using bicubic interpolation for 4x spatial super-resolution.
    """

    def __init__(self, scale_factor: int = 4, in_nc: int = 4, out_nc: int = 4):
        super().__init__()
        self.scale_factor = scale_factor
        self.in_nc = in_nc
        self.out_nc = out_nc

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: Tensor of shape (B, 4, H, W)
        Returns:
            dict containing "sr" tensor of shape (B, 4, 4*H, 4*W) and "logvar" as None.
        """
        sr = F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode="bicubic",
            align_corners=False,
        )
        return {"sr": sr, "logvar": None}