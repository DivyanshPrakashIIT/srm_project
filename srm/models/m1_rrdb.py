from typing import Dict, Optional
import torch
import torch.nn as nn


class ResidualDenseBlock_5C(nn.Module):
    """5-convolution layer Residual Dense Block (RDB)."""

    def __init__(self, num_filters: int = 64, gc: int = 32, res_scale: float = 0.2):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_filters, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_filters + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_filters + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_filters + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_filters + 4 * gc, num_filters, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), dim=1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), dim=1))
        return x5 * self.res_scale + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block (RRDB)."""

    def __init__(self, num_filters: int = 64, gc: int = 32, res_scale: float = 0.2):
        super().__init__()
        self.res_scale = res_scale
        self.rdb1 = ResidualDenseBlock_5C(num_filters, gc, res_scale)
        self.rdb2 = ResidualDenseBlock_5C(num_filters, gc, res_scale)
        self.rdb3 = ResidualDenseBlock_5C(num_filters, gc, res_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * self.res_scale + x


class RRDBNet(nn.Module):
    """
    Residual-in-Residual Dense Block Network (RRDBNet) for Sentinel-2 Super-Resolution.
    """

    def __init__(
        self,
        in_nc: int = 4,
        out_nc: int = 4,
        num_filters: int = 64,
        num_blocks: int = 6,
        gc: int = 32,
        scale_factor: int = 4,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.in_nc = in_nc
        self.out_nc = out_nc

        # Shallow feature extraction
        self.conv_first = nn.Conv2d(in_nc, num_filters, 3, 1, 1)

        # RRDB trunk
        self.rrdb_trunk = nn.Sequential(
            *[RRDB(num_filters=num_filters, gc=gc) for _ in range(num_blocks)]
        )
        self.conv_body = nn.Conv2d(num_filters, num_filters, 3, 1, 1)

        # Upsampling block for 4x spatial expansion
        if scale_factor == 4:
            self.upconv1 = nn.Conv2d(num_filters, num_filters * 4, 3, 1, 1)
            self.pixel_shuffle1 = nn.PixelShuffle(2)
            self.upconv2 = nn.Conv2d(num_filters, num_filters * 4, 3, 1, 1)
            self.pixel_shuffle2 = nn.PixelShuffle(2)
        elif scale_factor == 2:
            self.upconv1 = nn.Conv2d(num_filters, num_filters * 4, 3, 1, 1)
            self.pixel_shuffle1 = nn.PixelShuffle(2)
        else:
            raise ValueError(f"Unsupported scale_factor: {scale_factor}")

        self.conv_hr = nn.Conv2d(num_filters, num_filters, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_filters, out_nc, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (B, in_nc, H, W)
        Returns:
            dict containing "sr" tensor of shape (B, out_nc, scale_factor*H, scale_factor*W)
            and "logvar" as None.
        """
        fea = self.conv_first(x)
        trunk = self.conv_body(self.rrdb_trunk(fea))
        fea = fea + trunk

        if self.scale_factor == 4:
            fea = self.lrelu(self.pixel_shuffle1(self.upconv1(fea)))
            fea = self.lrelu(self.pixel_shuffle2(self.upconv2(fea)))
        elif self.scale_factor == 2:
            fea = self.lrelu(self.pixel_shuffle1(self.upconv1(fea)))

        out = self.conv_last(self.lrelu(self.conv_hr(fea)))
        return {"sr": out, "logvar": None}