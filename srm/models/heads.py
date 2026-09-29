"""
Uncertainty Prediction Heads for Super-Resolution Mapping.
Provides heads for estimating heteroscedastic noise variance.
"""

import torch
import torch.nn as nn


class UncertaintyHead(nn.Module):
    """
    Predicts log-variance (logvar) tensor representing heteroscedastic noise variance
    for super-resolution reconstruction outputs.
    """

    def __init__(self, in_channels: int = 64, out_channels: int = 4):
        """
        Args:
            in_channels: Number of input feature channels.
            out_channels: Number of output spectral channels (default: 4 for Sentinel-2 B2, B3, B4, B8).
        """
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature tensor of shape (B, in_channels, H, W)
        Returns:
            logvar: Log-variance tensor of shape (B, out_channels, H, W)
        """
        return self.head(x)