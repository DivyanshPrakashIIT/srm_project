from typing import Optional
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class SRMDegradation(nn.Module):
    """
    Simulates Sentinel-2 optical image degradation.
    Performs mild Gaussian blurring, 4x bicubic downsampling, and optional additive Gaussian noise.
    Output is strictly clipped to [0.0, 1.0].
    """

    def __init__(
        self,
        scale_factor: int = 4,
        blur_kernel_size: int = 3,
        blur_sigma: float = 0.5,
        noise_std: float = 0.0,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        self.noise_std = noise_std

    def forward(self, hr_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hr_tensor: High-resolution tensor of shape (4, H, W) or (B, 4, H, W)
                       with values in range [0.0, 1.0].
        Returns:
            lr_tensor: Downsampled low-resolution tensor of shape (4, H/4, W/4) or (B, 4, H/4, W/4).
        """
        is_batched = hr_tensor.dim() == 4
        x = hr_tensor if is_batched else hr_tensor.unsqueeze(0)

        # 1. Mild Gaussian blur
        if self.blur_kernel_size > 0 and self.blur_sigma > 0.0:
            x = TF.gaussian_blur(
                x,
                kernel_size=[self.blur_kernel_size, self.blur_kernel_size],
                sigma=[self.blur_sigma, self.blur_sigma],
            )

        # 2. Clean 4x Bicubic downsampling
        _, _, h, w = x.shape
        target_h = h // self.scale_factor
        target_w = w // self.scale_factor

        x = torch.nn.functional.interpolate(
            x,
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )

        # 3. Optional synthetic additive Gaussian noise
        if self.noise_std > 0.0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        # 4. Enforce strict reflectance bounds [0.0, 1.0]
        x = torch.clamp(x, 0.0, 1.0)

        if not is_batched:
            x = x.squeeze(0)

        return x.to(dtype=torch.float32)