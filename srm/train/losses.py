"""
Physics-Informed and Compound Loss Functions for Sentinel-2 Super-Resolution Mapping.
Includes SAM Loss, Heteroscedastic Uncertainty Loss, Perceptual Loss, and Compound Loss.
"""

from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAMLoss(nn.Module):
    """
    Spectral Angle Mapper (SAM) Loss.
    Measures spectral vector similarity across all 4 bands [B2, B3, B4, B8].
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted SR tensor of shape (B, C, H, W)
            target: Ground truth HR tensor of shape (B, C, H, W)
        Returns:
            SAM loss scalar (in radians).
        """
        dot_product = torch.sum(pred * target, dim=1)
        norm_pred = torch.norm(pred, p=2, dim=1)
        norm_target = torch.norm(target, p=2, dim=1)

        cos_theta = dot_product / (norm_pred * norm_target + self.eps)
        cos_theta = torch.clamp(cos_theta, -1.0 + self.eps, 1.0 - self.eps)
        sam_map = torch.acos(cos_theta)
        return torch.mean(sam_map)


class HeteroscedasticLoss(nn.Module):
    """
    Heteroscedastic Uncertainty Loss.
    Formulation:
        L_unc = 0.5 * exp(-logvar) * ||y_HR - y_SR||_1 + 0.5 * logvar
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: Predicted SR tensor of shape (B, C, H, W)
            target: Target HR tensor of shape (B, C, H, W)
            logvar: Predicted log-variance tensor of shape (B, C, H, W)
        Returns:
            Heteroscedastic uncertainty loss scalar.
        """
        diff = torch.abs(target - pred)
        loss = 0.5 * torch.exp(-logvar) * diff + 0.5 * logvar
        return torch.mean(loss)


class PerceptualLoss(nn.Module):
    """
    Perceptual Feature Loss using VGG16 feature extraction on RGB channels [B4, B3, B2].
    """

    def __init__(self, feature_layers: list = [3, 8, 15]):
        super().__init__()
        self.feature_layers = feature_layers
        self.vgg = None
        try:
            import torchvision.models as models

            weights = getattr(models, "VGG16_Weights", None)
            vgg16 = models.vgg16(weights=weights.DEFAULT if weights else True)
            features = list(vgg16.features.children())
            max_layer = max(feature_layers) + 1
            self.vgg = nn.Sequential(*features[:max_layer]).eval()
            for param in self.vgg.parameters():
                param.requires_grad = False
        except Exception:
            self.vgg = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted SR tensor of shape (B, 4, H, W)
            target: Ground truth HR tensor of shape (B, 4, H, W)
        Returns:
            Perceptual feature loss scalar.
        """
        if self.vgg is None:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        # Extract RGB channels [B4, B3, B2] -> indices [2, 1, 0]
        pred_rgb = pred[:, [2, 1, 0], :, :]
        target_rgb = target[:, [2, 1, 0], :, :]

        loss = 0.0
        x_pred = pred_rgb
        x_target = target_rgb

        for idx, layer in enumerate(self.vgg):
            x_pred = layer(x_pred)
            x_target = layer(x_target)
            if idx in self.feature_layers:
                loss += F.l1_loss(x_pred, x_target)

        return loss


class SRMCompoundLoss(nn.Module):
    """
    SRM Compound Loss combining L1, SAM, Heteroscedastic Uncertainty, and Perceptual losses.
    """

    def __init__(
        self,
        w_l1: float = 1.0,
        w_sam: float = 0.1,
        w_unc: float = 1.0,
        w_perc: float = 0.0,
    ):
        super().__init__()
        self.w_l1 = w_l1
        self.w_sam = w_sam
        self.w_unc = w_unc
        self.w_perc = w_perc

        self.l1_loss = nn.L1Loss()
        self.sam_loss = SAMLoss()
        self.unc_loss = HeteroscedasticLoss()
        self.perc_loss = PerceptualLoss()

    def forward(
        self,
        preds: Union[torch.Tensor, Dict[str, torch.Tensor]],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            preds: Model output dict containing 'sr' and 'logvar' tensors, or tensor 'sr'.
            target: Target HR image tensor (B, 4, H, W).
        Returns:
            Combined total loss scalar tensor.
        """
        if isinstance(preds, dict):
            sr = preds["sr"]
            logvar = preds.get("logvar", None)
        else:
            sr = preds
            logvar = None

        l1 = self.l1_loss(sr, target)
        sam = self.sam_loss(sr, target)

        total_loss = self.w_l1 * l1 + self.w_sam * sam

        if logvar is not None and self.w_unc > 0:
            unc = self.unc_loss(sr, target, logvar)
            total_loss = total_loss + self.w_unc * unc

        if self.w_perc > 0:
            perc = self.perc_loss(sr, target)
            total_loss = total_loss + self.w_perc * perc

        return total_loss

    def get_components(
        self,
        preds: Union[torch.Tensor, Dict[str, torch.Tensor]],
        target: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Computes and returns individual loss terms as float values.
        """
        if isinstance(preds, dict):
            sr = preds["sr"]
            logvar = preds.get("logvar", None)
        else:
            sr = preds
            logvar = None

        l1_val = self.l1_loss(sr, target).item()
        sam_val = self.sam_loss(sr, target).item()
        unc_val = self.unc_loss(sr, target, logvar).item() if logvar is not None else 0.0
        perc_val = self.perc_loss(sr, target).item() if self.w_perc > 0 else 0.0

        total_val = (
            self.w_l1 * l1_val
            + self.w_sam * sam_val
            + self.w_unc * unc_val
            + self.w_perc * perc_val
        )

        return {
            "total": total_val,
            "l1": l1_val,
            "sam": sam_val,
            "unc": unc_val,
            "perc": perc_val,
        }