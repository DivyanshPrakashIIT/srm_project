"""
Downstream Utility Evaluation Module for Sentinel-2 Super-Resolution Mapping (SRM).

Evaluates the real-world utility of 2.5m super-resolved imagery for land cover
classification and building/edge segmentation according to project contracts.
"""

from typing import Optional
import torch


class DownstreamEvaluator:
    """
    Evaluator for downstream computer vision tasks operating on super-resolved
    Sentinel-2 tensors with shape (B, 4, H, W) or predicted classification/segmentation masks.
    """

    def __init__(self, num_classes: int = 5, eps: float = 1e-7) -> None:
        """
        Args:
            num_classes: Total number of classes for segmentation/classification.
            eps: Small epsilon value to avoid division by zero.
        """
        self.num_classes = num_classes
        self.eps = eps

    def evaluate_segmentation_miou(
        self,
        pred: torch.Tensor,
        target_mask: torch.Tensor,
        num_classes: Optional[int] = None,
    ) -> float:
        """
        Computes Mean Intersection over Union (mIoU) for building and land cover edge segmentation.

        Args:
            pred: Predicted class indices (B, H, W), logits/probabilities (B, C, H, W),
                  or super-resolved tensor (B, 4, H, W).
            target_mask: Ground truth high-resolution target mask of shape (B, H, W).
            num_classes: Optional override for class count.

        Returns:
            mIoU score averaged across all classes present in targets or predictions.
        """
        n_cls = num_classes if num_classes is not None else self.num_classes

        # Convert multi-channel logits or tensors to class indices
        if pred.ndim == 4:
            pred_mask = torch.argmax(pred, dim=1)
        else:
            pred_mask = pred

        pred_mask = pred_mask.long()
        target_mask = target_mask.long()

        ious = []
        for c in range(n_cls):
            pred_c = pred_mask == c
            target_c = target_mask == c

            intersection = (pred_c & target_c).sum().float().item()
            union = (pred_c | target_c).sum().float().item()

            if union == 0:
                continue

            iou = intersection / (union + self.eps)
            ious.append(iou)

        if not ious:
            return 1.0 if torch.equal(pred_mask, target_mask) else 0.0

        return float(sum(ious) / len(ious))

    def evaluate_classification_f1(
        self,
        pred: torch.Tensor,
        target_label: torch.Tensor,
        num_classes: Optional[int] = None,
    ) -> float:
        """
        Computes Macro F1-Score for multi-class land cover classification.

        Args:
            pred: Predicted class indices (B,) or class logits/probabilities (B, C).
            target_label: Ground truth high-resolution labels of shape (B,).
            num_classes: Optional override for class count.

        Returns:
            Macro-averaged F1-score across all active classes.
        """
        n_cls = num_classes if num_classes is not None else self.num_classes

        if pred.ndim == 2:
            pred_labels = torch.argmax(pred, dim=1)
        else:
            pred_labels = pred

        pred_labels = pred_labels.long()
        target_label = target_label.long()

        f1_scores = []
        for c in range(n_cls):
            tp = ((pred_labels == c) & (target_label == c)).sum().float().item()
            fp = ((pred_labels == c) & (target_label != c)).sum().float().item()
            fn = ((pred_labels != c) & (target_label == c)).sum().float().item()

            if tp + fp + fn == 0:
                continue

            precision = tp / (tp + fp + self.eps)
            recall = tp / (tp + fn + self.eps)

            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1 = 2 * (precision * recall) / (precision + recall + self.eps)
                f1_scores.append(f1)

        if not f1_scores:
            return 1.0 if torch.equal(pred_labels, target_label) else 0.0

        return float(sum(f1_scores) / len(f1_scores))