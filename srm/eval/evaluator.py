import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import torch
from torch.utils.data import DataLoader

from srm.eval.metrics import compute_all_metrics


class SRMEvaluator:
    """
    Standardized Evaluation Harness for Sentinel-2 Super-Resolution Mapping models.
    Executes model inference over validation DataLoaders, aggregates metrics,
    and produces evaluation reports.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: Optional[Union[str, torch.device]] = None,
        scale_factor: float = 4.0,
    ):
        self.model = model
        self.dataloader = dataloader
        self.scale_factor = scale_factor
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)

    def evaluate(self) -> Dict[str, Dict[str, float]]:
        """
        Runs validation loop, computes full-reference metrics, and aggregates mean/std.

        Returns:
            Dict mapping metric names to {'mean': float, 'std': float}.
        """
        self.model.eval()
        batch_metrics: Dict[str, list] = {}

        with torch.no_grad():
            for batch in self.dataloader:
                # Handle varying DataLoader return formats
                if isinstance(batch, (tuple, list)):
                    lr_imgs, hr_imgs = batch[0], batch[1]
                elif isinstance(batch, dict):
                    lr_imgs, hr_imgs = batch["lr"], batch["hr"]
                else:
                    raise ValueError(f"Unsupported batch format: {type(batch)}")

                lr_imgs = lr_imgs.to(self.device)
                hr_imgs = hr_imgs.to(self.device)

                # API Contract: Model returns dict(sr=sr_tensor, logvar=...)
                outputs = self.model(lr_imgs)
                sr_imgs = outputs["sr"] if isinstance(outputs, dict) else outputs

                # Clamp prediction strictly to valid reflectance range [0.0, 1.0]
                sr_imgs = torch.clamp(sr_imgs, 0.0, 1.0)

                metrics = compute_all_metrics(sr_imgs, hr_imgs, scale_factor=self.scale_factor)

                for k, v in metrics.items():
                    if k not in batch_metrics:
                        batch_metrics[k] = []
                    batch_metrics[k].append(v)

        summary = {}
        for k, values in batch_metrics.items():
            summary[k] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }

        return summary

    def export_report(
        self, output_path: Union[str, Path], summary: Optional[Dict[str, Any]] = None
    ) -> None:
        """Saves evaluation summary as formatted JSON file."""
        if summary is None:
            summary = self.evaluate()

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(summary, f, indent=4)