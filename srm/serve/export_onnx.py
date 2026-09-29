import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional, Union
from pathlib import Path


class ModelWrapper(nn.Module):
    """
    Wrapper module to guarantee standard tuple output (sr, logvar)
    for PyTorch ONNX export compatibility.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.model(x)
        if isinstance(out, dict):
            return out["sr"], out["logvar"]
        elif isinstance(out, (tuple, list)):
            return out[0], out[1]
        return out, torch.zeros_like(out)


def export_to_onnx(
    model: torch.nn.Module,
    save_path: Union[str, Path],
    dummy_input_shape: Tuple[int, ...] = (1, 4, 32, 32),
    opset_version: int = 14,
    dynamic_axes: Optional[Dict] = None,
) -> Path:
    """
    Exports a PyTorch model for Sentinel-2 Super-Resolution to ONNX format.

    Args:
        model: PyTorch model returning dict(sr=tensor, logvar=tensor) or tuple.
        save_path: Destination path for the .onnx model file.
        dummy_input_shape: Input tensor shape (Batch, Channels, Height, Width). Default (1, 4, 32, 32).
        opset_version: ONNX operator set version.
        dynamic_axes: Dynamic axes dictionary for ONNX exporter.

    Returns:
        Path object pointing to exported ONNX model.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
    dummy_input = torch.randn(*dummy_input_shape, device=device)

    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "sr": {0: "batch_size"},
            "logvar": {0: "batch_size"},
        }

    wrapped_model = ModelWrapper(model)

    torch.onnx.export(
        wrapped_model,
        dummy_input,
        str(save_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["sr", "logvar"],
        dynamic_axes=dynamic_axes,
    )

    return save_path