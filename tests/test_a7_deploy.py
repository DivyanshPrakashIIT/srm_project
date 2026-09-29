import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from srm.serve.export_onnx import export_to_onnx, ModelWrapper
from srm.apps.app import compute_ndvi, compute_ndwi, run_inference


class DummySRModel(nn.Module):
    """Mock PyTorch Super-Resolution model for testing ONNX export contracts."""
    def __init__(self, scale_factor: int = 4):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(4, 4, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        sr = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        sr = self.conv(sr)
        logvar = torch.zeros_like(sr)
        return {"sr": sr, "logvar": logvar}


def test_model_wrapper():
    model = DummySRModel()
    wrapped = ModelWrapper(model)
    x = torch.randn(1, 4, 32, 32)
    sr, logvar = wrapped(x)

    assert sr.shape == (1, 4, 128, 128)
    assert logvar.shape == (1, 4, 128, 128)


def test_onnx_export(tmp_path: Path):
    pytest.importorskip("onnx")
    model = DummySRModel()
    save_path = tmp_path / "model_test.onnx"
    
    exported_path = export_to_onnx(
        model=model,
        save_path=save_path,
        dummy_input_shape=(1, 4, 32, 32),
        opset_version=14,
    )

    assert exported_path.exists()
    assert exported_path.stat().st_size > 0


def test_onnx_runtime_execution(tmp_path: Path):
    pytest.importorskip("onnxruntime")
    import onnxruntime as ort

    model = DummySRModel()
    save_path = tmp_path / "model_test.onnx"
    export_to_onnx(model=model, save_path=save_path)

    session = ort.InferenceSession(str(save_path))
    dummy_input = np.random.randn(2, 4, 32, 32).astype(np.float32)

    outputs = session.run(None, {"input": dummy_input})
    
    assert len(outputs) == 2
    assert outputs[0].shape == (2, 4, 128, 128)
    assert outputs[1].shape == (2, 4, 128, 128)


def test_spectral_index_helpers():
    img = np.zeros((4, 10, 10), dtype=np.float32)
    img[1] = 0.4  # Green
    img[2] = 0.2  # Red
    img[3] = 0.8  # NIR

    ndvi = compute_ndvi(img)
    ndwi = compute_ndwi(img)

    assert np.allclose(ndvi, 0.6, atol=1e-5)
    assert np.allclose(ndwi, -0.33333334, atol=1e-5)


def test_app_inference_pipeline():
    lr_patch = np.random.uniform(0, 1, size=(4, 32, 32)).astype(np.float32)
    sr_patch, logvar = run_inference(lr_patch, model_choice="Bicubic Baseline")

    assert sr_patch.shape == (4, 128, 128)
    assert logvar.shape == (4, 128, 128)