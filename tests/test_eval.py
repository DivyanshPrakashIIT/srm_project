import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from srm.eval.evaluator import SRMEvaluator
from srm.eval.metrics import (
    compute_all_metrics,
    compute_ergas,
    compute_psnr,
    compute_sam,
    compute_ssim,
)


@pytest.fixture
def sample_hr_batch():
    """Generates a synthetic high-resolution ground truth batch (B, 4, 128, 128)."""
    torch.manual_seed(42)
    return torch.rand((2, 4, 128, 128), dtype=torch.float32)


def test_perfect_reconstruction_metrics(sample_hr_batch):
    """Verifies that identical tensors yield theoretical optimal metric values."""
    hr = sample_hr_batch
    metrics = compute_all_metrics(hr, hr)

    # Perfect reconstruction assertions
    assert metrics["psnr"] >= 99.0
    assert metrics["psnr_b2"] >= 99.0
    assert pytest.approx(metrics["ssim"], abs=1e-4) == 1.0
    assert pytest.approx(metrics["sam"], abs=1e-4) == 0.0
    assert pytest.approx(metrics["ergas"], abs=1e-4) == 0.0


def test_noisy_degradation_metrics(sample_hr_batch):
    """Verifies that adding Gaussian noise reduces metrics predictably."""
    hr = sample_hr_batch
    noisy_sr = torch.clamp(hr + torch.randn_like(hr) * 0.1, 0.0, 1.0)

    metrics = compute_all_metrics(noisy_sr, hr)

    assert metrics["psnr"] < 40.0
    assert metrics["ssim"] < 1.0
    assert metrics["sam"] > 0.0
    assert metrics["ergas"] > 0.0


def test_individual_metric_functions(sample_hr_batch):
    hr = sample_hr_batch
    sr = torch.clamp(hr + 0.05, 0.0, 1.0)

    psnr_dict = compute_psnr(sr, hr)
    assert "psnr" in psnr_dict and "psnr_b2" in psnr_dict

    ssim_val = compute_ssim(sr, hr)
    assert isinstance(ssim_val, float) and -1.0 <= ssim_val <= 1.0

    sam_val = compute_sam(sr, hr)
    assert isinstance(sam_val, float) and sam_val >= 0.0

    ergas_val = compute_ergas(sr, hr)
    assert isinstance(ergas_val, float) and ergas_val >= 0.0


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 4x bicubic-like interpolation dummy layer
        self.up = torch.nn.Upsample(scale_factor=4, mode="bicubic", align_corners=False)

    def forward(self, x):
        return {"sr": self.up(x), "logvar": None}


def test_srm_evaluator(tmp_path):
    lr_data = torch.rand((4, 4, 32, 32), dtype=torch.float32)
    hr_data = torch.rand((4, 4, 128, 128), dtype=torch.float32)

    dataset = TensorDataset(lr_data, hr_data)
    dataloader = DataLoader(dataset, batch_size=2)

    model = DummyModel()
    evaluator = SRMEvaluator(model, dataloader, device="cpu")

    summary = evaluator.evaluate()

    assert "psnr" in summary and "mean" in summary["psnr"]
    assert "ssim" in summary and "std" in summary["ssim"]

    # Test report export
    report_file = tmp_path / "report.json"
    evaluator.export_report(report_file, summary)
    assert report_file.exists()