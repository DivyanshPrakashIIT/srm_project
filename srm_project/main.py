"""
main.py — Sentinel-2 Super-Resolution Mapping (SRM): End-to-End Orchestrator

Wires together the four phases documented in SRM_Sentinel2_Implementation_Manual.pdf:
  Phase 1  dataset.py        -> DatasetConfig / build_dataloader (with synthetic fallback)
  Phase 2  models_v3.py      -> ASDDPM_CNP (diffusion) + RRDBGenerator4Band (GAN)
  Phase 3  segmentation.py   -> UNetXceptionResNet building-footprint head
  Phase 4  evaluation_v2.py  -> PSNR/SSIM (align_crops=True) + MC epistemic uncertainty

Run:
    python main.py                              # quick smoke test, synthetic data, small models
    python main.py --data-dir data/raw_tiles     # use real GeoTIFF tiles if present
    python main.py --full-models --diffusion-timesteps 100 --mc-samples 8

This is a scaffold/verification runner, not a production training loop: no
checkpointing, distributed training, or mixed precision is wired in (see the
manual's Executive Summary for what production deployment should add).
"""

import argparse
import sys

import torch

try:
    from dataset import DatasetConfig, build_dataloader
    _DATASET_MODULE_AVAILABLE = True
    _DATASET_IMPORT_ERROR = None
except ImportError as e:
    # dataset.py does `import rasterio` unconditionally at module scope, so if
    # rasterio (a native GDAL binding, occasionally finicky to install) isn't
    # present, importing dataset.py fails here -- before any try/except inside
    # load_phase1_batch() would ever run. Catch it at this level too so the
    # synthetic-tensor fallback still works even without rasterio installed.
    _DATASET_MODULE_AVAILABLE = False
    _DATASET_IMPORT_ERROR = e
from models_v3 import (
    DiffusionSchedule,
    ASDDPM_CNP,
    diffusion_training_step,
    ddpm_sample,
    RRDBGenerator4Band,
    RaDiscriminator4Band,
    VGG4BandFeatureExtractor,
    generator_step,
    discriminator_step,
)
from segmentation import UNetXceptionResNet, ParcelSegmenter
from evaluation_v2 import (
    evaluate_reconstruction_with_alignment,
    LPIPS4Band,
    ensemble_uncertainty_map,
)


def parse_args():
    p = argparse.ArgumentParser(description="SRM end-to-end pipeline runner")
    p.add_argument("--data-dir", default="data/raw_tiles",
                    help="Directory of 10m, 4-band GeoTIFF tiles (Phase 1). "
                         "Falls back to synthetic tensors if empty/missing.")
    p.add_argument("--batch-size", type=int, default=1,
                    help="Kept at 1 by default: ASDDPM's ADTBlock runs full "
                         "self-attention at up to 128x128 spatial resolution "
                         "(16,384 tokens) even in --small-models mode, which is "
                         "slow on CPU (see README 'Known performance caveat').")
    p.add_argument("--device", default=None,
                    help="cuda | cpu | mps. Default: auto-detect.")
    p.add_argument("--full-models", dest="small_models", action="store_false",
                    help="Use full-width models from the manual (slow on CPU). "
                         "Default is a small config for fast verification.")
    p.add_argument("--diffusion-timesteps", type=int, default=3,
                    help="T for the DiffusionSchedule. Manual default is 100; "
                         "kept very small here purely so ddpm_sample() finishes "
                         "in a reasonable time on CPU. Each unit of T costs one "
                         "full forward pass through ASDDPM_CNP.")
    p.add_argument("--mc-samples", type=int, default=2,
                    help="K: number of reverse-diffusion samples drawn for the "
                         "epistemic uncertainty map. Cost multiplies with "
                         "--diffusion-timesteps (K full ddpm_sample() runs). "
                         "Must be >=2: evaluation_v2.ensemble_uncertainty_map() "
                         "uses torch.std() with default (unbiased) correction, "
                         "which returns NaN for every pixel when K=1 — this is "
                         "an existing bug in evaluation_v2.py, not something "
                         "this script works around.")
    p.set_defaults(small_models=True)
    return p.parse_args()


def get_device(explicit):
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ------------------------------------------------------------------------------
# Phase 1 — Data
# ------------------------------------------------------------------------------
def load_phase1_batch(args):
    """Loads a real (LR, HR) batch from GeoTIFF tiles if available; otherwise
    generates synthetic 4-channel mockup tensors of shape (B,4,51,51) / (B,4,256,256)
    so the rest of the pipeline can be exercised immediately.
    """
    print("\n=== Phase 1: Data Engineering ===")
    if not _DATASET_MODULE_AVAILABLE:
        print(f"[Phase 1] dataset.py could not be imported "
              f"({type(_DATASET_IMPORT_ERROR).__name__}: {_DATASET_IMPORT_ERROR}). "
              f"Likely missing/broken rasterio install.")
        print("[Phase 1] Falling back to synthetic mock tensors for pipeline verification.")
        b = args.batch_size
        lr = torch.rand(b, 4, 51, 51)
        hr = torch.rand(b, 4, 256, 256)
        print(f"[Phase 1] Synthetic LR shape: {tuple(lr.shape)}  HR shape: {tuple(hr.shape)}")
        return lr, hr, True
    try:
        cfg = DatasetConfig(hr_dir=args.data_dir)
        loader = build_dataloader(cfg, batch_size=args.batch_size, num_workers=0)
        batch = next(iter(loader))
        print(f"[Phase 1] Loaded real batch from '{args.data_dir}' "
              f"({len(loader.dataset)} tiles found).")
        print(f"[Phase 1] LR shape: {tuple(batch['lr'].shape)}  HR shape: {tuple(batch['hr'].shape)}")
        return batch["lr"], batch["hr"], False
    except Exception as e:
        print(f"[Phase 1] No usable GeoTIFF tiles in '{args.data_dir}' "
              f"({type(e).__name__}: {e}).")
        print("[Phase 1] Falling back to synthetic mock tensors for pipeline verification.")
        b = args.batch_size
        lr = torch.rand(b, 4, 51, 51)    # matches degrade_factor=5 on a 256x256 HR tile
        hr = torch.rand(b, 4, 256, 256)
        print(f"[Phase 1] Synthetic LR shape: {tuple(lr.shape)}  HR shape: {tuple(hr.shape)}")
        return lr, hr, True


# ------------------------------------------------------------------------------
# Phase 2 — Super-resolution: ASDDPM (diffusion) + RRDB (GAN)
# ------------------------------------------------------------------------------
def run_phase2(lr, hr, args, device):
    print("\n=== Phase 2: Super-Resolution Models ===")

    # --- Model A: ASDDPM ---
    print("\n--- Model A: ASDDPM (Adaptive Semantic-Enhanced Diffusion) ---")
    if args.small_models:
        base_ch, ch_mults, num_heads = 16, (1, 2), 2
    else:
        base_ch, ch_mults, num_heads = 64, (1, 2, 4, 8), 8

    schedule = DiffusionSchedule(timesteps=args.diffusion_timesteps, device=device)
    asddpm = ASDDPM_CNP(base_ch=base_ch, ch_mults=ch_mults,
                         num_res_blocks=1, temb_dim=base_ch, num_heads=num_heads).to(device)
    opt_diff = torch.optim.Adam(asddpm.parameters(), lr=1e-4)

    train_loss = diffusion_training_step(asddpm, schedule, hr.to(device), lr.to(device), opt_diff)
    print(f"[Phase 2A] Diffusion training step OK. Loss: {train_loss:.4f}")

    sr_diffusion = ddpm_sample(asddpm, schedule, lr.to(device), out_shape=(lr.shape[0], 4, 256, 256))
    print(f"[Phase 2A] DDPM ancestral sampling OK. SR shape: {tuple(sr_diffusion.shape)}")

    # --- Model B: 4-Band Relativistic Real-ESRGAN ---
    print("\n--- Model B: 4-Band Relativistic Real-ESRGAN ---")
    if args.small_models:
        gan_base_ch, num_rrdb = 16, 2
    else:
        gan_base_ch, num_rrdb = 64, 23

    generator = RRDBGenerator4Band(in_ch=4, out_ch=4, base_ch=gan_base_ch,
                                    num_rrdb=num_rrdb, scale=5).to(device)
    discriminator = RaDiscriminator4Band(in_ch=4, base_ch=gan_base_ch).to(device)
    vgg = VGG4BandFeatureExtractor(layer_idx=5, pretrained_rgb=False, freeze=True).to(device)

    opt_g = torch.optim.Adam(generator.parameters(), lr=1e-4)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    sr_gan, loss_g = generator_step(generator, discriminator, vgg, lr.to(device), hr.to(device), opt_g)
    loss_d = discriminator_step(discriminator, hr.to(device), sr_gan, opt_d)
    print(f"[Phase 2B] Generator step OK. Loss: {loss_g:.4f}")
    print(f"[Phase 2B] Discriminator step OK. Loss: {loss_d:.4f}")
    print(f"[Phase 2B] SR (GAN) shape: {tuple(sr_gan.shape)} "
          f"(51x51 -> replicate-padded 52x52 -> PixelShuffle(5) -> crop to 256x256; "
          f"only the top-left 255x255 region is the true un-padded active area — see Phase 4)")

    return asddpm, schedule, sr_diffusion, sr_gan


# ------------------------------------------------------------------------------
# Phase 3 — Downstream segmentation
# ------------------------------------------------------------------------------
def run_phase3(sr_output, device):
    print("\n=== Phase 3: Downstream Segmentation ===")
    seg_model = UNetXceptionResNet(in_ch=4, num_classes=2).to(device)
    seg_model.eval()
    with torch.no_grad():
        logits = seg_model(sr_output)
    print(f"[Phase 3] UNetXceptionResNet OK. Input: {tuple(sr_output.shape)} "
          f"-> logits: {tuple(logits.shape)}")

    binary_mask = logits.argmax(dim=1)  # (B, H, W), values in {0, 1}
    print(f"[Phase 3] Binary classification map shape: {tuple(binary_mask.shape)}")

    parcel_segmenter = ParcelSegmenter()
    sr_single_tile = sr_output[0].detach().cpu().numpy()
    parcel_results = parcel_segmenter.predict_parcels(sr_single_tile)
    print(f"[Phase 3] ParcelSegmenter (YOLO11x-seg) OK. Keys: {list(parcel_results.keys())}")

    return binary_mask


# ------------------------------------------------------------------------------
# Phase 4 — Scientific evaluation + uncertainty
# ------------------------------------------------------------------------------
def run_phase4(sr_output, hr, asddpm, schedule, lr, args, device):
    print("\n=== Phase 4: Scientific Evaluation & Uncertainty ===")

    # Tier 1: PSNR/SSIM with align_crops=True. This crops both tensors to the
    # 255x255 active grid so the replicate-padding used to reach an integer
    # 5x PixelShuffle factor (see Phase 2B) never leaks into the reconstruction
    # metric, and so a possible half-pixel bicubic-resample shift can't
    # artificially suppress the score.
    lpips_evaluator = LPIPS4Band().to(device)
    metrics = evaluate_reconstruction_with_alignment(
        sr_output, hr.to(device), align_crops=True, lpips_evaluator=lpips_evaluator
    )
    print(f"[Phase 4] Tier 1 (aligned, {metrics['Evaluation_Grid']} grid): "
          f"PSNR={metrics['PSNR']:.4f} dB  SSIM={metrics['SSIM']:.4f}  "
          f"LPIPS={metrics['LPIPS']:.4f}")

    # Epistemic uncertainty: K independent reverse-diffusion samples -> per-pixel std.
    if args.mc_samples < 2:
        print(f"\n[Phase 4] WARNING: --mc-samples={args.mc_samples}. "
              f"torch.std() with a single sample and default (unbiased=True) "
              f"correction returns NaN for every pixel (0 degrees of freedom). "
              f"Proceeding anyway, but expect NaNs below.")
    print(f"\n[Phase 4] Running {args.mc_samples}-sample Monte Carlo uncertainty "
          f"estimate ({args.diffusion_timesteps} diffusion steps each)...")
    mean_sr, uncertainty = ensemble_uncertainty_map(
        asddpm, schedule, lr.to(device), out_shape=(lr.shape[0], 4, 256, 256), k=args.mc_samples
    )
    print(f"[Phase 4] Uncertainty map shape: {tuple(uncertainty.shape)}  "
          f"mean per-pixel std: {uncertainty.mean().item():.5f}  "
          f"max per-pixel std: {uncertainty.max().item():.5f}")

    return metrics, uncertainty


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")
    print(f"Model size profile: {'small (fast verification)' if args.small_models else 'full (per manual)'}")

    lr, hr, used_synthetic = load_phase1_batch(args)

    asddpm, schedule, sr_diffusion, sr_gan = run_phase2(lr, hr, args, device)

    # Feed the diffusion model's SR output forward into Phase 3 (either SR head works;
    # ASDDPM output is used here since Phase 4's uncertainty map is also ASDDPM-based).
    binary_mask = run_phase3(sr_diffusion, device)

    metrics, uncertainty = run_phase4(sr_diffusion, hr, asddpm, schedule, lr, args, device)

    print("\n=== Pipeline Summary ===")
    print(f"Data source:        {'synthetic mock tensors' if used_synthetic else args.data_dir}")
    print(f"SR (diffusion):     {tuple(sr_diffusion.shape)}")
    print(f"SR (GAN):           {tuple(sr_gan.shape)}")
    print(f"Segmentation mask:  {tuple(binary_mask.shape)}")
    print(f"PSNR / SSIM:        {metrics['PSNR']:.4f} dB / {metrics['SSIM']:.4f}")
    print(f"Mean uncertainty:   {uncertainty.mean().item():.5f}")
    print("\nAll four phases ran successfully end-to-end.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nPipeline failed — traceback below:", file=sys.stderr)
        raise
