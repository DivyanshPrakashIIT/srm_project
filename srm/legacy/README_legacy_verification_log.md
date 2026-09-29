# SRM Project — Setup, Execution & Verification Log

End-to-end runner for the Sentinel-2 Super-Resolution Mapping pipeline
described in `SRM_Sentinel2_Implementation_Manual.pdf` (`dataset.py`,
`models_v3.py`, `segmentation.py`, `evaluation_v2.py`).

**Status as of 29 Aug 2026:** all four phases verified end-to-end on CPU, on
GPU (NVIDIA RTX 2050, 4GB), and against a real Sentinel-2 L2A tile. No model
has been trained yet — see [Section 7](#7-current-status--what-this-is-not-yet)
before presenting this anywhere.

---

## 1. Directory structure

```
srm_project/
├── dataset.py            # Phase 1
├── models_v3.py          # Phase 2
├── segmentation.py       # Phase 3
├── evaluation_v2.py      # Phase 4
├── main.py               # orchestrator
├── fetch_real_tile.py    # pulls a real Sentinel-2 tile from Planetary Computer
├── requirements.txt
├── data/
│   ├── raw_tiles/        # 10m, 4-band GeoTIFF tiles go here (Phase 1 input)
│   └── scl_masks/        # optional Scene Classification Layer masks
├── vector_outputs/
│   ├── predicted/        # predicted polygon shapefiles/GeoJSON (Phase 4, Tier 3)
│   └── reference/        # ground-truth polygon shapefiles/GeoJSON (Phase 4, Tier 3)
├── checkpoints/          # model weights you save during training
└── outputs/              # scratch space for generated rasters/maps
```

Commands to create this from scratch and place the source files:

```powershell
mkdir data\raw_tiles, data\scl_masks
mkdir vector_outputs\predicted, vector_outputs\reference
mkdir checkpoints, outputs

move dataset.py, models_v3.py, segmentation.py, evaluation_v2.py, main.py, fetch_real_tile.py .\srm_project\
cd srm_project
```

## 2. Dependencies

```powershell
pip install torch torchvision numpy rasterio ultralytics scipy geopandas shapely pystac-client planetary-computer --break-system-packages
```

(`--break-system-packages` only needed on systems with an externally-managed
Python; drop it in a venv/conda env, which is recommended for real training
later on.)

**GPU (optional but recommended for anything beyond a smoke test):**
```powershell
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
Match the `cuXXX` suffix to what `nvidia-smi` reports as your driver's max
supported CUDA version. Verify with:
```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**If `rasterio` or `geopandas` fail to install:** both wrap native GDAL/GEOS
libraries and occasionally fail on pip alone. `main.py` still runs without
`rasterio` (falls back to synthetic tensors), but Phase 4's Tier-3 vector
validation needs `geopandas`/`shapely` to do anything beyond print a mock
result. On Ubuntu/Debian: `sudo apt install gdal-bin libgdal-dev` first. On
macOS: `brew install gdal` first, or use `conda install -c conda-forge
rasterio geopandas`.

## 3. Getting real data (optional but recommended)

`fetch_real_tile.py` pulls one real Sentinel-2 L2A scene from Microsoft
Planetary Computer's public STAC catalog — **no account or API key needed** —
and writes it as 256×256 4-band GeoTIFF tile(s) into `data/raw_tiles/`.

```powershell
python fetch_real_tile.py --lat 12.9716 --lon 77.5946 --window-px 256
```

- `--lat` / `--lon` — center point of the area you want (any location works;
  cities/coastlines tend to have more recent, less-cloudy scenes than open
  ocean).
- `--window-px 256` — pulls just enough for one 256×256 tile per band (fast,
  small download). Omit it (defaults to 1024) to pull a bigger area and get
  several tiles cut from the same scene.
- `--max-cloud` — max cloud cover %, default 15. Raise it (e.g. `--max-cloud
  40`) if no scene is found near your location.

It prints which scene it found (ID, cloud cover %, date) and how many tiles
it wrote. Verify with `dir data\raw_tiles`.

## 4. Running the pipeline

```powershell
python main.py                                    # quick verification, synthetic data, small models, CPU or GPU
python main.py --device cuda                       # same, but on GPU
python main.py --device cuda --data-dir data/raw_tiles   # use real tiles fetched above
python main.py --full-models --diffusion-timesteps 100 --mc-samples 8 --device cuda   # per-manual config — GPU required, see caveats below
```

Step by step:
1. Open a terminal, `cd` into `srm_project/`.
2. Verify the environment: `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`.
3. (Optional) Run `fetch_real_tile.py` per Section 3 to get real data.
4. Run `main.py` with whichever flags fit. On first run it downloads an
   ImageNet-pretrained VGG19 checkpoint for the perceptual/LPIPS loss (needs
   internet once); if that fails, it prints a warning and falls back to
   random VGG weights — expected, doesn't stop the run. It'll also download
   YOLO11x-seg weights (~119MB) the first time Phase 3 runs.
5. Read the phase-by-phase console output. Phase 1 will say either
   `"Falling back to synthetic mock tensors"` or `"Loaded real batch from
   '<dir>'"` — that line tells you which data source was actually used.
6. A "Pipeline Summary" block prints at the end if all four phases completed.

## 5. What `main.py` actually does

- **Phase 1** — builds `DatasetConfig`/`build_dataloader` against `--data-dir`.
  If the directory is empty, doesn't exist, or `dataset.py` itself can't be
  imported (e.g. `rasterio` missing), it transparently falls back to synthetic
  `(B,4,51,51)` LR / `(B,4,256,256)` HR tensors so the rest of the pipeline
  can still be exercised.
- **Phase 2** — runs one training step and one full `ddpm_sample()` for
  ASDDPM, and one generator/discriminator step for the RRDB GAN, validating
  the 256×256 padded output shape.
- **Phase 3** — feeds the diffusion model's SR output through
  `UNetXceptionResNet` for a binary classification map, and through
  `ParcelSegmenter` (mock mode if `ultralytics`/weights aren't available).
- **Phase 4** — computes PSNR/SSIM/LPIPS with `align_crops=True` (crops both
  tensors to the 255×255 active grid before scoring, so the 1-pixel
  replicate-pad used to make the 5× PixelShuffle integer-friendly never
  leaks into the fidelity metrics), then runs a `--mc-samples`-way Monte
  Carlo ensemble through `ddpm_sample()` to build a per-pixel epistemic
  uncertainty map.

## 6. Verification log

Every run below actually happened — either in my sandbox (noted "sandbox,
CPU, no GPU available there") or on the user's own Windows 11 laptop
(HP Victus, Intel i5, NVIDIA RTX 2050 4GB).

| # | Config | Where | Data | Result |
|---|---|---|---|---|
| 1 | default (`T=3, mc=2, batch=1`) | Sandbox, CPU | synthetic | Pass, ~2 min. PSNR 4.79 dB / SSIM 0.014 |
| 2 | default | User, CPU | synthetic | Pass. PSNR 4.79 dB / SSIM 0.014 |
| 3 | `--device cuda` (default sizes) | User, RTX 2050 | synthetic | Pass, fast, no OOM |
| 4 | `--device cuda --diffusion-timesteps 50 --mc-samples 4` | User, RTX 2050 | synthetic | Pass, no OOM |
| 5 | `--device cuda --full-models --diffusion-timesteps 100 --mc-samples 8` | User, RTX 2050 | synthetic | Tier-1 metrics completed (PSNR 4.78 dB / SSIM 0.0135). MC uncertainty step hit ~3955/4096 MiB VRAM, ran 30+ min at 100% GPU util without OOMing, user cancelled before it finished. Full config works but is impractically slow on 4GB VRAM at mc-samples=8. |
| 6 | `fetch_real_tile.py --lat 12.9716 --lon 77.5946 --window-px 256` | User | — | Found scene `S2C_MSIL2A_20260427T050651_R019_T43PGQ_20260427T102509`, 0.2% cloud, 27 Apr 2026. Wrote 1 real 256×256 tile. |
| 7 | `--device cuda --data-dir data/raw_tiles` (small models) | User, RTX 2050 | real Sentinel-2 tile from run 6 | Pass. `"Loaded real batch from 'data/raw_tiles' (1 tiles found)"`. PSNR 5.18 dB / SSIM 0.0085 |

**Practical config recommendations from this log**, for this specific
4GB-VRAM card:
- Day-to-day iteration / debugging: default or stage-2-style settings
  (`--diffusion-timesteps 50 --mc-samples 4`) — fast, no OOM risk.
- Full-model numbers without the multi-hour wait: `--full-models
  --diffusion-timesteps 100 --mc-samples 2` (2×100=200 passes instead of
  8×100=800 — Tier-1 PSNR/SSIM don't depend on `--mc-samples` at all, only
  the uncertainty ensemble does).
- Real training epochs at full config: needs more VRAM than this card has
  headroom for comfortably — plan on cloud GPU access, or keep
  `--small-models` for local work.

## 7. Current status — what this IS and IS NOT

**Verified (see log above):**
- All four phases wire together correctly — shapes, dtypes, and interfaces
  match across every phase boundary.
- Runs correctly on CPU and on a real consumer GPU.
- Runs correctly on a real Sentinel-2 L2A tile (not just synthetic data) —
  `rasterio` reading, DN→reflectance normalization, and the degradation
  simulator all executed against real satellite imagery.
- The trickier implementation details (51×51→260×260→crop-256 replicate
  padding, the 255×255 aligned evaluation crop) behave exactly as designed.

**NOT yet true — don't claim these:**
- **No model has been trained.** Every run above is one forward(+backward)
  step, not a training loop. PSNR/SSIM values (4.8–5.2 dB) are meaningless
  as accuracy numbers — they're comparing an untrained model's noisy output
  against a target, not measuring real reconstruction quality.
- **Nothing is "deployed."** There's no trained model serving predictions.
  What exists is a verified scaffold.
- **Phase 4's Tier-3 vector validation has never run** — `main.py` doesn't
  exercise `area_based_confusion_matrix` / `count_based_centroid_validation`
  since there are no predicted/reference polygon files yet.
- **No actual training loop exists** — `main.py` runs one step per model per
  invocation. Real training needs epochs, checkpointing, LR scheduling, none
  of which is wired in (matching the manual's own framing of the code as a
  scaffold, not a production trainer).

**Accurate way to describe this project out loud:** *"I implemented all four
phases from the manual and verified them end-to-end on CPU, GPU, and against
real Sentinel-2 imagery. Next step is building out an actual training loop
and running it on a real dataset."* That's true, specific, and holds up to
follow-up questions.

## 8. Known issues in the provided source files (not caused by main.py)

1. **ASDDPM's attention is expensive by design.** `ADTBlock` in
   `models_v3.py` runs full self-attention over *every* decoder resolution,
   including near-full-resolution stages (128×128 = 16,384 tokens even in
   the small test config; up to 256×256 = 65,536 tokens in the full config).
   A single training step took ~63s on CPU in early testing, and the full
   `T=100, mc_samples=8` GPU run in row 5 above ran 30+ minutes without
   finishing. If training speed matters, consider restricting `ADTBlock` to
   only the lowest 1–2 decoder resolutions — the common pattern in most
   diffusion U-Nets (e.g. Stable Diffusion, ADM) — rather than every stage.
2. **`ensemble_uncertainty_map()` returns `NaN` at `k=1`.**
   `evaluation_v2.py` calls `samples.std(dim=0)` with PyTorch's default
   `unbiased=True` correction, dividing by `(k-1)`; at `k=1` that's division
   by zero. `main.py` defaults `--mc-samples` to `2` and warns if overridden
   to `1`, but the underlying function has no guard itself — worth a
   one-line fix (`unbiased=False`, or an explicit `k>=2` check) if you'll
   ever call it standalone with `k=1`.

Neither breaks correctness for `k>=2` / reasonable `T` — they're a
performance characteristic and a small edge-case bug, not wiring problems.