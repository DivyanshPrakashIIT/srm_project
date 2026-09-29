# STATE.md: Living Status Board

**Owner:** A0. **Updated:** 2026-09-29 (Week 0). Owners update their own card in the same PR as their work.
Status: `TODO` | `IN-PROGRESS` | `BLOCKED` | `REVIEW` | `DONE`.
The A1-A9 scopes below are the proposed split; A0 confirms in Week 0.

| ID | Workstream | Folder | Status | Blocked by |
|---|---|---|---|---|
| A0 | Integration, contracts, CI | root, `srm/contracts.py`, `tests/` | IN-PROGRESS | none |
| A1 | Data acquisition (STAC fetch, tiling) | `srm/data/` | TODO | A0 |
| A2 | Dataset, degradation, loaders | `srm/data/` | TODO | A0 |
| A3 | Model M1 | `srm/models/` | TODO | A0, A2 |
| A4 | Model shared blocks (M1/M2) | `srm/models/` | TODO | A0 |
| A5 | Model M2 and training loops | `srm/models/`, `srm/train/` | TODO | A0, A2 |
| A6 | Evaluation and metrics | `srm/eval/` | TODO | A0 |
| A7 | Downstream apps (segmentation, parcels) | `srm/apps/` | TODO | A0 |
| A8 | Serving, GeoTIFF product | `srm/serve/` | TODO | A0 |
| A9 | Web front end | `srm/web/` | TODO | A8 |

## Task cards

### A0: Integrator / Architect / CI
- [x] Freeze CONTRACTS.md v1.0.0, CODEOWNERS, DECISIONS.md
- [ ] Move legacy code to `srm/legacy/`; run `make setup`, `make test`, `make e2e` on a clean clone
- [ ] Enable branch protection: CI green and CODEOWNERS review required
- **Done when:** all workstreams merge into a green `make e2e`.

### A1: Data acquisition
- [ ] Port `fetch_real_tile.py` to `srm/data/fetch.py` (STAC, `[B2,B3,B4,B8]`, 4-band GeoTIFF tiles)
- [ ] Tile at LR 32/64 grid sizes per CONTRACTS C3
- **Done when:** `configs/data.yaml` drives a reproducible tile download.

### A2: Dataset and degradation
- [ ] Port `dataset.py` with clean 4x degradation (HR 128 -> LR 32); `spectral_shuffle_prob=0`
- [ ] `validate_pair` passes on every batch
- **Done when:** loader emits contract-valid `(B,4,32,32)` / `(B,4,128,128)`.

### A3: Model M1
- [ ] Baseline model (`configs/model_m1.yaml`) returning `dict(sr, logvar)`
- **Done when:** passes `validate_model_output` at train and inference sizes.

### A4: Shared blocks
- [ ] Clean 4x upsampling head (no replicate-pad hack); ADT attention only at <=32x32
- **Done when:** blocks consumed by both M1 and M2.

### A5: Model M2 and training
- [ ] Port ASDDPM-CNP as M2 (`configs/model_m2.yaml`); training loop, checkpointing
- **Done when:** short training run reduces loss on synthetic data.

### A6: Evaluation
- [ ] Port `evaluation_v2.py`; fix k=1 NaN; remove 5x/255 crop logic
- **Done when:** PSNR/SSIM/LPIPS and uncertainty tested with pytest.

### A7: Apps
- [ ] Port `segmentation.py` to consume SR `(B,4,256,256)` per contract
- **Done when:** segmentation runs on a contract-valid SR tile.

### A8: Serving
- [ ] Inference service producing the 5-band COG (C5) via `assemble_product_array`
- **Done when:** output GeoTIFF passes the e2e product check.

### A9: Web
- [ ] Map viewer for the SR product and the uncertainty band
- **Blocked by:** A8 product format sample.

## Change log
- 2026-09-29: Week 0 scaffold created by A0.
