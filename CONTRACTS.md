# CONTRACTS.md: Frozen System-Wide Interface Contracts

**Status:** FROZEN as of Week 0. **Owner:** A0 (Integrator). Code-level source of truth: `srm/contracts.py`.
Any change requires a new ADR in `DECISIONS.md`, an A0-approved PR, and a version bump below.

**Contract version:** `1.0.0`

## C1. Band ordering
| Index | Band | Colour | Native res |
|---|---|---|---|
| 0 | B2 | Blue | 10 m |
| 1 | B3 | Green | 10 m |
| 2 | B4 | Red | 10 m |
| 3 | B8 | NIR | 10 m |

- Order is `[B2, B3, B4, B8]` everywhere: files on disk, tensors, model I/O, product GeoTIFF.
- No code may permute, shuffle or re-order channels (see ADR-0002).

## C2. Radiometry
- Dtype `float32`, surface reflectance `rho = DN / 10000` (Sentinel-2 L2A).
- Values clipped to **[0, 1]** at dataset output, model output and product write.
- Non-finite values (NaN/Inf) are forbidden in tensors crossing a module boundary.
- Only the dataset layer converts DN -> reflectance. Nothing else divides by 10000.

## C3. Tensor shapes (fixed 4x scale)
| Context | LR (input) | HR / SR (output) |
|---|---|---|
| Training | `(B, 4, 32, 32)` | `(B, 4, 128, 128)` |
| Inference tile | `(B, 4, 64, 64)` | `(B, 4, 256, 256)` |

- Scale factor is exactly `4`. Output side length = 4 x input side length. No padding or crop hacks.
- Layout NCHW, `float32`.

## C4. Model output API
```python
out = model(lr)              # lr: (B, 4, h, w) float32 in [0, 1]
out["sr"]      # (B, 4, 4h, 4w) float32 in [0, 1]
out["logvar"]  # (B, 4, 4h, 4w) float32, log sigma^2 per pixel per band, finite
```
- Return type is a `dict` with exactly the keys `sr` and `logvar`.
- Deterministic models (no uncertainty head) return `logvar = zeros_like(sr)`. The key is never omitted.
- Sampling models (diffusion) fold sampling and ensembling **inside** `forward`. Callers never see the sampler.
- Per-pixel uncertainty: `sigma_bar = mean over bands of exp(0.5 * logvar)`, shape `(B, 1, 4h, 4w)`.

## C5. Product GeoTIFF
Cloud-Optimized GeoTIFF (COG), 5 bands, `float32`:
| Band | Content |
|---|---|
| 1 | SR B2 (Blue) |
| 2 | SR B3 (Green) |
| 3 | SR B4 (Red) |
| 4 | SR B8 (NIR) |
| 5 | sigma_bar, mean per-pixel uncertainty (C4) |

- CRS is inherited from the source tile. Pixel size is exactly 1/4 of the source (10 m -> 2.5 m); the affine transform is scaled accordingly with the origin unchanged.
- Bands 1-4 in [0, 1]; band 5 >= 0.
- Band descriptions set to `B2,B3,B4,B8,sigma_bar`.

## C6. Ownership of contract changes
Only A0 edits this file. Workstream owners raise a `contract-change` issue; A0 decides and records an ADR.

## Open items for A0 to confirm (not yet frozen)
- Nodata value for the product GeoTIFF (proposal: none; tiles are fully valid).
- Compression for COG (proposal: DEFLATE + predictor 3).
