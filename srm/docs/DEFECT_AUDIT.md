# Legacy Defect Audit (srm/legacy -> new modules)

| # | File | Defect | Fix in new code | Owner |
|---|---|---|---|---|
| D1 | `models_v3.py` (`RRDBGenerator4Band`, `scale=5`), `main.py`, `dataset.py` | 5x PixelShuffle with a 51->260 replicate-pad, then crop to 256 | Clean 4x head; 32->128 and 64->256; no pad/crop (ADR-0001) | A4, A2 |
| D2 | `evaluation_v2.py` (`align_crops`, 255x255) | Metric alignment crop exists only to hide the pad hack | Drop; score full `(B,4,128,128)` | A6 |
| D3 | `dataset.py` (`spectral_shuffle_prob=0.15`) | Channel shuffle destroys red-edge/NIR semantics | Set to 0, delete the option (ADR-0002) | A2 |
| D4 | `evaluation_v2.py` (`ensemble_uncertainty_map`: `samples.std(dim=0)`) | NaN when k=1 (unbiased std divides by k-1) | `k>=2` check or `unbiased=False`; test k=1 explicitly | A6 |
| D5 | `models_v3.py` (`ADTBlock`) | Full self-attention at every decoder resolution (up to 65,536 tokens) | Full attention only at <=32x32; convolutional or windowed elsewhere | A4, A5 |
| D6 | `models_v3.py` (all models) | Output is a tensor, not `dict(sr, logvar)` | Conform to CONTRACTS C4 | A3, A5 |

Verify each fix with a pytest case in the owning workstream.
