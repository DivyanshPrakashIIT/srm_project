# DECISIONS.md: Architecture Decision Records

Format: Status / Context / Decision / Consequences. Append only; supersede with a new ADR.

## ADR-0001: Switch from 5x to 4x super-resolution scale
- **Status:** Accepted (Week 0)
- **Context:** The legacy pipeline used 5x scale (10 m -> 2 m). PixelShuffle(5) needed a 51x51 -> 260x260 replicate-pad, then crop to 256. Evaluation needed a 255x255 alignment crop so the padding would not leak into metrics. This is fragile and dimensionally awkward.
- **Decision:** Fixed 4x scale. Training LR `(B,4,32,32)` -> HR `(B,4,128,128)`; inference LR 64 -> SR 256. All sizes are power-of-two friendly, so no padding or cropping is needed.
- **Consequences:** Remove the replicate-pad hack and the 255x255 crop. Product pixel size is 2.5 m. Legacy 5x checkpoints are incompatible. Frozen in CONTRACTS C3.

## ADR-0002: Remove spectral shuffle augmentation
- **Status:** Accepted (Week 0)
- **Context:** Legacy `dataset.py` randomly permuted channels (`spectral_shuffle_prob=0.15`). Band semantics matter downstream: NIR (B8) and red carry vegetation-index and red-edge information, so permuting them corrupts the physical meaning.
- **Decision:** Never shuffle spectral channels. `spectral_shuffle_prob` defaults to 0, and the option is to be deleted when `dataset.py` is ported. Only flips and 90-degree rotations remain.
- **Consequences:** Band order `[B2,B3,B4,B8]` is a hard invariant (CONTRACTS C1). Regularisation must come from other means.

## ADR template
- **Status:** Proposed | Accepted | Superseded by ADR-XXXX
- **Context / Decision / Consequences:** ...
