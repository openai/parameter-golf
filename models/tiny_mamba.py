"""Mamba-based variant — Placeholder for Phase 4.

Mamba-3 is reported to be 2x more parameter-efficient than transformers
at <10M parameter scale. Worth exploring for the 16MB budget.

TODO (Phase 4):
- Implement Mamba block (selective SSM)
- Same interface as TinyGPT (forward(idx, targets) -> logits, loss)
- Support weight sharing / looped architecture
- Compare BPB vs transformer at equal param count
"""

# Implementation coming in Phase 4
raise NotImplementedError("Mamba not yet implemented — Phase 4")
