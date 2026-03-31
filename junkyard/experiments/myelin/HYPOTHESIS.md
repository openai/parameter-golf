# Myelin Sheath: Fibonacci Node Spacing in Skip Connections

## Biological inspiration
Saltatory conduction jumps between nodes of Ranvier at NON-UNIFORM intervals.
Signal fidelity maintained, transmission speed increases dramatically.
Internodal segments are passive (myelinated — just pass through).

## Architecture
Current: encoder-decoder skip connections fire at uniform intervals (every layer).
Proposed: Fibonacci-spaced "nodes" — only layers at Fibonacci indices get full skip
connections; intermediate layers are myelinated (no skip, just residual pass-through).

For 11-layer model, Fibonacci positions: 1, 2, 3, 5, 8
Layers 4, 6, 7, 9, 10, 11: myelinated (skip_weight clamped to 0, not learned).

Early layers: dense skip nodes (fast local refinement).
Deep layers: sparse skip nodes (long-range integration only).
Ratio of skip:non-skip layers ≈ φ.

φ bonus: Fibonacci IS the golden ratio sequence. Structurally exact.
         skip_count / total_layers → φ as layers → ∞.

## Key hyperparameters
- MYELIN_FIBONACCI_SKIPS = "1,2,3,5,8"  (which layers get active skip connections)
- All other green hyperparameters unchanged.

## Implementation
In GPT.__init__: after creating skip_weights, zero-init and freeze the non-Fibonacci ones.
```python
fibonacci_nodes = {1, 2, 3, 5, 8}  # 1-indexed decoder layers
for i, w in enumerate(self.skip_weights):
    if (i+1) not in fibonacci_nodes:
        nn.init.zeros_(w)
        w.requires_grad_(False)  # myelinated — frozen at 0
```

## Buildability: ★★★★☆ — ~10 lines in GPT.__init__
Risk: may hurt if skip connections are load-bearing. Run vs green baseline.
