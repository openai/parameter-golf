# Astrocyte: Tiny Parallel Gating Network

## Biological inspiration
Astrocytes (~10:1 ratio to neurons) don't compute — they modulate synaptic strength,
clear noise, synchronize firing. They're the infrastructure layer. Never touches
hidden states directly — only modulates the main network's attention.

## Architecture
A tiny "astrocyte" network (~2% of model params, ~300K) runs in parallel:
- Input: attention entropy of each head at each layer (computed from existing attn scores)
- Output: per-head multiplicative Q/K scales fed back to attention projections
- Never touches hidden states directly

Astrocyte hidden dims scale by 1/φ per layer: 512 → 316 → 195 → 120.
Total ~300K extra params (2% of base 15M).

The astrocyte sees the FULL attention entropy map (num_layers × num_heads) and outputs
a scale vector (num_layers × num_heads). Main network Q/K projections are multiplied by
these scales before attention computation.

φ bonus: Astrocyte dims follow 1/φ geometric sequence: 512, 316, 195, 120.

## Key hyperparameters
- ASTROCYTE_ENABLED = 1
- ASTROCYTE_HIDDEN = 512  (first dim, rest follow /φ progression)
- ASTROCYTE_LR = 0.025    (same as scalar_lr, separate optimizer group)

## Implementation notes
New class AstrocyteNet(nn.Module):
  - Linear(num_layers * num_heads, 512) → ReLU
  - Linear(512, 316) → ReLU
  - Linear(316, num_layers * num_heads)  (outputs scales, init near 1.0)

Requires attention to return entropy or raw scores — needs hook into CausalSelfAttention.
Highest implementation complexity of the five.

## Buildability: ★★★☆☆ — ~50 lines, needs attn entropy extraction
