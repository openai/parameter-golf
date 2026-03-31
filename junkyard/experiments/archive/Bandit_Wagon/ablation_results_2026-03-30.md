# Bandit_Wagon Width/Depth Ablations — 2026-03-30

**Setup:** seed=444, 500 steps, warmdown=0, SKIP_GPTQ=1, CRAWLER_QUANT_INT8=1, mlp_mult=6.0
**Metric:** int6_sliding_window BPB (stride=64) — proxy only, directional

| ARM   | Label                    | Params     | Size (int6+zstd) | INT6_SW_BPB    |
|-------|--------------------------|------------|------------------|----------------|
| BW-00 | dim=512, 4F+1C (anchor)  | ~15.9M     | ~5.8MB           | **1.18616**\*  |
| BW-01 | dim=576, 4F+1C (narrow+) | 18,101,228 | 5,931,618 B      | 1.60381587     |
| BW-02 | dim=640, 4F+1C (wide)    | 22,157,740 | 6,649,057 B      | 1.63302742     |
| BW-03 | dim=512, 5F+1C (depth+1) | 16,823,860 | 5,888,703 B      | **1.54404070** |
| BW-04 | dim=512, 6F+1C (depth+2) | 19,185,724 | 6,497,859 B      | 1.56887339     |

\* Anchor at full 600s run (8000 steps, 8×H100). Proxy arms are directional only.

## Ranking (proxy, lower is better)
1. BW-03 — 5F+1C (depth +1): **1.54404**
2. BW-04 — 6F+1C (depth +2): 1.56887
3. BW-01 — dim=576 (width narrow): 1.60382
4. BW-02 — dim=640 (width wide): 1.63303

## Key Signals
- **Depth beats width** at every tested point
- **5F+1C wins over 6F+1C** — adding a 6th feedforward block hurts (overparameterized for the budget)
- **Width expansions both hurt** — 576 and 640 both trail the depth arms; 576 < 640 so narrower is better when forcing width
- BW-03 at 5.88MB stays inside 8MB budget with room to spare

## Notes
- BW-02 (dim=640) overshoots 8MB at 6.65MB int6+zstd — tight if full-run compresses less
- BW-03 is the recommended winner for Bandit_Wagon_II investigation
- Proxy inflation rule applies: do not promote without gate run
