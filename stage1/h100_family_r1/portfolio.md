# H100 Family R1 Portfolio

Superseded note:
This package is a provisional config-heavy scaffold.
It is not the canonical Stage 1 family after the mechanism-level recut in `stage1/`.

This is the first single-GPU H100 deployment family for Stage 1.

It follows the Enigma pattern:

- one pure baseline,
- seven distinct single-mechanism probes,
- no composites yet,
- one slot per main neighborhood,
- enough breadth to update the main Bayesian questions.

## Proxy Caveat

These runs are intentionally `1xH100` scouts.
They are not perfect proxies for final `8xH100` leaderboard conditions because `world_size` changes gradient-accumulation behavior in [train_gpt.py](/Users/ankit/Documents/dev/RL/nanoe/nanoevolve/pgolf/parameter-golf/train_gpt.py).

Use them to:

- rank families,
- kill clear losers,
- detect throughput and quantization effects,
- decide what deserves aligned `8xH100` promotion.

Do not treat them as final proof.

## Active Slate

| Slot | Role | Hypothesis | Run Name | Why included |
| --- | --- | --- | --- | --- |
| P0 | baseline | `none` | `baseline` | pure comparator |
| P1 | architecture | `H01` | `h01_depth10_width480` | matched-size depth-up width-down scout |
| P2 | architecture | `H03` | `h03_kv2_width544` | more aggressive GQA with reinvested width |
| P3 | structural | `H05` | `h05_alt_share` | alternate-layer sharing wildcard |
| P4 | throughput | `H08` | `h08_seq512` | shortest clean test of the 600-second bottleneck |
| P5 | optimizer | `H11` | `h11_adaptive_muon` | adaptive Muon compute schedule |
| P6 | artifact | `H17` | `h17_quant_bytes` | byte-focused quantizer tightening |
| P7 | artifact | `H18` | `h18_quant_quality` | quality-focused quantizer improvement |

## Main Questions

1. Does parameter allocation beat local optimizer work in this regime?
2. Is shorter context the cleanest route to better final score?
3. Is Muon compute too expensive for the 10-minute budget?
4. Is the final frontier already blocked by export quality or export bytes?
5. Can parameter sharing survive training without collapsing quality?
