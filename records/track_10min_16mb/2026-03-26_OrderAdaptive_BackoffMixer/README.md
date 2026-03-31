# Order-Adaptive BackoffMixer

## Results

| Seed | val_bpb | Eval time |
|------|---------|-----------|
| 42   | 0.5437  | ~391s     |
| 1337 | 0.5450  | ~391s     |
| 2024 | 0.5434  | ~391s     |
| **Mean** | **0.5440** | |
| Std  | 0.0008  | |

Artifact: ~16.0 MB. Train: 600s on 8xH100 SXM. Eval: ~391s.

## Architecture

11-layer transformer with XSA-all, full MHA (8/8 heads), LeakyReLU(0.5)^2, 3.5x MLP expansion. int5 quantization + compression. EMA, Tight SWA, Soft-Round QAT.

## Eval-Time Scoring

Order-adaptive entropy-gated BackoffNgramMixer with multi-order backoff (2-7 gram). Per-order entropy thresholds for mixing weight selection. Score-first, backward-looking, deterministic.

## Acknowledgments

Standing on the shoulders of many contributors to this competition:

- @abaybektursun — PR #549 (base architecture, Legal TTT framework, Parallel Muon)
- @deanbrr — PR #659, #779 (original n-gram eval cache concept, BackoffNgramMixer, drift-free TTT)
- @Asukabot0 — PR #715, #727 (XSA-all, backoff concept, entropy-adaptive alpha)
- @gowtham0992 — PR #606 (int5 + Soft-Round QAT)
- @signalrush — PR #414 (EMA training recipe)
- @sofiabod — PR #518 (LeakyReLU activation)
- @thwu1 — PR #180 (mixed quantization, BigramHash, SmearGate)
- @RoyiRa — PR #700 (TTT framework extensions)
- @Christopher-Lee-McClendon — PR #461 (TTT recipe)
- @raahilshah — PR #162 (int6 quantization baseline)

This competition has been an incredible collaborative learning experience. Every improvement builds on ideas shared openly by the community.
