# Record: LeakyReLU² + Legal Score‑First TTT + N‑gram Backoff Cache + Gated Attention

**val_bpb = 0.9641** (3‑seed mean, std 0.0007)

## Results (3‑seed validation)

| Seed | val\_bpb | val\_loss | Artifact Size | Train Steps | Train Time |
|------|---------|----------|--------------|-------------|------------|
| 1337 | 0.9642  | 1.6274   | 15,982,044 B | 7,185       | 599,384 ms |
| 42   | 0.9648  | 1.6285   | 15,977,267 B | 7,182       | 599,761 ms |
| 2025 | 0.9634  | 1.6261   | 15,989,583 B | 7,196       | 599,618 ms |
| **Mean** | **0.9641** | **1.62735** | — | — | — |
| **Std**  | **0.0007** | — | — | — | — |

**Statistical significance**: mean 0.9641 bpb (1.6274 nats) vs current merged top 1.1147 bpb (1.8822 nats, [PR #1019](https://github.com/openai/parameter-golf/pull/1019)) → Δ = −0.2548 nats, Welch t = −328.3, df = 2.93, p ≪ 0.01. Required improvement threshold ≥ 0.005 nats ([official rule](https://github.com/openai/parameter-golf/blob/main/README.md#L191)); this Δ exceeds it by 51×.

## Technique

- **Architecture**: 11L, 512d, GQA 8H/4KV, MLP 3×, LeakyReLU(0.5)², XSA‑5 (layers 6–10), Value Residual, Gated Attention, SmearGate, VE(128) on layers 8/9/10, BigramHash(2048), Partial RoPE(16/64), LN Scale, MTP‑2, EMA(0.9985). Tied embeddings. Muon optimizer.
- **N‑gram eval cache** (community precedent: [PR #727](https://github.com/openai/parameter-golf/pull/727)):
  - Multi‑order backoff (orders 2–9): highest matching order wins, cascade down on miss.
  - Laplace (add‑1) smoothing: returns a valid probability for any context match, even if the target token was never seen. The scored probability does **not** depend on oracle knowledge of the target.
  - Entropy‑adaptive alpha: `α = 0.08 + 0.65 × σ(2 × (H − 3.5))`. High entropy → trust n‑gram more; low entropy → trust neural model.
  - Zero artifact cost: cache is built entirely at eval time from already‑scored tokens. No stored weights or tables.
  - Score‑first, backward‑looking: `ngram_cache.update()` is called only *after* scoring each chunk.
- **Legal score‑first TTT**: SGD (lr=0.002, momentum=0.9), 3 epochs, 32K‑token chunks, stride 64, cosine LR decay.
- **Quantization**: int6 per‑row + lzma compression. CROWN‑Q penalty during late training.

## Compliance

- Training time: all seeds ≤ 600,000 ms (599,384 / 599,761 / 599,618). **Note**: the logged `train_time` starts after 20 warmup steps and model compilation. If the challenge judges end‑to‑end wallclock (including compile + warmup), the actual margin is narrower than these numbers suggest.
- Artifact size: all seeds ≤ 16,000,000 B (15,982,044 / 15,977,267 / 15,989,583).
- Score‑first TTT: each validation token is scored under `torch.inference_mode()` before any model update.
- N‑gram cache legality: **contested**. The cache is backward‑looking only, uses zero artifact bytes, and produces Laplace‑smoothed probabilities that form a proper normalized distribution. [PR #727](https://github.com/openai/parameter-golf/pull/727) (closed, 0.9674 bpb) used the same technique and spawned followup PRs (#753, #778, #782, #786). However, OpenAI opened [issue #677](https://github.com/openai/parameter-golf/issues/677) on 2026‑03‑25 questioning the legality of eval‑time cache methods. This submission may face review scrutiny regardless of score validity.
- Phase‑1 TTT (`TTT_PHASE1_ENABLED`): disabled by default (rule‑violating).
- No network access during training or eval beyond local `nvidia-smi`.

## Reproduce

The script auto‑resolves data paths relative to the repo root (via `_REPO_ROOT`), so it works from both the repo root and from within `records/track_10min_16mb/<submission>/`.

```bash
# From repo root after cloning:
cd parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024

# Seed 1337
SEED=1337 RUN_ID=seed_1337 VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-31_LeakyReLU2_LegalTTT_NGramCache_XSA/train_gpt.py

# Seed 42
SEED=42 RUN_ID=seed_42 VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-31_LeakyReLU2_LegalTTT_NGramCache_XSA/train_gpt.py

# Seed 2025
SEED=2025 RUN_ID=seed_2025 VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-31_LeakyReLU2_LegalTTT_NGramCache_XSA/train_gpt.py
```

Alternatively, override paths explicitly:
```bash
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
SEED=1337 VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Hardware: 8× H100 SXM (RunPod), CUDA 12.8, PyTorch 2.9+.

## Key Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `MAX_WALLCLOCK_SECONDS` | 600.0 | Hard training time cap (seconds) |
| `NGRAM_CACHE` | 1 | Enable N‑gram backoff cache |
| `NGRAM_ORDER` | 9 | Max n‑gram order |
| `TTT_ENABLED` | 1 | Enable legal score‑first TTT |
| `TTT_PHASE1_ENABLED` | 0 | **Off** — violates rules if enabled |
| `XSA_LAST_N` | 5 | Layers using exclusive self‑attention |
| `VE_ENABLED` | 1 | Value embedding on layers 8/9/10 |
| `QAT_ENABLED` | 0 | Quantization‑aware training |

## Eval Timing Budget (8×H100)

| Phase | Time |
|-------|------|
| Training (wallclock‑capped) | ≤ 600 s |
| Standard eval (int6 roundtrip + sliding window s64) | ~82 s |
| Legal TTT + N‑gram cache | ~420 s |
| **Total eval (timed phases)** | **~502 s** |

**Note**: the ~502 s figure covers only the timed eval phases. `torch.compile` warmup and model deserialization add additional overhead (~5–15 s) that occurs outside these timed blocks. Total end‑to‑end eval is estimated at ~515–520 s.

## Credits

Built on [modded‑nanogpt](https://github.com/KellerJordan/modded-nanogpt). Key technique credits: [PR #727](https://github.com/openai/parameter-golf/pull/727) (N‑gram backoff + entropy‑adaptive alpha), [PR #549](https://github.com/openai/parameter-golf/pull/549) (LeakyReLU² + TTT + Muon SOTA stack), [PR #461](https://github.com/openai/parameter-golf/pull/461) (score‑first TTT protocol), [PR #659](https://github.com/openai/parameter-golf/pull/659) (original N‑gram cache).
