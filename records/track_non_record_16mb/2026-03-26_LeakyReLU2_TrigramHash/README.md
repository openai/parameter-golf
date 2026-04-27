# LeakyReLU² + TrigramHashEmbedding

**val_bpb: 1.1448** (3-seed mean, sliding window stride 64) | **~15.6 MB** | validated on 1×H100 NVL 96GB with proportional wallclock

## Summary

I built on the PR #414 stack (11L EMA + GPTQ-lite) and PR #549 (LeakyReLU²) with a novel hash-based **TrigramHashEmbedding** that injects local 3-token context at the input layer. The trigram embedding extends BigramHashEmbedding (PR #198) from 2-grams to 3-grams, capturing richer local patterns like common word fragments and frequent phrases. Combined with LeakyReLU(0.5)² and the existing int6 + lzma compression pipeline, this achieves **1.1448 bpb** (3-seed mean) under the 16MB artifact cap.

## Results (1×H100 NVL 96GB, proportional wallclock 4054s)

| Seed | Steps | step_avg | Sliding val_bpb | Sliding val_loss | RT val_bpb | Artifact |
|------|-------|----------|-----------------|-----------------|------------|----------|
| 1337 | 4,145 | 978.15ms | 1.14586680 | 1.93474298 | 1.16934996 | 15,642,196 |
| 42 | 4,142 | 978.95ms | 1.14561735 | 1.93432179 | 1.16925444 | 15,587,464 |
| 2025 | 4,142 | 978.79ms | 1.14305660 | 1.92999808 | 1.16676605 | 15,591,832 |
| **Mean** | **4,143** | **978.63ms** | **1.14484692** | **1.93302095** | **1.16845682** | **15,607,164** |

All 3 artifacts under 16,000,000 bytes.

## Key Innovation: TrigramHashEmbedding

The main novel contribution. XOR-hashes three consecutive token IDs into a compact embedding table, adding local trigram context to the standard unigram token embedding:

```python
class TrigramHashEmbedding(nn.Module):
    def __init__(self, trigram_vocab_size, trigram_dim, model_dim):
        super().__init__()
        self.trigram_vocab_size = trigram_vocab_size
        self.embed = nn.Embedding(trigram_vocab_size, trigram_dim)  # 2048×48
        nn.init.zeros_(self.embed.weight)
        self.proj = CastedLinear(trigram_dim, model_dim, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.03))

    def trigram_hash(self, tokens):
        t = tokens.to(torch.int32)
        mod = self.trigram_vocab_size - 1
        out = torch.empty_like(t)
        out[..., 0] = out[..., 1] = mod  # no trigram context for first 2 tokens
        out[..., 2:] = torch.bitwise_xor(
            torch.bitwise_xor(48271 * t[..., 2:], 31547 * t[..., 1:-1]),
            17389 * t[..., :-2]
        ) % mod
        return out.long()

    def forward(self, token_ids):
        h = self.embed(self.trigram_hash(token_ids))
        return self.proj(h) * self.scale.to(dtype=h.dtype)
```

**How it works:** For each position $i \geq 2$, the hash function computes `(48271·t[i] XOR 31547·t[i-1] XOR 17389·t[i-2]) mod 2047`, mapping the trigram $(t_{i-2}, t_{i-1}, t_i)$ into one of 2048 buckets. The embedding is looked up, projected to model dimension, and scaled by a learnable gate (initialized at 0.03). Output is added to the standard token embedding.

**Design choices:**
- **Zero-init**: Both embedding and projection start at zero — the model falls back to unigram embeddings early and gradually learns to use trigram context.
- **Learnable scale gate**: Controls mixing strength; prevents the trigram signal from destabilizing early training.
- **Compact**: 2048×48 table + 48→512 projection adds only ~123K parameters, well within the 16MB budget after int6 quantization.

This extends BigramHashEmbedding (PR #198) to 3-token windows. In my ablations, a larger 6144×80 table gave −0.00214 bpb vs the baseline, but 2048×48 fits the artifact budget better.

## Architecture

Built on PR #414 stack with LeakyReLU² from PR #549:

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV GQA) |
| MLP | 3× expansion with **LeakyReLU(0.5)²** |
| **TrigramHash** | **2048×48 → project to 512** |
| BigramHash | 2048×128 |
| XSA | Last 4 layers |
| Partial RoPE | 16/64 dims |
| LN Scale | 1/√(layer+1) |
| Value Embedding | 128d on layers 9–10 |
| Tokenizer | 1024 BPE, seq_len 2048 |
| Tied embeddings | Yes |
| Weight averaging | EMA(0.997) + Tight SWA (every 50 steps) |
| Quantization | GPTQ-lite int6 + LZMA |
| Eval | Sliding window, stride 64 |

## Training Configuration

| Setting | Value |
|---------|-------|
| Optimizer | Muon (matrices) + AdamW (scalars/embeds) |
| Muon momentum | 0.99 (warmup from 0.92 over 1500 steps) |
| Muon WD | 0.04 |
| Batch tokens | 786,432 |
| Grad accum | `8 // world_size` (= 1 on 8×H100, 8 on 1×H100) |
| Warmdown iters | 3500 |
| EMA decay | 0.997 |
| Late QAT | Last ~150 steps |

## Note on Compute

I did not have access to 8×H100 SXM nodes, so I validated this submission on a single H100 NVL 96GB using a **proportional wallclock** approach to match the 8×H100 training trajectory:

1. **Gradient accumulation**: The script auto-sets `grad_accum = 8 // world_size`. On 8×H100 this is 1 (pure DDP); on 1×H100 it is 8 (accumulate 8 microbatches). Same effective batch of ~786K tokens/step.
2. **Wallclock calibration**: I measured step_avg on my H100 NVL at ~978ms/step vs ~145ms/step on 8×H100. Ratio ≈ 6.76×. So I set `MAX_WALLCLOCK_SECONDS=4054` (= 600 × 6.76) to match the 600s training window.
3. **Matched training phase**: This produces ~4,143 steps in 4054s on 1×H100, matching the ~4,100–4,250 steps expected in 600s on 8×H100. The LR warmdown, SWA, and late QAT all trigger at the same relative phase.

The script defaults to `MAX_WALLCLOCK_SECONDS=600` and should reproduce equivalent results on 8×H100 SXM via `torchrun --nproc_per_node=8 train_gpt.py` with no env overrides.

## How to Run

```bash
cd parameter-golf
pip install torch numpy sentencepiece lzma

# Download data
python data/download.py

# 8×H100 SXM (10 min, default settings)
torchrun --standalone --nproc_per_node=8 \
  records/track_non_record_16mb/2026-03-26_LeakyReLU2_TrigramHash/train_gpt.py

# 1×H100 NVL (matched, ~68 min)
MAX_WALLCLOCK_SECONDS=4054 SEED=1337 \
  python records/track_non_record_16mb/2026-03-26_LeakyReLU2_TrigramHash/train_gpt.py
```

## Files

- `train_gpt.py` — Training script (70,472 bytes)
- `submission.json` — Submission metadata
- `requirements.txt` — Dependencies
- `README.md` — This file

## Compliance

- [x] Artifact ≤ 16,000,000 bytes (max 15,642,196)
- [x] No validation data access during training
- [x] No external downloads during evaluation
- [x] Fully reproducible from provided code
- [x] ≤ 10 min on 8×H100 (validated via proportional wallclock on 1×H100)
