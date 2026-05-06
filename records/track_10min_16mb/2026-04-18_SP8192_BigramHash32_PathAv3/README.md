# Record: SP8192 + BigramHash d=32 + Path A v3 Aggressive Passthrough Quantization — val_bpb 1.07882 (3-seed mean)

**val_bpb = 1.07882** (3-seed mean, std 0.000143) | **mean 15,993,825 B (15.99 MB)** | 8× H100 80GB SXM | Legal Score-First TTT

Beats the merged SOTA ([2026-04-09 SP8192 record by @bigbag](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/README.md), 3-seed mean 1.08100) by **−0.00218 bpb / −0.00564 nats per token** on a 3-seed mean, clearing the 0.005-nat record threshold with one-sided **z = −3.00, p = 0.00136** (p < 0.01 required).

## 3-Seed Results (8× H100 80GB SXM, PyTorch 2.9.1+cu128, Legal Score-First TTT)

### Core (TTT) table

| Seed | Steps | Pre-TTT sliding bpb | **Post-TTT bpb** | TTT gain | TTT time | Artifact (B) |
|---:|---:|---:|---:|---:|---:|---:|
| 42   | 4393 | 1.08015 | **1.07887** | −0.00128 | 336.1 s | 15,991,203 |
| 314  | 4393 | 1.08024 | **1.07893** | −0.00131 | 335.5 s | 15,994,170 |
| 999  | 4403 | 1.07998 | **1.07866** | −0.00132 | 333.6 s | 15,996,103 |
| **mean** | | **1.08012** | **1.07882** | **−0.00130** | **335.1 s** | **15,993,825** |
| **std**  | |         | **0.000143** |          |          |            |

### Diagnostics

| Seed | Post-EMA bpb | Quant roundtrip bpb | Sliding bpb | TTT val_loss (nats) | Code bytes | Total submission (B) | Train ms | Eval ms (q+sl+ttt) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42   | 1.08584 | 1.09678 | 1.08015 | 2.78662485 | 18,097 | 15,991,203 | 588,110 | 480,408 |
| 314  | 1.08580 | 1.09679 | 1.08024 | 2.78678778 | 18,097 | 15,994,170 | 588,031 | 479,495 |
| 999  | 1.08561 | 1.09662 | 1.07998 | 2.78608265 | 18,097 | 15,996,103 | 588,029 | 477,724 |
| **mean** | **1.08575** | **1.09673** | **1.08012** | **2.78650** | — | **15,993,825** | **588,057** | **479,209** |

## Key Innovation: Path A v3 Aggressive Passthrough Quantization

Two complementary changes on top of the [2026-04-09 SP8192 stack](../2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/README.md):

### 1. `BIGRAM_DIM = 32`

BigramHashEmbedding dimension reduced from the common d=48/64 to **d=32**. Smaller bigram projection regularizes the hashed n-gram signal and frees ~262 KB of raw bigram parameters (compressed ~3 KB, modest on size but also lets `bigram.proj` be even smaller for the Path A v3 int8 treatment). Pre-quant post-EMA is preserved at ~1.0858, within noise of the d=48 baseline.

### 2. Path A v3 Aggressive Passthrough Quantization (primary contribution)

The canonical [PR #1394](https://github.com/openai/parameter-golf/pull/1394) / bigbag stack leaves the following tensors as **fp16 passthrough** (1 tensor per transformer block layer plus a few scalars), consuming ~40 KB in the compressed artifact:

- **Control tensors (per-tensor int8)**: `attn_scale`, `mlp_scale`, `resid_mix`, `skip_gates`, `skip_weights`. Each is a small 1-D array with a narrow dynamic range. Quantized to int8 with a single fp32 per-tensor scale — reconstruction error dominated by scale quantization is negligible (< 1e-4 relative on all tensors).
- **Small 2-D matrices (per-row int8)**: `bigram.proj` (512 × 32 = 16 K params), `attn_gate_proj`, `smear_gate.weight`. These are dense but small and excluded from Hessian-aware GPTQ by the `numel() <= 65536` threshold. Quantized to int8 with per-row fp16 scales.
- **`gptq_mixed_quantize`** and **`dequantize_mixed`** in the submitted training script are modified to dispatch these categories before falling back to fp16 passthrough. Everything else (int6 attn/MLP matrices, int8 tok_emb, int6 bigram.embed) is unchanged.
- **LZMA self-extracting wrapper** over a python-minified source: 53,508 raw → 52,775 minified → 14,384 LZMA → **18,097 bytes** wrapped. (Same technique as @bigbag's record.)

**Quantization quality cost:** measured to 5 d.p., the quantized roundtrip bpb is **unchanged** between baseline and Path A v3 (1.0968 in both). The Path A v3 modifications are effectively zero-cost in BPB while saving ~40 KB on the artifact.

**Net size effect:** total submission averages 15,993,825 B across 3 seeds (6,175 B under the 16,000,000 cap). Prior SP8192 runs without Path A v3 at the same `EMBED_BITS=8` configuration sit at ~16,065 KB (~65 KB over). Path A v3 is what makes int8 token-embeddings legal for this architecture.

## Architecture

11L × 512d × 8H / 4KV, MLP 4×, LeakyReLU(0.5)² activation, Partial RoPE (16 / 64 dims), tied token embeddings, logit softcap = 30. Skip gates (sigmoid-gated U-Net connections). Depth recurrence: encoder `[0,1,2,3,4,5,3,4]`, decoder `[5,3,4,5,6,7,8,9,10]` (loops layers 3–5, activated at step ~1950 = 45% training). Parallel residuals from layer 7. **BigramHashEmbedding with 16,384 buckets × d=32**. AttnOutputGate (width 12, source=proj). SmearGate (width 12). SentencePiece-BPE 8192.

## Training

MuonEq-R (row-normalized Muon, Newton-Schulz 5 steps) for matrices; AdamW for embeddings/scalars. Warmdown 72% of training; EMA decay 0.9965. QK-Gain init 5.0 (learnable per-head). Weight decay 0.085 / 0.095 (embed / matrix). ~4393–4403 steps in 588 s on 8× H100 SXM (`MAX_WALLCLOCK_SECONDS=600` minus 12 s GPTQ reserve).

## Quantization

Full-Hessian GPTQ with SDClip (`clip = k × std(row)`):

- **Matrices** (attn/MLP): int6, `matrix_clip_sigmas = 12.85` (@clarkkev PR #1394)
- **Token embeddings**: int8, `embed_clip_sigmas = 20.0`
- **bigram.embed**: int6 per-row simple scale
- **Path A v3 additions** (this PR): per-tensor int8 for control scalars, per-row int8 for small 2-D matrices (see Key Innovation section)

Byte-shuffle + Brotli-11 on the quantized state-dict blob. Self-extracting LZMA wrapper on the minified source.

## Test-Time Training (Score-First, Legal)

Per [Issue #1017](https://github.com/openai/parameter-golf/issues/1017) / [PR #549](https://github.com/openai/parameter-golf/pull/549) / [PR #461](https://github.com/openai/parameter-golf/pull/461) precedent:

```python
for chunk_idx, chunk_windows in enumerate(chunks):
    # Phase 1: SCORE (under no_grad, no parameter update)
    with torch.inference_mode():
        nll = model.forward_logits(batch).cross_entropy(targets)
    loss_sum += nll.sum()

    # Phase 2: TRAIN (only on the chunk just scored)
    if not is_last_chunk:
        for _ in range(ttt_epochs):   # 3 epochs
            for x, y in chunk_seqs:
                loss = model(x, y)
                loss.backward()
                optimizer.step()      # SGD, lr=0.005, momentum=0.9
```

1,238 chunks × 32,768 tokens × 3 epochs. Strict score-before-update ordering; no token is ever trained on before it is scored. Mean TTT time 335 s per seed (well within 600 s eval budget).

## Rule Compliance

Per [repo README](../../../README.md) and [Issue #1017](https://github.com/openai/parameter-golf/issues/1017):

- **Condition 1 — Causality** ✅ Strictly causal forward pass. Sliding-window eval never references future tokens for current-position scoring.
- **Condition 2 — Normalized distribution** ✅ Standard softmax over full 8,192 vocab. No n-gram cache, no logit biasing, no multi-pass rescoring.
- **Condition 3 — Score before update** ✅ Every TTT chunk is scored under `inference_mode()` before any parameter update. Gradient updates only use already-scored tokens.
- **Condition 4 — Single pass** ✅ Each val token is scored exactly once. No rescoring, no cache lookups.

Additional:
- **No SLOT** (standard or causal) — no eval-time delta optimization
- **No pre-quant TTT** on val data — model is quantized once; TTT adapts the quantized model at eval time only
- **No ETLB** (eval-time logit bias)
- **No n-gram cache** or tilt
- **Seed choice conventional** — matches @bigbag 2026-04-09 exactly (42, 314, 999); no seed brute-forcing
- **Artifact < 16,000,000 bytes** on all 3 seeds (margins: 8,797 / 5,830 / 3,897 B)
- **Training ≤ 600 s** on all 3 seeds (588,029–588,110 ms actual)
- **Eval ≤ 600 s** on all 3 seeds (quantized + sliding + TTT = 477,724–480,408 ms)

## Statistical Evidence

Three independent seeds on a canonical 128-shard sp8192 tokenization of the `willdepueoai/parameter-golf` fineweb export:

```
Seed 42:  val_bpb = 1.07886574, val_loss = 2.78662485 nats/token, total_bytes = 15,991,203, train_time_ms = 588,110
Seed 314: val_bpb = 1.07892882, val_loss = 2.78678778 nats/token, total_bytes = 15,994,170, train_time_ms = 588,031
Seed 999: val_bpb = 1.07865582, val_loss = 2.78608265 nats/token, total_bytes = 15,996,103, train_time_ms = 588,029

Mean bpb      = 1.07881679
Std bpb       = 0.000143 (sample, n=3, n-1=2)
SEM bpb       = 0.0000826
Mean val_loss = 2.78649843 nats/token
bpb / val_loss ratio = 0.387159 (per-pod byte-count mapping)

Merged SOTA (bigbag 2026-04-09 3-seed mean) = 1.08100 bpb
Observed delta                              = 0.00218 bpb  =  0.00564 nats/token  (> 0.005-nat threshold)
Threshold in bpb at our ratio               = 0.001936 bpb
Mean bpb required to clear threshold        = 1.079064
Our mean bpb                                = 1.078817
Margin past threshold                       = 0.000247 bpb  =  0.000637 nats/token

One-sided z (lower tail)                    = (1.078817 − 1.079064) / 0.0000826 = −2.998
One-sided p-value                           = 0.00136
Required: p < 0.01                          →  CLEARED
```

## Environment

```
torch                2.9.1+cu128
CUDA                 12.8
NVIDIA driver        575.57.08
brotli               1.2.0
sentencepiece        0.2.1
python-minifier      (latest)
NVIDIA H100 80 GB HBM3 SXM × 8 with NVLink (18 links × 26.562 GB/s)
NCCL all-reduce 256 MB: ~424 GB/s bus bandwidth (near-peak NVLink4)
```

## Reproduction

```bash
# 1. Install deps
pip install --break-system-packages brotli python-minifier sentencepiece huggingface_hub

# 2. Clone competition repo + generate canonical sp8192 data
git clone https://github.com/openai/parameter-golf.git repo
cd repo

cat > data/tokenizer_specs_sp8192.json <<'EOF'
{"tokenizers":[{"name":"sp_bpe_8192","dataset_suffix":"sp8192","vocab_size":8192}]}
EOF

python3 data/download_hf_docs_and_tokenize.py \
    --repo-id willdepueoai/parameter-golf \
    --remote-root datasets \
    --output-root ./data \
    --tokenizer-config data/tokenizer_specs_sp8192.json \
    --skip-byte \
    --chunk-tokens 100000000 \
    --tokenizer-train-docs 1000000

# 3. Run 3 seeds
for SEED in 42 314 999; do
  SEED=$SEED DATA_DIR=./data/ RUN_ID=seed${SEED} \
    ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 \
    TTT_ENABLED=1 SLIDING_WINDOW_ENABLED=1 VAL_LOSS_EVERY=4000 \
    BIGRAM_VOCAB_SIZE=16384 BIGRAM_DIM=32 \
    GATE_ATTN_OUT=1 GATE_WIDTH=12 GATE_ATTN_SRC=proj \
    SMEAR_GATE=1 SMEAR_GATE_WIDTH=12 \
    EMBED_BITS=8 EMBED_CLIP_SIGMAS=20.0 COMPRESSOR=brotli \
    torchrun --standalone --nproc_per_node=8 train_gpt.py \
      2>&1 | tee logs/train_seed${SEED}.log
done
```

The provided `train_gpt.py` is an 18,097-byte LZMA self-extracting wrapper. The equivalent full source (53,508 B) is `train_gpt_stacked_v2_fixed.py` for review.

## Credits

- **@clarkkev** — PR #1394: SP8192 base stack + GPTQ SDClip + int6 matrices / int8 embeddings + MuonEq-R + SP8192 tokenizer recipe.
- **@bigbag** — 2026-04-09 SP8192 record: 3-layer depth recurrence + parallel residuals + QK-Gain 5.25 + legal TTT on the SP8192 stack. (Direct ancestor of this submission.)
- **@dexhunter** — PR #1331, #1437: 3-layer depth recurrence; PR #1413: legal TTT on SP8192.
- **@Robby955** — PR #1412: parallel residuals on SP8192. **@msisovic** — PR #1204: parallel residuals concept.
- **@Christopher-Lee-McClendon** — PR #461: legal score-first TTT framework. **@abaybektursun** — PR #549: merged precedent for legal TTT.
- **@MarioPaerle** — PR #1667: AttnOutputGate used in this architecture.

## Our contribution

Two modifications on top of the @bigbag / @clarkkev SP8192 lineage:

1. **Path A v3 aggressive passthrough quantization** in `gptq_mixed_quantize` and `dequantize_mixed` — per-tensor int8 for five control-tensor families (`attn_scale`, `mlp_scale`, `resid_mix`, `skip_gates`, `skip_weights`) and per-row int8 for three small 2-D matrices (`bigram.proj`, `attn_gate_proj`, `smear_gate.weight`). Net effect: the full bigbag-style int8 token-embedding + int6 matrix recipe now fits ≤ 16 MB with ~6 KB margin, preserving the full TTT BPB of the baseline.
2. **BigramHashEmbedding `d = 32`** (vs common d=48 / d=64 in the lineage) — modest regularization + complementary size savings that free a few KB for Path A v3 to work with.
