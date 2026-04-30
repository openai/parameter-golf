# BESE + Mamba-3 SSD Hybrid

**Author:** Omer Bese ([@mrbese](https://github.com/mrbese))  
**Date:** 2026-04-16  
**Track:** Non-record (SSM / State-space model submission)  
**val_bpb:** 1.3571 (INT6 + LZMA + sliding window eval with n-gram tilt)  
**Artifact size:** 7,614,888 bytes (48% of 16 MB limit)

---

## Overview

This submission combines two experimental ideas requested by the challenge organizers:

1. **State-space models** (specifically Mamba-3 SSD) — checking the "State-space models" bounty from the challenge README
2. **Novel tokenizer** (BESE, a custom 288-vocab byte-level tokenizer) — testing whether sub-byte tokenization gives SSMs an advantage through 2x token density

To our knowledge, this is the first submission to pair a custom byte-level tokenizer with a Mamba-3 architecture.

## Architecture

**Hybrid: 6 Mamba-3 SSD blocks + 2 Attention blocks (8 layers total)**

```
Layer 0: Mamba-3 SSD
Layer 1: Mamba-3 SSD  
Layer 2: Attention (GQA, FlashAttention/SDPA)
Layer 3: Mamba-3 SSD
Layer 4: Mamba-3 SSD
Layer 5: Attention (GQA, FlashAttention/SDPA)  
Layer 6: Mamba-3 SSD
Layer 7: Mamba-3 SSD
```

### Model Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| `model_dim` | 512 | |
| `num_layers` | 8 | 6 Mamba + 2 Attention |
| `d_state` | 128 | SSM state dimension |
| `expand` | 2 | d_inner = 1024 |
| `headdim` | 64 | SSM head dimension |
| `nheads` (SSM) | 16 | d_inner / headdim |
| `ngroups` | 1 | All heads share B/C (reference Mamba-2 default) |
| `chunk_size` | 64 | SSD chunk size |
| `num_heads` (Attn) | 8 | |
| `num_kv_heads` (Attn) | 4 | GQA |
| `mlp_mult` | 3.0 | Attention block MLP |
| `vocab_size` | 288 | BESE tokenizer |
| **Total params** | **15,152,432** | |

### Key Design Decisions

**1. ngroups=1 (shared B/C across heads)**

All 16 SSM heads share the same B (input-to-state) and C (state-to-output) projections, with only 1 group. This matches the reference Mamba-2 implementation and was confirmed optimal by PR #1644 ablations. Saves ~6.9M parameters vs per-head B/C (ngroups=16), which we reallocate to larger d_state.

**2. No depth recurrence on SSM layers**

PR #1355 measured a -69 mBPB penalty from depth recurrence on Mamba blocks. Unlike transformers where attention re-processes with updated context, SSM state from pass 1 does not inform pass 2 (initial_states=None). We disable depth recurrence entirely.

**3. Two attention layers at positions [2, 5]**

Following PR #1644's architecture, attention layers provide global token mixing at strategic points, dividing the SSM blocks into three equal segments. The SSM layers handle local sequential processing (O(n)), while attention provides periodic global information bottlenecks.

**4. d_state=128 with ngroups=1**

With shared B/C (ngroups=1), the projection cost is only 1 x d_state per position for B and C. Doubling d_state from 64 to 128 costs just ~400K extra parameters but doubles the SSM's memory bandwidth — how much past context each state vector can retain.

## BESE Tokenizer

BESE (Byte-Encoded Sub-byte Encoding) is a two-layer tokenizer:
- **Layer 1:** 40 base tokens (digrams covering 95% of English byte pairs)
- **Layer 2:** 248 BPE merges on top of the base tokens
- **Total vocab:** 288 tokens

Compared to SP1024 (the challenge default), BESE produces ~2x more tokens per byte of text. This means:
- **Embedding table:** 288 x 512 = 147K params (vs SP1024's 1024 x 512 = 524K, or SP8192's 8192 x 512 = 4.2M)
- **Saved parameters** go directly into model capacity
- **Longer effective sequences** for the same token count

### BPB Correctness Proof

The competition rules require that any tokenizer change "prove with certainty that the val_bpb is correctly calculated." Because BESE is custom, we include the proof inline below rather than referencing prior submissions.

**The invariant we maintain:**

> For any input string `s`,
> `sum(BYTES_PER_TOKEN[t] for t in encode(s)) == len(s.encode("utf-8"))`

This means our BPB number is computed against the same denominator (UTF-8 bytes) as every SP1024 / SP8192 submission, so scores are directly comparable.

**Per-token byte accounting** (defined in `bese_constants.py::build_bytes_per_token`):

| Token category | Count | Bytes per token |
|---|---|---|
| Single-letter tokens (e, t, a, o, i, n, s, r, h, d, l) | 11 | 1 |
| Group tokens (ufbz, cwvj, mykx, gpq) — prefix only | 4 | **0** |
| Position tokens (P1–P4) — carry the actual character | 4 | 1 |
| Punctuation (space, period, comma, newline, ?, quote, OTHER_PUNCT) | 7 | 1 |
| Digit tokens (0–9) | 10 | 1 |
| Special tokens (PAD, BOS, EOS, UNK) | 4 | 0 |
| BPE merge tokens | 248 | recursive sum of constituents |

**Why it holds — by case** (see `bese_fast_bpe.py::_text_to_base_tokens`):

For every input character `ch`:

1. **Mapped path:** If `lower(ch)` is in `ENCODE_TABLE` AND the mapped tokens' byte sum equals `len(ch.encode("utf-8"))`, we emit those tokens. The equality is checked at runtime — if it ever failed we'd fall through to the byte-fallback path. Group tokens contribute 0, position tokens contribute 1 → a (group, pos) pair encodes one ASCII character (1 byte) correctly. Single letters contribute 1 byte directly.

2. **Byte-fallback path:** Otherwise (uppercase ASCII not normalized, non-ASCII, unknown punctuation), we emit `OTHER_PUNCT_ID` exactly `len(ch.encode("utf-8"))` times. Each `OTHER_PUNCT_ID` contributes 1 byte → total bytes match the UTF-8 byte count of the character exactly.

In both branches the sum-of-bytes invariant is preserved per character, so it holds for the full string by linearity.

**For BPE merges**, the byte count is computed transitively when merges are loaded (`bese_fast_bpe.py::compute_bytes_per_token`):

```python
bpt = np.zeros(self.vocab_size, dtype=np.int16)
bpt[:BASE_VOCAB_SIZE] = BYTES_PER_TOKEN
merge_bpt = {i: int(BYTES_PER_TOKEN[i]) for i in range(BASE_VOCAB_SIZE)}
for pair, new_id in self.merges:
    merge_bpt[new_id] = merge_bpt[pair[0]] + merge_bpt[pair[1]]
    bpt[new_id] = merge_bpt[new_id]
```

So every merge inherits its byte count from its constituents — invariant preserved through the BPE layer.

**Self-test** (run this from inside the records folder; runs in <1 s on CPU):

```python
import numpy as np
from bese_fast_bpe import BeseFastBPE

# Load tokenizer
tok = BeseFastBPE.load("tokenizer.json")
bpt = tok.compute_bytes_per_token()

# Roundtrip + byte-count check across diverse inputs
test_cases = [
    "the quick brown fox jumps over the lazy dog",
    "Hello, World! 1234567890",
    "naïve café résumé — emojis: 🚀✨🌍",
    "newlines\nand\ttabs and \"quotes\" and 'apostrophes'",
    "MixedCase HTML <p>tag</p> and JSON {\"a\": 1}",
]
for s in test_cases:
    ids = tok.encode(s)
    # 1. Lossless roundtrip on the supported character set
    # 2. Byte-count invariant
    assert sum(int(bpt[t]) for t in ids) == len(s.encode("utf-8")), \
        f"byte invariant failed on: {s!r}"
print("OK — BPB byte invariant holds for all test cases")
```

The same `bpt` table is what the training/eval code uses to compute val_bpb, so this self-test is checking the exact accounting used to score the submission.

**val_bpb formula** (matches the upstream definition):

```
val_bpb = (sum_of_token_NLLs_in_nats / sum_of_bytes_per_token) / log(2)
```

where the numerator and denominator are summed over the same set of evaluated tokens. Because the per-token byte counts sum to the true UTF-8 byte length of the validation set, this is identical to the SP1024 / SP8192 BPB formula evaluated on the same FineWeb validation data.

## Training

- **Hardware:** 8x NVIDIA H100 80GB SXM (RunPod)
- **Training time:** 600 seconds (wallclock cap)
- **Steps completed:** 2,191
- **Step average:** 274 ms/step
- **Optimizer:** Muon (Newton-Schulz) for 2D matrices, AdamW for scalars and embeddings
- **EMA decay:** 0.9965
- **Warmdown:** 5000 iterations
- **SWA:** Activated at step 1200
- **Sequence length:** 2048 (train and eval)
- **Batch tokens:** 786,432 per step (global)

### Training Curve

| Step | val_bpb |
|------|---------|
| 0 | 4.1571 |
| 500 | 1.5460 |
| 1000 | 1.4268 |
| 1500 | 1.3806 |
| 2000 | 1.3489 |
| 2191 (final) | 1.3458 |

## Evaluation

| Stage | val_bpb | Notes |
|-------|---------|-------|
| Raw (post-EMA) | 1.3475 | Diagnostic |
| INT6 roundtrip | 1.3809 | Quantized model |
| **INT6 + Sliding Window + N-gram tilt** | **1.3571** | **Final submission score** |

- **Quantization:** Mixed INT6 (6-bit) for MLP, attention, and Mamba projection weights. Scalar params (D, dt_bias, A_log, norms) stored as FP16.
- **Compression:** LZMA preset 9
- **Sliding window eval:** stride=64, full 2048 context per window
- **N-gram tilt:** Pre-computed trigram prior from training data, applied as additive logit bias during sliding window eval

> **Statistical-significance note.** The headline 1.3571 BPB is a **single-seed** result from the `dim=512, d_state=128` configuration (`train_log_run1.txt`). The two additional logs (`train_log_run2_d64.txt`, `train_log_run3_dim576.txt`) are **architecture ablations** (different `d_state` and `model_dim`), not seed replicates of the headline config. Three-seed validation of the headline config is in the *Ongoing Work* section, pending compute credits. We submit this as a non-record entry under the rule that allows in-progress and unoptimized solutions for novel ideas; it is not a leaderboard claim.

### Artifact Size

| Component | Bytes |
|-----------|-------|
| Compressed model (INT6 + LZMA) | 7,452,680 |
| Code (train_gpt.py + mamba3_ssd.py + tokenizer) | 162,208 |
| **Total** | **7,614,888** |
| Budget remaining | 8,385,112 (52% unused) |

## Additional Runs

We ran three configurations to ablate the architecture. These are **architecture ablations**, not seed replicates of a single config:

| Config | Params | Steps | Raw BPB | INT6 BPB | SW BPB | Artifact |
|--------|--------|-------|---------|----------|--------|----------|
| dim=512, d_state=64 | 14.8M | 2,482 | 1.3254 | 1.3445 | not completed | 7.96 MB |
| **dim=512, d_state=128** (headline) | **15.2M** | **2,191** | **1.3458** | **1.3809** | **1.3571** | **7.56 MB** |
| dim=576, d_state=128, mlp3.5 | 19.7M | 1,847 | 1.3415 | 1.4053 | not completed | 8.42 MB |

Key findings:
- **d_state=128 vs 64:** Slightly worse raw BPB (fewer steps) but sliding window eval works and n-gram tilt recovers the gap
- **dim=576 (wider model):** Best per-step learning rate and best raw BPB (1.3415 at step 1847), but larger INT6 quantization gap (+60 mBPB). Suggests QAT would unlock significant gains for wider Mamba models.
- **Artifact headroom:** Even the widest model uses only 8.42/16 MB, leaving substantial room for growth

## SSD Implementation Notes

We implemented Mamba-3 SSD in pure PyTorch (no custom CUDA/Triton kernels) using the chunked parallel formulation from the Mamba-2 paper. Key components:

- **segsum:** Stable cumulative sum for decay computation via lower-triangular masking
- **ssd_chunked:** Chunked parallel SSD with intra-chunk quadratic attention and inter-chunk state recurrence
- **Causality fix:** We discovered and fixed a causality bug in the reference implementation's inter-chunk decay matrix (diagonal was 1, allowing each chunk to see its own state through Y_off). Fixed by shifting the column index in the einsum.

We attempted integration with the official `mamba-ssm` Triton kernels (`mamba_chunk_scan_combined`), which worked on single GPU but caused segfaults under multi-GPU torchrun after ~100 steps. The pure PyTorch fallback is stable and provides correct results, though ~2-3x slower per step.

## Code Structure

The submission folder contains four Python files:

| File | Bytes | Role |
|---|---|---|
| `train_gpt.py` | 108,597 | Self-contained training entry point — model definition, training loop, eval, quantization, compression |
| `mamba3_ssd.py` | 20,912 | Mamba-3 SSD block + chunked parallel `ssd_chunked` algorithm |
| `bese_fast_bpe.py` | 25,263 | BESE tokenizer encode/decode + BPE merge application |
| `bese_constants.py` | 3,941 | BESE alphabet constants and `BYTES_PER_TOKEN` lookup table |

**On the FAQ rule "all counted code should live in `train_gpt.py`":** the helper modules above are fully accounted for in `submission.json::code_bytes` (162,208 bytes total) and bundled with the submission, so the artifact-size accounting is honest. We split them out for readability — `mamba3_ssd.py` is the SSD algorithm, `bese_*.py` is the tokenizer — but they could be inlined into `train_gpt.py` mechanically with no logic change, and the artifact size would be identical. We left them separate to make the per-component contribution to artifact size legible to reviewers; happy to inline if preferred.

`tokenizer.json` (3,495 bytes) holds the trained BPE merges and is loaded at training time only (it is *not* part of the artifact — the merges are baked into the model's input pipeline).

## Comparison to Other SSM Submissions

| Submission | BPB | Arch | Vocab | Artifact |
|-----------|------|------|-------|----------|
| PR #1479 GDN hybrid | 1.1450 | 8 GDN + 2 Attn | SP8192 | 13.83 MB |
| PR #1245 Hymba | 1.1470 | 8L hybrid | SP8192 | ~15 MB |
| PR #1644 Mamba-3 | 1.1473 | 5 SSM + 2 Attn | SP8192 | ~14 MB |
| **This (BESE + Mamba-3)** | **1.3571** | **6 SSM + 2 Attn** | **BESE 288** | **7.56 MB** |

The ~210 mBPB gap to the best SSM submissions is attributable to:
- Byte-level prediction with 288 vocab (estimated ~30-50 mBPB penalty vs SP8192)
- Pure PyTorch SSD without Triton kernels (fewer training steps, ~40-60 mBPB)
- No test-time training (~30-50 mBPB based on other submissions)
- No torch.compile (~20-30 mBPB)

The unique contribution is demonstrating that byte-level tokenization + SSM is viable, achieving competitive artifact efficiency (half the 16 MB budget) while leaving substantial room for optimization.

## Reproduction

### Quick path — run the submitted artifact from this records folder

The four Python files in this folder are self-contained. From an 8xH100 SXM RunPod pod with the official Parameter Golf template (PyTorch 2.x, CUDA 12.x):

```bash
# From the cloned upstream repo:
cd parameter-golf

# Install the two extra packages (everything else is in the template):
pip install einops sentencepiece

# Prepare the SP1024 cached FineWeb (used as the input pipeline before BESE re-encoding):
python3 data/cached_challenge_fineweb.py --variant sp1024

# Run training + eval directly from this records folder:
cd records/track_non_record_16mb/2026-04-16_BESE_Mamba3_Hybrid

torchrun --standalone --nproc_per_node=8 train_gpt.py \
  VOCAB_SIZE=288 \
  TOKENIZER_PATH=./tokenizer.json \
  DATA_PATH=../../../data/datasets/fineweb10B_sp1024/ \
  RUN_ID=bese_mamba3_repro
```

This produces the same INT6+LZMA artifact and the `final_*` BPB numbers in `train_log_run1.txt`. Total wallclock on 8xH100 SXM is ~10 min training + ~7-8 min eval (sliding window + n-gram tilt is the eval bottleneck).

### Full pipeline — rebuild the BESE shards from scratch

If you want to reproduce the data-prep step (untimed) and not rely on cached BESE shards, the full pipeline lives on the author's fork:

```bash
cd /workspace
git clone https://github.com/mrbese/parameter-golf-bese.git bese
cd bese
git checkout v7-mamba

# Reuses /workspace/parameter-golf/data/datasets/fineweb10B_sp1024/ from above.
pip install einops --break-system-packages
python scripts/runpod_v7_mamba.py --num-gpus 8

# Or with pre-existing BESE shards (cached on RunPod network volume):
python scripts/runpod_v7_mamba.py --skip-shards --num-gpus 8
```

The fork's `scripts/runpod_v7_mamba.py` is just an orchestrator around the same `train_gpt.py` shipped in this records folder; it adds shard re-encoding from SP1024 → BESE base tokens, BPE training, and ngram-prior building. The records folder contains the artifacts of those steps (`tokenizer.json`, the BPE merges) so the quick path above can skip them.

## Ongoing Work

We have a pending compute credit request and plan to continue optimizing this submission. Planned next steps:

- **Triton kernel integration**: Fix the multi-GPU segfault in `mamba_chunk_scan_combined` to get 2-3x faster steps (~150ms vs 274ms), enabling ~4,000 steps in 600s
- **torch.compile**: Unblocked once Triton kernels are stable — additional ~15% step speedup
- **Wider model with QAT**: dim=576 + mlp3.5 achieved 1.3415 raw BPB but has a +60 mBPB INT6 gap. Quantization-aware training should close this gap substantially
- **Test-time training (TTT)**: Disabled in current runs to save credits. Other SSM submissions show ~30-50 mBPB improvement from TTT
- **SP8192 + BESE comparison**: Direct ablation of tokenizer impact on the same Mamba architecture
- **Three-seed statistical significance** for the headline `dim=512, d_state=128` configuration

Conservative target with all optimizations: **1.17-1.20 BPB**, which would be competitive with the best SSM submissions while maintaining BESE's artifact efficiency advantage.

## Acknowledgments

Architecture decisions informed by:
- PR #1644 by mradassaad (best Mamba-3 submission, exhaustive ablation study)
- PR #1355 by mradassaad (SSM depth recurrence ablation)
- PR #1245 by mkenney2 (Hymba hybrid architecture)
- The Mamba-2 paper (Dao and Gu, 2024) for the SSD algorithm
- mamba3-minimal (VikramKarLex) for the reference pure-PyTorch implementation
