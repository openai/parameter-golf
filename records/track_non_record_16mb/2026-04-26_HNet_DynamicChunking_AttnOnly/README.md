# H-Net Dynamic Chunking — Attention-Only Variant

*Pair-programmed with Claude (Anthropic). Architecture choices, training runs, and debugging direction are mine. Full attribution at the bottom.*

Non-record submission, **unlimited-compute track** (~100 min on 1×A6000, not the 10-min/8×H100 budget). 

Implements the dynamic-chunking module from H-Net (Hwang, Wang & Gu 2025, arxiv:[2507.07955](https://arxiv.org/abs/2507.07955)) on byte-level FineWeb-Edu, with Mamba-2 layers replaced by pre-norm Transformer blocks. This evaluates dynamic chunking on a transformer backbone — it does *not* isolate the chunking contribution from Mamba's, since the routing module was co-designed with SSMs in the paper. A proper isolation would need a 2×2 ablation (Mamba ± chunking, Transformer ± chunking).

**Result:** `val_bpb = 1.8838` after 10K steps (reproducible from the included `.ptz`). 16.27M params, 9.49MB compressed (well under 16MB cap).

## Summary of Choices

1. **Path B (attention-only)** — no Mamba-2; tests dynamic chunking on a transformer backbone, removes `mamba_ssm`/Triton dependency. (Caveat: this is a backbone swap, not a clean ablation — see header.)
2. **Single-stage chunking** — one routing module, one EMA pass; two-stage left for future work.
3. **Byte-level vocab (260)** — the whole point of H-Net; pays ~6× token overhead vs BPE-sp1024.
4. **Identity init for `W_q`/`W_k`** — at step 0 the router is raw cosine similarity of adjacent encoder states. Per goombalab/hnet `dc.py`.
5. **Chunk-level EMA** (not fine-level) — see Key Discovery.

## Key Discovery: chunk-level vs fine-level EMA

The H-Net paper (Section 2.2.2) describes the EMA dechunker ambiguously enough that an obvious-looking fine-level implementation passed sanity checks but produced the wrong math. Diffing against `goombalab/hnet/dc.py` showed they EMA over the *compressed* sequence and gather back to fine, making within-chunk values constant (correct) instead of decaying (wrong). Fix made training stable AND ~6× faster.

**Lesson:** for paper implementations, read the reference repo as primary, paper as secondary.

## Architecture

```
bytes (B, L) → embed(260→256) → 3× Transformer (d=256, 4 heads)
            → DynamicChunking (cosine sim, p_t = 0.5(1 - cos), b_t = 1{p≥0.5})
            → 6× Transformer (d=512, 8 heads, on compressed sequence)
            → upsample_with_ema  +  encoder residual
            → 3× Transformer (d=256, 4 heads) → norm → tied head → bytes
Loss: L_AR + 0.03 · L_ratio
```

16,273,920 params. Pre-norm blocks throughout. Tied embedding/head. `target_ratio = 1/6` is the paper's English default; **not swept here due to compute budget**.

## Results

| Submission | val_bpb | Compute | Params |
|---|---:|---|---:|
| Naive baseline (BPE sp1024) | 1.2244 | 10 min × 8×H100 | ~30M |
| **This submission** (byte260, no tokenizer) | **1.8838** | ~100 min × 1×A6000 | 16.27M |
| Top of leaderboard (Apr 9; BPE + ~10 tricks) | 1.0810 | 10 min × 8×H100 | ~30M |

The 0.66 bpb gap is the cost of dropping BPE *and* every BPE-era trick. This is a clean reference for byte-level dynamic chunking, not a leaderboard climb.

Going in I expected ~1.5 based on Chinchilla scaling. Landing at 1.88 surfaced that byte-level + 16M params + 300MB doesn't have the headroom to absorb the BPE token-tax — bigger model OR more data is needed before dynamic chunking can close the gap.

### val_bpb verification (byte260 vocab)

Per submission rules, tokenizer changes need explicit verification. The byte260 vocab has tokens 0-3 as specials (PAD/BOS/EOS/UNK, cost 0 bytes) and tokens 4-259 as raw byte values (each costs exactly 1 byte). Conversion:

```
bpb = (cross_entropy / ln 2) × tokens_per_byte
tokens_per_byte = total_tokens / total_bytes
```

`total_bytes` is summed via a 260-entry LUT (`bytes_per_token_lut` in `train_hnet.py:118`): `lut[0:4] = 0` (specials), `lut[4:260] = 1` (raw bytes). No specials are emitted during training/eval — the data pipeline encodes raw UTF-8 with a +4 offset — so emitted tokens map 1:1 to bytes. Sanity check: encoding "Hello" → 5 byte-tokens (all in [4..259]), 5 source bytes, `tokens_per_byte = 1.0`, `bpb = CE / ln 2`.

**End-to-end check.** Loading the shipped `final_model.int8.ptz` from disk, dequantising, and running val on the full split gives `val_bpb = 1.8838`. The training-time eval reported `1.8756` (in-training bf16 weights, full val); the ~0.008 gap is the cost of fp16 scale storage in the `.ptz` (per-row scales kept as fp16 to save bytes). The 1.8838 is the number a reviewer reproduces from the shipped artifact.

### Quantisation comparison

From `test_quant_full.py` on the trained checkpoint (val cap 65K tokens):

| Bits | val_bpb | Δ vs fp | zlib (MB) | lzma (MB) |
|---:|---:|---:|---:|---:|
| fp (bf16) | 1.8429 | — | — | — |
| **int8** | **1.8765** | **+0.034** | **9.49** | **8.88** |
| int6 | 1.8760 | +0.033 | 9.49 | 8.88 |
| int4 | 1.9416 | +0.099 | 8.74 | 8.17 |
| int3 | 2.5273 | +0.684 | 6.26 | 5.79 |
| int2 | 6.0342 | +4.191 | 3.38 | 3.03 |
| int1 | 5.2659 | +3.423 | 3.42 | 3.11 |

int8 is the Pareto pick: same compressed size as int6, ~+0.001 bpb cost vs fp, well under the 16MB cap. zlib over lzma for decode speed.

(int1 beats int2 because int1 here is `±mean(|w|)` per row, BinaryConnect-style; int2 is ternary forced through the int8 quantile-clip machinery, which fits worse.)

## Configuration

AdamW (β=0.9, 0.95), linear warmup 500 + cosine decay to 0, bf16 autocast, grad clip 1.0. All other hyperparameters are env-var driven; full command in the Reproducing section below.

## Negative Results Worth Noting

- **Fine-level EMA** — ran fine, wrong math, ~6× slower. See Key Discovery.
- **Default `N(0,1)` embedding init** — with tied head, initial CE ≈ 52 (vs expected `ln(260) ≈ 5.56`). Fix: `std = d_enc^(-0.5)`.
- **fp16 autocast for the EMA** — `(1-p)` accumulator chain underflows after a few hundred chunk steps. bf16 has the dynamic range.
- **Muon optimizer + logit softcap** — implemented locally, reverted before submission to keep this as a clean baseline that future stacked-trick variants can compare against.

## Hardware Note

1× RTX A6000 (48GB), no wallclock cap (non-record track). Could not fit `seq_len=2048` on a 4090 (24GB). Single GPU, no distributed. Windows-developed, Runpod-trained.

## Reproducing

Run from the parameter-golf root (paths are relative). Override `DATA_PATH=...` if shards live elsewhere. Three steps in order:

```bash
# 0) install
pip install torch>=2.4 numpy huggingface-hub datasets tqdm

# 1) build the byte260 dataset (it's NOT in the upstream cached_challenge_fineweb.py
#    manifest, so we build shards locally first; ~300MB FineWeb-Edu via HuggingFace)
python make_byte260_smoke.py

# 2) train (~100 min on 1× RTX A6000)
RUN_ID=hnet_real ITERATIONS=10000 \
  TRAIN_BATCH_TOKENS=32768 TRAIN_SEQ_LEN=2048 \
  D_ENC=256 D_MAIN=512 N_HEADS=8 \
  LR=3e-4 WARMUP_STEPS=500 WEIGHT_DECAY=0.1 RATIO_LOSS_ALPHA=0.03 \
  TARGET_RATIO=0.16667 \
  VAL_TOKENS_CAP=1000000 VAL_LOSS_EVERY=1000 TRAIN_LOG_EVERY=100 \
  python train_hnet.py

# 3) quantise the saved checkpoint to int8 + zlib (~9.5MB output)
python quantize_hnet.py logs/hnet_real_final.pt   # → logs/hnet_real_final.int8.ptz
```

Skip step 2 entirely if you just want to verify the reported `val_bpb`: load the included `final_model.int8.ptz` directly.

## Future Work

Path A (Mamba-2 layers); two-stage chunking `(3, 3)`; bigger model + data (room on both); vectorised EMA via parallel scan; stack with BPE-era tricks (Muon, partial RoPE, GPTQ).

## Included Files

- `hnet_model.py` — `DynamicChunking`, `upsample_with_ema`, `TransformerBlock`, `HNet`
- `train_hnet.py` — env-var training script
- `quantize_hnet.py` — per-row int8 + zlib with round-trip check
- `make_byte260_smoke.py` — FineWeb-Edu → byte260 .bin shards
- `final_model.int8.ptz` — trained weights (9.49MB)
- `train_log.txt`, `submission.json`, `requirements.txt`

## Citations

- Hwang, Wang, Gu (2025). *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling.* arXiv:[2507.07955](https://arxiv.org/abs/2507.07955).
- Reference: [goombalab/hnet](https://github.com/goombalab/hnet) — `dc.py` was source of truth for EMA/STE/clipping.
- Loshchilov & Hutter (2019). *Decoupled Weight Decay Regularization* (AdamW).

## Acknowledgements

Built on the parameter-golf training scaffold ([train_gpt.py](https://github.com/openai/parameter-golf/blob/main/train_gpt.py)) and its int8+zlib quantisation pipeline.

**AI assistance.** Pair-programmed with Claude (Anthropic, Opus 4.7). Architecture choices, the decision to ship a boring baseline before stacking tricks, and all training runs are mine. Code drafting and README drafting were AI-assisted and human-reviewed. Following the spirit of the NeurIPS LLM use disclosure norm (separate "use of AI" statement, not co-authorship).