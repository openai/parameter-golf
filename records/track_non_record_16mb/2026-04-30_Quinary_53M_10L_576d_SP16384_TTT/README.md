# Non-record: Quinary quantization + SP16384 + per-group lrzip + TTT - bpb 1.1384

**Quinary {-2,-1,0,+1,+2} weights (5-state, base-5 packed) + 10L (5 Encoder + 5 Decoder) 576d U-Net + Muon + 4× relu² MLP + Tied Embed (380→576) + Poly5 Softcap + YaRN 2048 + SP16384 BPE + FP8 QAT + 5-bit Scale Quant + Layout-Aware Per-Stream Archive + Score-First TTT (3 epochs, fp16-calibration-only)**

**bpb 1.1384 ± 0.0009 std** (3-seed TTT mean) | **15.72 MB** total artifact max (all 3 seeds FIT) | 8×H100 SXM, 7,800 steps in 599s + ~3.6 min TTT-eval

## Results (3 seeds, 8×H100 SXM)

| Seed | TTT BPB    | RT BPB | Total bytes |
|------|------------|--------|-------------|
| 42   | **1.1381** | 1.1626 | 15,714,938  |
| 1337 | **1.1394** | 1.1633 | 15,721,124  |
| 7    | **1.1378** | 1.1622 | 15,724,839  |
| **Mean ± std** | **1.1384 ± 0.0009** | 1.1627 ± 0.0006 |  |

## Motivation

Quinary {-2,-1,0,+1,+2} is one step above ternary {-1,0,+1}: 5 levels per parameter instead of 3. Per-symbol entropy floor: log₂5 ≈ 2.32 bits/param vs log₂3 ≈ 1.59. Fewer parameters fit in 16 MB, but each is finer-grained. We ran the comparison to see which side wins at this budget.

Beyond the formal comparison, there was the sport of it: how far can the extreme-quantization regime actually be pushed under the restrictions of the competition (16 MB total artifact, 10-min training, 10-min eval on 8×H100 SXM)? The current leaderboard frontier has drifted far away from it (PR #1855, PR #1851, et al. use int6 + LQER rank-4 corrections rather than ternary/binary), and quinary is the smallest step away from ternary that still stays squarely inside the discrete regime. We wanted to test whether that smallest step buys enough to be worth taking.

Empirically: chunked roundtrip BPB drops from **1.1842** (ternary record, PR #640) to **1.1627** (this submission, 3-seed mean) — a **−0.022 bpb** architectural win — and score-first TTT on the calibration parameters adds another **−0.024 bpb** on top, landing at **1.1384**.

## Diff from the ternary record (PR #640)

This submission is a direct quinary fork of [@CiprianFlorin-Ifrim's ternary record](https://github.com/openai/parameter-golf/pull/640). Everything not listed in the table below was inherited unchanged: U-Net topology with per-block residual mix and learned skip weights, 4× relu² MLP, Muon (3 NS steps), factored tied embedding, polynomial-5 softcap with z-loss, YaRN (max_len=2048, base=5000), fused QKV, FP8 QAT for non-quantized linears, FlashAttention-3, 599s wallclock cap.

| | Ternary record | This submission |
|---|---|---|
| Quantization | ternary {-1,0,+1}, log₂3 ≈ 1.585 bpw entropy floor | quinary {-2,-1,0,+1,+2}, log₂5 ≈ 2.322 bpw entropy floor; raw storage ≈ 2.667 bits/param via 3 quins per byte (post-archive cost depends on the entropy coder) |
| Group size | 128 | 192 (3 groups per c_qkv row) |
| Scale storage | fp16 | 5-bit log-delta scale quant (−141 KB, +2.1 mBPB TTT cost) |
| Tokenizer | SP8192 | SP16384 |
| Model dim | 768 | 576 |
| GQA (Q:KV) | 8:4 | 6:3 |
| Embed bottleneck | 254 | 380 |
| Compression | single-blob LZMA | layout-aware per-stream v2 archive (header `0x03`) |
| Eval | stride-16 sliding + temperature scaling | score-first TTT (3 epochs, fp16 calibration params, 42,364 ≈ 0.08% of model) |
| Param count | 73.7M | 52.8M |

## Architecture (config)

| | |
|---|---|
| Layers | 10 (5 encoder + 5 decoder, symmetric U-Net) |
| Model dim | 576 |
| Heads | 6 query / 3 KV (GQA), head_dim=96 |
| MLP | 4× expansion, hidden=2304, relu² activation |
| Embed | tied, 16384 vocab, 380→576 bottleneck |
| RoPE | YaRN, base=5000, max_len=2048 |
| Softcap | poly5, cap=10 |
| Quinary group size | 192, per-group absmean (`scale_correction` exists per group but is inert by design — see Score-first TTT) |
| Optimizer | Muon (matrix params), Adam (scalars + tied embed) |
| Batch / seq | 524 288 tok / 1024 |
| Wallclock cap | 599 s |

## Tokenizer (SP16384)

Custom-trained SentencePiece BPE with `vocab_size=16384`, trained from scratch on FineWeb-10B using the upstream [`data/download_hf_docs_and_tokenize.py`](https://github.com/openai/parameter-golf/blob/main/data/download_hf_docs_and_tokenize.py) pipeline (a thin wrapper around `sentencepiece` BPE training). The tokenizer model + pre-tokenized shards are published on the HF dataset repo [`deniskurlov/parameter-golf-fineweb-sp16384`](https://huggingface.co/datasets/deniskurlov/parameter-golf-fineweb-sp16384); `setup.sh` pulls the `canonical/` subset (~23 GB total — 117 train shards + 1 val shard + the tokenizer `.model`/`.vocab`) and that is sufficient to reproduce the run. The competition's 16 MB artifact cap applies only to the model bundle; the tokenizer + tokenized shards are pre-published infrastructure that any reproducer downloads once.

Doubling the vocab vs the ternary record's SP8192 reduces tokens-per-byte from ~0.30 to ~0.246 (fewer cross-entropy terms in the BPB sum) at the cost of a 2× larger embedding matrix — bounded here by the 380→576 factored-tied-embedding bottleneck.

## Per-stream compression (header byte `0x03`)

For each quinary tensor, two-stage choose-min:

1. **Layout selection** (LZMA-screened): generate 4 candidate byte-layouts and pick the layout with the smallest LZMA9-compressed size.
   - **base5** (canonical, 3 symbols/byte) and **base5_T** (transpose, then base-5 pack — wins when columns are more locally similar than rows, common on MLP projections).
   - **bitmask** = three bit-planes: `zero_mask | sign_bits[over nonzeros] | mag2_bits[over nonzeros]`, giving the entropy coder homogeneous planes to model independently. Plus **bitmask_T**.
2. **Compressor selection**: compress the winning layout with both `lzma9` and `lrzip-zpaq -L9`; keep whichever is smaller.

(LZMA-screen rather than full 4×2 keeps serialize wallclock bounded — `lrzip` can be slow on bad streams; we only invoke it on the chosen layout.)

For `c_qkv.weight`, the row block is split into independent Q / K / V sub-payloads first (Q, K, V have different trained weight distributions, so optimal layouts differ per part). The legacy single-blob lrzip path was the source of the seed-7 OVER cliff (~33% of seeds went OVER 16 MB); per-stream v2 FITS at ~15.65 MB (model only) / ~15.72 MB (model+code) across all 3 seeds we tried.

## Score-first TTT

After loading the artifact, freeze all quantized and FP8 weights. Adapt only the **fp16 calibration parameters** — `attn_scale`, `mlp_scale`, `resid_mix`, `q_gain`, `skip_weights`, `vocab_bias` — **42,364 values, ≈0.08% of the model**. Process the val stream in 1,134 chunks of 32k tokens with sliding-window stride=16; for each chunk, **grade tokens first, then train** (legal under the rules). 3 SGD epochs, lr=0.005, momentum 0.9. Eval time ~215 s of the 600 s eval budget. (These params are tagged `CTP` in the code — a holdover constant name from the ternary base.)

> **Note on `scale_correction`:** the inherited `QuinaryLinear` carries a per-group fp32 `scale_correction` parameter (~190k values across all quinary layers, init=1.0). Its gradient is blocked by the STE detach in the forward pass, so it never updates from 1.0 in either training or TTT. We tested fixing the STE on 2026-05-01 (one seed): no TTT benefit, ~2 mBPB training-side regression. Reverted. The parameter is therefore inert by design; it remains in the state-dict at value 1.0 (compresses to a few KB) and is excluded from the TTT optimizer's adapted-parameter set.

### Validation accounting

The validation manifest has **37,147,047** token IDs, hence **37,147,046** possible next-token target positions. The eval loop truncates the stream to a multiple of `train_seq_len=1024`, scoring **37,146,624** target tokens — the final partial 1024-window (the trailing **422** target tokens) is omitted. From the `verify_bpb.py` output:

| | targets | bytes |
|---|---:|---:|
| Full untruncated stream | 37,147,046 | 151,080,891 |
| Eval-scored slice (`(N-1)//1024 × 1024`) | 37,146,624 | 151,078,879 |
| Dropped tail (last partial window) | 422 | 2,012 |

The dropped fraction of the byte denominator is **2,012 / 151,080,891 ≈ 1.33 × 10^{-5}**.

Bound on the resulting BPB bias: BPB = NLL / byte_count, with both numerator and denominator scaling proportionally with token count, so the bias only comes from any *deviation* of the dropped-tail's average loss from the global average. For a 422-token sample of 37M, that worst-plausible deviation gives **|ΔBPB| ≲ 2 × 10^{-5} BPB** — well below the current 3-seed std of **0.0009 BPB** and far below the 0.005 BPB record-submission improvement threshold. The bias is also identical across all 3 seeds (same val stream, same truncation), so seed-to-seed variance is unaffected; only absolute BPB carries the constant offset.

Fresh logs report `eval_tokens` and `eval_bytes` after both the roundtrip eval and TTT eval so the actual scored slice is auditable line-by-line, and `verify_bpb.py`'s `exact eval slice` `lut_bytes` cross-checks the `eval_bytes:N` value `train_gpt.py` writes at runtime.

### Verifying the BPB denominator (custom tokenizer)

Because we use a custom-trained SentencePiece BPE (SP16384), the per-token byte LUTs in `build_luts` are reviewer-auditable. Run:

```bash
python3 verify_bpb.py
```

The script independently rebuilds the `(base_bytes, has_leading_space, is_boundary_token)` LUTs from the `.model` file using the **same shard loader as `train_gpt.py`** (256-int32 / 1024-byte header, magic `20240520`, version `1`), then for each of several stream slices it compares **two byte counts**:

- the LUT-based sum used by `eval_val` and TTT eval (`base_bytes_lut[tgt] + has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]`);
- the SentencePiece decoder's UTF-8 byte count for the same tokens, with documents decoded BOS-by-BOS so the LUT's "no leading-space charge after a boundary" rule and the decoder's "no synthetic leading space at sequence start" behavior are aligned.

The slices checked are: **the exact slice scored by `eval_val`** (truncated to a `train_seq_len=1024` target multiple — this is the slice whose byte count drives the headline BPB), the full untruncated stream, and BOS-delimited document prefixes (single doc, first 10/100/1000 docs). The script also reports the tokenizer's SHA-256 for provenance and counts UNK tokens (must be 0 — the LUT assigns 0 bytes to UNK, which would silently inflate BPB).

If every slice reports `delta = +0`, the LUT denominator matches SentencePiece decoding on the scored tokenized validation slice — which means BPB reduces to `cross_entropy_in_nats / (decoded_bytes × ln 2)`. **Note**: this proves *internal consistency* between the LUT and the SP decoder for the supplied tokenizer + supplied tokenized shards. It does not by itself prove that the shards correspond to the original canonical FineWeb validation bytes under the intended tokenizer — that is a separate provenance question, addressed by the tokenizer SHA-256 line below and by hosting the tokenized shards at the public HF repo `deniskurlov/parameter-golf-fineweb-sp16384` so reviewers can compare against an independent retokenization.

**Verified output, on the canonical sp16384 stack (2026-05-01)**:

```
tokenizer    : data/canonical/tokenizers/fineweb_16384_bpe.model (vocab=16384)
tokenizer sha: abaec140336563026d65c1b7192d47a2b8c81a3bbad0f4d1cd1d852364ac432a
BOS id=1 ('<s>')  EOS id=2 ('</s>')  UNK id=3
val shards   : 1 (data/canonical/datasets/fineweb10B_sp16384/fineweb_val_000000.bin ... )
LUT stats    : byte-fallback=256  control/unknown/unused=4  with-leading-space=11664  boundary=4
shard tokens : 37,147,047
BOS positions: 50,000; first=0
UNK count    : 0

eval slice   : train_seq_len=1024 target_count=37,146,624 omitted_tail_targets=422
  PASS  exact eval slice                     targets=37,146,624 lut_bytes= 151,078,879 decoded_bytes= 151,078,879 delta=+0 start_boundary=True
  PASS  full untruncated stream              targets=37,147,046 lut_bytes= 151,080,891 decoded_bytes= 151,080,891 delta=+0 start_boundary=True
  PASS  doc at first BOS                     targets=       295 lut_bytes=       1,339 decoded_bytes=       1,339 delta=+0 start_boundary=True
  PASS  first 10 BOS docs                    targets=     5,592 lut_bytes=      22,919 decoded_bytes=      22,919 delta=+0 start_boundary=True
  PASS  first 100 BOS docs                   targets=    72,484 lut_bytes=     291,960 decoded_bytes=     291,960 delta=+0 start_boundary=True
  PASS  first 1000 BOS docs                  targets=   706,682 lut_bytes=   2,851,493 decoded_bytes=   2,851,493 delta=+0 start_boundary=True

ALL CHECKS PASS — LUT bytes match SentencePiece decoder bytes on the eval slice.
```

End-to-end audit cross-check: the verifier's `exact eval slice lut_bytes = 151,078,879` is **bit-identical** to the runtime `eval_bytes:151,078,879` printed by `train_gpt.py:eval_val` and TTT eval for every seed. Cross-tokenizer sanity: rerunning the same verifier against the upstream openai/parameter-golf SP1024 stack on the same FineWeb val source gives identical byte counts of `1,339 / 22,919 / 291,960 / 2,851,493` for the first 1/10/100/1000 documents (different tokenizer, same source bytes — the decoder agrees).

## Setup and Run

```bash
# Environment setup (lrzip + Python deps + FlashAttention-3 + dataset)
bash setup.sh

# Single seed
SEED=42 bash run.sh

# 3-seed sweep
for SEED in 42 1337 7; do
  RUN_ID=quinary_seed${SEED} SEED=$SEED bash run.sh
done
```

<details>
<summary>Representative run command (subset of <code>run.sh</code> env vars — see <code>run.sh</code> for the authoritative full set)</summary>

The block below mirrors the env vars `run.sh` actually passes (model shape, optimizer, TTT, etc.). A few minor knobs that `run.sh` also passes through (`ADAM_LR`, `ADAM_WD`, `BATCH_TOKENS_START`, `BATCH_SCHEDULE_FRACTION`, `SEQ_LEN_START`, `SEQ_SCHEDULE_FRACTION`, `VAL_LOSS_EVERY`, `TRAIN_LOG_EVERY`, `CHURN_LOG_EVERY`, `VAL_MAX_TOKENS`) are not duplicated here — `run.sh` is the authoritative source. As of Tier-4, all defaults in `train_gpt.py:Hyperparameters` also match the canonical SP16384 config, so a bare `torchrun --standalone --nproc_per_node=8 train_gpt.py` (no env vars) reproduces the submission.

```bash
RUN_ID=quinary_seed42 \
DATA_PATH=./data/canonical/datasets/fineweb10B_sp16384 \
TOKENIZER_PATH=./data/canonical/tokenizers/fineweb_16384_bpe.model \
VOCAB_SIZE=16384 \
BITNET_GROUP_SIZE=192 \
EMBED_DIM=380 \
NUM_LAYERS=10 \
MODEL_DIM=576 \
NUM_KV_HEADS=3 \
NUM_HEADS=6 \
MLP_MULT=4 \
MATRIX_OPTIMIZER=muon \
MUON_BACKEND_STEPS=3 \
MUON_MOMENTUM=0.95 \
MUON_MOMENTUM_WARMUP_START=0.85 \
MUON_MOMENTUM_WARMUP_STEPS=500 \
MUON_WD=0.0 \
MATRIX_LR=0.035 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.02 \
WARMDOWN_FRACTION=0.2 \
LOGIT_SOFTCAP=10 \
QK_GAIN_INIT=5.0 \
ROPE_TYPE=yarn \
YARN_MAX_LEN=2048 \
ROPE_BASE=5000 \
TRAIN_BATCH_TOKENS=524288 \
TRAIN_SEQ_LEN=1024 \
ITERATIONS=10000 \
WARMUP_STEPS=5 \
MAX_WALLCLOCK_SECONDS=599 \
TIE_EMBEDDINGS=1 \
HEAD_LR=0.02 \
ACTIVATION=relu2 \
SOFTCAP_TYPE=poly \
TTT_STEPS=3 \
TTT_LR=0.005 \
TTT_TOKENS=32768 \
SCALE_QUANT_BITS=5 \
SEED=42 \
COMPILE_MODE=default \
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

</details>

## File Layout

```
.
├── README.md                          # this file
├── submission.json                    # OpenAI-format submission metadata
├── setup.sh                           # apt + pip + FlashAttention-3 + HF dataset download
├── run.sh                             # canonical training+eval entry point
├── train_gpt.py                       # complete training, compression, and TTT pipeline
├── verify_bpb.py                      # standalone reviewer-runnable BPB-LUT check
├── requirements.txt                   # Python dependency pin
├── fineweb_16384_bpe.model            # bundled tokenizer (sha256 abaec140…), provenance copy
├── fineweb_16384_bpe.vocab            # bundled vocab (provenance copy)
├── quinary_seed42.txt                 # 3-seed training/TTT logs (one per seed)
├── quinary_seed1337.txt
└── quinary_seed7.txt
```

The bundled `fineweb_16384_bpe.{model,vocab}` files are inspection / provenance copies — they should be byte-identical to what `setup.sh` downloads from `deniskurlov/parameter-golf-fineweb-sp16384` into `./data/canonical/tokenizers/`. Tokenizer files are *data*, not code, so they don't count toward the 16 MB cap (`bytes_code + compressed_model_bytes`); their SHA-256 (`abaec140336563026d65c1b7192d47a2b8c81a3bbad0f4d1cd1d852364ac432a`) is also reported by `verify_bpb.py` for cross-checking.

## Compliance

- [x] Artifact ≤ 16,000,000 bytes (15,724,839 — max across the 3 verified seeds, seed=7; per-seed values in the Results table; margin = 275,161 bytes)
- [x] Training ≤ 10 minutes (599,436–599,772 ms wallclock across 3 seeds)
- [x] Evaluation ≤ 10 minutes (TTT eval 212,250–215,427 ms; non-TTT roundtrip ~80 s)
- [x] Score-first TTT (CTP params only adapt on tokens *already* graded)
- [x] No network calls during evaluation
- [x] No external compute
- [x] No access to validation data during training. Validation shards are loaded into memory at startup but are consumed only by the post-training eval / TTT functions; under the canonical run `VAL_LOSS_EVERY=0`, so no validation tokens enter gradient updates before they have been scored under the score-first TTT pattern.
- [x] Reproducibly runs end-to-end from `bash setup.sh && bash run.sh` on a fresh 8×H100 SXM pod
- [x] BPB byte-count LUTs match the SentencePiece decoder's UTF-8 output exactly on the slice scored by `eval_val` (`exact eval slice` check) and on BOS-aligned document slices (run `python3 verify_bpb.py` to reproduce; see "Verifying the BPB denominator" above)

## Our contribution

All adaptations to the quinary case are ours. Concretely:

- **Quinary `QuinaryLinear`** — 5-level absmean-scaled STE quantization (`clamp(-2, 2)`) replacing the ternary `clamp(-1, 1)`. Per-group (192) absmean scaling. The inherited `scale_correction` per-group multiplier is kept in the state-dict for backwards-compatibility but is inert by design (see Score-first TTT note).
- **Base-5 packing** — three quins per byte (max symbol value 4·1 + 4·5 + 4·25 = 124, fits in `uint8`), with a paired unpacker. **Raw storage 8/3 ≈ 2.667 bits/param**; the entropy floor for 5 equiprobable symbols is log₂5 ≈ 2.322 bits/param, so the raw packing leaves ~0.34 bits/param of headroom for the downstream entropy coder. Ternary base-3 packing is the analogous 5/3 ≈ 1.667 raw bits / log₂3 ≈ 1.585 floor.
- **{-2,-1,0,+1,+2} bitmask plane decomposition** — alternative encoding as three concatenated bit-planes: `zero_mask | sign_bits[over nonzeros] | mag2_bits[over nonzeros]`. Each plane has homogeneous bit statistics so the entropy coder models them independently rather than fighting a multimodal mixture.
- **LZMA-screened layout selection** — for every quinary tensor, materialize all 4 layouts (`base5`, `base5_T`, `bitmask`, `bitmask_T`), screen them by LZMA9-compressed size, then run LZMA9 vs lrzip-zpaq only on the selected layout. Bounded-cost heuristic with an LZMA floor — *not* an exhaustive 4×2 search; in principle could miss a (layout, lrzip) pair that beats (best-LZMA-layout, lrzip), but in practice the LZMA floor caps the worst case at the canonical base5+LZMA encoding.
- **5-bit log-delta scale quantization** — per-group fp16 scales replaced with anchor + 5-bit log-delta; saves ~141 KB at +2.1 mBPB TTT cost. Net Pareto-positive.
- **Quinary fork of the architecture itself** — the rebalancing of `model_dim` (768→576), GQA ratio (8:4 → 6:3), and embedding bottleneck (254 → 380) for the higher per-param bit cost of quinary, plus the SP8192→SP16384 tokenizer choice and `group_size` 128→192 to fit the new model dim cleanly.

Empirical effect of the bundle is in the Diff and Results tables above. The seed-7 OVER cliff (single-blob lrzip OVERed on ~33% of seeds, including seed=7 at 17.23 MB) goes away under the per-stream v2 archive — all 3 seeds we tried FIT at 15.71–15.73 MB total.

Note: with the v2 archive, seed=7 is now actually the **best**-fitting and best-scoring seed (TTT BPB = 1.1378), not the worst — the prior cliff was an artifact of single-blob compression interacting with seed=7's specific weight distribution, fully absorbed by the per-stream layout choose-min.

## Acknowledgements

This submission stands on others' work. The architectural foundation, the compression pipeline's core ideas, and the score-first TTT pattern are all upstream of us; see "Our contribution" above for the quinary-specific extensions.

- **@CiprianFlorin-Ifrim** — [PR #640: Ternary U-Net record (1.1570 BPB)](https://github.com/openai/parameter-golf/pull/640) — architectural base. 
Inherited: the U-Net topology (5+5 enc/dec with learned skip weights and per-block residual mix), Muon optimizer settings, FP8 QAT for non-quantized linears, factored tied embedding (with a narrower bottleneck here), polynomial-5 softcap with z-loss, YaRN positional encoding, fused QKV projection, and FlashAttention-3 wiring. 
- **@codemath3000** — [PR #1855: per-group lrzip+brotli compression pipeline (1.06108 BPB)](https://github.com/openai/parameter-golf/pull/1855) — compression pipeline core ideas. Inherited: splitting the artifact into multiple compressed payloads instead of one monolithic blob, treating per-tensor byte layout as an optimization axis, and using lrzip's ZPAQ back-end for long-range cross-payload deduplication.
- **Score-first TTT lineage** — [@abaybektursun's PR #549](https://github.com/openai/parameter-golf/pull/549) and [@clarkkev's PR #1394 SP8192 stack](https://github.com/openai/parameter-golf/pull/1394) — origin of the legal "train only on tokens already graded" pattern.