# Mamba2 SSM + Attention Hybrid (SP8192) — Non-Record Submission

> A single-attention Mamba2/SSM hybrid trained at SP8192 in 600 seconds on 8×H100. This submission documents the architecture's strong pre-quant quality (sliding `val_bpb` ≈ 1.10) and the compression cliff that prevents it from fitting cleanly under the 16 MB cap.


This is a non-record attempt but could be useful. This is all I could muster up in 3 sundays and limited compute budget. My goal was to match or come as close to transfomers using a SSM focused attempt. 

The readme is AI generated so apologies for the slop. 




## Submitted-Run Numbers

| Metric | Value |
|---|---|
| **Pre-quant sliding `val_bpb`** | **1.1005** |
| **Pre-quant standard `val_bpb`** | **1.1053** |
| **Post-quant sliding `val_bpb`** | **1.2938** |
| **Post-quant standard `val_bpb`** | **1.2985** |
| **Compressed artifact size** | **16,094,692 bytes** (~95 KB over the 16,000,000-byte cap) |
| Training time | 611 s on 8×H100 (`MAX_WALLCLOCK_SECONDS=600`) |
| Steps | 4,300 |
| Total params | 45,651,000 |
| Sequence length (train + eval) | 8,192 |
| Vocab | SP8192 |
| Compression | role-bit packed (int4 FFN gate, int4 FFN down, int5 Mamba/attn, int6 embed) + LZMA preset 9 extreme |

The full training log is in `train.log`. The exact env override used for the submitted run is documented in the `Reproduction` section near the bottom of this README.

### Why this is a *non-record* submission

The submitted artifact is **just over the 16 MB cap** (16,094,692 bytes ≥ 16,000,000). A more aggressive quantization variant (`Run #1` in `train.log`, int3 FFN gate / auto-prune to 15.85 MB) does fit at 10,910,872 bytes (10.4 MB) but post-quant quality collapses to 2.12 sliding `val_bpb`. The architecture is competitive at full precision; the remaining work is purely on the compression side. We submit this as a research contribution under the 16 MB non-record track to document SSM-hybrid behavior in the constraint envelope.

---

## Background

This README was originally written as an experiment chronicle of all SSM/attention-hybrid runs we made during the challenge, not just the submitted one. The high-level conclusions and table of all single-run experiments below give the broader context that informed the submitted configuration.

The work focused on Mamba/SSM + attention hybrids, compression, and optimizer behavior. These were **exploratory single-run experiments**, not record-quality submissions. Unless otherwise stated, each result is from **one run only**, typically one seed, and should be treated as directional rather than statistically conclusive. We did **not** run multi-seed means, confidence intervals, or record-style validation.

## High-level conclusion

The strongest pre-quant result I found came from the old/full-capacity architecture:

```text
10 layers
9 Mamba2/SSM blocks + 1 attention block at layer 6
full independent SwiGLU FFN in every block
SP8192 tokenizer
D_MODEL=512
D_STATE=64
SEQ_LEN=8192
Muon on large 2D matrices
AdamW on scalar/control/embed groups
```

This reached roughly:

```text
prequant standard BPB: ~1.105
prequant sliding BPB:  ~1.100
```
There is a variant in ablations somewhere which got to 1.089 as well but honestly, I forgot which one. 


Compression is the main unresolved bottleneck. The model has about **45.65M parameters**, so even good byte-int6 + LZMA compression landed around **24.6 MB**, well above the 16 MB cap. More aggressive compression is needed, but the smaller/shared/narrow FFN architectures usually lost too much quality.

A second major conclusion is that **SSM optimizer behavior matters a lot**. Pure AdamW was faster but much worse in our short-run setting, while naive high-LR AdamW for Mamba projections catastrophically hurt performance. The best current optimizer path remains Muon-heavy, but we suspect SSMs need more careful Muon/AdamW hybridization or trust-clipped Muon updates.

## Experimental caveat

These experiments are not leaderboard-record attempts. They were meant to rapidly test architecture and optimizer hypotheses. Therefore:

- Most rows are **single-seed, single-run** measurements.
- Results are not averaged over seeds.
- Some scripts changed multiple factors at once, especially during early exploration.
- Several TTT, Mamba3, and compression runs were deliberately diagnostic rather than submission-ready.
- Reported BPB values should be read as rough ablation evidence, not final records.

## Result summary

| Experiment / variant | Params | Key setup | Steps | Standard BPB | Sliding BPB | Notes |
|---|---:|---|---:|---:|---:|---|
| Full FFN clean / full Muon baseline | 45.65M | 10L, FFN all layers, attn layer 6, Muon big matrices | 4200 | 1.1051 | **1.1003** | Best clean prequant architecture anchor |
| Full FFN clean, byte-int6/LZMA | 45.65M | Same as full baseline; byte INT6/INT8 + LZMA | 4300 | 1.1036 pre / 1.1148 post | **1.0987 pre / 1.1100 post** | Quality good, size 24.6 MB, still over cap |
| Full FFN clean, BF16 storage | 45.65M | BF16 storage for FFN/attn/embed/bigram, Mamba kept fp32 | 4300 | 1.1186 | 1.1138 | Slight speedup, large BPB loss; not worth keeping |
| Full FFN, Muon FFN+attn, AdamW Mamba high LR | 45.65M | Mamba moved to AdamW with too-high LR | 4500 | ~1.247 | ~1.245 | Catastrophic; AdamW LR was Muon-scale and too hot |
| Sparse FFN baseline | 26.77M | FFN only on layers 6 and 9 | 5400 | 1.1383 | 1.1334 | Much smaller/faster but loses substantial quality |
| Sparse FFN + carryover eval | 26.77M | Direct-kernel stateful-overlap carryover | 5400 | — | 1.1393 carryover | Carryover worse than sliding; SSM state not trained for this |
| J tied-dense | 30.19M | Shared SSM FFN on SSM layers + attention correction at 2,5 | 3900 | 1.1280 | 1.1226 | Best shared-FFN model, still far behind full FFN |
| J-384 | 29.92M | J with attention-correction dim 384 | 3800 | 1.1309 | 1.1256 | Shrinking attention correction hurt; little speed gain |
| Shared final FFN + small attention correction | 27.29M | Sparse FFN with shared final FFN reuse and 256-dim attn corr | 4600 | 1.1373 | 1.1319 | Small improvement over sparse, not enough |
| Fusion A | 30.30M | SSM + gated attention correction at layers 2,5 | 4670 | 1.1328 | 1.1272 | Attention correction helped; TTT was bad |
| Fusion B | 29.25M | Attention-before-Mamba injection | ~5090 | 1.1597 | 1.1563 | Dropped |
| Fusion C | 27.00M | Hard attention layers 2,5 + QK conditioning | ~5260 | 1.1394 | 1.1340 | Better size/quality tradeoff than B, worse than A/J/full |
| E1 carrier | 19.42M | Narrow SSM carrier, too few full sequence mixers | 3610 | 1.1974 | 1.1924 | Collapsed; not enough real SSM/FFN capacity |
| E2 carrier + memory tokens | 19.94M | E1 + SSM memory tokens | 3790 | 1.1998 | 1.1947 | Memory tokens did not help |
| E3 carrier + cache distill | 19.94M | E2 + train-only repetition/cache distillation | 3850 | 1.2064 | 1.2014 | Distill objective hurt in current form |
| K35 TaskMuon | 34.65M | Tied head, EMBED_DIM=320, no bigram, SSM FFN hidden 896 | ~4000+ | poor | poor | Under-learned; too aggressively shrunk |
| K35 wider TaskMuon | 38.19M | EMBED_DIM=384, bigram on, SSM FFN hidden 1024 | 4300 | 1.1597 | 1.1546 | Still bad; architecture/optimizer interaction weak |
| K35 pure AdamW, hot LR | 38.19M | AdamW-only, FFN/attn LR 0.0025 | 5000 | 1.1684 | — | Faster but poor generalization |
| K35 pure AdamW, cooler LR | 38.19M | AdamW-only, FFN/attn LR 0.001 | 4900 | 1.1896 | 1.1846 | Faster but much worse |
| Old 28.2M int6 packed/Brotli run | 28.21M | 2-attn, FFN mult 2, compressed int6 | 4300 | 1.1534 pre / 1.1831 post | 1.1480 pre / 1.1779 post | Nearly fit size, but quality too weak |

## Architecture learnings

### Full independent FFNs matter

The biggest architectural lesson is that full independent FFNs are doing a lot of work. The full-FFN baseline was both the best and surprisingly efficient:

```text
45.65M params, FFN everywhere → ~1.1003 sliding BPB
```

Attempts to share, sparsify, or aggressively compress FFNs structurally usually hurt:

- FFN only on layers 6 and 9: ~1.1334 sliding BPB.
- Shared SSM FFN / J: ~1.1226 sliding BPB.
- Post-hoc shared-base + rank-64 SVD residuals: catastrophic, post-SVD BPB above 2.0.
- Trainable compact FFN rank-96 was undertrained and weak, around ~1.1385 sliding BPB before optimizer fixes.

Conclusion: do not simply remove FFN capacity. If model size must drop, use compression or carefully trained structure, not naive sharing/post-hoc SVD.

### Fewer/full blocks may be better than same depth with weak FFNs

K35 tried to keep 10 layers while shrinking most SSM FFNs. That performed much worse than expected, even at 38M parameters. The lesson appears to be:

```text
Full FFN expressivity > preserving layer count with narrow FFNs
```

A future compressed architecture may be better as **fewer layers with full FFNs**, rather than 10 layers with many weakened FFNs.

### Attention correction helps, but not enough

Fusion A and J showed that extra attention correction paths can help. However, attention correction is not free, and replacing real SSM or FFN capacity with attention did not consistently improve results.

Useful observations:

- Fusion A was the best of the fusion sweep at ~1.1272 sliding BPB.
- Shrinking J attention correction from 512 to 384 hurt quality with little speed gain.
- Removing the early attention correction in J worsened performance.

Conclusion: attention correction is useful, but it does not replace full FFN capacity.

### Carrier-style SSM memory was too weak

The E1/E2/E3 carrier experiments collapsed around 1.19–1.20 BPB. They replaced too much real sequence/FFN capacity with narrow carriers and FFN-only blocks.

Conclusion: SSMs can be useful, but narrow carrier side channels are not enough unless the main sequence mixer remains strong.

## SSM-specific learnings

### SSM state carryover did not help yet

Stateful-overlap carryover was implemented using a direct Mamba chunk-scan path, but carryover eval was worse than sliding eval in the sparse baseline:

```text
sliding:   ~1.1334 BPB
carryover: ~1.1393 BPB
```

This suggests the model was not trained to use nonzero carried Mamba states. If carryover is revisited, it likely needs training-time state carry with detached recurrent states, not only eval-time carryover.

### seq_idx fixes were principled but not a big BPB lever

Adding doc-boundary `seq_idx` reset and explicit Mamba2 knobs was cleaner, but the BPB difference compared to no reset/default dt was small. The best practical setting remains:

```text
SEQ_IDX_ENABLED=1
DT_MIN=0.0005
DT_MAX=0.05
HEADDIM=64
D_STATE=64
```

### Mamba3 was not useful in this pass

A Mamba3 port was attempted because public Mamba3-style SSMs can be faster, but the local build/API path was not clean enough and the experiment did not produce a useful result. The conclusion is not that Mamba3 is bad; only that this attempt was not a productive path within this time budget.

## Optimizer learnings

### Muon is still the strongest known optimizer for this setup

The strongest model used Muon on large 2D matrices:

```text
matrix=45 Muon
scalar/control AdamW
embed AdamW
```

This consistently beat pure AdamW and AdamW-heavy hybrids in our short-run setup.

### Pure AdamW was faster but much worse

Pure AdamW K35 runs achieved much higher throughput and more steps, but validation was much worse:

```text
~5000 steps, ~4.3M tok/s, but ~1.18 sliding BPB
```

This shows AdamW is computationally attractive, but our AdamW configuration and/or architecture did not converge to a good basin in 600 seconds.

### High-LR AdamW for Mamba is catastrophic

Moving all Mamba matrices to AdamW with LR values like 0.006 for `mamba.in_proj` / `mamba.out_proj` destroyed performance. Those values are Muon-scale, not AdamW-scale.

If AdamW is used for SSM internals, it likely needs much lower rates:

```text
Mamba in/out projections: 3e-4 to 8e-4
Mamba dynamics:           5e-5 to 2e-4
```

### Full-Muon + trust clipping is the best current optimizer direction

The current best optimizer hypothesis is not “remove Muon.” It is:

```text
Keep full-Muon routing for large matrices.
Add per-role update trust clipping.
Use lower trust ratios for Mamba in/out projections.
```

This directly targets observed loss oscillations without throwing away the optimizer that produced the best BPB.

Suggested role trust ratios:

```text
FFN:        0.020
Attention:  0.015
Mamba in:   0.008
Mamba out:  0.008
Embed proj: 0.010
Bigram:     0.010
Generic:    0.010
```

### EMA failed in the current script

EMA with decay 0.9965 produced an unusable evaluation in one run:

```text
live BPB: ~1.1035
EMA BPB:  ~4.36
```

The likely issue is averaging from step 1 and/or averaging fragile recurrent/control parameters. If revisited, use late-start EMA, exclude Mamba dynamics, or checkpoint soup instead of naive full EMA.

## Compression learnings

### byte-int6/LZMA preserved quality but was too large

The best compression quality result was byte-stored INT6/INT8 with LZMA:

```text
prequant sliding:  ~1.0987 BPB
postquant sliding: ~1.1100 BPB
artifact:          24,604,112 bytes
```

This proved the quantization loss can be manageable. The size, not the BPB, is the remaining problem.

### zstd/int8 was too large

A full int8+zstd path compressed to about 37.3 MB. It was less destructive but nowhere near the 16 MB cap.

### BF16 storage during training hurt quality

Selective BF16 storage for non-Mamba projection weights cast about 26.5M parameters to BF16 storage, mostly FFNs. It gave only a small throughput gain but hurt sliding BPB by roughly 13 mBPB. Full FFN weights appear to need FP32 storage during training with the current Muon path.

### Aggressive hybrid low-bit compression is the next final-attempt path

The current final compression attempt targets the full-Muon/full-FFN model with:

```text
FFN gate_up: packed INT3
FFN down:    packed INT4
Mamba:       packed INT4 with protected dynamics rows
Attention:   packed INT5
Embeddings:  packed INT6 with opt-clip
Small params: fp16
Compressor:  LZMA-9 extreme
```

This is aggressive and may damage BPB, but it is the right scale of compression needed to push a 45M model toward 16 MB.

## TTT learnings

The LoRA score-first TTT path was harmful in the current implementation:

```text
Fusion A sliding: ~1.1272
Fusion A TTT:     ~1.1835
```

TTT also consumed a large amount of evaluation time. Until the TTT objective is fixed, it should remain disabled for architecture/compression sweeps.

Likely issues:

- TTT trained on too many overlapping context tokens.
- LoRA targets included unstable modules.
- Adaptation LR/WD was too high.
- Rank-local chunking was correct, but the update objective was not aligned enough with scoring.

A safer future TTT path would adapt only tiny calibration parameters such as norms, q_gain, residual scales, or final bias.

## Profiling learnings

Profiling showed the full model is not dominated by one operation. Costs were split across:

- Mamba backward / scan kernels.
- FFN and output/head dense matmuls.
- Muon optimizer step.
- Attention backward.

The full-FFN model was actually faster end-to-end than some smaller shared/adapter models because it uses large dense GEMMs that H100 handles well. Smaller low-rank/adapted models often introduced many small/skinny operations and optimizer overhead.

## Current best direction

The most promising path is:

```text
1. Keep full independent FFN architecture as the quality anchor.
2. Keep full-Muon optimizer, but add per-role trust clipping.
3. Use aggressive low-bit hybrid compression and dynamics protection.
4. Avoid pure AdamW, naive EMA, current LoRA TTT, and structural FFN sharing unless new evidence appears.
```

The leading open question is whether the 45.65M full-FFN model can be compressed under 16 MB while preserving enough of the ~1.10 prequant BPB. If the aggressive quantization path loses too much BPB, the next architecture direction should be fewer layers with full FFNs rather than the K35-style “same depth, narrower FFNs” approach.

## Reproducibility notes

All major runs used:

```text
8 x H100 80GB
SP8192 tokenizer
FineWeb challenge data
SEQ_LEN=8192 for the strongest runs
train_batch_tokens(global)=524288 unless noted
600 second training cap
sliding eval stride=64 for serious comparisons
```

Because these are exploratory, non-record runs, the reported table should be used to guide follow-up ablations only. A record attempt would need:

- Multiple seeds.
- Mean BPB reporting.
- Strict final artifact accounting.
- Official eval path consistency.
- No diagnostic shortcuts.
- Repeated compression/TTT validation after decompression.

---

## Reproduction

### Setup (fresh 8×H100 pod)

```bash
# 1. Clone the parameter-golf repo
cd /workspace
git clone https://github.com/openai/parameter-golf.git parameter-golf-repo

# 2. Install Python deps (H100 only, parallel build)
cd parameter-golf-repo
TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=16 \
  pip install -r requirements.txt --no-build-isolation
# This submission also requires:
pip install mamba-ssm causal-conv1d einops zstandard

# 3. Download SP8192 data + tokenizer (lives in a fork)
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 80
```

### Submitted-run command (Run #2 — moderate int4/5 quant, no auto-prune)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
SEQ_LEN=8192 \
N_UNIQUE_BLOCKS=10 \
ATTN_LAYER_IDXS=6 \
FFN_MULT=3 \
FFN_ACTIVE_POLICY=all \
EMBED_DIM=512 \
BIGRAM_ENABLED=1 \
HEADDIM=64 \
DT_MIN=0.0005 \
DT_MAX=0.05 \
SEQ_IDX_ENABLED=1 \
DOC_BOUNDARY_TOKEN=1 \
CARRYOVER_EVAL_ENABLED=0 \
SCORE_FIRST_TTT_ENABLED=0 \
TTT_ENABLED=0 \
RUN_POSTQUANT=1 \
QUANT_FORMAT=fullmuon_aggressive_lzma \
QUANT_BITS_FFN_GATE=4 \
QUANT_BITS_FFN_DOWN=4 \
QUANT_BITS_MAMBA=5 \
QUANT_BITS_ATTN=5 \
QUANT_BITS_EMBED=6 \
QUANT_BITS_MATRIX=5 \
QUANT_PACK_ALL=1 \
QUANT_PASSTHROUGH_NUMEL=65536 \
QUANT_K_FFN_GATE=12.85 \
QUANT_K_FFN_DOWN=12.85 \
QUANT_K_MAMBA=12.85 \
QUANT_K_ATTN=12.85 \
QUANT_K_EMBED=20.0 \
QUANT_SCALE_FLOOR_MULT=1.0 \
QUANT_PROTECT_DYNAMICS=1 \
QUANT_OPTCLIP_EMBED=1 \
QUANT_OPTCLIP_STEPS=8 \
QUANT_AUTO_PRUNE_TO_BYTES=0 \
QUANT_TARGET_BYTES=15850000 \
QUANT_LZMA_PRESET=9 \
QUANT_LZMA_EXTREME=1 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### "Fits the cap" variant (Run #1 — aggressive int3 + auto-prune; quality collapses)

Same as above but with:

```bash
QUANT_BITS_FFN_GATE=3 \
QUANT_BITS_MAMBA=4 \
QUANT_AUTO_PRUNE_TO_BYTES=1 \
QUANT_AUTO_PRUNE_FRACS=0,0.25,0.40,0.55,0.70,0.85
```

This produces a 10,910,872-byte artifact (under the cap) but post-quant sliding `val_bpb` collapses to 2.1248. Provided in `train.log` for completeness.
