# Notable Non-Record Submission: 1.0960 BPB — Muon + gated Krylov

Muon with a small gated Krylov correction on square, nonnormal slices. Standard SentencePiece GPT path, AR self-generated Full-Hessian GPTQ, selective `±1` pruning, and sliding-window evaluation.

**val_bpb: 1.09596320** (sliding, seed=`1337`) | **15,957,504 bytes** | **1xA100 80GB, 8h 52m**

> This is a non-record submission. It fits under the `16,000,000` byte artifact cap, but it does not satisfy the challenge's main leaderboard wallclock requirement of `10 minutes on 8xH100 SXM`.

## Results

| Metric | Value |
|--------|-------|
| Sliding BPB | `1.09596320` |
| Sliding val_loss | `1.85048306` |
| Step-20000 val_bpb | `1.1166` |
| Post-EMA val_bpb | `1.1156` |
| Int6 roundtrip exact BPB | `1.11953265` |
| Artifact bytes | `15,957,504` |
| Compressed model bytes | `15,817,800` |
| Code bytes | `139,704` |
| Parameters | `26,993,756` |
| Peak allocated VRAM | `29,336 MiB` |
| Training time | `31,932,706 ms` (`8h 52m 12.706s`) |
| Average step time | `1596.64 ms` |

The exact training log for this run is [train_seed1337.log]. The current `train_gpt.py` in this folder includes a small CPU-import compatibility guard so the record imports cleanly during Python 3.10 / CPU smoke tests; that changes the code-byte count but does not affect the SentencePiece execution path used for the logged run.

## Main Idea

The optimizer stays in the Muon family. The change is not a replacement of Newton-Schulz with a different optimizer; it is a narrow correction path:

1. Compute the standard Muon direction on the banked matrix weights.
2. For square slices only, estimate nonnormality from the commutator `W^T W - W W^T`.
3. Use a Hutchinson estimator to decide whether the slice is sufficiently nonnormal.
4. Choose a small adaptive Krylov rank.
5. Build a residual-direction correction and blend it back into the Muon direction with a small coefficient.

In practice this worked best as a conservative hybrid. Muon remained the base geometry, and the Krylov branch only fired on a subset of slices.

## Architecture

| Component | Setting | First introduced by |
|-----------|---------|---------------------|
| Layers | 11 (512d, 8 GQA heads, 4 KV heads) | Baseline |
| MLP | 3× (1536) with LeakyReLU(0.5)² | [#493](https://github.com/openai/parameter-golf/pull/493) @parinzee |
| Attention | XSA on all 11 layers | [#478](https://github.com/openai/parameter-golf/pull/478) @gowtham0992 |
| BigramHash | 3072 × dim=112 | [#1019](https://github.com/openai/parameter-golf/pull/1019) lineage (concept: [#162](https://github.com/openai/parameter-golf/pull/162) @raahilshah) |
| RoPE | Partial (16/64 dims) | [#315](https://github.com/openai/parameter-golf/pull/315) @jfprincz |
| LN Scale | 1/√(layer+1) | [#315](https://github.com/openai/parameter-golf/pull/315) @jfprincz |
| VE128 | Layers 9-10 | [#374](https://github.com/openai/parameter-golf/pull/374) @unnir |
| SmearGate | Position-mixing gate | [#65](https://github.com/openai/parameter-golf/pull/65) @aquariouseworkman |
| U-Net skips | Encoder-decoder connections | [#289](https://github.com/openai/parameter-golf/pull/289) |
| Weight avg | EMA(0.997) + Tight SWA(every 50) | [#401](https://github.com/openai/parameter-golf/pull/401) @newjordan |
| Quantization | Full Hessian GPTQ int6 (AR self-gen calibration) | [#1019](https://github.com/openai/parameter-golf/pull/1019) lineage (GPTQ: [#535](https://github.com/openai/parameter-golf/pull/535) @raahilshah) |
| Compression | LZMA preset=9 | [#160](https://github.com/openai/parameter-golf/pull/160) @ChaseWNorton |
| Warmdown | 4000 iterations | [#364](https://github.com/openai/parameter-golf/pull/364) @shikhar1729 |
| Optimizer | Parallel Muon + Parameter Banking + **gated Krylov residual correction** | **This work**, built on [#399](https://github.com/openai/parameter-golf/pull/399) @abaybektursun |
| Late QAT | STE at LR scale < 0.15 | [#286](https://github.com/openai/parameter-golf/pull/286) @chris-buckley |
| Selective pruning | ±1 values by reconstruction error | [#609](https://github.com/openai/parameter-golf/pull/609) @saml212 |
| Flash Attention 3 | Hopper warp-specialized kernels | [#122](https://github.com/openai/parameter-golf/pull/122) @mtybadger |


## What Actually Ran

This result used:

- the standard SentencePiece `sp1024` tokenizer path
- 11 layers, 512 dim, 8 attention heads, 4 KV heads
- 3x MLP with LeakyReLU(0.5)^2
- XSA across all 11 layers
- BigramHash, SmearGate, VE128, partial RoPE, U-Net skips
- EMA after training
- AR self-generated Full-Hessian GPTQ int6 export
- selective `±1` pruning to fit the official byte cap

The exact script snapshot used for the run is [train_gpt.py]. It is the historical single-file training script copied from the A100 box, not the current evolving repo root script.

### Brief HNet Result

The later learned-HNet branch was measured on the same A100 box and finished at:

| Variant | Sliding BPB | Int6 roundtrip exact BPB | Artifact |
|---------|-------------|--------------------------|----------|
| SentencePiece + Muon + gated Krylov | `1.09596320` | `1.11953265` | `15,957,504` |
| HNet + Muon + gated Krylov | `1.42700113` | `1.51636243` | `15,554,948` |

So HNet was comfortably under the byte cap, but much worse in quality. The main issue was not compression size; it was that the HNet path changed the representation and throughput in a way that hurt this setup.

## Run Command

```bash
TARGET_MB=15.2587890625 \
MUON_KRYLOV_ENABLED=1 \
MUON_KRYLOV_ALPHA=0.05 \
MUON_KRYLOV_ETA_THRESHOLD=0.03 \
MUON_KRYLOV_WARMUP_STEPS=1000 \
MUON_KRYLOV_DECISION_EVERY=100 \
MUON_KRYLOV_EVERY=2 \
MUON_KRYLOV_HUTCHINSON_SAMPLES=2 \
MUON_KRYLOV_RANK_MAX=4 \
MUON_KRYLOV_RANK_SCALE=1.0 \
VAL_LOSS_EVERY=2000 \
TRAIN_LOG_EVERY=200 \
WARMUP_STEPS=20 \
SEED=1337 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Bottom Line

The useful result here is simple: keep the strong 11L SP stack, keep Muon as the main optimizer, and add only a small gated Krylov residual correction on top. That combination produced a strong under-cap non-record score of **1.09596320 BPB** on a single A100 training for ~9 hours.
