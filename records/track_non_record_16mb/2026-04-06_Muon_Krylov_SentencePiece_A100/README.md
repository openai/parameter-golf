# Notable Non-Record Submission: 1.0960 BPB — Muon + gated Krylov

Muon with a small gated Krylov correction on square, nonnormal slices. Standard SentencePiece GPT path, AR self-generated Full-Hessian GPTQ, selective `±1` pruning, and sliding-window evaluation.

**val_bpb: 1.09596320** (sliding, seed=`1337`) | **15,925,099 bytes** | **1xA100 80GB, 8h 52m**

> This is a non-record submission. It fits under the `16,000,000` byte artifact cap, but it does not satisfy the challenge's main leaderboard wallclock requirement of `10 minutes on 8xH100 SXM`.

## Results

| Metric | Value |
|--------|-------|
| Sliding BPB | `1.09596320` |
| Sliding val_loss | `1.85048306` |
| Step-20000 val_bpb | `1.1166` |
| Post-EMA val_bpb | `1.1156` |
| Int6 roundtrip exact BPB | `1.11953265` |
| Artifact bytes | `15,925,099` |
| Compressed model bytes | `15,817,800` |
| Code bytes | `107,299` |
| Parameters | `26,993,756` |
| Peak allocated VRAM | `29,336 MiB` |
| Training time | `31,932,706 ms` (`8h 52m 12.706s`) |
| Average step time | `1596.64 ms` |

The exact training log for this run is [train_seed1337.log].

## Main Idea

The optimizer stays in the Muon family. The change is not a replacement of Newton-Schulz with a different optimizer; it is a narrow correction path:

1. Compute the standard Muon direction on the banked matrix weights.
2. For square slices only, estimate nonnormality from the commutator `W^T W - W W^T`.
3. Use a Hutchinson estimator to decide whether the slice is sufficiently nonnormal.
4. Choose a small adaptive Krylov rank.
5. Build a residual-direction correction and blend it back into the Muon direction with a small coefficient.

In practice this worked best as a conservative hybrid. Muon remained the base geometry, and the Krylov branch only fired on a subset of slices.

## What Actually Ran

This result did **not** use HNet. It used:

- the standard SentencePiece `sp1024` tokenizer path
- 11 layers, 512 dim, 8 attention heads, 4 KV heads
- 3x MLP with LeakyReLU(0.5)^2
- XSA across all 11 layers
- BigramHash, SmearGate, VE128, partial RoPE, U-Net skips
- EMA after training
- AR self-generated Full-Hessian GPTQ int6 export
- selective `±1` pruning to fit the official byte cap

The exact script snapshot used for the run is [train_gpt.py]. It is the historical single-file training script copied from the A100 box, not the current evolving repo root script.

## Why SentencePiece Beat HNet Here

An HNet branch was tested later and performed much worse on this setup. The main issue was throughput and representation mismatch:

- the HNet path decoded SentencePiece shards back into bytes
- that changed the effective batch and context semantics
- HNet saw less text per optimizer step while paying for extra encoder/chunker/decoder machinery

For this English FineWeb regime and this budget, fixed SentencePiece was the better trade.

### Brief HNet Result

The later learned-HNet branch was measured on the same A100 box and finished at:

| Variant | Sliding BPB | Int6 roundtrip exact BPB | Artifact |
|---------|-------------|--------------------------|----------|
| SentencePiece + Muon + gated Krylov | `1.09596320` | `1.11953265` | `15,925,099` |
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

The useful result here is simple: keep the strong SentencePiece GPT stack, keep Muon as the main optimizer, and add only a small gated Krylov residual correction on top. That combination produced a strong under-cap non-record score of **1.09596320 BPB** on a single A100.
