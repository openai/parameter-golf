# SP8192 + Per-Layer GPTQ Clip + Unfrozen Score-First TTT

**val_bpb = 1.0849** (3-seed mean, std 0.00022) | **~15.96 MB** | 8xH100 SXM

## 3-Seed Results

| Seed | Sliding BPB | **TTT BPB** | Artifact bytes | Eval time |
|------|-------------|-------------|----------------|-----------|
| 4    | 1.08671     | **1.08463** | 15,961,223     | 553.2s    |
| 30   | 1.08721     | **1.08504** | 15,963,763     | 557.4s    |
| 2026 | 1.08699     | **1.08498** | 15,963,896     | 553.3s    |
| **Mean** | **1.08697** | **1.08488** | **15,962,961** | |
| **Std**  | **0.00025** | **0.00022** | | |

Merged SOTA (PR #1493 @bigbag): **1.0810 BPB**. This submission lands at
**+0.0039** vs merged SOTA — not a new record, contributed for the two
new primitives it introduces (per-layer clip sigmas, eval wallclock guard)
and for a recipe the community can sweep from.

## Key Techniques

1. **Base architecture** — inherits from PR #1735 (@Grad62304977): SP8192,
   11L × 512d, 8 heads / 4 KV, MLP 4x, partial RoPE (16/64), tied
   embeddings, logit softcap 30, depth recurrence layers 3-5, parallel
   residuals from layer 7, XSA last 11 layers, QK-Gain 5.25.
2. **SpinQuant** (PR #1695 @dexhunter) — per-matrix Hadamard rotation
   regenerated from `(seed, name, dim)` at dequant time; zero storage,
   one matmul per layer at load.
3. **GPTQ + SDClip** (@clarkkev PR #1394) — int6 matrices with
   **per-category clip sigmas** (this submission): MLP=12.0, attn=13.5,
   emb=15.0. Tuned for per-category outlier distributions.
4. **int7 token embedding** (`EMBED_BITS=7 EMBED_CLIP_SIGMAS=15`) — ~190 KB
   of artifact headroom at +0.002 BPB cost at the embedding level.
5. **Attention output gate** (PR #1667 @Grad62304977) — per-head, per-dim
   `2·σ(attn_gate)`, zero-init identity.
6. **Legal score-first TTT** (PR #549 @abaybektursun) — SGD, momentum 0.9,
   cosine LR, **TTT_FREEZE_BLOCKS=0 TTT_LR=0.010 TTT_EPOCHS=5** (this
   submission tunes these knobs). Score-before-update ordering: each
   val chunk is scored under `torch.no_grad()` BEFORE the SGD update
   touches weights; adaptation only shifts scoring of chunks c+1..N.
7. **Eval wall-clock budget guard** (this submission) — times score-only
   and adaptation costs separately (after a 5-chunk warmup). When
   elapsed + estimated-next-adapt + remaining-chunks × score-only-cost
   would exceed 600s − 20s safety, adaptation is truncated at the
   current chunk. Scoring continues for every remaining chunk (legality
   requires every val token contributes to BPB). Skipping happens at
   the cosine-LR tail where per-chunk LR is already near zero, so the
   BPB impact is negligible. Decision is rank-synced via
   `dist.all_reduce(MAX)` to keep NCCL in lockstep.

## Legality (Issue #1017 Conditions 1-4)

- **C1 single left-to-right pass** — chunks 0..N-1 iterated in order.
- **C3 score-before-update** — scoring under `torch.no_grad()` precedes
  the SGD step in `eval_val_ttt`.
- **C4 normalized distribution** — logit softcap + `F.cross_entropy`
  over the full vocab; every val token contributes to the reported BPB.
- **Training budget rule 7** — Hessian calibration and GPTQ run inside
  the 600s training budget (`gptq_reserve_seconds=60`), not eval.
- **Eval wall-clock ≤ 600s** — enforced via `MAX_EVAL_SECONDS=600` guard
  as described above.

## Architecture

11L × 512d × 8H / 4KV, MLP 4x, ReLU² MLP, partial RoPE (16/64 dims),
layerwise LN scale, tied embeddings, logit softcap 30.0, attention output
gate (per-head 2σ gate), SpinQuant rotation pre-GPTQ. Depth recurrence:
encoder `[0,1,2,3,4,5,3,4]` decoder `[5,3,4,5,6,7,8,9,10]` — layers 3-5
loop, activated around step 2016 (frac=0.35). Parallel residuals from
layer 7: attention and MLP operate on same pre-residual input. Skip
gates (sigmoid-gated U-Net connections).

## Training

Muon for matrices (momentum 0.99, warmup 1500 steps from 0.92), AdamW
for embeddings + scalars. Single training pass of ≤ 600s wall-clock.
Hessian calibration + GPTQ quantization performed inside the 600s
training budget (`gptq_reserve_seconds=60`).

## Reproduce

```bash
export SPINQUANT_ENABLED=1 \
       EMBED_BITS=7 EMBED_CLIP_SIGMAS=15 \
       MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.5 \
       TTT_ENABLED=1 TTT_EPOCHS=5 TTT_FREEZE_BLOCKS=0 TTT_LR=0.010 \
       MAX_WALLCLOCK_SECONDS=600 MAX_EVAL_SECONDS=600

for seed in 4 30 2026; do
  SEED=$seed torchrun --standalone --nproc_per_node=8 train_gpt.py
done
```

Per-seed logs: `train_seed{4,30,2026}.log`. The reported BPB is the
`quantized_ttt val_bpb:` line. `Total submission size quantized+brotli:`
confirms the artifact cap.
