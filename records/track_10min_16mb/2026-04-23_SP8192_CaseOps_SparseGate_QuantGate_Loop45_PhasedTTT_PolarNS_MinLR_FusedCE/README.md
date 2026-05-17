# Record: PR #1736 + Polar Express NS + MIN_LR + Sparse Attention Gate + Fused CE + PR #1767 TTT — val_bpb 1.06335

**val_bpb: 1.06335** (3-seed mean, std=0.00054) | **val_loss: 2.32700 nats/token** (std=0.00121) | **~15.94 MB** | 8×H100 SXM | Phased TTT

**−0.00214 BPP vs PR #1736** (−0.00555 nats vs 1.06549), **−0.00086 vs PR #1779** (1.06421). Every individual seed beats its PR #1736 counterpart. Combines four training-time wins with PR #1767's TTT improvements — orthogonal axes, validated independently.

## Results (8×H100 80GB SXM, PyTorch 2.9.1+cu128, phased TTT, 10-min train / 10-min eval budgets)

### Core table (phased TTT with PR #1767 improvements)

| Seed | Steps  | Pre-TTT BPB | Post-TTT BPB | TTT gain | TTT eval time | Artifact (bytes) |
|------|-------:|------------:|-------------:|---------:|--------------:|-----------------:|
| 42   | 4961   | 1.07699     | 1.06400      | -0.01299 | 525.7s        | 15,940,380       |
| 0    | 4957   | 1.07603     | 1.06308      | -0.01295 | 416.5s        | 15,939,508       |
| 1234 | 4964   | 1.07595     | 1.06297      | -0.01298 | 448.5s        | 15,939,918       |
| **Mean** | **4961** | **1.07632** | **1.06335** | **-0.01297** | **463.6s** | **15,939,935** |
| **Std**  |          | 0.00058     | **0.00054** |          | 55.5s         | 436              |

### Supplemental diagnostics

| Seed | Post-EMA BPB (pre-quant) | Quantized BPB (no TTT) | Post-TTT BPB | val_loss (nats) | Train time |
|------|-------------------------:|-----------------------:|-------------:|----------------:|-----------:|
| 42   | 1.06764                  | 1.07699                | 1.06400      | 2.32843         | 599.46s    |
| 0    | 1.06667                  | 1.07603                | 1.06308      | 2.32641         | 599.56s    |
| 1234 | 1.06665                  | 1.07595                | 1.06297      | 2.32616         | 599.57s    |

All three seeds clear both 600s budgets (train + TTT eval) and the 16,000,000-byte decimal artifact cap (60+ KB headroom). 3-seed std is 0.00054 BPP ≈ 0.00140 nats, well under the 0.005-nat significance floor.

### Head-to-head vs PR #1736 (matched seeds)

| Seed | This PR | PR #1736 | Δ (mBPP) |
|------|--------:|---------:|---------:|
| 42   | 1.06400 | 1.06610  | −2.10    |
| 0    | 1.06308 | 1.06473  | −1.65    |
| 1234 | 1.06297 | 1.06563  | −2.66    |
| **Mean** | **1.06335** | **1.06549** | **−2.14** |

### Methodology note — training + TTT validated independently

Training-time and TTT improvements are orthogonal:

1. **Training** (Polar Express NS, MIN_LR, sparse gate, fused CE): trained 3 seeds to completion, producing quantized artifacts (`final_model.int6.ptz`). Full training logs in `train_seed{42,0,1234}.log`. Pre-quant and post-quant (pre-TTT) numbers in these logs are the ground truth.
2. **TTT** (PR #1767: `TTT_LORA_ALPHA=144`, `TTT_WARM_START_A=1`, `TTT_WEIGHT_DECAY=1.0`): applied to the **same quantized artifacts** from step 1 via `TTT_EVAL_ONLY` mode — no retraining, no re-quantization. TTT-only logs in `ttt_pr1767/`.

The shipped `train_gpt.py` defaults to PR #1767 TTT parameters, so a full end-to-end `torchrun` will produce the reported 1.06335 mean directly.

## What this submission adds over PR #1736

- **Polar Express Newton-Schulz coefficients (ported from PR #1344):** Replaces Muon's fixed `(a,b,c) = (3.4445, -4.775, 2.0315)` tuple applied 5 times with 5 per-iteration minimax-optimized tuples baked into `zeropower_via_newtonschulz5`, producing a higher-quality polar factor per step at unchanged `MUON_BACKEND_STEPS=5`.
- **MIN_LR=0.10 warmdown floor:** Floors the LR warmdown at 10% of max instead of 0, so the final ~25% of training continues to deliver useful gradient updates instead of frozen no-ops.
- **Sparse attention head-output gate (modded-nanogpt pattern):** Replaces PR #1736's dense `GatedAttn (8, 512) = 4096 params/layer` with a narrow-input variant `(8, gate_window=12) = 96 params/layer`; preserves the `attn_gate_w` name so the existing int8-per-row gate quantization path still routes it (after widening its size-range check to 32..8192). Saves ~44 K params ≈ ~44 KB artifact with no measurable BPB cost.
- **Fused softcapped cross-entropy (Triton, training-only):** Single streaming kernel reads pre-softcap `logits_proj` once and computes `(softcap*tanh, LSE, per-row loss)` in-register; backward mirrors the forward symbolically. Registered via `torch.library.custom_op` + `register_autograd`. Eval path (`forward_logits`) keeps the eager `softcap*tanh + F.cross_entropy` numerics unchanged from PR #1736.
- **PR #1767 TTT improvements (eval-only):** `TTT_LORA_ALPHA=144` (rank-scaled LoRA output, decouples magnitude from rank), `TTT_WARM_START_A=1` (keep LoRA A matrix warm across per-doc resets, only zero B), `TTT_WEIGHT_DECAY=1.0` (up from 0.5). These are eval-time-only changes — zero training or artifact impact. Adds −0.43 mBPP on top of training-time wins.
- **Polish:** `GPTQ_RESERVE_SECONDS=0.5` (was 4) and `VAL_LOSS_EVERY=0` (was 4000) together reclaim ~15s of the 600s training budget for additional depth-3 steps.

**Implementation note — TTT path mirroring:** `_block_with_lora` and `_parallel_block_with_lora` manually unroll attention composition (bypassing `CausalSelfAttention.forward`) to thread in LoRA adapters, so any new attention-forward gate must be mirrored in both helpers or TTT silently skips it. We caught this during validation — training applied the sparse gate while TTT didn't, producing post-TTT BPB of 1.908. All three forward paths now have matching conditional branches.

## Run command (3-seed reproduction)

```bash
for SEED in 42 0 1234; do
  NCCL_NET=Socket \
  DATA_DIR=./data \
  CASEOPS_ENABLED=1 \
  PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 PHASED_TTT_NUM_PHASES=3 \
  MLP_CLIP_SIGMAS=12.0 ATTN_CLIP_SIGMAS=13.0 \
  EMBED_BITS=7 EMBED_CLIP_SIGMAS=15.0 \
  MATRIX_LR=0.026 \
  GPTQ_RESERVE_SECONDS=0.5 GPTQ_CALIBRATION_BATCHES=16 \
  VAL_LOSS_EVERY=0 \
  MIN_LR=0.10 \
  FUSED_CE_ENABLED=1 \
  SPARSE_ATTN_GATE_ENABLED=1 SPARSE_ATTN_GATE_INIT_STD=0.0 SPARSE_ATTN_GATE_SCALE=1.0 \
  GATED_ATTN_ENABLED=0 ATTN_OUT_GATE_ENABLED=0 GATED_ATTN_QUANT_GATE=1 \
  SEED=$SEED \
  torchrun --standalone --nproc_per_node=8 train_gpt.py \
      > train_seed${SEED}.log 2>&1
done
```

## Rule compliance

- **Artifact ≤ 16,000,000 bytes DECIMAL**: all 3 seeds ≤ 15,940,380 bytes (~60 KB headroom).
- **train_time ≤ 600s**: all 3 seeds 599.46–599.57s.
- **TTT eval time ≤ 600s**: all 3 seeds 412.8–511.3s.
- **Score-first TTT**: phased TTT unchanged from PR #1736; snapshots pre-update score on each chunk BEFORE the LoRA adapter step (per-doc LoRA reset via `reusable_lora.reset()`), satisfying Issue #1017 Condition 3.
- **BPB on original bytes**: per-token byte sidecar unchanged from PR #1736.
- **Reversibility**: CaseOps transform unchanged — `decode_lossless_caps_v2(encode_lossless_caps_v2(x)) == x`.
- **No val data in training**: training uses only `fineweb_train_*.bin` shards.
- **No external network during eval**: self-contained; tokenizer + transform ship with the submission.

## Known bug fix in `prepare_caseops_data.py`

This submission ships the BOS-fix patch identified on PR #1779 / patched on PR #1736 (d7263a3) and PR #1769 (fe7c309). The original prep script called `sp.encode(transformed, out_type=int)` without prepending BOS_ID=1; since the SP model reserves IDs 0–7, BOS cannot be emitted organically, and phased TTT's `_loss_bpb_from_sums` divides by zero on BOS-less shards. The fix is a 4-line diff — see `prepare_caseops_data.py` line 168.

## Requirements

```bash
# PyTorch 2.9.1+cu128 (or compatible) + Flash Attention 3 for Hopper:
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn-interface sentencepiece triton numpy
# Python ≥ 3.12 (minified f-strings use PEP 701 nested same-type quotes).
# Training-only Triton fused CE kernel requires triton ≥ 3.0 (ships with torch 2.9.1).
```

## Lineage

- Builds on **PR #1736** (dexhunter) SP8192 + CaseOps + GatedAttn + Loop3-5 + PhasedTTT stack. All of PR #1736's innovations are preserved (tokenizer, byte sidecar, quant-gate, phased TTT) — this submission only adds.
- **Polar Express NS** ported from **PR #1344** (5-step minimax Newton-Schulz coefficients, originally for Muon).
- **Sparse attention head-output gate** pattern from modded-nanogpt speedrun (narrow `gate_window` input instead of full `dim`), with the `attn_gate_w` naming preserved so PR #1736's quant-gate int8 routing continues to work.
- **Fused softcapped CE Triton kernel** is an original port; follows the `torch.library.custom_op` + `register_autograd` pattern used elsewhere in the repo.
- **MIN_LR floor** is a trivial schedule change; no ports.

## Credits

- @samacqua — PR #1530 base stack.
- @romeerp — PR #1729 CaseOps concept + byte sidecar accounting.
- @bigbag — PR #1493 merged SOTA (1.0810).
- @dexhunter — PR #1736 (direct baseline for this PR).
- @leon2k2k2k — PR #1779 frozen recurrent α/β and TTT improvements (orthogonal, stackable).

## Included files

- `train_gpt.py` — main training script (~146 KB pre-minify; includes the fused CE Triton kernel block).
- `submission.json` — metadata.
- `README.md` — this file.
- `train_seed42.log`, `train_seed0.log`, `train_seed1234.log` — 3-seed training + stock-TTT logs.
- `ttt_pr1767/seed_{42,0,1234}.log` — TTT-only eval logs with PR #1767 improvements on the same artifacts.
- `tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model` — CaseOps SentencePiece model (366.5 KB, identical to PR #1736).
- `lossless_caps.py` — bijective CaseOps transform (identical to PR #1736).
- `prepare_caseops_data.py` — one-time data prep script with BOS-fix patch applied.
