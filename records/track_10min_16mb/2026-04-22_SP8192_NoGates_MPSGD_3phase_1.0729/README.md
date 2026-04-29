# SP8192 + No Gates + Multi-Phase Global SGD TTT

**val_bpb: 1.07285** (3-seed mean, std 0.00051) | **~15.94 MB** | 8xH100 SXM | Multi-Phase Global SGD TTT (Track B)

This record combines the base architecture from PR #1667 (MarioPaerle) with the Multi-Phase Global SGD TTT path from PR #1626 (dexhunter), with both SmearGate and AttnOutGate disabled. No tokenizer changes (vanilla SP8192). No Casefold or CaseOps. No SLOT.

## Results (8xH100 80GB SXM, Kansas City US, PyTorch 2.9.1+cu128, FA3)

| Seed | Steps | Train time | Post-TTT val_bpb | Post-TTT val_loss | Eval time | Artifact (bytes) |
|------|------:|-----------:|-----------------:|------------------:|----------:|-----------------:|
| 1337 | 4827  | 587.52s    | 1.07333739       | 2.77254196        | 429.1s    | 15,935,536       |
| 42   | 4839  | 587.16s    | 1.07287895       | 2.77135776        | 338.7s    | 15,935,501       |
| 0    | 4832  | 587.16s    | 1.07232205       | 2.76991921        | 385.1s    | 15,943,766       |
| **Mean** | **4833** | **587.28s** | **1.07285** | **2.77127** | **384.3s** | **15,938,268** |
| **Std**  |          |           | **0.00051**      | **0.00131**       |            | 4,805            |

All three seeds clear the 600s train budget, the 600s eval budget, and the 16,000,000-byte decimal artifact cap. The 3-seed std of 0.00051 BPB is well inside the 0.005-nat significance floor.

## What this submission is

This is a disciplined combinatorial submission that establishes two data points at full 8xH100 production scale:

1. **MP-SGD 3-phase TTT beats single-phase score-first TTT by 0.0028 BPB** on the same base architecture (single-phase run on the same pod produced 1.07612, this run produced 1.07334 for seed 1337).
2. **Disabling SmearGate and AttnOutGate from PR #1667's base does not hurt this configuration.** Reasoning for this came from community observations that PR #1736 and PR #1756 shipped with both gates plumbed but flagged off in their winning runs; I validated the direction on Spark ablations first, then reproduced at H100 production scale.

It does not attempt a novel architecture. It isolates a specific hypothesis (MP-SGD over single-phase TTT) and answers it at full scale.

## Lineage / attribution

- **PR #1667 @MarioPaerle** — SP8192 base architecture, 11L x 512d x 8H / 4KV, Partial RoPE 16/64, Loop L3-5, Parallel Residuals L7+, QK-Gain 5.25, MuonEq-R optimizer, Skip gates, SmearGate and AttnOutGate (both disabled in this submission), base score-first TTT scaffold, GPTQ int6 / int7 embeddings, Brotli-11 compression
- **PR #1626 @dexhunter** — Multi-Phase Global SGD TTT (`eval_val_ttt_phased`, `train_val_ttt_global_sgd_distributed`, the per-batch `BatchedTTTLoRA` with reset, phased boundaries, global SGD on scored documents only)
- **PR #1019 @abaybektursun** — the currently merged record-track rank 1

I ported the MP-SGD functions from PR #1626 verbatim into the PR #1667 base, preserved the per-chunk score-before-update ordering exactly, and added env-var gates so `PHASED_TTT_ENABLED=1` selects the phased path and the default (0) uses the existing single-phase path. Nothing was rewritten or simplified from PR #1626's TTT code.

## Issue #1017 compliance (Track B)

All four conditions addressed:

1. **Condition 1 (Strict causal dependence):** LoRA state at chunk `t` is constructed only from the prefix. Base model weight updates via `train_val_ttt_global_sgd_distributed` happen only at phase boundaries and operate on tokens from documents whose scoring already completed (`local_scored_docs` is populated after each batch's inner chunk loop completes). No future tokens influence any past score.
2. **Condition 2 (Full normalized distribution):** Standard softmax over the full sentencepiece vocabulary `Σ` of size 8192. No bucket normalization, no hash-bin redistribution, no `x_t`-contingent completion. The output distribution at position `t` is determined independently of the realized token.
3. **Condition 3 (Score-before-update):** Per-chunk: forward pass on the current chunk runs under `torch.no_grad()` path for accumulation into `loss_sum`, and the LoRA gradient step runs only after that accumulation is complete (`if needs_train:` guard, which is false on the last chunk of each document). Global level: `train_val_ttt_global_sgd_distributed` is invoked at phase boundaries on tokens from already-scored documents, not on live tokens. The last chunk of each training slice is explicitly skipped (`is_last_chunk: continue`) as a protective measure.
4. **Condition 4 (Single left-to-right pass):** Each batch is claimed exactly once via `_claim_next_batch` (atomic file-lock counter). No rescoring loop. `loss_sum` is append-only throughout evaluation.

The MP-SGD code paths in this submission are unchanged from PR #1626, which has already been accepted as Issue #1017 Track B compliant in the community.

## Hardware / reproducibility

- **Pod:** 8x NVIDIA H100 80GB HBM3 SXM in Kansas City, Missouri (US-MO-1 datacenter)
- **Per-GPU GEMM (pod-test.sh measurement):** 0.21 ms bf16 4096x4096 (about 657 TFLOPS per GPU)
- **NVLink:** 18 bonded NVLinks per GPU pair (NV18 all-pairs)
- **CPU:** Intel Xeon Platinum 8470, 208 threads
- **Torch:** 2.9.1+cu128, Triton 3.5.1, flash_attn_interface prebuilt wheel from https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
- **Image:** `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`

## Run command (per seed)

```bash
# Env defaults reproduce the submission exactly:
SEED=<seed> \
TTT_ENABLED=1 \
PHASED_TTT_ENABLED=1 \
PHASED_TTT_NUM_PHASES=3 \
PHASED_TTT_PREFIX_DOCS=2000 \
GLOBAL_TTT_LR=0.001 \
GLOBAL_TTT_EPOCHS=1 \
SMEAR_GATE=0 \
GATE_ATTN_OUT=0 \
DATA_DIR=/workspace/track-a/data/ \
ARTIFACT_DIR=<output dir> \
RUN_ID=<run id> \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Attribution notes

- The `train_gpt.py` in this folder contains two development-only shims that are inert on H100: (1) a Flash Attention backend auto-detect that falls through from FA3 to FA2 to SDPA based on `torch.cuda.get_device_capability` (activates only on `cc[0]==12`, Blackwell), and (2) a Triton block-size override for `linear_leaky_relu_square_kernel` (activates only on `cc[0]==12`). Both are no-ops on H100 Hopper and do not affect the submission path. They exist so the same file can be developed on a Blackwell dev box (where FA3 runtime kernels fail) without forking the code.
- No changes to the core model architecture, training loop, quantization pipeline, or evaluation code relative to PR #1667 and PR #1626.

## Delta vs the MP-SGD source (PR #1626)

- PR #1626 with vanilla SP8192 reports val_bpb 1.07193 (single seed in the PR log; I did not rerun it).
- This submission's 3-seed mean is 1.07285. The ~0.001 gap is within the 3-seed std (0.00051 here) plus what I'd expect from the seed mix we used (1337, 42, 0) vs PR #1626's seed choice.
- I did not introduce SmearGate or AttnOutGate (both disabled). I did not introduce CaseOps (vanilla SP8192). The only deliberate change to the MP-SGD recipe is inheriting PR #1667's base config defaults (for example, `MATRIX_LR=0.04`, `EMBED_LR=0.05`, `MUON_WD=0.095`, which differ slightly from PR #1626's defaults).
