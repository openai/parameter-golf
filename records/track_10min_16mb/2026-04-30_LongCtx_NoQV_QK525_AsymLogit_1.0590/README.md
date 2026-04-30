# LongCtx No-QV QK5.25 + AsymLogit

This record folder contains a three-seed rerun of the LongCtx No-QV
QK5.25 + AsymLogit configuration. The submitted script is self-contained in
this folder and was rerun from this directory on 8xH100.

## Result

| Seed | Stop step | Train time | Pre-quant BPB | Quant BPB | Final TTT BPB | Eval time | Artifact bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 4879 | 596.034s | 1.06203086 | 1.07040043 | 1.05857451 | 397.4s | 15,988,724 |
| 0 | 4868 | 596.139s | 1.06262576 | 1.07106641 | 1.05915199 | 396.3s | 15,988,166 |
| 1234 | 4880 | 596.069s | 1.06240646 | 1.07107313 | 1.05924929 | 417.7s | 15,992,777 |

Mean final validation BPB: **1.05899193**.

Population std over the three seeds: **0.00029782**. The largest observed
submission artifact is **15,992,777 bytes**, below the 16,000,000 byte limit.

## Method

The final configuration combines a compact CaseOps/SP8192 language model with
legal score-first TTT and a size-aware quantization path. The main frozen choices
are:

- CaseOps/SP8192 tokenization with byte-sidecar BPB accounting.
- Sparse attention gating, BOS-fixed SmearGate, skip gates, LQER correction,
  int7 embeddings, and mixed-precision GPTQ.
- 2560-token eval and TTT windows.
- No-QV TTT masking, keeping K/O/MLP adaptation active.
- `TTT_LOCAL_LR_MULT=0.75`, `TTT_LORA_RANK=80`, and
  `PHASED_TTT_PREFIX_DOCS=3000`.
- `QK_GAIN_INIT=5.25`, `WARMDOWN_FRAC=0.85`, and `MIN_LR=0.1`.
- Eval-only asymmetric logit rescale and AWQ-lite protected quantization.
- Per-group lrzip compression, with artifact size checked on every clean seed.

The constants were selected by local HPO and ablation runs, then frozen before
the clean three-seed rerun above. Public Parameter Golf submissions and records
were used as references while building this configuration.

## Reproduce

Prepare the CaseOps dataset once:

```bash
python prepare_caseops_data.py --local-dir /workspace/caseops_data
```

Run a seed from this folder:

```bash
SEED=42 \
CASEOPS_ROOT=/workspace/caseops_data \
RUN_ID=longctx_noqv_qk525_asym_seed42 \
./run_current_candidate.sh
```

`run_current_candidate.sh` expands the exact environment variables and launches:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Logs

- `train_seed42_clean_rerun.log`: clean seed-42 rerun, final BPB `1.05857451`.
- `train_seed0_clean_rerun.log`: clean seed-0 rerun, final BPB `1.05915199`.
- `train_seed1234_clean_rerun.log`: clean seed-1234 rerun, final BPB
  `1.05924929`.
