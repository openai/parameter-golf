# RUNBOOK (8xH100, 10-minute budget)

This runbook documents how to rerun and verify the submission in this folder.

## 1) Environment

- GPU: 8x H100 SXM
- Python deps: `torch`, `sentencepiece`, `numpy`
- Working dir:

```bash
cd /workspace/parameter-golf/records/track_non_record_16mb/2026-04-02_Meadow_TextDiffusion_Retrodiction_TTT_DepthRecurrence
```

## 2) AR + Retrodiction (5L d=256, v4096)

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py \
  --train_budget_secs 540 \
  --steps 99999 \
  --grad_accum 1 \
  --microbatch_tokens 65536 \
  --val_every 500 \
  --val_tokens 1000000 \
  --data_dir /workspace/data_v4096_full \
  --tokenizer_path /workspace/bpe_v4096.model \
  --save_path model_5L_v4096.npz \
  --save_int6_path model_5L_v4096_int6.lzma
```

Expected outputs:
- `model_5L_v4096.npz`
- `model_5L_v4096_int6.lzma`
- train log with `FINAL val_bpb`

## 3) Shared AR+CDM (single model, SP1024)

```bash
torchrun --standalone --nproc_per_node=8 train_cdm.py \
  --train_budget_secs 540 \
  --steps 99999 \
  --grad_accum 1 \
  --microbatch_tokens 65536 \
  --val_every 500 \
  --val_tokens 1000000 \
  --data_dir /data/datasets/fineweb10B_sp1024 \
  --tokenizer_path /data/tokenizers/fineweb_1024_bpe.model \
  --save_path shared_ar_cdm.npz \
  --save_int6_path shared_ar_cdm_int6.lzma
```

Expected outputs:
- `shared_ar_cdm.npz`
- `shared_ar_cdm_int6.lzma`
- train log with `FINAL val_bpb`

## 4) Evaluation

Sequential Unmasking script uses fixed paths/checkpoint names in-code:

```bash
python3 eval_sequential_unmasking.py
```

TTT evaluation:

```bash
python3 eval_ttt.py \
  --model_path model_5L_v4096.npz \
  --model_dim 512 \
  --val_tokens 500000 \
  --ttt_lr 3e-4 \
  --ttt_steps 1
```

## 5) Verification checklist

1. Training budget enforcement:
   - log shows budget trigger near 540s.
2. Metric extraction:
   - log includes `FINAL val_bpb`.
3. Artifact size:
   - `ls -lh *.lzma` and confirm compressed model is under 16MB.
4. PR metadata consistency:
   - `submission.json` values match reported outputs.

## 6) Optional reproducibility protocol

- Run 3-5 repeats (separate logs).
- Report mean/std for `val_bpb`.
- Attach all logs to the PR thread for auditability.
