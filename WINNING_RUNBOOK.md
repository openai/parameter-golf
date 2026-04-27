# Parameter Golf Winning Runbook ($25 Edition)

This runbook is optimized for low budget and high decision quality.
Goal: make one real SOTA attempt without wasting credits.

## 0) Non-negotiables

- Beat current SOTA with margin and significance, not one lucky seed.
- Keep artifact under `16,000,000` bytes (decimal MB).
- Never use validation or train data illegally during quantization/eval.
- Prefer cheap filtering first, then expensive confirmation.

## 1) Budget split

- Phase A (cheap filtering): `$8`
- Phase B (1xH100 confirmation): `$9`
- Phase C (final 8xH100 reproducibility): `$8` (or wait for grant)

If you get OpenAI credits, expand Phase C to 3-seed evidence.

## 2) Exact baseline to start from

Use this folder as your starting point:

- `records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/`

Do not start from old baseline scripts.

## 3) Runpod setup (first pod)

1. Launch cheap single-GPU pod first (L40/4090/5090 class).
2. SSH in and run:
   - `cd /workspace`
   - `git clone https://github.com/openai/parameter-golf.git`
   - `cd parameter-golf`
   - `python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1`
3. Copy your working `train_gpt.py` candidate into a new local work folder.

## 4) Experiment matrix (run in this order)

Use `commands/runpod_experiments.sh`.

Design principle:
- Change one high-impact axis at a time.
- Keep all other vars fixed.
- Promote only stable gains.

Priority axes:
- `GPTQ_CALIB_BATCHES`: 192/256/320
- `GPTQ_BLOCK_SIZE`: 128/256
- `BIGRAM_DIM`: 96/112/128 (with `BIGRAM_VOCAB_SIZE=3072`)
- `WARMDOWN_ITERS`: 3500/4000/4500
- `TARGET_MB`: 15.85/15.90

## 5) Stop/go rules (strict)

- If run regresses by `>= 0.0015 bpb` vs control: stop that branch.
- If run improves by `< 0.0007 bpb`: do not promote.
- Promote only if improved in 2 seeds (cheap pod is fine for this check).
- Spend H100 only on top 1-2 configs.

## 6) Promotion checklist before H100

- Script runs clean with no dependency errors.
- Final lines print `val_bpb` and compressed model size.
- Artifact clearly below 16MB target.
- No rule-violating data access in quantization/eval path.

## 7) Submission checklist

Create a new folder in `records/track_10min_16mb/<date>_<name>/` with:

- `README.md` (what changed, why, exact command)
- `submission.json`
- `train_gpt.py`
- `train.log` (or multiple logs for significance)
- `requirements.txt` only if non-default deps were needed

## 8) Your daily cadence (copy exactly)

1. 6 cheap ablations (Phase A).
2. Pick top 2 and re-run with new seeds.
3. Move best to 1xH100 (Phase B).
4. If still positive, run final reproducibility pass (Phase C or grant credits).
5. Submit PR the same day while evidence is fresh.

## 9) One hard rule

Do not chase tiny LR/WD decimal tweaks until your quantization + calibration stack is already clearly beating your current best.
