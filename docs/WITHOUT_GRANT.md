# Working without compute credits

You can still make real progress before 8×H100 time.

## 1. Validate the submission script (CPU, ~5s)

From the repo root (with `torch` installed):

```bash
python3 scripts/validate_submission.py
```

This checks syntax, line count, sliding-window coverage (including full val token count), `forward_logits` vs `forward`, STE gradients, and int6 roundtrip.

## 2. Train on Apple Silicon (MLX)

The official MLX path uses root `train_gpt_mlx.py`, not the CUDA submission in `records/`. Use MLX to debug data loading, shapes, and training dynamics:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install mlx numpy sentencepiece huggingface-hub datasets tqdm
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
RUN_ID=mlx_smoke ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 \
  python3 train_gpt_mlx.py
```

Use default `VAL_BATCH_SIZE` (524288) so the single end-of-run validation does not take forever.

## 3. Push your fork and open a draft PR

You do not need a score to start review prep:

1. Clone your fork: `git clone https://github.com/0xjaishy/parameter-golf.git`
2. Branch: `git checkout -b submission/your-name`
3. Add only your `records/...` folder + any docs
4. Commit and `git push -u origin submission/your-name`
5. Open a PR to `openai/parameter-golf` — mark as draft until `train.log` and `submission.json` metrics are filled

## 4. Cheap GPU smoke (optional)

Rent a **single** H100 or A100 for an hour on RunPod (paid, not grant) to catch CUDA-only bugs (DDP, compile, NCCL) before an 8-GPU run.

## 5. When you have 8×H100

Run the submission `train_gpt.py` from its record folder with `torchrun --nproc_per_node=8`, copy logs into the same folder, update `submission.json`, and mark the PR ready.
