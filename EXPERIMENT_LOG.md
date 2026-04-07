# Experiment Log

Use this file to keep a running record of experiment configs, outcomes, and takeaways.
Append a new entry after every experiment.

## Entry Template

### YYYY-MM-DD - experiment_name
- Status: pending | running | completed | failed
- Script: `...`
- Config: `...`
- Dataset: `...`
- Goal: ...
- Key settings: ...
- Result summary: ...
- Metrics: ...
- Artifacts: `...`
- Findings: ...
- Next step: ...

## Entries

### 2026-04-06 - diffusion_tiny_synthetic_smoketest
- Status: completed
- Script: [`train_diffusion.py`](/Users/archit/Local/parameter-golf/train_diffusion.py)
- Config: [`configs/diffusion_tiny.env`](/Users/archit/Local/parameter-golf/configs/diffusion_tiny.env)
- Dataset: synthetic repeated-pattern debug data
- Goal: verify the week-1 MLX diffusion trainer runs end-to-end on a tiny overfit setup
- Key settings: `TRAIN_SEQ_LEN=64`, `TRAIN_BATCH_TOKENS=2048`, `ITERATIONS=400`, `NUM_LAYERS=4`, `MODEL_DIM=128`, `NUM_DIFFUSION_STEPS=16`, `MASK_SCHEDULE=linear`
- Result summary: training completed successfully after one sample-path bug fix for NumPy 2.x compatibility
- Metrics: `val_loss` `3.5647 -> 0.2150`, final step `400/400`, saved model size `2,115,960` bytes
- Artifacts: [`logs/diffusion_tiny_diffusion.txt`](/Users/archit/Local/parameter-golf/logs/diffusion_tiny_diffusion.txt), [`logs/diffusion_tiny_diffusion_mlx.npz`](/Users/archit/Local/parameter-golf/logs/diffusion_tiny_diffusion_mlx.npz)
- Findings: core training loop, corruption process, validation, sampling, and serialization all work on the tiny synthetic setup; MLX execution requires unsandboxed Metal access in this environment
- Next step: run the FineWeb local config and record loss curve, sample quality, throughput, and any stability issues

### 2026-04-06 - diffusion_local_5min_progress
- Status: completed
- Script: [`train_diffusion.py`](/Users/archit/Local/parameter-golf/train_diffusion.py)
- Config: [`configs/diffusion_local.env`](/Users/archit/Local/parameter-golf/configs/diffusion_local.env)
- Dataset: FineWeb `sp1024`, `TRAIN_SHARDS=1`
- Goal: verify that the week-1 diffusion model makes early training progress on a real FineWeb shard within a short Mac-friendly run
- Key settings: `RUN_ID=diffusion_local_5min_progress`, `ITERATIONS=100`, `TRAIN_SEQ_LEN=256`, `TRAIN_BATCH_TOKENS=16384`, `NUM_LAYERS=6`, `MODEL_DIM=256`, `NUM_DIFFUSION_STEPS=32`, `MASK_SCHEDULE=cosine`, `LEARNING_RATE=0.0003`, `VAL_LOSS_EVERY=0`, `VAL_AT_START=0`, `VAL_AT_END=0`, `SAMPLE_EVERY=25`
- Result summary: run completed successfully and showed clear early optimization progress on real data
- Metrics: train loss `6.9802 -> 6.0468` over `100` steps, throughput roughly `13k -> 22k` then `15k -> 19k` tokens/sec, saved model size `13,133,268` bytes
- Artifacts: [`logs/diffusion_local_5min_progress_diffusion.txt`](/Users/archit/Local/parameter-golf/logs/diffusion_local_5min_progress_diffusion.txt), [`logs/diffusion_local_5min_progress_diffusion_mlx.npz`](/Users/archit/Local/parameter-golf/logs/diffusion_local_5min_progress_diffusion_mlx.npz)
- Findings: samples remained noisy but text-shaped and became more coherent by step 100; loss improvement is real but modest, which is expected for a short first-pass run on one shard; full validation is too expensive for short exploratory runs, so `VAL_AT_START=0` and `VAL_AT_END=0` are useful additions for quick iteration
- Next step: run a slightly longer one-shard experiment with periodic validation only after training has started, or tune learning rate / batch size to improve the early loss slope

### 2026-04-06 - diffusion_local_smallval_5min_check
- Status: completed
- Script: [`train_diffusion.py`](/Users/archit/Local/parameter-golf/train_diffusion.py)
- Config: [`configs/diffusion_local_smallval.env`](/Users/archit/Local/parameter-golf/configs/diffusion_local_smallval.env)
- Dataset: FineWeb `sp1024`, `TRAIN_SHARDS=1`, reduced validation subset
- Goal: verify that local periodic validation works correctly on a small validation subset before starting a longer run
- Key settings: `RUN_ID=diffusion_local_smallval_5min_check`, `ITERATIONS=100`, `TRAIN_SEQ_LEN=256`, `TRAIN_BATCH_TOKENS=16384`, `VAL_MAX_TOKENS=262144`, `VAL_LOSS_EVERY=25`, `SAMPLE_EVERY=25`, `MAX_WALLCLOCK_SECONDS=300`
- Result summary: run completed successfully with reduced validation enabled; both training loss and validation loss moved in the right direction
- Metrics: `val_loss` `6.9734 -> 6.0787 -> 6.0240 -> 6.0231`, train loss `6.9802 -> 5.9735` by step 50, saved model size `13,133,268` bytes
- Artifacts: [`logs/diffusion_local_smallval_5min_check_diffusion.txt`](/Users/archit/Local/parameter-golf/logs/diffusion_local_smallval_5min_check_diffusion.txt), [`logs/diffusion_local_smallval_5min_check_diffusion_mlx.npz`](/Users/archit/Local/parameter-golf/logs/diffusion_local_smallval_5min_check_diffusion_mlx.npz)
- Findings: smaller validation is working correctly and is fast enough for local sanity checks; periodic validation still adds enough overhead that the 5-minute wallclock cap ended this run early around step 51, so longer runs should either increase the wallclock cap or validate less frequently
- Next step: launch a longer one-shard run with `VAL_MAX_TOKENS` enabled and a looser wallclock budget, or keep the 5-minute budget and reduce validation frequency to every 50-100 steps

### 2026-04-06 - diffusion_local_long
- Status: completed
- Script: [`train_diffusion.py`](/Users/archit/Local/parameter-golf/train_diffusion.py)
- Config: [`configs/diffusion_local_long.env`](/Users/archit/Local/parameter-golf/configs/diffusion_local_long.env)
- Dataset: FineWeb `sp1024`, `TRAIN_SHARDS=1`, reduced validation subset
- Goal: check whether the week-1 diffusion baseline continues improving over a longer local run with periodic small-validation checkpoints
- Key settings: `ITERATIONS=400`, `TRAIN_SEQ_LEN=256`, `TRAIN_BATCH_TOKENS=16384`, `VAL_MAX_TOKENS=262144`, `VAL_LOSS_EVERY=100`, `SAMPLE_EVERY=100`, `MAX_WALLCLOCK_SECONDS=1800`
- Result summary: run completed cleanly, but optimization plateaued early; validation improved quickly at first and then mostly flattened
- Metrics: train loss `6.9802 -> 5.9979`; val loss `6.9734 -> 6.0147 -> 6.0093 -> 6.0075 -> 6.0061`; throughput mostly `37k-41k tok/s`; saved model size `13,133,268` bytes
- Artifacts: [`logs/diffusion_local_long_diffusion.txt`](/Users/archit/Local/parameter-golf/logs/diffusion_local_long_diffusion.txt), [`logs/diffusion_local_long_console.txt`](/Users/archit/Local/parameter-golf/logs/diffusion_local_long_console.txt), [`logs/diffusion_local_long_diffusion_mlx.npz`](/Users/archit/Local/parameter-golf/logs/diffusion_local_long_diffusion_mlx.npz)
- Findings: most gains happen in the first 50-100 steps; after that both train and validation losses oscillate in a narrow band near `6.0`; samples become more text-like but remain far from fluent, suggesting the model is learning local token statistics but not yet denoising strongly enough
- Next step: tune the optimization and corruption recipe before scaling duration further, especially learning rate, mask schedule, diffusion-step count, and batch size

### 2026-04-07 - overnight_diffusion_apr07_e01_256d_256seq_lr5e4_16steps
- Status: completed
- Script: [`train_diffusion.py`](/Users/archit/Local/parameter-golf/train_diffusion.py)
- Config: [`configs/overnight/e01_256d_256seq_lr5e4_16steps.env`](/Users/archit/Local/parameter-golf/configs/overnight/e01_256d_256seq_lr5e4_16steps.env)
- Dataset: FineWeb `sp1024`, local overnight suite
- Goal: Higher learning rate plus fewer diffusion steps on the stable 256x256 baseline.
- Key settings: seq=256 dim=256 layers=6 heads=8 batch_tokens=16384 diff_steps=16 schedule=cosine max_mask=1.0 lr=0.0005 iterations=800
- Result summary: overnight suite auto-entry
- Metrics: train_loss=`5.8211`, val_loss=`5.8108`, model_bytes=`13125076`
- Artifacts: [`overnight_diffusion_apr07_e01_256d_256seq_lr5e4_16steps_diffusion.txt`](/Users/archit/Local/parameter-golf/logs/overnight_diffusion_apr07/overnight_diffusion_apr07_e01_256d_256seq_lr5e4_16steps_diffusion.txt), [`overnight_diffusion_apr07_e01_256d_256seq_lr5e4_16steps_console.txt`](/Users/archit/Local/parameter-golf/logs/overnight_diffusion_apr07/overnight_diffusion_apr07_e01_256d_256seq_lr5e4_16steps_console.txt), [`overnight_diffusion_apr07_e01_256d_256seq_lr5e4_16steps_diffusion_mlx.npz`](/Users/archit/Local/parameter-golf/logs/overnight_diffusion_apr07/overnight_diffusion_apr07_e01_256d_256seq_lr5e4_16steps_diffusion_mlx.npz)
- Findings: pending morning review
- Next step: compare against the rest of the overnight suite

### 2026-04-07 - overnight_diffusion_apr07_e02_256d_256seq_lr5e4_16steps_mask08
- Status: completed
- Script: [`train_diffusion.py`](/Users/archit/Local/parameter-golf/train_diffusion.py)
- Config: [`configs/overnight/e02_256d_256seq_lr5e4_16steps_mask08.env`](/Users/archit/Local/parameter-golf/configs/overnight/e02_256d_256seq_lr5e4_16steps_mask08.env)
- Dataset: FineWeb `sp1024`, local overnight suite
- Goal: Make denoising easier by combining a higher learning rate with fewer diffusion steps and a lower maximum mask rate.
- Key settings: seq=256 dim=256 layers=6 heads=8 batch_tokens=16384 diff_steps=16 schedule=cosine max_mask=0.8 lr=0.0005 iterations=800
- Result summary: overnight suite auto-entry
- Metrics: train_loss=`5.4625`, val_loss=`5.4395`, model_bytes=`13125076`
- Artifacts: [`overnight_diffusion_apr07_e02_256d_256seq_lr5e4_16steps_mask08_diffusion.txt`](/Users/archit/Local/parameter-golf/logs/overnight_diffusion_apr07/overnight_diffusion_apr07_e02_256d_256seq_lr5e4_16steps_mask08_diffusion.txt), [`overnight_diffusion_apr07_e02_256d_256seq_lr5e4_16steps_mask08_console.txt`](/Users/archit/Local/parameter-golf/logs/overnight_diffusion_apr07/overnight_diffusion_apr07_e02_256d_256seq_lr5e4_16steps_mask08_console.txt), [`overnight_diffusion_apr07_e02_256d_256seq_lr5e4_16steps_mask08_diffusion_mlx.npz`](/Users/archit/Local/parameter-golf/logs/overnight_diffusion_apr07/overnight_diffusion_apr07_e02_256d_256seq_lr5e4_16steps_mask08_diffusion_mlx.npz)
- Findings: pending morning review
- Next step: compare against the rest of the overnight suite

### 2026-04-07 - overnight_diffusion_apr07_e03_384d_256seq_lr3e4_32steps
- Status: completed
- Script: [`train_diffusion.py`](/Users/archit/Local/parameter-golf/train_diffusion.py)
- Config: [`configs/overnight/e03_384d_256seq_lr3e4_32steps.env`](/Users/archit/Local/parameter-golf/configs/overnight/e03_384d_256seq_lr3e4_32steps.env)
- Dataset: FineWeb `sp1024`, local overnight suite
- Goal: Isolate model width by scaling to 384 dimensions while keeping the original 32-step corruption setup.
- Key settings: seq=256 dim=384 layers=6 heads=8 batch_tokens=16384 diff_steps=32 schedule=cosine max_mask=1.0 lr=0.0003 iterations=800
- Result summary: overnight suite auto-entry
- Metrics: train_loss=`5.8214`, val_loss=`5.7787`, model_bytes=`29132500`
- Artifacts: [`overnight_diffusion_apr07_e03_384d_256seq_lr3e4_32steps_diffusion.txt`](/Users/archit/Local/parameter-golf/logs/overnight_diffusion_apr07/overnight_diffusion_apr07_e03_384d_256seq_lr3e4_32steps_diffusion.txt), [`overnight_diffusion_apr07_e03_384d_256seq_lr3e4_32steps_console.txt`](/Users/archit/Local/parameter-golf/logs/overnight_diffusion_apr07/overnight_diffusion_apr07_e03_384d_256seq_lr3e4_32steps_console.txt), [`overnight_diffusion_apr07_e03_384d_256seq_lr3e4_32steps_diffusion_mlx.npz`](/Users/archit/Local/parameter-golf/logs/overnight_diffusion_apr07/overnight_diffusion_apr07_e03_384d_256seq_lr3e4_32steps_diffusion_mlx.npz)
- Findings: pending morning review
- Next step: compare against the rest of the overnight suite

### 2026-04-07 - overnight_diffusion_apr07_e04_384d_256seq_lr5e4_16steps
- Status: completed
- Script: [`train_diffusion.py`](/Users/archit/Local/parameter-golf/train_diffusion.py)
- Config: [`configs/overnight/e04_384d_256seq_lr5e4_16steps.env`](/Users/archit/Local/parameter-golf/configs/overnight/e04_384d_256seq_lr5e4_16steps.env)
- Dataset: FineWeb `sp1024`, local overnight suite
- Goal: Primary best-bet run: more width, fewer diffusion steps, and a higher learning rate.
- Key settings: seq=256 dim=384 layers=6 heads=8 batch_tokens=16384 diff_steps=16 schedule=cosine max_mask=1.0 lr=0.0005 iterations=800
- Result summary: overnight suite auto-entry
- Metrics: train_loss=`5.5503`, val_loss=`5.5842`, model_bytes=`29120212`
- Artifacts: [`overnight_diffusion_apr07_e04_384d_256seq_lr5e4_16steps_diffusion.txt`](/Users/archit/Local/parameter-golf/logs/overnight_diffusion_apr07/overnight_diffusion_apr07_e04_384d_256seq_lr5e4_16steps_diffusion.txt), [`overnight_diffusion_apr07_e04_384d_256seq_lr5e4_16steps_console.txt`](/Users/archit/Local/parameter-golf/logs/overnight_diffusion_apr07/overnight_diffusion_apr07_e04_384d_256seq_lr5e4_16steps_console.txt), [`overnight_diffusion_apr07_e04_384d_256seq_lr5e4_16steps_diffusion_mlx.npz`](/Users/archit/Local/parameter-golf/logs/overnight_diffusion_apr07/overnight_diffusion_apr07_e04_384d_256seq_lr5e4_16steps_diffusion_mlx.npz)
- Findings: pending morning review
- Next step: compare against the rest of the overnight suite

### 2026-04-07 - overnight_diffusion_apr07_e05_384d_256seq_lr5e4_16steps_mask08
- Status: completed
- Script: [`train_diffusion.py`](/Users/archit/Local/parameter-golf/train_diffusion.py)
- Config: [`configs/overnight/e05_384d_256seq_lr5e4_16steps_mask08.env`](/Users/archit/Local/parameter-golf/configs/overnight/e05_384d_256seq_lr5e4_16steps_mask08.env)
- Dataset: FineWeb `sp1024`, local overnight suite
- Goal: Primary best-bet variant with easier corruption pressure via MAX_MASK_RATE=0.8.
- Key settings: seq=256 dim=384 layers=6 heads=8 batch_tokens=16384 diff_steps=16 schedule=cosine max_mask=0.8 lr=0.0005 iterations=800
- Result summary: overnight suite auto-entry
- Metrics: train_loss=`5.2916`, val_loss=`5.2695`, model_bytes=`29120212`
- Artifacts: [`overnight_diffusion_apr07_e05_384d_256seq_lr5e4_16steps_mask08_diffusion.txt`](/Users/archit/Local/parameter-golf/logs/overnight_diffusion_apr07/overnight_diffusion_apr07_e05_384d_256seq_lr5e4_16steps_mask08_diffusion.txt), [`overnight_diffusion_apr07_e05_384d_256seq_lr5e4_16steps_mask08_console.txt`](/Users/archit/Local/parameter-golf/logs/overnight_diffusion_apr07/overnight_diffusion_apr07_e05_384d_256seq_lr5e4_16steps_mask08_console.txt), [`overnight_diffusion_apr07_e05_384d_256seq_lr5e4_16steps_mask08_diffusion_mlx.npz`](/Users/archit/Local/parameter-golf/logs/overnight_diffusion_apr07/overnight_diffusion_apr07_e05_384d_256seq_lr5e4_16steps_mask08_diffusion_mlx.npz)
- Findings: pending morning review
- Next step: compare against the rest of the overnight suite

### 2026-04-07 - overnight_diffusion_apr07_e06_384d_256seq_8layers_lr4e4_16steps
- Status: completed
- Script: [`train_diffusion.py`](/Users/archit/Local/parameter-golf/train_diffusion.py)
- Config: [`configs/overnight/e06_384d_256seq_8layers_lr4e4_16steps.env`](/Users/archit/Local/parameter-golf/configs/overnight/e06_384d_256seq_8layers_lr4e4_16steps.env)
- Dataset: FineWeb `sp1024`, local overnight suite
- Goal: Add depth on top of the wider model to test whether denoising benefits more from depth than duration.
- Key settings: seq=256 dim=384 layers=8 heads=8 batch_tokens=16384 diff_steps=16 schedule=cosine max_mask=1.0 lr=0.0004 iterations=700
- Result summary: overnight suite auto-entry
- Metrics: train_loss=`5.6017`, val_loss=`5.6607`, model_bytes=`38560300`
- Artifacts: [`overnight_diffusion_apr07_e06_384d_256seq_8layers_lr4e4_16steps_diffusion.txt`](/Users/archit/Local/parameter-golf/logs/overnight_diffusion_apr07/overnight_diffusion_apr07_e06_384d_256seq_8layers_lr4e4_16steps_diffusion.txt), [`overnight_diffusion_apr07_e06_384d_256seq_8layers_lr4e4_16steps_console.txt`](/Users/archit/Local/parameter-golf/logs/overnight_diffusion_apr07/overnight_diffusion_apr07_e06_384d_256seq_8layers_lr4e4_16steps_console.txt), [`overnight_diffusion_apr07_e06_384d_256seq_8layers_lr4e4_16steps_diffusion_mlx.npz`](/Users/archit/Local/parameter-golf/logs/overnight_diffusion_apr07/overnight_diffusion_apr07_e06_384d_256seq_8layers_lr4e4_16steps_diffusion_mlx.npz)
- Findings: pending morning review
- Next step: compare against the rest of the overnight suite

### 2026-04-07 - overnight_diffusion_apr07_e07_512d_256seq_lr3e4_16steps
- Status: completed
- Script: [`train_diffusion.py`](/Users/archit/Local/parameter-golf/train_diffusion.py)
- Config: [`configs/overnight/e07_512d_256seq_lr3e4_16steps.env`](/Users/archit/Local/parameter-golf/configs/overnight/e07_512d_256seq_lr3e4_16steps.env)
- Dataset: FineWeb `sp1024`, local overnight suite
- Goal: Exploit the available local runtime by pushing embedding width substantially higher at the same sequence length.
- Key settings: seq=256 dim=512 layers=6 heads=8 batch_tokens=16384 diff_steps=16 schedule=cosine max_mask=1.0 lr=0.0003 iterations=700
- Result summary: overnight suite auto-entry
- Metrics: train_loss=`5.6274`, val_loss=`5.6841`, model_bytes=`51406828`
- Artifacts: [`overnight_diffusion_apr07_e07_512d_256seq_lr3e4_16steps_diffusion.txt`](/Users/archit/Local/parameter-golf/logs/overnight_diffusion_apr07/overnight_diffusion_apr07_e07_512d_256seq_lr3e4_16steps_diffusion.txt), [`overnight_diffusion_apr07_e07_512d_256seq_lr3e4_16steps_console.txt`](/Users/archit/Local/parameter-golf/logs/overnight_diffusion_apr07/overnight_diffusion_apr07_e07_512d_256seq_lr3e4_16steps_console.txt), [`overnight_diffusion_apr07_e07_512d_256seq_lr3e4_16steps_diffusion_mlx.npz`](/Users/archit/Local/parameter-golf/logs/overnight_diffusion_apr07/overnight_diffusion_apr07_e07_512d_256seq_lr3e4_16steps_diffusion_mlx.npz)
- Findings: pending morning review
- Next step: compare against the rest of the overnight suite

### 2026-04-07 - overnight_diffusion_apr07_e08_256d_512seq_lr5e4_16steps
- Status: completed
- Script: [`train_diffusion.py`](/Users/archit/Local/parameter-golf/train_diffusion.py)
- Config: [`configs/overnight/e08_256d_512seq_lr5e4_16steps.env`](/Users/archit/Local/parameter-golf/configs/overnight/e08_256d_512seq_lr5e4_16steps.env)
- Dataset: FineWeb `sp1024`, local overnight suite
- Goal: Test whether longer context alone helps the diffusion model more than extra width.
- Key settings: seq=512 dim=256 layers=6 heads=8 batch_tokens=16384 diff_steps=16 schedule=cosine max_mask=1.0 lr=0.0005 iterations=700
- Result summary: overnight suite auto-entry
- Metrics: train_loss=`5.8982`, val_loss=`5.8956`, model_bytes=`13125076`
- Artifacts: [`overnight_diffusion_apr07_e08_256d_512seq_lr5e4_16steps_diffusion.txt`](/Users/archit/Local/parameter-golf/logs/overnight_diffusion_apr07/overnight_diffusion_apr07_e08_256d_512seq_lr5e4_16steps_diffusion.txt), [`overnight_diffusion_apr07_e08_256d_512seq_lr5e4_16steps_console.txt`](/Users/archit/Local/parameter-golf/logs/overnight_diffusion_apr07/overnight_diffusion_apr07_e08_256d_512seq_lr5e4_16steps_console.txt), [`overnight_diffusion_apr07_e08_256d_512seq_lr5e4_16steps_diffusion_mlx.npz`](/Users/archit/Local/parameter-golf/logs/overnight_diffusion_apr07/overnight_diffusion_apr07_e08_256d_512seq_lr5e4_16steps_diffusion_mlx.npz)
- Findings: pending morning review
- Next step: compare against the rest of the overnight suite

### 2026-04-07 - overnight_diffusion_apr07_e09_384d_512seq_lr4e4_16steps
- Status: completed
- Script: [`train_diffusion.py`](/Users/archit/Local/parameter-golf/train_diffusion.py)
- Config: [`configs/overnight/e09_384d_512seq_lr4e4_16steps.env`](/Users/archit/Local/parameter-golf/configs/overnight/e09_384d_512seq_lr4e4_16steps.env)
- Dataset: FineWeb `sp1024`, local overnight suite
- Goal: Combine the two most plausible scale directions: wider embeddings and longer context.
- Key settings: seq=512 dim=384 layers=6 heads=8 batch_tokens=16384 diff_steps=16 schedule=cosine max_mask=1.0 lr=0.0004 iterations=600
- Result summary: overnight suite auto-entry
- Metrics: train_loss=`5.9041`, val_loss=`5.8938`, model_bytes=`29120212`
- Artifacts: [`overnight_diffusion_apr07_e09_384d_512seq_lr4e4_16steps_diffusion.txt`](/Users/archit/Local/parameter-golf/logs/overnight_diffusion_apr07/overnight_diffusion_apr07_e09_384d_512seq_lr4e4_16steps_diffusion.txt), [`overnight_diffusion_apr07_e09_384d_512seq_lr4e4_16steps_console.txt`](/Users/archit/Local/parameter-golf/logs/overnight_diffusion_apr07/overnight_diffusion_apr07_e09_384d_512seq_lr4e4_16steps_console.txt), [`overnight_diffusion_apr07_e09_384d_512seq_lr4e4_16steps_diffusion_mlx.npz`](/Users/archit/Local/parameter-golf/logs/overnight_diffusion_apr07/overnight_diffusion_apr07_e09_384d_512seq_lr4e4_16steps_diffusion_mlx.npz)
- Findings: pending morning review
- Next step: compare against the rest of the overnight suite

### 2026-04-07 - overnight_diffusion_apr07_e10_384d_512seq_lr4e4_16steps_mask08
- Status: completed
- Script: [`train_diffusion.py`](/Users/archit/Local/parameter-golf/train_diffusion.py)
- Config: [`configs/overnight/e10_384d_512seq_lr4e4_16steps_mask08.env`](/Users/archit/Local/parameter-golf/configs/overnight/e10_384d_512seq_lr4e4_16steps_mask08.env)
- Dataset: FineWeb `sp1024`, local overnight suite
- Goal: Best-bet long-context variant with reduced corruption severity for an easier denoising target.
- Key settings: seq=512 dim=384 layers=6 heads=8 batch_tokens=16384 diff_steps=16 schedule=cosine max_mask=0.8 lr=0.0004 iterations=600
- Result summary: overnight suite auto-entry
- Metrics: train_loss=`5.7893`, val_loss=`5.7500`, model_bytes=`29120212`
- Artifacts: [`overnight_diffusion_apr07_e10_384d_512seq_lr4e4_16steps_mask08_diffusion.txt`](/Users/archit/Local/parameter-golf/logs/overnight_diffusion_apr07/overnight_diffusion_apr07_e10_384d_512seq_lr4e4_16steps_mask08_diffusion.txt), [`overnight_diffusion_apr07_e10_384d_512seq_lr4e4_16steps_mask08_console.txt`](/Users/archit/Local/parameter-golf/logs/overnight_diffusion_apr07/overnight_diffusion_apr07_e10_384d_512seq_lr4e4_16steps_mask08_console.txt), [`overnight_diffusion_apr07_e10_384d_512seq_lr4e4_16steps_mask08_diffusion_mlx.npz`](/Users/archit/Local/parameter-golf/logs/overnight_diffusion_apr07/overnight_diffusion_apr07_e10_384d_512seq_lr4e4_16steps_mask08_diffusion_mlx.npz)
- Findings: pending morning review
- Next step: compare against the rest of the overnight suite

### 2026-04-07 - overnight_diffusion_apr07_suite
- Status: completed
- Script: [`train_diffusion.py`](/Users/archit/Local/parameter-golf/train_diffusion.py)
- Config: overnight suite in [`configs/overnight`](/Users/archit/Local/parameter-golf/configs/overnight)
- Dataset: FineWeb `sp1024`, `TRAIN_SHARDS=1`, reduced validation subset `VAL_MAX_TOKENS=262144`
- Goal: identify the most promising next direction by ablating corruption severity, diffusion-step count, width, depth, and sequence length
- Key settings: 10 sequential runs covering `MODEL_DIM` in `{256,384,512}`, `TRAIN_SEQ_LEN` in `{256,512}`, `NUM_DIFFUSION_STEPS` in `{16,32}`, `MAX_MASK_RATE` in `{1.0,0.8}`, and modest LR/depth changes
- Result summary: the strongest improvement came from making corruption easier; width helped after that, while longer sequence length and extra depth did not
- Metrics:
  - best: `e05_384d_256seq_lr5e4_16steps_mask08` with `val_loss=5.2695`, `train_loss=5.2916`
  - second: `e02_256d_256seq_lr5e4_16steps_mask08` with `val_loss=5.4395`
  - third: `e04_384d_256seq_lr5e4_16steps` with `val_loss=5.5842`
  - worst: `e08_256d_512seq_lr5e4_16steps` with `val_loss=5.8956`
- Artifacts: [`logs/overnight_diffusion_apr07/summary.tsv`](/Users/archit/Local/parameter-golf/logs/overnight_diffusion_apr07/summary.tsv), [`logs/overnight_diffusion_apr07/runner.txt`](/Users/archit/Local/parameter-golf/logs/overnight_diffusion_apr07/runner.txt)
- Findings:
  - lowering `MAX_MASK_RATE` from `1.0` to `0.8` was the biggest single gain
  - at `256d,256seq,16steps`, `MAX_MASK_RATE=0.8` improved `val_loss` from `5.8108` to `5.4395`
  - at `384d,256seq,16steps`, `MAX_MASK_RATE=0.8` improved `val_loss` from `5.5842` to `5.2695`
  - increasing width from `256` to `384` helped when paired with the easier corruption setup
  - `32` diffusion steps underperformed `16` in this local regime
  - increasing sequence length to `512` hurt badly at the current batch budget
  - increasing depth from `6` to `8` layers did not help
  - sample quality improved substantially on the best mask-0.8 runs, but generation is still noisy and not yet fluent
- Next step: adopt `384d, 256seq, 16steps, cosine, MAX_MASK_RATE=0.8, LR=5e-4` as the new baseline and run focused follow-up ablations around mask rate, batch size, and diffusion-step count

### 2026-04-07 - followup_diffusion_20260407_110256_f01_best_baseline_repeat
- Status: completed
- Script: [`train_diffusion.py`](/Users/archit/Local/parameter-golf/train_diffusion.py)
- Config: [`configs/followup/f01_best_baseline_repeat.env`](/Users/archit/Local/parameter-golf/configs/followup/f01_best_baseline_repeat.env)
- Dataset: FineWeb `sp1024`, local overnight suite
- Goal: Reconfirm the strongest overnight recipe as the new baseline.
- Key settings: seq=256 dim=384 layers=6 heads=8 batch_tokens=16384 diff_steps=16 schedule=cosine max_mask=0.8 lr=0.0005 iterations=800
- Result summary: overnight suite auto-entry
- Metrics: train_loss=`5.2979`, val_loss=`5.2733`, model_bytes=`29120212`
- Artifacts: [`followup_diffusion_20260407_110256_f01_best_baseline_repeat_diffusion.txt`](/Users/archit/Local/parameter-golf/logs/followup_diffusion_20260407_110256/followup_diffusion_20260407_110256_f01_best_baseline_repeat_diffusion.txt), [`followup_diffusion_20260407_110256_f01_best_baseline_repeat_console.txt`](/Users/archit/Local/parameter-golf/logs/followup_diffusion_20260407_110256/followup_diffusion_20260407_110256_f01_best_baseline_repeat_console.txt), [`followup_diffusion_20260407_110256_f01_best_baseline_repeat_diffusion_mlx.npz`](/Users/archit/Local/parameter-golf/logs/followup_diffusion_20260407_110256/followup_diffusion_20260407_110256_f01_best_baseline_repeat_diffusion_mlx.npz)
- Findings: pending morning review
- Next step: compare against the rest of the overnight suite

### 2026-04-07 - followup_diffusion_20260407_110256_f02_best_mask07
- Status: completed
- Script: [`train_diffusion.py`](/Users/archit/Local/parameter-golf/train_diffusion.py)
- Config: [`configs/followup/f02_best_mask07.env`](/Users/archit/Local/parameter-golf/configs/followup/f02_best_mask07.env)
- Dataset: FineWeb `sp1024`, local overnight suite
- Goal: Test whether slightly easier corruption improves on the new best run.
- Key settings: seq=256 dim=384 layers=6 heads=8 batch_tokens=16384 diff_steps=16 schedule=cosine max_mask=0.7 lr=0.0005 iterations=800
- Result summary: overnight suite auto-entry
- Metrics: train_loss=`5.1463`, val_loss=`5.0936`, model_bytes=`29120212`
- Artifacts: [`followup_diffusion_20260407_110256_f02_best_mask07_diffusion.txt`](/Users/archit/Local/parameter-golf/logs/followup_diffusion_20260407_110256/followup_diffusion_20260407_110256_f02_best_mask07_diffusion.txt), [`followup_diffusion_20260407_110256_f02_best_mask07_console.txt`](/Users/archit/Local/parameter-golf/logs/followup_diffusion_20260407_110256/followup_diffusion_20260407_110256_f02_best_mask07_console.txt), [`followup_diffusion_20260407_110256_f02_best_mask07_diffusion_mlx.npz`](/Users/archit/Local/parameter-golf/logs/followup_diffusion_20260407_110256/followup_diffusion_20260407_110256_f02_best_mask07_diffusion_mlx.npz)
- Findings: pending morning review
- Next step: compare against the rest of the overnight suite

### 2026-04-07 - followup_diffusion_20260407_110256_f03_best_more_batch
- Status: completed
- Script: [`train_diffusion.py`](/Users/archit/Local/parameter-golf/train_diffusion.py)
- Config: [`configs/followup/f03_best_more_batch.env`](/Users/archit/Local/parameter-golf/configs/followup/f03_best_more_batch.env)
- Dataset: FineWeb `sp1024`, local overnight suite
- Goal: Use the best recipe with a larger token batch to test whether optimization smoothness improves.
- Key settings: seq=256 dim=384 layers=6 heads=8 batch_tokens=32768 diff_steps=16 schedule=cosine max_mask=0.8 lr=0.0005 iterations=800
- Result summary: overnight suite auto-entry
- Metrics: train_loss=`5.1443`, val_loss=`5.0742`, model_bytes=`29120212`
- Artifacts: [`followup_diffusion_20260407_110256_f03_best_more_batch_diffusion.txt`](/Users/archit/Local/parameter-golf/logs/followup_diffusion_20260407_110256/followup_diffusion_20260407_110256_f03_best_more_batch_diffusion.txt), [`followup_diffusion_20260407_110256_f03_best_more_batch_console.txt`](/Users/archit/Local/parameter-golf/logs/followup_diffusion_20260407_110256/followup_diffusion_20260407_110256_f03_best_more_batch_console.txt), [`followup_diffusion_20260407_110256_f03_best_more_batch_diffusion_mlx.npz`](/Users/archit/Local/parameter-golf/logs/followup_diffusion_20260407_110256/followup_diffusion_20260407_110256_f03_best_more_batch_diffusion_mlx.npz)
- Findings: pending morning review
- Next step: compare against the rest of the overnight suite

### 2026-04-07 - followup_diffusion_20260407_110256_f04_best_12steps
- Status: completed
- Script: [`train_diffusion.py`](/Users/archit/Local/parameter-golf/train_diffusion.py)
- Config: [`configs/followup/f04_best_12steps.env`](/Users/archit/Local/parameter-golf/configs/followup/f04_best_12steps.env)
- Dataset: FineWeb `sp1024`, local overnight suite
- Goal: Reduce diffusion-step count further to test whether a simpler denoising task helps more.
- Key settings: seq=256 dim=384 layers=6 heads=8 batch_tokens=16384 diff_steps=12 schedule=cosine max_mask=0.8 lr=0.0005 iterations=800
- Result summary: overnight suite auto-entry
- Metrics: train_loss=`5.3573`, val_loss=`5.2496`, model_bytes=`29117140`
- Artifacts: [`followup_diffusion_20260407_110256_f04_best_12steps_diffusion.txt`](/Users/archit/Local/parameter-golf/logs/followup_diffusion_20260407_110256/followup_diffusion_20260407_110256_f04_best_12steps_diffusion.txt), [`followup_diffusion_20260407_110256_f04_best_12steps_console.txt`](/Users/archit/Local/parameter-golf/logs/followup_diffusion_20260407_110256/followup_diffusion_20260407_110256_f04_best_12steps_console.txt), [`followup_diffusion_20260407_110256_f04_best_12steps_diffusion_mlx.npz`](/Users/archit/Local/parameter-golf/logs/followup_diffusion_20260407_110256/followup_diffusion_20260407_110256_f04_best_12steps_diffusion_mlx.npz)
- Findings: pending morning review
- Next step: compare against the rest of the overnight suite

### 2026-04-07 - followup_diffusion_20260407_110256_f05_best_more_batch_mask07
- Status: completed
- Script: [`train_diffusion.py`](/Users/archit/Local/parameter-golf/train_diffusion.py)
- Config: [`configs/followup/f05_best_more_batch_mask07.env`](/Users/archit/Local/parameter-golf/configs/followup/f05_best_more_batch_mask07.env)
- Dataset: FineWeb `sp1024`, local overnight suite
- Goal: Combine the two most promising follow-up directions: larger batch and slightly easier corruption.
- Key settings: seq=256 dim=384 layers=6 heads=8 batch_tokens=32768 diff_steps=16 schedule=cosine max_mask=0.7 lr=0.0005 iterations=800
- Result summary: overnight suite auto-entry
- Metrics: train_loss=`4.8846`, val_loss=`4.7833`, model_bytes=`29120212`
- Artifacts: [`followup_diffusion_20260407_110256_f05_best_more_batch_mask07_diffusion.txt`](/Users/archit/Local/parameter-golf/logs/followup_diffusion_20260407_110256/followup_diffusion_20260407_110256_f05_best_more_batch_mask07_diffusion.txt), [`followup_diffusion_20260407_110256_f05_best_more_batch_mask07_console.txt`](/Users/archit/Local/parameter-golf/logs/followup_diffusion_20260407_110256/followup_diffusion_20260407_110256_f05_best_more_batch_mask07_console.txt), [`followup_diffusion_20260407_110256_f05_best_more_batch_mask07_diffusion_mlx.npz`](/Users/archit/Local/parameter-golf/logs/followup_diffusion_20260407_110256/followup_diffusion_20260407_110256_f05_best_more_batch_mask07_diffusion_mlx.npz)
- Findings: pending morning review
- Next step: compare against the rest of the overnight suite

### 2026-04-07 - followup_diffusion_20260407_110256_suite
- Status: completed
- Script: [`train_diffusion.py`](/Users/archit/Local/parameter-golf/train_diffusion.py)
- Config: follow-up suite in [`configs/followup`](/Users/archit/Local/parameter-golf/configs/followup)
- Dataset: FineWeb `sp1024`, `TRAIN_SHARDS=1`, reduced validation subset `VAL_MAX_TOKENS=262144`
- Goal: refine the new best recipe by testing lower mask pressure, larger token batch, and fewer diffusion steps
- Key settings: all runs used `MODEL_DIM=384`, `TRAIN_SEQ_LEN=256`, `NUM_LAYERS=6`, cosine schedule, and one shard; ablations varied `MAX_MASK_RATE` in `{0.8,0.7}`, `TRAIN_BATCH_TOKENS` in `{16384,32768}`, and `NUM_DIFFUSION_STEPS` in `{16,12}`
- Result summary: all 5 follow-up experiments matched or beat the overnight best; the strongest improvement came from combining `MAX_MASK_RATE=0.7` with `TRAIN_BATCH_TOKENS=32768`
- Metrics:
  - baseline repeat `f01`: `val_loss=5.2733`
  - lower mask rate `f02`: `val_loss=5.0936`
  - larger batch `f03`: `val_loss=5.0742`
  - fewer diffusion steps `f04`: `val_loss=5.2496`
  - larger batch + lower mask rate `f05`: `val_loss=4.7833`
- Artifacts: [`logs/followup_diffusion_20260407_110256/summary.tsv`](/Users/archit/Local/parameter-golf/logs/followup_diffusion_20260407_110256/summary.tsv), [`logs/followup_diffusion_20260407_110256/runner.txt`](/Users/archit/Local/parameter-golf/logs/followup_diffusion_20260407_110256/runner.txt)
- Findings:
  - the overnight best recipe reproduced cleanly (`f01` vs overnight `e05` was effectively equal)
  - lowering `MAX_MASK_RATE` from `0.8` to `0.7` gave a meaningful gain on its own
  - increasing `TRAIN_BATCH_TOKENS` from `16384` to `32768` also gave a meaningful gain on its own
  - combining both changes was strongly synergistic and improved `val_loss` by about `0.49` versus the overnight best (`5.2695 -> 4.7833`)
  - reducing diffusion steps from `16` to `12` helped only slightly and was much less important than mask rate or batch size
  - the best run also showed clearly better sample text, though generation is still noisy and not yet fluent
- Next step: promote `384d,256seq,16steps,cosine,MAX_MASK_RATE=0.7,TRAIN_BATCH_TOKENS=32768,LR=5e-4` to the new local best recipe and explore around that point, especially `MAX_MASK_RATE` in the `0.6-0.75` range and possibly a longer run at the same settings

### 2026-04-07 - baseline_update_after_followup
- Status: completed
- Script: [`train_diffusion.py`](/Users/archit/Local/parameter-golf/train_diffusion.py)
- Config: [`configs/diffusion_local_best.env`](/Users/archit/Local/parameter-golf/configs/diffusion_local_best.env), [`configs/diffusion_local_long.env`](/Users/archit/Local/parameter-golf/configs/diffusion_local_long.env)
- Dataset: FineWeb `sp1024`, `TRAIN_SHARDS=1`, reduced validation subset
- Goal: promote the best known local recipe and prepare longer-run mask-rate comparisons
- Key settings: baseline now uses `MODEL_DIM=384`, `TRAIN_SEQ_LEN=256`, `TRAIN_BATCH_TOKENS=32768`, `NUM_DIFFUSION_STEPS=16`, cosine schedule, `MAX_MASK_RATE=0.7`, `LEARNING_RATE=5e-4`, `GRAD_ACCUM_STEPS=4`
- Result summary: config files updated to the new best recipe and a new 3-run longer-suite added for `MAX_MASK_RATE` in `{0.65, 0.70, 0.75}`
- Metrics: current best reference remains `f05_best_more_batch_mask07` with `val_loss=4.7833`
- Artifacts: [`configs/longrun/manifest.txt`](/Users/archit/Local/parameter-golf/configs/longrun/manifest.txt), [`scripts/run_longrun_diffusion_suite.sh`](/Users/archit/Local/parameter-golf/scripts/run_longrun_diffusion_suite.sh)
- Findings: the most informative next longer runs are local comparisons around mask rate, not architectural changes
- Next step: run either the single longer baseline or the 3-run longrun suite and compare stability versus final validation loss

### 2026-04-07 - longrun_setup_update
- Status: completed
- Script: [`train_diffusion.py`](/Users/archit/Local/parameter-golf/train_diffusion.py)
- Config: [`configs/longrun/manifest.txt`](/Users/archit/Local/parameter-golf/configs/longrun/manifest.txt)
- Dataset: FineWeb `sp1024`, reduced validation subset
- Goal: extend the longer-run suite to better use a 5-hour local window
- Key settings: all longrun configs now use `ITERATIONS=2000` and `MAX_WALLCLOCK_SECONDS=5400`; suite expanded from 3 runs to 4 runs by adding a `TRAIN_SHARDS=2` variant at the current best mask rate
- Result summary: long-run suite now compares `MAX_MASK_RATE` in `{0.65, 0.70, 0.75}` plus one `2`-shard robustness run at `MAX_MASK_RATE=0.7`
- Metrics: current best reference remains `f05_best_more_batch_mask07` with `val_loss=4.7833`
- Artifacts: [`configs/longrun/l04_mask070_2shards.env`](/Users/archit/Local/parameter-golf/configs/longrun/l04_mask070_2shards.env), [`scripts/run_longrun_diffusion_suite.sh`](/Users/archit/Local/parameter-golf/scripts/run_longrun_diffusion_suite.sh)
- Findings: beyond mask-rate comparison, the next most useful question is whether the best recipe remains strong when exposed to more training-shard diversity
- Next step: run the updated 4-run longrun suite if the machine is free for the next ~5 hours

### 2026-04-07 - longrun_diffusion_20260407_142138_suite
- Status: completed
- Script: [`train_diffusion.py`](/Users/archit/Local/parameter-golf/train_diffusion.py)
- Config: longrun suite in [`configs/longrun`](/Users/archit/Local/parameter-golf/configs/longrun)
- Dataset: FineWeb `sp1024`, reduced validation subset `VAL_MAX_TOKENS=262144`; three `1`-shard mask-rate runs plus one `2`-shard control run
- Goal: test whether longer training preserves the mask-rate trend and determine which nearby mask rate is best in the longer regime
- Key settings: all runs used `MODEL_DIM=384`, `TRAIN_SEQ_LEN=256`, `TRAIN_BATCH_TOKENS=32768`, `NUM_LAYERS=6`, `NUM_DIFFUSION_STEPS=16`, cosine schedule, `LEARNING_RATE=5e-4`, `GRAD_ACCUM_STEPS=4`, `ITERATIONS=2000`
- Result summary: lower mask rate continued to help strongly at longer horizons; `MAX_MASK_RATE=0.65` was the best run by a clear margin
- Metrics:
  - `l01_mask065`: `val_loss=3.9445`, `train_loss=4.0670`
  - `l02_mask070`: `val_loss=4.1197`, `train_loss=4.2555`
  - `l03_mask075`: `val_loss=4.2977`, `train_loss=4.4306`
  - `l04_mask070_2shards`: `val_loss=4.1222`, `train_loss=4.2502`
- Artifacts: [`logs/longrun_diffusion_20260407_142138/summary.tsv`](/Users/archit/Local/parameter-golf/logs/longrun_diffusion_20260407_142138/summary.tsv), [`logs/longrun_diffusion_20260407_142138/runner.txt`](/Users/archit/Local/parameter-golf/logs/longrun_diffusion_20260407_142138/runner.txt)
- Findings:
  - the mask-rate ordering was stable over 2000 steps: `0.65` beat `0.70`, which beat `0.75`
  - relative to the best follow-up run (`4.7833`), the longrun suite improved substantially: `0.65` reached `3.9445`
  - `MAX_MASK_RATE=0.70` on `2` shards was effectively tied with `1` shard, which suggests the recipe is not brittle to that small increase in training-data diversity
  - the best run kept improving throughout the full run and did not appear fully saturated by step 2000
  - sample text improved noticeably, though generation is still not clean natural language
- Next step: promote `MAX_MASK_RATE=0.65` to the new baseline and consider one longer confirmation run at that setting before resuming broader architecture searches

### 2026-04-07 - longrun_diffusion_20260407_142138_l01_mask065
- Status: completed
- Script: [`train_diffusion.py`](/Users/archit/Local/parameter-golf/train_diffusion.py)
- Config: [`configs/longrun/l01_mask065.env`](/Users/archit/Local/parameter-golf/configs/longrun/l01_mask065.env)
- Dataset: FineWeb `sp1024`, local overnight suite
- Goal: Longer run around the new best recipe with a slightly easier corruption target.
- Key settings: seq=256 dim=384 layers=6 heads=8 batch_tokens=32768 diff_steps=16 schedule=cosine max_mask=0.65 lr=0.0005 iterations=2000
- Result summary: overnight suite auto-entry
- Metrics: train_loss=`4.0670`, val_loss=`3.9445`, model_bytes=`29120212`
- Artifacts: [`longrun_diffusion_20260407_142138_l01_mask065_diffusion.txt`](/Users/archit/Local/parameter-golf/logs/longrun_diffusion_20260407_142138/longrun_diffusion_20260407_142138_l01_mask065_diffusion.txt), [`longrun_diffusion_20260407_142138_l01_mask065_console.txt`](/Users/archit/Local/parameter-golf/logs/longrun_diffusion_20260407_142138/longrun_diffusion_20260407_142138_l01_mask065_console.txt), [`longrun_diffusion_20260407_142138_l01_mask065_diffusion_mlx.npz`](/Users/archit/Local/parameter-golf/logs/longrun_diffusion_20260407_142138/longrun_diffusion_20260407_142138_l01_mask065_diffusion_mlx.npz)
- Findings: pending morning review
- Next step: compare against the rest of the overnight suite

### 2026-04-07 - longrun_diffusion_20260407_142138_l02_mask070
- Status: completed
- Script: [`train_diffusion.py`](/Users/archit/Local/parameter-golf/train_diffusion.py)
- Config: [`configs/longrun/l02_mask070.env`](/Users/archit/Local/parameter-golf/configs/longrun/l02_mask070.env)
- Dataset: FineWeb `sp1024`, local overnight suite
- Goal: Longer run of the current best recipe.
- Key settings: seq=256 dim=384 layers=6 heads=8 batch_tokens=32768 diff_steps=16 schedule=cosine max_mask=0.7 lr=0.0005 iterations=2000
- Result summary: overnight suite auto-entry
- Metrics: train_loss=`4.2555`, val_loss=`4.1197`, model_bytes=`29120212`
- Artifacts: [`longrun_diffusion_20260407_142138_l02_mask070_diffusion.txt`](/Users/archit/Local/parameter-golf/logs/longrun_diffusion_20260407_142138/longrun_diffusion_20260407_142138_l02_mask070_diffusion.txt), [`longrun_diffusion_20260407_142138_l02_mask070_console.txt`](/Users/archit/Local/parameter-golf/logs/longrun_diffusion_20260407_142138/longrun_diffusion_20260407_142138_l02_mask070_console.txt), [`longrun_diffusion_20260407_142138_l02_mask070_diffusion_mlx.npz`](/Users/archit/Local/parameter-golf/logs/longrun_diffusion_20260407_142138/longrun_diffusion_20260407_142138_l02_mask070_diffusion_mlx.npz)
- Findings: pending morning review
- Next step: compare against the rest of the overnight suite

### 2026-04-07 - longrun_diffusion_20260407_142138_l03_mask075
- Status: completed
- Script: [`train_diffusion.py`](/Users/archit/Local/parameter-golf/train_diffusion.py)
- Config: [`configs/longrun/l03_mask075.env`](/Users/archit/Local/parameter-golf/configs/longrun/l03_mask075.env)
- Dataset: FineWeb `sp1024`, local overnight suite
- Goal: Longer run around the new best recipe with slightly harder corruption.
- Key settings: seq=256 dim=384 layers=6 heads=8 batch_tokens=32768 diff_steps=16 schedule=cosine max_mask=0.75 lr=0.0005 iterations=2000
- Result summary: overnight suite auto-entry
- Metrics: train_loss=`4.4306`, val_loss=`4.2977`, model_bytes=`29120212`
- Artifacts: [`longrun_diffusion_20260407_142138_l03_mask075_diffusion.txt`](/Users/archit/Local/parameter-golf/logs/longrun_diffusion_20260407_142138/longrun_diffusion_20260407_142138_l03_mask075_diffusion.txt), [`longrun_diffusion_20260407_142138_l03_mask075_console.txt`](/Users/archit/Local/parameter-golf/logs/longrun_diffusion_20260407_142138/longrun_diffusion_20260407_142138_l03_mask075_console.txt), [`longrun_diffusion_20260407_142138_l03_mask075_diffusion_mlx.npz`](/Users/archit/Local/parameter-golf/logs/longrun_diffusion_20260407_142138/longrun_diffusion_20260407_142138_l03_mask075_diffusion_mlx.npz)
- Findings: pending morning review
- Next step: compare against the rest of the overnight suite

### 2026-04-07 - longrun_diffusion_20260407_142138_l04_mask070_2shards
- Status: completed
- Script: [`train_diffusion.py`](/Users/archit/Local/parameter-golf/train_diffusion.py)
- Config: [`configs/longrun/l04_mask070_2shards.env`](/Users/archit/Local/parameter-golf/configs/longrun/l04_mask070_2shards.env)
- Dataset: FineWeb `sp1024`, local overnight suite
- Goal: Check whether the new best recipe still holds up when trained across two FineWeb shards instead of one.
- Key settings: seq=256 dim=384 layers=6 heads=8 batch_tokens=32768 diff_steps=16 schedule=cosine max_mask=0.7 lr=0.0005 iterations=2000
- Result summary: overnight suite auto-entry
- Metrics: train_loss=`4.2502`, val_loss=`4.1222`, model_bytes=`29120212`
- Artifacts: [`longrun_diffusion_20260407_142138_l04_mask070_2shards_diffusion.txt`](/Users/archit/Local/parameter-golf/logs/longrun_diffusion_20260407_142138/longrun_diffusion_20260407_142138_l04_mask070_2shards_diffusion.txt), [`longrun_diffusion_20260407_142138_l04_mask070_2shards_console.txt`](/Users/archit/Local/parameter-golf/logs/longrun_diffusion_20260407_142138/longrun_diffusion_20260407_142138_l04_mask070_2shards_console.txt), [`longrun_diffusion_20260407_142138_l04_mask070_2shards_diffusion_mlx.npz`](/Users/archit/Local/parameter-golf/logs/longrun_diffusion_20260407_142138/longrun_diffusion_20260407_142138_l04_mask070_2shards_diffusion_mlx.npz)
- Findings: pending morning review
- Next step: compare against the rest of the overnight suite
