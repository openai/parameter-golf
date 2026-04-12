# `seeds_run/` — §3.1 multi-seed verification artifacts

This folder contains the small text-only artifacts of the §3.1 multi-seed verification reported in the v3.5 README. It is meant to make the §3.1 numbers reviewer-checkable without requiring any external download.

## What is in this folder

- `run_p5.sh` — the orchestration script that ran the 5 fresh shared-model trainings (`11L_w0.3` × seeds `{1337, 42, 2024, 7, 100}`) and the 5 corresponding CF evaluations
- `run_p5.out` — wrapper stdout from the run_p5 invocation, including the per-run training final `val_bpb` and per-run CF Total summary lines
- `run_phase_b.sh` — the orchestration script that ran the 1 fresh causal-only control training (`11L_w0` `SEED=1337`) and its CF evaluation
- `run_phase_b.out` — wrapper stdout from run_phase_b
- `logs/11L_w03_s{1337,42,2024,7,100}_train.log` — full training stdout for each shared seed (loss curves, val_bpb every 500 steps, final val_bpb, throughput, FLOPs)
- `logs/11L_w0_s1337_train.log` — full training stdout for the fresh control
- `eval/11L_w03_s{1337,42,2024,7,100}_cf.log` — CF evaluation results for each shared seed (JSON: `pure_ar_bpb`, `cf_ar_part`, `cf_cdm_part`, `cf_total`, `cf_vs_ar_pct`)
- `eval/11L_w0_s1337_cf.log` — CF evaluation result for the fresh control
- `eval/*_eval.out` — wrapper stdout from each CF evaluation invocation

Every number quoted in §3.1 of the v3.5 README is grep-able from these logs.

## What is NOT in this folder (and why)

The `.npz` final-state model files (~114 MB each) and the `step_final.pt` torch checkpoints (~109 MB each) for all 6 runs are NOT included in this PR folder. Six of each totals ~1.3 GB and exceeds what is reasonable to commit to a parameter-golf PR. They are stored locally in the author's `meadow-golf` research diary repo at `experiments/2026-04-09_matched_ablation/p5_results/{npz,ckpt}/` and are byte-identical (verified after download from the H100 SXM training pod) to what was used to produce the CF evaluations in `eval/`.

If a reviewer wants to independently re-run the CF evaluation on the exact `.npz` / `.pt` files used in §3.1, contact the author (akai@fawstudio.com) for a Hugging Face dataset upload — this is a small request to fulfill on-demand and avoids bloating the PR folder with model weights that almost no reviewer will actually want to download.

## How to reproduce §3.1 from scratch (no checkpoints needed)

Both `run_p5.sh` and `run_phase_b.sh` are verbatim pod-side orchestration scripts. They expect the cloned experiment checkout at `/workspace/meadow-golf/experiments/2026-04-09_matched_ablation`, and they rely on the v3.5 copies of `train_cdm.py`, `train_ablation_runner.py`, and `eval_cf_ablation.py` in that directory. On a 1×H100 SXM pod:

```bash
# 1. Clone meadow-golf and overlay the v3.5 train_cdm.py / train_ablation_runner.py /
#    eval_cf_ablation.py (which include the final-checkpoint save fix, seed-patched runner,
#    and the .npz loader, respectively).
git clone https://github.com/akaiHuang/meadow-golf
cd meadow-golf/experiments/2026-04-09_matched_ablation

# 2. Download the v4096 dataset (~20 GB, ~1 min on H100)
hf download akaiii/meadow-golf-v4096 --repo-type dataset --local-dir /workspace/gv4096

# 3. Run the 5 shared seeds
SCRIPT_DIR=. DATA_DIR=/workspace/gv4096/data \
  TOKENIZER=/workspace/gv4096/bpe_v4096.model \
  OUT_DIR=/workspace/out CKPT_DIR=/workspace/ckpt LOG_DIR=/workspace/logs \
  bash run_p5.sh

# 4. Run the 1 control seed
bash run_phase_b.sh
```

Total wall time on a single 1×H100 SXM pod: ~70 min (5 shared trainings ~50 min + 1 control training ~9 min + 6 CF evals ~10 min). Total self-funded compute cost: ~$3.50 at $2.99/hr.
