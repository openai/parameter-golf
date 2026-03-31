# RASCAL Single-H100 Ablation Matrix (Seed 444, 1200 iters)

Date: 2026-03-31
Source run: `experiments/SOTA/2026-03-30_JUNKYARD_RAT_RASCAL_II_nogptq/logs/ablate_single_h100_20260330_234105`
Source CSV: `/workspace/parameter-golf-lab/experiments/SOTA/2026-03-30_JUNKYARD_RAT_RASCAL_II_nogptq/logs/ablate_single_h100_20260330_234105/summary.csv`

## Run context
- torch=`2.11.0+cu130`
- cuda=`13.0`
- visible_gpus=`1`
- nproc_per_node=`1`
- seed=`444`
- iterations=`1200`
- train_batch_tokens=`786432`
- train_seq_len=`2048`
- max_wallclock_seconds=`0`
- data_path=`/workspace/parameter-golf-lab/data/datasets/fineweb10B_sp1024`
- tokenizer_path=`/workspace/parameter-golf-lab/data/tokenizers/fineweb_1024_bpe.model`

## Dynamic ablations

| case | step_avg_ms | post_ema_bpb | delta_step_vs_base_ms | delta_bpb_vs_base | int6_zstd_bytes | notes |
|---|---:|---:|---:|---:|---:|---|
| baseline | 793.06 | 1.3284 | +0.00 | +0.0000 | 12,820,858 | control |
| loader_cache2 | 793.52 | 1.3284 | +0.46 | +0.0000 | 12,818,568 | no benefit on 1-shard setup |
| loader_cache4 | 792.22 | 1.3282 | -0.84 | -0.0002 | 12,821,443 | slight speed + tiny quality gain |
| muon_ns4 | 792.15 | 1.3284 | -0.91 | +0.0000 | 12,815,073 | best speed, no quality loss |
| muon_ns3 | 792.55 | 1.3406 | -0.51 | +0.0122 | 11,278,035 | quality regression, reject |
| compile_off | 1918.79 | 1.3287 | +1125.73 | +0.0003 | 12,821,923 | massive slowdown, reject |

## Static modeled items (no training run)
- `comm_padding_model`: modeled only.
- `replicated_allreduce_model`: modeled only.

## Practical takeaway
- Keep compile enabled.
- `MUON_BACKEND_STEPS=4` is the cleanest speed win with quality parity in this run.
- `COPRIME_MAX_LOADED_SHARDS=4` gives a smaller but positive signal on this single-shard local setup.
- Do not use `MUON_BACKEND_STEPS=3` (quality drop).

## Suggested next validation
- 3-seed confirm on single H100: baseline vs `muon_ns4` vs `loader_cache4+muon_ns4`.
- If stable, promote to race runner and verify under 600s cap.
