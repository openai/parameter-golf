# Champion And Ablation Log Summary

Source scope:
- `logs/champion*.txt`
- `logs/ablation_*iter1500.txt`

## Baseline

- forked from https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon

## Strategy Legend

- `champion_*`: earlier baseline or champion-style runs.
- `ablation_*`: targeted follow-up experiments that isolate one idea at a time.
- `style_disabled`: no style pathway.
- `style_add`: inject the style summary additively into the hidden state.
- `style_scale`: use the style summary to multiplicatively scale activations.
- `style_film`: FiLM-style modulation, meaning the style path can learn both scale and shift.
- `w32` or `w128`: style summary window length.
- `d0`, `d32`, `d64`: style bottleneck width. `d0` means no bottleneck.
- `b32`, `b128`: older naming for a style bottleneck width.
- `gated`: style modulation is passed through a learned gate.
- `v2_dual_gate`: newer style pathway with dual summaries, short-window context, and sample gating enabled.
- `v2_dual_nogate`: same as `v2_dual_gate`, but the sample gate is disabled.
- `v2_mean_gate`: same newer style pathway, but with mean summary instead of dual summary.
- `freq_mid`: adds an auxiliary frequency-domain loss focused on the middle-frequency band.
- `localmem`: adds local attention in early layers plus a depth-memory pathway.
- `orig_train10m`, `style_train10m`, `style_small_train10m`: shorter 500-step runs used as earlier baselines.

## What Looks Helpful

- At `iter=1500, seed=1337`, the best finished run is `freq_mid` with `final val_bpb = 1.2390`, followed very closely by `v2_dual_gate` at `1.2407`. Both are clearly better than the older style variants, which cluster around `1.256` to `1.262`.
- Within the new `v2` family, gating helps. `v2_dual_gate` beats `v2_dual_nogate` by `0.0031 val_bpb` at the same iteration count and nearly the same model size.
- Among the older style variants, `film w128 d0` is the strongest finished plain style setup. Adding bottlenecks like `d32`, `d64`, or `b128` usually hurts a bit.
- Compared with `style_disabled`, the old style variants help only modestly at `1500` steps. The newer `v2` style design helps much more.
- At `iter=3500, seed=42`, `style_film_w128_d0` is slightly better than `style_disabled`, but the gap is very small. That suggests style still helps, but the long-run advantage is subtle rather than dramatic.
- `localmem` and `v2_mean_gate` are still incomplete, so they should not be used for final conclusions yet.

Note:
- `final_val_*` comes from the last logged `step:X/X val_loss ... val_bpb ...` line.
- `int6_rt_*` comes from `final_int6_roundtrip_exact`.
- `submission_bytes` is total code plus serialized model size from the log.
- Rows are grouped by `iter` and `seed`.
- Within each table, completed runs are sorted by `final_val_bpb` ascending. Incomplete runs are listed afterward.

## Iter 1500, Seed 1337

| strategy | status | model_params | final_val_loss | final_val_bpb | int6_rt_loss | int6_rt_bpb | code_bytes | submission_bytes | notes | log |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| freq_mid | complete | 27518045 | 2.0920 | 1.2390 | 2.18796546 | 1.29583639 | 100360 | 12035256 | freq-loss mid band | [log](logs/ablation_freq_mid_seed1337_iter1500.txt) |
| v2_dual_gate | complete | 28043358 | 2.0949 | 1.2407 | 2.18774453 | 1.29570554 | 98950 | 12275470 | summary=dual, sample_gate=True | [log](logs/ablation_v2_dual_gate_seed1337_iter1500.txt) |
| v2_dual_nogate | complete | 28042333 | 2.1002 | 1.2438 | 2.20260569 | 1.30450716 | 98950 | 12190442 | summary=dual, sample_gate=False | [log](logs/ablation_v2_dual_nogate_seed1337_iter1500.txt) |
| style_film_w128_d0 | complete | 27518045 | 2.1215 | 1.2565 | 2.50351052 | 1.48271995 | 95667 | 9520143 | baseline + naive style | [log](logs/champion_style_film_w128_d0.txt) |
| baseline | complete | 26993756 | 2.1260 | 1.2592 | 2.56619677 | 1.51984628 | 95667 | 9159163 | no style path | [log](logs/champion_style_disabled.txt) |
| style_scale_w128_d32 | complete | 27026525 | 2.1264 | 1.2594 | 2.53749284 | 1.50284619 | 95667 | 9195831 | scale mode, dim 32 | [log](logs/champion_style_scale_w128_d32.txt) |
| style_film_w128_d64 | complete | 27092061 | 2.1266 | 1.2595 | 2.55107364 | 1.51088950 | 95667 | 9314699 | film mode, dim 64 | [log](logs/champion_style_film_w128_d64.txt) |
| style_scale_w128_d0 | complete | 27255901 | 2.1272 | 1.2599 | 2.51152725 | 1.48746790 | 95667 | 9336415 | scale mode, dim 0 | [log](logs/champion_style_scale_w128_d0.txt) |
| style_add_w128_b0 | complete | 27255901 | 2.1275 | 1.2600 | 2.58229374 | 1.52937981 | 95667 | 9319115 | add mode | [log](logs/champion_style_add_w128_b0.txt) |
| style_scale_w128_d64 | complete | 27059293 | 2.1285 | 1.2606 | 2.57704877 | 1.52627344 | 95667 | 9264431 | scale mode, dim 64 | [log](logs/champion_style_scale_w128_d64.txt) |
| style_film_w128_d32 | complete | 27042909 | 2.1288 | 1.2608 | 2.54873898 | 1.50950679 | 95667 | 9224547 | film mode, dim 32 | [log](logs/champion_style_film_w128_d32.txt) |
| style_add_w128_b128 | complete | 27124829 | 2.1315 | 1.2624 | 2.50705928 | 1.48482172 | 95667 | 9358403 | add mode, bottleneck 128 | [log](logs/champion_style_add_w128_b128.txt) |
| localmem | incomplete | 27583582 | - | - | - | - | - | - | step 350/1500, local attention + depth memory | [log](logs/ablation_localmem_seed1337_iter1500.txt) |
| v2_mean_gate | incomplete | 27518558 | - | - | - | - | - | - | startup only, summary=mean | [log](logs/ablation_v2_mean_gate_seed1337_iter1500.txt) |
| style_add_w128_b32 | incomplete | 27255901 | - | - | - | - | - | - | step 10/1500 | [log](logs/champion_style_add_w128_b32.txt) |
| style_scale_w32_d32 | incomplete | 27026525 | - | - | - | - | - | - | step 50/1500 | [log](logs/champion_style_scale_w32_d32.txt) |

## Iter 3500, Seed 42

| strategy | status | model_params | final_val_loss | final_val_bpb | int6_rt_loss | int6_rt_bpb | code_bytes | submission_bytes | notes | log |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| style_film_w128_d0 | complete | 27518045 | 1.9780 | 1.1715 | 1.99727780 | 1.18290041 | 95667 | 15362523 | film mode, long run | [log](logs/champion_style_film_w128_d0_seed42_iter3500.txt) |
| style_disabled | complete | 26993756 | 1.9802 | 1.1728 | 2.00045933 | 1.18478470 | 95667 | 14820859 | no style, long run | [log](logs/champion_style_disabled_seed42_iter3500.txt) |

## Iter 500, Seed 1337

| strategy | status | model_params | final_val_loss | final_val_bpb | int6_rt_loss | int6_rt_bpb | code_bytes | submission_bytes | notes | log |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| style_w128_gated | complete | 27255901 | 2.7544 | 1.6313 | 12.10147742 | 7.16717657 | 92731 | 5398479 | gated style, window 128 | [log](logs/champion_style_w128_gated.txt) |
| style_w32_gated | complete | 27255901 | 2.7567 | 1.6326 | 12.09212407 | 7.16163699 | 92731 | 5397127 | gated style, window 32 | [log](logs/champion_style_w32_gated.txt) |
| style_w128_gated_b32 | complete | 27026525 | 2.7572 | 1.6330 | 12.29871753 | 7.28399328 | 93789 | 5294889 | gated style, dim 32 | [log](logs/champion_style_w128_gated_b32.txt) |
| style_train10m | complete | 27255900 | 2.7630 | 1.6364 | 8.31215293 | 4.92292516 | 91635 | 5387819 | style baseline, short train | [log](logs/champion_style_train10m.txt) |
| orig_train10m | complete | 26993756 | 2.7733 | 1.6425 | 12.65652219 | 7.49590535 | 91610 | 5234074 | no style, short train | [log](logs/champion_orig_train10m.txt) |
| style_small_train10m | complete | 24040989 | 2.7878 | 1.6511 | 11.83664680 | 7.01032896 | 93789 | 4922097 | smaller model, short train | [log](logs/champion_style_small_train10m.txt) |



## Iter 3500, Seed 2025

| strategy | status | model_params | final_val_loss | final_val_bpb | int6_rt_loss | int6_rt_bpb | code_bytes | submission_bytes | notes | log |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| style_film_w128_d0 | incomplete | 27518045 | - | - | - | - | - | - | step 600/3500, film mode, long run | [log](logs/champion_style_film_w128_d0_seed2025_iter3500.txt) |
