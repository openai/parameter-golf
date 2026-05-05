# BIJEPAX-lite 3-seed candidate

Config matches successful seed 42 run:
- script: our_submission/train_gpt_v15_bijepax.py
- DISABLE_COMPILE=1
- CASEOPS_ENABLED=1
- PPM_MIXER_ENABLED=1 order=5 H=0.999 L=0.18 T=0.80
- TTT_ENABLED=0
- LQER_TOP_K=1
- BIJEPAX_ENABLED=1 weight=0.01 start=0.35 end=0.80 fwd_hops=4 bwd_hops=4 cycle=0 head_dim=32 stride=64 lr=0.001

Existing seed 42:
- run: v15_bijepaxlite_lqer1_nocompile_s42_20260501_022405
- final ppm_sliding val_bpb: 0.97234287
- artifact bytes: 15997180
- eval time: 502131ms
- rc: 0

Queued seeds: 314, 999


## v15_bijepaxlite_lqer1_nocompile_s314_20260501_025209
- started: 2026-05-01T02:52:09Z
- log: /workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s314_20260501_025209/train.log
- finished: 2026-05-01T03:14:07Z
- rc: 0
- scores: diagnostic pre-quantization post-ema val_loss:2.42899363 val_bpb:1.10988323 eval_time:9910ms;Total submission size quantized+pergroup: 15999539 bytes;diagnostic quantized val_loss:2.44155528 val_bpb:1.11562304 eval_time:9926ms;ppm_mixer val_bpb:0.97206308 eval_time:453715ms order=5 H=0.999 L=0.18 T=0.8 N_tokens=47851520 N_sidecar_bytes=151074499;ppm_sliding val_loss:2.45044876 val_bpb:0.97206308 eval_time:499038ms;

## v15_bijepaxlite_lqer1_nocompile_s999_20260501_031407
- started: 2026-05-01T03:14:07Z
- log: /workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s999_20260501_031407/train.log
- finished: 2026-05-01T03:36:01Z
- rc: 0
- scores: diagnostic pre-quantization post-ema val_loss:2.43314506 val_bpb:1.11178015 eval_time:9911ms;Total submission size quantized+pergroup: 15997593 bytes;diagnostic quantized val_loss:2.44582432 val_bpb:1.11757370 eval_time:11393ms;ppm_mixer val_bpb:0.97373767 eval_time:451054ms order=5 H=0.999 L=0.18 T=0.8 N_tokens=47851520 N_sidecar_bytes=151074499;ppm_sliding val_loss:2.45502055 val_bpb:0.97373767 eval_time:496384ms;

## Final scrape
/workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s314_20260501_025209/train.log:  artifact_dir: /workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s314_20260501_025209
/workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s314_20260501_025209/train.log:  logfile: /workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s314_20260501_025209/v15_bijepaxlite_lqer1_nocompile_s314_20260501_025209.txt
/workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s314_20260501_025209/train.log:  model_path: /workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s314_20260501_025209/final_model.pt
/workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s314_20260501_025209/train.log:  quantized_model_path: /workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s314_20260501_025209/final_model.int6.ptz
/workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s314_20260501_025209/train.log:  run_id: v15_bijepaxlite_lqer1_nocompile_s314_20260501_025209
/workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s314_20260501_025209/train.log:Total submission size quantized+pergroup: 15999539 bytes
/workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s314_20260501_025209/train.log:diagnostic quantized val_loss:2.44155528 val_bpb:1.11562304 eval_time:9926ms
/workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s314_20260501_025209/train.log:ppm_mixer val_bpb:0.97206308 eval_time:453715ms order=5 H=0.999 L=0.18 T=0.8 N_tokens=47851520 N_sidecar_bytes=151074499
/workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s314_20260501_025209/train.log:ppm_sliding val_loss:2.45044876 val_bpb:0.97206308 eval_time:499038ms
/workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s42_20260501_022405/train.log:  artifact_dir: /workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s42_20260501_022405
/workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s42_20260501_022405/train.log:  logfile: /workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s42_20260501_022405/v15_bijepaxlite_lqer1_nocompile_s42_20260501_022405.txt
/workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s42_20260501_022405/train.log:  model_path: /workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s42_20260501_022405/final_model.pt
/workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s42_20260501_022405/train.log:  quantized_model_path: /workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s42_20260501_022405/final_model.int6.ptz
/workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s42_20260501_022405/train.log:  run_id: v15_bijepaxlite_lqer1_nocompile_s42_20260501_022405
/workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s42_20260501_022405/train.log:Total submission size quantized+pergroup: 15997180 bytes
/workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s42_20260501_022405/train.log:diagnostic quantized val_loss:2.44116551 val_bpb:1.11544494 eval_time:10342ms
/workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s42_20260501_022405/train.log:ppm_mixer val_bpb:0.97234287 eval_time:456845ms order=5 H=0.999 L=0.18 T=0.8 N_tokens=47851520 N_sidecar_bytes=151074499
/workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s42_20260501_022405/train.log:ppm_sliding val_loss:2.45118426 val_bpb:0.97234287 eval_time:502131ms
/workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s999_20260501_031407/train.log:  artifact_dir: /workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s999_20260501_031407
/workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s999_20260501_031407/train.log:  logfile: /workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s999_20260501_031407/v15_bijepaxlite_lqer1_nocompile_s999_20260501_031407.txt
/workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s999_20260501_031407/train.log:  model_path: /workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s999_20260501_031407/final_model.pt
/workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s999_20260501_031407/train.log:  quantized_model_path: /workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s999_20260501_031407/final_model.int6.ptz
/workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s999_20260501_031407/train.log:  run_id: v15_bijepaxlite_lqer1_nocompile_s999_20260501_031407
/workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s999_20260501_031407/train.log:Total submission size quantized+pergroup: 15997593 bytes
/workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s999_20260501_031407/train.log:diagnostic quantized val_loss:2.44582432 val_bpb:1.11757370 eval_time:11393ms
/workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s999_20260501_031407/train.log:ppm_mixer val_bpb:0.97373767 eval_time:451054ms order=5 H=0.999 L=0.18 T=0.8 N_tokens=47851520 N_sidecar_bytes=151074499
/workspace/parameter-golf/our_submission/1000/runs/v15_bijepaxlite_lqer1_nocompile_s999_20260501_031407/train.log:ppm_sliding val_loss:2.45502055 val_bpb:0.97373767 eval_time:496384ms
