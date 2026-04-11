# H100 Frontier Result Summary — 2026-03-26

## Objective
Validate whether the recent H100 stabilization changes convert previously failing frontier runs into legal, end-to-end training/eval/export results, and identify the highest-value current direction under tight RunPod budget constraints.

## Stabilization Changes Used
Commit:
`97732ceef462ee7a4e64b3a1d9f16a1e696aed2f`

Patch summary:
- Preserved FlashAttention eligibility by forcing aligned attention dtypes in `flash_attn_interface.py`
- Disabled `torch.compile(...)` paths in:
  - `train_gpt_frontier_control.py`
  - `train_gpt_frontier_gptq.py`

Reason:
- Prior failures were caused by two issues:
  1. compile-time / inductor memory blowups
  2. mixed attention dtypes (`q/k=float32`, `v=bfloat16`) forcing fallback to the memory-heavy math attention path

After the fix:
- runtime used `attention_backend:flash_attn`
- attention dtype stayed `torch.bfloat16`
- H100 runs completed legally

## Best Result
Run name:
`h100_ppm_half_run_directed`

Preset:
`sota_plus_ppm_multiorder`

Scale:
`half_run`

Status:
`legal`

Best final metric:
`legal_ttt_exact val_bpb: 2.65555886`

Other final metrics:
- `final_int6_sliding_window_exact val_bpb: 2.69223778`
- `final_int6_roundtrip_exact val_bpb: 3.53425201`

Artifact / size:
- exported bytes: `4677472`
- code bytes: `117470`
- total artifact bytes: `4794942`
- remaining headroom to 16 MB: `11205058`

Training summary:
- wallclock-capped run
- stopped at `step 115`
- `train_time: 300392ms`
- peak memory allocated: `40153 MiB`
- peak memory reserved: `40986 MiB`

Interpretation:
- PPM remains the strongest validated direction so far
- legal score-first TTT provides meaningful additional gain over sliding-window eval
- the half-run materially improved on smoke performance and justified the H100 validation pass

## Comparison to Other Validated Runs
### PPM smoke
Run:
`h100_smoke_ppm_dtypefix_retry`

Best:
`legal_ttt_exact val_bpb: 2.89186739`

### RotaryFix smoke
Run:
`h100_smoke_rotaryfix_dtypefix_retry`

Best:
`legal_ttt_exact val_bpb: 3.84790756`

### Improvement
Compared with legal PPM smoke:
- `2.89186739 -> 2.65555886`
- absolute gain: `0.23630853 bpb`

Compared with legal RotaryFix smoke:
- PPM half-run is decisively better

## What Was Learned
1. The H100 issue was not lack of VRAM alone; it was primarily a runtime-path problem.
2. The decisive fix was:
   - avoiding compile-heavy paths
   - keeping attention on FlashAttention bf16 instead of falling back to math attention
3. Once stabilized, the H100 produced a strong legal half-run result quickly enough to be worth the spend.
4. Under current evidence, PPM is the lead direction.

## Recommended Next Action
Current best candidate to preserve and build from:
- `h100_ppm_half_run_directed`

Recommended immediate stance:
- do not spend more by default
- only resume paid runs for one of these targeted reasons:
  1. matched H100 `half_run` for RotaryFix as a fair A/B against PPM
  2. one longer directed PPM run if budget allows and a further gain seems worth chasing

## Repo-State Note
The H100 stabilization diff from base commit `3f7d478` to `97732ce` should be preserved alongside this note because it was necessary for all successful H100 frontier runs.

Key files changed:
- `flash_attn_interface.py`
- `train_gpt_frontier_control.py`
- `train_gpt_frontier_gptq.py`
