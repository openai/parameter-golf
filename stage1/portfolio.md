# Stage 1 Portfolio

This is the evidence-updated Stage 1 portfolio after reading the public records.

## Selection Rule

Choose the top 5 based on:

- distinct mechanism, not just distinct knob value
- high information gain on unresolved bottlenecks
- enough feasibility to implement now
- broad coverage across training, optimizer, export, and evaluation

## Desired Mix

- 1 conservative architecture baseline family
- 1 training-geometry family
- 1 optimizer family
- 1 export or artifact family
- 1 evaluation family
- 1 wildcard or secondary structural family

## Active Top 5

| Slot | Role | Family | Target region | Why selected now | Expected signal | Acceptance test | Kill condition | Overlap check | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | scout | `M02` long-context training geometry | `Hyperparameters`, loader usage in main loop | public records strongly support longer training context, and this is already being tested on A100 | better final `val_bpb` despite fewer total steps | survives A100 scouting and is competitive after matched retune | slower and worse with no compensating quality | only active geometry family | pending |
| P2 | exploit | `M09` export-aware quantization redesign | quantization and dequantization path | public wins show fp16 embedding passthrough and selective keep-float rules are first-order | smaller post-quant gap or materially better bytes at same score | round-tripped score improves or bytes fall without damage | pre-quant looks good but round-tripped score regresses | unique export family | pending |
| P3 | exploit | `M06` selective optimizer partitioning | optimizer split in main | public top run validates AdamW-for-some, Muon-for-others as a real family | better convergence and/or post-quant robustness without throughput collapse | wins against matched optimizer baselines | extra complexity with no stable gain | unique vs M07 | pending |
| P4 | exploit | `M11` evaluation-time context | `eval_val`, sliding-window path, eval seq-len path | clearly validated by the public leaderboard and missing from the earlier ontology | major score lift with unchanged or minimally changed training | reproducible eval gain that stays within challenge rules | gain depends on a bug or collapses under exact replay | unique evaluation family | pending |
| P5 | wildcard | `M08` wallclock-aware schedule | `lr_mul`, warmdown logic, validation schedule | schedule now looks like a real lever for post-quant quality, not a background retune | better final score through better weight geometry and legal wallclock use | improves round-tripped score with no artifact penalty | only moves pre-quant or hurts training too much | unique vs M02 and M09 | pending |

## Deferred But Still In Final Eleven

- `M01` matched-budget architecture reallocation
  - reason: still important, but the public frontier already says geometry/export/eval deserve priority first
  - promote if the live A100 runs show room beyond context length alone

- `M04` nonuniform stack roles
  - reason: strong idea, but more invasive than M03 and less clean as a first structural scout
  - promote if the winning stack seems to benefit from depth-role specialization

- `M07` adaptive Muon compute
  - reason: still likely real, but public evidence made partitioning stronger than pure Muon scheduling
  - promote if optimizer partition helps but Muon cost still looks mistimed

- `M03` parameter reuse
  - reason: public evidence weakens it for the first 10-minute wave
  - promote if longer-context and export surfaces saturate

- `M10` tokenizer/data frontier
  - reason: high upside, higher setup cost
  - promote after at least one export and one eval read

- `M05` control-tensor simplification
  - reason: still plausible, but the public frontier currently adds smarter controls rather than removing them
  - promote if export remains tight after the stronger public tricks are adopted

## Elimination Rule For This Loop

- Do not run two slots that only differ by one local retune in the same region.
- Do not let all active slots collapse back into `Hyperparameters`.
- If one family wins only pre-quant but loses after export, it does not count as a survivor.
- A100 is for family ranking, not for final claims.
