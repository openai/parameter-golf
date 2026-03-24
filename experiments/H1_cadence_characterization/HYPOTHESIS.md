# H1: Cadence Characterization on 4x2 (RC-0)

## Question
What is cadence doing to BPB in a balanced 4f+2cx2 recursive system?

## Prediction
Cadence 2 (C/N alternating) is near-optimal because:
- Cadence 1 (all C): doubles compute but ref never gets N-step outbound gradient.
  The PD channel is always in "write" mode, never "read." Expect worse BPB per wall-second.
- Cadence 2: balanced read/write on the PD channel. N-steps let the ref's gradient
  propagate back through the crawler without competing with the C-step consensus update.
- Cadence 3-4: starves the ref of C-step updates. The deliberation mechanism goes dormant.
  Expect delib_scale to plateau or decay.

We expect a U-shaped curve: BPB worst at cadence 1 (compute waste) and cadence 4
(PD starvation), best at cadence 2 or 3.

## Architecture (held constant)
```
NUM_FLAT_LAYERS=4  NUM_CRAWLER_LAYERS=2  CRAWLER_LOOPS=2
MODEL_DIM=640  NUM_HEADS=10  NUM_KV_HEADS=5  MLP_MULT=4
XSA_LAST_N=2  VE_LAYERS=0,1
```

## Arms

| Arm | DIAG_FIXED_CADENCE | C-step ratio | Parent |
|-----|-------------------|--------------|--------|
| cad1 | 1 | 100% (all C) | RC-0 |
| cad2 | 2 | 50% (C/N) | RC-0 (control) |
| cad3 | 3 | 33% (C/N/N) | RC-0 |
| cad4 | 4 | 25% (C/N/N/N) | RC-0 |

## Scale
0.25 (150s wallclock, 625 warmdown, TTT/distill OFF)

## Diagnostic Focus
1. `delib_scale` trajectory — does PD stay alive across cadences?
2. `fast_val_bpb` at wall-clock matched checkpoints
3. `train_loss` split by `is_crawl` — are C-steps helping or hurting?
4. Total steps achieved (cadence 1 will get fewer)
5. `quant_gap` — does cadence affect quantization friendliness?

## Verdict
_To be filled after runs complete._

| Arm | Steps | fast_val_bpb | sliding_bpb | post_ema_bpb | quant_gap | delib_scale_final | Verdict |
|-----|-------|-------------|-------------|-------------|-----------|-------------------|---------|
| cad1 | | | | | | | |
| cad2 | | | | | | | |
| cad3 | | | | | | | |
| cad4 | | | | | | | |
