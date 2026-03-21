# Parameter Golf Research Handoff

Date: 2026-03-20

## Objective

Optimize the challenge's real target:

- best post-quantized `val_bpb`
- under a hard fixed-time training budget
- under a hard artifact-byte cap

This means the useful currencies are:

- quality gained per second
- quality gained per stored byte
- quality retained after quantization/export

The local MLX loop is a filter, not a perfect H100 simulator, but it has been good enough to kill bad ideas and identify one very strong branch.

## Current Best Local Branch

Backbone:

- `DEPTH_SHARE_MODE=cycle`
- `DEPTH_UNIQUE_LAYERS=3`
- `DEPTH_SHARE_HEAVY_ONLY=1`

Confirmed stack on top:

- `MUON_WEIGHT_DECAY=0.01`
- `SMEARGATE=1`
- `SMEARGATE_INIT=-3.0`
- `MOD_KEEP=0.75`
- `MOD_CORE=1`

Interpretation:

- share the expensive attention/MLP cores across depth
- keep cheap per-layer controls unique
- add a tiny embedding-level previous-token blend
- route MLP compute on the middle shared core only, keeping roughly `75%` of tokens active there

## Best Numbers

### Baseline controls

- Baseline `180s`, seed `1337`: `2.16834965`
  - log: `logs/stage6_seq_base_180s_s1337.txt`
  - compressed model: `8,558,684` bytes
- Baseline `180s`, seed `2026`: `2.18171034`
  - log: `logs/stage6_seq_base_180s_s2026.txt`
  - compressed model: `8,535,407` bytes

### Heavy-share backbone

- `heavycycle3`, seed `1337`: `2.13989464`
  - log: `logs/stage8_heavycycle3_180s_s1337.txt`
  - compressed model: `3,390,661` bytes
- `heavycycle3`, seed `2026`: `2.13784642`
  - log: `logs/stage8_heavycycle3_180s_s2026.txt`
  - compressed model: `3,400,406` bytes

### Add Muon weight decay

- `heavycycle3 + wd=0.01`, seed `1337`: `2.13540884`
  - log: `logs/stage10_heavycycle3_wd001_180s_s1337.txt`
  - compressed model: `3,406,265` bytes

### Add SmearGate-lite

- `heavycycle3 + wd + smear(-3.0)`, seed `1337`: `2.12786663`
  - log: `logs/stage13_heavycycle3_wd001_smearm3_180s_s1337.txt`
  - compressed model: `3,386,835` bytes
- `heavycycle3 + wd + smear(-2.0)`, seed `1337`: `2.14924756`
  - log: `logs/stage13_heavycycle3_wd001_smearm2_180s_s1337.txt`
  - conclusion: gate must stay conservative

### Add MoD-lite

- `heavycycle3 + wd + smear(-3.0) + mod(keep=0.75, core=1)`, seed `1337`: `2.02522139`
  - log: `logs/stage16_heavycycle3_wd001_smearm3_mod075_180s_s1337.txt`
  - compressed model: `3,658,495` bytes
- same config, seed `2026`: `2.02095004`
  - log: `logs/stage16_heavycycle3_wd001_smearm3_mod075_180s_s2026.txt`
  - compressed model: `3,665,283` bytes
- `keep=0.875`, seed `1337`: `2.02301388`
  - log: `logs/stage16_heavycycle3_wd001_smearm3_mod0875_180s_s1337.txt`
  - compressed model: `3,648,012` bytes

Current best local result:

- `2.02095004` at `180s` local matched time
- branch: `heavycycle3 + wd + smear(-3.0) + mod(0.75, core=1)`

## Important Ablations

### MoD-lite placement

- `core=1` is best
- `core=2`: `2.03106751`
  - log: `logs/stage16_heavycycle3_wd001_smearm3_mod075_core2_180s_s1337.txt`
- `core=0`: `2.07331353`
  - log: `logs/stage16_heavycycle3_wd001_smearm3_mod075_core0_180s_s1337.txt`
  - also much slower

Conclusion:

- the gain is not generic "route any layer"
- the middle shared core is the right place to gate MLP compute

### KV schedule

- uniform `KV_HEAD_SCHEDULE=2`: `2.14991956`
  - log: `logs/stage14_heavycycle3_wd001_smearm3_kv2_180s_s1337.txt`
- non-uniform `KV_HEAD_SCHEDULE=4,2,4`: `2.13113807`
  - log: `logs/stage14_heavycycle3_wd001_smearm3_kv424_180s_s1337.txt`

Conclusion:

- not worth carrying forward vs the current best branch

## Eval / Compression Findings

### Sliding-window eval

On the earlier `heavycycle3` checkpoint:

- flat `1024`: `2.13989464`
  - log: `logs/eval_stage8_heavycycle3_flat1024_s1337.txt`
- sliding `1024/256`: `2.13631236`
  - log: `logs/eval_stage8_heavycycle3_slide256_1024_s1337.txt`

Conclusion:

- sliding eval is likely real and should be carried forward
- it has not yet been cleanly restacked on the newest MoD-lite branch because long local eval tails were awkward on the Mac

### PTQ mixed low-bit

On the `smear(-3.0)` checkpoint:

- int8 control: `2.12786663`, `3,386,835` bytes
  - log: `logs/eval_stage15_smearm3_int8_s1337.txt`
- mixed `MLP 6-bit / attention+embed 8-bit`: `2.17621863`, `2,695,599` bytes
  - log: `logs/eval_stage15_smearm3_m6mix_default_s1337.txt`
- same mixed recipe with `INT8_CLIP_PERCENTILE=99.99`: `2.17631718`, `2,695,511` bytes
  - log: `logs/eval_stage15_smearm3_m6mix_clip9999_s1337.txt`

Conclusion:

- PTQ-only mixed low-bit is not good enough
- if compression is revisited, it should be training-aware: QAT-lite, quant-noise, or outlier smoothing on GPU

## Dead / Deprioritized Ideas

These were tested locally and are not worth more Mac time right now:

- full AttnRes and most selective AttnRes variants
- naive latent memory and recurrent memory
- naive shared-depth recurrence
- naive width scaling
- sequence curriculum as a main lever
- MLP FiLM
- MLP LoRA / low-rank delta
- MTP
- `MLP 3x`
- aggressive KV reduction as a main idea
- PTQ-only mixed low-bit export

Representative bad numbers:

- MTP `0.25`: `2.20793677`
  - log: `logs/stage12_heavycycle3_wd001_mtp2w025_180s_s1337.txt`
- MTP `0.5`: `2.30722590`
  - log: `logs/stage12_heavycycle3_wd001_mtp2w05_180s_s1337.txt`
- `MLP 3x` on `heavycycle3 + wd`: `2.14172339`
  - log: `logs/stage11_heavycycle3_wd001_mlp3_180s_s1337.txt`

## Current Thesis

The project is no longer in the "broad ideation" phase.

The current working thesis is:

- fast shared-heavy backbone
- cheap per-layer controls stay unique
- tiny embedding-side inductive bias helps
- conditional compute helps a lot if placed carefully
- export should stay simple unless training-aware compression proves itself

The current branch is not "just heavier tuning." It is a different architecture family:

- shared-heavy / unique-light depth tying
- cheap local bigram-like bias via SmearGate
- compute reallocation via MoD-lite

## Recommended Next Steps

### Local, if any

Only two local tasks still look worth doing:

1. clean sliding-window eval on the current best MoD-lite checkpoint
2. maybe one extra-depth reinvestment test with the same 3 shared cores, only if time remains

Everything else is now lower value than CUDA/H100 porting.

### Immediate next engineering step

Port the current winning branch into `train_gpt.py`:

- heavy shared cores
- unique-light per-layer wrappers
- Muon weight decay
- SmearGate-lite
- MoD-lite (`mod_core=1`, `mod_keep≈0.75`)

### H100-side priorities

Once the port exists, the next branches should be:

1. confirm the branch under real CUDA wallclock
2. retest sliding-window eval on the best CUDA checkpoint
3. explore training-aware compression, not more PTQ calibration
4. consider FP8 / runtime systems improvements only after the branch is stable

## Suggested H100 Experiment Order

1. PyTorch baseline reproduction
2. heavycycle3 reproduction
3. heavycycle3 + wd
4. heavycycle3 + wd + smear
5. heavycycle3 + wd + smear + MoD-lite
6. sliding-window eval on the best checkpoint
7. QAT-lite / compression-aware branch

## Bottom Line

The local winner is:

- `heavycycle3 + MUON_WEIGHT_DECAY=0.01 + SMEARGATE(init=-3.0) + MoD-lite(keep=0.75, core=1)`

This branch is replicated across two seeds and has a large margin over every earlier local branch. It is now strong enough that the main bottleneck is no longer more local idea search; it is CUDA/H100 validation and making low-bit export work without giving back the gains.
