# Record candidate: PR #1855 base + activation-aware GPTQ mixed precision (step-matched)

**Matched-step 3-seed mean val_bpb: 1.06081076** (std 0.00089) | **~15.99 MB** | 8×H100 SXM | full TTT eval

This submission keeps the PR #1855 training recipe unchanged and only changes quantization. The quantization change is an activation-aware mixed-precision GPTQ path:

1. collect per-input-channel activation RMS during the existing GPTQ calibration pass
2. score candidate column groups with an AWQ-style heuristic
   - `weight_score = mean(abs(w), dim=0)`
   - `saliency = act_rms * weight_score`
   - `group_score = saliency[start:end].sum()`
3. select one salient `64`-column group
4. quantize that group at `int8` inside the same full-tensor GPTQ solve
5. keep stock PR #1855 LQER on top of the resulting AWQ-aware GPTQ base

The motivation for this writeup is that end-to-end reruns of the PR #1855 base showed enough pretrained-model variance that tiny final-score deltas were hard to interpret cleanly. Rather than claim a training win from a slightly luckier rerun, this submission matches the original PR #1855 seeds and stop steps and compares the quantized model at those exact step counts.

## Results

### Step-matched comparisons against PR #1855

| Seed | Stop step | Prequant BPB (PR1855) | Prequant BPB (AWQ) | Quantized BPB (PR1855) | Quantized BPB (AWQ) | Post-TTT BPB (PR1855) | Post-TTT BPB (AWQ) | Artifact bytes (PR1855) | Artifact bytes (AWQ) |
|------|----------:|----------------------:|-------------------:|-----------------------:|--------------------:|----------------------:|-------------------:|------------------------:|---------------------:|
| 42   | 4945 | 1.06395844 | 1.06384082 | 1.07254371 | **1.07225564** | 1.05989454 | **1.05957221** | 15,897,259 | 15,985,824 |
| 0    | 4932 | 1.06544819 | 1.06555331 | 1.07406724 | **1.07403531** | 1.06124613 | 1.06127329 | 15,900,947 | 15,983,935 |
| 1234 | 4917 | 1.06596989 | 1.06574247 | 1.07477929 | **1.07427091** | 1.06208695 | **1.06158679** | 15,907,550 | 15,996,559 |
| **Mean** | **4931** | **1.06512551** | **1.06504553** | **1.07379675** | **1.07352062** | **1.06107587** | **1.06081076** | **15,901,918** | **15,988,772** |

### Quantization-tax view

- PR #1855 mean quantization tax:
  - `1.07379675 - 1.06512551 = 0.00867124`
- AWQ mean quantization tax:
  - `1.07352062 - 1.06504553 = 0.00847509`

So the activation-aware GPTQ recipe recovers about `0.00019615` BPB of mean quantization tax on the matched-step 3-seed suite, while staying under the 16 MB cap on every seed.

At final post-TTT, the matched-step means are:

- PR #1855: `1.06107587`
- activation-aware GPTQ: `1.06081076`

for a mean reduction of `0.00026511` BPB.

## What changed

Compared to the PR #1855 base stack, the functional change is in `train_gpt.py`:

- add activation-stat collection during the existing GPTQ calibration pass
- add exact mixed-bit GPTQ support for a selected group inside the same Hessian-based solve
- keep stock LQER behavior on top of the AWQ-aware quantized base
- add `FORCE_STOP_STEP` to support step-matched evaluation

No training hyperparameters were changed for these runs. The base model recipe is the PR #1855 seed-matched recipe.

## Reproducing

This record folder assumes the same CaseOps sp8192 dataset/tokenizer used by PR #1855, sourced from Hugging Face:

- dataset repo: `romeerp/parameter-golf-caseops-v1`
- variant: `sp8192_lossless_caps_caseops_v1_reserved`

The three runs in this folder use:

- seed `42`, `FORCE_STOP_STEP=4945`
- seed `0`, `FORCE_STOP_STEP=4932`
- seed `1234`, `FORCE_STOP_STEP=4917`

The quantization knobs are:

- `AWQ_LITE_ENABLED=1`
- `AWQ_LITE_BITS=8`
- `AWQ_LITE_GROUP_TOP_K=1`
- `AWQ_LITE_GROUP_SIZE=64`
- stock PR #1855 LQER settings:
  - `LQER_ENABLED=1`
  - `LQER_ASYM_ENABLED=1`
  - `LQER_RANK=4`
  - `LQER_FACTOR_BITS=4`
  - `LQER_ASYM_GROUP=64`
  - `LQER_TOP_K=3`

## Included files

- `train_gpt.py` — modified training/quantization script
- `README.md` — this writeup
- `submission.json` — structured metadata
- `requirements.txt` — Python dependencies reference
- `train_seed42.log`, `train_seed0.log`, `train_seed1234.log` — full matched-step run logs
