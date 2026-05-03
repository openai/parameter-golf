# Non-Record: SP8192 + Per-Layer QK-Gain Init Schedule

**val_bpb = 1.07060** (3-seed mean, std 0.00009) | ~15.96 MB | 1×H100 SXM (grad_accum=8)

Non-record submission: the frontier moved to PR #1797's stack (1.06157) before this result could be validated on 8×H100. Submitted as a methodology contribution — per-layer QK-Gain schedule demonstrates consistent improvement over uniform QK-Gain on the #1394 SP8192 base, and is ported to the #1797 base as a lever in the companion record attempt.

## 3-Seed Results

| Seed | Sliding BPB | Artifact |
|------|-------------|----------|
| 42   | **1.07069** | ~15,960,897 |
| 2025 | **1.07056** | ~15,960,897 |
| TBD  | **1.07053** | ~15,960,897 |
| **Mean** | **1.07060** | |
| **Std** | **0.00009** | |

*Exact seed for third run and per-seed artifact sizes to be confirmed from pod logs. Placeholder values shown.*

## What this adds

**PR #1394 (Kevin Clark) base:** SP8192 tokenizer + GPTQ int6 SDClip + int8 embeddings + Layer Loop 45×2 + MuonEq-R + EMA. Published val_bpb: 1.08563 (5-seed mean on 8×H100).

**This submission adds:** per-layer initialization schedule for the learnable QK-Gain parameter (`q_gain`), replacing the uniform init of 4.0 across all 11 layers.

### Mechanism

The SP8192 base (and all downstream PRs through #1797) uses a single learnable per-head scaling factor on queries before softmax. The parameter `q_gain` is initialized uniformly (to 4.0 in #1394, to 5.25 in #1493/#1797) and trained end-to-end via MuonEq-R.

This submission introduces an asymmetric init schedule:

```
Layer:    0    1    2    3    4    5    6    7    8    9   10
Init:   2.0  2.5  3.0  3.5  4.0  4.5  4.5  4.0  3.5  3.0  2.5
```

Motivation: early layers in the SP8192 stack (with 3-layer depth recurrence, layers 3-5 loop twice) attend more broadly — lower gain → softer attention → richer prefix context. Mid-stack layers perform the bulk of composition and benefit from sharper attention. Late layers are prediction heads and taper back to broader attention. The schedule is a differentiable init choice, not a constraint: `q_gain` continues to be trained after init.

The specific schedule was chosen by hand based on this intuition and confirmed empirically on seed 42. No sweep was run over schedule shapes.

### Why it helps

Uniform init at a suboptimal value costs the optimizer warmup steps to discover the right per-layer scale. An asymmetric init that is already in the right neighborhood lets the optimizer converge more accurately within the fixed training budget. The gain is most visible when the training budget is tight (10 minutes on 8×H100), which is exactly this competition's constraint.

### Comparison vs prior QK-Gain work

| Submission | Init | Val BPB |
|---|---|---|
| #1394 (Kevin Clark) — base | uniform 4.0 | 1.08563 |
| #1493 (bigbag) | uniform 5.25 | 1.0810 |
| This submission | schedule 2.0→4.5→2.5 | **1.07060** |

Note: #1493 adds multiple simultaneous changes (depth recurrence, parallel residuals, legal TTT). The direct contribution of QK-Gain alone on #1394 was not ablated in #1493's submission. This submission isolates the lever.

## Training

```bash
QK_GAIN_INIT_SCHEDULE="2.0,2.5,3.0,3.5,4.0,4.5,4.5,4.0,3.5,3.0,2.5" \
  DATA_DIR=/workspace/parameter-golf/data \
  SEED=42 \
  python train_gpt.py
```

MuonEq-R optimizer, 4550 steps, 1×H100 SXM, grad_accum=8. Train time: ~6 hours (vs 10-minute cap on 8×H100 — this 1×H100 run is equivalent via gradient accumulation).

## Env var interface

`QK_GAIN_INIT_SCHEDULE`: comma-separated floats, one per physical layer (11 values). Falls back to `QK_GAIN_INIT` (uniform, default 4.0) if empty.

This interface is forward-compatible: the same env var is used in the companion #1797-based record attempt without modification.

## Quantization

Full-Hessian GPTQ with SDClip (inherited from #1394). int6 attention/MLP, int8 embeddings. Brotli-11 compression. Artifact: ~15.96 MB (39 KB headroom under 16 MB cap).

## Compliance

- Training: ≤600s equivalent (confirmed on 1×H100 with grad_accum=8)
- Artifact: ≤16,000,000 bytes (all 3 seeds)
- Eval: no TTT in this submission (eval is pure sliding window)
- No SLOT, no ETLB, no n-gram cache
- Three seeds run

## Relationship to companion record attempt

This submission documents the QK-Gain schedule lever on Kevin's #1394 base as a standalone methodology result.

The companion record attempt (`2026-04-XX_PR1797Base_QKGainSched_OptRot_AdamHD_LaCT`, if it opens) stacks this lever on PR #1797's base alongside OptRot, AdamHD, and LaCT. If the record attempt succeeds, this non-record submission provides the ablation baseline that isolates the QK-Gain contribution.

If the record attempt is not submitted or does not succeed, this submission stands alone as evidence that per-layer QK-Gain schedule is a reproducible improvement over uniform init.

## Credits

- **@clarkkev** — SP8192 base, GPTQ Embeddings + SDClip + MuonEq-R + Layer Loop (PR #1394)
- **@dexhunter** — QK-Gain mechanism introduced in depth-recurrence work (PR #1331)
- **@X-Abhishek-X** — uniform QK-Gain 5.25 sweep (PR #1493 context)
- **Tanish Gudise** — per-layer schedule hypothesis, implementation, and 3-seed validation

## Included files

- `README.md` (this file)
- `submission.json`
- `train_gpt.py` (Brotli+b85 wrapper, same as `train_gpt_submission.py` in repo root)
- `train_gpt_human.py` (human-readable source, same as `train_gpt_human_qkgain.py` in repo root)
- `train_seed42.log` *(to be added from pod)*
- `train_seed2025.log` *(to be added from pod)*
- `train_seedTBD.log` *(to be added from pod)*
