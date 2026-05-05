# BIJEPAX-lite JEPA + SP8192 CaseOps PPM

This record captures a Claude-designed, JEPA-inspired training regularizer layered onto the SP8192 CaseOps + per-group compression + PPM sliding stack.

The key idea is deliberately narrow: add a small bidirectional hidden-state prediction objective during training, then remove it entirely from the artifact. The final submitted model is still the quantized base GPT, scored causally by the PPM sliding evaluator.

Thanks to Claude for designing the BIJEPAX-lite auxiliary objective and helping shape the experiment, to Codex for implementation, auditing, packaging, and run coordination, and to the Parameter Golf community for the public ideas this stack builds on. The inherited stack uses public Parameter Golf work by @clarkkev, @bigbag, @codemath3000, @OE-GOD, @remg1997, @joshuaswanson, @MarioPaerle, @classiclarryd, @simonbissonnette, @dexhunter, @romeerp, @samacqua, @renqianluo, @jorge-asenjo, @Omrigotlieb, @AnirudhRahul, @ndokutovich, and @H1cSuNtDr4C0n3S. See `REFERENCES.md` for the detailed lineage.

## Score

| Seed | Final `ppm_sliding val_bpb` | Quantized diagnostic | Artifact bytes | Train stop | Eval time | Exit |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | `0.97234287` | `1.11544494` | `15,997,180` | `2014` steps / `599.843s` | `502.131s` | `0` |
| 314 | `0.97206308` | `1.11562304` | `15,999,539` | `2012` steps / `599.586s` | `499.038s` | `0` |
| 999 | `0.97373767` | `1.11757370` | `15,997,593` | `2013` steps / `599.821s` | `496.384s` | `0` |

Three-seed mean:

```text
0.97271454
```

Sample standard deviation:

```text
0.00089703
```

All three runs are under the strict decimal `16,000,000` byte cap and under the 600s eval cap.

## What changed

BIJEPAX-lite adds a training-only auxiliary module:

- hop-4 forward hidden-state prediction
- hop-4 backward hidden-state prediction
- cosine embedding-space loss
- LayerNorm-stabilized two-layer predictor heads
- no cycle head in the submitted lightweight config
- active from `35%` to `80%` of the wallclock training schedule
- separate optimizer for the auxiliary predictor
- predictor heads are never serialized into the artifact

## Lineage and attribution

The submitted JEPA branch is a custom training-objective experiment on top of an inherited Parameter Golf stack:

- SP8192 tokenizer, recurrence, QK gain, and compact GPT training lineage from PR #1394, PR #1493, and PR #1855.
- Causal byte-PPM mixer lineage from PR #1795, PR #1959, and PR #1991.
- SmearGate / attention output gate lineage from modded-nanogpt @classiclarryd and PR #1667, plus the BOS cross-document leak fix discussed in PR #2014 / the PR #1797 base audit.
- Per-group `lrzip` compression lineage from PR #1586 through PR #1667 / PR #1729-style grouped serialization work.
- LQER/AWQ/asymmetric-rescale and related quantization/optimization pieces from PR #1530, PR #1797, PR #1886, PR #1923, and PR #1855.
- JEPA-Lite local-competition precedent from PR #2027.
- Online n-gram tilt / scoring overlay ideas from PR #1145 and PR #1967, although the submitted score uses the PPM path with TTT disabled.

Our specific addition is the Claude-designed BIJEPAX-lite training-only auxiliary objective: bidirectional hop-4 hidden-state prediction with cosine loss and LayerNorm-stabilized predictor heads, removed from the serialized artifact.

Submitted config:

```bash
DISABLE_COMPILE=1
CASEOPS_ENABLED=1
VOCAB_SIZE=8192
TTT_ENABLED=0
PPM_MIXER_ENABLED=1
PPM_ORDER=5
PPM_H=0.999
PPM_L=0.18
PPM_T=0.80
LQER_TOP_K=1
BIJEPAX_ENABLED=1
BIJEPAX_WEIGHT=0.01
BIJEPAX_START_FRAC=0.35
BIJEPAX_END_FRAC=0.80
BIJEPAX_FWD_HOPS=4
BIJEPAX_BWD_HOPS=4
BIJEPAX_CYCLE=0
BIJEPAX_HEAD_DIM=32
BIJEPAX_STRIDE=64
BIJEPAX_LR=0.001
```

`LQER_TOP_K=1` is used to preserve artifact headroom. Seed 314 is the tightest artifact at `15,999,539` bytes.

## Legality notes

The JEPA component itself is training-only:

- `MultiDirectionalBiJEPAX` is created as a standalone module, not as a child of the base GPT.
- Serialization saves `base_model.state_dict()`, so BIJEPAX predictor weights are absent from `final_model.int6.ptz`.
- The auxiliary loss uses hidden states from training batches only.
- `TTT_ENABLED=0`, so there is no validation-set test-time training in this record.
- SmearGate BOS masking is present to avoid packed-document cross-boundary leakage.

The final score uses the existing causal PPM sliding path. The inherited compliance risk, if any, is the existing SP8192 CaseOps + PPM byte-sidecar lane rather than the BIJEPAX-lite auxiliary regularizer.

See `LEGALITY_AUDIT.md` and `STATIC_AUDIT_NOTES.md` for the audit details.

## References

This is a custom Claude-designed JEPA-inspired auxiliary objective, not a claimed faithful reproduction of a specific BiJEPA paper.

The design is inspired by:

- I-JEPA: https://arxiv.org/abs/2301.08243
- LLM-JEPA: https://arxiv.org/abs/2509.14252
- MC-JEPA: https://arxiv.org/abs/2307.12698

See `REFERENCES.md` for wording notes and claims we intentionally avoid.

## Included files

- `train_gpt.py` - exact script used for the submitted runs
- `train_seed42.log`, `train_seed314.log`, `train_seed999.log` - run logs
- `full_seed42.txt`, `full_seed314.txt`, `full_seed999.txt` - full captured source/log snapshots
- `submission.json` - leaderboard metadata
- `LEGALITY_AUDIT.md` - compliance audit
- `STATIC_AUDIT_NOTES.md` - static code review notes
- `REFERENCES.md` - prior work and PR wording guidance
- `JEPA.mp4` - short visual/demo asset for the PR

## Exact final lines

Seed 42:

```text
Total submission size quantized+pergroup: 15997180 bytes
diagnostic quantized val_loss:2.44116551 val_bpb:1.11544494 eval_time:10342ms
ppm_mixer val_bpb:0.97234287 eval_time:456845ms order=5 H=0.999 L=0.18 T=0.8 N_tokens=47851520 N_sidecar_bytes=151074499
ppm_sliding val_loss:2.45118426 val_bpb:0.97234287 eval_time:502131ms
```

Seed 314:

```text
Total submission size quantized+pergroup: 15999539 bytes
diagnostic quantized val_loss:2.44155528 val_bpb:1.11562304 eval_time:9926ms
ppm_mixer val_bpb:0.97206308 eval_time:453715ms order=5 H=0.999 L=0.18 T=0.8 N_tokens=47851520 N_sidecar_bytes=151074499
ppm_sliding val_loss:2.45044876 val_bpb:0.97206308 eval_time:499038ms
```

Seed 999:

```text
Total submission size quantized+pergroup: 15997593 bytes
diagnostic quantized val_loss:2.44582432 val_bpb:1.11757370 eval_time:11393ms
ppm_mixer val_bpb:0.97373767 eval_time:451054ms order=5 H=0.999 L=0.18 T=0.8 N_tokens=47851520 N_sidecar_bytes=151074499
ppm_sliding val_loss:2.45502055 val_bpb:0.97373767 eval_time:496384ms
```
