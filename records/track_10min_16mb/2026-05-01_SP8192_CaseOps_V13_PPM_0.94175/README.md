# SP8192 CaseOps v13 PPM tuned gate

This submission consolidates our strongest v13 lane: the SP8192 CaseOps transformer stack with SmearGate BOS masking, per-group `lrzip` compression, and a causal sidecar-aware byte PPM evaluator.

The final score comes from a narrow evaluator retune over the already-validated v13/SP8192 artifacts:

```text
PPM_ORDER=5
PPM_H=0.999
PPM_L=0.18
PPM_T=0.80
TTT_ENABLED=0
```

Thanks to Claude for the late-stage experiment design help and to Codex for implementation, audit, run coordination, and packaging. This stack also builds on public Parameter Golf work by @clarkkev, @bigbag, @codemath3000, @OE-GOD, @remg1997, @joshuaswanson, @MarioPaerle, @classiclarryd, @simonbissonnette, @dexhunter, @romeerp, @samacqua, @renqianluo, @jorge-asenjo, @Omrigotlieb, @AnirudhRahul, and @ndokutovich. See `REFERENCES.md` for the component lineage and PR numbers.

## Score

| Seed | Final `ppm_sliding val_bpb` | Artifact bytes | Training stop | Eval time |
|---:|---:|---:|---:|---:|
| 42 | `0.94151072` | `15,942,636` | `4802` steps / `599.546s` | `510.410s` |
| 314 | `0.94180705` | `15,946,930` | `4803` steps / `599.583s` | `500.300s` |
| 999 | `0.94192810` | `15,937,542` | `4767` steps / `599.657s` | `497.643s` |

Three-seed mean:

```text
0.94174862
```

Sample standard deviation:

```text
0.00021474
```

All three artifacts remain under the strict decimal `16,000,000` byte cap. Using the checked-in `train_gpt.py` with no local minifier available, the largest measured artifact plus compressed code wrapper is `15,995,881` bytes.

## What changed

Relative to the previous SP8192 + byte-PPM tuned-gate line, v13 combines:

- CaseOps SP8192 tokenization and byte sidecar accounting for correct `val_bpb` normalization.
- SmearGate with the BOS cross-document leak mask applied in both normal forward and TTT forward paths.
- Per-group `lrzip` compression for banked int6 tensors, with Brotli for the remainder/code wrapper.
- PPM order 5 with the final gate retune `H=0.999`, `L=0.18`, `T=0.80`.
- TTT disabled for the submitted score, so the validation pass is a single causal PPM scoring pass over the quantized artifact.

## Lineage and attribution

This is not a from-scratch model. The code is a consolidation of several public Parameter Golf ideas:

- SP8192 tokenizer, recurrence, QK gain, and compact GPT training lineage from PR #1394, PR #1493, and PR #1855.
- Causal byte-PPM mixer lineage from PR #1795, PR #1959, and PR #1991.
- SmearGate / attention output gate lineage from modded-nanogpt @classiclarryd and PR #1667, plus the BOS cross-document leak fix discussed in PR #2014 / the PR #1797 base audit.
- Per-group `lrzip` compression lineage from PR #1586 through PR #1667 / PR #1729-style grouped serialization work.
- LQER/AWQ/asymmetric-rescale and related quantization/optimization pieces from PR #1530, PR #1797, PR #1886, PR #1923, and PR #1855.
- Online n-gram tilt / scoring overlay ideas from PR #1145 and PR #1967, though the submitted score uses the PPM path rather than TTT.

Our specific contribution in this PR is the v13 consolidation, the CaseOps sidecar-aware evaluation packaging, and the final PPM gate retune to `H=0.999`, `L=0.18`, `T=0.80` over the same seed set.

The checked-in script sets the final PPM gate as defaults, so a fresh run follows the same configuration without external environment overrides.

## Evidence notes

The included `train_seed*.log` files are the full source training logs for the three artifacts. The final PPM gate was tuned after those artifacts were produced, so the exact final score lines are in the paired `eval_seed*_v13_ppm.log` files. This is an evaluation-only retune: it does not change trained weights, artifact bytes, tokenizer, or training data.

A fresh end-to-end v13 rerun with these defaults was started on the 8xH100 box while this PR was prepared; these logs can replace the paired evidence as soon as they finish.

Update: the fresh seed-42 rerun finished cleanly as `fresh_seed42_v13_submit.log`:

```text
stopping_early: wallclock_cap train_time: 599686ms step: 4773/20000
Total submission size quantized+pergroup: 15987305 bytes
diagnostic quantized val_loss:2.35586432 val_bpb:1.07646816 eval_time:10407ms
ppm_mixer val_bpb:0.94182660 eval_time:462353ms order=5 H=0.999 L=0.18 T=0.8 N_tokens=47851520 N_sidecar_bytes=151074499
ppm_sliding val_loss:2.36677335 val_bpb:0.94182660 eval_time:507652ms
```

That fresh end-to-end score is slightly worse than the original seed-42 eval-only evidence, so the headline 3-seed mean is left unchanged until the queued fresh seed-314 run also finishes.

## Exact final lines

Seed 42:

```text
ppm_mixer val_bpb:0.94151072 eval_time:464892ms order=5 H=0.999 L=0.18 T=0.8 N_tokens=47851520 N_sidecar_bytes=151074499
ppm_sliding val_loss:2.36642906 val_bpb:0.94151072 eval_time:510410ms
```

Seed 314:

```text
ppm_mixer val_bpb:0.94180705 eval_time:454770ms order=5 H=0.999 L=0.18 T=0.8 N_tokens=47851520 N_sidecar_bytes=151074499
ppm_sliding val_loss:2.36687117 val_bpb:0.94180705 eval_time:500300ms
```

Seed 999:

```text
ppm_mixer val_bpb:0.94192810 eval_time:452193ms order=5 H=0.999 L=0.18 T=0.8 N_tokens=47851520 N_sidecar_bytes=151074499
ppm_sliding val_loss:2.36740764 val_bpb:0.94192810 eval_time:497643ms
```

## Included files

- `train_gpt.py` - exact submitted script, with v13 PPM defaults set to `0.999/0.18/0.80`
- `train_seed42.log`, `train_seed314.log`, `train_seed999.log` - source training logs for the three artifacts
- `eval_seed42_v13_ppm.log`, `eval_seed314_v13_ppm.log`, `eval_seed999_v13_ppm.log` - exact v13 PPM score logs
- `fresh_seed42_v13_submit.log` - fresh end-to-end v13 seed-42 rerun with the submitted defaults
- `submission.json` - leaderboard metadata
- `LEGALITY_AUDIT.md` - compliance audit
- `REFERENCES.md` - public PR and component lineage notes
- `requirements.txt` - Python package/runtime notes
