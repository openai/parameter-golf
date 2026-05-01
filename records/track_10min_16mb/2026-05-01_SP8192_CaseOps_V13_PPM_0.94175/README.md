# SP8192 CaseOps v13 PPM tuned gate

This submission consolidates our strongest v13 lane: the SP8192 CaseOps transformer stack with SmearGate BOS masking, per-group `lrzip` compression, and a causal sidecar-aware byte PPM evaluator.

The final score is backed by three fresh end-to-end v13 reruns with the submitted defaults:

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
| 42 | `0.94182660` | `15,987,305` | `4773` steps / `599.686s` | `507.652s` |
| 314 | `0.94146034` | `15,983,753` | `4770` steps / `599.628s` | `516.897s` |
| 999 | `0.94197117` | `15,988,348` | `4772` steps / `599.644s` | `519.029s` |

Three-seed mean:

```text
0.94175270
```

Sample standard deviation:

```text
0.00026331
```

All three fresh artifacts remain under the strict decimal `16,000,000` byte cap. The largest fresh measured artifact plus compressed code wrapper is `15,988,348` bytes.

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

The included `fresh_seed*_v13_submit.log` files are full fresh end-to-end runs with the submitted PPM defaults in `train_gpt.py`. The older `train_seed*.log` and paired `eval_seed*_v13_ppm.log` files are retained as lineage/eval-retune evidence, but the headline score below uses the cleaner fresh rerun set.

```text
seed 42:
stopping_early: wallclock_cap train_time: 599686ms step: 4773/20000
Total submission size quantized+pergroup: 15987305 bytes
diagnostic quantized val_loss:2.35586432 val_bpb:1.07646816 eval_time:10407ms
ppm_mixer val_bpb:0.94182660 eval_time:462353ms order=5 H=0.999 L=0.18 T=0.8 N_tokens=47851520 N_sidecar_bytes=151074499
ppm_sliding val_loss:2.36677335 val_bpb:0.94182660 eval_time:507652ms

seed 314:
stopping_early: wallclock_cap train_time: 599628ms step: 4770/20000
Total submission size quantized+pergroup: 15983753 bytes
diagnostic quantized val_loss:2.35632034 val_bpb:1.07667653 eval_time:9243ms
ppm_mixer val_bpb:0.94146034 eval_time:471320ms order=5 H=0.999 L=0.18 T=0.8 N_tokens=47851520 N_sidecar_bytes=151074499
ppm_sliding val_loss:2.36627199 val_bpb:0.94146034 eval_time:516897ms

seed 999:
stopping_early: wallclock_cap train_time: 599644ms step: 4772/20000
Total submission size quantized+pergroup: 15988348 bytes
diagnostic quantized val_loss:2.35838976 val_bpb:1.07762211 eval_time:8788ms
ppm_mixer val_bpb:0.94197117 eval_time:473888ms order=5 H=0.999 L=0.18 T=0.8 N_tokens=47851520 N_sidecar_bytes=151074499
ppm_sliding val_loss:2.36682950 val_bpb:0.94197117 eval_time:519029ms
```

The earlier eval-only three-seed mean was `0.94174862`; the fresh end-to-end mean is `0.94175270`. The difference is only `0.00000408` bpb, and the fresh set is the cleaner evidence for review.

## Exact final lines

Seed 42:

```text
ppm_mixer val_bpb:0.94151072 eval_time:464892ms order=5 H=0.999 L=0.18 T=0.8 N_tokens=47851520 N_sidecar_bytes=151074499
ppm_sliding val_loss:2.36642906 val_bpb:0.94151072 eval_time:510410ms
fresh ppm_mixer val_bpb:0.94182660 eval_time:462353ms order=5 H=0.999 L=0.18 T=0.8 N_tokens=47851520 N_sidecar_bytes=151074499
fresh ppm_sliding val_loss:2.36677335 val_bpb:0.94182660 eval_time:507652ms
```

Seed 314:

```text
ppm_mixer val_bpb:0.94180705 eval_time:454770ms order=5 H=0.999 L=0.18 T=0.8 N_tokens=47851520 N_sidecar_bytes=151074499
ppm_sliding val_loss:2.36687117 val_bpb:0.94180705 eval_time:500300ms
fresh ppm_mixer val_bpb:0.94146034 eval_time:471320ms order=5 H=0.999 L=0.18 T=0.8 N_tokens=47851520 N_sidecar_bytes=151074499
fresh ppm_sliding val_loss:2.36627199 val_bpb:0.94146034 eval_time:516897ms
```

Seed 999:

```text
ppm_mixer val_bpb:0.94192810 eval_time:452193ms order=5 H=0.999 L=0.18 T=0.8 N_tokens=47851520 N_sidecar_bytes=151074499
ppm_sliding val_loss:2.36740764 val_bpb:0.94192810 eval_time:497643ms
fresh ppm_mixer val_bpb:0.94197117 eval_time:473888ms order=5 H=0.999 L=0.18 T=0.8 N_tokens=47851520 N_sidecar_bytes=151074499
fresh ppm_sliding val_loss:2.36682950 val_bpb:0.94197117 eval_time:519029ms
```

## Included files

- `train_gpt.py` - exact submitted script, with v13 PPM defaults set to `0.999/0.18/0.80`
- `train_seed42.log`, `train_seed314.log`, `train_seed999.log` - source training logs for the three artifacts
- `eval_seed42_v13_ppm.log`, `eval_seed314_v13_ppm.log`, `eval_seed999_v13_ppm.log` - exact v13 PPM score logs
- `fresh_seed42_v13_submit.log` - fresh end-to-end v13 seed-42 rerun with the submitted defaults
- `fresh_seed314_v13_submit.log` - fresh end-to-end v13 seed-314 rerun with the submitted defaults
- `fresh_seed999_v13_submit.log` - fresh end-to-end v13 seed-999 rerun with the submitted defaults
- `submission.json` - leaderboard metadata
- `LEGALITY_AUDIT.md` - compliance audit
- `REFERENCES.md` - public PR and component lineage notes
- `requirements.txt` - Python package/runtime notes
