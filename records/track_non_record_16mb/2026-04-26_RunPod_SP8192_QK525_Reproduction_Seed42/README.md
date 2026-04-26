# Non-Record RunPod Reproduction: SP8192 QK-Gain 5.25 Seed 42

This is a non-record reproduction/evidence bundle from a single budget-capped
RunPod pass. It is not a new SOTA claim and should not be submitted to the
leaderboard as an original record.

The run used the public `SP8192 + 3-Layer Recurrence + Parallel Residuals +
Legal TTT` record script as a baseline reproduction target. The useful outcome
is infrastructure validation: the 8xH100 SXM pod trained under the 600 second
limit and produced a compressed artifact under the 16,000,000 byte cap after the
missing Brotli dependency was installed.

## Result Summary

| Item | Value |
|------|------:|
| Seed | 42 |
| GPUs | 8x NVIDIA H100 80GB HBM3 |
| Train shards | 128 |
| Train stop | 4575/20000 steps |
| Train wallclock | 588080 ms |
| Pre-quant post-EMA bpb | 1.08710081 |
| Quantized chunked bpb | 1.09924840 |
| Compressed model bytes | 15976010 |
| Total counted bytes | 15992604 |

The original training run completed the training phase and failed during
packaging because the pod image did not include the Python `brotli` module. A
checkpoint recovery pass installed `brotli`, packaged `final_model.pt`, and
completed the standard quantized eval, but then failed before the sliding-window
and TTT eval lines because the temporary recovery wrapper triggered a
TorchDynamo import issue.

## Included Files

- `README.md` - this file.
- `submission.json` - metadata for this non-record reproduction bundle.
- `requirements.txt` - missing runtime dependencies observed during the run.
- `train_gpt.py` - runnable reproduction script, stored as the compressed
  LZMA/base85 wrapper used by the source record.
- `train_gpt_unpacked.py` - audit-only unpacked source extracted from
  `train_gpt.py`.
- `train_gpt_unpacked.sha256` - SHA-256 digest for the unpacked source.
- `verify_unpacked_source.py` - verifies that the unpacked source matches the
  compressed wrapper payload and hash.
- `train_seed42_runpod.log` - original 8xH100 training log.
- `pack_seed42_recovery.log` - checkpoint packaging and partial eval log.

## Compliance Status

- Artifact under 16,000,000 bytes: yes.
- Training under 600 seconds on 8xH100: yes.
- Three-seed statistical evidence: no.
- Complete sliding-window/TTT eval from this run: no.
- New technique or SOTA claim: no.

## Reproduction Notes

Install the missing packaging dependencies before launching the run:

```bash
pip install -r requirements.txt
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
```

Prepare the SP8192 cached FineWeb data:

```bash
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 128
```

Run on 8xH100:

```bash
SEED=42 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

`train_gpt.py` is intentionally the compressed two-line LZMA/base85 wrapper from
the source record, so it executes the decompressed payload directly. For review,
inspect `train_gpt_unpacked.py` instead. To verify it matches the wrapper:

```bash
python3 verify_unpacked_source.py
```

## Why This Is Non-Record

The repository README asks new SOTA records to beat the existing SOTA by at
least 0.005 nats, provide enough logs for statistical significance, and
reproducibly run in under 10 minutes on 8xH100s. This bundle only provides one
seed and reproduces a published top approach, so it is packaged as transparent
run evidence rather than a competition record.
