# Non-record: SP8192 + RandProj384 tied embeddings + Pairwise-QK Muon -- Single-seed negative result

This is a non-record submission testing two new ideas on the SP8192 / CaseOps / legal-TTT stack:

- random-projection tied embeddings (`RandProj384`)
- pairwise-head Muon orthogonalization for Q/K (`PairMuonQK`)

This run completed training and quantization on 8xH100 SXM within the 10-minute training cap and produced a legal sub-16MB artifact, but it was not competitive with the frontier. I am submitting it as a negative result because the failure is clear and informative.

## Single-seed result

Seed: `42`

- train steps: `1724`
- train wallclock: `599714 ms`
- in-run full validation: `val_loss 2.4662`, `val_bpb 1.1269`
- post-EMA diagnostic: `val_loss 2.47020597`, `val_bpb 1.12868936`
- quantized model size: `15,399,365` bytes
- total submission size: `15,438,770` bytes
- headroom to 16MB: `561,230` bytes

## What happened

The artifact fit comfortably under the size limit, but model quality regressed too far from the public frontier before quantization and before TTT could help.

The post-training legal TTT eval path also did not complete robustly on this stack:

- larger TTT batch hit OOM during adaptation
- smaller TTT batch progressed but was too slow to be practical

Because of that, I am not claiming a final post-TTT score.

## Why this is still useful

This result directly constrains the design space:

- aggressive latent tied-embedding compression was destructive (`1.1269` pre-TTT BPB, more than `0.06` worse than the strongest open public frontier)
- pairwise Q/K Muon orthogonalization did not preserve frontier behavior
- parameter savings alone are insufficient; pre-quantization quality matters

## Why this is non-record

- single seed only
- not competitive with current SOTA
- no successful final TTT evaluation
- submitted as an interesting negative result rather than a leaderboard claim

## Included files

- `train_gpt.py`
- `requirements.txt`
- helper files required by the run
- `train_seed42.log`
- `ttt_eval_seed42_fail.log`
- `submission.json`
