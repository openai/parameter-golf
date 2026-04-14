# PR 1251 — Non-record: Online Hessian GPTQ (val_bpb=1.1349)

**Author:** Josue Alexander Ibarra (2026-04-02)
**Claimed BPB:** Base 1.1390, TTT 1.1349. Artifact 14.9MB.
**Seeds:** not stated in commit message
**Hardware:** not stated

## Files retrieved
- `records__track_10min_16mb__2026-04-01_ApproachL_MaxPerf__train_gpt.py`

No README in PR. Only the `train_gpt.py` file was added (1834 insertions).

## Claimed changes (from commit message, verbatim)
> NEGATIVE RESULT: Online Hessian GPTQ accumulates H during training to eliminate post-training GPTQ calibration. But the overhead (112ms vs 95ms/step) costs more training steps than it saves. Base 1.1390, TTT 1.1349. Artifact 14.9MB.

## Notes from train_gpt.py docstring
- File header: "Approach L: Maximum Performance. Parallel Muon (no DDP) + Online Hessian GPTQ + all Approach I innovations."
- MAX_WALLCLOCK_SECONDS default 583.0 — "Reserve 17s for online hessian overhead + GPTQ finalize"
- Online Hessian: every N steps after 20% of training, run ONE uncompiled forward pass
