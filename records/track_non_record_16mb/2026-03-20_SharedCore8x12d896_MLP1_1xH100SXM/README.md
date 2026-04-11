This record captures a non-record `16MB` submission built from a shared-core recurrent transformer and validated on `1xH100 SXM`.

This is a cloud-validated checkpoint rather than a leaderboard claim. The main leaderboard requires the final train-and-eval run to fit the `8xH100 / 10 minute` setting.

Model summary:
- `VOCAB_SIZE=1024`
- `STORED_LAYERS=8`
- `RECURRENCE_DEPTH=12`
- `MODEL_DIM=896`
- `MLP_MULT=1`
- tied input/output embeddings
- short cloud-ranking run at `200` steps under the `16,000,000` byte artifact cap

Why it is interesting:
- It validates a near-cap shared-core architecture on real cloud hardware rather than only on a local proxy.
- The same branch repeated at essentially the same exact post-roundtrip score on the same hardware class.

Environment:
- Hardware: single `H100 SXM`
- Stack observed during the run: `Python 3.11.10`, `torch 2.4.1+cu124`
- Dataset/tokenizer: published `fineweb10B_sp1024` cached export
- Extra dependencies: none beyond the base repo stack used for this run

Key metrics:
- Pre-quant eval at step `200`: `val_loss:3.3188`, `val_bpb:1.9656`
- Exact post-roundtrip eval: `val_loss:3.37650478`, `val_bpb:1.99975632`
- Train time: `37555ms` (`step_avg:187.78ms`)
- Peak memory: `4210 MiB allocated`, `4462 MiB reserved`
- Serialized model int8+zlib: `12902422 bytes`
- Code size: `57902 bytes`
- Total submission size int8+zlib: `12960324 bytes`

Repeat evidence:
- `train_repeat.log` reran the same branch and finished at `final_int8_zlib_roundtrip_exact val_bpb:1.99970894`
- Repeat total submission size int8+zlib: `12960820 bytes`

Why this is non-record:
- The run was executed on `1xH100`, not on the required `8xH100` leaderboard setting.
- This checkpoint is intended to justify further `8xH100` testing, not replace it.

Included files:
- `train_gpt.py` (exact code snapshot used for the run)
- `train.log` (primary remote training log)
- `train_repeat.log` (repeat run showing stability)
- `submission.json` (metadata)
