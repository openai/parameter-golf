Non-record Modal run captured from dashboard logs.

This run achieved strong quality but is not leaderboard-valid for 16MB due to artifact size.

Run metadata:
- App ID: `ap-7GgwPNSXR9TJqDNlPWoWxQ`
- Hardware/time: 8xH100, 10-minute cap
- Train stop: `step:6900/20000`, `step_avg:86.96ms`, `train_time:600027ms`
- Peak memory: `20920 MiB allocated`, `21266 MiB reserved`

Quality metrics from log:
- Pre-quant at stop: `val_loss:1.9340`, `val_bpb:1.1454`
- Int6 roundtrip exact: `val_loss:1.93554747`, `val_bpb:1.14634024`
- Int6 sliding window exact (stride 64): `val_loss:1.89606859`, `val_bpb:1.12296159`

Size metrics from log:
- Serialized model: `107098843 bytes`
- Code size: `75697 bytes`
- Serialized model int6+zstd: `20830583 bytes`
- Total submission size int6+zstd: `20906280 bytes`
- Over 16MB limit by: `4906280 bytes`

Included files:
- `submission.json`
- `train.log` (captured dashboard log)
- `train_gpt.py`
