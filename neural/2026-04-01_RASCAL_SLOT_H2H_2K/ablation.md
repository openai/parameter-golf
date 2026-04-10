# Ablation: RASCAL_SLOT_H2H_2K
Date: 2026-04-01
Track: neural
Parent: neural/2026-03-31_Rascal_III_SLOT/

## Run

```bash
SEED=444 NPROC_PER_NODE=8 bash neural/2026-04-01_RASCAL_SLOT_H2H_2K/run.sh
```

## Expected key log lines

- `Serialized model: ...`
- `Serialized model int6+zstd: ...`
- `h2h_shared_artifact_bytes:...`
- `h2h_sliding_window_base_exact ...`
- `h2h_sliding_window_slot8steps_exact ...`
- `h2h_sliding_window_delta_exact ...`

## Result

Status: [ ] pending  [ ] pass  [ ] fail
artifact_bytes_shared:
base_bpb_exact:
slot_bpb_exact:
delta_bpb_exact:
notes:
