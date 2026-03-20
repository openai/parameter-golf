# Modal Ops Log

## 2026-03-19

### Objective
Move autoresearch orchestration from RunPod-heavy operation to Modal, keep multiple finite research lanes alive in parallel, and preserve enough state in git that progress is not trapped in live containers or chat.

### Modal orchestration changes
- Added [modal_lane_orchestrator.py](/Users/simon/Code/parameter-golf/modal_lane_orchestrator.py) to launch lane batches on Modal H100s using a persistent Modal volume.
- Updated [run_lane.sh](/Users/simon/Code/parameter-golf/run_lane.sh) to:
  - run Python unbuffered for live logs
  - honor externally provided lane/stage/namespace overrides
  - support focused `eval_window_discovery` and `storage_export_discovery` lanes
- Added lane-specific research programs:
  - [program_record.md](/Users/simon/Code/parameter-golf/program_record.md)
  - [program_eval.md](/Users/simon/Code/parameter-golf/program_eval.md)
  - [program_eval_window.md](/Users/simon/Code/parameter-golf/program_eval_window.md)
  - [program_storage.md](/Users/simon/Code/parameter-golf/program_storage.md)
  - [program_storage_export.md](/Users/simon/Code/parameter-golf/program_storage_export.md)
  - [program_nonrecord.md](/Users/simon/Code/parameter-golf/program_nonrecord.md)

### Bugs fixed
- Fixed remote log buffering so Modal apps stream real autoresearch progress.
- Fixed namespace override handling so Modal lanes can resume from intended seed histories.
- Fixed `core_record_discovery` seeding so it uses promoted code as a starting point without inheriting an impossible `600s` BPB gate for `180s` proxy runs.

### Strategy split
- `core_record_discovery`: fast-core record-track proxy search
- `eval_time_discovery`: general eval-time calibration
- `eval_window_discovery`: dedicated sliding-window evaluation exploration
- `storage_discovery`: broad export/quantization search
- `storage_export_discovery`: dedicated mixed-precision/export-path search
- `core_nonrecord_promotion`: longer-horizon non-record core search

### Notable lane progress observed during Modal rollout
- `storage_discovery` improved from `1.6496` to `1.4601`, then to `1.4427`
- `eval_time_discovery` improved from `1.4936` to `1.4583`, then to `1.4578`
- `storage_export_discovery` early result reported by user: `1.6496 -> 1.4580`
- `eval_window_discovery` early result reported by user: `1.4936 -> 1.4599`

### Active / recent app IDs
- `ap-o41PS1RSK82lCgmDoFCInR`: `core_record_discovery`
- `ap-ake6WNRy3CeEkg1XBg6W2G`: `eval_time_discovery`
- `ap-YTDrV0xN9nInJmbYV4YZu2`: `eval_window_discovery`
- `ap-UDRP202S6UZgiKWgm9jein`: `storage_export_discovery`
- Earlier `storage_discovery` and `core_nonrecord_promotion` apps were launched successfully; relaunches may be needed as batches stop.

### Monitoring
From repo root:

```bash
source .venv-modal/bin/activate
modal app list
modal app logs <APP_ID> --timestamps
modal app stop <APP_ID>
```

### Notes
- Large local experiment trees under `autoresearch/*/experiments` were intentionally not checkpointed here.
- Record-track search is now explicitly treated as a different optimization problem from non-record search.
