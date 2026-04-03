# DeepFloor

**Non-record submission candidate.**

DeepFloor is a compact recurrent multi-view language model that explores a different part of the Parameter Golf design space: repeated QKV+O recurrent blocks, either periodic floor attention or a fused recurrent accumulator, plus an explicit stability stack for long tied-depth execution.

The current checked-in candidate is the small-box `fused_d32_v2` run on real `enwik8`:

- `val_bpb = 7.9221`
- `test_bpb = 8.1786`
- `artifact_bytes = 8448`
- `bytes_total = 56477`

This is not a record claim. The point of the submission is to package a real, reproducible DeepFloor architecture and its execution surface as a `track_non_record_16mb` contribution.

Key checked-in artifacts:

- [submission.json](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/records/track_non_record_16mb/2026-04-03_DeepFloor/submission.json)
- [candidate_result_seed1337.json](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/records/track_non_record_16mb/2026-04-03_DeepFloor/candidate_result_seed1337.json)
- [train_seed1337.log](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/records/track_non_record_16mb/2026-04-03_DeepFloor/train_seed1337.log)
- [RESULTS.md](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/records/track_non_record_16mb/2026-04-03_DeepFloor/RESULTS.md)
- [PR_BODY.md](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/records/track_non_record_16mb/2026-04-03_DeepFloor/PR_BODY.md)

Workflow:

1. Run the local gate:

```bash
./run_local_suite.sh
```

2. Run the fixed comparison matrix locally or on a remote box:

```bash
./run_matrix.sh
```

3. Summarize the outputs:

```bash
./report.sh
```

4. On a small GPU box, reuse the same runner with `--device cuda`:

```bash
./run_smallbox_suite.sh
```

5. On a full 8-GPU box, run the larger end-to-end suite:

```bash
./run_fullbox_suite.sh
./report_fullbox.sh
```

6. Run the submission-shaped DeepFloor entrypoint from this folder:

```bash
./.venv/bin/python ./freeze_submission_snapshot.py
ENWIK8_PATH=/workspace/data/enwik8 \
OUTPUT_JSON=./train_result.json \
DEVICE=cuda \
TRAIN_STEPS=16 \
EVAL_BATCHES=8 \
python3 ./train_gpt.py
```

7. Run the submission preflight and build `submission.json` from a real result:

```bash
./.venv/bin/python ./freeze_submission_snapshot.py
./run_submission_preflight.sh
./.venv/bin/python ./build_submission_json.py --result-json ./train_result.json
```

8. Run the current checked-in submission candidate and produce `train_seed1337.log`, `candidate_result_seed1337.json`, and `submission.json`:

```bash
ENWIK8_PATH=/workspace/data/enwik8 \
PYTHON_BIN=python3 \
./run_submission_candidate.sh
```

9. Manage pod lifecycles with the lease-aware RunPod helpers:

```bash
./runpod_create_smallbox.sh
./runpod_start_smallbox.sh <pod-id>
./runpod_extend_smallbox.sh <pod-id>
./runpod_lease_status.sh [pod-id]
./runpod_sync_smallbox.sh <pod-id>
./runpod_stop_smallbox.sh <pod-id>
./runpod_bootstrap.sh
./runpod_delete_smallbox.sh <pod-id>

./runpod_create_fullbox.sh
./runpod_start_fullbox.sh <pod-id>
./runpod_extend_fullbox.sh <pod-id>
./runpod_sync_fullbox.sh <pod-id>
./runpod_stop_fullbox.sh <pod-id>
./runpod_bootstrap_fullbox.sh
./runpod_delete_fullbox.sh <pod-id>
```

Notes:
- Local scripts default to the repo `.venv`.
- If no `ENWIK8_PATH` is set, the suite runner generates a tiny byte fixture for smoke coverage.
- `freeze_submission_snapshot.py` vendors the current repo-root DeepFloor implementation into [deepfloor_snapshot.py](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/records/track_non_record_16mb/2026-04-03_DeepFloor/deepfloor_snapshot.py) so the submission entrypoint can run from this record folder without importing repo-root code.
- `train_gpt.py` imports the frozen [deepfloor_snapshot.py](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/records/track_non_record_16mb/2026-04-03_DeepFloor/deepfloor_snapshot.py), not the mutable repo root.
- On this checkout, prefer `./.venv/bin/python ./train_gpt.py` for local verification; the `python3` example is aimed at the prepared pod image where the dependencies are already present.
- `build_submission_json.py` counts both [train_gpt.py](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/records/track_non_record_16mb/2026-04-03_DeepFloor/train_gpt.py) and the frozen [deepfloor_snapshot.py](/Users/kennethmalloy/Local%20Documents/Developer/parameter-golf/records/track_non_record_16mb/2026-04-03_DeepFloor/deepfloor_snapshot.py) in `bytes_code`.
- `run_submission_preflight.sh` is the cheapest end-to-end gate for the record folder itself: compile, tiny run, metadata build, and `16,000,000`-byte cap check.
- The small-box script is intentionally conservative: it runs the same unit and smoke matrix gates before any longer end-to-end work.
- The fullbox script follows the same pattern, then fans out one `deepfloor-recipe-evolution` job per listed GPU using profile/seed pairs and writes per-job logs under `runs/fullbox/launch_logs`.
- `report_fullbox.sh` summarizes `smoke/`, `matrix/`, and `evolution/` together so we can review the whole 8-GPU run from one command.
- `runpod_create_smallbox.sh` and `runpod_start_smallbox.sh` always arm an auto-stop lease. By default they use `LEASE_MINUTES=120`; override with `--lease-minutes N` or `LEASE_MINUTES=N`.
- `runpod_create_fullbox.sh` and `runpod_start_fullbox.sh` do the same for 8-GPU pods with a longer default lease (`360` minutes create/start, `120` minutes extend) and larger default storage sizing.
- `runpod_extend_smallbox.sh` adds another lease window without restarting the pod, so overlapping leases are safe when an agent needs more time.
- `runpod_lease_status.sh` shows the tracked expiry window and active lease count for each leased pod.
- `runpod_sync_smallbox.sh` and `runpod_sync_fullbox.sh` pull the run artifacts back to the matching local `runs/` directory without stopping the pod.
- `runpod_stop_smallbox.sh` and `runpod_stop_fullbox.sh` are now harvest gates, not raw power switches: they rsync the remote run directory locally first and only stop the pod if that sync succeeds.
