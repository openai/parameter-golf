# DeepFloor

This record folder is the execution surface for the first DeepFloor build-out.

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

8. Manage pod lifecycles with the lease-aware RunPod helpers:

```bash
./runpod_create_smallbox.sh
./runpod_start_smallbox.sh <pod-id>
./runpod_extend_smallbox.sh <pod-id>
./runpod_lease_status.sh [pod-id]
./runpod_stop_smallbox.sh <pod-id>
./runpod_bootstrap.sh
./runpod_delete_smallbox.sh <pod-id>

./runpod_create_fullbox.sh
./runpod_start_fullbox.sh <pod-id>
./runpod_extend_fullbox.sh <pod-id>
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
