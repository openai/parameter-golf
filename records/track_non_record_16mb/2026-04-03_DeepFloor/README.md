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

6. Manage pod lifecycles with the lease-aware RunPod helpers:

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
- The small-box script is intentionally conservative: it runs the same unit and smoke matrix gates before any longer end-to-end work.
- The fullbox script follows the same pattern, then fans out one `deepfloor-recipe-evolution` job per listed GPU using profile/seed pairs and writes per-job logs under `runs/fullbox/launch_logs`.
- `report_fullbox.sh` summarizes `smoke/`, `matrix/`, and `evolution/` together so we can review the whole 8-GPU run from one command.
- `runpod_create_smallbox.sh` and `runpod_start_smallbox.sh` always arm an auto-stop lease. By default they use `LEASE_MINUTES=120`; override with `--lease-minutes N` or `LEASE_MINUTES=N`.
- `runpod_create_fullbox.sh` and `runpod_start_fullbox.sh` do the same for 8-GPU pods with a longer default lease (`360` minutes create/start, `120` minutes extend) and larger default storage sizing.
- `runpod_extend_smallbox.sh` adds another lease window without restarting the pod, so overlapping leases are safe when an agent needs more time.
- `runpod_lease_status.sh` shows the tracked expiry window and active lease count for each leased pod.
