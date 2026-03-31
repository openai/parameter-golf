# RunPod Multi-Pod Launch Plan

## Goal
Run multiple research lanes in parallel without mixing histories or objectives.

The recommended setup is:
- 2 always-on discovery pods
- 1 always-on storage/eval pod
- 1 on-demand promotion pod

This keeps cost contained while preserving real diversity in the search.

## Recommended Pods

### Pod 1: `core_discovery`
Purpose:
- continue the current search
- explore model shape, optimizer, schedule, recurrence, and tying

Command on pod:

```bash
cd /workspace/parameter-golf
BACKGROUND=1 ./run_lane.sh core_discovery
```

Exact env:

```bash
AUTORESEARCH_LANE=core
AUTORESEARCH_STAGE=discovery
AUTORESEARCH_NAMESPACE=core_discovery
EXPERIMENT_SECONDS=180
MAX_EXPERIMENTS=100
GPUS=1
VAL_LOSS_EVERY=0
```

Notes:
- seed this from the current best in `autoresearch/train_gpt.best.py`
- this is the direct successor to your current runs

### Pod 2: `storage_discovery`
Purpose:
- search quantization-aware and compression-aware ideas
- accept small BPB regressions only if artifact bytes improve materially

Command on pod:

```bash
cd /workspace/parameter-golf
BACKGROUND=1 ./run_lane.sh storage_discovery
```

Exact env:

```bash
AUTORESEARCH_LANE=storage
AUTORESEARCH_STAGE=discovery
AUTORESEARCH_NAMESPACE=storage_discovery
EXPERIMENT_SECONDS=180
MAX_EXPERIMENTS=50
GPUS=1
VAL_LOSS_EVERY=0
MAX_QUANTIZATION_GAP=0.08
STORAGE_MAX_REGRESSION=0.003
STORAGE_MIN_SIZE_IMPROVEMENT=250000
```

Notes:
- this lane should prefer codebook, export, or quantization-friendly changes

### Pod 3: `eval_time_discovery`
Purpose:
- search cache, copy, pointer, and dynamic-evaluation ideas
- reject anything that improves BPB only by exploding eval time

Command on pod:

```bash
cd /workspace/parameter-golf
BACKGROUND=1 ./run_lane.sh eval_time_discovery
```

Exact env:

```bash
AUTORESEARCH_LANE=eval_time
AUTORESEARCH_STAGE=discovery
AUTORESEARCH_NAMESPACE=eval_time_discovery
EXPERIMENT_SECONDS=180
MAX_EXPERIMENTS=50
GPUS=1
VAL_LOSS_EVERY=0
MAX_EVAL_TIME_MS=60000
```

Notes:
- keep this separate from `core`
- treat eval wallclock as a hard budget

### Pod 4: `core_promotion` (on demand)
Purpose:
- re-test only the strongest candidates on longer 1xH100 runs
- reduce short-horizon false positives before 8xH100 work

Command on pod:

```bash
cd /workspace/parameter-golf
BACKGROUND=1 ./run_lane.sh core_promotion
```

Exact env:

```bash
AUTORESEARCH_LANE=core
AUTORESEARCH_STAGE=promotion
AUTORESEARCH_NAMESPACE=core_promotion
EXPERIMENT_SECONDS=600
MAX_EXPERIMENTS=20
GPUS=1
VAL_LOSS_EVERY=0
```

Notes:
- do not keep this pod running full time at first
- turn it on when discovery finds something worth validating

## Recommendation

Start now:
- `core_discovery`
- `storage_discovery`
- `eval_time_discovery`

Start later:
- `core_promotion`

Do not start yet:
- `representation_discovery`

Reason:
- the representation lane still needs a correctness harness before it can safely keep wins

## Cost Envelope

At the currently observed rate of about `$2.69/hr` for a 1xH100 pod:
- 3 always-on discovery pods cost about `$8.07/hr`
- a temporary promotion pod adds about `$2.69/hr`

## Seeding Strategy

All launched lanes should be seeded from the current best:
- `autoresearch/train_gpt.best.py`
- `autoresearch/history.jsonl`

`run_lane.sh` already does this by:
- copying the best file into the namespace if needed
- reading the best kept BPB from the seed history and exporting it as `BASELINE_BPB`

## Namespace Layout

Each pod writes to its own namespace:

```text
autoresearch/core_discovery/
autoresearch/storage_discovery/
autoresearch/eval_time_discovery/
autoresearch/core_promotion/
```

Each namespace gets:
- `history.jsonl`
- `experiments/`
- `logs/`
- `train_gpt.best.py`
- `autoresearch.out` when launched in background

## Suggested Launch Order

1. Sync the latest repo to each pod.
2. Run setup once if the pod is fresh:

```bash
cd /workspace/parameter-golf
bash setup_runpod.sh
```

3. Launch the lane:

```bash
cd /workspace/parameter-golf
BACKGROUND=1 ./run_lane.sh core_discovery
```

4. Tail logs:

```bash
tail -f /workspace/parameter-golf/autoresearch/core_discovery/autoresearch.out
```

## Suggested Promotion Rule

Promote a discovery result to `core_promotion` only if at least one is true:
- it is a clear BPB win and stays comfortably under the byte cap
- it improves both BPB and quantization gap
- it wins repeatedly across nearby settings
- it has a clear mechanism that should scale to longer runs

## Suggested 8xH100 Rule

Promote to full `8xH100 / 600s` validation only after:
- it survives `core_promotion`
- it still looks good post-quantization
- it has no obvious throughput or eval-time risk
