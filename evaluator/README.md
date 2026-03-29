# Parameter Golf Evaluator Stack

This folder contains the v0 scored interface for Parameter Golf experiments.

The stack is:

1. `run_candidate.py`
   - executes one candidate family with a concrete environment/config

2. `evaluate_candidate.py`
   - wraps the run in a structured evaluator result
   - emits machine-readable JSON
   - only assigns a real score when artifact and runtime constraints pass
   - records evaluation legality metadata such as `evaluation_mode`, `uses_ttt`, and `legal_eval_ok`

3. `optuna_search.py`
   - uses the structured evaluator as the optimization interface

4. `bootstrap_skydiscover_records.py`
   - turns the curated record registry into a SkyDiscover-compatible checkpoint
   - intended as the first warm-start population for future EvoX runs

5. `runpod_remote_eval.py`
   - remote single-candidate evaluator intended to run on a GPU pod
   - uses plain Python, not `uv`, to reduce remote setup requirements

6. `runpod_client_eval.py`
   - ships a candidate script to a remote RunPod pod over SSH/SCP
   - invokes `runpod_remote_eval.py` remotely and returns the JSON result locally

7. `skydiscover_runpod_eval.py`
   - SkyDiscover-compatible `evaluate(program_path)` wrapper around the RunPod client
   - lets EvoX use the remote H100 pod as its evaluator

The first supported candidate family is:

- `exp01_mixed_export`

This is intentional. The current evidence says `exp01` is the only promising family so far.

## RunPod Path

If you want SkyDiscover or Optuna to evaluate candidates on a remote H100:

1. Provision the pod contents:

```bash
cd parameter-golf
bash scripts/provision_runpod_evaluator.sh
```

This syncs the local workspace tree to the pod instead of copying only a few evaluator files, so the remote side sees the same scripts and context you have locally.

2. Evaluate one candidate remotely:

```bash
cd parameter-golf
python evaluator/runpod_client_eval.py experiments/exp01_mixed_export/train_gpt.py
```

3. Use the pod directly from SkyDiscover:

```bash
PYTHONPATH=skydiscover uv run python -m skydiscover.cli \
  parameter-golf/evaluator/skydiscover_runpod_eval.py \
  --config parameter-golf/skydiscover_evox_smoke.yaml \
  --checkpoint parameter-golf/skydiscover_bootstrap_smoke/checkpoint_0 \
  --search evox \
  --iterations 20 \
  --output parameter-golf/skydiscover_runs/evox_runpod_20
```

## Why This Exists

The challenge is not just model quality. It is:

- pre-export quality
- post-roundtrip quality
- artifact size
- train time
- eval time

So the evaluator emits all of these, not just a single score.

Constraint handling:

- `artifact_ok`, `train_time_ok`, `eval_time_ok`, and `legal_eval_ok` are treated as hard constraints
- if a candidate finishes but violates any of them, the result status becomes `constraint_failed`
- `constraint_failed` candidates get a poison score so search cannot treat them as valid wins

Evaluation legality handling:

- default `evaluation_mode` is `standard_roundtrip`
- default `uses_ttt` is `false`
- default `legal_eval_ok` is `true` only for standard non-TTT evaluation
- TTT-style runs must declare themselves explicitly and set legality intentionally
- `optuna_search.py` rejects evaluator results whose eval legality is false or unknown

## Example

```bash
cd parameter-golf
uv run python evaluator/evaluate_candidate.py \
  --family exp01_mixed_export \
  --smoke
```

```bash
cd parameter-golf
uv run python evaluator/optuna_search.py --trials 8
```
