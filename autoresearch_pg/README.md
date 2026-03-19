# autoresearch_pg

`autoresearch_pg` is a thin, Parameter-Golf-specific fork of the `autoresearch`
idea. It does not replace the existing training stack. Instead, it wraps the
current repository's [train_gpt.py](../train_gpt.py) in a candidate workspace,
run harness, and scoring pipeline that optimize the challenge's real objective:

- valid `final_int8_zlib_roundtrip_exact val_bpb`
- total artifact bytes under `16_000_000`
- reproducible tiered promotion from cheap proxies to full `8xH100` runs

The first version is intentionally conservative:

- the agent edits only candidate-local `train_gpt.py`
- tokenizer search is documented but not yet automated
- every run is captured in `autoresearch_pg/runs/`
- promoted runs can be materialized into `records/`

## Layout

```text
autoresearch_pg/
  config/
  lib/
  scripts/
  candidates/
  runs/
  state/
  README.md
  program.md
```

## Basic Flow

1. Bootstrap a candidate from the current root `train_gpt.py`.
2. Let the agent edit only `autoresearch_pg/candidates/<candidate_id>/train_gpt.py`.
3. Run `smoke_local`, then `proxy_1gpu_fast`, then `proxy_1gpu_full`, then `track_8gpu_600s`.
4. Score using post-quant `val_bpb`, bytes, and quantization gap.
5. Convert a promoted full run into a `records/...` folder.

## Commands

Bootstrap a fresh candidate:

```bash
python3 autoresearch_pg/scripts/bootstrap_candidate.py --candidate baseline_copy
```

Run a candidate on the local smoke tier:

```bash
python3 autoresearch_pg/scripts/run_candidate.py --candidate baseline_copy --tier smoke_local
```

Check whether a candidate clears a promotion bar and optionally launch the next tier:

```bash
python3 autoresearch_pg/scripts/promote_candidate.py \
  --candidate baseline_copy \
  --from-tier proxy_1gpu_fast \
  --to-tier proxy_1gpu_full
```

Render the internal leaderboard:

```bash
python3 autoresearch_pg/scripts/render_leaderboard.py
```

List the live template registry:

```bash
python3 autoresearch_pg/scripts/ingest_template.py --list
```

Create or update one template:

```bash
python3 autoresearch_pg/scripts/ingest_template.py \
  --template-id recipe_fast_decay_v2 \
  --family training_recipe \
  --template-type env_template \
  --env WARMUP_STEPS=64 \
  --env WARMDOWN_ITERS=256 \
  --env MATRIX_LR=0.05
```

Force a template-backed portfolio cycle:

```bash
python3 autoresearch_pg/scripts/run_portfolio.py \
  --tier smoke_mlx_local \
  --template recipe_fast_decay
```

Create a record-style folder from the best valid full run:

```bash
python3 autoresearch_pg/scripts/make_record.py \
  --candidate baseline_copy \
  --tier track_8gpu_600s \
  --track track_non_record_16mb \
  --name MyCandidate \
  --author "Your Name" \
  --github-id yourgithub
```

## Design Notes

- Candidate runs are isolated. The root `train_gpt.py` is never edited by the harness.
- The score is challenge-native. The best candidate is the one with the lowest
  valid post-quant `val_bpb`, not the lowest pre-export loss.
- Promotion is cheap-first. The scaffold is designed to avoid wasting `8xH100`
  time on ideas that fail on smoke or `1xH100` proxies.

## Next Steps

The current scaffold leaves room for:

- tokenizer search and dataset re-export wrappers
- agent-generated candidate notes and hypothesis tracking
- richer promotion rules
- multi-run statistical promotion instead of single-run gating
- reusable research templates and template ingestion
- template-guided portfolio cycles
