# Parameter-Golf V8 WebSignal BPE Entropy MicroMix

Dataset repo: `8Planetterraforming/Parameter-Golf-V8-WebSignal-BPE-Entropy-MicroMix`

This repository is laid out as a **flat Hugging Face dataset repo**.

## File list

- `train.jsonl`
- `validation.jsonl`
- `test.jsonl`
- `train.txt`
- `validation.txt`
- `test.txt`
- `v8_micro_0p02pct.txt`
- `v8_micro_0p05pct.txt`
- `v8_micro_0p10pct.txt`
- `build_v8_micro_mix.py`
- `run_v8_seed42_probe.sh`
- `probe_plan.md`
- `dataset_design.md`
- `stats.json`
- `dataset_infos.json`
- `source_sanitization.md`
- `upload_to_hf.md`

## Plain-text artifacts

Use the flat text files directly:

- `train.txt`
- `validation.txt`
- `test.txt`
- `v8_micro_*.txt`

## Recommended micro-mix rates

- `0.02%`
- `0.05%`
- `0.10%`

## Probe pass condition

For seed-42 probing, keep the gate unchanged:

- seed42 must beat `1.08041364` before running a 3-seed proof.

## Notes

This dataset documentation does **not** claim this dataset already beats SOTA.
