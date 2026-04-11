# Non-Record Submission: Random-Map Adapter Train/Eval Ablations

This folder captures an 8xH100 ablation study around low-parameter adapter mechanisms added to the stock 9-layer/512-dim baseline. The code snapshot adds two related ideas:

1. train-time `random_diag` adapters on `q`, `v`, and `lm_head`
2. document-aware evaluation adapters for test-time training, with both LoRA and random-map variants

The main result is negative-but-useful: in this setup, the **LoRA TTT control** remained the best performer at **1.1917 val_bpb**, while the best fully random-map condition (**random-map train + random-map TTT**) reached **1.2008 val_bpb**. Both counted artifacts stayed under the 16MB cap.

## Best Result

- Best overall score: **1.1917 val_bpb** (`lora_ttt`)
- Best random-map score: **1.2008 val_bpb** (`random_train_ttt`)
- Counted artifact size (best run): **15,883,798 bytes**
- Steps reached (best run): **13,507**
- Hardware: **8xH100**
- Wallclock cap: **600s**

## What Changed in `train_gpt.py`

- Added train-time random-map adapters via `TRAIN_ADAPTER_KIND=random_diag`
- Added batched document-aware TTT evaluation with:
  - `TTT_ADAPTER_KIND=lora`
  - `TTT_ADAPTER_KIND=random_diag`
- Targeted the same adapter sites across methods:
  - attention `q`
  - attention `v`
  - `lm_head`
- Used rank-8 adapters for all ablations
- Used chunked score-first document evaluation (`TTT_CHUNK_SIZE=256`, `TTT_EVAL_SEQ_LEN=1024`, `TTT_BATCH_SIZE=64`)

## Ablation Results

| Ablation | Train adapter | Eval adapter | Roundtrip val_bpb | Final TTT val_bpb | Notes |
|---|---|---:|---:|---:|---|
| `baseline` | none | none | 1.2268 | — | Reference run; final roundtrip validated cleanly |
| `lora_ttt` | none | LoRA | 1.2267 | **1.1917** | Best overall |
| `random_train` | random-map | none | 1.2310 | — | Train-time random-map adapters alone hurt roundtrip score |
| `random_train_ttt` | random-map | random-map | 1.2330 | **1.2008** | Best random-map condition |
| `random_ttt` | none | random-map | — | — | Interrupted before completion |

Useful deltas:

- `lora_ttt` vs baseline roundtrip: **-0.0351 bpb**
- `random_train_ttt` vs baseline roundtrip: **-0.0260 bpb**
- `random_train_ttt` vs `lora_ttt`: **+0.0091 bpb**
- `random_train` vs baseline roundtrip: **+0.0042 bpb**

## Interpretation

The random-map parameterization was viable enough to support meaningful document-time adaptation, but it did **not** beat the LoRA control in this implementation. The strongest random-map result came from combining train-time and eval-time random-map adapters, suggesting the representation is at least usable, but the train-time random-map adapters by themselves were slightly harmful.

In short:

- document-aware TTT is the main win here
- LoRA remained the stronger adapter family
- random-map adapters are interesting, but not yet competitive with the LoRA control

## Comparison to Current Repo SOTA

From the root `README.md` at the time of writing:

- current 10-minute leaderboard SOTA: **1.1194**
- historical LoRA TTT submission: **1.1928**
- notable unlimited-compute non-record baseline: **1.2074**

So this ablation set:

- is **not** competitive with the current SOTA record
- slightly improves on the earlier public LoRA TTT result (**1.1917 vs 1.1928**)
- comfortably beats the previously listed 4-hour non-record baseline (**1.1917 vs 1.2074**), though that run targeted a different non-record setting

## Known Issues / Negative Results

- The `TTT_ADAPTER_KIND=none` path still calls the document-aware eval function at the end of the script, which crashes after the clean roundtrip validation lines because `adapter=None` returns a scalar loss instead of per-token losses. This affected `baseline` and `random_train`, but their counted roundtrip metrics were already emitted before the crash.
- The `random_ttt` run was interrupted before completion, so there is no final scored result for that condition in this folder.

## Reproduction

Run all ablations from this directory:

```bash
python run_all_ablations.py
```

Or a single condition:

```bash
python run_all_ablations.py --only lora_ttt
python run_all_ablations.py --only random_train_ttt
```

The wrapper launches:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

with a default wallclock cap of `600.0` seconds and the environment overrides defined in `run_all_ablations.py`.

## Included Files

- `train_gpt.py` — code snapshot for this ablation study
- `run_all_ablations.py` — wrapper used to launch the ablations
- `plot_ablation_losses.py` — summary/plot generation script
- `ablations_summary.tsv` — parsed summary of the completed runs
- `logs/*.log` — raw ablation logs
- `submission.json` — submission metadata
