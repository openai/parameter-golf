# Static Audit Notes

## Network / external access

Search terms checked in `train_gpt_v15_bijepax.py`:

```text
requests, urllib, http, https, wget, curl, socket, huggingface, datasets, load_dataset
```

No network download path appears in the script.

The only `subprocess.run` uses found are:

- `lrzip` compression/decompression for the local model artifact.
- optional `pyminify` code minification fallback for the code wrapper.

These are local tool invocations, not network calls.

## Data access

Training:

- `DocumentPackingLoader` reads `fineweb_train_*.bin`.
- GPTQ calibration uses `ShuffledSequenceLoader`, also training data.
- BIJEPAX-lite loss reads only hidden states from training batches.

Validation/evaluation:

- `ValidationData` reads `fineweb_val_*.bin`.
- CaseOps reads `fineweb_val_bytes_*.bin` for byte accounting.
- PPM sliding evaluation reads validation tokens only during final evaluation.

No code path was found where BIJEPAX-lite trains on validation data.

## TTT paths

The file contains TTT utilities, but this submission sets:

```text
TTT_ENABLED=0
```

So TTT paths are not used in the submitted BIJEPAX-lite score.

## Serialization

The final compressed model is built from `base_model.state_dict()`.

BIJEPAX-lite predictor heads are held in `bijepax_module`, not as `base_model` children. Therefore they are not saved into `final_model.pt` or the int6 artifact.

## Current concern list

- The strict artifact cap headroom is very small. Seed 42 headroom was `2820` bytes.
- `lrzip` must exist in the evaluation environment. This matches the per-group compressor lane already used in the current stack.
- Claims about "BiJEPA 4x chaotic systems" remain unverified and should not go into the PR.
- If we submit a record PR, we need the seed 314 and 999 logs plus mean/std before finalizing `submission.json`.
