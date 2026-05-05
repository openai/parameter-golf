# Legality Audit - BIJEPAX-lite

## Verdict

Current read: **likely legal/submittable**, assuming the existing CaseOps byte-sidecar/PPM lane is accepted.

The BIJEPAX-lite addition itself is low-risk because it is training-only and has no evaluation-time access to future validation tokens.

## Challenge rules checked

From the repository README:

- Submission artifact size is code bytes plus compressed model bytes.
- The cap is strict decimal `16,000,000` bytes.
- Evaluation may not use training data unless paid for inside the artifact.
- Validation data may not be used during training.
- Evaluation must complete within 10 minutes on 8xH100, separate from the 10-minute training cap.
- Test-time methods must score before updating on validation tokens.

## Artifact size

Seed 42:

- `Serialized model quantized+pergroup: 15955181 bytes`
- `Total submission size quantized+pergroup: 15997180 bytes`
- Strict cap: `16000000 bytes`
- Headroom: `2820 bytes`

This is tight but under cap.

`LQER_TOP_K=1` was used specifically to create byte headroom. Earlier BIJEPA without this trim packaged at `16,004,902` bytes and was not submittable.

## Training-only JEPA auxiliary

Relevant implementation:

- `class MultiDirectionalBiJEPAX`
- `def bijepax_weight_at`
- `train_model(...): bijepax_module = MultiDirectionalBiJEPAX(...)`
- `step_fn(...): loss = ce_loss + bijepax_module(hidden, ...)`

The predictor module is created outside `base_model`:

```python
bijepax_module = MultiDirectionalBiJEPAX(...).to(device).bfloat16()
bijepax_opt = torch.optim.Adam(bijepax_module.parameters(), ...)
```

It is not assigned as a child module of `base_model`, so `base_model.state_dict()` does not contain BIJEPAX predictor weights.

Serialization only saves `base_model.state_dict()`:

```python
torch.save(base_model.state_dict(), h.model_path)
sd_cpu = _unbank_state_dict(base_model.state_dict(), h.num_layers)
```

So the JEPA predictor heads are not present in the final artifact.

## No validation leakage during training

Training batches come from `DocumentPackingLoader(h, device)`.

Validation data is loaded for periodic/terminal validation, but the BIJEPAX training loss only uses hidden states from training microbatches:

```python
x, y, cu_seqlens, _max_seqlen = train_loader.next_batch(...)
ce_loss, hidden = forward_with_hidden(x, y, ...)
loss = ce_loss + bijepax_module(hidden, ...)
```

The BIJEPAX module does not read validation tokens, validation bytes, or validation sidecars.

## Evaluation path

Final score uses the existing PPM sliding evaluator:

- `eval_val_ppm_sliding`
- `ppm_mixer val_bpb`
- `ppm_sliding val_loss / val_bpb`

The PPM mixer operates score-before-update over the scored target stream. The implementation computes neural log probabilities first, then the byte mixer walks bytes in order and updates its tables after scoring each byte.

Legality risk is therefore concentrated in whether reviewers accept this existing PPM/CaseOps scoring lane, not in BIJEPAX-lite.

## Cross-document leak check

The SmearGate cross-document leak fix is present in both hidden and TTT paths:

```python
not_bos = (input_ids[:, 1:] != BOS_ID).to(x.dtype).unsqueeze(-1)
x = torch.cat([x[:, :1], x[:, 1:] + g * x[:, :-1] * not_bos], dim=1)
```

TTT is disabled for this candidate (`TTT_ENABLED=0`), but the symmetric fix is still present.

## Eval compile

The run uses `DISABLE_COMPILE=1`. Post-serialize evaluation also honors this:

```python
if os.environ.get("DISABLE_COMPILE", "0") == "1":
    log("eval_compile:disabled_by_env")
    compiled_model = eval_model
    compiled_forward_logits = eval_model.forward_logits
```

This avoids the compile stall encountered in the first BIJEPAX attempts.

## Risks / reviewer-facing caveats

- The artifact headroom is only `2820` bytes on seed 42. Do not add substantial code unless compression is rechecked.
- The PR should avoid unverifiable claims such as "BiJEPA proved 4x better on chaotic systems" unless the exact source is provided.
- The submission should clearly say the JEPA module is an auxiliary training regularizer, not an eval-time bidirectional predictor.
- If the competition reviewers consider the PPM/CaseOps byte-sidecar lane non-compliant, this candidate inherits that risk.
