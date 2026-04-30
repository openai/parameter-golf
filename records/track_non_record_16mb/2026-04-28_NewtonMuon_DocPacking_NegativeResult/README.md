# Newton-Muon × Document-Packing — Negative Result

**Result: technique strongly REGRESSES (+0.0378 nat) under PR #1874's document-packed loader.** Filed as a non-record submission with full diagnostic logs so other competitors don't burn compute on the same dead end.

> **What this submission is, in one paragraph.**
> An honest negative result. We tried to graft hook-based Newton–Schulz residual orthogonalization ("Newton-Muon") onto PR #1874's full stack as part of our weekend's exploration. It regressed val_bpb by +0.0378 nat in a controlled same-seed A/B. Rather than bury the result, we are filing it here with the full diagnostic logs and a root-cause analysis. The same-seed reproduction baseline — produced by the same `train_gpt.py` source with `NEWTON_MUON_ENABLED=0` — is included for direct comparison and is the same configuration as the reproduction in our companion record submission.

---

## TL;DR

| Config (seed=42, identical data and hyperparameters otherwise) | quantized + TTT-phased val_bpb |
|--------|-------------------------------:|
| PR #1874 baseline                                | **1.06928** |
| PR #1874 + `NEWTON_MUON_ENABLED=1`               | **1.10705** |
| **Δ**                                            | **+0.0378 nat (worse)** |

Both runs use identical seed, dataset, batch size, step count, and hardware. The only delta is the `NEWTON_MUON_ENABLED=1` env var.

---

## Root Cause

The Newton-Muon implementation we tested uses a **forward-pre-hook on every Linear module** to accumulate a per-module integer counter `_nm_K_count` and trigger Newton–Schulz preconditioning every K-th forward pass. This is incompatible with PR #1874's training pipeline because:

1. **Document packing produces variable `cu_seqlens` per step.** PR #1874's loader concatenates documents into mixed-length sequences and passes per-step `cu_seqlens` into FlashAttention 3. Each unique `cu_seqlens` shape is already one source of dynamo specialization.

2. **`_nm_K_count` is a Python int attribute.** `torch._dynamo` treats integer attributes on `nn.Module` as **static** — it specializes on their value. Every step the hook does `module._nm_K_count += n`, which dynamo sees as a new value, triggering a new graph specialization.

3. **The recompile limit is hit within ~10 steps.** On 8 ranks, each transformer block hits `config.recompile_limit=16` almost immediately. From [`train_nm_default.log`](train_nm_default.log):

   ```
   torch._dynamo hit config.recompile_limit (16)
       function: 'forward'
       last reason: 0/15: self._modules['blocks']._modules['0']._modules['attn']
                          ._modules['attn_gate_proj']._nm_K_count == 1474560
   ```

4. **Cascade failure.** Dynamo falls back to eager for the affected blocks → `fullgraph=True` is silently violated → FlashAttention 3 fused kernels stop emitting cleanly → step time inflates ~2.4× → fewer steps fit in the 600s budget → final val_bpb regresses by +0.0378 nat.

The PyTorch hint in the warning is on point:

> HINT: torch.compile considers integer attributes of the nn.Module to be static. If you are observing recompilation, you might want to make this integer dynamic using `torch._dynamo.config.allow_unspec_int_on_nn_module = True`, or convert this integer into a tensor.

We tried `allow_unspec_int_on_nn_module = True` early in development. It suppressed the recompile warnings but the underlying graph fragmentation persisted because dynamo still has to reconcile the variable-length `cu_seqlens` against a now-dynamic counter, and FA3's specialized paths get bypassed regardless.

---

## Why This Is Worth Filing

PR #1874's stack is currently the best public score on the SP8192 track in absolute terms (1.06766 single-seed; we reproduced it independently to 1.06907 single-seed and 1.06996 3-seed mean). Newton-Muon-style optimizers are a frequently-suggested next step in leaderboard discussions and in PR #1900's threads. Anyone porting Newton-Muon onto PR #1874 — or onto any base that uses a document-packing loader with variable `cu_seqlens` — will hit this same wall.

The fix is non-trivial. It probably requires moving the K-counter and the preconditioning trigger out of the compiled region entirely (e.g. into the `optimizer.step()` boundary, not a forward hook), which is enough of a redesign that we did not attempt it within our compute budget.

---

## What's Compatible

- **Static-shape forward** (single fixed seq_len, no document packing): the hook approach likely works there because `cu_seqlens` doesn't change and `_nm_K_count` becomes the only specialization source. We did not test this, but `allow_unspec_int_on_nn_module = True` should cover it.
- **Optimizer-step-based preconditioning** (not hook-based): triggered from `optimizer.step()`, runs outside any compiled region. The suggested fix above. We did not implement it.

---

## Reproduction

Identical to the record submission's reproduction steps; the only delta is the env var:

```bash
# Negative-result run (Newton-Muon enabled)
NEWTON_MUON_ENABLED=1 NEWTON_MUON_CAPTURE_EVERY=4 \
  SEED=42 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee train_nm_default.log

# Baseline (PR #1874 stack, NM disabled — identical seed, identical everything else)
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tee train_baseline_seed42.log
```

The diff between the two runs is exactly one env var.

---

## Files

- `README.md` (this file)
- `submission.json` — machine-readable metadata, including the diagnostic log lines
- `train_gpt.py` — PR #1874 source with the Newton-Muon graft (134 KB unwrapped, kept readable for the root-cause analysis)
- `train_nm_default.log` — full Newton-Muon run, val_bpb 1.10705
- `train_nm_smoke.log` — short capture run that surfaces the dynamo recompile diagnostics in detail
- `train_baseline_seed42.log` — PR #1874 baseline (NM disabled), val_bpb 1.06928. Identical seed and data — direct A/B comparison.
- `models/` — pre-trained `.int6.ptz` artifacts so a reviewer can eval-only without retraining (see "Eval-only verification" below).

### `models/` directory

| File | What it is | Size | Reported val_bpb |
|------|-----------|-----:|-----------------:|
| `models/nm_default.int6.ptz`              | Newton-Muon enabled, full 600s training, seed=42 | 15,928,150 B | 1.10705 |
| `models/nm_smoke.int6.ptz`                | Newton-Muon enabled, short 180s smoke run        | 15,943,987 B | (smoke; not the headline number) |
| `models/baseline_pr1874_seed42.int6.ptz`  | PR #1874 baseline, NM disabled, seed=42 (the A/B comparison artifact) | 15,921,161 B | 1.06928 |

These are checked into the submission so any reviewer can inspect the trained artifacts directly without having to retrain. Including model artifacts is not standard practice on this leaderboard; we're including them here because the value of a negative-result submission is "anyone can verify the failure mode," and shipping the artifacts gives the reviewer two independent ways to do that (re-run the script, or eval the shipped weights).

### How to use the shipped artifacts

The included `train_gpt.py` does **not** ship with an explicit `EVAL_ONLY` flag — its pipeline is `train → quantize → eval` end-to-end (the int6 artifact is written to `final_model.int6.ptz` near the bottom of training, then loaded and eval'd by the same process). To eval one of the shipped `.int6.ptz` artifacts without retraining, point the script at it via the `final_model.int6.ptz` filename it expects, and use the existing `deserialize(h, device)` helper at `train_gpt.py:2139`. A minimal harness looks like:

```python
# eval_shipped_artifact.py — sketch, not shipped
import shutil, train_gpt as TG
shutil.copy("models/nm_default.int6.ptz", "final_model.int6.ptz")
h = TG.Hyperparameters()  # same env-var-driven config as training
TG.set_logging_hparams(h)
device = TG.setup_distributed()
eval_model = TG.deserialize(h, device)
TG.run_sliding_eval(eval_model, h, device)         # sliding-window eval
TG.run_phased_ttt_eval(eval_model, h, device)      # phased TTT eval
```

(Function names follow PR #1874's structure; exact entrypoints may need adjusting against the source.) For most reviewers, **re-running `train_gpt.py` from scratch on a fresh seed=42 is the simpler verification path**, since the script is already wired end-to-end and the regression is large (+0.0378 nat) and stable. The artifacts in `models/` are primarily archival evidence of the runs that produced the reported logs.

### Direct inspection without GPUs (verified)

The `.int6.ptz` files are produced by PR #1874's `serialize()` (in `train_gpt.py`): a torch-saved `{"w": <quant_result>, "m": <quant_meta>}` dict, byte-shuffled with stride 2, then brotli-compressed. To read on CPU you need to undo those two steps in reverse:

```python
# verified on 2026-04-28 against models/nm_default.int6.ptz
import brotli, io, torch, numpy as np

_BSHF_MAGIC = b"BSHF"

def _byte_unshuffle(data):  # mirrors train_gpt.py:_byte_unshuffle
    if len(data) < 5 or data[:4] != _BSHF_MAGIC:
        return data
    stride = data[4]
    if stride < 2:
        return data[5:]
    payload = np.frombuffer(data, dtype=np.uint8, offset=5)
    n = len(payload)
    out = np.empty(n, dtype=np.uint8)
    src_off = 0
    for pos in range(stride):
        chunk_len = (n - pos + stride - 1) // stride
        out[pos::stride] = payload[src_off:src_off + chunk_len]
        src_off += chunk_len
    return out.tobytes()

with open("models/nm_default.int6.ptz", "rb") as f:
    raw = brotli.decompress(f.read())
state = torch.load(io.BytesIO(_byte_unshuffle(raw)), map_location="cpu", weights_only=False)

print(list(state.keys()))                          # ['w', 'm']
print(len(state["w"]), "quantized tensor entries") # 207
print(list(state["w"].keys())[:4])
# ['blocks.0.attn.c_q.weight.q',
#  'blocks.0.attn.c_q.weight.scale',
#  'blocks.0.attn.proj.weight.q',
#  'blocks.0.attn.proj.weight.scale']
print(list(state["m"].items())[:1])
# [('blocks.0.attn.c_q.weight', 'gptq (int6)')]
```

This is enough to confirm the artifacts are well-formed int6 GPTQ-quantized models with the expected layer structure on a laptop, no GPU required. The byte-shuffle step (`_byte_shuffle`/`_byte_unshuffle` at `train_gpt.py:1976-2002`) is part of PR #1874's compression pipeline, not something we added.

---

## Compute Cost of This Negative Result

~$12 of 8×H100 SXM time on RunPod (one full 600 s training run + one short 180 s smoke run + diagnostic capture). Posted publicly so the next person doesn't repeat it.

---

## Hardware

8 × H100 80 GB SXM (RunPod), PyTorch 2.9.1 + CUDA 12.8, FlashAttention 3 (`cu128_torch291` wheel).

---

## Acknowledgements

Thanks to **@AjAnubolu** for [PR #1874](https://github.com/openai/parameter-golf/pull/1874) (the base stack we grafted onto). Newton-Muon idea credit goes to the broader Newton–Schulz orthogonalization literature in the leaderboard discussion.

Submitted by:
- **Saicharan Ramineni** ([@GodlyDonuts](https://github.com/GodlyDonuts))
- csramineni@gmail.com
