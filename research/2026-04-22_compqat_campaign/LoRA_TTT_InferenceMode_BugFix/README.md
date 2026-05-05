> This is a supporting research note, not a Parameter Golf leaderboard submission.  
> It documents an inference-mode / TTT interaction found during experimentation and is included for context from the same campaign.

# Non-Record Engineering Contribution: `torch.inference_mode` Breaks Score-First LoRA-TTT — A Bug Report and Fix

**Author:** yevh ([@yevh](https://github.com/yevh)) | **Contribution type:** engineering bug fix + integration code | **Track:** non-record

---

> **Context.** This is a standalone engineering contribution carved out from a larger Parameter Golf research campaign. It documents a real, general PyTorch footgun that anyone implementing score-first Test-Time Training with a rotary positional encoding cache will hit — and provides the one-character fix.
>
> **Full research narrative where this bug was discovered** (building LoRA-TTT on top of [PR #1493](https://github.com/openai/parameter-golf/pull/1493)'s stack): [Blog post — OpenAI Parameter Golf: what I built, what worked, what didn't](./article.md).

---

## The bug in one line

`torch.inference_mode()` silently breaks score-first Test-Time Training in any transformer that uses a buffered rotary positional embedding cache. The first forward pass that runs inside `inference_mode` poisons the rotary cache buffer; every subsequent forward pass that touches that cache inherits the poison, and the backward pass fails with a cryptic error.

Error signature:

```
RuntimeError: Cannot set version_counter for inference tensor
```

Fix (one call-site change):

```diff
-with torch.inference_mode():      # poisons rotary cache buffer
+with torch.no_grad():             # safe; same semantics for our purposes
    score_phase_forward(...)
```

---

## Why this matters

The score-first TTT protocol ([PR #549](https://github.com/openai/parameter-golf/pull/549) by @abaybektursun) is the only eval-time adaptation the competition accepts as "legal." It has a strict structure:

1. **Score** a chunk under a gradient-disabling context, accumulating validation loss
2. **Train** trainable parameters on those same tokens (the model is allowed to update *after* scoring, not before)
3. **Move on** — never re-score a chunk after training on it

PyTorch's idiomatic "no-gradient" context is `torch.inference_mode()`. It's documented as stricter than `torch.no_grad()` — it not only disables gradient tracking but also blocks many mutating operations. The natural choice for the score phase. And it's what I chose on my first implementation.

The failure doesn't surface immediately. The score phase completes cleanly. The problem appears a few steps into the *train* phase, when the model performs a forward pass that touches the rotary cache, and the backward pass crashes.

## What's actually happening

The rotary positional embedding module in a typical modern transformer ([PR #1493](https://github.com/openai/parameter-golf/pull/1493)'s implementation follows this pattern) keeps a cos/sin lookup table as a buffer. The table is populated lazily — on the first forward pass that requests a given sequence length, the module computes cos/sin for that length and caches it:

```python
class Rotary(nn.Module):
    def forward(self, seq_len, device, dtype):
        if self._cos_cached is None or self._seq_len_cached != seq_len:
            # compute cos, sin here — tensors are created during forward pass
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
        return self._cos_cached, self._sin_cached
```

`torch.inference_mode()` marks tensors *created inside the context* as "inference tensors" — a special flag that permanently disqualifies them from autograd. When the first forward pass that populates `_cos_cached` runs inside `inference_mode`, the cached cos/sin tensors acquire this flag and keep it after the context exits.

Every subsequent forward pass — including ones in the *train* phase, where autograd is active and gradients are needed — uses these poisoned cached tensors. The backward pass then fails because it cannot propagate gradients through inference-mode tensors.

## The fix

`torch.no_grad()` disables gradient tracking without tainting tensors created inside it. It preserves the score-first protocol's legality guarantees identically:
- No gradients computed during scoring
- Weights cannot be updated by a non-existent backward pass
- No leakage from training to scoring (model weights are the same before and after)

Only difference: no "inference tensor" flag on tensors created inside. So the rotary cache, once populated during the score phase, is a normal tensor that the train-phase backward pass can work with.

## Where this was observed and how to verify the fix

**The bug surfaced on**: 8×H100 SXM, PyTorch **2.9.1** + CUDA 12.8, `flash_attn_3` attention kernels, [PR #1493](https://github.com/openai/parameter-golf/pull/1493)'s Rotary module, during Phase 2 of my research campaign (April 22, 2026). The crash happened consistently in the chunk-2 backward pass of the LoRA-TTT eval after the chunk-1 score phase populated the rotary cache under `torch.inference_mode()`. Full cloud log is in the linked research repo.

**On newer PyTorch (2.11+) the bug may not reproduce** in a minimal CPU example — it's possible recent PyTorch versions have softened the inference-tensor tainting rules, or that the specific combination of `flash_attn_3` + `torch.compile` on CUDA is required to trigger it. Whichever way: the `torch.no_grad()` form is *always* correct for score-first TTT, and is the form [PR #549](https://github.com/openai/parameter-golf/pull/549) and subsequent TTT-integrating records should use going forward. The `inference_mode()` form is *sometimes* correct and *sometimes* wrong depending on the PyTorch version and model — which is exactly the shape of a footgun worth documenting.

**To verify the fix works in a full LoRA-TTT integration**: run [`test_lora_ttt_integration.py`](test_lora_ttt_integration.py). It attaches LoRA adapters to a CPU mini-transformer, runs the score-first protocol end-to-end (score with `no_grad`, train LoRA adapter parameters, verify base weights unchanged), and checks that hook cleanup + `requires_grad` restoration work correctly. Takes ~5 seconds.

## Where this code landed

The fix is integrated into a larger Parameter Golf research submission: the native `eval_val_lora_ttt` function in the accompanying submission's [`train_gpt.py`](../2026-04-22_SP8192_CompQAT_PR1493/train_gpt.py) uses `torch.no_grad()` correctly. The LoRA-TTT mechanism itself — low-rank adapters on attention projections, updated chunk-by-chunk under the score-first protocol — is also integrated there, though not exercised in the submitted 3-seed runs (which use [PR #549](https://github.com/openai/parameter-golf/pull/549)'s full-parameter TTT for cross-recipe consistency).

The LoRA-TTT integration ships with an end-to-end CPU test (`tests/test_lora_ttt_integration.py` in the research repo) that would have caught this bug — and a related wrapper-signature bug — in under a second. The test is included here as [`test_lora_ttt_integration.py`](test_lora_ttt_integration.py) for reference.

## Scope and honesty

What this contribution is:

- A real, general PyTorch footgun with a real fix
- Working, tested LoRA-TTT integration code that ships with a CPU integration test
- Reproducible in a minimal setting

What this contribution is *not*:

- A head-to-head 8×H100 comparison of LoRA-TTT vs full-parameter TTT (I did not run this — my submitted 3-seed runs all use the full-parameter TTT from [PR #549](https://github.com/openai/parameter-golf/pull/549) for consistency). Whether LoRA-TTT beats full-parameter TTT at frontier scale remains an open empirical question.
- A claim about BPB improvement from the LoRA-TTT mechanism itself — this PR is about correctness of the score-first protocol under LoRA adapters, not about whether the adapters win.

Earlier exploratory evidence on a weaker base (SP1024, 1×H100) showed LoRA-TTT delivering a −0.034 BPB lift, stronger than full-parameter TTT at the same scale. But SP1024 is not SP8192, and 1×H100 is not 8×H100. Until someone runs the head-to-head at frontier scale (~$10 of cloud compute), don't infer anything about the mechanism's competitive value from this PR.

## Attribution

- Score-first TTT legal framework: [@abaybektursun ([PR #549](https://github.com/openai/parameter-golf/pull/549))](https://github.com/openai/parameter-golf/pull/549)
- LoRA adaptation base concept: [@samacqua ([PR #116](https://github.com/openai/parameter-golf/pull/116))](https://github.com/openai/parameter-golf/pull/116)
- Rotary cache pattern that surfaces this bug: standard across most PRs from #287 onward

## Files in this directory

- `README.md` — this document
- `test_lora_ttt_integration.py` — full integration test on a CPU mini-model. Exercises the score-first protocol end-to-end with `no_grad` (the correct form). Verifies base weights unchanged, hooks cleaned up, requires_grad restored. ~5 seconds.
