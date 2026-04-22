# TTT LoRA Rank Sweep

**Status:** candidate — ready to spec
**Created:** 2026-04-22
**Expected Δ:** unknown — never tested
**Cost:** eval-only, ~$5–10

---

## TTT Frontier State (as of 2026-04-22)

### What the competition has established

**Phase 1 — Score-first post-quant SGD (merged era)**
PR #461 introduced the legal framework: score a chunk first, then train on it with SGD. PR #549 merged this (1.1194). PR #1493 (our baseline ancestor, 1.0810) inherited it. Gains ~−0.002 bpb.

**Phase 2 — Per-document LoRA (what we use)**
PR #1530 (samacqua): replace global SGD with per-doc LoRA adapters. Fresh LoRA per document, train on early chunks, predict later chunks. −0.008 nats improvement. This is the mechanism in our #1736 baseline.

**Phase 3 — Phased global SGD on top**
PR #1610 (romeerp) + PR #1626 (dexhunter): global SGD pass on base model between LoRA phases. Small additional gains ~−0.0008 bpb. This gives us `PHASED_TTT_NUM_PHASES=3, PREFIX_DOCS=2000`.

**Phase 4 — Pre-quant TTT (illegal)**
PR #1735/#1738/#1758: adapt full-precision model on all val data before quantizing. Claimed −0.054 bpb. **Ruled illegal** — Issue #1017 Track B explicitly bans "any mechanism whose useful state is built from evaluation tokens, including test-time training." Even the author of #1735 (AjAnubolu) conceded in PR comments: "The Track B language most likely covers this even with a frozen post-GPTQ artifact, since the frozen weights were themselves built from val data." Treat as DQ'd.

### What has never been ablated (legitimate space)

| Parameter | Current value | Never tested |
|---|---|---|
| `TTT_LORA_RANK` | 96 | 32 / 64 / 128 / 256 / 512 |
| `TTT_CHUNK_SIZE` | 48 | 16 / 32 / 96 |
| `PHASED_TTT_PREFIX_DOCS` | 2000 | 500 / 1000 / 3000 |
| `TTT_GRAD_STEPS` | 1 | 2 / 3 |
| `TTT_EVAL_SEQ_LEN` | 2048 | 1024 / 4096 |

The entire legitimate TTT hyperparam space was abandoned because everyone chased pre-quant. It's wide open.

---

## Key finding: pass-specific LoRA is already implemented

An earlier version of this file proposed adding pass-specific LoRA adapters as a new idea. **After reading the code, this is already the default behavior.**

`BatchedTTTLoRA` computes `num_slots` as:
```python
if getattr(model, "looping_active", False):
    num_slots = len(model.encoder_indices) + len(model.decoder_indices)
```

`encoder_indices` contains the full unrolled sequence including repeats:
```
[0, 1, 2,  3, 4, 5,  3, 4, 5,  3, 4, 5,  6, 7, ...]
 pre-loop  pass-1   pass-2   pass-3   post-loop
```

And `forward_ttt` uses a simple incrementing `slot` counter — so each position in the unrolled sequence gets its own dedicated LoRA adapter. Each recurrence pass through each looped layer already has a unique adapter that receives unique gradients from the backward pass through the full recurrent computation graph.

**The idea as originally conceived is already there. No code change needed.**

---

## The actual open question: rank

Since pass-specific adapters already exist, the real unexplored variable is **rank**. Every clean PR in the competition has used rank ~64–96 without ever sweeping it.

Two competing hypotheses:

**Hypothesis A — rank is too low:**
Each pass-specific adapter trains on ~1/3 of the document's chunks (only the chunks where that pass runs). With limited per-document context (documents are short), rank 96 may not have enough capacity to capture the adaptation signal specific to each pass's dynamics.

**Hypothesis B — rank is too high:**
With per-doc reset and short documents, each adapter trains on very few gradient steps. High rank = more parameters per gradient step = potential overfitting or noisy updates within a document. Lower rank (32–64) might generalize better.

Neither has been tested. The optimal rank is unknown.

### What the α values suggest

From spec 017's learned α endpoint:
```
Pass-1 (first recurrence):  [1.078, 1.273, 1.430]  # amplify, grows with depth
Pass-2 (second recurrence): [1.016, 0.965, 0.832]  # dampen, shrinks with depth
```

Pass-1 and pass-2 through the same layer are doing qualitatively different things. The pass-specific adapters should in principle be learning different corrections for each. Whether rank 96 is enough (or too much) to learn these corrections cleanly from per-document context is the open question.

---

## Screening plan

**Eval-only — no training needed.** Reuse spec 008's `final_model.int6.ptz`.

Sweep: `TTT_LORA_RANK` ∈ {32, 64, 96, 128, 256}

Run each as a standalone eval (no pod training, just load checkpoint → TTT → report post-TTT bpb). Current rank 96 is the control.

Secondary sweep if rank signal is found: `TTT_GRAD_STEPS` ∈ {1, 2, 3} at the winning rank.

**Cost:** ~$2–3 per eval run × 5 rank values = ~$10–15 total. All parallelizable on the same pod.

---

## Risks

- **Null result:** rank may be in a flat region around 96 and no value moves the needle. Informative but no gain.
- **Interaction with checkpoint:** the optimal rank for spec 008's checkpoint may differ from spec 021's (recur-alpha changes hidden state dynamics). Worth re-running the winner on 021's checkpoint if it promotes.

---

## Next steps

1. Run rank sweep (eval-only) on spec 008 checkpoint
2. If any rank beats 96 by ≥ 0.0005 bpb: re-run on 021's checkpoint to confirm
3. If signal found: freeze rank as a config change, stack onto next full pipeline run
