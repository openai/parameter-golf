---
title: Hail Mary Variant V15 — Tensor-Gated STE (The "Did QAT Ever Turn On?" Test)
date: 2026-04-19
pod: right_copper_warbler (8×A100 SXM, $11.95/hr)
author: Claude Cowork (synthesis of V4 source reading + GPT DeepResearch Q-match lane + Kimi swarm STE-failure mode)
status: primary recommendation for tonight's remaining ~3 runs
---

# TL;DR

V4's QAT branch may never have executed. Not "underperformed" — **never ran**. The class-level `_qat_enabled: bool` flag is read inside a `torch.compile(fullgraph=True, dynamic=False)` graph; at trace time the flag is `False`, so the STE branch is dead-code-eliminated before training starts. Flipping `CastedLinear._qat_enabled = True` at step 1000 mutates a Python class attribute that the compiled graph no longer references. The `qat:activated step:1000` log line fires, but nothing downstream changes.

If this is correct, V4's banked 1.1803 BPB is a *V3-with-QAT-route-logging* number, not a QAT number, and the entire QAT lane of the project is unexplored.

**Primary variant — V15:** One-file diff replacing the class-level Python bool with a tensor-valued mask inside the STE expression. The branch becomes a `mask * (w_q - w).detach()` multiply that is present in the compiled graph in both states, and the mask is a buffer whose value is flipped at step 1000 (tensor writes are visible to the compiled forward). Byte-identical forward-path to V3 while mask=0. Full STE once mask=1. No recompile. One A100 SXM run proves it.

The forensic test comes for free: if V15(mask=0 forever) vs V15(mask=1 @ step 1000) differ at step 5000, QAT has an effect and V4 was silently broken. If they don't differ, the STE is inert and the whole late-QAT lane is a dead end worth confirming before burning more H100 budget.

---

# The Forensic Hypothesis (the "where did you get this" angle)

From V4 source, `train_breadcrumb_recur_ema_stochdepth_bigramhash_int5mlp_qat.py`:

- Line 854: `_qat_enabled: bool = False` (class attribute, Python bool, default False)
- Line 858: `if (CastedLinear._qat_enabled and self.training and ...)` inside `CastedLinear.forward`
- Line 1346: `compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)`
- Line 1520: `CastedLinear._qat_enabled = True` at step 1000

`torch.compile` with `fullgraph=True` traces the forward at first invocation. Class-attribute reads on Python bools are specialized as constants during Dynamo's guard generation. The condition is folded. At step 1000 the Python attribute flips, but:
1. The compiled graph contains no conditional on the flag — the `if False` branch was pruned pre-CUDA.
2. Dynamo's guard on `_qat_enabled == False` holds because the read happens at trace-time on a class attribute that Python resolves *before* the recompile-guard check fires on the next forward. There is no guard on "class attribute `_qat_enabled` has changed."
3. Even if a recompile were triggered, it would need a fresh first-invocation to re-trace — the training loop doesn't reset the compiled wrapper.

The V4 docstring at lines 86-93 flags this exactly:

> "Per-instance _qat_clip_range is configured BEFORE torch.compile. If the compile caches a code path that captures `CastedLinear._qat_enabled` by value instead of attribute read, the gate may not activate under compile. Mitigation: the class-level attribute is read fresh on every forward (`CastedLinear._qat_enabled` not `self._qat_enabled`); torch.compile specializes on class-level attribute reads correctly in recent builds. First sign of failure: validation BPB does not diverge from V3 after QAT_START_STEP."

The mitigation ("reads fresh") does not work. Dynamo specializes on class attribute reads at trace time regardless of whether they're accessed through `self.` or `ClassName.`; both resolve to `LOAD_ATTR` against an object whose value is read once and baked in unless a guard is installed. Guards on mutable class attributes are not installed by default.

**The test:** run V4 at QAT_ENABLE=1 and QAT_ENABLE=0 with identical seeds for 5000 steps. If validation BPB is identical (or differs only by seed noise), QAT never turned on.

Doug likely already has one or both of those numbers from prior runs. If they match, V4's entire QAT lane has been inert and the 15.78MB legal run is a V3 run in QAT clothing.

---

# V15 — The Fix

**One-file change, one function, ~15 lines.** Replace the Python-bool gate with a tensor mask that participates in the compiled graph as a runtime value.

## Design

- Module-level `_QAT_MASK = torch.zeros(1)` tensor, registered as a non-persistent buffer on each `CastedLinear` module at build time (shared object so one `.fill_(1.0)` flips all modules).
- `CastedLinear.forward` always computes `w_q` (under `torch.no_grad()`) when the module is training, 2D, >65k params, not skipped. The `mask` tensor is the multiplier:

  ```python
  w = w + mask * (w_q - w).detach()
  ```

  - `mask=0.0`: `w + 0 * detached = w`. Byte-identical to V3 to fp roundoff (one extra elementwise multiply-add on a zero tensor; numerically exact in bf16 because `0 * x = 0` and `w + 0 = w` at the cast width).
  - `mask=1.0`: full STE. `w + (w_q - w).detach()` forward equals `w_q`, backward gradient flows through `w`.
- `_QAT_MASK.fill_(1.0)` at step `qat_start_step` flips all modules at once. Buffer writes are visible to the compiled forward because buffers are graph inputs, not Python constants.
- When `QAT_ENABLE=0`, the mask is never flipped and stays at 0.0 — forward is byte-identical to V3 for the entire run (preserves V3 baseline reproducibility).

## Cost analysis

- Under mask=0: one elementwise multiply + one add on each CastedLinear weight per forward. For ~20M params across ~60 linear layers, this is ~0.5–1% overhead measured in wallclock. Negligible vs 600s budget.
- Under mask=1: identical cost to V4 when V4 actually runs QAT.
- Artifact size: **unchanged**. QAT trains weights that round better at serialization time; it does not change the serialization path. V4 at 15.78MB legal stays at 15.78MB.

## Expected numbers (A100 SXM → H100 SXM extrapolation)

| Metric | V4 (claimed) | V4 (actual, hypothesis) | V15 |
|--------|--------------|------------------------|-----|
| Val BPB, pre-quant, step 5000 | ~1.195 | **== V3 within noise** | drifts from V3 post-step-1000 |
| Legal artifact BPB | 1.1803 | 1.1803 (same as V3) | **1.170–1.178 target** |
| Legal artifact size | 15.78 MB | 15.78 MB | 15.78 MB (identical quantizer) |
| Wallclock/step, A100 SXM bf16 | ~65 ms | ~65 ms | ~66 ms |

If V15(mask=1) and V15(mask=0) match on A100 at step 5000, the STE has no effect even with this fix — meaning signalrush/jfprincz-style late-QAT is flat on Shepherd topology at this scale and Doug can retire the QAT lane entirely. That is also a useful result (Fork C outcome from Programmers Room v2 terminology: honest negative).

If they diverge and mask=1 wins by >0.003 BPB at step 5000, promote to H100 for the full 20k-step run with QAT_START_STEP={500, 1000, 2000} sweep on remaining budget.

---

# Exact Diff vs `train_breadcrumb_recur_ema_stochdepth_bigramhash_int5mlp_qat.py`

Save this as `/workspace/variants/train_v15_tensor_gated_ste.py` — copy of V4 with these edits.

### Edit 1 — Add module-level mask tensor (insert around line 200, after env parsing)

```python
# V15: Tensor-gated STE mask. Shared across all CastedLinear instances. Registered
# as a non-persistent buffer so torch.compile sees it as a graph input. Writing
# _QAT_MASK.fill_(1.0) at step qat_start_step flips all instances simultaneously
# and the change is visible to the compiled forward (unlike a class-level Python
# bool, which Dynamo specializes as a constant at trace time under fullgraph=True).
_QAT_MASK: Tensor = torch.zeros(1, dtype=torch.float32)
```

### Edit 2 — Replace `CastedLinear` class (replace lines 837–871)

```python
class CastedLinear(nn.Linear):
    # V15: Tensor-gated STE. Mask is a non-persistent buffer shared across all
    # CastedLinear instances. mask=0.0 -> byte-identical to V3 (multiply by zero).
    # mask=1.0 -> full fake-quant + STE.
    #
    # The prior V4 approach (class-level _qat_enabled: bool) does NOT work under
    # torch.compile(fullgraph=True, dynamic=False): Dynamo specializes the Python
    # bool read as a constant at trace time, pruning the branch before the step-1000
    # flip can take effect. Tensor multiplies are always in the graph.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Share the module-level tensor via register_buffer. Same underlying
        # storage across all instances because we pass the same object.
        self.register_buffer("_qat_mask", _QAT_MASK, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if (self.training
                and w.ndim == 2
                and self.weight.numel() > 65536
                and not getattr(self, "_qat_skip", False)):
            clip = int(getattr(self, "_qat_clip_range", 31))
            with torch.no_grad():
                w32 = self.weight.float()
                row_max = w32.abs().amax(dim=1)
                scale = (row_max / float(clip)).clamp_min(1.0 / float(clip))
                w_q = (torch.clamp(torch.round(w32 / scale[:, None]),
                                   -(clip + 1), clip) * scale[:, None]).to(x.dtype)
            # Tensor-gated STE. mask=0 -> w (byte-identical). mask=1 -> w_q forward, w backward.
            mask = self._qat_mask.to(x.dtype)
            w = w + mask * (w_q - w).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)
```

### Edit 3 — Replace step-1000 activation (replace lines 1508–1522)

```python
    # V15: QAT activation via tensor write instead of class-bool mutation. The
    # buffer write is visible to the compiled forward on the next invocation
    # because buffers are graph inputs, not Python constants.
    _qat_activated = False

    # ...inside the training step loop, before loss computation:
    if args.qat_enable == 1 and not _qat_activated and step >= args.qat_start_step:
        _QAT_MASK.fill_(1.0)
        _qat_activated = True
        log0(f"qat:activated step:{step} mask:{_QAT_MASK.item():.1f}")
```

### Edit 4 — Move mask to device after model is on CUDA (add near line 1314, before `compiled_model = torch.compile(...)`)

```python
    # V15: mask must live on the same device as the weights for the multiply to
    # avoid a CPU-GPU stream hop inside the compiled graph.
    global _QAT_MASK
    _QAT_MASK = _QAT_MASK.to(device=next(base_model.parameters()).device)
    # Re-register on every CastedLinear so all instances point at the device copy.
    for _m in base_model.modules():
        if isinstance(_m, CastedLinear):
            _m._qat_mask = _QAT_MASK
```

### Edit 5 — Update QAT_ENABLE startup log (around line 1310, the existing qat_enable log line)

```python
    log0(f"v15:tensor_gated_ste qat_enable:{args.qat_enable} "
         f"qat_start_step:{args.qat_start_step} "
         f"routed_mlp:{_qat_route_counts['mlp']} routed_attn:{_qat_route_counts['attn']} "
         f"routed_bigram:{_qat_route_counts['bigram']} routed_skip:{_qat_route_counts['skip']} "
         f"mlp_qat_clip:{_mlp_clip} mask_device:{_QAT_MASK.device}")
```

That's the entire change. ~40 lines of edit, all in one file, no new dependencies.

---

# Torchrun Commands (A100 SXM, pod right_copper_warbler)

Matches Doug's existing pattern (seed 1337, 600s wallclock, 20k step cap, log to `/workspace/logs/`).

## Run 1 — V15 QAT-on (the hypothesis test)

```bash
cd /workspace && \
MAX_WALLCLOCK_SECONDS=600 MAX_STEPS=20000 \
QAT_ENABLE=1 QAT_START_STEP=1000 MLP_QUANT_BITS=5 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 \
  variants/train_v15_tensor_gated_ste.py \
  2>&1 | tee /workspace/logs/v15_qat_on_1337.log
```

## Run 2 — V15 QAT-off sanity (must match V3 byte-for-byte)

```bash
cd /workspace && \
MAX_WALLCLOCK_SECONDS=600 MAX_STEPS=20000 \
QAT_ENABLE=0 MLP_QUANT_BITS=5 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 \
  variants/train_v15_tensor_gated_ste.py \
  2>&1 | tee /workspace/logs/v15_qat_off_1337.log
```

Run 2 is the control. Its step-5000 val BPB MUST equal the V3 step-5000 val BPB from Doug's earlier V3 1337 logs. If it doesn't, there's a bug in the mask-path that isn't faithfully implementing `w + 0*x = w` and V15 is disqualified.

## Run 3 — Decision branch

- If Run 1 val BPB @ step 5000 differs from Run 2 val BPB @ step 5000 by >0.003 BPB in either direction: **QAT has effect**. Fire Run 3 as V15 QAT-on, QAT_START_STEP=500 (earlier activation to accumulate more QAT steps in 600s wallclock). Best of the three goes to H100 SXM for a full 20k-step champion run.
- If Run 1 ≈ Run 2: **QAT is inert on Shepherd at this scale**. Fire Run 3 as backup variant V16 (below). Document V15 result as Fork C honest-negative and retire the QAT lane.

---

# Backup Variants (fire only if V15 triage resolves in first 2 runs)

## V16 — Int7 Embeddings + Adaptive Clip (GPT DeepResearch Q-int7 lane)

Changes to V3 baseline (not V4):
- Embedding quantizer switches from fp16 passthrough to int7 (clip=63), saving ~200KB on the 50304-token embedding table. Artifact drops from 15.78MB toward ~15.5MB, freeing budget for an 11th layer or wider MLP.
- GPTQ-SDClip with `k = k0 + alpha * log(row_numel)` per-row adaptive clip instead of fixed k=12.85. Alpha ~0.15 typical. Shifts the clip tighter for larger rows where quantization noise averages out.

Ships as a quantizer patch in the `mixed_quantize_*` functions at serialization time. Zero training-loop change. Pure byte-reclaim. Expected BPB delta: –0.005 to –0.010 if the embedding saves are spent on a single extra breadcrumb layer or MLP width bump.

**Risk:** existing embeddings were not trained to tolerate int7 noise. Expect ~0.010 BPB *penalty* from pure int7 swap, offset by the gain from spending the saved bytes on capacity. Net is a gamble.

## V17 — Export-Matched STE (GPT DeepResearch Q-match lane, stricter than V4)

V15 assumes the signalrush STE form matches the post-training quantizer. It doesn't exactly — the post-training quantizer applies `amax.clamp_min(epsilon)` and the V4 STE uses `(row_max / clip).clamp_min(1/clip)`, which are algebraically equivalent only when epsilon and 1/clip match. V17 refactors both paths to call a single shared `_quantize_row(w, clip)` function so the STE forward is *bit-identical* to the post-training forward. Rules out any drift between training-time fake-quant and serialization-time real-quant.

Ships as a second-pass on V15 once V15 confirms QAT activates. Combine only if V15 lands; otherwise pointless.

---

# A100 vs H100 Caveat (read before committing all 3 runs)

Doug's pod is 8×A100 SXM. Contest eval is 8×H100 SXM. Three implications:

1. **Absolute BPB from A100 runs is not submittable.** A100 is ~1.6x slower on bf16 matmul-heavy. Under `MAX_WALLCLOCK_SECONDS=600` on A100, the training will hit ~12,500 steps instead of 20,000. The 600s-bound BPB on A100 is *worse* than the 600s-bound BPB on H100 because fewer steps complete.
2. **Relative BPB deltas transfer.** V15(mask=0) vs V15(mask=1) A100 delta predicts H100 delta to first order. This is the correct measurement for tonight.
3. **Champion run must be H100.** Once V15 lands on A100, the submittable run is a 20k-step H100 run with the same config. Budget ~$30 on a single 8×H100 hour. Plan for that on Monday.

The wallclock cap binds to A100 physics, not H100 physics. Don't submit any number from tonight's runs to the leaderboard. The signal is delta-vs-control, not absolute.

---

# What makes this the Hail Mary

Every other variant in the queue (B1, V2, V11, V8) is incremental territory — layer counts, Muon sweeps, bigram routing. V15 is a *retroactive* variant: it asks whether V4's claimed QAT ever actually happened, and if it didn't, it opens an unexplored lane that the entire signalrush/jfprincz reference line sits on. The GPT DeepResearch report ranks Q-match (export-matched STE) as the #1 highest-EV intervention; V15 is the necessary prerequisite — confirming the STE runs at all before spending budget matching it to the exporter.

The wow-factor hook: "V4's QAT branch is dead code under torch.compile. Here's the one-run test and one-line fix."

Cost: 3 A100 SXM runs @ ~$2/run = ~$6. Payoff: either (a) validates a 0.003+ BPB improvement lane that was believed to already be banked but never ran, or (b) retires the QAT lane cleanly so the remaining 11 days don't waste budget on it.

---

# Slot-in summary (paste this into the other Claude session)

> V15 — Tensor-Gated STE. Replaces V4's class-level `_qat_enabled: bool` with a `torch.zeros(1)` buffer multiplied into the STE expression. Under `torch.compile(fullgraph=True)` the class bool gets specialized as a constant at trace time; the branch is dead code and the step-1000 flag flip does nothing. The tensor mask stays in the compiled graph in both states, so the step-1000 `fill_(1.0)` actually activates QAT. Run V15(QAT=1) vs V15(QAT=0) seed-matched; if they diverge, V4's 1.1803 banked number was V3-in-QAT-clothing and the QAT lane is unexplored. Diff is ~40 lines in one file. Pod: 8×A100 SXM right_copper_warbler. 3 runs: QAT-on, QAT-off sanity, decision branch. A100 numbers are relative-delta only; champion promotes to H100 next.

Sources:
- [HAIL MARY GPT DeepResearch report](computer:///sessions/jolly-wonderful-feynman/mnt/The Constant Saga/HAIL MARY USE FROM GPT deep-research-report.md)
- [V4 training script (target of diff)](computer:///sessions/jolly-wonderful-feynman/mnt/The Constant Saga/Parameter_Golf_Variants_2026-04-19/train_breadcrumb_recur_ema_stochdepth_bigramhash_int5mlp_qat.py)
- [Programmers Room v2 prompt (Shepherd Clause framing)](computer:///sessions/jolly-wonderful-feynman/mnt/The Constant Saga/PROGRAMMERS_ROOM_MASTER_PROMPT_v2.md)
