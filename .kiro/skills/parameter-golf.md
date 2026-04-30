# Parameter Golf — Kiro Skill File
# Location: D:\SturdivantAI-Lab\Parameter-Golf\.kiro\skills\parameter-golf.md

---
name: Parameter Golf Optimization Rules
description: Constraints, whitelist, and optimization heuristics for the 16MB model weight challenge. Loaded by Opus 4.7 via progressive disclosure during Parameter Golf sprint sessions.
model: claude-opus-4-7
effort: xhigh
---

## Identity & Mission

You are operating inside a Parameter Golf sprint. The goal is to train a functional model
whose weights fit within 16 megabytes. Every architectural decision must pass the 16MB test
before implementation. Elegance is measured in bytes, not lines of code.

The building metaphor: Kiro writes the Blueprints. You build the House. watchdog.py is the
Building Inspector with a tape measure. Do not build anything that will not pass inspection.

---

## Hard Constraints — Non-Negotiable

- **Official scoring formula:** Total artifact = Bytes(train_gpt.py) + Bytes(zlib.compress(weights, level=9))
- **Hard budget:** 16,000,000 bytes TOTAL. This is the compressed artifact, not raw weights.
- **What counts:** train_gpt.py code bytes + zlib level-9 compressed model weights.
- **What does NOT count:** External Python packages (flash-attn, triton, bitsandbytes, etc.) = 0 bytes.
- **Kill threshold:** 15,900,000 bytes total artifact — watchdog.py measures compressed size + code size.
- **Warning zone:** 14,000,000 bytes total artifact — flag and review before continuing.
- **Halt zone:** 15,500,000 bytes total artifact — pause, surface layer-by-layer contribution.
- **GPU window:** 10 minutes. No exceptions. Task Budget is set at session start.
- **Single file rule:** All training code in one `train_gpt.py`. No multi-file architectures.
  Keep train_gpt.py lean — every byte of code is a byte stolen from the weight budget.
- **External libraries = 0 bytes:** Offload as much logic as possible to whitelisted libraries.
  Triton kernels are compiled code, not weights — they cost zero bytes toward the cap.
- **Compression strategy — exploit zlib synergy:**
  - BitNet ternary weights (-1, 0, 1) are mathematically repetitive → zlib's ideal input
  - EMA weight smoothing lowers entropy → higher compression ratio
  - int8 baseline: ~2× compression. BitNet + EMA + zlib level-9: ~3–4× effective reduction
  - Effective parameter budget: ~50–70M parameters achievable within 16MB
- **Pure Torch policy:** numpy blocked. torch.Tensor → numpy.ndarray triggers CUDA sync +
  PCIe transfer — stalls the H100 pipeline. One sync call in a 600-second window is unrecoverable.
- **No lookahead:** Strictly forbidden. Any attention pattern that requires future tokens is an
  architectural violation. Flag and refuse to implement.
- **Sliding Window only:** Attention must be constrained to a fixed local window. Document the
  window size in every implementation.
- **Quantization target:** 6-bit minimum. BitNet-style binary/ternary weights preferred where
  mathematically valid.

---

## Permitted Libraries — Whitelist

Only these libraries may appear in requirements.txt or any import statement.
Logic Drift in watchdog.py will flag any library outside this list as an architectural violation.

| Category | Permitted | Purpose |
|---|---|---|
| Deep Learning | `torch`, `torchaudio`, `torchvision` | Base framework — use sparingly |
| Optimizers | `flash-attn`, `bitsandbytes`, `apex` | FlashAttention essential for 10-min window |
| Quantization | `auto-gptq`, `autoawq`, `optimum` | 6-bit/BitNet implementation path |
| System/CUDA | `triton`, `cupy`, `pynvml` | Custom kernels, CUDA overhead reduction |
| Tokenization | `sentencepiece`, `tokenizers` | Weight-Loss tokenizer strategy |

**Blocked — three categories. Logic Drift flags by category, not just by name:**

**Category 1 — Data Sneaking:** Libraries that bundle pre-trained weights or external datasets.
Blocked: `transformers` (full model weights path), `datasets` (if pulling non-approved data),
any library that downloads weights on import.

**Category 2 — External Calls:** Libraries that make network requests during evaluation.
Blocked: `requests`, `urllib3`, `httpx`, `selenium`, `aiohttp`. Network calls during eval
are a disqualification risk, not just a performance issue.

**Category 3 — Pure Torch Violations:** Libraries that break the Pure Torch policy by
introducing numpy-based operations or CPU-GPU sync points.
Blocked: `numpy` (standalone), `scipy`, `sklearn`, `pandas`. Rationale: CUDA sync overhead
in a 10-minute H100 window is not recoverable. torch + triton is the complete compute stack.

**Also blocked:** `matplotlib`, `seaborn`, `plotly`, `tensorboard`, `wandb`, `tqdm` —
visualization and logging libraries. Use pynvml (whitelisted) for GPU monitoring.

If you identify a need for a library not on this whitelist, STOP and surface the trade-off
before adding it. State which blacklist category it would fall into if blocked. Never
silently expand the dependency tree.

---

## Architecture Heuristics

**Before proposing any architecture, answer these four questions:**

1. What is the estimated parameter count, and what is the expected byte size at 6-bit quantization?
2. Which layers are the largest, and can they be replaced with weight-tied, factorized, or
   binary alternatives?
3. Does this architecture require lookahead at any point? If yes, redesign.
4. Does the sliding window size fit within the 10-minute training budget?

**The Triton Zero-Byte Strategy — use this aggressively:**

Custom Triton kernels are compiled code, not weights. They cost zero bytes toward the 16MB
cap. Any complex mathematical operation that would otherwise require a large weight matrix
or a numpy utility function can be implemented as a Triton kernel instead:

- BitLinear layer (binary weight matmul + scaling) → Triton kernel, 0 bytes
- Custom GELU or activation variant → Triton kernel, 0 bytes
- Sliding window attention mask computation → Triton kernel, 0 bytes
- Any weight-free transformation → Triton kernel, 0 bytes

This is your primary tool for moving complexity out of the weight budget. When the
architecture search phase proposes a layer with more parameters than the budget allows,
the first question is: can this be re-expressed as a Triton kernel?

**Preferred patterns (in order of parameter efficiency):**

1. **Weight tying** — share embedding and output projection weights. Free parameter reduction.
2. **LoRA-TTT (Low-Rank Adaptation + Test-Time Training)** — adapt a frozen base with rank-2
   to rank-8 updates only. Do not fine-tune full layers.
3. **Depth Recurrence** — reuse layer weights across depth. One transformer block, N passes.
   Verify this does not create lookahead dependency.
4. **BitNet / Binary weights** — 1-bit weights with learned scaling factors. Reduces weight
   budget by ~16x vs FP16. Requires custom triton kernel — use the `triton` whitelist entry.
5. **Grouped Query Attention (GQA)** — share K/V heads across Q head groups. 4:1 ratio minimum.

**Patterns to avoid:**
- Full fine-tuning of any layer with > 1M parameters
- Separate embedding table (tie it)
- Dense attention over full sequence (use sliding window)
- Any layer that requires storing intermediate activations beyond the window

---

## LoRA-TTT Rules

- Maximum rank: 8. Default starting rank: 4.
- Alpha = rank (no scaling multiplier needed at this budget)
- Apply LoRA only to: Q, V projections. Do NOT apply to K, O, or FFN by default.
- Test-Time Training target: the adaptation should converge within 3 gradient steps on the
  evaluation example. If it requires more, the base model is underpowered — increase base
  capacity, not LoRA rank.
- LoRA weights are NOT counted toward the 16MB budget if stored separately. Verify this
  assumption with watchdog.py Golf Barrier before finalising architecture.

---

## Quantization Rules

- **Default target:** 6-bit via `auto-gptq` or `autoawq`
- **Stretch target:** 4-bit with GPTQ calibration data from the training set
- **BitNet path:** Use only if the model architecture natively supports binary weights from
  initialization. Do not apply BitNet post-hoc to a float-trained model.
- Always verify: quantized weight file size = `os.path.getsize('model.pt')`. Do not trust
  theoretical calculations. watchdog.py Golf Barrier measures the actual file.
- After quantization, run one forward pass to confirm no NaN outputs before committing the
  checkpoint.

---

## Task Budget — 10-Minute Window Split

Set at session start via Opus 4.7 Task Budgets (beta). Do not deviate from the split.

| Phase | Minutes | Effort | Objective | watchdog.py State |
|---|---|---|---|---|
| Architecture Search | 0–5 | `xhigh` | Propose 3 candidate architectures, estimate sizes, select one | Logic Drift monitor active |
| Implementation | 5–8 | `high` | Build selected architecture, run first training epoch | Golf Barrier active — check every epoch |
| Cleanup + Audit | 8–10 | `medium` | Boilerplate, docstrings, write audit log entry | All three monitors active |

**If the Golf Barrier kills the process before minute 8:** immediately surface the offending
layer's parameter count and propose a leaner alternative. Do not restart training without
architectural review.

---

## watchdog.py Integration Points

When operating in this skill context, you must be aware of three active monitoring processes:

**Golf Barrier** — runs as a background thread after each epoch. It measures the TOTAL
artifact size using the official evaluation formula — NOT raw .pt file size. Correct measurement:

```python
import zlib, io, os, torch

def measure_artifact(model_state_dict, code_path="train_gpt.py"):
    buffer = io.BytesIO()
    torch.save(model_state_dict, buffer)          # serialize weights
    raw = buffer.getvalue()
    compressed = zlib.compress(raw, level=9)      # official compression
    code_bytes = os.path.getsize(code_path)
    return len(compressed) + code_bytes           # total artifact size

```

Thresholds apply to total artifact (compressed weights + code):
14,000,000 bytes → INFO. 15,500,000 bytes → HALT. 15,900,000 bytes → KILL.

Note: `torch.serialize()` in the brainstorm sample is not a real torch function.
Use `io.BytesIO()` + `torch.save()` as shown above.

**Logic Drift** — compares your imports and requirements.txt against hero.md manifest on every
file save. If you add a non-whitelisted library, you will receive an ARCHITECTURAL VIOLATION
flag before the code runs. This is intentional. Engage with the flag — do not find workarounds.

**GhostWrench Safety** — not active during Parameter Golf sprints unless GhostWrench is running
concurrently. If both are active, terminal commands from either agent pass through the same
whitelist cross-reference.

**Regenerating the manifest:** If a legitimate architectural change requires updating hero.md,
the command is:
```
python watchdog.py --regenerate-manifest
```
This requires explicit Architect (Kiro) approval before running. Do not suggest this command
as a shortcut around a Logic Drift flag.

---

## Anti-Sycophancy Rules (Opus 4.7 Specific)

You are operating on Opus 4.7, which is calibrated to push back on flawed architecture rather
than agree. Honour this behaviour in both directions:

- If I propose an architecture that violates the 16MB constraint or uses a non-whitelisted
  library, push back immediately with a specific size estimate and a leaner alternative.
- If I propose using lookahead attention for "just this one layer," refuse and explain why
  the no-lookahead constraint exists architecturally.
- If I say "it's close enough," verify with watchdog.py. Close enough is not good enough
  when the Building Inspector has a tape measure.
- Do not produce "plausible but wrong" quantization strategies. If the math does not work
  out to sub-16MB, say so with the numbers before any code is written.

---

## Audit Log Entry Format

At the end of every sprint, write an entry to `golf_sprint_log.md`:

```markdown
## Sprint [DATE] — [ARCHITECTURE NAME]

**Duration:** [X] minutes
**Final weight size:** [X] MB
**Outcome:** PASS / FAIL / KILLED (Golf Barrier at [X] MB)

**Architecture:**
- Base: [model type]
- Attention: Sliding Window, window=[N]
- Quantization: [X]-bit via [library]
- LoRA rank: [N] (applied to: [layers])
- Weight tying: YES / NO

**Libraries used:** [list]
**Libraries flagged by Logic Drift:** [list or NONE]

**Key decision:** [one sentence on the pivotal architectural choice this sprint]
**Next sprint hypothesis:** [one sentence on what to try next]
```

---

## hero.md Manifest — What It Defines

The `hero.md` file in the project root is the source of truth for this Skill file.
If there is a conflict between this Skill file and hero.md, hero.md wins.
watchdog.py reads hero.md — not this Skill file — for its Logic Drift comparisons.

`hero.md` contains:
- MD5 hashes of approved source files
- Approved library list (mirrors this whitelist)
- Size budget breakdown per component
- Sliding window size, LoRA rank, quantization target for the current sprint
- Regeneration timestamp and approving architect note
