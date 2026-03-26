# ARTEMIS CODE REVIEW
**Reviewer:** Artemis 🏹 — Code Quality & Structural Refinement  
**Date:** 2026-03-24  
**Context:** OpenAI Parameter Golf — train best LM in 16MB, 10 min on 8×H100  
**Requirement:** All code in ONE self-contained `train_gpt.py` file

---

## Summary Table

| File | Lines | Distributed Ready | Self-Contained | Critical Bugs | Verdict |
|------|-------|-------------------|----------------|---------------|---------|
| `train_gpt_model4.py` | 719 | ⚠️ Near-ready | ✅ Yes | 2 (MLP bug + missing barrier) | **Fix 2 bugs → submit** |
| `train_gpt_model1.py` | 322 | ❌ No | ❌ No (5 external deps) | 3 | **Not competition-ready** |
| `train_gpt_model2.py` | 1390 | ✅ Yes | ⚠️ Near (sentencepiece) | 2 (eval_model mismatch + ema var) | **Fix 2 bugs → strong candidate** |
| `train_gpt_model3.py` | 176 | ❌ Delegated | ❌ No (hardcoded path) | 3 | **Not competition-ready as-is** |

---

## FILE 1: `train_gpt_model4.py` — Optimized Transformer (Safe Baseline)

### Architecture
Standard GPT with: GQA attention, RoPE, RMSNorm, Bigram hash embedding, SmearGate, U-Net skip connections, EMA, Muon optimizer, late QAT, int6+zstd export.

### ✅ What Works
- Distributed init (`dist.init_process_group`) and DDP wrapping correct
- Muon optimizer has div-by-zero protection: `max(update.size(1), 1)` ✅  
- EMA initialized and updated from unwrapped `model` (not DDP `train_model`) ✅
- Warmup reset logic correctly saves/restores model + optimizer states ✅
- Wallclock cap + warmdown schedule both implemented ✅
- Self-contained: stdlib + torch + optional zstandard ✅
- Smoke test path works cleanly ✅

---

### 🔴 CRITICAL Bug #1 — MLP applies `.square()` twice (double exponentiation)

**Location:** `MLP.forward()` — lines ~370-371

```python
# CURRENT (WRONG):
def forward(self, x: Tensor) -> Tensor:
    x = torch.relu(self.fc(x)).square()  # x = relu(Wx)^2
    return self.proj(x.square())          # x = relu(Wx)^4  ← BUG

# CORRECT (ReLU² activation):
def forward(self, x: Tensor) -> Tensor:
    x = torch.relu(self.fc(x)).square()  # x = relu(Wx)^2
    return self.proj(x)                   # project squared activation
```

**Impact:** The model computes `relu(x)^4` instead of `relu(x)^2`. This is not what the comment says (`# ReLU² activation`). The loss will be higher than it should be throughout training because the MLP output is severely clipped/saturated at large values. This silently degrades performance — training won't crash, it'll just underperform. Compare to `train_gpt_model2.py` lines 595-596 which has the correct implementation.

**Fix:** Remove the `.square()` from the final line.

---

### 🔴 CRITICAL Bug #2 — Missing distributed barrier before `destroy_process_group`

**Location:** After training loop, lines ~698-715

```python
# CURRENT (WRONG):
if TORCH_AVAILABLE:
    model.load_state_dict(...)
    if rank == 0:
        # rank0 does eval + quantize + serialize — takes variable time
        sample = torch.randint(...)
        slide_loss = eval_sliding_window(...)       # slow
        quantized, meta = mixed_quantize(...)       # slow
        buf = io.BytesIO()
        torch.save(...)
        blob = zstandard.ZstdCompressor(level=22).compress(raw)  # very slow
        print(...)
if distributed:
    dist.destroy_process_group()  # ← ranks 1-7 reach here BEFORE rank0 finishes!
```

**Impact:** On 8×H100, ranks 1-7 call `dist.destroy_process_group()` while rank 0 is still running `ZstdCompressor(level=22).compress()` (which can take several seconds). This causes an **NCCL error** or **process hang** — the run will terminate abnormally and the quantized model may not be saved.

**Fix:**
```python
# CORRECT: Add barrier before the rank-0-only block
if distributed:
    dist.barrier()  # ← all ranks wait here
if TORCH_AVAILABLE:
    model.load_state_dict(...)
    if rank == 0:
        # ... eval + export ...
if distributed:
    dist.destroy_process_group()
```

---

### ⚠️ WARNING #1 — Final validation uses random synthetic tokens, not real validation data

**Location:** Lines ~700-703

```python
sample = torch.randint(0, args.vocab_size, (args.train_seq_len + 1,), device=device)
slide_loss = eval_sliding_window(model.eval(), sample, args.train_seq_len, args.eval_stride)
```

`val_files` is defined in `Hyperparameters` but never loaded or used. The `sliding_eval_loss` printed to stdout is computed on **random noise tokens** — it's meaningless as a quality metric. The competition judging system presumably runs its own eval, so this won't disqualify you, but you lose the ability to track real progress during training runs.

**Fix:** Load validation data and compute real validation loss. The infrastructure for this already exists in `train_gpt_model2.py` (`load_validation_tokens`, `eval_val`, `build_sentencepiece_luts`) — port it in.

---

### ⚠️ WARNING #2 — Wallclock break is not synchronized across ranks

**Location:** Line ~696-697

```python
if args.max_wallclock_seconds > 0 and not smoke and (time.perf_counter() - start) > args.max_wallclock_seconds:
    break
```

Each rank independently checks its own wall clock. In theory this is fine (all ranks do the same amount of work per step), but minor clock drift could cause one rank to break out of the training loop while others continue, leading to a DDP deadlock on the next `all_reduce`. In practice on homogeneous H100 hardware this is unlikely to trigger, but `train_gpt_model2.py` has the correct pattern (`dist.all_reduce(reached_cap_tensor)`) — prefer that approach.

---

### ℹ️ MINOR — `require_backward_grad_sync` guarded by `isinstance(train_model, DDP)` — correct but verbose

**Location:** Line ~671

```python
if distributed and isinstance(train_model, DDP):
    train_model.require_backward_grad_sync = micro == grad_accum_steps - 1
```

The double check (`distributed and isinstance(...)`) is redundant since `maybe_wrap_ddp` returns DDP iff `distributed`. Not a bug, just slightly verbose.

---

### VERDICT: Model4
**2 bugs to fix. After fixes, this is competition-ready for 8×H100.**
- Fix MLP double-square (5 min)
- Add distributed barrier before export (2 min)
- Optionally: add real validation eval

---

---

## FILE 2: `train_gpt_model1.py` — Codec Architecture (N-gram + Transformer Residual)

### Architecture
Multi-phase codec: dictionary building → Kneser-Ney n-gram LM → residual BitNet transformer correction → LPC preprocessing. Trains only on "hard" (high-entropy) batches.

### 🔴 CRITICAL Bug #1 — NOT SELF-CONTAINED: 5 external import dependencies

**Location:** Lines 19-22

```python
from builds.model1_phase1_dictionary import build_dictionary, verify_dictionary
from builds.model1_phase2_ngram import KneserNeyNGram
from builds.model1_phase3_transformer import ResidualBitNetTransformer
from builds.model1_phase4_lpc import LinearPredictiveCoder
```

These imports pull from 4 separate files in `builds/` (~572 lines total). The competition requires a **single file `train_gpt.py`**. This file cannot be submitted as-is.

**Fix:** Inline all 4 build files into `train_gpt_model1.py`. This will make it ~900 lines — well within the 1500-line limit.

---

### 🔴 CRITICAL Bug #2 — ZERO distributed training support

**Location:** No `dist` imports, no `DDP`, no `RANK`/`WORLD_SIZE` handling anywhere.

The competition runs `torchrun --nproc_per_node=8`. This file runs on a **single GPU only**. On 8×H100 it will only use 1 GPU, wasting 87.5% of available compute.

**Impact:** Training speed is ~8× slower than it could be. With only 10 minutes of wallclock, this is catastrophic for competition performance.

**Fix:** This is a significant rewrite. The codec architecture as designed doesn't parallelise trivially (n-gram fitting is sequential, LPC is per-corpus). Consider:
1. Fitting n-gram on rank 0, broadcasting state to all ranks
2. Sharding the "hard batch" training across ranks
3. Only parallelising the neural residual training

---

### 🔴 CRITICAL Bug #3 — `sentencepiece` imported inside `run_pipeline()` but required at top level

**Location:** Line ~277

```python
def run_pipeline(args: Hyperparameters) -> dict[str, object]:
    ...
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
```

The `build_sentencepiece_luts` function is defined in the same file and used only in `run_pipeline`. However, any call path through `run_pipeline` requires `sentencepiece`. If this library is not available on the competition server, the entire pipeline crashes at runtime (not at import time — late failure).

More critically: `sentencepiece` is needed to compute `val_bpb`, but the **training** itself (n-gram + transformer) doesn't need it. The byte-per-byte metric computation will crash if `sentencepiece` is unavailable.

---

### ⚠️ WARNING #1 — N-gram `next_distribution` called in a Python loop per token per batch item

**Location:** `CodecModel.ngram_logits()` lines ~120-135, and `build_hard_token_batches()` lines ~145-158

```python
for b in range(batch):
    history: list[int] = []
    for t in range(seq_len):
        result = self.ngram.next_distribution(history)  # Python call per token!
```

With `seq_len=1024` and `batch_size=8`, this is 8192 Python-level function calls per forward pass. The KneserNey n-gram has no GPU acceleration. This will make training extremely slow. With 10 minutes wallclock, this likely limits to only a few hundred training steps total.

---

### ⚠️ WARNING #2 — `residual_dist = base_probs_t + 0.01 * res` shape mismatch risk

**Location:** Line ~154

```python
res = residual[start : start + seq_len].to(device=device).unsqueeze(-1)  # (seq_len, 1)
residual_dist = base_probs_t + 0.01 * res  # (seq_len, vocab_size) + 0.01*(seq_len, 1)
```

Broadcasting works here (1 → vocab_size), but semantically this is adding a **scalar LPC residual** uniformly across all vocabulary probabilities. This doesn't make architectural sense — the LPC residual should influence specific vocabulary items, not shift all probabilities uniformly. This produces a valid (non-crashing) computation but likely does not improve model quality.

---

### ⚠️ WARNING #3 — Only trains on first training shard, not the full dataset

**Location:** `run_pipeline()` line ~238

```python
train_tokens = load_data_shard(train_files[0])  # ← only first file
hard_batches = build_hard_token_batches(train_tokens, ngram, args, device)
```

The n-gram is fit on all shards, but the hard-batch residual training only uses `train_files[0]`. With 10B tokens across many shards, this severely limits what the transformer can learn.

---

### ℹ️ MINOR — No max_wallclock_seconds enforcement

No timer check during training. If n-gram fitting takes longer than expected, the 10-minute cap will be exceeded silently.

---

### VERDICT: Model1
**Not competition-ready. Requires major work:**
1. Inline all 4 `builds/` dependencies (~2-4 hrs)
2. Add distributed training (~4-8 hrs)
3. Reconsider n-gram Python loop performance (likely a blocker for 10-min run)
4. This architecture is innovative but may not be viable in the competition's constraint envelope without architectural changes.

---

---

## FILE 3: `train_gpt_model2.py` — Adaptive Recursive Transformer

### Architecture
Single shared block executed adaptively 1–12 times per token (controlled by a learned depth router or Gumbel-softmax), frequency codebook for common tokens, Muon optimizer, EMA, SWA, int6+zstd export, full distributed training support.

### ✅ What Works
- Full 8×H100 distributed training ✅
- Synchronized wallclock cap (`dist.all_reduce(reached_cap_tensor)`) ✅
- Correct codebook broadcast to all ranks ✅
- Correct `require_backward_grad_sync` pattern ✅
- Warmup reset logic correct ✅
- EMA state initialized and updated correctly ✅
- Barrier before `destroy_process_group` ✅
- Strong quantization with self-contained roundtrip verification ✅

---

### 🔴 CRITICAL Bug #1 — `eval_model` instantiation missing recursive-architecture params

**Location:** Lines ~1324-1335

```python
eval_model = GPT(
    vocab_size=args.vocab_size, ...
    mtp_num_heads=0, mtp_loss_weight=0.0,
    bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
    xsa_last_n=args.xsa_last_n,
    rope_dims=args.rope_dims, ln_scale=args.ln_scale, dtg=args.dtg_enabled,
    ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
    # ← MISSING: fixed_depth, max_depth, use_router, router_hidden_dim,
    #            router_aux_loss_weight, codebook_size, shared_block_count
)
```

**Impact:** With default values (`codebook_size=0` vs training's `codebook_size=2048`, `use_router=True` at default dims, etc.), the `eval_model` will have a **different state_dict structure** than the trained model. The subsequent `eval_model.load_state_dict(deq_state, strict=True)` will raise a `RuntimeError: Missing/Unexpected keys` and the entire post-training eval block crashes.

This is the most dangerous bug because it occurs after training completes — you'd lose the trained model's evaluation and submission artifact.

**Fix:**
```python
eval_model = GPT(
    vocab_size=args.vocab_size, ...
    fixed_depth=args.fixed_depth,
    max_depth=args.max_depth,
    use_router=args.use_router,
    router_hidden_dim=args.router_hidden_dim,
    router_aux_loss_weight=args.router_aux_loss_weight,
    codebook_size=args.codebook_size,
    shared_block_count=args.shared_block_count,
    mtp_num_heads=0, mtp_loss_weight=0.0,
    bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
    xsa_last_n=args.xsa_last_n,
    rope_dims=args.rope_dims, ln_scale=args.ln_scale, dtg=args.dtg_enabled,
    ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers,
).to(device).bfloat16()
```

---

### 🔴 CRITICAL Bug #2 — `ema_decay` is hardcoded, ignoring `args.ema_decay`

**Location:** Lines ~1173-1174

```python
ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}
ema_decay = 0.997  # ← hardcoded, ignores args.ema_decay
```

`Hyperparameters.ema_decay` is defined as `float(os.environ.get("EMA_DECAY", 0.997))` (line 99), meaning it's configurable via env var. But the training loop uses a local variable `ema_decay = 0.997` instead of `args.ema_decay`. The env var setting is silently ignored.

This is a functional bug: if you tune EMA decay via `EMA_DECAY=0.995`, it won't take effect. With only 20,000 steps at 10 minutes, EMA tuning matters.

**Fix:** Replace `ema_decay = 0.997` with `ema_decay = args.ema_decay`.

---

### ⚠️ WARNING #1 — `sentencepiece` is a hard mandatory import

**Location:** Line 22

```python
import sentencepiece as spm
```

Unlike model4, `sentencepiece` is an unconditional top-level import. If not installed on the competition server, the script fails at startup. The competition environment likely has it, but it should be wrapped in a try/except for robustness.

---

### ⚠️ WARNING #2 — Muon `g.size(0) / g.size(1)` uses Python integer division (acceptable but check for 1D params)

**Location:** Line ~152

```python
g *= max(1, g.size(0) / g.size(1)) ** 0.5
```

For 1D parameters (scalars/vectors), `zeropower_via_newtonschulz5` is called which transposes, but `g.size(1)` on a 1D tensor would fail. However, Muon in model2 only receives 2D `matrix_params` filtered by `p.ndim == 2`, so this should never encounter 1D tensors. ⚠️ Verify the filtering at param group construction is correct.

---

### ⚠️ WARNING #3 — `torch.compile` on `zeropower_via_newtonschulz5` at module level

**Location:** Top of `main()`, line ~935

```python
zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)
```

This mutates the global function reference. This is fine for a single training run but could cause issues if the function is imported elsewhere. Not a competition blocker.

---

### ⚠️ WARNING #4 — `fullgraph=True` compile on eval_model may fail with dynamic depth routing

**Location:** Line ~1341

```python
compiled_eval = torch.compile(eval_model, dynamic=False, fullgraph=True)
```

The recursive depth router uses `gate.max().item() == 0` (Python-level control flow that breaks graph capture) and Gumbel-softmax sampling. With `fullgraph=True`, torch.compile will fail if there are graph breaks. This will crash the post-training eval.

**Fix:** Use `fullgraph=False` for the recursive model.

---

### ℹ️ MINOR — `ve_enabled`, `ve_dim`, `ve_layers` accepted by `GPT.__init__` but never implemented

The GPT class accepts `ve_enabled`, `ve_dim`, `ve_layers` as constructor parameters but never creates any VE (vocabulary embedding?) modules. These params are dead code. Not a bug, but if VE was intended, it's missing.

---

### ℹ️ MINOR — EMA not synchronized across ranks in distributed training

EMA is updated per-rank on `base_model.state_dict()`. Since DDP keeps parameters synchronized via `all_reduce` during backward, the EMA states on all ranks will be identical after each optimizer step. This is correct behavior, not a bug — just noting it's local per-rank computation that happens to stay in sync.

---

### VERDICT: Model2
**2 critical bugs, easy fixes. After fixing, this is the strongest candidate.**
- Fix `eval_model` missing params (10 min — just copy the training GPT instantiation)
- Fix `ema_decay = 0.997` → `ema_decay = args.ema_decay` (30 sec)
- Fix `fullgraph=True` → `fullgraph=False` on eval compile (30 sec)
- The architecture (adaptive recursion + codebook) is genuinely novel and competition-worthy

---

---

## FILE 4: `train_gpt_model3.py` — SSM/Transformer Hybrid

### Architecture
HybridGPT with `ssm_layers=8` SSM (selective state space model with MoE hash routing) + 3 transformer attention layers, bigram embedding, SmearGate. Delegates all training infrastructure to a **dynamically loaded external trainer** via `importlib`.

### 🔴 CRITICAL Bug #1 — Hardcoded absolute path to external trainer

**Location:** Lines 15-22

```python
HOST_PATH = Path(
    "/home/ferrante42/.openclaw/workspace/parameter-golf/records/track_10min_16mb/"
    "2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/train_gpt.py"
)
_SPEC = importlib.util.spec_from_file_location("model3_host", HOST_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Unable to load host trainer from {HOST_PATH}")
host = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(host)
```

This path does not exist on the competition server. The script will raise `RuntimeError` on the first import. **This file cannot be run on any machine except yours.**

---

### 🔴 CRITICAL Bug #2 — NOT SELF-CONTAINED: imports from `builds.model3_lib`

**Location:** Line 12

```python
from builds.model3_lib import HybridGPT, env_int, run_smoke_training
```

`builds/model3_lib.py` is 478 lines and itself imports from `train_gpt` (the baseline). Competition requires single-file submission.

---

### 🔴 CRITICAL Bug #3 — Three-layer dependency chain prevents standalone execution

The full dependency chain is:
```
train_gpt_model3.py
  └─ imports builds/model3_lib.py
       └─ imports train_gpt.py (baseline, expected at workspace root)
  └─ dynamically loads records/.../train_gpt.py (HOST_PATH, hardcoded absolute path)
```

Even if you fix the absolute path, `model3_lib` requires `train_gpt` to be importable from `sys.path`. The competition server only has the single submitted file.

---

### ⚠️ WARNING #1 — Monkey-patching `host.GPT = Model3GPT` works in CPython but is fragile

**Location:** Lines 174-176

```python
host.GPT = Model3GPT
host.Hyperparameters = Hyperparameters
host.main()
```

This pattern works in CPython because `host.main()` uses `host`'s module `__dict__` to resolve `GPT`. However:
- If `host.main()` calls any helper function that references `GPT` by closure (not by module lookup), the patch won't apply
- Compiled functions (`torch.compile`) may have already captured the original `GPT` reference
- If `host` is reloaded or re-executed, patches are lost

Verify that `host.main()` references `GPT` as a module-level name (i.e., `GPT(...)` not `some_func_that_captured_GPT()`). From code inspection, `host.main()` uses `base_model = GPT(...)` which resolves through the module dict — so this *does* work correctly in practice. But it's fragile architecture.

---

### ⚠️ WARNING #2 — `Hyperparameters.state_dim = model_dim` creates a class-level dependency on class variable ordering

**Location:** Lines 28-33

```python
class Hyperparameters(host.Hyperparameters):
    num_layers = int(os.environ.get("NUM_LAYERS", 11))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    state_dim = int(os.environ.get("STATE_DIM", model_dim))  # model_dim evaluated at class body time
```

`state_dim = int(os.environ.get("STATE_DIM", model_dim))` uses `model_dim` as a default by referencing the **class variable** `model_dim` defined 2 lines above. This works in Python class bodies (evaluated in order), but if `STATE_DIM` env var is not set, `state_dim` gets the *default* value of `model_dim` (512) **at class definition time**, not the potentially-overridden runtime value. If `MODEL_DIM` is overridden at runtime, `state_dim` will still be 512 unless `STATE_DIM` is also explicitly set. This is a subtle config bug.

---

### ℹ️ MINOR — SSM sequential loop is incompatible with `torch.compile`

The `SelectiveSSMExpert.forward()` in `model3_lib.py` runs a Python `for t in range(seq_len):` loop. `torch.compile` cannot efficiently compile Python loops that iterate over the sequence dimension. This will fall back to eager mode or produce a very slow compiled graph. Performance will be significantly worse than expected.

---

### VERDICT: Model3
**Not competition-ready. Requires full consolidation into a single file.**
The actual hybrid SSM architecture (in `builds/model3_lib.py`) is well-designed. The work needed:
1. Inline `builds/model3_lib.py` (~478 lines) into the submission file
2. Inline the referenced `host` training loop from `records/...train_gpt.py` or from `train_gpt.py`
3. Remove the `importlib` dynamic loading
4. Fix absolute path dependency
5. Estimated effort: 3-5 hours to consolidate properly

---

---

## Cross-Cutting Issues

### Competition Compliance Summary

| Requirement | Model4 | Model1 | Model2 | Model3 |
|-------------|--------|--------|--------|--------|
| Single file | ✅ | ❌ | ✅ | ❌ |
| 16MB size limit | ✅ (exports compressed) | Unknown | ✅ (verified) | Unknown |
| 10-min wallclock | ✅ | ❌ (no enforcement) | ✅ | Delegated |
| 8×H100 distributed | ⚠️ (barrier missing) | ❌ | ✅ | Delegated |
| No hardcoded paths | ✅ | ✅ | ✅ | ❌ |

### Recommended Priority Order (to get something on the leaderboard)

1. **Ship Model4 immediately** after applying the 2 critical fixes. It's the safest path.
2. **Develop Model2 in parallel** — fix the 2 critical bugs + fullgraph issue. Best architecture, needs ~30 min of fixes.
3. **Consolidate Model3 into a single file** if you want the SSM hybrid — it's genuinely different. ~4-6 hours of work.
4. **Model1 needs a major rethink** — the n-gram Python loop won't scale to 10 minutes of H100 training.

### OOM Risk Assessment

- **Model4:** Low. Standard transformer with GQA, seq_len=2048, batch=48 seqs/GPU. Fine on H100.
- **Model2:** Medium. `max_depth=12` means the single block runs 12 times, storing all intermediate activations for backprop through depth. With `model_dim=512`, this is manageable but monitor memory.
- **Model3:** Low for SSM layers (state-based, no attention), medium for the 3 attention layers.
- **Model1:** High risk from Python loop overhead causing timeout, not OOM.

---

*Review complete. Structural coordinates mapped. Bugs logged. Ready for fixes.*  
*— Artemis 🏹*
