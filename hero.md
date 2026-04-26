# hero.md — Parameter Golf Master Manifest
# Source of truth for watchdog.py Logic Drift comparisons.
# If conflict exists between this file and parameter-golf.md skill file, hero.md wins.
# Regenerate with: python watchdog.py --regenerate-manifest
# Regeneration requires explicit Architect (Kiro) approval — never use as shortcut around a Logic Drift flag.

---

## Section 1 — Regeneration Metadata

regenerated_at: 2026-04-24T19:20:49.550295Z
regenerated_by: Cowork (pre-sprint bootstrap)
architect_signoff: Removed numpy from train_gpt.py — pure Torch replacement. Renamed audit log to watchdog_audit.jsonl.

---

## Section 2 — Approved Library List

Logic Drift blocks any library NOT on this list from appearing in requirements.txt or any import statement.
Three-category blacklist is enforced by category, not just by name.

### Whitelist

| Category       | Library          | Purpose                                      |
|----------------|------------------|----------------------------------------------|
| Deep Learning  | torch            | Base framework — primary compute stack       |
| Deep Learning  | torchaudio       | Audio utilities (permitted, use sparingly)   |
| Deep Learning  | torchvision      | Vision utilities (permitted, use sparingly)  |
| Optimizers     | flash-attn       | FlashAttention — essential for 10-min window |
| Optimizers     | bitsandbytes     | 8-bit/4-bit quantization                     |
| Optimizers     | apex             | Mixed precision, fused optimizers            |
| Quantization   | auto-gptq        | GPTQ quantization path                       |
| Quantization   | autoawq          | AWQ quantization path                        |
| Quantization   | optimum          | HuggingFace optimization bridge              |
| System/CUDA    | triton           | Custom kernels — 0 bytes toward cap          |
| System/CUDA    | cupy             | CUDA array operations                        |
| System/CUDA    | pynvml           | GPU memory monitoring in watchdog.py         |
| Tokenization   | sentencepiece    | SentencePiece tokenizer                      |
| Tokenization   | tokenizers       | HuggingFace fast tokenizers                  |
| Standard Lib   | zlib             | Compression measurement (official formula)   |
| Standard Lib   | io               | BytesIO for artifact measurement             |
| Standard Lib   | os               | File size measurement                        |
| Standard Lib   | hashlib          | MD5 hashing for Logic Drift                  |
| Standard Lib   | re               | Import statement parsing                     |
| Standard Lib   | json             | Audit log serialisation                      |
| Standard Lib   | threading        | Background monitor threads                   |
| Standard Lib   | time             | Epoch timing                                 |
| Standard Lib   | datetime         | Audit log timestamps                         |
| Standard Lib   | pathlib          | File path handling                           |
| Standard Lib   | sys              | Process control                              |

### Blacklist — Two Severity Tiers (Logic Drift flags by severity, not by category alone)

Severity distinction matters: a Tier A breach risks disqualification, a Tier B breach only costs training time. Watchdog should surface them differently (red vs. yellow) so the sprint signal is honest.

**Tier A — Competition Rule Violations (Disqualification Risk, RED)**

Libraries here violate the official OpenAI Parameter Golf rules. A submission that imports any of these at eval time is not a valid record. Watchdog flags these as critical.

| Library          | Rule Violated                                                              |
|------------------|----------------------------------------------------------------------------|
| transformers     | README §171: bundled pretrained weights violate "no external downloads"    |
| datasets (if pulling non-approved data) | README §171: external dataset access prohibited at eval   |
| requests         | README §171, §184: network calls during evaluation are prohibited          |
| urllib3          | Same as requests                                                           |
| httpx            | Same as requests                                                           |
| aiohttp          | Same as requests                                                           |
| selenium         | Same as requests, plus browser automation                                  |

Rationale: these fail the air-gap and artifact-containment requirements in README.md §171 ("No external downloads, training dataset access, or network calls are allowed during evaluation. The artifact must be fully self-contained and reproducible"). A single call here is disqualifying.

**Tier B — Performance Discipline (Optimization Risk Only, YELLOW)**

Libraries here are **legal** under official rules — README §194 explicitly allows any library that doesn't violate evaluation/compute/size rules ("importing FlashAttention, etc. is completely fine"). They are blocked by Kiro for the 10-min 8×H100 training window only, to protect the 600-second wallclock budget. Watchdog flags these as warnings, not violations. Removing them makes the sprint faster; it does not make the submission legal (it is already legal).

| Library       | Why Avoided                                                                   |
|---------------|-------------------------------------------------------------------------------|
| numpy         | torch.Tensor → ndarray triggers CUDA sync + PCIe transfer; unrecoverable in 600s |
| scipy         | Depends on numpy; inherits the same sync cost                                 |
| sklearn       | Depends on numpy; plus heavy import cost                                      |
| pandas        | Depends on numpy; plus object-dtype overhead                                  |
| tqdm          | Print/IO overhead per step; acceptable in dev, blocked in sprint              |
| matplotlib    | Not needed in training loop                                                   |
| seaborn       | Not needed in training loop                                                   |
| plotly        | Not needed in training loop                                                   |
| tensorboard   | Not needed in training loop                                                   |
| wandb         | Network + logging overhead; acceptable in dev, blocked in sprint              |

Rationale: none of these libraries are prohibited by OpenAI. They are self-imposed performance constraints. A "Tier B violation" in the audit log means "you are leaving sprint-time on the table", not "your submission is invalid".

---

## Section 3 — Size Budget

Official scoring formula: Total = Bytes(train_gpt.py) + Bytes(zlib.compress(weights, level=9))
Hard budget: 16,000,000 bytes TOTAL.
External libraries = 0 bytes toward cap.
Triton kernels = 0 bytes toward cap (compiled code, not weights).

| Component         | Budget (bytes) | Notes                                         |
|-------------------|----------------|-----------------------------------------------|
| train_gpt.py code | ≤ 500,000      | Every byte of code steals from weight budget  |
| Compressed weights| ≤ 15,500,000   | zlib level-9 compressed model state dict      |
| TOTAL             | ≤ 16,000,000   | Hard limit — Golf Barrier kills at 15,900,000 |

Thresholds (applied to total artifact = code + compressed weights):
- 14,000,000 bytes → INFO log
- 15,500,000 bytes → HALT — pause and surface layer-by-layer contribution
- 15,900,000 bytes → KILL — terminate training, log offending architecture

---

## Section 4 — Architectural Constraints (Current Sprint)

sprint_id: SPRINT-002
sprint_kind: ablation_study
target_baseline: upstream SOTA at commit 8b148a0 — val_bpb 0.9485, 3-seed mean

### Naming reconciliation (commit-message labels → code-grounded labels)

Upstream commit messages use names like "Scylla", "GPTQ-6bit", "XSA-all", "FA3". A direct read of `train_gpt.py` shows these labels do not map to discrete swappable modules:

| Upstream label | What's actually in the code |
|----------------|-----------------------------|
| GPTQ-6bit      | INT8 per-row quantization in `quantize_state_dict_int8` (no GPTQ machinery, no 6-bit) |
| XSA-all        | Grouped Query Attention via `F.scaled_dot_product_attention(..., enable_gqa=True)` (no sliding window) |
| FA3            | PyTorch SDPA dispatcher with `enable_flash_sdp(True)` (no FA3 import) |
| Scylla         | Single `Block` class with `RMSNorm` + `attn_scale` + `mlp_scale` + learnable `resid_mix` (no separate Scylla module) |
| Muon (unlabelled in upstream commit msg) | The actual distinguishing optimizer choice — borrowed from modded-nanogpt |

The matrix below uses the code-grounded labels. The upstream label is kept in parentheses as the bridge term for the paper's Related Work section.

### Ablation Matrix

| Row | Toggle (env var)                  | Default         | Component (upstream label)            | Seeds |
|-----|-----------------------------------|-----------------|---------------------------------------|-------|
| B0  | (none — all defaults)             | —               | Baseline replication                  | 5     |
| A1  | `QUANTIZE_WEIGHTS=none`           | `int8`          | INT8 quantization (≈ "GPTQ")          | 3     |
| A2  | `NUM_KV_HEADS=8`                  | `4`             | GQA 4:1 KV sharing (≈ "XSA")          | 3     |
| A3  | `SDPA_BACKEND=math`               | `flash`         | FlashAttention SDPA backend (≈ "FA3") | 3     |
| A4  | `OPTIMIZER=adamw`                 | `muon`          | Muon optimizer                        | 3     |
| A5  | `QUANT_SCHEME=per_tensor`         | `per_row`       | Per-row scale within INT8 (≈ "GPTQ-4bit precision frontier") | 3     |
| A6  | `TIE_EMBEDDINGS=0`                | `1`             | Embedding tying                       | 3     |
| C1  | `QUANTIZE_WEIGHTS=none` + `NUM_KV_HEADS=8` | —      | Quant × GQA interaction               | 3     |
| C2  | `NUM_KV_HEADS=8` + `SDPA_BACKEND=math`     | —      | GQA × kernel interaction              | 3     |

significance_threshold: p < 0.05 against B0 (Welch's t-test, two-sided)
output_per_run: structured JSON {bpb, train_s, eval_s, artifact_bytes, seed, config_hash}
output_aggregate: ablation summary table for paper Section 4

### Sprint 002 Discipline

- Single-component changes only per row; no compounded edits within an individual ablation.
- Both wallclock budgets (training 600s, eval 600s) enforced — exceeding either disqualifies the row.
- All ablations are env-var toggleable in `train_gpt.py` (no forks). The Hyperparameters class already reads env vars via `os.environ.get(...)`; new toggles follow the same pattern.
- No new dependencies introduced. Each toggle resolves to an alternative already importable: `QUANTIZE_WEIGHTS=none` → bf16/zlib raw passthrough, `NUM_KV_HEADS=8` → MHA via existing GQA path, `SDPA_BACKEND=math` → math kernel via `enable_math_sdp(True)`, `OPTIMIZER=adamw` → `torch.optim.AdamW` for matrix params.
- A5 (per-row vs per-tensor INT8 scale) preserves the post-training quantization paradigm; lower-bit quantization (4-bit/2-bit/ternary/BitNet) is explicitly out-of-scope for Sprint 002 — reserved as the Q3 paper proposal.
- A4 isolates the Muon contribution honestly. The Block-level architectural ablation upstream calls "Scylla minus" was dropped: there is no clean swap for the `resid_mix`/`attn_scale`/`mlp_scale` triplet that doesn't compound multiple changes.

### Time Budget (per submission run, 8×H100 SXM)

training_wallclock_target: 600s
eval_wallclock_target: 600s
combined_run_budget: 1200s

Source: README §194 — "We won't accept submissions that take more than 10 minutes on 8xH100 to evaluate (Note: This limit is in addition to the 10 minutes of training time allowed!)"

Both budgets are independent hard ceilings; exceeding either disqualifies the run. Watchdog Golf Barrier MUST enforce both, not training alone. Eval cost is dominated by FineWeb validation pass — any architecture choice that inflates eval-time inference (e.g., depth recurrence at inference, multi-pass refinement, large beam search) trades against this budget directly.

Preferred architecture patterns (priority order):
1. Weight tying — free parameter reduction
2. LoRA-TTT rank 4–8 on Q, V only
3. Depth recurrence — reuse layer weights across depth
4. BitNet / ternary weights — 1-bit with learned scaling via Triton kernel
5. Grouped Query Attention — 4:1 KV sharing minimum

---

## Section 5 — Source File Hashes

| File                   | MD5 Hash                         | Last verified |
|------------------------|----------------------------------|---------------|
| train_gpt.py           | 48f9a03b9c971f447c2181b821e43da2                 | 2026-04-24 |
| watchdog.py            | 0567086f431b163485a301cfa3f41ef9                 | 2026-04-24 |
| requirements.txt       | 084fe1fc84c2fc35c08497947a45dfff                 | 2026-04-24 |
| .kiro/skills/parameter-golf.md | 823d5ddcd0a5b91a4daccfa8c79a51d2                 | 2026-04-24 |
                                                                                                                                                      