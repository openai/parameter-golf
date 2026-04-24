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

### Blacklist — Three Categories (Logic Drift flags by category)

**Category 1 — Data Sneaking:**
transformers, datasets (if pulling non-approved data), any library that downloads weights on import.
Rationale: bundled pre-trained weights or external datasets violate competition rules.

**Category 2 — External Calls:**
requests, urllib3, httpx, selenium, aiohttp.
Rationale: network calls during evaluation are a disqualification risk.

**Category 3 — Pure Torch Violations:**
numpy, scipy, sklearn, pandas.
Rationale: torch.Tensor → numpy.ndarray triggers CUDA sync + PCIe transfer.
One sync call in a 600-second H100 window is not recoverable.

**Also blocked (visualisation/logging):**
matplotlib, seaborn, plotly, tensorboard, wandb, tqdm.
Rationale: not needed in production training loop; tqdm acceptable in dev but blocked in sprint.

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

sprint_id: SPRINT-001
quantization_target: 6-bit (via auto-gptq or autoawq)
lora_rank: 4 (applied to Q, V projections only)
attention_type: sliding_window
sliding_window_size: TBD — set at sprint start
weight_tying: YES — embedding and output projection share weights
lookahead: FORBIDDEN — any attention pattern requiring future tokens is a violation
no_lookahead_enforcement: strict

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
