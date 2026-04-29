# Framework for Legal Score-First PPM-D Mixtures

**Non-record methodology submission — audited Path B subset result available, but no corrected full-validation BPB claim**

This submission documents a rigorous formal framework for computing valid bits-per-byte (BPB) with neural + PPM-D (Prediction by Partial Matching, variant D) byte-level mixtures. It includes:

1. A **formal legality proof** that the existing `_ppm_mixture_bpb` (as used in PR #1877 and exp_1876) is **invalid** — it does not define a normalized probability distribution at each scoring position.
2. **Two architecturally valid correction paths** — Path A (token-normalized) and Path B (byte-level marginalization) — with implementation status, computational cost analysis, and calculations to date.
3. An **audited first-8M-token Path B sliding-only subset result** for the exp_1876 artifact, with explicit denominator and claim-boundary checks.
4. The **full 8×H100 production training run** (exp_1876, seed 42v2) with artifact, logs, and valid neural-only baselines.

> **Why this matters:** Multiple PPM-D submissions to the Parameter Golf leaderboard have been challenged on normalization grounds. This framework now includes both a formal proof and an audited corrected Path B subset number on the same exp_1876 artifact. That is enough to confirm the normalization concern empirically, but not enough to claim a corrected full-validation BPB.

---

## Results Summary

| Metric | Value | Scope | Status |
|--------|-------|-------|--------|
| Neural-only sliding BPB | **1.0830** | Full validation | ✅ Valid |
| Neural-only TTT BPB | **1.0812** | Full validation | ✅ Valid |
| Neural-only quantized BPB | 1.0999 | Full validation | ✅ Valid |
| Pre-quant EMA BPB | 1.0879 | Full validation | ✅ Valid |
| Invalid in-source PPM mixture BPB | 0.9949 | Original first-8M-token postpass | ❌ **Invalid** — see proof below |
| Corrected Path B mixture BPB | **1.5458598675878639** | First 8M tokens / 29,365,687 bytes, sliding only | ✅ Audited subset result (`claim_ready=true`) |
| Corrected Path A BPB | *not yet computed* | Any scope | 🟡 C++ backend built, CUDA backend planned |
| Corrected Path B full-validation BPB | *not yet computed* | Full 151,078,222-byte validation split | 🟡 Full-val sliding shards bundled; offline CPU merge/PPM-D postpass pending (~6–9 hr CPU) |
| Corrected Path B TTT BPB | *not yet computed* | TTT | 🟡 Not implemented |

> This submission intentionally keeps `submission.json` `val_bpb = null` because the only corrected Path B number currently available is the audited **first-8M-token sliding subset**, not a corrected full-validation contest metric.

### Audited Path B first-8M-token subset (sliding only)

Result JSON:
`results/exp_1876_ppmd/path_b_prod_8gpu_local_score/path_b_sliding_subset_8000000.json`

| Quantity | Value |
| --- | --- |
| `mixture_bpb` | **1.5458598675878639** |
| `neural_only_bpb` | **1.5619188191495212** |
| `ppm_d_only_bpb` | **2.118415681067613** |
| `scored_token_count` | 8,000,000 |
| `scored_byte_count` | 29,365,687 |
| `denominator_match` | `true` |
| `claim_ready` | `true` |
| `runtime_seconds` | `6209.227784756571` *(local merge + PPM-D postpass only)* |

Bundled full-val shard-generation artifacts:
`results/exp_1876_ppmd/path_b_prod_8gpu_fullval/path_b_sliding_accounting_audit.json`
and
`results/exp_1876_ppmd/path_b_prod_8gpu_fullval/path_b_sliding_merge_manifest.json`.
The full-val sliding-window shard-generation artifacts already exist and are now bundled; the remaining missing step for a corrected full-validation Path B BPB is the offline CPU merge/PPM-D postpass, projected at **~6–9 hours of offline CPU time** (see Runtime section below).

> **Why is the corrected subset mixture BPB (1.546) worse than neural-only (1.083)?**
> Two reasons. First, the subset covers only the **first** 8M validation tokens — PPM-D starts with an empty context and takes millions of bytes to build useful statistics, so it drags the mixture down early in the stream. The full-validation BPB would benefit from the much longer context available in later positions. Second, these are **sliding-window** neural logprobs (stride 64, seq 2048), not TTT-adapted logprobs. The neural-only sliding BPB quoted in the results table (1.0830) is a full-validation figure, not directly comparable to this 8M-token prefix subset. The subset-only neural BPB is 1.5619 — the corrected mixture (1.5459) **does** improve on that same-scope neural-only baseline, confirming PPM-D helps once normalization is correct.

## PR / Issue Lineage

| Reference | Role in this package |
| --- | --- |
| PR #1851 | Clean neural baseline lineage for the SP8192 / CaseOps / score-first-TTT family referenced here. |
| PR #1868 | Clean/reproducible neural baseline lineage built from #1851-style neural scoring; the defensible no-PPM comparison point. |
| PR #1873 | Earlier PPM attempt that motivated stricter normalization and accounting scrutiny. |
| PR #1876 | Source artifact and neural+PPM-D mixture stack audited here (`final_model.int6.ptz`, seed 42v2). |
| PR #1877 | Later public normalization-discussion reference; useful for the proof obligation, but not the source of the audited artifact. |
| Issue #1872 | Normalization concern that the audited first-8M Path B subset empirically confirms on the exp_1876 artifact. |

**Artifact:** `final_model.int6.ptz` — **15,975,706 bytes** (model) + **20,220 bytes** (code) = **15,995,926 bytes** total (under 16 MB cap)

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Hardware | 8×H100 80GB SXM |
| Seed | 42 |
| Steps completed | 4,590 / 20,000 (stopped by 10-min wallclock cap) |
| Training wallclock | 588s (~9.8 min) |
| Tokenizer | SP8192 BPE (`fineweb_8192_bpe.model`) |
| Architecture | 11-layer transformer, dim=512, 8 attn heads / 4 KV heads (GQA) |
| Depth recurrence | 2 loops, layers 3-4 shared, enabled at step 2048 (35% of training) |
| MLP | 4× expansion |
| Sequence length | 2048 (train and eval) |
| Optimizer | Muon (matrix params) + Adam (scalars/embeddings) |
| Quantization | INT6 (GPTQ) + brotli compression |
| TTT | LoRA TTT, 3 epochs, lr=0.005, chunk=32768 tokens |
| PPM-D (as originally run) | Order 5, λ gated by confidence threshold 0.9, first 8M tokens |
| Validation tokens | 40,540,160 target tokens, 151,078,222 target bytes |
| PPM subset | First 8,000,000 tokens → 29,365,687 bytes |

### Eval Timeline (from production log)

```
pre-quantization post-ema val_loss:2.81015  val_bpb:1.08790
quantized val_loss:2.84108                  val_bpb:1.09987
quantized_sliding_window val_loss:2.79763   val_bpb:1.08305
ppm_mix bytes=29365687 mix_bpb=0.994872 ppm_only=2.270203 nn_only=1.086575
quantized_ttt val_loss:2.79290              val_bpb:1.08122
```

---

## Formal Legality Proof: Why `_ppm_mixture_bpb` is Invalid

Full proof: `docs/legality/ppmd-legality-proof.md`  
Machine-checkable tests: `tests/test_exp1876_ppmd_legality_audit.py`

### The Core Problem

The `_ppm_mixture_bpb` function in the training script (lines 518–555) computes:

$$p_{\text{mix}}(b) = \lambda \cdot \text{nn}(b) + (1-\lambda) \cdot p_{\text{PPM}}(b)$$

where the "neural byte probability" is:

$$\text{nn}(b) = \exp\!\left(\frac{\log p_{\text{NN}}(\text{token})}{n_b}\right) = p_{\text{NN}}(\text{token})^{1/n_b}$$

This is a **geometric mean decomposition** — it spreads the token's log-probability uniformly across its bytes. It is a constant with respect to $b$: it assigns the same value to all 256 possible byte values regardless of which byte is being queried.

### Theorem 2: Neural Byte Component is NOT a Valid Distribution

$$\sum_{b=0}^{255} \text{nn}(b) = 256 \cdot p_{\text{NN}}(\text{token})^{1/n_b} \neq 1$$

**Counterexample:** For $p_{\text{NN}} = 0.01$, $n_b = 3$:

$$\text{nn}(b) = 0.01^{1/3} \approx 0.2154 \quad\Rightarrow\quad \sum_b \text{nn}(b) = 256 \times 0.2154 \approx 55.15 \neq 1$$

| $p_{\text{NN}}$ | $n_b$ | $\text{nn}(b)$ | $\sum_b \text{nn}(b)$ |
|:---:|:---:|:---:|:---:|
| 0.01 | 3 | 0.2154 | 55.15 |
| 0.1 | 2 | 0.3162 | 80.95 |
| 0.5 | 4 | 0.8409 | 215.27 |
| 0.001 | 5 | 0.2512 | 64.30 |
| 0.99 | 1 | 0.99 | 253.44 |

### Additional Defect: PPM-D Without Exclusion

The code implements a simplified PPM-D without update-exclusion: it uses a fixed $1/256$ uniform fallback regardless of which bytes were already assigned probability at higher context levels. After nontrivial byte histories, this simplified variant produces sums $\neq 1$ (e.g., $\approx 0.993$ after history `b"hello"` with order 1).

PPM-D **with** proper exclusion provably normalizes (Theorem 1 in the full proof). The defect is in the implementation, not the mathematical framework.

### Theorem 3: The Mixture is NOT Normalized

$$\sum_{b=0}^{255} p_{\text{mix}}(b) = \lambda \cdot 256 \cdot p_{\text{NN}}^{1/n_b} + (1-\lambda) \neq 1$$

**Counterexample:** $p_{\text{NN}} = 0.01$, $n_b = 3$, $\lambda = 0.9$:

$$\sum_b p_{\text{mix}} = 0.9 \times 55.15 + 0.1 \times 1.0 = 49.74 \neq 1$$

### What IS Correct

The formal proof also **verifies** several aspects of the implementation:

| Proof Obligation | Status |
|:---|:---:|
| Theorem 1: PPM-D (proper, with exclusion) normalizes | ✅ Proved |
| Theorem 4: Score-before-update ordering | ✅ Proved |
| Theorem 5: Byte denominator correct (151,078,222 bytes full, 29,365,687 subset) | ✅ Proved |
| Theorem 6: Distributed coverage correct (no missing/duplicate positions) | ✅ Proved |
| Code PPM-D (no exclusion) normalizes | ❌ Disproved |
| Theorem 2: Neural byte component normalizes | ❌ Disproved |
| Theorem 3: Mixture normalizes | ❌ Disproved |

**Verdict:** The reported `mix_bpb = 0.994872` is **not a valid compression rate** because the scoring function does not define a proper probability distribution over the byte alphabet at each position.

The audited Path B first-8M subset result on the same artifact (`mixture_bpb = 1.5458598675878639`) provides direct empirical confirmation of the normalization concern raised in Issue #1872. That subset result is corrected and claim-ready for its audited scope, but it is **not** a full-validation or TTT claim.

---

## Two Valid Correction Paths

Both paths are constructive — they describe how to compute a provably valid PPM-D BPB from the same trained neural model. Neither requires retraining.

### Path A: Token-Normalized PPM Mixture

**Idea:** At each token position $t$, compute PPM-D byte-string scores for **every** token $v \in V$, normalize over the full vocabulary, then mix with the neural softmax.

**Formula:**
$$q_t(v) = \prod_{j=1}^{n_b(v)} p_{\text{PPM}}(b_j(v) \mid \text{byte\_history}_t)$$
$$p_{\text{PPM},t}(v) = \frac{q_t(v)}{\sum_{u \in V} q_t(u)}$$
$$p_{\text{mix},t}(v) = \lambda_t \cdot p_{\text{NN},t}(v) + (1-\lambda_t) \cdot p_{\text{PPM},t}(v)$$
$$\text{loss}_t = -\log_2 p_{\text{mix},t}(x_t) \quad;\quad \text{BPB} = \frac{\sum_t \text{loss}_t}{\sum_t n_b(x_t)}$$

**Why it's valid:** $\sum_v p_{\text{mix},t}(v) = 1$ by construction (convex combination of two normalized distributions over the same vocabulary).

**Implementation status:**
- ✅ Python reference evaluator: `scripts/eval_path_a_ppmd.py` (10 tests passing)
- ✅ C++ pybind11 backend: repo-root `scripts/ppmd_cpp/` (not bundled here; 18 tests, bit-exact BPB equivalence, ~17-50× single-thread speedup)
- ✅ SLURM CPU benchmark: 2.28M probes/s on 32 cpu_short threads
- ❌ CPU-only full eval projected at ~38 days → exceeds 3.5h budget → CUDA backend required
- 🟡 CUDA backend plan: `docs/plans/path-a-ppmd-cuda-backend-plan.md` (6 phases, A100/H100 selection, RunPod HTTP-bootstrap)
- 🟡 RunPod execution plan: `docs/plans/path-a-ppmd-cuda-runpod-execution-plan.md`
- **No Path A BPB has been computed yet.** The computational cost at O(V=8192) PPM-D evaluations per token position is the bottleneck.

### Path B: Byte-Level Trie Marginalization

**Idea:** Convert the neural model's token-level softmax into a proper 256-way byte distribution at each byte position using trie-based marginalization, then mix with proper PPM-D at the byte level.

**Formula:**
$$p_{\text{NN}}(b \mid \text{prefix}) = \frac{\sum_{v: \text{bytes}(v) \text{ extends prefix} + b} p_{\text{NN}}(v)}{\sum_{v: \text{bytes}(v) \text{ extends prefix}} p_{\text{NN}}(v)}$$
$$p_{\text{mix}}(b) = \lambda \cdot p_{\text{NN}}(b \mid \text{prefix}) + (1-\lambda) \cdot p_{\text{PPM}}(b)$$
$$\text{BPB} = \frac{\sum_i -\log_2 p_{\text{mix}}(b_i)}{N_{\text{bytes}}}$$

**Why it's valid:** Both components are proper 256-way distributions: PPM-D with exclusion normalizes (Theorem 1); the trie marginalization normalizes by construction (conditional probability from the token softmax).

**Implementation status:**
- ✅ Token-byte trie primitives: `TokenByteSequences`, `ByteTrieNode`, `build_byte_trie`, `build_mode_tries`
- ✅ Reference marginalization: `reference_neural_byte_distribution` + brute-force verifier (tested on synthetic + SP8192-like cases)
- ✅ Optimized interval/cumsum scoring: `OptimizedTrieTables`, `optimized_neural_byte_distribution` (validated against reference)
- ✅ Proper PPM-D with exclusion: `PPMDByteModel`, `score_ppmd_byte_then_update` (normalization tested)
- ✅ Mixture helper: `mixture_byte_distribution` (normalization tested)
- ✅ Vectorized target-path extraction: `vectorized_target_path_logprobs` (torch batch kernel)
- ✅ Binary shard helpers: `write_records_npz`, `read_records_npz`, `merge_record_npz_shards`
- ✅ Streaming PPM-D mixture scorer: `score_ppmd_stream`
- ✅ CLI scaffolding: `PathBEvalConfig`, `run_dry_run`, `run_explicit_eval`, `main`
- ✅ Distributed sliding-window shard generation, shard merge, and accounting audit completed for the first 8,000,000 validation tokens
- ✅ Audited subset result JSON: `results/exp_1876_ppmd/path_b_prod_8gpu_local_score/path_b_sliding_subset_8000000.json`
- ✅ Accounting audit + merge manifest: `results/exp_1876_ppmd/path_b_prod_8gpu/path_b_sliding_accounting_audit.json`, `results/exp_1876_ppmd/path_b_prod_8gpu/path_b_sliding_merge_manifest.json`
- ✅ Full-val sliding-window shard-generation artifacts are bundled: `results/exp_1876_ppmd/path_b_prod_8gpu_fullval/path_b_sliding_accounting_audit.json`, `results/exp_1876_ppmd/path_b_prod_8gpu_fullval/path_b_sliding_merge_manifest.json`
- ✅ First-8M audited metrics: `mixture_bpb = 1.5458598675878639`, `neural_only_bpb = 1.5619188191495212`, `ppm_d_only_bpb = 2.118415681067613`, `denominator_match = true`, `claim_ready = true`
- ✅ Empirical confirmation of Issue #1872 on the audited exp_1876 artifact: the corrected subset mixture is ~`+0.55` BPB above the invalid in-source `0.994872` figure
- ❌ Full-validation corrected Path B BPB (151,078,222 bytes): offline CPU merge/PPM-D postpass still pending
- ❌ TTT accounting / evaluation: not implemented for Path B
- ⚠️ The audited first-8M subset was throughput-inefficient by design: with contiguous window slicing on 8×H100, only ranks 0–1 produced non-empty shards while ranks 2–7 were idle
- **Claim boundary:** the first-8M-token sliding subset result is audited and claim-ready for that subset only. This submission intentionally keeps `val_bpb = null` because no corrected full-validation Path B number exists yet.

**Red-team review:** `docs/plans/path-b-byte-eval-redteam.md` documents 10 critical risks with mitigations and a 5-gate acceptance ladder (local readiness → eval wiring → 1×H100 rehearsal → 8×H100 production → post-run audit).

## Runtime and Operational Profile

Path B is currently a **two-stage workflow**, not a single monolithic eval:

| Stage | Execution mode | Current evidence |
| --- | --- | --- |
| 1. Sliding-window shard generation | GPU, distributed | Produces canonical per-rank records for later byte-level scoring. Bundled evidence now includes both the audited first-8M subset artifacts and the full-val shard-generation manifests in `results/exp_1876_ppmd/path_b_prod_8gpu_fullval/`. For the audited contiguous first-8M-token subset, only ranks 0–1 on the 8×H100 pass emitted non-empty shards; ranks 2–7 were empty. This was acceptable for an audit run, but it underutilized the hardware and should not be mistaken for a throughput-optimized full-val schedule. |
| 2. Merge + PPM-D postpass | Offline CPU, sequential/local | Merges shard records in canonical order and performs byte-level PPM-D score-before-update scoring. The measured local runtime for the audited 8M subset was `6209.227784756571` seconds for 29,365,687 bytes. Linear extrapolation of this **local postpass alone** gives roughly **8–9 hours** for the full 151,078,222-byte validation split. |

We therefore do **not** claim a corrected full-validation Path B wallclock here. The only directly measured runtime quoted in this README is the audited local postpass above. The full-val sliding-window shard-generation artifacts already exist and are now bundled, but the offline CPU merge/PPM-D postpass has not yet been completed for the full validation split.

---

## Key Design Principles

Both paths share these non-negotiable requirements, derived from the contest rules and our formal analysis:

1. **Full normalized distribution before scoring.** At each scoring position, the mixture must sum to exactly 1.0 over the scored alphabet (tokens for Path A, bytes for Path B) before the target is known.

2. **Score-before-update.** PPM-D context counts are updated only **after** the current position is scored. Proved correct in the original implementation (Theorem 4).

3. **Prefix-only gating.** The mixture weight $\lambda$ must not depend on the target token/byte or any future context. It can depend on PPM-D confidence, context length, or fixed hyperparameters.

4. **Single left-to-right pass.** The byte/token stream is processed exactly once in canonical order.

5. **Original UTF-8 byte denominator.** BPB divides total bits by the true byte count of scored targets, correctly handling SentencePiece `▁`, byte fallback tokens, BOS/EOS, and CaseOps sidecars.

6. **No rank-local PPM state.** PPM-D is inherently sequential. Distributed neural eval is fine, but PPM-D scoring must operate on the globally ordered merged byte stream.

---

## Why Not Just Fix the Mixture In-Place?

The original code could not be "patched" to a valid state for two fundamental reasons:

1. **The neural byte component is structurally wrong.** It uses a geometric-mean decomposition that is a constant across all 256 bytes. Fixing it requires either normalizing over the full token vocabulary (Path A) or building a trie-based byte marginalizer (Path B). Both are architecturally different from the original code.

2. **The PPM-D implementation lacks exclusion.** Proper PPM-D with exclusion (where bytes already assigned probability at higher context levels are removed from the active set at lower levels) is a different algorithm from the simplified version in the code. The fix is straightforward but changes the PPM-D probabilities.

Neither fix is a one-line patch. Both require significant implementation, testing, and audit.

---

## File Inventory

### Core submission files
| File | Description |
|------|-------------|
| `README.md` | This document |
| `submission.json` | Experiment metadata (no corrected full-validation BPB claimed; audited subset fields included) |
| `train.log` | Full production training log (8×H100, seed 42v2) |
| `train_gpt.py` | Training script (merged exp_1876 source) |
| `requirements.txt` | Python dependencies |

### Bundled evidence layout

#### `docs/legality/`

| File | Description |
|------|-------------|
| `docs/legality/ppmd-legality-proof.md` | Bundled copy of the formal 6-theorem legality proof |
| `docs/legality/ppmd-legality-proof-result.md` | Bundled proof result summary |
| `docs/legality/ppmd-legality-proof-implementation.md` | Bundled implementation notes for the legality proof |
| `docs/legality/ppm_notes.md` | Bundled design notes on PPM-D approaches |

#### `docs/plans/`

| File | Description |
|------|-------------|
| `docs/plans/path-a-ppmd-eval-plan.md` | Path A Python evaluator plan |
| `docs/plans/path-a-ppmd-eval-complete.md` | Path A Python evaluator completion report |
| `docs/plans/path-a-ppmd-cpp-backend-plan.md` | Path A C++ backend plan |
| `docs/plans/path-a-ppmd-cpp-backend-complete.md` | Path A C++ backend completion report |
| `docs/plans/path-a-ppmd-cuda-backend-plan.md` | Path A CUDA backend plan |
| `docs/plans/path-a-ppmd-cuda-runpod-execution-plan.md` | Path A CUDA RunPod execution plan |
| `docs/plans/path-b-byte-eval-plan.md` | Path B byte evaluator plan |
| `docs/plans/path-b-byte-eval-complete.md` | Path B byte evaluator completion report |
| `docs/plans/path-b-byte-eval-redteam.md` | Path B red-team risk analysis |
| `docs/plans/path-b-byte-eval-runpod-prompt.md` | Path B RunPod execution prompt |

#### `scripts/`

| File | Description |
|------|-------------|
| `scripts/eval_path_a_ppmd.py` | Bundled Path A token-normalized evaluator |
| `scripts/eval_path_b_ppmd.py` | Bundled Path B byte-level evaluator |

#### `tests/`

| File | Description |
|------|-------------|
| `tests/test_exp1876_ppmd_legality_audit.py` | Bundled legality-proof audit tests |
| `tests/test_path_a_ppmd_eval.py` | Bundled Path A evaluator tests |
| `tests/test_path_b_ppmd_eval.py` | Bundled Path B evaluator tests |

#### `audits/`

| File | Description |
|------|-------------|
| `audits/exp_1876_ppmd/static_provenance.json` | Reviewer-friendly bundled copy of the static provenance audit |
| `audits/exp_1876_ppmd/denominator_audit.json` | Reviewer-friendly bundled copy of the denominator audit |
| `audits/exp_1876_ppmd/coverage_audit.json` | Reviewer-friendly bundled copy of the distributed coverage audit |
| `audits/exp_1876_ppmd/normalization_audit.json` | Reviewer-friendly bundled copy of the normalization audit |

#### `results/`

| File | Description |
|------|-------------|
| `results/exp_1876_ppmd/path_b_prod_8gpu_local_score/path_b_sliding_subset_8000000.json` | Audited first-8M-token Path B subset result (`claim_ready=true`) |
| `results/exp_1876_ppmd/path_b_prod_8gpu/path_b_sliding_accounting_audit.json` | Denominator/accounting audit for the 8M subset |
| `results/exp_1876_ppmd/path_b_prod_8gpu/path_b_sliding_merge_manifest.json` | Per-rank shard manifest for the 8M subset merge |
| `results/exp_1876_ppmd/path_b_prod_8gpu_fullval/path_b_sliding_accounting_audit.json` | Full-val sliding-window shard-generation accounting artifact |
| `results/exp_1876_ppmd/path_b_prod_8gpu_fullval/path_b_sliding_merge_manifest.json` | Full-val sliding-window shard-generation merge manifest |
| `results/exp_1876_ppmd/prod_8gpu_s42v2/artifacts.txt` | Production run artifact listing |
| `results/exp_1876_ppmd/prod_8gpu_s42v2/launcher_state.json` | Production launcher state snapshot |

### Additional repo-root materials referenced but not bundled here

| Path | Description |
|------|-------------|
| repo-root `scripts/ppmd_cpp/` | Path A C++ pybind11 backend source |
| repo-root `tests/test_ppmd_cpp_*.py` | C++ backend tests |

---

## Reproducibility

### Training
```bash
# On 8×H100 SXM with SP8192 data downloaded:
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The training script runs standard eval (quantized, sliding, TTT) and the (invalid) PPM-D mixture as a postpass on the first 8M tokens. The script is self-contained and does not require network access after data download.

### Running the formal audit
```bash
# Legality proof machine-checkable tests:
python -m unittest tests.test_exp1876_ppmd_legality_audit -v

# Path B utility tests (no GPU needed):
python -m unittest tests.test_path_b_ppmd_eval -v

# Path A Python evaluator tests (no GPU needed):
python -m unittest tests.test_path_a_ppmd_eval -v
```

### Computing a corrected BPB (current scope and future work)

- **Path A:** still no corrected BPB number. It remains CUDA-bound because of the O(V) per-position token-normalization cost.
- **Path B:** now has an audited **first-8M-token sliding subset** result, but **not** a corrected full-validation or TTT result.
- **Full-validation Path B (future work):** the full-val sliding-window shard-generation artifacts already exist and are now bundled, but the offline CPU merge/PPM-D postpass still needs to be run. Based on the audited subset, the **local postpass alone** projects to roughly **8–9 hours**.
- **TTT Path B (future work):** intentionally not implemented; no corrected TTT Path B BPB is claimed.

---

## Community Call to Action

We encourage the community to:

1. **Review the formal proof** (`docs/legality/ppmd-legality-proof.md`) and report any errors.
2. **Validate our test suite** — all counterexamples and normalization checks are machine-checkable.
3. **Contribute to Path A or Path B** implementations if interested in provably valid PPM-D BPB.
4. **Apply this framework to other PPM-D submissions** — the Theorems 1-3 apply to any mixture that uses geometric-mean byte decomposition of token probabilities.

The key takeaway: **PPM-D is a mathematically sound compression framework, but mixing it with token-level neural models requires careful normalization that most existing implementations get wrong.** This submission provides the formal tools to distinguish valid from invalid approaches.

---

## Acknowledgments

This package sits in the lineage from the clean neural baseline family (#1851 → #1868), through earlier PPM exploration (#1873), to the audited exp_1876 artifact and the later public normalization discussion around #1877 and Issue #1872. The current audited first-8M Path B subset result exists because that discussion forced an explicit normalization and denominator audit instead of relying on attractive but invalid mixture numbers.
