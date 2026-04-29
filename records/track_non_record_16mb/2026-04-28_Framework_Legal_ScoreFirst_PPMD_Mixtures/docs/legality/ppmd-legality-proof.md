# PPM-D Legality Proof: exp_1876

**Audit target:** `results/exp_1876_ppmd/prod_8gpu_s42v2/`  
**Source under analysis:** `results/exp_1876_ppmd/train_gpt_merged.py`, function `_ppm_mixture_bpb` (lines 518–555)  
**Machine-checkable tests:** `tests/test_exp1876_ppmd_legality_audit.py`, class `Exp1876PpmdLegalityPhase3NormalizationTest`  
**Audit output:** `audits/exp_1876_ppmd/audit_outputs/normalization_audit.json`

---

## Definitions and Setup

### Notation

- $V = 8192$: SP8192 token vocabulary size
- $\mathcal{B} = \{0, 1, \ldots, 255\}$: byte alphabet
- $p_{\text{NN}}(v \mid \text{prefix})$: neural model's token-level softmax probability for token $v$
- $\text{nll}(i)$: neural model's negative log-likelihood in nats for the realized target token at position $i$
- $\text{bytes}(v)$: UTF-8 byte representation of token $v$ (possibly with leading space)
- $n_b(v) = |\text{bytes}(v)|$: number of bytes in token $v$'s representation
- $p_{\text{PPM}}(b \mid \text{ctx})$: PPM-D byte probability for byte $b$ given context
- $\lambda$: mixture weight (gated by PPM confidence, prefix-only)

### What `_ppm_mixture_bpb` computes

At each token position $i$, the function:

1. Converts token probability to per-byte log-probability:
   $$\text{nn\_log\_p} = \frac{\log p_{\text{NN}}(\text{token}_i)}{n_b(\text{token}_i)}$$

2. Computes PPM-D byte probability for each byte $b_t$ of the target token:
   $$\text{ppm\_log\_p}_t = \log p_{\text{PPM}}(b_t \mid \text{byte\_history}_{<t})$$

3. Forms the mixture (in log-space):
   $$\log p_{\text{mix}}(b_t) = \log\bigl(\lambda \cdot e^{\text{nn\_log\_p}} + (1-\lambda) \cdot e^{\text{ppm\_log\_p}_t}\bigr)$$

4. Accumulates the negative log mixture: `mix_nll -= log_mix`

5. Updates PPM-D counts AFTER scoring.

### Central question

Does $p_{\text{mix}}$ define a valid probability distribution over $\mathcal{B}$ at each byte scoring position $t$? Formally:
$$\sum_{b=0}^{255} p_{\text{mix}}(b \mid \text{state}_t) \stackrel{?}{=} 1$$

---

## Theorem 1: PPM-D with Exclusion Defines a Valid Byte Distribution (PROVED)

**Statement.** For any PPM-D state $S$ (context counts and window) and any order $K \geq 0$, the probability distribution defined by PPM-D with update-exclusion over $\mathcal{B} = \{0, \ldots, 255\}$ satisfies:
$$\sum_{b=0}^{255} p_{\text{PPM}}(b \mid S) = 1$$

**Proof.** We prove normalization by showing that at each backoff level, the mass assigned to seen symbols plus the escape mass equals the incoming mass, and the uniform fallback absorbs all remaining escape mass.

Define the backoff chain from the longest context $c_K$ down to $c_0$ (the empty context), followed by a uniform fallback at level $-1$.

At each level $k$ (processing from $K$ down to $0$):
- Let $E_k$ be the set of bytes already assigned probability at levels $> k$.
- Let $A_k = \{b \in \mathcal{B} \setminus E_k : b \in \text{counts}(c_k)\}$ be the active seen bytes.
- Let $U_k = (\mathcal{B} \setminus E_k) \setminus A_k$ be the active unseen bytes.

**Case 1:** $\text{counts}(c_k)$ does not exist. Pass through to level $k-1$ with unchanged escape mass.

**Case 2:** $A_k = \emptyset$ (all seen bytes at this level are already excluded). Pass through.

**Case 3:** $U_k = \emptyset$ (all remaining unassigned bytes are seen at this level). Distribute all remaining escape mass proportionally by count. No further escape. Total mass consumed = escape mass incoming, and the process terminates.

**Case 4:** $|A_k| > 0$ and $|U_k| > 0$. Let $n_k = \sum_{b \in A_k} \text{counts}(c_k)[b]$ and $d_k = |A_k|$. Then:
- Each $b \in A_k$ gets: $\text{esc} \cdot \frac{\text{counts}(c_k)[b]}{n_k + d_k}$
- Escape to level $k-1$: $\text{esc} \cdot \frac{d_k}{n_k + d_k}$
- Mass consumed at this level: $\text{esc} \cdot \frac{n_k}{n_k + d_k}$
- Check: $\frac{n_k}{n_k + d_k} + \frac{d_k}{n_k + d_k} = 1$ ✓

At the uniform fallback (level $-1$):
- Let $R = \mathcal{B} \setminus \bigcup_k E_k$ be the remaining unassigned bytes.
- Each $b \in R$ gets $\text{esc}_{\text{final}} / |R|$.
- If $|R| = 0$, the escape mass is zero (guaranteed by Case 3 above).

**Totaling:** At each level, incoming escape mass = mass assigned to symbols + outgoing escape mass. By induction from the highest to the lowest level, the total probability assigned to all 256 bytes equals the initial escape mass of 1.0. $\blacksquare$

**Machine-checkable verification:** `synthetic_ppm_d_normalization_test()` enumerates all 256 byte probabilities for 21 diverse synthetic states (7 histories × 3 orders). All sums equal $1.0 \pm 10^{-12}$.

### Note on the actual code implementation

The `_ppm_mixture_bpb` function implements a **simplified PPM-D without exclusion**: it uses a fixed $1/256$ uniform fallback regardless of which bytes were seen at higher context levels. This simplified variant does **not** normalize in general. For example, after history `b"hello"` with order 1, the non-exclusion sum is approximately $0.993$ rather than $1.0$. This is an additional normalization defect beyond the neural component issue proved below.

---

## Theorem 2: Neural Byte Component is NOT a Valid Distribution (PROVED — COUNTEREXAMPLE)

**Statement.** The "neural byte probability" used in `_ppm_mixture_bpb`:
$$\text{nn}(b) = \exp\!\left(\frac{\log p_{\text{NN}}(\text{token})}{n_b}\right) = p_{\text{NN}}(\text{token})^{1/n_b}$$
is **not** a valid probability distribution over $\mathcal{B}$.

**Proof.** The value $\text{nn}(b) = p_{\text{NN}}(\text{token})^{1/n_b}$ is a **constant** with respect to $b$ — it does not depend on which byte $b$ is being queried. Therefore:

$$\sum_{b=0}^{255} \text{nn}(b) = 256 \cdot p_{\text{NN}}(\text{token})^{1/n_b}$$

For this to equal 1, we would need:
$$p_{\text{NN}}(\text{token})^{1/n_b} = \frac{1}{256} \quad \Longleftrightarrow \quad p_{\text{NN}}(\text{token}) = \left(\frac{1}{256}\right)^{n_b}$$

This is not satisfied for general token probabilities.

**Concrete counterexample:** $p_{\text{NN}} = 0.01$, $n_b = 3$:
$$\text{nn}(b) = 0.01^{1/3} \approx 0.2154$$
$$\sum_{b=0}^{255} \text{nn}(b) = 256 \times 0.2154 \approx 55.15 \neq 1$$

**Additional counterexamples** (all verified by `synthetic_neural_byte_counterexample()`):

| $p_{\text{NN}}$ | $n_b$ | $\text{nn}(b)$ | $\sum_{b} \text{nn}(b)$ |
|:---:|:---:|:---:|:---:|
| 0.01 | 3 | 0.2154 | 55.15 |
| 0.1 | 2 | 0.3162 | 80.95 |
| 0.5 | 4 | 0.8409 | 215.27 |
| 0.001 | 5 | 0.2512 | 64.30 |
| 0.99 | 1 | 0.99 | 253.44 |

**Root cause.** The code computes:
```python
token_logp = -float(nll_nats[i])        # log p_NN(target_token)
per_byte_logp = token_logp / n_bytes     # spread uniformly over bytes
```
(Lines 530–531 of `train_gpt_merged.py`)

This is a **geometric mean decomposition** of the realized token's probability, not a conditional byte distribution. It answers "what constant per-byte log-probability, when summed over $n_b$ bytes, recovers the token log-probability?" — not "what is the probability of byte $b$ at position $t$?" $\blacksquare$

---

## Theorem 3: The Mixture is NOT a Normalized Distribution (PROVED — COUNTEREXAMPLE)

**Statement.** The mixture:
$$p_{\text{mix}}(b) = \lambda \cdot \text{nn}(b) + (1-\lambda) \cdot p_{\text{PPM}}(b)$$
is **not** a normalized probability distribution over $\mathcal{B}$.

**Proof.** Summing over all 256 bytes:
$$\sum_{b=0}^{255} p_{\text{mix}}(b) = \lambda \sum_{b} \text{nn}(b) + (1-\lambda) \sum_{b} p_{\text{PPM}}(b)$$

From Theorem 1 (with proper exclusion): $\sum_b p_{\text{PPM}}(b) = 1$.  
From Theorem 2: $\sum_b \text{nn}(b) = 256 \cdot p_{\text{NN}}^{1/n_b} \neq 1$.

Therefore:
$$\sum_{b=0}^{255} p_{\text{mix}}(b) = \lambda \cdot 256 \cdot p_{\text{NN}}^{1/n_b} + (1-\lambda) \neq 1$$

for any $\lambda \in (0, 1)$ and general $p_{\text{NN}}$, $n_b$.

**Concrete counterexample:** $p_{\text{NN}} = 0.01$, $n_b = 3$, $\lambda = 0.9$ (using uniform PPM):
$$\sum_{b} p_{\text{mix}}(b) = 0.9 \times 55.15 + 0.1 \times 1.0 = 49.64 + 0.1 = 49.74 \neq 1$$

The mixture massively overcounts because the neural component assigns $\approx 0.2154$ to each of 256 byte values, far exceeding a proper probability mass. $\blacksquare$

---

## Theorem 4: Score-Before-Update Ordering is Satisfied (PROVED)

**Statement.** In `_ppm_mixture_bpb`, the PPM-D context counts are updated **after** the mixture score is accumulated for each byte.

**Proof.** Static source analysis of `train_gpt_merged.py` (verified in Phase 1, re-confirmed here):

Within the byte loop `for t in range(total_bytes):` (line 540):

1. **Read counts** (line 542): `counts = ctx_counts.get(ctx)` — reads existing PPM-D state
2. **Compute PPM probability** (line 545): `prob_here = counts[b] / denom`
3. **Compute neural probability** (line 547): `nn_log_p = byte_nn_logp[t]`
4. **Accumulate mixture score** (line 551): `mix_nll -= log_mix`
5. **Update PPM-D counts** (line 554): `d[b] = d.get(b, 0) + 1`
6. **Update window** (line 555): `window.append(b)`

Source character offsets confirm strict ordering: score operations (step 4) occur at a lower offset than update operations (steps 5–6). The byte being scored is never used to update the PPM-D state before its own probability is computed. $\blacksquare$

---

## Theorem 5: Byte Denominator is Correct (PROVED — Phase 2)

Proved in Phase 2 (`audits/exp_1876_ppmd/audit_outputs/denominator_audit.json`):
- Full validation target bytes: 151,078,222 (matches known reference)
- First 8M PPM subset bytes: 29,365,687 (matches production log)
- Byte counting uses `base_bytes_lut[target] + (has_leading_space[target] & ~is_boundary_token[prev])`, correctly accounting for SentencePiece `▁` handling.

---

## Theorem 6: Distributed Coverage is Correct (PROVED — Phase 2)

Proved in Phase 2 (`audits/exp_1876_ppmd/audit_outputs/coverage_audit.json`):
- World-size 1 and 8 produce identical scored-position streams (SHA-256 verified)
- No missing or duplicate scored positions in either configuration
- PPM subset selection covers exactly positions 0 through 7,999,999

---

## Overall Verdict

| Proof Obligation | Status | Implication |
|:---|:---:|:---|
| Theorem 1: PPM-D (proper, with exclusion) normalizes | **PROVED** | The mathematical PPM-D framework is sound |
| Code PPM-D (no exclusion) normalizes | **DISPROVED** | The actual implementation leaks probability mass |
| Theorem 2: Neural byte component normalizes | **DISPROVED** | Geometric-mean decomposition ≠ byte distribution |
| Theorem 3: Mixture normalizes | **DISPROVED** | Inherits defects from both components |
| Theorem 4: Score-before-update ordering | **PROVED** | No look-ahead violation |
| Theorem 5: Byte denominator correct | **PROVED** | Denominator accounting is sound |
| Theorem 6: Distributed coverage correct | **PROVED** | No coverage gaps or duplicates |

**The `_ppm_mixture_bpb` function does NOT define a valid probability distribution at each scoring position.** The mixture score is therefore not a proper bits-per-byte metric in the information-theoretic sense.

There are **two independent normalization failures**:

1. **The neural byte component** (Theorem 2) spreads a token-level log-probability uniformly across bytes, producing a constant value for all 256 byte values. This is the geometric mean of the token probability, not a conditional byte distribution.

2. **The code's PPM-D implementation** (noted under Theorem 1) uses a simplified form without exclusion, which also fails to normalize when the byte history has non-trivial context statistics.

Even if the PPM-D implementation were corrected to use proper exclusion (which would fix issue 2), the mixture would still fail to normalize due to the neural component (issue 1).

---

## Implications and Recommendations

### For contest validity

The `mix_bpb = 0.994872` reported by `_ppm_mixture_bpb` is **not a valid compression rate** because the scoring function does not define a proper probability distribution over the byte alphabet at each position. Under the contest rules requiring "a full normalized distribution over the official alphabet" before scoring, this metric cannot be accepted as a record.

### For potential fixes

Two architecturally valid approaches exist (as outlined in `plans/ppm_notes.md`):

**Path A — Token-level normalized PPM mixture:** At each token position, compute $q_t(v) = \prod_{j} p_{\text{PPM}}(b_j(v) \mid \text{history})$ for every token $v \in V$, normalize to get $p_{\text{PPM},t}(v)$, then mix with $p_{\text{NN},t}(v)$. This is expensive ($O(V)$ PPM evaluations per token) but provably valid.

**Path B — Proper byte-level predictor:** Convert the neural model to a true 256-way byte distribution using token-trie marginalization: $p_{\text{NN}}(b \mid \text{byte\_prefix}) = \sum_{v: \text{bytes}(v) \text{ extends prefix}+b} p_{\text{NN}}(v) / \sum_{v: \text{bytes}(v) \text{ extends prefix}} p_{\text{NN}}(v)$. This gives a proper per-byte distribution but requires maintaining a byte-level view of the token vocabulary.

Either path would require significant implementation changes and re-evaluation.
