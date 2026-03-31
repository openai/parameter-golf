# Cambrian: Biology Concepts × DeltaNet Chunk Seams

**Premise:** Standard attention has no natural injection points — it's one flat pass with no "between" moments. DeltaNet's chunked recurrent processing creates **seams**: moments where the model must decide what state to carry forward. Our biology concepts were designed for exactly these decisions. This is the architecture where they belong.

**Target:** Beat PR #875 (1.0226 BPB) using DeltaNet recurrence + our Muon + XSA + n-gram stack + bio seam controllers.

---

## Base Architecture (Cambrian-0)

```
Input
  ↓
[GatedDeltaNet × 8]   ← chunked recurrence, state passes between chunks
  ↓                         chunk size: 64 → 128 → 256 (curriculum)
[XSA Attention × 3]   ← our cross-sequence attention on top
  ↓
Muon optimizer        ← our parallel Newton-Schulz
  ↓
entropy-adaptive 9-gram eval  ← our combined BPB metric
```

At every chunk boundary (the **seam**):
```
S_c → [seam operations] → S_{c+1}
```

This seam is where all four bio concepts inject their signal.

---

## The Delta Rule (Reference)

```
S_t = S_{t-1} + v_t ⊗ k_t - (S_{t-1} k_t) ⊗ k_t
         ↑                      ↑
      WRITE new value         ERASE old value at k_t
```

- S_t: recurrent state matrix (d_k × d_v) — the compressed memory
- k_t: key vector — WHERE to read/write/erase
- v_t: value vector — WHAT to write
- The erase term `(S_{t-1} k_t) ⊗ k_t` removes whatever was stored at k_t

Between chunks: S is passed forward intact. At seams: biology intervenes.

---

## Concept 1: Astrocyte Seam Controller

### Hypothesis
Astrocytes in biology regulate synaptic strength based on local chemical activity. At each chunk seam, a tiny network reads the outgoing state's activation profile and modulates how aggressively the next chunk erases vs preserves memory. Dense/active chunks (lots written) → preserve. Sparse/repetitive chunks → allow more erasure.

### Mechanism
```python
# At seam after chunk c:
state_norms = S_c.norm(dim=-1)          # (d_k,) — what's stored and how strongly
state_summary = state_norms / state_norms.max().clamp(min=1e-6)
astro_scales = AstrocyteNet(state_summary)  # (num_delta_layers,) in (0.5, 1.5)

# Scale the β (erase gate) in next chunk per layer:
β_next[layer] = β_base * astro_scales[layer]
```

AstrocyteNet: `Linear(d_k → d_k//φ → num_delta_layers)`, init output near 1.0.

### Why this helps
The default erase gate β is fixed per-position. The astrocyte makes it dynamic per-chunk: it can "stiffen" memory during hard passages (rare tokens, complex syntax) and "loosen" it during easy ones (repetitive structure). The model learns WHEN to be a good student vs when to trust its existing memory.

### Ablation Ladder
| ID | Config | Expected delta vs baseline |
|----|--------|---------------------------|
| A0 | Cambrian-0 (DeltaNet, no bio) | — |
| A1 | + Astrocyte → scales erase gate β only | -0.005 to -0.015 |
| A2 | + Astrocyte → scales erase AND write gates | -0.010 to -0.025 |
| A3 | + Astrocyte reads full state summary (not just norms) | -0.015 to -0.030 |
| A4 | + Astrocyte with oracle pull (slow EMA of past scales) | unknown |

---

## Concept 2: Myelin Fibonacci Chunk Bridges

### Hypothesis
Myelin in biology wraps specific axons — not all of them — to dramatically speed signal transmission along long-range pathways. Fibonacci-spaced chunk seams get a direct residual bridge that bypasses the compression step. Information that needs to travel far through the sequence doesn't have to fight through repeated erase/write cycles at every seam — it has a fast highway.

### Mechanism
```python
PHI = (1 + 5**0.5) / 2
fibonacci_seams = {1, 2, 3, 5, 8, 13, 21, 34}  # chunk indices

# At seam after chunk c:
if c in fibonacci_seams:
    # Fibonacci bridge: add direct residual to state
    S_next = S_c + skip_weight * H_c_mean.unsqueeze(-1)  # H_c_mean: mean hidden (d_k,)
else:
    S_next = S_c  # normal delta update, no bridge
```

`skip_weight`: learned scalar, initialized to 0.0 (starts as pure DeltaNet, grows as needed).

### Why this helps
DeltaNet compresses everything through the bottleneck of k/v projections. Long-range dependencies that don't fit in the key-value geometry get lost. The Fibonacci bridges create bypass lanes that preserve the raw hidden state summary across those specific seams. Irrational Fibonacci spacing means these lanes appear at irregular intervals — no periodic pattern for the model to exploit.

Non-Fibonacci seams get standard delta compression. The model learns to route long-range signal through the bridges and short-range signal through the delta state.

### Ablation Ladder
| ID | Config | Expected delta vs baseline |
|----|--------|---------------------------|
| M0 | Cambrian-0 | — |
| M1 | + Fibonacci bridges, fixed weight=0.0 init | ~0 (sanity check) |
| M2 | + Fibonacci bridges, learned weight | -0.005 to -0.020 |
| M3 | + Fibonacci bridges + non-Fib erase boost (β×1.2) | -0.010 to -0.025 |
| M4 | Replace Fibonacci with uniform bridges (every seam) | control — expect worse |
| M5 | Replace Fibonacci with random-spaced bridges (same count) | control — expect worse |

M4 and M5 are controls to verify Fibonacci spacing specifically matters (not just "any bridges").

---

## Concept 3: Clonal Selection State Amplification

### Hypothesis
In immunology, clonal selection amplifies B-cells that successfully bind an antigen — rare specific patterns get amplified while common ones are pruned. In the DeltaNet state, some key positions accumulate high norm (they've been written to strongly and rarely erased) — these are the "specialist" memories. At each seam, amplify the top-K specialist positions before passing to the next chunk.

### Mechanism
```python
K = round(d_k / PHI**5)  # ≈ d_k / 11.09 — specialist fraction from φ

# At seam after chunk c:
state_norms = S_c.norm(dim=-1)          # (d_k,) — strength of each memory slot
topk_vals, topk_idx = state_norms.topk(K)
clonal_mask = torch.zeros(d_k, device=S_c.device)
clonal_mask[topk_idx] = 1.0

# Amplify specialists:
S_next = S_c * (1.0 + clonal_scale * clonal_mask.unsqueeze(-1))
```

`clonal_scale`: learned scalar, init 0.0 (no effect at start).

### Why this helps
The erase gate treats all memory slots equally — it erases proportional to query similarity regardless of how important that slot is. Clonal selection breaks this symmetry: slots that have been strongly written and rarely queried are clearly encoding rare, important patterns. Boosting them before the next chunk ensures they survive and remain accessible for the hard tokens ahead.

### Ablation Ladder
| ID | Config | Expected delta vs baseline |
|----|--------|---------------------------|
| C0 | Cambrian-0 | — |
| C1 | + Clonal amplification, K=fixed(d_k//11), scale=fixed(0.1) | -0.003 to -0.010 |
| C2 | + Clonal amplification, learned scale | -0.005 to -0.015 |
| C3 | + Clonal amplification + bottom-K suppression | -0.008 to -0.020 |
| C4 | + Adaptive K (proportional to chunk entropy) | -0.010 to -0.025 |
| C5 | K=all (amplify everything equally) | control — expect ~0 |

C5 is the null control: if amplifying everything equally works as well as top-K, it's not clonal selection, just a scale.

---

## Concept 4: Circadian φ-Gated State Flow

### Hypothesis
Circadian rhythms in biology use irrational phase relationships to prevent synchronization lock-in — different biological systems oscillate at φ-related frequencies so they never perfectly align and create pathological resonance. DeltaNet's recurrent state can lock into periodic attractors if the model learns to rely on fixed-period patterns. A φ-spaced gate on state magnitude at each seam prevents this.

### Mechanism
```python
PHI = (1 + 5**0.5) / 2

# Precompute base phases (fixed, irrational spacing):
# base_phase[c] = 2π × φ × c / total_chunks

# At seam after chunk c:
gate = 1.0 + tanh(amp) * cos(base_phase[c] + learned_phase)
# amp: learned scalar, init 0.0 → gate starts at 1.0 (no effect)
# learned_phase: learned scalar per layer, init 0.0

S_next = S_c * gate
```

### Why this helps
Without gating, the recurrent state can settle into a regime where certain key positions are always active or always inactive (periodic attractor). The φ-spaced gate applies a gentle varying modulation that disrupts these locked states without destroying information (gate stays near 1.0 due to tanh + zero init). The irrationality of φ guarantees no two chunks have the same gate value — the model can't learn to exploit a periodic pattern.

### Ablation Ladder
| ID | Config | Expected delta vs baseline |
|----|--------|---------------------------|
| R0 | Cambrian-0 | — |
| R1 | + φ-gate, fixed amp=0.05 | -0.002 to -0.008 |
| R2 | + φ-gate, learned amp + phase | -0.005 to -0.015 |
| R3 | + per-layer φ-gate (each DeltaNet layer own phase) | -0.008 to -0.020 |
| R4 | φ → 2 (rational spacing control) | control — expect worse than R2 |
| R5 | φ → random phases (not learned) | control |

R4 is critical: if integer-spaced gates work as well as φ-spaced, the irrationality argument is wrong.

---

## Full Ablation Ladder (Cambrian-N)

Clean sequential build to isolate each contribution:

| ID | Architecture | Target BPB |
|----|-------------|-----------|
| C0 | DeltaNet baseline (8×GDN + 1×Attn, our Muon) | ~1.10 |
| C1 | C0 + Myelin Fibonacci bridges | ~1.09 |
| C2 | C1 + Circadian φ-gate | ~1.08 |
| C3 | C2 + Clonal Selection top-K | ~1.07 |
| C4 | C3 + Astrocyte seam controller | ~1.06 |
| C5 | C4 + XSA cross-sequence attention | ~1.04 |
| C6 | C5 + entropy-adaptive 9-gram eval | ~0.44 |

C6 is the submission-legal combined score. If each bio concept contributes even half its expected delta, C6 beats our current SOTA (0.4489 ngram9) by a meaningful margin.

---

## Implementation Order

1. **Port DeltaNet kernel** from PR #875 (chunk recurrence + state passing) — this is the foundation
2. **Cambrian-0**: DeltaNet + our Muon + eval stack, verify beats green baseline
3. **Add Myelin** (M2): simplest seam operation, good sanity check
4. **Add Circadian** (R2): gate on top of Myelin
5. **Add Clonal** (C2): amplification on top
6. **Add Astrocyte** (A2): seam controller last (most complex, depends on stable state dynamics)
7. **Full ablation run** on H100 once stack is verified

---

## Connection to PR #875

We are NOT copying their code. We are:
- Studying their chunked kernel approach for the recurrence mechanism
- Adding bio seam controllers that they don't have
- Keeping our Muon optimizer (they use AdamW)
- Keeping our XSA + entropy-adaptive n-gram eval (they have neither)
- Using our data pipeline and tokenizer

Their 1.0226 was achieved without any of our eval stack. Adding our n-gram system to a Cambrian model should push the combined score substantially below 0.44.
