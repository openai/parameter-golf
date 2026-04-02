# Non-Record: Text Diffusion (CDM) + Retrodiction + Coarse-to-Fine Dual Brain

Wishlist items addressed: **Text Diffusion**, **TTT**, **Depth Recurrence**

---

## 1. AR + Retrodiction (v4096 tokenizer)

| Config | val_bpb | +TTT | Size |
|--------|---------|------|------|
| 5L d=256 v4096 | **1.497** | **1.492** | 2.9MB |

**Retrodiction**: each step trains forward AR + reversed-sequence AR simultaneously.

```python
loss = causal_lm_loss(model, tokens) + 0.3 * causal_lm_loss(model, tokens.flip(1))
```

Inspired by the Petz recovery map from quantum information theory — a model that predicts both directions learns more robust representations. At eval, only forward is used.

- Paper: [Petz Recovery Map](https://github.com/akaiHuang/petz-recovery-unification)
- Tools: [tau-chrono](https://github.com/akaiHuang/tau-chrono)

**Hardware**: 1×H100 SXM, 540s (10-min budget), no torch.compile. 3,473 steps, 228M tokens, 4.2M params.

---

## 2. Text Diffusion (CDM) with Sequential Unmasking Eval

| Config | Eval Method | val_bpb | Size |
|--------|-------------|---------|------|
| 5L d=256 CDM (SP1024) | Sequential Unmasking | **2.570** | 2.2MB |

> **Note**: CDM uses SP1024 tokenizer (2.45 bytes/tok), AR above uses v4096 (3.38 bytes/tok). BPB values are not directly comparable across tokenizers because the byte-counting denominator differs.

**Training**: bidirectional transformer with random token replacement (= D3PM uniform noise ELBO).

```python
mask_ratio = uniform(0.1, 1.0)
mask = rand(B, T) < mask_ratio
noisy = where(mask, randint(0, V, (B,T)), tokens)
logits = model(noisy, is_causal=False)  # bidirectional
loss = cross_entropy(logits[mask], tokens[mask])
```

**Sequential Unmasking eval** — exact log P(x) via chain rule:

```
For t = 1..L:
  input = [x_1, ..., x_{t-1}, rand_t, ..., rand_L]
  accumulate log P(x_t) = log softmax(model(input))[t, x_t]

BPB = -Σ log P(x_t) / total_bytes / ln(2)
```

This decomposes `log P(x) = Σ log P(x_t | x_{<t})` — the same chain rule as AR. The random fill at positions ≥ t is averaged over R=3 draws for variance reduction. Because the model was trained on random-replaced inputs, it has learned to ignore the noise at unresolved positions. The resulting BPB is an approximately comparable metric to standard AR BPB.

**Hardware**: 1×H100 SXM, 540s training. Sequential Unmasking eval on M1 Max 64GB.

---

## 3. Coarse-to-Fine (AR + CDM)

| Config | val_bpb | Size | Log |
|--------|---------|------|-----|
| 11L AR + 11L CDM (SP1024) | **1.263** | ~31MB (2 models) | no log saved* |
| 5L AR + 11L CDM (SP1024) | **1.358** | ~19MB | no log saved* |

\* Evaluated interactively on H100 during development. Reproducible by running `eval_sequential_unmasking.py` with the provided model checkpoints. We acknowledge the lack of saved logs weakens auditability.

> Both exceed 16MB. Presented for research interest only.

**Eval**: AR predicts skeleton (every 2nd token, causal), CDM fills gaps (bidirectional):

```
AR:  P(x_0), P(x_2), P(x_4), ...   via causal attention
CDM: P(x_1|skeleton), P(x_3|skeleton), ...  via bidirectional attention

Total BPB = (Σ AR_NLL + Σ CDM_NLL) / total_bytes / ln(2)
```

Valid as compression: decoder reconstructs skeleton with AR, then fills gaps with CDM.

**Hardware**: 1×H100 SXM, 540s per model (AR and CDM trained separately).

---

## 4. Shared-Weight Dual Brain (AR + CDM in one model)

| Config | AR val_bpb | Coarse-to-Fine val_bpb | Size |
|--------|------------|------------------------|------|
| 5L d=256 Shared (SP1024) | 1.568 | **1.503** | 2.3MB |

One model, two modes — trained with both losses each step:

```python
loss_ar  = causal_lm_loss(model(tokens, is_causal=True), tokens)
loss_cdm = cdm_loss(model(noisy, is_causal=False), tokens, mask)
loss = loss_ar + loss_cdm
```

Same weights serve as AR (causal, left brain) and CDM (bidirectional, right brain). Coarse-to-Fine eval as described in Section 3.

CF 1.503 evaluated interactively on H100. No log saved — reproducible by running Coarse-to-Fine eval with the shared model checkpoint.

**Hardware**: 1×H100 SXM, 540s (10-min budget).

---

## Additional Techniques Explored

- **TTT (Test-Time Training)**: Full-model AdamW adaptation on graded validation tokens. 1.497 → 1.492 (−0.3%).
- **Depth Recurrence**: 6 unique weight sets × 17 passes (1.8× parameter efficiency). Tested on M1 Max with 768d model. Checkpoints saved but final BPB not formally recorded.
- **Custom v4096 BPE Tokenizer**: 3.38 bytes/tok vs SP1024 2.45. Same architecture SP1024 1.533 → v4096 1.497 (−2.4%).

---

## Files

| File | Description |
|------|-------------|
| `train_gpt.py` | AR + Retrodiction training (PyTorch) |
| `train_cdm.py` | Shared AR+CDM dual-brain training (PyTorch) |
| `eval_sequential_unmasking.py` | Sequential Unmasking eval |
| `eval_ttt.py` | TTT eval |
| `bpe_v4096.model` | Custom tokenizer (294KB) |
| `train.log` | H100 training log (5L v4096 AR) |

**Author**: Sheng-Kai Huang ([@akaiHuang](https://github.com/akaiHuang))
