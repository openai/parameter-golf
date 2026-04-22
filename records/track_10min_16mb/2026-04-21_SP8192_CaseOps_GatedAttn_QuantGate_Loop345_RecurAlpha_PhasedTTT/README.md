# SP8192 + CaseOps + Gated Attention + Quant Gate + Loop345 + Recur-Alpha + Phased TTT

**val_bpb = pending** | **~16MB** | 8xH100

## Summary

Adds **Recur-Alpha** to the PR #1736 stack (CaseOps + GatedAttn + QuantGate + Loop345 + PhasedTTT, val_bpb=1.06549). Recur-Alpha is a learnable scalar per looped block, initialized to zero, that adds a weighted copy of each block's first-visit activation to its subsequent recurrence passes. It introduces GRU-like carry into the depth recurrence with zero effect at initialization.

The only code change from PR #1736:

```python
# Block.__init__ — one new parameter per block
self.recur_alpha = nn.Parameter(torch.zeros(1))

# forward_logits / forward_ttt — carry logic in encoder and decoder loops
carry = {}
for i in enc_iter:
    x = block(...)
    if self.looping_active:
        if i in carry:
            x = x + self.blocks[i].recur_alpha.to(dtype=x.dtype) * carry[i]
        carry[i] = x
    skips.append(x)

# decoder (non-parallel branch only):
x = block(...)
if self.looping_active and i in carry:
    x = x + self.blocks[i].recur_alpha.to(dtype=x.dtype) * carry[i]
```

## Why Recur-Alpha

In plain depth recurrence, the second and third passes through a looped block recompute entirely from the current hidden state — no memory of earlier passes is carried forward. Recur-Alpha gives each block a trainable gate to blend prior-visit residuals into the current pass. Zero initialization ensures the model starts identical to the base and can learn the amount of carry through training.

The idea originates in PR #1714 (Anakintano), where it was implemented on the older SP8192 3-layer recurrence stack and showed strong pre-TTT results (1.0857) but TTT evaluation was never completed due to exhausted compute. This PR composes Recur-Alpha with the newer CaseOps + phased TTT stack for the first time.

## Parameter Cost

- 3 looped blocks (layers 3, 4, 5) × 1 scalar each = **3 parameters**
- ndim=1 → excluded from GPTQ and Muon; trained by scalar AdamW
- Artifact size impact: negligible (< 100 bytes uncompressed, compressed to ~0)

## Full Technique Stack

1. **SP8192** tokenizer (SentencePiece BPE, vocab 8192)
2. **CaseOps** — bijective lossless case preprocessing with TITLE/ALLCAPS/CAPNEXT/ESC operator tokens; BPB scored on original UTF-8 bytes via per-token byte sidecar
3. **3-Layer Depth Recurrence** — layers 3, 4, 5 looped ×2 (17 virtual layers), activates at 35% training
4. **Recur-Alpha** — learned carry scalar per looped block (init=0) *(novel addition)*
5. **Gated Attention** — per-head sigmoid output gate (Qwen-style), init_std=0.01
6. **Quant Gate** — int8-per-row quantization of attn_gate_w tensors
7. **Parallel Residuals** — GPT-J style from layer 8
8. **QK-Gain 5.0** — learned per-head query scalar
9. **Full-Hessian GPTQ** — int6 matrices, int8 embeddings, SDClip
10. **MuonEq-R** — row-normalized Muon + AdamW
11. **Phased TTT** — score-first LoRA SGD, per-doc reset, cosine LR decay
12. **Byte-shuffle + Brotli** compression

## Reproduction

```bash
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

# Prepare CaseOps data (once)
python prepare_caseops_data.py

# Train (one seed)
SEED=42 CASEOPS_ENABLED=1 GATED_ATTN_ENABLED=1 GATED_ATTN_QUANT_GATE=1 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- **@dexhunter** — PR #1736 CaseOps + GatedAttn + QuantGate + Loop345 + PhasedTTT base (val_bpb 1.06549)
- **@samacqua** — PR #1530 SP8192 base stack
- **@romeerp** — PR #1729 CaseOps concept + byte sidecar
- **@MarioPaerle** — PR #1667 attention gate pattern
- **Anakintano** — PR #1714 Recur-Alpha concept and implementation
- **@bigbag** — PR #1493 prior merged SOTA
