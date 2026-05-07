# Model 7: "The Immune System" — Build Spec

**Classification:** PRIVATE — DO NOT SUBMIT UNTIL ENDGAME
**Target bpb:** Unknown (experimental)
**Approach:** Combinatorial weight generation from templates
**Nature Analog:** V(D)J recombination — immune system generates millions of antibodies from ~400 gene segments

---

## Core Concept

Your immune system doesn't store a separate antibody for every pathogen. It stores ~400 gene segments and COMBINES them to create millions of unique antibodies. Each antibody is assembled from 3-4 randomly selected segments joined together.

We do the same: instead of storing a fixed weight matrix at each layer, store a library of small weight "templates" and dynamically combine them per-token to generate that token's effective weights.

## Architecture

### Template Library (12 MB)
- 128 weight templates, each ~96KB
- Each template is a small matrix (512 × 16 or similar)
- Templates are shared across all layers
- Stored at int6 quantization

### Recombination Router (0.5 MB)
- Per layer: small MLP (512 → 64 → 4) that selects which templates to combine
- Output: 4 template indices + mixing weights
- Different tokens activate different template combinations
- Total: 11 layers × ~45KB = ~0.5 MB

### Assembly Mechanism
For each token at each layer:
1. Router selects 4 templates + weights
2. Templates are combined: W_effective = sum(alpha_i * template_i) for i in selected
3. W_effective is used as the layer's weight matrix for this token
4. Different tokens get different effective weights

### Output Head + Embeddings (3.5 MB)
- Tied embeddings (fp16): ~1 MB
- Output projection, biases, norms: ~0.5 MB
- Buffer for quantization overhead: ~2 MB

## Training Strategy

1. Initialize templates randomly (orthogonal)
2. Train templates + routers end-to-end
3. Templates learn to be useful "building blocks" that compose well
4. Routers learn which combinations work for which token types
5. Gradient flows through both the selection and the templates

## Parameter Budget

| Component | Size |
|-----------|------|
| 128 templates (int6) | ~12 MB |
| Recombination routers | ~0.5 MB |
| Embeddings + output | ~1.5 MB |
| Norms, biases, overhead | ~2 MB |
| **Total** | **~16 MB** |

## Why This Could Work

- Combinatorial explosion: 128 choose 4 = 10.7M unique combinations PER LAYER
- Each token effectively sees a different model — massive effective capacity
- Templates learn shared structure, combinations create specialization
- Nature proves this works: 400 gene segments → millions of unique antibodies
- Potentially much more expressive than a single fixed weight matrix

## Key Risks
- Dynamic weight generation is slow (need to assemble weights per token per layer)
- Router training may be unstable (Gumbel-softmax for discrete selection)
- Templates may not learn to compose well
- Gradient through discrete selection is noisy (STE)
- May be too slow to fit in 10-minute training window

## Output
- `train_gpt_model7.py`
