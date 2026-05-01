# References and Prior Work

Use this file for claims we can defend in the PR.

## Primary references

1. **I-JEPA: Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture**
   - arXiv: https://arxiv.org/abs/2301.08243
   - Authors: Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, Nicolas Ballas.
   - Relevant claim: JEPA predicts target representations from context representations instead of reconstructing input pixels/tokens.

2. **LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures**
   - arXiv: https://arxiv.org/abs/2509.14252
   - Authors: Hai Huang, Yann LeCun, Randall Balestriero.
   - Relevant claim: JEPA-style embedding-space objectives can be combined with LLM training objectives and reported stronger/robuster performance across multiple text tasks/models.

3. **MC-JEPA: A Joint-Embedding Predictive Architecture for Self-Supervised Learning of Motion and Content Features**
   - arXiv: https://arxiv.org/abs/2307.12698
   - Authors: Adrien Bardes, Jean Ponce, Yann LeCun.
   - Relevant claim: JEPA objectives can be composed with another auxiliary objective in a shared encoder; this is a useful analogy for our "CE plus training-only representation prediction" setup.

4. **Parameter Golf README / rules**
   - GitHub: https://github.com/openai/parameter-golf/blob/main/README.md
   - Relevant constraints: strict decimal 16,000,000-byte artifact cap; no validation data during training; eval under 10 minutes; record submissions normally need multi-seed evidence.

## Competition-local ideas inherited or combined

The BIJEPAX-lite submission is not a from-scratch stack. It layers a JEPA-style training objective onto our existing SP8192 CaseOps + PPM stack.

Important inherited components to acknowledge:

- SP8192 / CaseOps tokenizer lane.
- Per-group compression with `lrzip`.
- GPTQ mixed quantization and LQER/AWQ settings.
- SmearGate cross-document leak fix.
- PPM sliding byte mixer.

## Claims to avoid unless sourced

Do **not** write these as facts unless we find exact sources:

- "BiJEPA cycle consistency has proven 4x better on chaotic systems."
- "Nobody in Parameter Golf has done backward JEPA prediction."
- "LLM-JEPA proves cosine similarity is strictly superior to MSE for all text."

Safer wording:

- "Inspired by JEPA and LLM-JEPA, we add a cosine embedding-space auxiliary objective."
- "We tested a bidirectional hidden-state prediction variant but submitted the lighter hop-4 version because it preserved training speed and artifact size."
- "The auxiliary predictor is training-only and is absent from the serialized artifact."

## Suggested PR phrasing

"This submission explores whether a JEPA-style representation prediction objective can act as a cheap training regularizer for the SP8192 CaseOps + PPM lane. The predictor heads are separate modules, optimized only during training, and are never serialized. Final scoring is still performed by the causal PPM sliding evaluator over the quantized model artifact."
