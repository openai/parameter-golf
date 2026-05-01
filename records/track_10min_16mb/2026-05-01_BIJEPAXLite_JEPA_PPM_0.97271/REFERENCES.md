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

- PR #1394 by @clarkkev: SP8192, GPTQ embeddings, depth recurrence, MuonEq-R, and related compact GPT training lineage.
- PR #1493 by @bigbag: SP8192 plus 3-layer recurrence, parallel residuals, QK gain 5.25, and the stronger recurrent base used by later SP8192 submissions.
- PR #1855 by @codemath3000: SP8192 plus LQER, sparse attention gate, BOS-fixed SmearGate, and the greedy hyperparameter stack.
- PR #1795 by @OE-GOD: strict-legal causal byte-level PPM adaptive-lambda mixer.
- PR #1959 by @remg1997: SP8192 plus byte-PPM mixer.
- PR #1991 by @joshuaswanson: SP8192 + byte-PPM tuned order/gate, including the order-5 PPM direction used here.
- modded-nanogpt @classiclarryd and PR #1667 by @MarioPaerle: SmearGate / attention-output-gate lineage.
- PR #1797 by @dexhunter and PR #2014 by @simonbissonnette: SmearGate packed-document BOS masking issue and fix lineage.
- PR #1586 by @dexhunter, PR #1667 by @MarioPaerle, and PR #1729 by @romeerp: per-group `lrzip` / grouped serialization lineage.
- PR #1530 by @samacqua, PR #1886 by @renqianluo, PR #1923 by @jorge-asenjo, and PR #1344 by @Omrigotlieb: LQER/AWQ/asymmetric-rescale/optimizer lineage reflected in the inherited stack.
- PR #1145 by @AnirudhRahul and PR #1967 by @ndokutovich: online n-gram tilt / scoring overlay ideas present in the broader code lineage.
- PR #2027 by @H1cSuNtDr4C0n3S: JEPA-Lite competition-local precedent. BIJEPAX-lite is a separate Claude-designed bidirectional/hop-4 variant, but this PR should be credited as prior JEPA-style work in the competition.

Our new contribution in this branch is the BIJEPAX-lite training-only auxiliary objective and the run package around it. The base GPT, tokenizer, compression, and PPM evaluator are inherited/adapted from the public lines above.

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
