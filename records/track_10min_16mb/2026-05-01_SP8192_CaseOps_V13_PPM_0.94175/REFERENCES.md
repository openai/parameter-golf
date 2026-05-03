# References and lineage

This submission builds on public Parameter Golf ideas rather than claiming a new standalone architecture. The list below is intentionally explicit so reviewers can separate inherited code/ideas from our final v13 changes.

## Base model and tokenizer lineage

- PR #1394 by @clarkkev: SP8192, GPTQ embeddings, depth recurrence, MuonEq-R, and related compact GPT training lineage.
- PR #1493 by @bigbag: SP8192 plus 3-layer recurrence, parallel residuals, QK gain 5.25, and the stronger recurrent base used by later SP8192 submissions.
- PR #1855 by @codemath3000: SP8192 plus LQER, sparse attention gate, BOS-fixed SmearGate, and the greedy hyperparameter stack that many late submissions build from.

## PPM / eval-time scoring lineage

- PR #1795 by @OE-GOD: strict-legal causal byte-level PPM adaptive-lambda mixer.
- PR #1959 by @remg1997: SP8192 plus byte-PPM mixer, bridging the PPM idea onto the later SP8192 neural stack.
- PR #1991 by @joshuaswanson: SP8192 + byte-PPM tuned order/gate, `0.94290` three-seed mean. v13 keeps the same core PPM direction and retunes the final gate to `H=0.999`, `L=0.18`, `T=0.80`.
- PR #1145 by @AnirudhRahul and PR #1967 by @ndokutovich: online n-gram tilt / scoring overlay ideas present in the code path, although the submitted score is from the PPM evaluator with TTT disabled.

## SmearGate and leakage fix lineage

- modded-nanogpt @classiclarryd: SmearGate idea referenced in the code comments.
- PR #1667 by @MarioPaerle: SmearGate + attention output gate integration into Parameter Golf.
- PR #1797 by @dexhunter: base audited for the packed-document SmearGate cross-boundary issue.
- PR #2014 by @simonbissonnette: public write-up of the BOS masking fix. v13 includes the BOS mask in both normal forward and TTT forward paths.

## Compression and quantization lineage

- PR #1586 by @dexhunter: per-layer adaptive GPTQ clip / int7 embeddings / MLR direction, referenced by the per-group compression lineage in this code.
- PR #1667 by @MarioPaerle and PR #1729 by @romeerp: per-group `lrzip` / grouped serialization lineage used for the submitted under-cap artifacts.
- PR #1530 by @samacqua: varlen attention, fused MLP, doc-independent TTT, and LQER-related lineage.
- PR #1886 by @renqianluo: fused softcap CE and WD stability notes reflected in comments/hyperparameters.
- PR #1923 by @jorge-asenjo: asymmetric logit rescale and AWQ-lite lineage.
- PR #1344 by @Omrigotlieb: Polar Express Newton-Schulz coefficients used in the optimizer path.

## Our changes

The main contribution here is the v13 consolidation, the sidecar-aware CaseOps evaluation packaging, and the final PPM gate retune:

```text
PPM_ORDER=5
PPM_H=0.999
PPM_L=0.18
PPM_T=0.80
TTT_ENABLED=0
```

Claude helped with late-stage experiment selection and write-up review. Codex handled implementation, audit, run coordination, packaging, and PR preparation.
