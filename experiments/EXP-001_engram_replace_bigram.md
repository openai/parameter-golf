# EXP-001: Engram Replaces BigramHash

## Metadata
- **Date**: 2026-04-06 (planned)
- **Branch**: exp/engram-replace-bigram
- **Parent**: exp/reproduce-sota
- **Priority**: P0
- **Estimated runs**: 1 dev + 1 full + 3 seed = 5 runs
- **Estimated cost**: ~$17
- **Paper**: DeepSeek Engram (arXiv:2601.07372)
- **Reference code**: https://github.com/deepseek-ai/Engram/blob/main/engram_demo_v1.py

## Hypothesis
Replacing BigramHash 3072x112 with a scaled-down Engram module will improve BPB by
0.002-0.005 because:
1. Trigram (3-gram) patterns capture more context than bigrams alone
2. Multi-head hashing with prime moduli reduces collision rate vs single hash
3. Learned gating selectively merges memory based on hidden state (vs blind addition)
4. The Engram paper shows this relieves early layers from static pattern reconstruction,
   effectively deepening the network for complex reasoning

## Null Hypothesis
Engram adds too many parameters (key/value projections, norms, conv) for the 16 MB byte
budget. The gating overhead costs more than the information gain. int6 quantization of
the larger embedding tables introduces more error. BPB is neutral or worse.

## Control Variables (what stays the same)
- 11 layers, 512d, 8 GQA heads, 4 KV heads
- XSA on all 11 layers, Partial RoPE 16/64
- LeakyReLU(0.5)^2 MLP, 3x expansion
- LN scale 1/sqrt(layer+1), SmearGate, VE128 layers 9-10
- U-Net skips, EMA(0.997) + SWA
- Full Hessian GPTQ int6 with AR self-gen calibration
- LZMA preset=9, selective pruning
- Warmdown 4000, Parallel Muon + Parameter Banking
- Late QAT at LR scale < 0.15
- Same training data, tokenizer, eval protocol

## Independent Variable (what changes)
Remove: BigramHash 3072x112 (~344K params, lines 680-712 of SOTA script)
Add: Engram module with these scaled-down settings:
- max_ngram_size = 3 (bigrams + trigrams)
- engram_vocab_size = [4096, 4096] (per ngram order)
- n_embed_per_ngram = 64
- n_head_per_ngram = 4 (16 dims per head)
- layer_ids = [0, 5] (early + mid layer)
- No ShortConv (save parameters), OR kernel_size=2 if bytes allow
- Gating: key projection + sigmoid gate (per the Engram paper)
- Value projection to model_dim=512

Estimated parameter delta: +~260K params (600K Engram - 344K BigramHash)
Must verify artifact fits under 16 MB after GPTQ + LZMA.

## Adaptation Notes (Engram -> Parameter Golf)
The Engram demo uses:
- DeepSeek-V3 tokenizer (129K vocab) -> we use SP1024 (1024 vocab), much smaller
- hc_mult=4 hyper-connections -> we don't have hyper-connections, use simple residual add
- Hidden size 1024 -> our model_dim is 512
- Engram at layers [1, 15] in a 30L model -> we pick layers [0, 5] in our 11L model

Key simplifications:
- Drop CompressedTokenizer (our vocab is already 1024, no dedup needed)
- Drop ShortConv (parameter budget)
- Drop hyper-connection dimensions (use single hidden state, not [B,L,HC,D])
- Keep: multi-head hash, multi-head embedding, gating mechanism, value projection

## Success Criteria
- BPB < 1.1147 (any improvement over reproduced baseline)
- Artifact size <= 15.95 MB
- No training slowdown > 5 ms/step (target: ~87 ms/step)

## Abort Criteria
- Artifact > 16 MB after GPTQ + LZMA (hard stop -- resize Engram)
- BPB > 1.1180 on dev run (0.003 worse -> abandon this config)
- Training > 95 ms/step (won't finish enough steps)

## Run Plan
1. DEV RUN: seed=314, 8xH100, full 600s
   - Goal: verify compilation, artifact fits, BPB estimate
2. DECISION GATE: artifact < 16MB AND BPB < 1.1180
3. TUNE (if needed): adjust vocab size, embed dim, or drop gating
4. FULL RUN: seed=314, 8xH100
5. DECISION GATE: BPB < reproduced baseline
6. SEED RUNS: seeds 314, 42, 999

## Implementation Plan
1. Copy SOTA train_gpt.py to new branch
2. Add NgramHashMapping class (adapted from Engram demo, lines 188-303)
3. Add MultiHeadEmbedding class (adapted from lines 305-324)
4. Add EngamModule class with gating (adapted from lines 326-378, minus ShortConv/HC)
5. Replace BigramHashEmbedding calls in GPT.__init__ and forward
6. Add Engram to _HessianGPT for GPTQ calibration
7. Add env vars: ENGRAM_VOCAB, ENGRAM_EMBED, ENGRAM_HEADS, ENGRAM_LAYERS
8. Test compilation locally (will fail on FA3, but syntax check passes)
9. Run on 8xH100

## Results
[Fill in after running]

## Post-Mortem
[Fill in after running]
