# 11L Adaptive Markov + Int6 Mixed Quant + 786K Batch

This record captures my `11L Adaptive Markov + Int6/Int8 Mixed Quant` submission for the `track_10min_16mb` setting.

## Motivation

The motivation behind this approach goes back to my undergraduate research work over 8 years ago, when I was involved in brain-computer interface research connected to cancer-focused applications. That work led to a $100k grant and introduced me to the challenge of modeling real sequential signals, especially noisy neural activity where local transitions still carry meaningful predictive structure.

During that time, I became fascinated by brain waves and the broader question of whether human behavior or neural patterns could be predicted from signal dynamics. That led me to practical Markov chain implementations in real research settings. The core idea stayed with me: even in complex systems, local state transitions can contain valuable predictive information if modeled carefully.

This submission comes from that same intuition. Instead of treating the language model as purely transformer-only, I wanted to explore whether a causal GPT could benefit from an explicit Markov-style component that captures short-range token transition structure, then combines it with the broader contextual reasoning of the transformer. In other words, this was not meant as a benchmark trick, but as an attempt to bring a real sequence-modeling intuition from earlier research into the competition in a practical and causal way.

## Approach Evolution

I did not arrive at the final model immediately. I first explored several versions of the idea, including first-degree and second-degree Markov variants, as well as gated and weighted approaches for combining local transition structure with transformer predictions. Those experiments helped clarify both the value and the limits of explicit short-range modeling in this setting.

The final submission uses the adaptive version of the hybrid. Compared with the earlier variants, it gave the best overall balance between local transition awareness, contextual flexibility, and performance under the competition’s wallclock and artifact constraints.

## Summary

This submission combines three main ideas:

1. **Adaptive Markov mixing**  
   A unigram Markov transition table is combined with transformer logits through an adaptive per-position mixing mechanism. The model uses transformer hidden states to predict how much Markov signal to trust, and also applies a confidence gate based on the top-2 Markov logit gap.

2. **Mixed int6/int8 quantization with zstd-22**  
   MLP and attention weights use per-row int6 quantization. Embeddings and the Markov transition table use int8 quantization. Small control tensors remain fp16 passthrough. Compression is handled with zstd level 22.

3. **786K token batch**  
   A larger batch size improved wallclock efficiency for the deeper model. Even with fewer total steps, the model processed enough total tokens within the 10-minute budget to improve final quality.

## Final Configuration

- Model kind: `gpt_markov`
- Layers: `11`
- Dim: `512`
- Heads: `8`
- KV heads: `4` (GQA)
- Sequence length: `1024`
- Batch: `786,432` tokens
- Markov settings:
  - `MARKOV_LR=0.008`
  - `MARKOV_MIX_INIT=0.05`
  - `MARKOV_GATE_THRESHOLD=0.20`
  - `MARKOV_GATE_TEMP=0.03`
- No QAT
- No EMA
- Quantization:
  - int6 per-row for MLP and attention weights
  - int8 for embeddings and Markov table
  - fp16 passthrough for small/control tensors
- Compression: `zstd` level 22

## Command

```bash
pip install zstandard

TRAIN_BATCH_TOKENS=786432 \
MODEL_KIND=gpt_markov \
NUM_LAYERS=11 \
MARKOV_LR=0.008 \
MARKOV_MIX_INIT=0.05 \
MARKOV2_BUCKETS=0 \
MARKOV_GATE_THRESHOLD=0.20 \
MARKOV_GATE_TEMP=0.03 \
QAT_START_FRAC=0 \
EMA_DECAY=0 \
torchrun --standalone --nproc_per_node=8 experiments/train_gpt_markov_adaptive.py
```

## Key Metrics

- Pre-quant eval:
  - `val_loss: 2.0394`
  - `val_bpb: 1.2078`
- Post-quant int8+zlib:
  - `val_loss: 2.04791199`
  - `val_bpb: 1.21288883`
  - over size limit
- Post-quant int6+zstd:
  - `val_loss: 2.05546777`
  - `val_bpb: 1.21736379`
- Steps completed: `7427/20000`
- Wallclock: `600.075s`
- Total submission size: `15,107,918 bytes`

## Architecture Details

### Adaptive Markov Mixing

The model combines a standard transformer with a unigram Markov transition table. The transformer hidden state is projected to a scalar mixing weight at each position, allowing the Markov contribution to vary with context. In addition, a confidence gate based on the Markov top-2 logit gap suppresses the Markov contribution when its signal is weak or ambiguous.

This gives the hybrid two useful properties:

- the transformer can still dominate on harder, context-heavy predictions
- the Markov component can help on easier local transitions such as common token continuations, punctuation, and other short-range patterns

### Mixed Quantization Strategy

| Component | Quantization | Reason |
|---|---|---|
| MLP weights | int6 per-row | large tensors, relatively quantization-tolerant |
| Attention Q/K/V/proj | int6 per-row | large 2D tensors |
| Embeddings | int8 per-row | more sensitive |
| Markov transition table | int8 per-row | sensitive to quantization noise |
| Control tensors | fp16 passthrough | small and precision-sensitive |

## Notes

- This approach is fully causal.
- The Markov component is used as an explicit short-range prior, not as a non-causal cache or lookahead mechanism.
- The final legal submission artifact is the `int6+zstd` export.
- `zstandard` is required for the legal export path.

## Included Files

- `README.md`
- `submission.json`
- `train_seed1337.log`
- `train_seed42.log`
- `train_seed7.log`
- `train_gpt.py`
