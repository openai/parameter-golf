# Non-Record Submission: AuxNet Baseline

This is a non-record submission, and the result is slightly worse than the published naive baseline. The experiment tests a specific hypothesis: a very small auxiliary network, running alongside the main LM, may be able to learn simple but high-value structure faster than the trunk and feed that information back as a cheap correction. Instead of making the base model larger, the goal is to see whether a tiny side network can learn an prior and use it to nudge next-token prediction in the right direction.

This idea was motivated by earlier small-LM experiments on TinyStories with an RTX 5090. Even when the model did not show a clear improvement in validation loss, the generated stories were often more coherent and more interesting qualitatively. That suggests a useful insight: since, the standard LM architectures are hyper optimized, when testing non-standard LM architectures, it is worth looking beyond aggregate train/val metrics, because some behavioral differences may show up more clearly in qualitative evaluation than in loss alone.

In this particular submission, we use the aux network to predict whether the next-token contains a space character or not. The aux network tries in two coupled ways:

1. It predicts a binary target: whether the ground-truth next token has a leading space (`has_leading_space_lut[target_id]`).
2. It uses the same low-dimensional features to generate a small residual edit in embedding space, projects that edit through the tied embedding matrix, and adds the resulting correction directly to the LM logits.

In code, this is the `AuxNet` module:

- `512 -> 32` bottleneck (`AUX_DIM=32`)
- tiny 32-dim MLP editor
- `32 -> 512` up-projection back into model space
- a separate 1-logit boundary classifier trained with BCE
- `edit_scale` initialized to zero so the whole module starts as a no-op

Conceptually, it is trying to learn a cheap "word-boundary prior." If the hidden state suggests the next piece is probably a word start, the residual logit edit can softly favor tokens whose tied embeddings line up with that direction. The auxiliary BCE loss encourages the bottleneck to carry exactly that boundary information.

## Things to try out

1. Predict a richer target than just `has_leading_space`. Good candidates include `start_of_word`, `continuation_piece`, punctuation/opening-quote classes, or a small multi-class boundary label.

2. Explore stronger aux-network designs for generating the corrections. The current module is intentionally tiny and simple; a better bottleneck, gating scheme, or edit head may turn the learned signal into a more useful logit adjustment.

## Configuration
The baseline LM architecture is unchanged except for a smear transformation applied to the token embeddings with a learned lower-triangular matrix. The goal is to encourage the embedding layer to build features progressively.

- `VOCAB_SIZE=1024`
- `NUM_LAYERS=9`
- `MODEL_DIM=512`
- `NUM_HEADS=8`
- `NUM_KV_HEADS=4`
- `MLP_MULT=2`
- `TRAIN_BATCH_TOKENS=524288`
- `TRAIN_SEQ_LEN=1024`
- `AUX_LOSS_WEIGHT=0.1`
- `AUX_DIM=32`
- `TIED_EMBED_LR=0.05`
- `MATRIX_LR=0.03`
- `SCALAR_LR=0.03`
- `HEAD_LR=0.0`
- `MAX_WALLCLOCK_SECONDS=600`

## Runs

Three independent training runs with different seeds. `pre_quant_val_bpb` is the final stop-time eval from the training log; `post_quant_val_bpb` is the exact `final_int8_zlib_roundtrip` metric used for the submission:

| Seed | val_loss | pre_quant_val_bpb | post_quant_val_bpb | step_stop | bytes_total |
|------|----------|-------------------|--------------------|-----------|-------------|
| 42 | 2.06974043 | 1.2249 | 1.22581687 | 13217 | 15,900,186 |
| 1227 | 2.07040067 | 1.2255 | 1.22620790 | 13232 | 15,924,267 |
| 2000 | 2.06864145 | 1.2245 | 1.22516599 | 13243 | 15,924,165 |
| **Mean** | **2.06959418** | **1.22497** | **1.22573025** | **13230.67** | **15,916,206** |
| **Std** | **0.00089** | **0.00050** | **0.00053** | - | - |

Additional run metrics:
- Peak memory: `10666 MiB allocated`
- Model params: `17,227,624`
- Code size: `55,890 bytes`
- Worst-case serialized model int8+zlib size: `15,868,377 bytes`

Code and run logs:
- `train_gpt.py` - exact code snapshot used for the runs
- `train_seed42.log`
- `train_seed1227.log`
- `train_seed2000.log`
- `submission.json`
