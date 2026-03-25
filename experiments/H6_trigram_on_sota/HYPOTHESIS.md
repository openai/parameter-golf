# H6: Trigram vs Bigram on SOTA (F1 Legal LB)

## Question
Does trigram hash embedding improve BPB over bigram on the 1.1190 model?

## Prediction
Trigram captures 3-character context per hash vs bigram's 2-character.
More local context = better input conditioning. At matched vocab size (1536),
trigram should improve by 0.001-0.003 BPB. Hash collisions increase but
high-frequency trigrams still get unique slots.

Risk: the model was tuned with bigram. Trigram changes the input distribution.
The improvement might be smaller than expected or require retuning.

## Arms (full 1.0 scale, 600s)
| Arm | Config | Change |
|-----|--------|--------|
| A (control) | F1 legal LB as-is | BIGRAM_VOCAB_SIZE=1536 |
| B (trigram) | F1 legal LB + trigram | TRIGRAM_VOCAB_SIZE=1536 |

Note: requires code change — F1 train_gpt.py uses BigramHashEmbedding.
Need to add TrigramHashEmbedding or make it configurable.

## Diagnostic Focus
- sliding_window BPB: does trigram beat 1.1190?
- Artifact size: trigram embedding same param count at matched vocab
- TTT interaction: does trigram help or hurt TTT adaptation?

## Status
NEEDS CODE CHANGE — BigramHash → configurable n-gram.

## Verdict
_To be filled after runs._
