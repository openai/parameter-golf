# Submission-Risky Directions

The source document explicitly calls out a few directions as risky or invalid for a serious submission:

- external-teacher distillation or offline logits from a larger model
- any evaluation trick that peeks ahead
- anything that effectively trains on future validation tokens
- full gradient-based test-time training on the validation set

The safer aggressive path is:

- causal cache-style adaptation
- long-context streamed evaluation
- EMA and QAT
- parameter reallocation inside the model
