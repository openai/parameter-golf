# Prompt For Improvement Ideas v2

Read `README.md` and `train_gpt.py` and act like you are trying to beat this baseline in the Parameter Golf challenge.

I do not want a broad brainstorm. I want your highest-conviction ideas for this exact script.

## Objective

- Minimize final `val_bpb`.

## Non-Negotiable Constraints

- Total artifact = code bytes + compressed model bytes must be under `16,000,000`.
- Submission and evaluation must run in under 10 minutes on `8xH100`.
- No network or external downloads at evaluation time.
- The approach must stay within the spirit of the challenge.
- Changes should fit primarily inside `train_gpt.py`.
- `train_gpt.py` must remain under 1500 lines.

## Current Baseline Facts

- 9 layers, width 512
- 8 query heads, 4 KV heads
- 2x MLP
- tied embeddings
- sequence length 1024
- bf16 training
- Muon for matrix params, Adam for embeddings and scalars
- int8 + zlib final export
- tokenizer-agnostic metric is `val_bpb`

## Your Task

- Give me the top 7 ideas you would actually try first.
- Rank them by expected value, where expected value means likely `val_bpb` gain relative to implementation cost, rule-risk, and time cost.
- Do not include generic tuning suggestions unless you can argue they are unusually important for this script.
- Prefer ideas that exploit the challenge setup: tiny artifact budget, tokenizer-agnostic evaluation, 10-minute cap, and permissive evaluation rules.

For each idea, give:

1. the core thesis in 1 sentence
2. what bottleneck in this baseline it attacks
3. why it should improve `val_bpb`
4. expected effect on:
   - training speed
   - evaluation speed
   - compressed artifact size
5. implementation difficulty from 1-5
6. rule-risk from 1-5
7. the smallest decisive experiment to test it
8. whether you think it belongs in:
   - baseline script improvement
   - serious record attempt
   - probably not worth trying

## Also Include

- 3 ideas that sound clever but you think are probably traps
- 3 concrete ablations you would run first before changing architecture
- your single best bet if I only try one thing
