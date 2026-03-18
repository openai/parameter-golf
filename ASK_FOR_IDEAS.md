# Prompt For Improvement Ideas

I want you to think like a ruthless parameter-golf competitor.

Please read `README.md` and `train_gpt.py` and propose the highest-upside ways to improve this baseline while staying fully within the challenge rules.

## Objective

- Improve final `val_bpb` on the challenge evaluation, not just train loss.

## Hard Constraints

- Total submission artifact must stay under `16,000,000` bytes.
- Counted artifact is code bytes plus compressed model bytes.
- Evaluation and submission must run in under 10 minutes on `8xH100`s.
- No external downloads or network calls during evaluation.
- The solution should remain in the spirit of the challenge.
- Changes should be realizable primarily inside `train_gpt.py`.
- `train_gpt.py` must stay under 1500 lines.

## Current Baseline

- 9-layer GPT, width 512
- 8 attention heads, 4 KV heads
- 2x MLP
- tied embeddings
- sequence length 1024
- bf16 training
- Muon for matrix params, Adam for embeddings/scalars
- int8 + zlib export for final artifact size
- tokenizer-agnostic evaluation via `val_bpb`

## What I Want From You

- Give me 10-20 concrete ideas, ranked by expected payoff.
- Prioritize non-obvious ideas over generic "tune LR" advice.
- For each idea, include:
  1. why it could improve `val_bpb`
  2. expected effect on training speed
  3. expected effect on compressed artifact size
  4. implementation complexity and risk
  5. whether it is likely to fit cleanly into this script
- Separate ideas into:
  - architecture changes
  - optimization and training changes
  - serialization and compression changes
  - evaluation and test-time compute tricks
  - tokenizer and data-interface changes, if allowed
- Call out any ideas that are likely invalid or borderline under the rules.

Be opinionated. I do not want a broad brainstorm; I want the few ideas you think have the best expected value for this exact codebase and challenge.
