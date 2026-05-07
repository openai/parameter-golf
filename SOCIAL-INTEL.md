# OpenAI Parameter Golf Social Intel Report
*Gathered Mar 24, 2026 by Hermes 🃏*

## 1. Parameter Golf Competition Discussion
- Primary hub: GitHub repo [openai/parameter-golf](https://github.com/openai/parameter-golf) (27+ forks in days).
  - Active PRs/experiments: SWT arch (KushagraLabs), auto-iterative loop (chrispyspearbit/ChrisGoesGolfing), dashboard [parameter-golf.github.io](https://parameter-golf.github.io/).
- HN mentions: Announcement posts (e.g., [item?id=47477706](https://news.ycombinator.com/item?id=47477706)), low engagement (1-12 pts). Hive agents topping leaderboard ([news.ycombinator.com/item?id=47472846](https://news.ycombinator.com/item?id=47472846), [hive.rllm-project.com/task/parameter-golf](https://hive.rllm-project.com/task/parameter-golf)).
- No Reddit/Discord/Twitter hits (rate-limited/blocked). Sparse forum buzz; focus on GitHub leaderboard.

## 2. NanoGPT Speedrun Community
- Official repo cites as inspiration: [NanoGPT Speedrunning](https://github.com/KellerJordan/modded-nanogpt).
- Key techniques applicable: Fast training mods (L(T) optimization → parameter-constrained L(N)).

## 3. Leaderboard Leaders' Public Presence
| Leader | Score | GitHub | Other |
|--------|-------|--------|-------|
| signalrush | 1.1228 (11L EMA + GPTQ-lite + warmdown) | [signalrush/parameter-golf](https://github.com/signalrush) (fork) | None found |
| jfprincz | 1.1248/1.1271 (Partial RoPE/XSA4/EMA) | [jfprincz/parameter-golf](https://github.com/jfprincz) (fork) | None |
| unnir | 1.1307 (Efficient Partial XSA) | [unnir](https://github.com/unnir) (ML PhD, tabularis-ai) | [vadimborisov.com](http://vadimborisov.com), [@vdmbrsv](https://twitter.com/vdmbrsv) |
| thwu1 | 1.1428 (10L Int5-MLP + BigramHash) | [thwu1](https://github.com/thwu1) (Berkeley EECS PhD) | [thwu1.github.io](https://thwu1.github.io/tianhaowu/) |
| Raahil Shah | 1.1458 (Int6 MLP3x + SmearGate) | No direct repo | None |
| aruniyer | 1.1502 (11L MLP3x + Int6 QAT) | No direct repo | None |

- Leaders mostly silent publicly; sharing via PRs/READMEs (e.g., GPTQ-lite clip search, Partial RoPE 16/64, Int6 MLP3x).

## 4. ML Twitter/X on Small Model Training
- Searches blocked/no content. HN proxy: Sparse; focus on challenge mechanics over techniques.

## 5. Competition Meta-Strategy
- No direct blogs/threads. Implicit: Compute grants ($1M OpenAI-sponsored), MLX local training, cloud (Runpod). Emphasize arch hacks (test-time compute, QAT, bitnets, tying).

## 6. Hidden Gems
- Forks experimenting: [ofirkris/parameter-golf](https://github.com/ofirkris/parameter-golf) (val_bpb 1.1377), [SuperKaggler/openai_challenge_parameter_golf](https://github.com/SuperKaggler/openai_challenge_parameter_golf) (small LLM design).
- Hive swarm: Collaborative agents evolving code ([hive.rllm-project.com](https://hive.rllm-project.com/)).
- Non-record: 4-hour baseline 1.2074 (Quasi10B from 50B).

**Pulse:** GitHub-centric, low external hype. Community iterating PRs on quantization (Int5/6, GPTQ-lite), MLP scaling (3x), hashing (BigramHash), SWA/EMA, custom gates (SmearGate). No leaked strategies beyond READMEs. Watch unnir/thwu1 for blogs.