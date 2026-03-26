# Parameter Golf — Batch Review Template

Use this after every 3-5 experiment runs. Feed this + experiment_log.jsonl to a free model.

---

## Review Prompt (copy this to a sub-agent)

```
You are reviewing a batch of Parameter Golf training experiments. 

CONTEXT: OpenAI Parameter Golf competition. Train best LM under 16MB + 10min 8×H100. Lower bpb = better. Current competition #1: ~1.07 bpb (pending PRs).

Here are the last N experiment results:
[PASTE experiment_log.jsonl entries here]

ANALYZE:
1. Which changes improved bpb? By how much?
2. Which changes hurt or had no effect?
3. What's our current bottleneck? (quantization? architecture? training schedule? evaluation?)
4. Based on the pattern of results, what should the NEXT batch of 3-5 experiments try?
5. Are there any techniques from our research files we haven't tried yet that address the current bottleneck?

OUTPUT:
- Bottleneck diagnosis (one sentence)
- Next 3-5 specific mutations to try (ranked by expected impact)
- Any web searches to run for the specific bottleneck
- Updated strategy notes
```

## After Review

1. Run recommended web searches
2. Generate next batch of code mutations
3. Hand to Codex for implementation
4. Run experiments
5. Log results to experiment_log.jsonl
6. Repeat

## Batch History

| Batch | Runs | Best bpb | Key Learning |
|-------|------|----------|--------------|
| 0 (baseline) | 1 | 1.3382 (1×H100) | Pipeline works, 2.65MB headroom |
