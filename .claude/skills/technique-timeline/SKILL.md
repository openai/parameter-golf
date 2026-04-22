---
name: technique-timeline
description: Deep chronological research on how a specific technique (e.g. "recurrence", "parallel residuals", "TTT", "quantization") evolved across PRs in openai/parameter-golf. Fetches actual PR pages, builds a phase-by-phase narrative, and surfaces what's been ablated vs what remains unknown. Use before speccing an idea to avoid reinventing the wheel.
argument-hint: <technique name or keyword>
---

# Technique Timeline

You are building a deep chronological research report on how a specific technique evolved in the openai/parameter-golf competition.

The user has asked about: **{{ args }}**

## What to produce

A structured, phase-by-phase narrative covering:
1. Who introduced the technique and when
2. What the first version looked like (architecture, config, score)
3. How it evolved PR-by-PR (config changes, ablations, score progression)
4. How it interacts with other techniques (recurrence, TTT, quantization, etc.)
5. What has been ablated vs what is assumed but untested
6. A summary table: date | PR | author | bpb | what changed | merged?
7. Open questions that remain as of today

## Search strategy

Start with targeted PR searches. Run these in parallel:

```
https://github.com/openai/parameter-golf/pulls?q=<keyword>&state=all&per_page=100
https://github.com/openai/parameter-golf/pulls?q=<keyword>&state=closed&per_page=100
```

Use synonyms and related terms. For example:
- "recurrence" → also search "loop", "layer loop", "looping", "depth recurrence"
- "parallel residuals" → also "parallel stream", "GPT-J", "XSA", "pre-residual", "dual stream"
- "TTT" → also "test-time training", "score-first", "phased TTT", "LoRA TTT"
- "quantization" → also "GPTQ", "SDClip", "int6", "int8", "Hessian"

For each relevant PR found, fetch the actual PR page to extract details:
```
gh pr view <N> --repo openai/parameter-golf --json number,title,author,body,createdAt,mergedAt,state
```

For PRs with code changes worth understanding, also fetch the diff:
```
gh pr diff <N> --repo openai/parameter-golf --patch
```

## What to extract per PR

- PR number, author login, date submitted, date merged (or "open"/"closed unmerged")
- Claimed bpb score (mean ± std, number of seeds)
- Exactly what the technique does in this PR (concrete mechanism, not marketing)
- Config values (layer indices, step fractions, hyperparams)
- How it differs from the prior version of the same technique
- Any ablations performed and their results
- Any negative results or failure modes documented
- Whether it was merged into main

## Narrative structure

Organize findings into phases based on natural breakpoints in the evolution (e.g. "first introduced", "first successful record", "key ablation study", "stable backbone — orthogonal work begins").

Within each phase, go PR-by-PR in chronological order.

End with:
- A **summary table** (date | PR# | author | bpb | key change | status)
- A **key findings** section: the 3–5 most important empirical discoveries
- An **open questions** section: what has never been cleanly ablated on the current stack
- A **potential opportunities** section: concrete experiments worth trying, grounded in the gaps found above. For each opportunity: what to change, what config value to try, what bpb delta might be plausible, and what risk/cost it carries. Flag if it's a one-liner config change vs requires code. Do not invent ideas from thin air — each opportunity must be motivated by a specific gap or untested assumption found in the research.

## Tone and format

- Write for a researcher who will use this to decide whether to spec a new idea
- Be specific: cite PR numbers, config values, bpb deltas
- Flag when something is "assumed but never ablated" vs "explicitly tested"
- Keep phase headers short; put detail in the body
- No speculation beyond what the PRs actually say

## Reminders

- Focus on the **record track** (`track_10min_16mb/`). Non-record PRs only if they contain directly relevant ablations or introduce the technique first.
- Our **current baseline is PR #1736** (dexhunter, 1.06549 bpb, unmerged). The **merged SOTA is PR #1493** (1.0810 bpb).
- Today's date is in the system context — use it to frame "recent" vs "established".
- Do not write idea files or specs — this skill is research-only. Hand findings back to the user.
