# Claude Code Skills Reference

Invoke any skill with `/skill-name` in the chat.

---

## `/parameter-golf`
Strategy guide, adversarial critic, and technique oracle for the OpenAI Parameter Golf competition.

**Use when:** asking about val_bpb improvement, artifact size, technique selection (GPTQ, TTT, XSA, BigramHash, Muon, int5/int6, QAT), phase planning, TTT legality, quantization, compression pipeline, 3-seed significance, submission checklist, or "what should I work on next."

**Knows:** current state (1.0882 val_bpb, 42.2MB artifact), the 8 failure patterns (P1–P8), full technique catalog with epistemic labels, and the 9-step compression pipeline.

---

## `/skill-creator`
Create new skills, improve existing ones, and measure skill performance.

**Use when:** you want to build a skill from scratch, iterate on an existing skill, run evals, benchmark with variance analysis, or optimize a skill's description for better triggering.

---

## `/update-config`
Configure the Claude Code harness via `settings.json`.

**Use when:** setting up automated behaviors ("from now on when X…"), managing permissions ("allow X", "add permission"), setting env vars, troubleshooting hooks, or editing `settings.json` / `settings.local.json`.

---

## `/less-permission-prompts`
Scan transcripts for frequent read-only Bash/MCP tool calls and add them to the permission allowlist.

**Use when:** you're getting too many permission prompts for commands you always approve.

---

## `/loop`
Run a prompt or slash command on a recurring interval (e.g. `/loop 5m /foo`). Omit the interval to let the model self-pace.

**Use when:** you want to poll for status, repeat a task on a schedule, or run something continuously (e.g. "check the deploy every 5 minutes").

---

## `/schedule`
Create, update, list, or run scheduled remote agents on a cron schedule.

**Use when:** you want a recurring remote agent, automated tasks, or cron jobs for Claude Code.

---

## `/claude-api`
Build, debug, and optimize Claude API / Anthropic SDK applications with prompt caching.

**Use when:** code imports `anthropic` / `@anthropic-ai/sdk`, you're working with the Claude API, tuning caching/thinking/tool use/batch/files, or migrating between model versions (4.5 → 4.6 → 4.7).

---

## `/simplify`
Review changed code for reuse, quality, and efficiency, then fix issues found.

**Use when:** you want a second pass on code you just wrote to trim dead weight and improve clarity.

---

## `/init`
Initialize a new `CLAUDE.md` file with codebase documentation.

**Use when:** starting a new project or repo that doesn't yet have a `CLAUDE.md`.

---

## `/review`
Review a pull request.

**Use when:** you want Claude to read a PR and give structured feedback on correctness, style, and risks.

---

## `/security-review`
Complete a security review of pending changes on the current branch.

**Use when:** you want to audit staged/committed diffs for OWASP-class vulnerabilities, injection risks, credential exposure, etc.

---

## `/keybindings-help`
Customize keyboard shortcuts and modify `~/.claude/keybindings.json`.

**Use when:** you want to rebind keys, add chord shortcuts, or change the submit key.
