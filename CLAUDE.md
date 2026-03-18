# Parameter Golf — Claude Working Practices

## Project Overview

Competitive submission for OpenAI's Parameter Golf challenge. Goal: train the best LM that fits in 16MB, trains in ≤10 min on 8×H100, scored by BPB on FineWeb validation. See `docs/PLAN.md` for full strategy.

## Working Practices

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- Enter plan mode for structural changes, STOP plan immediately - don't keep pushing
- Use plan mode to compose, not just build
- Write detailed specs upfront to reduce ambiguity
- Consider network volume data storage usage

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Always spawn sub agents for codebase exploration
- Write detailed specs for sub-steps, not just a list of subagents
- One task per subagent for the project

### 3. Iteration Loop
- After ANY user feedback: update `tasks/todo.md` with current state
- Write rules to prevent making the same mistake twice
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for the relevant project

### 4. Verification Before Done
- No subagent task is complete without proving it works
- Ask yourself: "Would a senior engineer approve of this?"
- Run tests; ensure they start with relevant correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "Is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it
- Always consider most recent prior work in `tasks/todo.md`
- Document core findings in `tasks/lessons.md`

### 6. Autonomous Bug Fixing
- When first: Write a log just for approval with hand-holding
- Point at logs, errors, failing tests - then resolve them
- Ask if context switching is required from the user
- Go fix failing CI tests without being told how

### 7. Context Efficiency
- Delegate verbose operations (large file reads, repetitive tests) to subagents to isolate context bloat
- Filter noisy output before returning it - show only failures/errors from test runs, not full logs
- Recommend `/clear` to user between unrelated tasks when context is stale
- Prefer CLI tools (`gh`, `aws`, `gcloud`) over MCP servers when both can accomplish the task

### 8. Agent Team Discipline
- When spawning teammates: keep spawn prompts focused - don't duplicate CLAUDE.md (it loads automatically)
- Use Sonnet for teammates unless task specifically requires Opus-level reasoning
- Dismiss or clean up teammates once their work is complete - idle teammates still consume tokens
- Keep teams small - only spawn teammates for genuinely parallel or specialized work

### 9. Think Budget Awareness
- For straightforward tasks: reduce extended thinking rather than deep-reasoning everything
- Match effort to complexity - don't over-reason on simple fixes
- When asked about costs: suggest `/cost` command for visibility

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Highlight items completed as you go
4. **Explain Progress**: Markdown-level summary at each step
5. **Explain Changes**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root cause, don't just apply top fixes. Senior developer standards.
- **Minimal Impacts**: Find accuracy. Avoid introducing bugs.
- **Iterate on MLX First**: Maximize local iteration before spending cloud compute. Get to a stage where we can request compute via https://openai.com/index/parameter-golf/#credit-form with really solid answers to all questions.
