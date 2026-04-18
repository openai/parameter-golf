# Skills Library

Drop any skill folder into your agent's `.claude/skills/` directory to activate it.

## Custom Skills

| Skill | Invoke | Purpose |
|-------|--------|---------|
| `parameter-golf` | `/parameter-golf` | Strategy guide + adversarial critic for the OpenAI Parameter Golf competition. Knows 8 failure patterns, full technique catalog, compression pipeline. |
| `skill-creator` | `/skill-creator` | Create, iterate, and benchmark new skills with quantitative evals. |

## Superpowers (from official plugin)

| Skill | Purpose |
|-------|---------|
| `brainstorming` | Structured ideation and option generation |
| `dispatching-parallel-agents` | Spawn multiple subagents for independent subtasks |
| `executing-plans` | Work through a written plan step by step |
| `finishing-a-development-branch` | Pre-PR checklist: tests, lint, docs, commit message |
| `receiving-code-review` | Process review feedback and address comments |
| `requesting-code-review` | Prepare and request structured code reviews |
| `subagent-driven-development` | Delegate complex tasks to specialized subagents |
| `systematic-debugging` | Root-cause debugging with hypothesis-driven approach |
| `test-driven-development` | TDD workflow: write tests first, then implement |
| `using-git-worktrees` | Parallel branches via worktrees without stashing |
| `using-superpowers` | Meta-skill: choose the right superpowers skill |
| `verification-before-completion` | Pre-completion checklist before marking task done |
| `writing-plans` | Draft implementation plans before coding |
| `writing-skills` | Write new skill SKILL.md files |

## Plugin Dev

| Skill | Purpose |
|-------|---------|
| `agent-development` | Build Claude Code agents |
| `command-development` | Create slash commands |
| `hook-development` | Write event hooks for Claude Code |
| `mcp-integration` | Integrate MCP servers |
| `plugin-settings` | Manage plugin settings and config |
| `plugin-structure` | Scaffold plugin directory structure |
| `skill-development` | Write and refine SKILL.md files |

## Bundled Skills (built into Claude Code CLI)

These don't have SKILL.md files — they're compiled into the Claude Code binary. Invoke them normally:

| Command | Purpose |
|---------|---------|
| `/update-config` | Edit `settings.json` — hooks, permissions, env vars |
| `/less-permission-prompts` | Scan transcripts → add read-only commands to allowlist |
| `/loop` | Run a prompt on a recurring interval |
| `/schedule` | Create cron-scheduled remote agents |
| `/claude-api` | Build/debug Anthropic SDK apps with prompt caching |
| `/simplify` | Review changed code for quality and trim dead weight |
| `/init` | Generate a `CLAUDE.md` for a new project |
| `/review` | Review a pull request |
| `/security-review` | Audit staged changes for OWASP-class vulnerabilities |
| `/keybindings-help` | Customize `~/.claude/keybindings.json` |

## Usage with Omniclaw / Nanoclaw

Copy skill folders into your agent's `.claude/skills/` directory:

```bash
cp -r skills/parameter-golf /path/to/your-agent/.claude/skills/
cp -r skills/dispatching-parallel-agents /path/to/your-agent/.claude/skills/
```

Or symlink for live updates:
```bash
ln -s $(pwd)/skills/parameter-golf /path/to/your-agent/.claude/skills/parameter-golf
```
