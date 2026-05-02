# CLAUDE.md

@AGENTS.md

## Claude Code Specific Rules

- Use Plan Mode before any task that touches more than one module.
- Prefer project skills under .claude/skills/ when available.
- Use subagents for exploration, review, and log-heavy work.
- Keep the main conversation clean: do not paste long logs; write logs to artifacts/logs and summarize.
- After every implementation task, update docs/engineering/PROJECT_STATE.md.
- Never claim completion until required validation commands pass or the failure is documented.