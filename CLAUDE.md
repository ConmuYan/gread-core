# CLAUDE.md

@AGENTS.md

## Environment

**Dedicated venv**: `/data1/mq/conda_envs/gread-core`
**Python**: 3.10.20 | **CUDA**: 12.4 | **PyTorch**: 2.6.0+cu124

All bash commands that run Python or pip MUST use the gread-core venv:
```bash
# Python
/data1/mq/conda_envs/gread-core/bin/python ...
# Pip
/data1/mq/conda_envs/gread-core/bin/pip ...
# Pytest
/data1/mq/conda_envs/gread-core/bin/pytest ...
```

For CUDA library resolution, prefix with:
```bash
NVIDIA_BASE=/data1/mq/conda_envs/gread-core/lib/python3.10/site-packages/nvidia
export LD_LIBRARY_PATH=$(find $NVIDIA_BASE -name "lib" -type d | tr '\n' ':')$LD_LIBRARY_PATH
```

Never use system python, conda base, or other environments for this project.

## Claude Code Specific Rules

- Use Plan Mode before any task that touches more than one module.
- Prefer project skills under .claude/skills/ when available.
- Use subagents for exploration, review, and log-heavy work.
- Keep the main conversation clean: do not paste long logs; write logs to artifacts/logs and summarize.
- After every implementation task, update docs/engineering/PROJECT_STATE.md.
- Never claim completion until required validation commands pass or the failure is documented.