# Context Hygiene

## Working Memory Management

### Principle
Keep working memory focused on the current task. Archive when done.

### Practices
1. **Task-Specific Context**: Load only what's needed
2. **Regular Cleanup**: Archive completed tasks
3. **ADR Creation**: Document decisions before moving on

## Decision Archiving

### When to Create ADR
- Architecture decisions
- Non-obvious implementation choices
- Trade-off resolutions
- API/design changes

### ADR Template
```markdown
# ADR_XXXX_title

## Status
Proposed | Accepted | Deprecated | Superseded

## Context
What is the issue?

## Decision
What did we decide?

## Consequences
What does this mean?
```

## Context Categories

### Priority Context (Always Loaded)
- Project type and domain
- Core principles (score-blind MEP, evidence contract)

### Working Context (Task-Specific)
- Current task description
- Relevant files/modules
- Active decisions

### Archived Context (ADRs, Docs)
- Past decisions
- Experiment results
- Design discussions

## Hygiene Checklist

- [ ] Working memory focused on current task
- [ ] Completed tasks archived
- [ ] Decisions documented in ADRs
- [ ] Specs updated for changes

---

*See `docs/decisions/` for existing ADRs*
