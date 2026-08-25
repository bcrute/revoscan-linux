<!-- coordinate-claude-work:start -->
## Codex planner/reviewer workflow

When `.coordination/README.md` exists, read it and the live files it names before
acting. Codex/ChatGPT is the planner and reviewer; the configured executor is the
product-code writer.

- Codex owns `.coordination/planner/` and `.coordination/reviews/`.
- The configured executor owns product edits. Its adapter owns
  `.coordination/coder/` when the runtime cannot safely do so itself.
- Codex owns the overall goal, assigns one bounded executor subgoal at a time, and
  sets `planner/goal.md` to `done` only after its completion criteria are met.
- After each implementation handoff, Codex reviews the complete diff and evidence,
  then requests corrections, assigns the next subgoal, or records the overall
  completion in `reviews/completion.md`.
- Codex does not edit product code unless the user explicitly suspends this role
  boundary.
- No agent commits, pushes, deploys, or changes external systems unless the active
  assignment explicitly authorizes it.
<!-- coordinate-claude-work:end -->
