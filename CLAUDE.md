<!-- coordinate-claude-work:start -->
## Claude implementation workflow

When `.coordination/README.md` exists, act as the implementation agent for the
active task packet supplied by the runner. Claude Code already loads normal
repository instructions. Do not preload the coordination history, goal,
roadmap, reports, or reviews unless the active assignment references them or a
concrete missing decision requires them.

- Implement only the active assignment and its current review corrections.
- Own product-code edits and `.coordination/coder/`; do not edit planner or review
  files.
- Update coder status to `implementing` before product edits; a concise
  `Current activity` is sufficient. Do not manually mirror Claude's native task
  list into coordination files. End by writing a truthful latest report and
  setting status to `review` or `blocked`.
- Run the focused evidence requested by the assignment. A check not run is
  `not run`, never passing.
- Use Claude Code's native orchestration. Proactively delegate genuinely
  independent work to the configured Sonnet subagents or teammates, with at most
  two active concurrently. Keep trivial or sequential work in the lead; the
  Opus lead owns integration and the final report.
- Preserve unrelated work. Do not commit, push, deploy, or mutate external systems
  unless the active assignment explicitly authorizes it.
- Stop on a missing owner decision, credential, destructive action, or material
  scope expansion and record the exact blocker.
- Never set the overall goal to `done`; that signal belongs to Codex after review.
<!-- coordinate-claude-work:end -->
