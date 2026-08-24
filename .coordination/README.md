# Coordination workflow

This directory is the durable mailbox between the owner, Codex planner/reviewer,
and the configured implementation executor. It is repository state, not a chat transcript.

## Roles and ownership

- Owner: product decisions, authority, and scope changes.
- Codex/ChatGPT: `.coordination/planner/` and `.coordination/reviews/`.
- Configured executor: product files. Its Coordinator adapter may write
  `.coordination/coder/` on the runtime's behalf.

Everyone may read all files. Only the named owner writes each coordination area.
Repository-level `AGENTS.md` and `CLAUDE.md` remain binding.

## Live files

- `PROJECT.md`: durable facts, commands, boundaries, and decisions.
- `planner/goal.md`: Codex's overall objective and the durable `done` signal.
- `planner/current-task.md`: the current executor-sized subgoal and review round.
- `coder/status.md`: the cross-system implementation signal and an optional
  concise current activity; it does not mirror a provider's native task list.
- `coder/latest-report.md`: the executor's most recent handoff.
- `reviews/latest.md`: Codex's verdict on that handoff.
- `reviews/completion.md`: Codex's user-facing overall-goal result.

## State machine

Overall goal: `idle -> active -> done | blocked`

Each subgoal: `ready -> implementing -> review -> accepted`

Codex may return `review -> changes_requested -> implementing` while retaining the
same task ID. Either agent may record `blocked` when progress needs owner authority
or a missing external capability.

Every executor invocation is one implementation handoff. Codex reviews its complete
result, then requests corrections, assigns the next subgoal, or sets the overall
goal to `done`. A watcher treats those file states as signals; it never invents a
task or a verdict itself.

## Watchers

The executor-side watcher launches the configured runtime only for `ready` or
`changes_requested` subgoals and exits when `planner/goal.md` says `done` or
`blocked`. The Codex-side watcher launches a review only when executor status says
`review` or `blocked`. They may run as separate processes, or one `both` watcher
may relay both sides. Runtime locks and status JSON live under `runtime/` and are
not committed.

In a terminal, the watcher presents a persistent dashboard for the overall goal,
roadmap, active acceptance contract, activity, timers, generated tokens, separate
context/cache counts, and any provider-reported native workers. Repeated stream
events for one message ID count once. Agent output is retained in
`runtime/relay.log`;
`--no-dashboard` restores line-oriented output. A failed or incomplete handoff
stops the watcher instead of retrying or waiting indefinitely.

With the Claude adapter, the automatic watcher supplies only the active task packet to
an Opus lead and lets Claude Code manage native Sonnet subagents. Native agent teams require an
interactive session; use `start_claude_team.py` alongside a Codex-only watcher
rather than recreating Claude's team task list or mailbox here.

With the mini-swe-agent adapter, Coordinator runs one noninteractive, step- and
wall-time-bounded agent, records its trajectory under `runtime/trajectories/`, and
writes the coder status/report from observed results. The local model does not own
coordination files and no nested mini-swe-agent workers are exposed in this integration.

## Local web app

The installed skill ships a localhost dashboard for this coordination directory:

```bash
coordinator serve --repo .
```

Run it from any initialized project; nothing is copied into the project. `--repo`
accepts an absolute path or a path relative to the current working directory, and
the server refuses to start unless that path contains `.coordination/README.md`.
Open `http://127.0.0.1:8765` in a browser. The page polls `/api/state` once per
second and shows the overall goal and progress, roadmap, current task contract,
coder status, latest review, live metrics, observed provider workers,
watcher status files, the managed watcher, and the tail of `runtime/relay.log`.

The only controls are start and stop for one fixed automatic watcher: the `both`
relay for this repository, with the dashboard disabled. The web app never chooses
a task, writes a verdict, or edits coordination files. It refuses to start a
second watcher while `runtime/watcher-both.lock` is held, so a watcher already
running in a terminal stays untouched. That watcher's output is appended to
`runtime/relay.log`, which is also what the page displays.

Ctrl-C stops the server, and closing the server stops any watcher it started; a
watcher started in a terminal is unaffected. The server binds `127.0.0.1` by
default and has no authentication, so it is for the owner's machine only. LAN
exposure is deliberately deferred and would need authentication first.

Native interactive Claude teams are not controllable from this page. They require
Claude's own terminal UI through `start_claude_team.py`; the web app manages only
the automatic watcher.

## Evidence and history

- A command not run is `not run`.
- Review the diff and behavior, not merely the coder report.
- Keep live files concise; preserve only useful accepted-task summaries under
  `history/`.
- Coordination records do not replace tests, CI, code review, or product docs.
