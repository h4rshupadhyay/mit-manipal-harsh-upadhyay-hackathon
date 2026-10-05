# Development, commit, and push workflow

## Intent and authority

The [approved plan](superpowers/plans/2026-10-04-financial-risk-engine.md)
requires small, coherent, tested commits. It does not require a branch per task,
a branch rename, or separate pushes of old commits. The task boundaries remain
authoritative; this document defines the publication and handoff procedure.

Going forward, the publication checkpoint is each completed, reviewed task,
provided pushing to the exact remote and branch is authorized. Commit cadence
and push cadence are different: a commit records a local development checkpoint;
a successful push publishes committed history and provides a remote checkpoint.
Neither means the application is deployed or merged into `main`.

## Workspace and branches

- Develop in the IDE-visible repository on a feature branch, keeping unfinished
  work off `main`. A branch isolates committed history; it does not back up or
  hide uncommitted edits. Record unfinished work before a handoff.
- Reuse the agreed feature branch across related tasks. Task-specific branches
  are optional, not mandated by the plan. Do not switch, rename, merge, or delete
  branches merely to make their names match the latest task.
- Use an isolated worktree only when needed and agreed; disclose its path so the
  user can inspect the code. Do not silently relocate development to `/tmp`.
- Preserve user changes. Inspect status before starting and before staging;
  never reset, clean, or discard unrelated changes to obtain a clean tree.

## One task: red → green → review → commit → push

1. **Establish scope.** Read [AGENTS.md](../AGENTS.md), the active task, and the
   resume ledger. Inspect `git status --short --branch` and recent commits.
   Record the base commit and exact permitted file scope. Stop for a scope
   decision if the task requires an undeclared interface or dependency change.
2. **Complete TDD.** Add one focused behavioral test, observe its expected
   failure, then make the smallest implementation that passes. Repeat as needed.
   Never commit the failing red phase or a temporary broken state.
3. **Verify and review.** Run the focused tests and applicable repository-wide
   checks from AGENTS.md; include the plan's phase gates. Inspect outputs, not
   just exit codes. Review against both the task specification and repository
   standards. Resolve findings and rerun checks affected by fixes. If a check
   is unavailable or skipped, disclose it; do not call it passed or declare an
   unmet required gate complete. Documentation-only tasks need diff and link
   review, not artificial prose tests or an unrelated full algorithm suite.
4. **Commit one coherent outcome.** Include the behavior or interface and its
   corresponding tests. Include documentation only when required by that
   behavior; otherwise use a separately requested documentation task. Stage
   explicit paths, inspect `git diff --cached --check` and `git diff --cached`,
   then use the plan's outcome-oriented message when provided. No blanket
   `git add .`, unrelated formatting, ingestion-through-UI mega-commits, or
   micro-commits of every edit. Every implementation commit must leave existing
   tests green. A necessary upstream repair may be its own coherent, tested
   commit; record the scope decision. If review finds a defect after a commit,
   add a green repair commit rather than rewriting published history.
5. **Publish the task checkpoint when authorized.** Verify the remote URL,
   destination branch, upstream, and outgoing commits. Confirm that the range
   contains only reviewed checkpoints and permitted artifacts, with no secrets,
   model weights, or restricted source data. Push the exact branch without
   force; do not use `--all`, push to `main`, or infer authority to publish to a
   new destination. Ask for explicit destination approval when missing or
   required by the permission gate. A non-fast-forward rejection requires
   inspection and a coordination decision, not an automatic force push/rebase.
6. **Verify and hand off.** Confirm the remote branch SHA equals the intended
   local checkpoint. Record the commit, verification evidence, review outcome,
   and actual publication status in the ledger. Report one of: uncommitted,
   committed but unpushed, or pushed and verified. Do not report a denied,
   timed-out, or merely attempted push as successful.

Run routine Git and verification commands directly. Use the model-routing and
narrow-context rules in AGENTS.md for implementation and review; do not spin up
an agent just to run tests or Git. Reuse valid verification evidence for the
same unchanged tree; rerun affected checks after edits, and retain final gate
evidence for the exact committed implementation.

## Publishing an existing backlog

A normal fast-forward push of the approved HEAD transfers all missing commits
with their original boundaries and order. It does not squash them. Separate
pushes of historical checkpoints do not recreate the original development-time
push events; use them only if the user specifically requests that publication
sequence. Do not rewrite, cherry-pick, or rename history to simulate it.

As observed on 2026-10-05, Tasks 2–22 continued on the legacy-named
`feat/task-1-contracts` branch, with 36 commits beyond its remote-tracking tip.
The attempted publication did not execute; destination approval was still
required. The branch name is not evidence that only Task 1 is implemented.
Renaming to a broader feature name was a suggestion, not an approved plan rule
or an authorized action. Reinspect Git for current state; this is a dated record.

## Limits, interruptions, and resumption

Before stopping, update `.superpowers/sdd/<plan>/progress.md` with the active
task, branch, base/current commit, changed paths, red/green state, verification
commands and results, unresolved findings, push status, and exact next action.
If interrupted in the red phase, leave it uncommitted and record that explicitly.
Do not create a broken commit merely to obtain a checkpoint.

On resume, inspect Git and the ledger before doing work. Continue the smallest
unfinished step; do not rerun completed tasks, repeat reviews of unchanged code,
or confuse a local commit with a remote backup. If publication is blocked,
record it and request the missing approval. Do not use another transport or
credential to bypass a denied push. Continue only separately authorized local
work, and carry the unpublished status into each handoff.

The ledger is ignored by Git. In the same checkout it survives a session or
account change, but a new clone will not contain it or uncommitted work. Transfer
the ledger and unfinished changes deliberately when changing workspaces.
Published commits plus tracked AGENTS.md, this workflow, and the plan provide
the portable instructions; live agents and chat state are not a resume record.
