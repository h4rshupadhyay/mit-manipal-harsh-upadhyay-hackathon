# Repository Guidance

## Before changing code

- Read `CONTEXT.md` for canonical domain language. Use those terms in public
  contracts, tests, documentation, and UI copy.
- For implementation work, read the approved design at
  `docs/superpowers/specs/2026-10-04-financial-risk-engine-design.md` and the
  task plan at `docs/superpowers/plans/2026-10-04-financial-risk-engine.md`.
  Implement only the requested task and preserve its declared interfaces for
  later tasks.
- Consult the relevant ADR under `docs/adr/` before changing architecture,
  calibration, valuation, validation, or deployment decisions.

## Engineering workflow

- Use test-driven development: add a focused failing test, verify the expected
  failure, implement the smallest passing change, then run the full applicable
  suite.
- Keep module seams typed with Pydantic records and small public interfaces.
  Financial calculations and policy decisions must remain deterministic and
  outside generative-model code paths.
- Treat timestamps, units, provenance, and version metadata as required domain
  data. Reject incomplete or inconsistent inputs instead of filling silent
  defaults.
- Keep Impact Score, Confidence, Portfolio Materiality, and Action Priority as
  separate concepts and fields.
- Preserve offline replay: analysis must never perform an implicit network
  refresh, and tests must run without network access unless explicitly marked
  as acquisition or model smoke tests.

## Data and artifacts

- Commit only redistributable, project-authored, synthetic, or derived data.
  Keep secrets, model weights, local databases, caches, large downloads, and
  restricted third-party text out of Git.
- Preserve source terms, hashes, snapshot identifiers, timestamps, and version
  metadata for every committed or generated artifact.
- Never overwrite an immutable snapshot or a locked final evaluation result;
  create a new version instead.

## Quality gate

Use the commands declared in `pyproject.toml`. Before completing a task, run
its focused tests and then the applicable repository-wide checks (`pytest`,
`ruff check`, and `mypy`). Read each command's output and report any skipped or
unavailable check explicitly.

Commit one coherent task at a time using the commit message from the approved
plan when one is provided.
