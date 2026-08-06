# ROGII Wellbore Geology

**Read [`AGENTS.md`](AGENTS.md) now. It is the canonical entry point and this file is only a
pointer to it.**

`AGENTS.md` is agent-agnostic on purpose, so Claude Code, Codex, and anyone else picking this
up work from the same instructions and there is no second copy to drift. Do not move content
back into this file.

The short version, so nothing is lost if only this file is auto-loaded:

- Predict TVT along horizontal wellbores. Pooled RMSE over 3,783,989 rows and 773 wells, lower
  is better. Deadline **2026-08-05**, submit new candidates by **2026-08-03**, selection locks
  **2026-08-04**.
- Run `make preflight` before anything else. It verifies the deployed protocol still reproduces
  8.2826, 8.7300 and 8.9641. If it fails, stop; nothing this checkout produces is admissible.
- For a fresh Claude Code session, run `make claude-handoff` after preflight, then read the
  dated brief it writes under `scratchpad/`.
- Read path is `AGENTS.md`, `QUEUE.md`, `LEDGER.tsv`, plus `SESSION_SUMMARY.md` for the live
  handoff. Nothing in `archive/` unless a grep sends you there.
- Run Python as `.venv/bin/python` from this directory. Never `uv run`, never `pip install`,
  never `sudo`. `make help` lists the maintained entry points.
