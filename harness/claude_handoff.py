"""Create a compact continuation brief for Claude Code or another agent.

The project keeps the long rules in ``AGENTS.md``. This command adds a fresh
operational snapshot so a new agent sees the active experiments, the current
best score, and the required next action without relying on chat history.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
AXES_PATH = ROOT / "harness" / "forward_axes.json"
TARGET_PATH = ROOT / "harness" / "forward_target.json"
LEDGER_PATH = ROOT / "LEDGER.tsv"
QUEUE_PATH = ROOT / "QUEUE.md"
SESSION_PATH = ROOT / "SESSION_SUMMARY.md"
AWS_REPORT = ROOT / "reports" / "aws_sequence_trials_2026_08_01.md"


def git_tail() -> str:
    """Return recent measurement commits without failing the handoff.

    Returns:
        One line per recent commit, or a diagnostic when Git is unavailable.
    """
    try:
        completed = subprocess.run(
            ["git", "log", "--oneline", "-8", "--", "."],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"git log unavailable: {exc}"
    return completed.stdout.strip()


def ledger_tail(lines: int = 8) -> str:
    """Return the latest ledger rows.

    Args:
        lines: Number of rows to include.

    Returns:
        Ledger tail text.
    """
    rows = LEDGER_PATH.read_text(encoding="utf-8").splitlines()
    return "\n".join(rows[-lines:])


def build_handoff(best_cv: float) -> str:
    """Build the continuation brief from current project state.

    Args:
        best_cv: Current best honest deployed protocol CV.

    Returns:
        Markdown handoff text.
    """
    axes = json.loads(AXES_PATH.read_text(encoding="utf-8"))
    target = json.loads(TARGET_PATH.read_text(encoding="utf-8"))
    open_axes = [entry for entry in axes if entry["status"] == "open"]
    active_text = "\n".join(
        f"* {entry['axis']}: {entry['next_action']}" for entry in open_axes
    ) or ("* NONE, and that is a HARNESS FAILURE, not a conclusion. Open a new axis and "
           "name its first bounded experiment before doing anything else.")
    aws_status = "AWS trial report is not present."
    if AWS_REPORT.exists():
        report_lines = AWS_REPORT.read_text(encoding="utf-8").splitlines()
        launch = [line for line in report_lines if "rogii-cnn-sdf-fall" in line]
        aws_status = "\n".join(launch[-4:]) or aws_status
    now = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    return f"""# Claude Code continuation brief

Generated: {now}
Working directory: `{ROOT}`

Read `AGENTS.md`, `QUEUE.md`, `LEDGER.tsv`, and `SESSION_SUMMARY.md` before changing code.
Run `make preflight` first. A failed known winner makes every new number inadmissible.

## Objective

**The objective is a GOLD medal, public LB {target.get('objective_lb', 5.899)}.** Not an
improvement, not a respectable finish. Current best honest CV is **{best_cv:.4f}** against a
working proxy target of {target['goal_cv']}. Deadline is **{target['deadline']}**.

**Stopping and deferring are the same failure.** These are NOT admissible reasons to stop work,
to defer an experiment, or to spend remaining time documenting instead of running something:
not enough time; too close to the deadline; post-deadline work; a multi-day build; too
speculative; we have plateaued; every axis is closed; the expected gain is too small; unlikely
to succeed; the picks are locked so the score is fixed.

Only four things end the work: the objective is reached, the deadline passes, the owner says
stop, or a NAMED external blocker prevents the next bounded experiment and you can state the
attempt that hit it.

A large idea is never deferred, it is DECOMPOSED until its first bounded experiment fits the
time available, and that experiment is run. `make breadth-status` fails outright if an open
axis defers its next action. When every axis closes, open a new one; a closed axis is a result,
not an ending.

Do not stop because a model family washes. Close only the measured route, then continue on a
different forward model or observation axis. Do not spend a Kaggle submission without explicit
user approval.

## Active axes

{active_text}

## Parallel AWS work

{aws_status}

Poll existing jobs before launching another one. Collect CNN artifacts from `model.tar.gz`, not
`output.tar.gz`. Verify the launcher `--help` flags and require a 773 well OOF plus the normal
rolled control before porting anything.

## Required method discipline

Use `.venv/bin/python`. Do not use `pip`, `uv run`, SMOTE, or destructive Git commands. Save
scripts before execution. Cross fit every learned choice by well. Use `make gate BASE=name` for
new bases. Prefer a fold zero falsifier before a full run. Record every material result in
`LEDGER.tsv`, `QUEUE.md`, and `SESSION_SUMMARY.md`, then commit it with a Conventional Commit.

Do not run the broad `make ensemble` search. It takes hours and cannot break the measured
recombination ceiling. Use a fixed candidate list and the nested half check instead.

## Recent ledger

```text
{ledger_tail()}
```

## Recent commits

```text
{git_tail()}
```

At the end of the turn, write the next concrete command into `QUEUE.md`, update the session
summary, and leave the checkout in a resumable state. If a Claude Code session is interrupted,
rerun `make claude-handoff` to regenerate this brief.
"""


def main() -> int:
    """Write and print a current Claude Code continuation brief.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-cv", type=float, default=7.1593)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "scratchpad" / (
        f"claude_handoff_{dt.datetime.now(dt.UTC).strftime('%Y_%m_%d_%H%M%S')}.md"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    text = build_handoff(args.best_cv)
    output.write_text(text, encoding="utf-8")
    print(f"wrote {output}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
