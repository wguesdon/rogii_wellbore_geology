"""Report and enforce the ROGII forward model campaign state.

ROGII does not benefit from a generic count of tabular model families. Its measured
recombination ceiling is already below the target. This checker therefore enforces
the relevant anti stopping condition: while CV remains above the user target and the
deadline is open, at least one distinct forward model or observation axis must remain
open and have a concrete next action.
"""

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
AXES_PATH = ROOT / "harness" / "forward_axes.json"
TARGET_PATH = ROOT / "harness" / "forward_target.json"
OVERRIDE_PATH = ROOT / "harness" / "campaign_override_log.tsv"
VALID_STATUSES = {"open", "tried", "closed"}


def load_json(path: Path) -> Any:
    """Load UTF 8 JSON from a project file.

    Args:
        path: JSON file to load.

    Returns:
        Parsed JSON content.
    """
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _has_quantified_bar(next_action: str) -> bool:
    """Return whether a next action names a numeric threshold a result could fail.

    Looks for a bar stated as feet, a correlation, a fraction or a percentage, near
    language that marks it as a threshold rather than an incidental measurement.

    Args:
        next_action: The axis's declared next action.

    Returns:
        True when a quantified bar is present.
    """
    lowered = next_action.lower()
    if not re.search(r"(bar|must|threshold|floor|below|above|exceed|beat)", lowered):
        return False
    return bool(re.search(r"\d*\.?\d+\s*(ft|feet|percent|%)", lowered)
                or re.search(r"(correlation|corr|fraction|ratio)[^.]{0,40}?\d*\.\d+", lowered)
                or re.search(r"\d*\.\d+[^.]{0,40}?(correlation|corr|fraction|ratio)", lowered))


def validate_axes(axes: list[dict[str, str]], target: dict[str, Any] | None = None) -> list[str]:
    """Return structural errors in the forward axis registry.

    An open axis must carry a next action that is executable NOW. Deferral language
    such as "post-deadline" or "multi-day build" is rejected, because deferring the
    only open axis is early stopping wearing a different hat: the registry still shows
    an open axis while no work is actually scheduled. A large idea is not deferred, it
    is decomposed until its first step fits the time available.

    An open axis must ALSO state a quantified bar. `AGENTS.md` rule 7: on 2026-08-02 a
    neighbour curvature field passed all three of its pre-registered bars, which asked
    for direction and controls, and was worth 0.0035 ft. A bar that never names the
    smallest gain worth having cannot fail an effect that is real and negligible, and
    this CV cannot resolve a per-well correction below ~0.10 ft.

    Args:
        axes: Parsed forward axis entries.
        target: Campaign target settings, for the deferral marker list.

    Returns:
        Human readable validation failures.
    """
    errors: list[str] = []
    seen: set[str] = set()
    markers = [str(m).lower() for m in (target or {}).get("deferral_markers", [])]
    enforce = bool((target or {}).get("next_action_must_be_executable_now", False))
    for entry in axes:
        axis = str(entry.get("axis", "")).strip()
        status = str(entry.get("status", "")).strip()
        evidence = str(entry.get("evidence", "")).strip()
        next_action = str(entry.get("next_action", "")).strip()
        close_reason = str(entry.get("close_reason", "")).strip()
        if not axis:
            errors.append("Axis without a name.")
        elif axis in seen:
            errors.append(f"Duplicate axis: {axis}.")
        seen.add(axis)
        if status not in VALID_STATUSES:
            errors.append(f"{axis}: invalid status '{status}'.")
        if not evidence:
            errors.append(f"{axis}: missing evidence.")
        if status == "open" and not next_action:
            errors.append(f"{axis}: open axis lacks a next action.")
        if status in {"tried", "closed"} and not close_reason:
            errors.append(f"{axis}: {status} axis lacks a close reason.")
        if status == "open" and enforce and next_action:
            lowered = next_action.lower()
            hit = [m for m in markers if m in lowered]
            if hit:
                errors.append(
                    f"{axis}: open axis defers its next action ({', '.join(hit)}). "
                    "An open axis must name a step that can START NOW. Decompose the "
                    "idea until its first bounded experiment fits the time available."
                )
            if not _has_quantified_bar(next_action):
                errors.append(
                    f"{axis}: open axis states no QUANTIFIED bar. Name the smallest gain "
                    "worth having, in ft or as an explicit correlation or fraction. "
                    "AGENTS.md rule 7: the curvature field passed every bar it was given "
                    "and was worth 0.0035 ft, because the bars asked whether the effect "
                    "was real and never how big."
                )
    return errors


def is_deadline_open(deadline: str) -> bool:
    """Return whether the UTC calendar deadline has not passed.

    Args:
        deadline: ISO calendar date for the competition deadline.

    Returns:
        True when today is on or before the deadline.
    """
    return dt.date.today() <= dt.date.fromisoformat(deadline)


def render_report(axes: list[dict[str, str]], target: dict[str, Any], best_cv: float) -> str:
    """Render the current campaign state for humans and automation.

    Args:
        axes: Validated forward axis entries.
        target: Campaign target settings.
        best_cv: Best honest deployed protocol cross fit CV.

    Returns:
        Multi line state report.
    """
    open_axes = [entry for entry in axes if entry["status"] == "open"]
    objective = target.get("objective", "target")
    lines = [
        f"OBJECTIVE: {objective}"
        + (f" (public LB {target['objective_lb']})" if "objective_lb" in target else ""),
        f"Best honest CV: {best_cv:.4f} / working proxy target {target['goal_cv']:.4f}",
        f"Deadline: {target['deadline']} ({'OPEN' if is_deadline_open(target['deadline']) else 'CLOSED'})",
        f"Open forward axes: {len(open_axes)} / minimum {target['minimum_open_axes_while_unreached']}",
    ]
    if len(open_axes) < int(target["minimum_open_axes_while_unreached"]) and \
            best_cv > float(target["goal_cv"]) and is_deadline_open(str(target["deadline"])):
        lines.append("")
        lines.append("BREADTH FAILURE: no open axis while the objective is unreached and the")
        lines.append("deadline is open. Open one and name its first bounded experiment.")
    if open_axes:
        lines.append("")
        lines.append("Required next actions:")
        for entry in open_axes:
            lines.append(f"  * {entry['axis']}: {entry['next_action']}")
    return "\n".join(lines)


def campaign_ready(axes: list[dict[str, str]], target: dict[str, Any], best_cv: float) -> bool:
    """Return whether a campaign completion statement is allowed.

    Args:
        axes: Validated forward axis entries.
        target: Campaign target settings.
        best_cv: Best honest deployed protocol cross fit CV.

    Returns:
        True only when the target is met or the deadline has passed.
    """
    if best_cv <= float(target["goal_cv"]):
        return True
    if not is_deadline_open(str(target["deadline"])):
        return True
    return False


def append_override(best_cv: float, reason: str) -> None:
    """Append a transparent campaign completion override.

    Args:
        best_cv: Best honest CV at the time of override.
        reason: User supplied reason for the override.
    """
    write_header = not OVERRIDE_PATH.exists()
    with OVERRIDE_PATH.open("a", encoding="utf-8") as handle:
        if write_header:
            handle.write("timestamp_utc\tbest_cv\treason\n")
        timestamp = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
        handle.write(f"{timestamp}\t{best_cv:.6f}\t{reason}\n")


def main() -> None:
    """Run the forward axis status report or completion check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best-cv", required=True, type=float)
    parser.add_argument("--lock", action="store_true")
    parser.add_argument("--override", default="")
    args = parser.parse_args()

    axes = load_json(AXES_PATH)
    target = load_json(TARGET_PATH)
    errors = validate_axes(axes, target)
    if errors:
        print("FORWARD AXIS REGISTRY INVALID:", file=sys.stderr)
        print("\n".join(f"  * {error}" for error in errors), file=sys.stderr)
        raise SystemExit(2)

    print(render_report(axes, target, args.best_cv))
    if not args.lock:
        return
    if campaign_ready(axes, target, args.best_cv):
        # A closing reason is worth MORE once the deadline has passed, not less: this is the
        # post-mortem. Until 2026-08-05 the override was dropped on this path, so the campaign
        # could not record its own closure and the final entry had to be written by hand.
        if args.override:
            append_override(args.best_cv, args.override)
            print("\nCAMPAIGN CHECK: COMPLETE OR DEADLINE CLOSED. Closing reason logged to "
                  "harness/campaign_override_log.tsv.")
            return
        print("\nCAMPAIGN CHECK: COMPLETE OR DEADLINE CLOSED.")
        return
    if args.override:
        append_override(args.best_cv, args.override)
        print("\nCAMPAIGN CHECK: OVERRIDDEN. The reason is logged in harness/campaign_override_log.tsv.")
        return
    print("\nCAMPAIGN CHECK: CONTINUE. The objective is unreached and the deadline is open.")
    print("")
    print("These are NOT admissible reasons to stop or to defer the next experiment:")
    for reason in target.get("inadmissible_stop_reasons", []):
        print(f"  x {reason}")
    print("")
    print("Only these are:")
    for reason in target.get("admissible_stop_reasons", []):
        print(f"  + {reason}")
    note = target.get("note_on_difficulty", "")
    if note:
        print("")
        print(note)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
