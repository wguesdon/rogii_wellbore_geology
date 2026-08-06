"""Greedy ensemble over gated bases, with the selection-bias guard. Prints what to deploy.

    .venv/bin/python harness/ensemble.py                      # axis NOGEOM, ledger candidates
    .venv/bin/python harness/ensemble.py --axis LEAK
    .venv/bin/python harness/ensemble.py --base a --base b     # explicit candidate pool

WHY THE GUARD EXISTS
--------------------
Screening N bases by blend-add and then reporting the winner on the same wells manufactures a
gain that is not there: a best-of-N maximum is a maximum of N noisy statistics. On 2026-07-26 a
57-base greedy on this data improved its selection half monotonically (-0.030, -0.043, -0.065,
-0.075) while the held-out half wandered (-0.009, +0.006, -0.019, +0.015), and the two directions
agreed on one base out of four and six.

So selection happens on one half of the WELLS and the number reported comes from the other half,
run in BOTH directions as two independent searches. The full-773 cross-fit number of the chosen
set is also printed, labelled OPTIMISTIC, only so it can be compared against the record.

THE CEILING YOU CANNOT BEAT
---------------------------
An unconstrained least-squares fit of all 60 banked bases to the truth, in sample, gives 7.9537.
No subset, weighting or shrinkage of existing bases goes below that. If this script's honest
number is not moving, the answer is a new base, not a better search.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

HARNESS = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS))
from gate import (  # noqa: E402
    ALPHA, AXES, GEOM_FED, INADMISSIBLE, LEDGER, N_FOLDS, Data, gateable)

HALF_SEED = 20260726


def ledger_candidates(axis: str) -> list[str]:
    """Bases the ledger marks KEEP or MARGINAL, minus anything ineligible for this axis."""
    if not LEDGER.exists():
        return []
    out = []
    with LEDGER.open() as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r.get("verdict") in ("KEEP", "MARGINAL"):
                n = r["base"]
                if n in INADMISSIBLE or (axis == "NOGEOM" and n in GEOM_FED):
                    continue
                if n not in out:
                    out.append(n)
    return out


def halves(d: Data) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic well-level half split, returning the two row-index arrays."""
    n = len(d.wells)
    perm = np.random.default_rng(HALF_SEED).permutation(n)
    a = np.concatenate([d.wells[i] for i in perm[: n // 2]])
    b = np.concatenate([d.wells[i] for i in perm[n // 2:]])
    return a, b


def nested(d: Data, bases: list[str], sel: np.ndarray, ev: np.ndarray) -> tuple[float, float]:
    """Score a base set in one direction of the nested split.

    Args:
        d: The loaded data.
        bases: Base set.
        sel: Rows of the selection half. Cross-fit within it; the only thing a search may see.
        ev: Rows of the held-out half, predicted from a Ridge fitted on the whole of `sel`.

    Returns:
        `(selection-half pooled RMSE, held-out pooled RMSE)`.
    """
    m = np.column_stack([d.col(b) for b in bases])
    stack = np.zeros(len(d.yr))
    sx, sy, sg = m[sel], d.yr[sel], d.groups[sel]
    pred = np.zeros(len(sel))
    for tr, va in GroupKFold(N_FOLDS).split(sx, groups=sg):
        r = Ridge(alpha=ALPHA, positive=True)
        r.fit(sx[tr], sy[tr])
        pred[va] = r.predict(sx[va])
    stack[sel] = pred
    r = Ridge(alpha=ALPHA, positive=True)
    r.fit(sx, sy)
    stack[ev] = r.predict(m[ev])
    out = d.postproc(stack)
    return d.rmse(out, sel), d.rmse(out, ev)


def greedy(d: Data, base: list[str], pool: list[str], sel: np.ndarray, ev: np.ndarray,
           rounds: int, tag: str) -> tuple[list[str], float, float]:
    """Forward selection scored ONLY on the selection half.

    Returns:
        `(chosen bases, held-out delta, selection-half delta)`.
    """
    s0, e0 = nested(d, base, sel, ev)
    chosen, cur, best_e = [], s0, e0
    rest = [p for p in pool if p not in base]
    for i in range(rounds):
        best = None
        for c in rest:
            s, e = nested(d, base + chosen + [c], sel, ev)
            if best is None or s < best[1]:
                best = (c, s, e)
        if best is None or best[1] >= cur - 1e-4:
            print(f"    [{tag}] round {i + 1}: nothing improves the selection half. STOP.")
            break
        c, s, e = best
        chosen.append(c)
        rest.remove(c)
        cur, best_e = s, e
        print(f"    [{tag}] + {c:34s} sel {s:.4f} ({s - s0:+.4f})   "
              f"HELD-OUT {e:.4f} ({e - e0:+.4f})")
    return chosen, best_e - e0, cur - s0


def main() -> int:
    """Run the guarded greedy on one axis and print the deployable base list."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--axis", choices=list(AXES), default="NOGEOM")
    ap.add_argument("--base", action="append", default=[], help="explicit candidate pool")
    ap.add_argument("--all", action="store_true", help="every gateable base, not just the ledger")
    ap.add_argument("--rounds", type=int, default=6)
    args = ap.parse_args()

    d = Data()
    start, record = AXES[args.axis]
    if args.base:
        pool = args.base
    elif args.all:
        pool = [b for b in gateable()
                if b not in start and not (args.axis == "NOGEOM" and b in GEOM_FED)]
    else:
        pool = ledger_candidates(args.axis)
    print(f"axis {args.axis}   start {start}")
    print(f"candidate pool: {len(pool)} bases  <- this is the best-of-N denominator")
    if not pool:
        print("nothing to search. Gate some bases first: harness/gate.py --base NAME")
        return 0

    got = d.rmse(d.run(start))
    good = abs(got - record) <= 0.002
    print(f"KNOWN-WINNER: record {record:.4f} got {got:.4f} "
          f"{'PASS' if good else '*** FAIL, stopping ***'}\n")
    if not good:
        return 1

    a, b = halves(d)
    print("greedy, two independent searches, each blind to the half it is scored on:")
    ch_ab, d_ab, s_ab = greedy(d, start, pool, a, b, args.rounds, "A->B")
    ch_ba, d_ba, s_ba = greedy(d, start, pool, b, a, args.rounds, "B->A")

    print()
    print("=" * 78)
    print("RESULT")
    print("=" * 78)
    print(f"  chosen A->B : {ch_ab}")
    print(f"  chosen B->A : {ch_ba}")
    print(f"  overlap     : {sorted(set(ch_ab) & set(ch_ba))}")
    print(f"  HELD-OUT delta: A->B {d_ab:+.4f}   B->A {d_ba:+.4f}   "
          f"mean {0.5 * (d_ab + d_ba):+.4f}")
    print(f"  (selection-half delta was {s_ab:+.4f} / {s_ba:+.4f}; the gap between those and")
    print(f"   the held-out numbers is exactly the selection bias the guard exists to expose)")

    keep = sorted(set(ch_ab) & set(ch_ba)) or sorted(set(ch_ab) | set(ch_ba))
    label = "intersection" if set(ch_ab) & set(ch_ba) else "union (no overlap, weak evidence)"
    final = start + keep
    cf = d.rmse(d.run(final))
    print()
    print(f"  DEPLOYABLE SET ({label}): {final}")
    print(f"  full-773 cross-fit: {cf:.4f}  ({cf - got:+.4f} vs the {args.axis} start)")
    print(f"  [that number selected on all wells, so it is OPTIMISTIC; the held-out delta above")
    print(f"   is the honest one]")
    print()
    print(f"  ceiling: an in-sample OLS over all 60 banked bases gives 7.9537. If the honest")
    print(f"  delta is flat, the answer is a NEW BASE, not a better search.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
