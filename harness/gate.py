"""Gate a candidate base on the deployed protocol and append one row to LEDGER.tsv.

This is the whole gate. There is no other one.

    .venv/bin/python harness/gate.py --base lgbmedrate_2026_07_26
    .venv/bin/python harness/gate.py --base foo --base bar        # several at once
    .venv/bin/python harness/gate.py --list                       # what is gateable

WHAT IT MEASURES, and why each column exists
--------------------------------------------
The DEPLOYED protocol, reproduced exactly: positive Ridge (alpha 1.0) cross-fit over
`GroupKFold(5)` by well, then `last_known + 0.76*stack + 0.24*pf_selector`, then per-well
savgol(17, 3), then the anchored robust degree-4 U-projection at blend 0.70, C 3.0.

  standalone   pooled RMSE of `last_known + base` alone. Recorded, NEVER gated on. Rule 2.
  errcorr      corr(base error, deployed blend error). Below ~0.85 is what earns a place.
  blendadd     pooled RMSE change when the base joins the Ridge. Both axes:
                 LEAK    on Pick-1 {lgbdivmed, xgbdivmed, geom_k16}, deployed 8.2826
                 NOGEOM  on the best no-geom stack {MED3 + realmlp}, deployed-protocol 8.7300
  null         the same blendadd with the base replaced by a within-well-ROLLED copy of
               itself, seeded from the well id. A base whose blendadd is reproduced by its own
               rolled copy is carrying per-well amplitude, not row-level content, and is a wash.
               This is Rule 3 and it is not optional; it is what caught `[UNTRIED]` #19.
  perfold      the five outer-fold blendadds on the LEAK axis. A gain carried by one fold is a
               much weaker claim than a broad one.

KNOWN-WINNER ARM
----------------
Before any candidate is measured, the two deployed stacks must reproduce 8.2826 and 8.9641 to
0.002. If they do not, the port is wrong and nothing else is admissible, so the run stops.

ADMISSIBILITY
-------------
Ten banked bases are inadmissible and three more have partial well coverage despite full length.
The list is below with a reason per base and it is enforced, not advisory. A base not on the
773-well grid, or carrying non-finite values, is rejected.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ROOT / "data" / "features"
PREDICTIONS = ROOT / "predictions"
LEDGER = ROOT / "LEDGER.tsv"

ALPHA, W_STACK, N_FOLDS = 1.0, 0.76, 5
SG_WINDOW, SG_POLY = 17, 3
PROJ_DEGREE, PROJ_BLEND, PROJ_ITERS, PROJ_C = 4, 0.70, 4, 3.0

AXES = {
    "LEAK": (["lgbdivmed", "xgbdivmed", "geom_k16"], 8.2826),
    "NOGEOM": (["lgbdivmed", "xgbdivmed", "cbdivmed", "realmlp_v1_s42"], 8.7300),
}
PICK2 = ["lgbdivv1", "cb_div_s42", "xgb_div_s42"]
PICK2_RECORD = 8.9641

# Enforced, not advisory. Each entry is a base that must never enter a stack, and why.
INADMISSIBLE = {
    "tomcumsum_md_oracle": "selects its offset using TRUTH; blend delta -2.6700",
    "tomcumsum_row_oracle": "selects its offset using TRUTH; identical to the md variant",
    "dzleak_const": "self-declared leak probe",
    "dzleak_lin": "self-declared leak probe",
    "dzleak_lastN": "self-declared leak probe",
    "dzleak_recent": "self-declared leak probe",
    "dzleak_theil": "self-declared leak probe",
    "geom_k16_field_random6": "random-block CV protocol control, optimistic by construction",
    "virtual_prefix_future_ridge": "not a base: the median-trio BLEND plus a constant correction",
    "pf_selector": "already enters the blend at the fixed 0.24 weight",
    "viterbi_v1_smoke3": "partial coverage: nonzero on 3 of 773 wells",
    "viterbi_v1_confidence_smoke3": "partial coverage: nonzero on 3 of 773 wells",
    "realmlp_v1_s42_smoke": "partial coverage: nonzero on 155 of 773 wells",
    # 2026-07-28. The fold-0 screen ranked this first at REAL-NULL -0.4377 on NOGEOM, five
    # times any honest base we own. It is an ORACLE: emit_untried19_oof_2026_07_27.py:133
    # fits the window shape by lstsq on `tvw`, the TRUE TVT, and its own docstring calls the
    # arm "TRUE local shape, the channel ceiling, not deployable". Any `_e1_` arm is truth-fed.
    "untried19_e1_lam0.3": "arm e1 fits its shape on TRUE TVT; oracle, not a predictor",
    "untried19_e1_lam0.3_f0": "arm e1 fits its shape on TRUE TVT; oracle, not a predictor",
}
# Admissible on the LEAK axis only: consumes the geom prior, so it re-imports the leak into a
# no-geom stack. notes/experiment_notes/cnn_field.md:169,174; aws/src/train_cnn_sdf.py:131.
GEOM_FED = {"cnn_surface_twin", "geom_k16", "geom_k16_field_kmeans8",
            "geom_k16_field_typewell6", "geom_k16_field_ward6"}

LEDGER_COLS = ["base", "date", "standalone", "errcorr_leak", "errcorr_nogeom",
               "blendadd_leak", "null_leak", "blendadd_nogeom", "null_nogeom",
               "folds_improved_leak", "verdict", "note"]


class Data:
    """The 773-well OOF grid plus the deployed post-processing.

    Attributes:
        y: True TVT, full length.
        lk: Last-known TVT anchor, full length.
        yr: Residual target `y - lk`.
        groups: Well id per row.
        wells: Per-well row-index arrays, MD-sorted.
        pf: The PF residual that enters the blend at `1 - W_STACK`.
    """

    def __init__(self) -> None:
        """Load targets, meta and the PF, and build MD-sorted per-well index arrays."""
        tg = pd.read_parquet(FEATURES / "targets_train.parquet",
                             columns=["last_known_tvt", "target_tvt"])
        meta = pd.read_parquet(FEATURES / "meta_train.parquet",
                               columns=["well_id", "MD", "Z"])
        self.y = tg["target_tvt"].to_numpy(np.float64)
        self.lk = tg["last_known_tvt"].to_numpy(np.float64)
        self.yr = self.y - self.lk
        self.groups = meta["well_id"].to_numpy()
        md = meta["MD"].to_numpy(np.float64)
        z = meta["Z"].to_numpy(np.float64)
        order = np.argsort(self.groups, kind="stable")
        edges = np.flatnonzero(
            np.r_[True, self.groups[order][1:] != self.groups[order][:-1], True])
        self.wells = []
        for i in range(len(edges) - 1):
            loc = order[edges[i]:edges[i + 1]]
            self.wells.append(loc[np.argsort(md[loc], kind="stable")])
        self.md, self.z = md, z
        self.well_ids = [self.groups[w[0]] for w in self.wells]
        self._cache: dict[str, np.ndarray] = {}
        self.pf = self.col("pf_selector")

    def col(self, name: str) -> np.ndarray:
        """Load and cache a base OOF, validating shape and finiteness."""
        if name not in self._cache:
            path = PREDICTIONS / f"oof_{name}.npy"
            if not path.exists():
                raise FileNotFoundError(f"no such base: {path}")
            a = np.load(path).astype(np.float64)
            if a.shape != self.y.shape:
                raise ValueError(f"{name}: shape {a.shape}, expected {self.y.shape}")
            if not np.isfinite(a).all():
                raise ValueError(f"{name}: contains non-finite values")
            self._cache[name] = a
        return self._cache[name]

    def put(self, name: str, arr: np.ndarray) -> None:
        """Inject a synthetic column, used for the rolled null."""
        self._cache[name] = arr

    def roll(self, name: str) -> str:
        """Register a within-well-rolled copy of `name` and return its handle.

        The shift is `int.from_bytes(well_id.encode()) % n`. Never `abs(hash())`: Python's
        hash is salted per process, so an `abs(hash())` control is not reproducible.
        """
        src, out = self.col(name), np.empty(len(self.y))
        for loc, wid in zip(self.wells, self.well_ids):
            seg = src[loc]
            out[loc] = np.roll(seg, int.from_bytes(str(wid).encode(), "big") % max(len(seg), 1))
        handle = f"__roll__{name}"
        self.put(handle, out)
        return handle

    def postproc(self, stack: np.ndarray) -> np.ndarray:
        """Blend with the PF, savgol per well, then the anchored robust U-projection."""
        tvt = self.lk + W_STACK * stack + (1.0 - W_STACK) * self.pf
        out = tvt.copy()
        for loc in self.wells:
            n = len(loc)
            if n >= SG_WINDOW:
                out[loc] = savgol_filter(tvt[loc], SG_WINDOW, SG_POLY)
        for loc in self.wells:
            m, zz = self.md[loc], self.z[loc]
            s = (m - m[0]) / max(m[-1] - m[0], 1e-6)
            u = out[loc] + zz
            fit = _robust_polyfit(s, u)
            out[loc] = (1.0 - PROJ_BLEND) * out[loc] + PROJ_BLEND * (fit - zz)
        return out

    def run(self, bases: list[str]) -> np.ndarray:
        """Cross-fit the Ridge over the outer folds and post-process. The deployed protocol."""
        m = np.column_stack([self.col(b) for b in bases])
        stack = np.zeros(len(self.yr))
        for tr, va in GroupKFold(N_FOLDS).split(m, groups=self.groups):
            r = Ridge(alpha=ALPHA, positive=True)
            r.fit(m[tr], self.yr[tr])
            stack[va] = r.predict(m[va])
        return self.postproc(stack)

    def rmse(self, pred: np.ndarray, rows: np.ndarray | None = None) -> float:
        """Pooled RMSE, optionally restricted to `rows`."""
        e = pred - self.y if rows is None else pred[rows] - self.y[rows]
        return float(np.sqrt(np.mean(e ** 2)))


def _robust_polyfit(s: np.ndarray, y: np.ndarray) -> np.ndarray:
    """IRLS degree-4 polynomial fit, matching the deployed U-projection."""
    m = np.isfinite(s) & np.isfinite(y)
    if int(m.sum()) < PROJ_DEGREE + 2:
        return y.copy()
    coef = np.polyfit(s[m], y[m], PROJ_DEGREE)
    for _ in range(PROJ_ITERS):
        r = y[m] - np.polyval(coef, s[m])
        sc = np.median(np.abs(r)) * 1.4826 + 1e-9
        w = 1.0 / (1.0 + (r / (PROJ_C * sc)) ** 2)
        coef = np.polyfit(s[m], y[m], PROJ_DEGREE, w=w)
    return np.polyval(coef, s)


def gateable() -> list[str]:
    """Every full-length finite base on disk that is not inadmissible."""
    out = []
    for p in sorted(PREDICTIONS.glob("oof_*.npy")):
        n = p.stem[4:]
        if n not in INADMISSIBLE:
            out.append(n)
    return out


def append_ledger(row: dict) -> None:
    """Append one row to LEDGER.tsv, writing the header if the file is new."""
    new = not LEDGER.exists()
    with LEDGER.open("a") as fh:
        if new:
            fh.write("\t".join(LEDGER_COLS) + "\n")
        fh.write("\t".join(str(row.get(c, "")) for c in LEDGER_COLS) + "\n")


def main() -> int:  # noqa: C901
    """Gate the requested bases and append their ledger rows."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", action="append", default=[])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    if args.list:
        for n in gateable():
            print(n)
        return 0
    if not args.base:
        ap.error("give --base NAME (repeatable) or --list")

    d = Data()
    print(f"{len(d.y)} rows, {len(d.wells)} wells\n")

    print("KNOWN-WINNER, the deployed stacks must reproduce their record:")
    ok = True
    refs = {}
    for axis, (bases, record) in AXES.items():
        pred = d.run(bases)
        got = d.rmse(pred)
        good = abs(got - record) <= 0.002
        ok &= good
        refs[axis] = (pred, got)
        print(f"  {axis:7s} record {record:.4f}  got {got:.4f}  "
              f"{'PASS' if good else '*** FAIL ***'}")
    p2 = d.rmse(d.run(PICK2))
    good = abs(p2 - PICK2_RECORD) <= 0.002
    ok &= good
    print(f"  {'PICK2':7s} record {PICK2_RECORD:.4f}  got {p2:.4f}  "
          f"{'PASS' if good else '*** FAIL ***'}")
    if not ok:
        print("\n*** the port is wrong; nothing below would be admissible. Stopping. ***")
        return 1
    print()

    fold_of_row = np.empty(len(d.y), int)
    for f, (_, va) in enumerate(GroupKFold(N_FOLDS).split(np.zeros((len(d.y), 1)),
                                                          groups=d.groups)):
        fold_of_row[va] = f

    for name in args.base:
        print("=" * 78)
        print(f"BASE {name}")
        print("=" * 78)
        if name in INADMISSIBLE:
            print(f"  INADMISSIBLE: {INADMISSIBLE[name]}")
            append_ledger(dict(base=name, date=date.today().isoformat(),
                               verdict="INADMISSIBLE", note=INADMISSIBLE[name]))
            continue
        try:
            base_err = d.lk + d.col(name) - d.y
        except (FileNotFoundError, ValueError) as exc:
            print(f"  REJECTED: {exc}")
            append_ledger(dict(base=name, date=date.today().isoformat(),
                               verdict="REJECTED", note=str(exc)))
            continue

        row = dict(base=name, date=date.today().isoformat(), note=args.note)
        row["standalone"] = f"{float(np.sqrt(np.mean(base_err ** 2))):.4f}"
        rolled = d.roll(name)
        print(f"  standalone {row['standalone']}  (recorded, never gated on)")

        for axis, (bases, _) in AXES.items():
            if axis == "NOGEOM" and name in GEOM_FED:
                print(f"  {axis}: SKIPPED, this base consumes the geom prior")
                row[f"blendadd_{axis.lower()}"] = "n/a"
                row[f"null_{axis.lower()}"] = "n/a"
                row[f"errcorr_{axis.lower()}"] = "n/a"
                continue
            ref_pred, ref = refs[axis]
            ec = float(np.corrcoef(ref_pred - d.y, base_err)[0, 1])
            add = d.rmse(d.run(bases + [name])) - ref
            null = d.rmse(d.run(bases + [rolled])) - ref
            row[f"errcorr_{axis.lower()}"] = f"{ec:.3f}"
            row[f"blendadd_{axis.lower()}"] = f"{add:+.4f}"
            row[f"null_{axis.lower()}"] = f"{null:+.4f}"
            print(f"  {axis:7s} err-corr {ec:6.3f}   blend-add {add:+.4f}   "
                  f"rolled null {null:+.4f}   REAL-NULL {add - null:+.4f}")
            if axis == "LEAK":
                pc = d.run(bases + [name])
                imp = sum(1 for f in range(N_FOLDS)
                          if d.rmse(pc, np.flatnonzero(fold_of_row == f))
                          < d.rmse(ref_pred, np.flatnonzero(fold_of_row == f)))
                row["folds_improved_leak"] = f"{imp}/5"
                print(f"  {'':7s} folds improved {imp}/5")

        adds = [float(row.get(f"blendadd_{a.lower()}", "0") or 0)
                for a in AXES if row.get(f"blendadd_{a.lower()}", "n/a") != "n/a"]
        nulls = [float(row.get(f"null_{a.lower()}", "0") or 0)
                 for a in AXES if row.get(f"null_{a.lower()}", "n/a") != "n/a"]
        margin = min(a - n for a, n in zip(adds, nulls)) if adds else 0.0
        row["verdict"] = ("KEEP" if margin <= -0.05 else
                          "MARGINAL" if margin <= -0.01 else "WASH")
        print(f"  VERDICT {row['verdict']}   (best REAL-NULL margin {margin:+.4f}; "
              f"KEEP needs <= -0.05, MARGINAL <= -0.01)")
        append_ledger(row)
        print()

    print(f"appended to {LEDGER.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
