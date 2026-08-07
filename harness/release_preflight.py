"""Validate a public ROGII harness checkout before measuring a candidate.

The release excludes data and generated artifact banks. This preflight reports
missing required inputs without crashing. Once those inputs are present, it
runs the three known winner checks implemented by ``harness.gate``.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FEATURES = ROOT / "data" / "features"
PREDICTIONS = ROOT / "predictions"
RAW_TRAIN = ROOT / "data" / "raw" / "train"
REQUIRED_PACKAGES = ("numpy", "pandas", "scipy", "sklearn")
KNOWN_WINNERS = {
    "LEAK": 8.2826,
    "NOGEOM": 8.7300,
    "PICK2": 8.9641,
}


def missing_artifacts() -> list[Path]:
    """Return required artifact paths that are not available.

    Returns:
        Missing files or directories needed to reproduce the known winner
        checks.
    """
    required = (
        FEATURES / "targets_train.parquet",
        FEATURES / "meta_train.parquet",
        PREDICTIONS / "oof_pf_selector.npy",
        RAW_TRAIN,
    )
    missing = [path for path in required if not path.exists()]
    if len(list(PREDICTIONS.glob("oof_*.npy"))) <= 50:
        missing.append(PREDICTIONS / "oof_*.npy (more than 50 required)")
    return missing


def check_environment() -> bool:
    """Check that the Python interpreter can import required packages.

    Returns:
        True if every package is importable and Python is at least version 3.11.
    """
    ok = sys.version_info >= (3, 11)
    print(f"python >= 3.11: {'PASS' if ok else 'FAIL'} ({sys.version.split()[0]})")
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
            print(f"import {package}: PASS")
        except ImportError as error:
            print(f"import {package}: FAIL ({error})")
            ok = False
    return ok


def check_known_winners() -> bool:
    """Reproduce the three recorded cross validation scores.

    Returns:
        True when every known winner score is within the release tolerance.
    """
    from harness.gate import AXES, PICK2, Data

    data = Data()
    stacks = {
        "LEAK": AXES["LEAK"][0],
        "NOGEOM": AXES["NOGEOM"][0],
        "PICK2": PICK2,
    }
    ok = True
    for name, bases in stacks.items():
        observed = data.rmse(data.run(bases))
        expected = KNOWN_WINNERS[name]
        passed = abs(observed - expected) <= 0.002
        print(
            f"{name}: {'PASS' if passed else 'FAIL'} "
            f"expected {expected:.4f}, observed {observed:.4f}"
        )
        ok = ok and passed
    return ok


def main() -> int:
    """Run the release preflight.

    Returns:
        Zero if the requested checks pass, otherwise one.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Validate the environment and artifact layout without scoring known winners.",
    )
    args = parser.parse_args()

    print(f"release preflight: {ROOT}")
    environment_ok = check_environment()
    missing = missing_artifacts()
    if missing:
        print("missing private competition artifacts:")
        for path in missing:
            print(f"  {path.relative_to(ROOT)}")
        return 1
    if args.quick:
        print("artifact layout: PASS")
        return 0 if environment_ok else 1
    winners_ok = check_known_winners()
    return 0 if environment_ok and winners_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
