# %% [markdown]
# # ROGII Wellbore Geology Prediction. PF multi divergence stack with U projection
#
# Thin submission notebook. All heavy training happened locally / on AWS.
# This notebook only does inference.
#
# 1. Discover test wells.
# 2. Build the v6 live feature frame for the requested rows.
# 3. Add live estimator-divergence columns.
# 4. Average each fitted list for lgbdivmed and xgbdivmed; predict the realmlp NN residual.
# 5. Ridge-combine the three residuals, blend with PF, project in anchored U space, and write `submission.csv`.
#
# Inputs.
#
# - Competition data at `/kaggle/input/rogii-wellbore-geology-prediction/`.
# - Bundled artifacts at `/kaggle/input/rogii-models-v6/`. Contains
#   `rogii_features.py`.
# - Divergence stack artifacts at `/kaggle/input/rogii-div-models-v1/`.
#   Contains `models_*.pkl`, `feature_names_*.json`, `divergence.py`, and
#   `ridge_meta_3div.json`.
# - RealMLP artifacts at `/kaggle/input/rogii-realmlp-models/` (model,
#   feature names, preprocess stats) and offline wheels at
#   `/kaggle/input/rogii-realmlp-wheels/` (pytabkit + lightning deps).

# %% [markdown]
# ## 1. Imports and configuration

# %%
from __future__ import annotations

import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path

for var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(var, "2")

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 200)

# rogii_features uses np.random.normal/uniform directly inside the particle
# filter. Without a fixed seed the test-time feature build is non-deterministic
# across submissions, which inflated v10's LB spread to 0.534. Pin the seed
# here so every Kaggle re-run produces the same submission.csv.
np.random.seed(42)

# Set FAST_DEBUG=True to smoke test on a few test wells locally.
FAST_DEBUG = False
MAX_TEST_WELLS = 3 if FAST_DEBUG else None

# Divergence stack artifacts. Each model has its own feature order and fitted
# model list. Ridge metadata stores the residual-space meta coefficients.
MODEL_NAMES = ["lgbdivmed", "xgbdivmed"]
RIDGE_META_NAME = "ridge_meta_3div.json"
RATECOUPLED_MODEL_NAME = "models_lgbmede2ratecoupledfeature_2026_08_01.pkl"
RATECOUPLED_FEATURE_NAME = "feature_names_lgbmede2ratecoupledfeature_2026_08_01.json"
DIV_MODEL_MARKERS = [f"models_{name}.pkl" for name in MODEL_NAMES]
DIV_FEATURE_MARKERS = [f"feature_names_{name}.json" for name in MODEL_NAMES]
DIV_MODEL_MARKER = DIV_MODEL_MARKERS[0]
APPLY_SG_SMOOTHING = True
SG_WINDOW = 17
SG_POLY = 3

# Private-safe projection borrowed from public notebook lineage, but tuned on our
# own full 773-well OOF. It smooths U = TVT + Z - anchor per well, then blends
# back into the stack/PF prediction. Expanded OOF tuning improved the current stack from 9.1444 to 8.9490
# with W_STACK 0.88, degree 6, projection blend 0.78, and robust C 3.0. Retuned 2026-08-04:
# the degree-4 blend-0.70 projection was mis-conditioning the path for the trust-datum
# estimator, whose worth rises -0.1079 -> -0.1678 and whose per-well datum correlation rises
# 0.2298 -> 0.2777. Ramped CV 7.51276 -> 7.37230, paired well bootstrap CI95
# [-0.2610, -0.0125], p_worse 0.0138, 5 of 5 GroupKFold folds.
APPLY_U_PROJECTION = True
U_PROJECTION_DEGREE = 6
U_PROJECTION_BLEND = 0.78
U_PROJECTION_ROBUST_ITERS = 4
U_PROJECTION_ROBUST_C = 3.0

print(f"FAST_DEBUG: {FAST_DEBUG}")
print(f"MAX_TEST_WELLS: {MAX_TEST_WELLS}")
print(f"MODEL_NAMES: {MODEL_NAMES}")
print(f"RIDGE_META_NAME: {RIDGE_META_NAME}")
print(f"APPLY_SG_SMOOTHING: {APPLY_SG_SMOOTHING} window={SG_WINDOW} poly={SG_POLY}")
print(
    "U projection: "
    f"enabled={APPLY_U_PROJECTION} degree={U_PROJECTION_DEGREE} "
    f"blend={U_PROJECTION_BLEND} robust_iters={U_PROJECTION_ROBUST_ITERS} "
    f"robust_c={U_PROJECTION_ROBUST_C}"
)

# %% [markdown]
# ## 2. Locate competition data and bundled artifacts

# %%
def find_data_root() -> Path:
    """Locate the competition data root on Kaggle or locally."""
    candidates = [
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path.cwd() / "data" / "raw",
        Path.cwd().parent / "data" / "raw",
        Path.cwd(),
    ]
    candidates.extend(Path.cwd().parents)
    for root in candidates:
        if (root / "test").is_dir() and (root / "sample_submission.csv").is_file():
            return root.resolve()
    raise FileNotFoundError("Could not locate competition data root.")


def find_artifacts_root() -> Path:
    """Locate the bundled v6 feature code artifacts dataset."""
    candidates = [
        Path("/kaggle/input/rogii-models-v6"),
        Path("/kaggle/input/datasets/wguesdon/rogii-models-v6"),
        Path.cwd() / "kaggle_datasets" / "rogii-models-v6",
        Path.cwd().parent / "kaggle_datasets" / "rogii-models-v6",
    ]
    marker = "rogii_features.py"
    for root in candidates:
        if (root / marker).is_file():
            return root.resolve()
    kaggle_root = Path("/kaggle/input")
    if kaggle_root.is_dir():
        for match in kaggle_root.rglob(marker):
            return match.parent.resolve()
    raise FileNotFoundError(f"Could not locate {marker} in /kaggle/input.")


def find_div_model_root() -> Path:
    """Locate the dataset that holds the divergence GBDT artifacts."""
    candidates = [
        Path("/kaggle/input/rogii-div-models-v1"),
        Path("/kaggle/input/datasets/wguesdon/rogii-div-models-v1"),
        Path.cwd() / "kaggle_datasets" / "rogii-div-models-v1",
        Path.cwd().parent / "kaggle_datasets" / "rogii-div-models-v1",
    ]
    required = [*DIV_MODEL_MARKERS, *DIV_FEATURE_MARKERS, RIDGE_META_NAME, "divergence.py"]
    for root in candidates:
        if all((root / name).is_file() for name in required):
            return root.resolve()
    kaggle_root = Path("/kaggle/input")
    if kaggle_root.is_dir():
        for match in kaggle_root.rglob(RIDGE_META_NAME):
            root = match.parent
            if all((root / name).is_file() for name in required):
                return root.resolve()
    missing_msg = ", ".join(required)
    raise FileNotFoundError(f"Could not locate complete divergence artifact set: {missing_msg}.")



def find_ratecoupled_model_root() -> Path:
    """Locate the private artifact dataset for the rate coupled LightGBM."""
    candidates = [
        Path("/kaggle/input/rogii-ratecoupled-gbdt"),
        Path("/kaggle/input/datasets/wguesdon/rogii-ratecoupled-gbdt"),
        Path.cwd() / "kaggle_datasets" / "rogii-ratecoupled-gbdt",
        Path.cwd().parent / "kaggle_datasets" / "rogii-ratecoupled-gbdt",
    ]
    for root in candidates:
        if (root / RATECOUPLED_MODEL_NAME).is_file() and (root / RATECOUPLED_FEATURE_NAME).is_file():
            return root.resolve()
    kaggle_root = Path("/kaggle/input")
    if kaggle_root.is_dir():
        for match in kaggle_root.rglob(RATECOUPLED_MODEL_NAME):
            if (match.parent / RATECOUPLED_FEATURE_NAME).is_file():
                return match.parent.resolve()
    raise FileNotFoundError("Could not locate the complete rate coupled LightGBM artifact set.")


# --- realmlp (RealMLP-TD NN base) live inference ---------------------------
# realmlp is a genuinely different family (pytabkit RealMLP-TD) trained on the
# base 137-feature set (a subset of the divergence feature order) predicting the
# residual to last_known_tvt. Adding it to the Ridge meta improved the full
# 773-well projected OOF from 8.9490 to 8.8144 (deployed-consistent) and from
# 8.9680 to 8.8492 (honest cross-fit). Artifacts live in rogii-realmlp-models;
# pytabkit installs offline from rogii-realmlp-wheels.
REALMLP_MODEL_NAME = "realmlp_v1_s42_all.pkl"
REALMLP_FEATURES_NAME = "realmlp_feature_names.json"
REALMLP_PREPROCESS_NAME = "realmlp_preprocess_np.npz"


def find_realmlp_root() -> Path:
    """Locate the dataset holding the realmlp model artifacts."""
    candidates = [
        Path("/kaggle/input/rogii-realmlp-models"),
        Path("/kaggle/input/datasets/wguesdon/rogii-realmlp-models"),
        Path.cwd() / "kaggle_datasets" / "realmlp_models",
        Path.cwd().parent / "kaggle_datasets" / "realmlp_models",
    ]
    for root in candidates:
        if (root / REALMLP_MODEL_NAME).is_file():
            return root.resolve()
    kaggle_root = Path("/kaggle/input")
    if kaggle_root.is_dir():
        for match in kaggle_root.rglob(REALMLP_MODEL_NAME):
            return match.parent.resolve()
    raise FileNotFoundError(f"Could not locate {REALMLP_MODEL_NAME}.")


# --- CNN-1D whole-well head live inference -------------------------------
# The one base in the repo that DECORRELATES: err-corr 0.574 against this blend where every
# GBDT sits at 0.90-0.96, and the only KEEP in LEDGER.tsv. Adding it moves the deployed
# protocol 8.7300 -> 8.6107 with both halves of the selection-bias guard agreeing.
#
# torch is already present in this image; the realmlp branch below installs pytabkit, which
# is built on it. What was NOT free is the inference code, and it has two traps that produce
# a kernel that RUNS and is silently wrong. Both are measured, see
# scripts/verify_cnn1d_inference_2026_07_28.py, which rebuilds fold 0 from the checkpoints
# using ONLY the six columns a hidden test well exposes and diffs against the banked OOF:
#   1. drift_mode must be `resid`. It changes the reconstruction, not any parameter shape, so
#      a strict=True load succeeds either way and the only symptom is a 95 ft mean error.
#   2. the shipped dataset.py must be the post-2026-07-28 one. The earlier version filled
#      eval_mask only when a TVT label was present, and Seq1DNet takes eval_mask as an INPUT,
#      so on a test well the model emitted pure carry-forward in both drift modes.
CNN1D_PKG_MARKER = "cnn_1d/kernel_infer.py"


def find_cnn1d_root() -> Path:
    """Locate the dataset holding the cnn_1d package and its fold checkpoints."""
    candidates = [
        Path("/kaggle/input/rogii-cnn1d-models"),
        Path("/kaggle/input/datasets/wguesdon/rogii-cnn1d-models"),
        Path.cwd() / "kaggle_datasets" / "rogii-cnn1d-models",
        Path.cwd().parent / "kaggle_datasets" / "rogii-cnn1d-models",
    ]
    for root in candidates:
        if (root / CNN1D_PKG_MARKER).is_file():
            return root.resolve()
    kaggle_root = Path("/kaggle/input")
    if kaggle_root.is_dir():
        for match in kaggle_root.rglob(CNN1D_PKG_MARKER):
            return match.parent.parent.resolve()
    raise FileNotFoundError(f"Could not locate {CNN1D_PKG_MARKER}.")


def load_cnn1d() -> tuple[object, list, object]:
    """Import the shipped cnn_1d package and load every fold checkpoint.

    Returns:
        Tuple ``(config, fold_models, predict_well_drift)``.
    """
    root = find_cnn1d_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from cnn_1d.dataset import Seq1DConfig
    from cnn_1d.kernel_infer import load_fold_models, predict_well_drift

    cfg = Seq1DConfig()
    models = load_fold_models(root, n_features=cfg.n_features)
    print(f"[cnn1d] root {root}; {len(models)} fold models; seq_len {cfg.seq_len}")
    return cfg, models, predict_well_drift


def ensure_pytabkit() -> None:
    """Import pytabkit, installing it offline from a bundled wheels dataset."""
    try:
        import pytabkit  # noqa: F401
        return
    except ImportError:
        pass
    import glob
    import subprocess

    hits = glob.glob("/kaggle/input/**/pytabkit*.whl", recursive=True)
    if not hits:
        raise FileNotFoundError("pytabkit wheel not found under /kaggle/input.")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--no-index",
        "--find-links", str(Path(hits[0]).parent), "pytabkit",
    ])
    import pytabkit  # noqa: F401


def predict_realmlp_residual(frame: pd.DataFrame) -> np.ndarray:
    """Return realmlp residual-to-last_known predictions for the live frame.

    Loads the refit-all RealMLP-TD model with its saved feature order and
    impute/standardize statistics, applies the identical preprocessing used in
    training (mean-impute then standardize), and predicts. The 137 features are a
    subset of the divergence feature set, so they already exist on ``frame``.

    Args:
        frame: Live feature frame containing all realmlp feature columns.

    Returns:
        Residual predictions aligned to ``frame`` rows.
    """
    ensure_pytabkit()
    root = find_realmlp_root()
    feature_names = json.loads((root / REALMLP_FEATURES_NAME).read_text())
    pp = np.load(root / REALMLP_PREPROCESS_NAME)
    fill = pp["fill"].astype(np.float64)
    mean = pp["mean"].astype(np.float64)
    scale = pp["scale"].astype(np.float64)
    missing = [c for c in feature_names if c not in frame.columns]
    if missing:
        raise KeyError(f"realmlp features missing from live frame: {missing[:8]}")
    matrix = frame.loc[:, feature_names].to_numpy(dtype=np.float64, copy=True)
    matrix = np.where(np.isfinite(matrix), matrix, fill)  # mean-impute NaN
    matrix = ((matrix - mean) / scale).astype(np.float32)  # standardize
    model = joblib.load(root / REALMLP_MODEL_NAME)
    pred = np.asarray(model.predict(matrix), dtype=np.float64).reshape(-1)
    print(f"[realmlp] residual mean {pred.mean():+.4f} std {pred.std():.4f}")
    return pred


if Path("/kaggle/input").is_dir():
    print("Inventory of /kaggle/input:")
    for child in sorted(Path("/kaggle/input").iterdir()):
        print(f"  {child}")
        if child.is_dir():
            for sub in sorted(child.iterdir())[:3]:
                print(f"    -> {sub.name}")

DATA_ROOT = find_data_root()
TEST_DIR = DATA_ROOT / "test"
TRAIN_DIR = DATA_ROOT / "train"
SAMPLE_SUB_PATH = DATA_ROOT / "sample_submission.csv"
ARTIFACTS_ROOT = find_artifacts_root()
DIV_ARTIFACTS_ROOT = find_div_model_root()
RATECOUPLED_ARTIFACTS_ROOT = find_ratecoupled_model_root()
print(f"DATA_ROOT:          {DATA_ROOT}")
print(f"TEST_DIR:           {TEST_DIR}")
print(f"ARTIFACTS_ROOT:     {ARTIFACTS_ROOT}")
print(f"DIV_ARTIFACTS_ROOT: {DIV_ARTIFACTS_ROOT}")

# Import rogii_features from the artifacts dataset so we share the exact
# code that produced the training features. Import divergence from the new
# model dataset when present.
sys.path.insert(0, str(DIV_ARTIFACTS_ROOT))
sys.path.insert(0, str(ARTIFACTS_ROOT))
import rogii_features as rf  # noqa: E402
try:
    from divergence import add_divergence_columns, DIVERGENCE_COLUMNS  # noqa: E402
except Exception:
    import itertools

    ESTIMATOR_COLUMNS = (
        "pf_selector", "pf_pf12_beam2", "pf_beam_oof", "pf_median",
        "pf_scale5", "pf_scale8", "pf_ancc", "pf_z", "plane_buda",
        "plane_egfdl", "dense50", "dense", "beam_mean", "beam_cons",
    )
    DIVERGENCE_COLUMNS = tuple(
        f"div_{left}__minus__{right}"
        for left, right in itertools.combinations(ESTIMATOR_COLUMNS, 2)
    ) + (
        "div_spread_std", "div_spread_range", "div_spread_mad",
        "div_spread_max", "div_spread_min",
    )

    def add_divergence_columns(est_df: pd.DataFrame) -> pd.DataFrame:
        """Build divergence columns when the shared module is unavailable."""
        missing = [c for c in ESTIMATOR_COLUMNS if c not in est_df.columns]
        if missing:
            raise KeyError(f"Missing estimator columns for divergence: {missing}")
        estimates = {
            name: np.nan_to_num(est_df[name].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            for name in ESTIMATOR_COLUMNS
        }
        matrix = np.column_stack([estimates[name] for name in ESTIMATOR_COLUMNS]).astype(np.float32)
        out = {}
        for left, right in itertools.combinations(ESTIMATOR_COLUMNS, 2):
            out[f"div_{left}__minus__{right}"] = (estimates[left] - estimates[right]).astype(np.float32)
        est_max = matrix.max(axis=1)
        est_min = matrix.min(axis=1)
        est_mean = matrix.mean(axis=1)
        out["div_spread_std"] = matrix.std(axis=1).astype(np.float32)
        out["div_spread_range"] = (est_max - est_min).astype(np.float32)
        out["div_spread_mad"] = np.mean(np.abs(matrix - est_mean[:, None]), axis=1).astype(np.float32)
        out["div_spread_max"] = est_max.astype(np.float32)
        out["div_spread_min"] = est_min.astype(np.float32)
        return pd.DataFrame(out, columns=list(DIVERGENCE_COLUMNS), index=est_df.index)

print(f"rogii_features version: {getattr(rf, '__version__', 'untagged')}")

# %% [markdown]
# ## 3. Identify required test rows and model artifact

# %%
print(f"Divergence stack models: {MODEL_NAMES}")
for marker in DIV_MODEL_MARKERS:
    print(f"Model artifact: {DIV_ARTIFACTS_ROOT / marker}")
print(f"Ridge meta artifact: {DIV_ARTIFACTS_ROOT / RIDGE_META_NAME}")

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
sample_sub["well_id"] = sample_sub["id"].str.rsplit("_", n=1).str[0]
sample_sub["row_index"] = sample_sub["id"].str.rsplit("_", n=1).str[1].astype(int)
test_row_map = rf.make_test_row_map(sample_sub)
print(f"Sample submission rows: {len(sample_sub)}")

# %% [markdown]
# ## 4. Fit FormationPlaneKNN and RowKNN on all training wells

# %%
t0 = time.time()
train_paths = sorted(TRAIN_DIR.glob("*__horizontal_well.csv"))
print(f"Training wells: {len(train_paths)}")
formation_imputer = rf.FormationPlaneKNN(train_paths)
row_imputer = rf.RowKNN(train_paths)
print(f"Imputers ready in {time.time() - t0:.1f}s "
      f"(centroids={len(formation_imputer.df)}, rows={len(row_imputer.ancc)})")

# %% [markdown]
# ## 5. Build test features per well

# %%
test_paths = sorted(TEST_DIR.glob("*__horizontal_well.csv"))
if MAX_TEST_WELLS is not None:
    test_paths = test_paths[:MAX_TEST_WELLS]
print(f"Test wells to score: {len(test_paths)}")

t0 = time.time()
parts: list[pd.DataFrame] = []
for i, path in enumerate(test_paths, start=1):
    # Re-seed per well so each well's particle filter sees identical
    # entropy regardless of preceding wells. Robust to glob ordering.
    np.random.seed(42)
    part = rf.build_features_for_well(
        path,
        split="test",
        formation_imputer=formation_imputer,
        row_imputer=row_imputer,
        test_row_map=test_row_map,
    )
    if not part.empty:
        parts.append(part)
    if i % 5 == 0:
        print(f"  {i}/{len(test_paths)} wells done ({time.time() - t0:.0f}s)")

test_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
print(f"\nTest feature frame: {test_df.shape} ({time.time() - t0:.0f}s)")

if test_df.empty:
    raise RuntimeError("Test feature build produced an empty frame.")

# %% [markdown]
# ## 6. Prepare feature selection helper

# %%
groups = rf.classify_columns(test_df.columns.tolist())
META_COLS = groups["meta"] + (["split"] if "split" in test_df.columns else [])
TARGET_COLS = groups["targets"]


def select_feature_columns(feature_df: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    """Return a sub-frame ordered exactly like the training feature columns."""
    missing = [c for c in expected if c not in feature_df.columns]
    if missing:
        raise KeyError(f"Test frame missing {len(missing)} features: {missing[:8]}...")
    return feature_df[expected].astype(np.float32)


baseline_test = test_df["baseline_tvt"].astype(np.float32).to_numpy()
n_test = len(test_df)
print(f"Test rows: {n_test}, baseline range [{baseline_test.min():.1f}, {baseline_test.max():.1f}]")


def robust_polyfit_predict(
    s: np.ndarray,
    y: np.ndarray,
    degree: int = 4,
    robust_iters: int = 4,
    robust_c: float = 2.0,
) -> np.ndarray:
    """Fit a robust polynomial and predict all input positions.

    Args:
        s: Normalized measured depth coordinates.
        y: Values to smooth in formation relative U space.
        degree: Polynomial degree.
        robust_iters: Number of iteratively reweighted least squares rounds.
        robust_c: Residual scale multiplier for robust weights.

    Returns:
        Predicted values at every input coordinate.
    """
    s = np.asarray(s, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(s) & np.isfinite(y)
    if int(mask.sum()) < int(degree) + 2:
        return y.copy()
    degree = int(min(int(degree), max(1, int(mask.sum()) - 2)))
    coef = np.polyfit(s[mask], y[mask], degree)
    for _ in range(int(robust_iters)):
        residual = y[mask] - np.polyval(coef, s[mask])
        scale = np.median(np.abs(residual)) * 1.4826 + 1e-6
        weights = 1.0 / (1.0 + (residual / (float(robust_c) * scale)) ** 2)
        coef = np.polyfit(s[mask], y[mask], degree, w=weights)
    pred = np.asarray(np.polyval(coef, s), dtype=np.float64)
    pred[~np.isfinite(pred)] = y[~np.isfinite(pred)]
    return pred


def project_prediction_by_well(
    prediction: np.ndarray,
    frame: pd.DataFrame,
    test_dir: Path,
    degree: int = 4,
    blend_weight: float = 0.75,
    robust_iters: int = 4,
    robust_c: float = 2.0,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Project per well predictions in anchored U space.

    U is defined as TVT plus Z minus the visible anchor. The projection is a
    low order robust polynomial by well. This keeps the hidden tail smooth in a
    geometry aware coordinate without using any public leaderboard overlay.

    Args:
        prediction: TVT predictions aligned to frame rows.
        frame: Test feature frame with well_id, row_index, MD, and Z columns.
        test_dir: Directory containing active test well CSVs.
        degree: Polynomial degree for the U smoother.
        blend_weight: Weight on the projected prediction.
        robust_iters: Number of robust fitting rounds.
        robust_c: Residual scale multiplier for robust weights.

    Returns:
        A tuple with projected predictions and a per well summary frame.
    """
    projected = np.asarray(prediction, dtype=np.float64).copy()
    rows: list[dict[str, object]] = []
    blend = float(np.clip(float(blend_weight), 0.0, 1.0))
    for wid, group_idx in frame.groupby("well_id", sort=False).groups.items():
        loc = np.asarray(list(group_idx), dtype=int)
        ordered = frame.iloc[loc].sort_values("MD")
        ordered_loc = ordered.index.to_numpy(dtype=int)
        try:
            hw_path = test_dir / f"{wid}__horizontal_well.csv"
            hw = pd.read_csv(hw_path, usecols=["TVT_input", "MD", "Z"])
            known = hw[hw["TVT_input"].notna()]
            if len(known) < 5:
                rows.append({"well_id": wid, "projected": False, "reason": "too_few_known_rows"})
                continue
            last_row = known.iloc[-1]
            anchor = float(last_row["TVT_input"]) + float(last_row["Z"])
            start_md = float(last_row["MD"])
            end_md = float(hw["MD"].iloc[-1])
            denom = max(end_md - start_md, 1e-6)
            s = (ordered["MD"].to_numpy(dtype=np.float64) - start_md) / denom
            z = ordered["Z"].to_numpy(dtype=np.float64)
            tvt = projected[ordered_loc].copy()
            u = tvt + z - anchor
            u_fit = robust_polyfit_predict(
                s,
                u,
                degree=degree,
                robust_iters=robust_iters,
                robust_c=robust_c,
            )
            tvt_projected = anchor + u_fit - z
            tvt_fit = (1.0 - blend) * tvt + blend * tvt_projected
            if not np.all(np.isfinite(tvt_fit)):
                rows.append({"well_id": wid, "projected": False, "reason": "non_finite"})
                continue
            adjustment = tvt_fit - tvt
            projected[ordered_loc] = tvt_fit
            rows.append({
                "well_id": wid,
                "projected": True,
                "reason": "ok",
                "rows": int(len(ordered_loc)),
                "mean_abs_adjustment": float(np.mean(np.abs(adjustment))),
                "max_abs_adjustment": float(np.max(np.abs(adjustment))),
            })
        except Exception as exc:
            rows.append({"well_id": wid, "projected": False, "reason": str(exc)[:160]})
    return projected.astype(np.float32), pd.DataFrame(rows)


# %% [markdown]
# ## 6b. Embedded strong PF (from src/pf_frontier.py)

# %%
"""Faithful reproduction of the public LB 8.860 PF / beam / selector frontier.

Ported from the public Kaggle notebook
``needless090/lb-8-860-rogii-sel15-256seeds`` (CleAAAAAA / needless090),
the strongest tree-tier physical model on the ROGII Wellbore Geology
Prediction leaderboard as of 2026-06-04.

The hidden test wells expose only ``[MD, X, Y, Z, GR, TVT_input]`` (the
formation-top columns and the ``TVT`` target are train-only), so the only
usable signal on scored wells is GR-vs-typewell tracking plus geometry.
This module treats the particle filter as a Monte-Carlo search: many seeds
are run, then weighted by total GR log-likelihood, and a per-well selector
chooses a likelihood temperature / beam / hold blend from whole-well
characteristics (n_eval, eval Z-span).

Provenance note: the numeric constants and the selector map are taken
verbatim from the public notebook; they were tuned on its author's CV.
"""


import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# --- selector constants (verbatim from the public notebook) --------------
SELECTOR_N_EVAL_THRESHOLD = 4840.0
SELECTOR_Z_SPAN_THRESHOLDS = (136.73000000000016, 185.5133333333342)
SELECTOR_BIN_VARIANTS = {
    0: "pf_scale_5_hold_0.2",
    1: "pf_scale_3_hold_0.15",
    2: "pf_scale_12_beam_0.2_hold_0.15",
    3: "pf_scale_5_hold_0.15",
    4: "pf_scale_5_beam_0.05_hold_0.05",
    5: "pf_scale_12_beam_0.2_hold_0.05",
}
SELECTOR_GLOBAL_VARIANT = "pf_scale_8_hold_0.2"
SELECTOR_SCALES = (3.0, 5.0, 8.0, 12.0)

# 14 beam configs: (beam_size, motion_cost, error_scale, savgol_radius)
BEAM_CONFIGS = [
    (10, 20.0, 144.0, 2),
    (10, 8.0, 64.0, 2),
    (8, 35.0, 220.0, 1),
    (10, 14.0, 90.0, 5),
    (20, 4.0, 36.0, 3),
    (12, 12.0, 100.0, 3),
    (15, 25.0, 180.0, 2),
    (20, 30.0, 200.0, 2),
    (15, 10.0, 80.0, 4),
    (25, 6.0, 50.0, 3),
    (10, 40.0, 300.0, 1),
    (12, 18.0, 120.0, 5),
    (30, 8.0, 70.0, 2),
    (10, 50.0, 400.0, 0),
]



# Embedded E2 rate coupled source SHA256: a45ee8b947fda42a88751c89f7f6b8fc3ffdf9102b4c17698bceb7371d195af4
E2_GRID_HALF = 100.0
E2_GRID_STEP = 0.5
E2_RATE_PER_100FT = 8.0
E2_WINDOWS = (400.0, 800.0)
E2_STRIDE_FRAC = 0.5
E2_MIN_WIN_ROWS = 50
E2_ROW_SUB = 2
E2_GR_DECORR_FT = 33.0
E2_MIN_EMIT_NODES = 3
E2_SMOOTH_RADIUS = 2
E2_LAM = 0.01
E2_WIDE_SLOPES = np.linspace(-0.06, 0.06, 13)
E2_WIDE_GAINS = np.linspace(-0.2, 2.0, 7)


def e2_smooth_gr(values: np.ndarray) -> np.ndarray:
    """Apply the predeclared five sample quadratic GR smoother.

    Args:
        values: Finite GR samples in their native path order.

    Returns:
        Smoothed samples, or the original samples when the path is too short.
    """
    window = 2 * E2_SMOOTH_RADIUS + 1
    if len(values) < window + 2:
        return values
    return savgol_filter(values, window, 2)


def e2_iter_windows(md: np.ndarray):
    """Yield admissible local matching windows.

    Args:
        md: Ascending eval measured depth samples.

    Yields:
        Window centroid, membership mask, and width in feet.
    """
    for width in E2_WINDOWS:
        if md[-1] - md[0] < width:
            continue
        for start in np.arange(md[0], md[-1] - width + 1e-9, width * E2_STRIDE_FRAC):
            selected = (md >= start) & (md < start + width)
            if int(selected.sum()) < E2_MIN_WIN_ROWS:
                continue
            yield float(md[selected].mean()), selected, width


def e2_ncc_profile(
    gr_window: np.ndarray,
    dmd: np.ndarray,
    neg_dz: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    cells: np.ndarray,
) -> np.ndarray:
    """Score every candidate datum with the E2 searched shape family.

    Args:
        gr_window: Calibrated horizontal GR within one local window.
        dmd: Measured depth centered inside the window.
        neg_dz: Negative Z centered inside the window.
        tw_tvt: Ascending typewell TVT coordinates.
        tw_gr: Smoothed typewell GR values.
        cells: Candidate TVT datum values at the window centroid.

    Returns:
        Maximum normalized cross correlation at each candidate cell.
    """
    centered = gr_window - gr_window.mean()
    norm = float(np.sqrt((centered * centered).sum()))
    best = np.full(len(cells), -2.0)
    if norm < 1e-9:
        return best
    for slope in E2_WIDE_SLOPES:
        for gain in E2_WIDE_GAINS:
            shape = slope * dmd + gain * neg_dz
            tvt = cells[:, None] + shape[None, :]
            reference = np.interp(
                tvt.ravel(), tw_tvt, tw_gr, left=np.nan, right=np.nan
            ).reshape(tvt.shape)
            valid = np.isfinite(reference).all(axis=1)
            if not valid.any():
                continue
            ref = reference[valid]
            ref = ref - ref.mean(axis=1, keepdims=True)
            ref_norm = np.sqrt((ref * ref).sum(axis=1))
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = np.where(ref_norm < 1e-9, 0.0, (ref @ centered) / (ref_norm * norm))
            best[np.flatnonzero(valid)] = np.maximum(best[np.flatnonzero(valid)], corr)
    return best


def e2_dp_track(score: np.ndarray, node_md: np.ndarray, cells: np.ndarray) -> np.ndarray:
    """Decode the fixed lambda E2 track through its bounded rate dynamic program.

    Args:
        score: Local E2 evidence with shape nodes by candidate cells.
        node_md: Ascending local evidence node positions.
        cells: Candidate absolute TVT datum values.

    Returns:
        Decoded absolute TVT at each node.
    """
    n_nodes, n_cells = score.shape
    dp = score[0].copy()
    back = np.zeros((n_nodes, n_cells), dtype=np.int32)
    for node in range(1, n_nodes):
        gap = max(abs(node_md[node] - node_md[node - 1]), 1e-6)
        norm = gap / 100.0
        band = max(int(np.floor(E2_RATE_PER_100FT * norm / E2_GRID_STEP)), 1)
        best = np.full(n_cells, -np.inf)
        arg = np.zeros(n_cells, dtype=np.int32)
        for delta in range(-band, band + 1):
            penalty = E2_LAM * (delta * E2_GRID_STEP) ** 2 / norm
            shifted = np.full(n_cells, -np.inf)
            if delta > 0:
                shifted[delta:] = dp[:n_cells - delta]
            elif delta < 0:
                shifted[:n_cells + delta] = dp[-delta:]
            else:
                shifted[:] = dp
            candidate = shifted - penalty
            update = candidate > best
            best[update] = candidate[update]
            arg[update] = -delta
        dp = score[node] + best
        back[node] = arg
    state = int(np.argmax(dp))
    path = np.empty(n_nodes, dtype=int)
    for node in range(n_nodes - 1, -1, -1):
        path[node] = state
        state = int(np.clip(state + int(back[node][state]), 0, n_cells - 1))
    return cells[path]


def predict_e2_residuals(hw: pd.DataFrame, tw: pd.DataFrame) -> dict[int, float]:
    """Predict E2 residuals for one hidden well using legal inference columns.

    Args:
        hw: Horizontal well frame containing MD, Z, GR, and TVT_input.
        tw: Typewell frame containing TVT and GR.

    Returns:
        Mapping from raw horizontal row index to TVT residual from last known.
        An empty mapping signals that carry forward must be used.
    """
    required_hw = {"MD", "Z", "GR", "TVT_input"}
    required_tw = {"TVT", "GR"}
    if not required_hw.issubset(hw.columns) or not required_tw.issubset(tw.columns):
        return {}
    known = hw["TVT_input"].notna().to_numpy()
    if not known.any() or known.all():
        return {}
    last_known_index = int(np.flatnonzero(known)[-1])
    pin = float(hw["TVT_input"].to_numpy(float)[last_known_index])
    md0 = float(hw["MD"].to_numpy(float)[last_known_index])

    tw_clean = tw[["TVT", "GR"]].dropna().sort_values("TVT")
    tw_tvt = tw_clean["TVT"].to_numpy(float)
    tw_gr = tw_clean["GR"].to_numpy(float)
    if len(tw_tvt) < 50:
        return {}
    unique = np.concatenate([[True], np.diff(tw_tvt) > 1e-9])
    tw_tvt = tw_tvt[unique]
    tw_gr = e2_smooth_gr(tw_gr[unique])

    gr_all = hw["GR"].to_numpy(float)
    tvt_input = hw["TVT_input"].to_numpy(float)
    calibration_rows = known & np.isfinite(gr_all) & np.isfinite(tvt_input)
    known_reference = np.interp(
        tvt_input[calibration_rows], tw_tvt, tw_gr, left=np.nan, right=np.nan
    )
    valid_calibration = np.isfinite(known_reference)
    if int(valid_calibration.sum()) < 20:
        return {}
    x = gr_all[calibration_rows][valid_calibration]
    if np.std(x) < 1e-6:
        return {}
    design = np.vstack([x, np.ones_like(x)]).T
    alpha, beta = np.linalg.lstsq(design, known_reference[valid_calibration], rcond=None)[0]

    eval_index = np.flatnonzero(~known)
    md_all = hw["MD"].to_numpy(float)[eval_index]
    z_all = hw["Z"].to_numpy(float)[eval_index]
    cal_all = alpha * gr_all[eval_index] + beta
    fit = np.isfinite(md_all) & np.isfinite(z_all) & np.isfinite(gr_all[eval_index])
    md = md_all[fit]
    z = z_all[fit]
    cal = e2_smooth_gr(cal_all[fit])
    if len(md) < 200:
        return {}

    windows = list(e2_iter_windows(md))
    if len(windows) < E2_MIN_EMIT_NODES:
        return {}
    centroids = np.array(sorted(centroid for centroid, _, _ in windows))
    node_md = np.unique(np.concatenate([[md0], centroids]))
    pin_node = int(np.argmin(np.abs(node_md - md0)))
    cells = pin + np.arange(-E2_GRID_HALF, E2_GRID_HALF + 1e-9, E2_GRID_STEP)
    score = np.zeros((len(node_md), len(cells)))
    for centroid, selected, width in windows:
        sampled = np.flatnonzero(selected)[::E2_ROW_SUB]
        md_window = md[sampled]
        z_window = z[sampled]
        cal_window = cal[sampled]
        finite = np.isfinite(cal_window)
        md_window = md_window[finite]
        z_window = z_window[finite]
        cal_window = cal_window[finite]
        if len(cal_window) < 20:
            continue
        dmd = md_window - float(md_window.mean())
        neg_dz = -(z_window - float(z_window.mean()))
        n_eff = max(width / E2_GR_DECORR_FT, 1.0)
        node = int(np.argmin(np.abs(node_md - centroid)))
        score[node] += n_eff * e2_ncc_profile(
            cal_window, dmd, neg_dz, tw_tvt, tw_gr, cells
        )

    pin_cell = int(np.argmin(np.abs(cells - pin)))
    score[pin_node] = -1e9
    score[pin_node, pin_cell] = 0.0
    track = e2_dp_track(score, node_md, cells)
    # The OOF emitter decodes its score only on rows with finite GR and Z, then
    # interpolates that track back to every finite MD eval row. Preserve that
    # distinction. A missing GR row still receives a legal path estimate.
    output_rows = np.isfinite(md_all)
    predicted = np.interp(md_all[output_rows], node_md, track)
    return {
        int(row): float(value - pin)
        for row, value in zip(eval_index[output_rows], predicted)
    }


def rate_posterior(
    score: np.ndarray,
    node_md: np.ndarray,
    node_z: np.ndarray,
    cells: np.ndarray,
    initial_rate: float,
) -> np.ndarray:
    """Return posterior TVT means for the fixed public rate state HMM.

    Args:
        score: E2 emissions by node and TVT cell.
        node_md: E2 node measured depths.
        node_z: Survey Z at E2 nodes.
        cells: Candidate TVT cells.
        initial_rate: Visible prefix estimate of d(TVT + Z)/dMD.

    Returns:
        Posterior mean TVT at each E2 node.
    """
    rates = np.linspace(-0.1, 0.1, 41, dtype=np.float64)
    n_nodes, n_cells = score.shape
    n_rates = len(rates)
    alpha = np.full((n_nodes, n_rates, n_cells), -np.inf, dtype=np.float64)
    start_p = int(np.argmax(score[0]))
    for rate_index, rate in enumerate(rates):
        alpha[0, rate_index, start_p] = score[0, start_p] - 0.5 * ((rate - initial_rate) / 0.01) ** 2

    transition: list[tuple[float, float, tuple[int, ...]]] = []
    for node in range(1, n_nodes):
        dm = max(float(node_md[node] - node_md[node - 1]), 1.0)
        dz = float(node_z[node] - node_z[node - 1])
        transition.append((dm, dz, tuple(range(n_rates))))
        rate_sigma = max(0.002 * np.sqrt(dm), 1e-9)
        pos_sigma = max(0.02, 0.35 * E2_GRID_STEP)
        for rate_index, rate in enumerate(rates):
            source = np.full(n_cells, -np.inf, dtype=np.float64)
            for previous_rate in range(max(0, rate_index - 1), min(n_rates, rate_index + 2)):
                rate_cost = -0.5 * ((rates[previous_rate] - rate) / rate_sigma) ** 2
                source = np.logaddexp(source, alpha[node - 1, previous_rate] + rate_cost)
            total = np.full(n_cells, -np.inf, dtype=np.float64)
            mean_move = rate * dm - dz
            center_delta = int(np.rint(mean_move / E2_GRID_STEP))
            for delta in range(center_delta - 2, center_delta + 3):
                cost = 0.5 * ((delta * E2_GRID_STEP - mean_move) / pos_sigma) ** 2
                shifted = np.full(n_cells, -np.inf, dtype=np.float64)
                if delta > 0:
                    shifted[delta:] = source[:-delta] - cost
                elif delta < 0:
                    shifted[:delta] = source[-delta:] - cost
                else:
                    shifted[:] = source - cost
                total = np.logaddexp(total, shifted)
            alpha[node, rate_index] = score[node] + total

    beta = np.zeros((n_nodes, n_rates, n_cells), dtype=np.float64)
    for node in range(n_nodes - 2, -1, -1):
        dm = max(float(node_md[node + 1] - node_md[node]), 1.0)
        dz = float(node_z[node + 1] - node_z[node])
        rate_sigma = max(0.002 * np.sqrt(dm), 1e-9)
        pos_sigma = max(0.02, 0.35 * E2_GRID_STEP)
        for previous_rate in range(n_rates):
            total = np.full(n_cells, -np.inf, dtype=np.float64)
            for rate_index in range(max(0, previous_rate - 1), min(n_rates, previous_rate + 2)):
                rate_cost = -0.5 * ((rates[previous_rate] - rates[rate_index]) / rate_sigma) ** 2
                next_term = score[node + 1] + beta[node + 1, rate_index] + rate_cost
                mean_move = rates[rate_index] * dm - dz
                center_delta = int(np.rint(mean_move / E2_GRID_STEP))
                for delta in range(center_delta - 2, center_delta + 3):
                    cost = 0.5 * ((delta * E2_GRID_STEP - mean_move) / pos_sigma) ** 2
                    shifted = np.full(n_cells, -np.inf, dtype=np.float64)
                    if delta > 0:
                        shifted[:-delta] = next_term[delta:] - cost
                    elif delta < 0:
                        shifted[-delta:] = next_term[:delta] - cost
                    else:
                        shifted[:] = next_term - cost
                    total = np.logaddexp(total, shifted)
            beta[node, previous_rate] = total

    mean = np.empty(n_nodes, dtype=np.float64)
    for node in range(n_nodes):
        joint = alpha[node] + beta[node]
        peak = float(np.max(joint))
        weight = np.exp(joint - peak)
        weight /= float(weight.sum())
        mean[node] = float(weight.sum(axis=0) @ cells)
    return mean


def predict_e2_ratehmm_posterior_residuals(hw: pd.DataFrame, tw: pd.DataFrame) -> dict[int, float]:
    """Predict public rate posterior E2 residuals using legal inference columns.

    Args:
        hw: Horizontal well frame containing MD, Z, GR, and TVT_input.
        tw: Typewell frame containing TVT and GR.

    Returns:
        Mapping from raw horizontal row index to TVT residual from last known.
        An empty mapping signals that carry forward must be used.
    """
    required_hw = {"MD", "Z", "GR", "TVT_input"}
    required_tw = {"TVT", "GR"}
    if not required_hw.issubset(hw.columns) or not required_tw.issubset(tw.columns):
        return {}
    known = hw["TVT_input"].notna().to_numpy()
    if not known.any() or known.all():
        return {}
    last_known_index = int(np.flatnonzero(known)[-1])
    pin = float(hw["TVT_input"].to_numpy(float)[last_known_index])
    md0 = float(hw["MD"].to_numpy(float)[last_known_index])
    z0 = float(hw["Z"].to_numpy(float)[last_known_index])

    tw_clean = tw[["TVT", "GR"]].dropna().sort_values("TVT")
    tw_tvt = tw_clean["TVT"].to_numpy(float)
    tw_gr = tw_clean["GR"].to_numpy(float)
    if len(tw_tvt) < 50:
        return {}
    unique = np.concatenate([[True], np.diff(tw_tvt) > 1e-9])
    tw_tvt = tw_tvt[unique]
    tw_gr = e2_smooth_gr(tw_gr[unique])

    gr_all = hw["GR"].to_numpy(float)
    tvt_input = hw["TVT_input"].to_numpy(float)
    calibration_rows = known & np.isfinite(gr_all) & np.isfinite(tvt_input)
    known_reference = np.interp(
        tvt_input[calibration_rows], tw_tvt, tw_gr, left=np.nan, right=np.nan
    )
    valid_calibration = np.isfinite(known_reference)
    if int(valid_calibration.sum()) < 20:
        return {}
    x = gr_all[calibration_rows][valid_calibration]
    if np.std(x) < 1e-6:
        return {}
    design = np.vstack([x, np.ones_like(x)]).T
    alpha, beta = np.linalg.lstsq(design, known_reference[valid_calibration], rcond=None)[0]

    eval_index = np.flatnonzero(~known)
    md_all = hw["MD"].to_numpy(float)[eval_index]
    z_all = hw["Z"].to_numpy(float)[eval_index]
    cal_all = alpha * gr_all[eval_index] + beta
    fit = np.isfinite(md_all) & np.isfinite(z_all) & np.isfinite(gr_all[eval_index])
    md = md_all[fit]
    z = z_all[fit]
    cal = e2_smooth_gr(cal_all[fit])
    if len(md) < 200:
        return {}

    windows = list(e2_iter_windows(md))
    if len(windows) < E2_MIN_EMIT_NODES:
        return {}
    centroids = np.array(sorted(centroid for centroid, _, _ in windows))
    node_md = np.unique(np.concatenate([[md0], centroids]))
    node_z = np.interp(node_md, np.r_[md0, md], np.r_[z0, z])
    pin_node = int(np.argmin(np.abs(node_md - md0)))
    cells = pin + np.arange(-E2_GRID_HALF, E2_GRID_HALF + 1e-9, E2_GRID_STEP)
    score = np.zeros((len(node_md), len(cells)))
    for centroid, selected, width in windows:
        sampled = np.flatnonzero(selected)[::E2_ROW_SUB]
        md_window = md[sampled]
        z_window = z[sampled]
        cal_window = cal[sampled]
        finite = np.isfinite(cal_window)
        md_window = md_window[finite]
        z_window = z_window[finite]
        cal_window = cal_window[finite]
        if len(cal_window) < 20:
            continue
        dmd = md_window - float(md_window.mean())
        neg_dz = -(z_window - float(z_window.mean()))
        n_eff = max(width / E2_GR_DECORR_FT, 1.0)
        node = int(np.argmin(np.abs(node_md - centroid)))
        score[node] += n_eff * e2_ncc_profile(
            cal_window, dmd, neg_dz, tw_tvt, tw_gr, cells
        )

    pin_cell = int(np.argmin(np.abs(cells - pin)))
    score[pin_node] = -1e9
    score[pin_node, pin_cell] = 0.0
    known_index = np.flatnonzero(known)
    tail_index = known_index[-30:]
    tail_md = hw["MD"].to_numpy(float)[tail_index]
    tail_z = hw["Z"].to_numpy(float)[tail_index]
    tail_tvt = hw["TVT_input"].to_numpy(float)[tail_index]
    dm_tail = np.diff(tail_md)
    valid_tail = dm_tail > 0
    if int(valid_tail.sum()) >= 3:
        initial_rate = float(np.median(
            (np.diff(tail_tvt)[valid_tail] + np.diff(tail_z)[valid_tail]) / dm_tail[valid_tail]
        ))
    else:
        initial_rate = 0.0
    track = rate_posterior(score, node_md, node_z, cells, initial_rate)
    # The OOF emitter decodes its score only on rows with finite GR and Z, then
    # interpolates that track back to every finite MD eval row. Preserve that
    # distinction. A missing GR row still receives a legal path estimate.
    output_rows = np.isfinite(md_all)
    predicted = np.interp(md_all[output_rows], node_md, track)
    return {
        int(row): float(value - pin)
        for row, value in zip(eval_index[output_rows], predicted)
    }


# This decoder deliberately mirrors the 773 well rate coupled OOF emitter. It
# remains separate from the public 41 state posterior above because the latter
# has a different state grid and rate random walk variance.
E2_RATECOUPLED_STATES = np.linspace(-0.08, 0.08, 9, dtype=np.float64)


def e2_ncc_profiles(
    gr_window: np.ndarray,
    dmd: np.ndarray,
    neg_dz: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    cells: np.ndarray,
    shapes: list[tuple[float, float]],
) -> np.ndarray:
    """Return one normalized correlation profile per local E2 shape.

    Args:
        gr_window: Calibrated horizontal GR for one local window.
        dmd: Centered measured depth values.
        neg_dz: Negative centered Z values.
        tw_tvt: Ascending typewell TVT coordinates.
        tw_gr: Smoothed typewell GR values.
        cells: Candidate absolute TVT datum values.
        shapes: Slope and survey gain pairs.

    Returns:
        An array of shape ``(len(shapes), len(cells))``. Invalid cells retain
        the E2 sentinel value negative two.
    """
    centered = gr_window - gr_window.mean()
    norm = float(np.sqrt((centered * centered).sum()))
    out = np.full((len(shapes), len(cells)), -2.0, dtype=np.float64)
    if norm < 1e-9:
        return out
    for shape_index, (slope, gain) in enumerate(shapes):
        tvt = cells[:, None] + slope * dmd[None, :] + gain * neg_dz[None, :]
        reference = np.interp(
            tvt.ravel(), tw_tvt, tw_gr, left=np.nan, right=np.nan
        ).reshape(tvt.shape)
        valid = np.isfinite(reference).all(axis=1)
        if not valid.any():
            continue
        ref = reference[valid]
        ref = ref - ref.mean(axis=1, keepdims=True)
        ref_norm = np.sqrt((ref * ref).sum(axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.where(
                ref_norm < 1e-9, 0.0, (ref @ centered) / (ref_norm * norm)
            )
        out[shape_index, np.flatnonzero(valid)] = corr
    return out


def e2_rate_profile_bins(
    dmd: np.ndarray,
    neg_dz: np.ndarray,
    shapes: list[tuple[float, float]],
) -> np.ndarray:
    """Assign each local E2 shape to its implied TVT plus Z rate state.

    Args:
        dmd: Centered measured depth values in one matching window.
        neg_dz: Negative centered Z values in the same window.
        shapes: Slope and survey gain pairs.

    Returns:
        Nearest rate state index for every local shape.
    """
    dz_dmd = float(np.polyfit(dmd, -neg_dz, 1)[0])
    shape_rates = np.asarray(
        [slope + (1.0 - gain) * dz_dmd for slope, gain in shapes],
        dtype=np.float64,
    )
    return np.abs(
        shape_rates[:, None] - E2_RATECOUPLED_STATES[None, :]
    ).argmin(axis=1)


def e2_rate_emission(
    profiles: np.ndarray,
    rate_bins: np.ndarray,
    n_eff: float,
) -> np.ndarray:
    """Convert separated local shape profiles into rate specific emissions.

    Args:
        profiles: Normalized correlation profiles by local shape and TVT cell.
        rate_bins: Rate state index assigned to each local shape.
        n_eff: Effective independent GR sample count in the window.

    Returns:
        Additive emission scores by rate state and TVT cell.
    """
    out = np.full(
        (len(E2_RATECOUPLED_STATES), profiles.shape[1]), -1e9, dtype=np.float64
    )
    for rate_index in range(len(E2_RATECOUPLED_STATES)):
        selected = profiles[rate_bins == rate_index]
        if len(selected):
            best = np.max(selected, axis=0)
            unsupported = best <= -1.5
            floor = float(best[~unsupported].min()) if (~unsupported).any() else -1.0
            out[rate_index] = n_eff * np.where(unsupported, floor, best)
    return out


def rate_coupled_posterior(
    emission: np.ndarray,
    node_md: np.ndarray,
    node_z: np.ndarray,
    cells: np.ndarray,
    initial_rate: float,
) -> np.ndarray:
    """Decode the nine state E2 rate coupled forward backward posterior.

    Args:
        emission: E2 evidence by node, rate state, and TVT cell.
        node_md: E2 node measured depths.
        node_z: E2 node Z coordinates.
        cells: Candidate TVT values.
        initial_rate: Known prefix estimate of d(TVT plus Z) divided by dMD.

    Returns:
        Posterior mean absolute TVT at each E2 node.
    """
    rates = E2_RATECOUPLED_STATES
    n_nodes, n_rates, n_cells = emission.shape
    alpha = np.full((n_nodes, n_rates, n_cells), -np.inf, dtype=np.float64)
    start = int(np.argmax(emission[0].max(axis=0)))
    for rate_index, rate in enumerate(rates):
        alpha[0, rate_index, start] = (
            emission[0, rate_index, start]
            - 0.5 * ((rate - initial_rate) / 0.01) ** 2
        )
    for node in range(1, n_nodes):
        dm = max(float(node_md[node] - node_md[node - 1]), 1.0)
        dz = float(node_z[node] - node_z[node - 1])
        rate_sigma = max(0.006 * np.sqrt(dm), 1e-9)
        pos_sigma = max(0.02, 0.35 * E2_GRID_STEP)
        for rate_index, rate in enumerate(rates):
            source = np.full(n_cells, -np.inf, dtype=np.float64)
            for previous in range(max(0, rate_index - 1), min(n_rates, rate_index + 2)):
                cost = -0.5 * ((rates[previous] - rate) / rate_sigma) ** 2
                source = np.logaddexp(source, alpha[node - 1, previous] + cost)
            total = np.full(n_cells, -np.inf, dtype=np.float64)
            mean_move = rate * dm - dz
            center = int(np.rint(mean_move / E2_GRID_STEP))
            for delta in range(center - 2, center + 3):
                cost = 0.5 * ((delta * E2_GRID_STEP - mean_move) / pos_sigma) ** 2
                shifted = np.full(n_cells, -np.inf, dtype=np.float64)
                if delta > 0:
                    shifted[delta:] = source[:-delta] - cost
                elif delta < 0:
                    shifted[:delta] = source[-delta:] - cost
                else:
                    shifted[:] = source - cost
                total = np.logaddexp(total, shifted)
            alpha[node, rate_index] = emission[node, rate_index] + total
    beta = np.zeros((n_nodes, n_rates, n_cells), dtype=np.float64)
    for node in range(n_nodes - 2, -1, -1):
        dm = max(float(node_md[node + 1] - node_md[node]), 1.0)
        dz = float(node_z[node + 1] - node_z[node])
        rate_sigma = max(0.006 * np.sqrt(dm), 1e-9)
        pos_sigma = max(0.02, 0.35 * E2_GRID_STEP)
        for previous in range(n_rates):
            total = np.full(n_cells, -np.inf, dtype=np.float64)
            for rate_index in range(max(0, previous - 1), min(n_rates, previous + 2)):
                rate_cost = -0.5 * ((rates[previous] - rates[rate_index]) / rate_sigma) ** 2
                next_term = emission[node + 1, rate_index] + beta[node + 1, rate_index] + rate_cost
                mean_move = rates[rate_index] * dm - dz
                center = int(np.rint(mean_move / E2_GRID_STEP))
                for delta in range(center - 2, center + 3):
                    cost = 0.5 * ((delta * E2_GRID_STEP - mean_move) / pos_sigma) ** 2
                    shifted = np.full(n_cells, -np.inf, dtype=np.float64)
                    if delta > 0:
                        shifted[:-delta] = next_term[delta:] - cost
                    elif delta < 0:
                        shifted[-delta:] = next_term[:delta] - cost
                    else:
                        shifted[:] = next_term - cost
                    total = np.logaddexp(total, shifted)
            beta[node, previous] = total
    mean = np.empty(n_nodes, dtype=np.float64)
    for node in range(n_nodes):
        joint = alpha[node] + beta[node]
        peak = float(np.max(joint))
        weight = np.exp(joint - peak)
        weight /= float(weight.sum())
        mean[node] = float(weight.sum(axis=0) @ cells)
    return mean


def predict_e2_ratecoupled_residuals(
    hw: pd.DataFrame, tw: pd.DataFrame
) -> dict[int, float]:
    """Predict the OOF matched nine state rate coupled E2 residual for one well.

    Args:
        hw: Horizontal well data with MD, Z, GR, and TVT_input.
        tw: Typewell data with TVT and GR.

    Returns:
        Residual predictions keyed by raw horizontal row index.
    """
    required_hw = {"MD", "Z", "GR", "TVT_input"}
    required_tw = {"TVT", "GR"}
    if not required_hw.issubset(hw.columns) or not required_tw.issubset(tw.columns):
        return {}
    known = hw["TVT_input"].notna().to_numpy()
    if not known.any() or known.all():
        return {}
    last_known_index = int(np.flatnonzero(known)[-1])
    pin = float(hw["TVT_input"].to_numpy(float)[last_known_index])
    md0 = float(hw["MD"].to_numpy(float)[last_known_index])
    z0 = float(hw["Z"].to_numpy(float)[last_known_index])
    tw_clean = tw[["TVT", "GR"]].dropna().sort_values("TVT")
    tw_tvt = tw_clean["TVT"].to_numpy(float)
    tw_gr = tw_clean["GR"].to_numpy(float)
    if len(tw_tvt) < 50:
        return {}
    unique = np.concatenate([[True], np.diff(tw_tvt) > 1e-9])
    tw_tvt = tw_tvt[unique]
    tw_gr = e2_smooth_gr(tw_gr[unique])
    gr_all = hw["GR"].to_numpy(float)
    tvt_input = hw["TVT_input"].to_numpy(float)
    calibration_rows = known & np.isfinite(gr_all) & np.isfinite(tvt_input)
    known_reference = np.interp(
        tvt_input[calibration_rows], tw_tvt, tw_gr, left=np.nan, right=np.nan
    )
    valid_calibration = np.isfinite(known_reference)
    if int(valid_calibration.sum()) < 20:
        return {}
    x = gr_all[calibration_rows][valid_calibration]
    if np.std(x) < 1e-6:
        return {}
    design = np.vstack([x, np.ones_like(x)]).T
    alpha, beta = np.linalg.lstsq(design, known_reference[valid_calibration], rcond=None)[0]
    eval_index = np.flatnonzero(~known)
    md_all = hw["MD"].to_numpy(float)[eval_index]
    z_all = hw["Z"].to_numpy(float)[eval_index]
    cal_all = alpha * gr_all[eval_index] + beta
    fit = np.isfinite(md_all) & np.isfinite(z_all) & np.isfinite(gr_all[eval_index])
    md = md_all[fit]
    z = z_all[fit]
    cal = e2_smooth_gr(cal_all[fit])
    if len(md) < 200:
        return {}
    windows = list(e2_iter_windows(md))
    if len(windows) < E2_MIN_EMIT_NODES:
        return {}
    centroids = np.array(sorted(centroid for centroid, _, _ in windows))
    node_md = np.unique(np.concatenate([[md0], centroids]))
    node_z = np.interp(node_md, np.r_[md0, md], np.r_[z0, z])
    pin_node = int(np.argmin(np.abs(node_md - md0)))
    cells = pin + np.arange(-E2_GRID_HALF, E2_GRID_HALF + 1e-9, E2_GRID_STEP)
    shapes = [
        (float(slope), float(gain))
        for slope in E2_WIDE_SLOPES
        for gain in E2_WIDE_GAINS
    ]
    emission = np.zeros(
        (len(node_md), len(E2_RATECOUPLED_STATES), len(cells)), dtype=np.float64
    )
    for centroid, selected, width in windows:
        sampled = np.flatnonzero(selected)[::E2_ROW_SUB]
        md_window = md[sampled]
        z_window = z[sampled]
        cal_window = cal[sampled]
        finite = np.isfinite(cal_window)
        md_window = md_window[finite]
        z_window = z_window[finite]
        cal_window = cal_window[finite]
        if len(cal_window) < 20:
            continue
        dmd = md_window - float(md_window.mean())
        neg_dz = -(z_window - float(z_window.mean()))
        n_eff = max(width / E2_GR_DECORR_FT, 1.0)
        node = int(np.argmin(np.abs(node_md - centroid)))
        profiles = e2_ncc_profiles(
            cal_window, dmd, neg_dz, tw_tvt, tw_gr, cells, shapes
        )
        emission[node] += e2_rate_emission(
            profiles, e2_rate_profile_bins(dmd, neg_dz, shapes), n_eff
        )
    pin_cell = int(np.argmin(np.abs(cells - pin)))
    emission[pin_node] = -1e9
    emission[pin_node, :, pin_cell] = 0.0
    known_index = np.flatnonzero(known)
    tail_index = known_index[-30:]
    tail_md = hw["MD"].to_numpy(float)[tail_index]
    tail_z = hw["Z"].to_numpy(float)[tail_index]
    tail_tvt = hw["TVT_input"].to_numpy(float)[tail_index]
    dm_tail = np.diff(tail_md)
    valid_tail = dm_tail > 0
    if int(valid_tail.sum()) >= 3:
        initial_rate = float(np.median(
            (np.diff(tail_tvt)[valid_tail] + np.diff(tail_z)[valid_tail])
            / dm_tail[valid_tail]
        ))
    else:
        initial_rate = 0.0
    output_rows = np.isfinite(md_all)
    predicted = np.interp(
        md_all[output_rows], node_md,
        rate_coupled_posterior(emission, node_md, node_z, cells, initial_rate),
    )
    return {
        int(row): float(value - pin)
        for row, value in zip(eval_index[output_rows], predicted)
    }

def tvt_from_contacts(hw_tr: pd.DataFrame, tw_tr: pd.DataFrame, ref_col: str = "EGFDU") -> pd.Series:
    """Reconstruct TVT from a formation contact (visible wells only).

    Uses a train-only formation-top column (default ``EGFDU``) and a known
    ``TVT`` to fit a single offset. Only valid where ``TVT`` and the
    formation columns are present (i.e. train wells), not hidden test wells.

    Args:
        hw_tr: Horizontal-well frame with ``Z``, ``TVT`` and ``ref_col``.
        tw_tr: Typewell frame with ``Geology`` and ``TVT``.
        ref_col: Formation-top column to anchor against.

    Returns:
        Reconstructed TVT as a pandas Series aligned to ``hw_tr``.
    """
    tw_g = tw_tr.dropna(subset=["Geology"])
    ref_tvt = tw_g[tw_g["Geology"] == ref_col]["TVT"].min()
    if np.isnan(ref_tvt):
        ref_col = tw_g["Geology"].iloc[0]
        ref_tvt = tw_g[tw_g["Geology"] == ref_col]["TVT"].min()
    offset = (hw_tr["TVT"] - (ref_tvt - (hw_tr["Z"] - hw_tr[ref_col]))).mean()
    return ref_tvt - (hw_tr["Z"] - hw_tr[ref_col]) + offset


def run_particle_filter(hw: pd.DataFrame, tw: pd.DataFrame, n_particles: int = 500, seed: int = 42):
    """Conservative particle filter tracking ``pos = TVT + Z`` via GR.

    GR gaps in the horizontal well are interpolated before tracking so the
    filter always has an observation (critical for wells with high NaN GR).

    Args:
        hw: Horizontal-well frame (``MD``, ``Z``, ``GR``, ``TVT_input``).
        tw: Typewell frame (``TVT``, ``GR``).
        n_particles: Number of particles.
        seed: RNG seed for this realization.

    Returns:
        Tuple ``(predictions, total_log_likelihood)`` where predictions is a
        full-length TVT array (known rows carry ``TVT_input``).
    """
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].values.astype(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).values.astype(float)

    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return hw["TVT_input"].values.astype(float).copy(), 0.0

    last = kn.iloc[-1]
    last_tvt = float(last["TVT_input"])
    last_Z = float(last["Z"])
    last_MD = float(last["MD"])

    tw_at_k = np.interp(kn["TVT_input"].values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn["GR"].fillna(0).values - tw_at_k), 10.0, 60.0))

    tail = kn.tail(30)
    dt = np.diff(tail["TVT_input"].values)
    dz = np.diff(tail["Z"].values)
    dm = np.diff(tail["MD"].values)
    m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0

    N = n_particles
    rng = np.random.default_rng(seed)
    ls = last_tvt + last_Z
    pos = ls + 2.0 * rng.standard_normal(N)  # wider init spread
    rate = ir + 0.01 * rng.standard_normal(N)
    w = np.ones(N) / N

    MOM = 0.998
    VN = 0.002
    PN = 0.005
    RP = 0.1
    RR = 0.001
    RESAMP = 0.5

    md_v = ev["MD"].values.astype(float)
    z_v = ev["Z"].values.astype(float)
    gr_interp = hw["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean())
    gr_v = gr_interp.values.astype(float)[ev.index]

    out_vals = hw["TVT_input"].values.astype(float).copy()
    res = np.empty(len(ev))
    prev_MD = last_MD
    log_lik = 0.0

    for i in range(len(ev)):
        dm_step = max(md_v[i] - prev_MD, 1.0)
        rate = MOM * rate + VN * rng.standard_normal(N)
        pos = pos + rate * dm_step + PN * rng.standard_normal(N)
        tvt_p = pos - z_v[i]
        tvt_p = np.clip(tvt_p, tw_tvt[0] - 100, tw_tvt[-1] + 100)
        pos = tvt_p + z_v[i]

        eg = np.interp(tvt_p, tw_tvt, tw_gr)
        d = (gr_v[i] - eg) / gs
        lk = np.exp(-0.5 * np.minimum(d ** 2, 600.0))
        lk = np.maximum(lk, 1e-300)
        avg_lk = float((w * lk).sum())
        log_lik += np.log(max(avg_lk, 1e-300))
        w = w * lk
        ws = w.sum()
        w = w / ws if ws > 0 else np.ones(N) / N

        n_eff = 1.0 / (w ** 2).sum()
        if n_eff < RESAMP * N:
            cum = np.cumsum(w)
            u0 = rng.uniform(0, 1.0 / N)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
            pos = pos[idx] + RP * rng.standard_normal(N)
            rate = rate[idx] + RR * rng.standard_normal(N)
            w = np.ones(N) / N

        res[i] = float(np.dot(w, pos - z_v[i]))
        prev_MD = md_v[i]

    out_vals[list(ev.index)] = res
    return out_vals, log_lik


def run_pf_lik_ensemble_scales(hw, tw, scales=SELECTOR_SCALES, n_particles=500,
                               n_seeds=256, return_stats=False):
    """Run a many-seed PF ensemble; return likelihood-weighted blends per scale.

    Args:
        hw: Horizontal-well frame.
        tw: Typewell frame.
        scales: Likelihood softmax temperatures.
        n_particles: Particles per PF run.
        n_seeds: Number of independent seeds.
        return_stats: If True, also return target-free instability stats.

    Returns:
        Dict mapping ``"pf_scale_<s>"`` (and ``"pf_mean"``) to full-length
        prediction arrays. If ``return_stats`` is True, the dict also holds
        ``seed_std`` (mean over eval rows of the across-seed std of the path,
        a target-free uncertainty signal) and ``lik_entropy`` (normalized
        Shannon entropy of the scale-5 likelihood softmax weights).
    """
    preds = []
    liks = []
    for s in range(n_seeds):
        p, ll = run_particle_filter(hw, tw, n_particles=n_particles, seed=s)
        preds.append(p)
        liks.append(ll)
    pred_arr = np.stack(preds, 0)
    liks = np.array(liks)
    liks_n = liks - liks.max()
    out = {}
    for scale in scales:
        weights = np.exp(liks_n / float(scale))
        weights /= weights.sum()
        out[f"pf_scale_{scale:g}"] = (weights[:, None] * pred_arr).sum(0)
    out["pf_mean"] = pred_arr.mean(0)
    if return_stats:
        ev_mask = hw["TVT_input"].isna().to_numpy()
        out["seed_std"] = float(np.mean(pred_arr[:, ev_mask].std(axis=0))) if ev_mask.any() else 0.0
        w5 = np.exp(liks_n / 5.0)
        w5 = w5 / w5.sum()
        nz = w5[w5 > 0]
        out["lik_entropy"] = float(-(nz * np.log(nz)).sum() / np.log(len(w5))) if len(w5) > 1 else 0.0
    return out


def beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs=10, mc=20.0, es=144.0, r=2):
    """Vectorized beam search for TVT tracking via GR matching.

    Args:
        hgr: Horizontal GR over eval rows (NaN-interpolated).
        tw_tvt: Typewell TVT grid (sorted).
        tw_gr: Typewell GR aligned to ``tw_tvt``.
        last_tvt: Last known TVT (search start).
        bs: Beam size.
        mc: Motion cost scale.
        es: GR error scale.
        r: Savitzky-Golay smoothing radius (0 disables).

    Returns:
        Array of tracked TVT values over the eval rows.
    """
    n = len(hgr)
    nt = len(tw_tvt)
    if n == 0:
        return np.array([last_tvt])

    if r > 0 and n > max(3, 2 * r + 1):
        win = min(2 * r + 1, n if n % 2 == 1 else n - 1)
        sgr = savgol_filter(hgr, win, min(2, win - 1))
    else:
        sgr = hgr.copy()

    si = int(np.argmin(np.abs(tw_tvt - last_tvt)))
    MOVES = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
    MC = mc * np.array([2.0, 1.0, 0.0, 1.0, 2.0])

    bidx = np.full(bs, si, dtype=np.int64)
    bcost = np.full(bs, np.inf)
    bcost[0] = 0.0
    bn = 1
    result = np.zeros(n)

    for step in range(n):
        gv = sgr[step]
        ni = bidx[:bn, None] + MOVES[None, :]
        ci = np.clip(ni, 0, nt - 1)
        valid = (ni >= 0) & (ni < nt)
        gr_e = (gv - tw_gr[ci]) ** 2 / es
        tot = bcost[:bn, None] + gr_e + MC[None, :]
        tot = np.where(valid, tot, np.inf)

        ni_f = ni.flatten()
        tot_f = tot.flatten()
        vf = valid.flatten()
        ni_f = ni_f[vf]
        tot_f = tot_f[vf]

        order = np.argsort(tot_f)
        ni_s = ni_f[order]
        tot_s = tot_f[order]
        _, first = np.unique(ni_s, return_index=True)
        ni_u = ni_s[first]
        tot_u = tot_s[first]

        kept = min(bs, len(ni_u))
        top = np.argpartition(tot_u, min(kept - 1, len(tot_u) - 1))[:kept]
        top = top[np.argsort(tot_u[top])]
        bidx[:kept] = ni_u[top]
        bcost[:kept] = tot_u[top]
        if kept < bs:
            bidx[kept:] = bidx[kept - 1]
            bcost[kept:] = np.inf
        bn = kept
        result[step] = tw_tvt[bidx[0]]

    return result


def run_beam_ensemble(hw, tw):
    """Average the 14 beam-search configs into a full-length prediction."""
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return hw["TVT_input"].values.astype(float).copy()

    last_tvt = float(kn.iloc[-1]["TVT_input"])
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].values.astype(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).values.astype(float)

    gr_all = hw["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean()).values.astype(float)
    hgr = gr_all[ev.index]

    beam_results = [beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r)
                    for (bs, mc, es, r) in BEAM_CONFIGS]
    beam_mean = np.stack(beam_results, 0).mean(0)

    out = hw["TVT_input"].values.astype(float).copy()
    out[list(ev.index)] = beam_mean
    return out


def selector_well_code(hw):
    """Map a well to a selector bin code and variant name."""
    eval_mask = hw["TVT_input"].isna().to_numpy()
    n_eval = float(eval_mask.sum())
    z_eval = hw.loc[eval_mask, "Z"].values.astype(float)
    z_span = float(np.nanmax(z_eval) - np.nanmin(z_eval)) if len(z_eval) else 0.0
    n_bin = int(n_eval > SELECTOR_N_EVAL_THRESHOLD)
    z_bin = int(np.searchsorted(SELECTOR_Z_SPAN_THRESHOLDS, z_span, side="right"))
    code = n_bin + 2 * z_bin
    variant = SELECTOR_BIN_VARIANTS.get(code, SELECTOR_GLOBAL_VARIANT)
    return code, variant, n_eval, z_span


def parse_selector_variant(name):
    """Parse a selector variant name into (scale, beam_weight, hold_weight)."""
    parts = name.split("_")
    scale = float(parts[2])
    beam_weight = 0.0
    hold_weight = 0.0
    if "beam" in parts:
        beam_weight = float(parts[parts.index("beam") + 1])
    if "hold" in parts:
        hold_weight = float(parts[parts.index("hold") + 1])
    return scale, beam_weight, hold_weight


def apply_selector_variant(name, pf_by_scale, tvt_beam, last_known_tvt):
    """Blend PF/beam/hold according to a selector variant name."""
    scale, beam_weight, hold_weight = parse_selector_variant(name)
    base = pf_by_scale.get(f"pf_scale_{scale:g}")
    if base is None:
        base = pf_by_scale[SELECTOR_GLOBAL_VARIANT.split("_beam_")[0].split("_hold_")[0]]
    pred = (1.0 - beam_weight) * base + beam_weight * tvt_beam
    pred = (1.0 - hold_weight) * pred + hold_weight * last_known_tvt
    return pred


def predict_well(hw, tw, n_particles=500, n_seeds=256):
    """Full hidden-well prediction: PF ensemble + beam + selector.

    Args:
        hw: Horizontal-well frame for the well.
        tw: Typewell frame for the well.
        n_particles: Particles per PF run.
        n_seeds: PF seeds.

    Returns:
        Dict with the selector prediction plus the component arrays
        (``pf_scale_*``, ``pf_mean``, ``beam``) for diagnostics.
    """
    pf_by_scale = run_pf_lik_ensemble_scales(hw, tw, n_particles=n_particles, n_seeds=n_seeds)
    tvt_beam = run_beam_ensemble(hw, tw)
    last_known = hw["TVT_input"].dropna()
    last_known_tvt = float(last_known.iloc[-1]) if len(last_known) else float(np.nanmean(pf_by_scale["pf_mean"]))
    _, variant, n_eval, z_span = selector_well_code(hw)
    selected = apply_selector_variant(variant, pf_by_scale, tvt_beam, last_known_tvt)
    out = {"selected": selected, "beam": tvt_beam, "variant": variant,
           "n_eval": n_eval, "z_span": z_span}
    out.update(pf_by_scale)
    return out



# %% [markdown]
# ## 6c. Public rate Viterbi decoder for untried19_ratehmm_public_sg5
#
# `rate_viterbi` is inlined verbatim from scripts/emit_untried19_ratehmm_2026_07_31.py,
# the emitter that produced the banked OOF. The wrapper below is this kernel's own
# posterior wrapper with a single identifier substituted, generated rather than copied
# so the two cannot drift. Verified bit-identical to the bank, 0.0 ft over 40 wells,
# by scripts/verify_ratehmm_public_port_2026_08_03.py.

# %%
# GRID_STEP is the E2 TVT cell step that rate_viterbi expects.
GRID_STEP = E2_GRID_STEP

def rate_viterbi(
    score: np.ndarray,
    node_md: np.ndarray,
    node_z: np.ndarray,
    cells: np.ndarray,
    initial_rate: float,
) -> np.ndarray:
    """Decode E2 emissions with the public notebook's TVT plus rate state.

    The rate tracks ``TVT + Z`` per measured depth. Its grid, initial width,
    random walk width, position width and five point spatial transition are
    copied from the public HMM configuration. This is a Viterbi readout rather
    than its expensive full forward backward tensor.

    Args:
        score: E2 emissions indexed by node and TVT cell.
        node_md: E2 node measured depth positions.
        node_z: Survey Z values at E2 nodes.
        cells: Candidate TVT cells.
        initial_rate: Last visible prefix estimate of d(TVT + Z)/dMD.

    Returns:
        One absolute TVT value at each E2 node.
    """
    rates = np.linspace(-0.1, 0.1, 41, dtype=np.float64)
    rate_step = float(rates[1] - rates[0])
    n_nodes, n_cells = score.shape
    n_rates = len(rates)
    negative = -1e30
    dp = np.full((n_rates, n_cells), negative, dtype=np.float64)
    start_p = int(np.argmax(score[0]))
    for rate_index, rate in enumerate(rates):
        prior = -0.5 * ((rate - initial_rate) / 0.01) ** 2
        dp[rate_index, start_p] = score[0, start_p] + prior

    back_pos = np.zeros((n_nodes, n_rates, n_cells), dtype=np.int16)
    back_rate = np.zeros((n_nodes, n_rates, n_cells), dtype=np.int8)
    for node in range(1, n_nodes):
        dm = max(float(node_md[node] - node_md[node - 1]), 1.0)
        dz = float(node_z[node] - node_z[node - 1])
        rate_sigma = max(0.002 * np.sqrt(dm), 1e-9)
        pos_sigma = max(0.02, 0.35 * GRID_STEP)
        current = np.full_like(dp, negative)
        for rate_index, rate in enumerate(rates):
            candidates = np.arange(max(0, rate_index - 1),
                                   min(n_rates, rate_index + 2))
            rate_cost = -0.5 * ((rates[candidates] - rate) / rate_sigma) ** 2
            stacked = dp[candidates] + rate_cost[:, None]
            best_choice = np.argmax(stacked, axis=0)
            best_rate = candidates[best_choice]
            source = stacked[best_choice, np.arange(n_cells)]
            mean_move = rate * dm - dz
            center_delta = int(np.rint(mean_move / GRID_STEP))
            best = np.full(n_cells, negative, dtype=np.float64)
            best_prev = np.zeros(n_cells, dtype=np.int16)
            for delta in range(center_delta - 2, center_delta + 3):
                move = delta * GRID_STEP
                cost = 0.5 * ((move - mean_move) / pos_sigma) ** 2
                if delta > 0:
                    candidate = source[:-delta] - cost
                    update = candidate > best[delta:]
                    target = np.flatnonzero(update) + delta
                    best[target] = candidate[update]
                    best_prev[target] = np.flatnonzero(update).astype(np.int16)
                elif delta < 0:
                    candidate = source[-delta:] - cost
                    update = candidate > best[:delta]
                    target = np.flatnonzero(update)
                    best[target] = candidate[update]
                    best_prev[target] = (target - delta).astype(np.int16)
                else:
                    update = source > best
                    best[update] = source[update]
                    best_prev[update] = np.arange(n_cells, dtype=np.int16)[update]
            current[rate_index] = score[node] + best
            back_pos[node, rate_index] = best_prev
            back_rate[node, rate_index] = best_rate.astype(np.int8)
        dp = current

    rate_index, cell_index = np.unravel_index(np.argmax(dp), dp.shape)
    path = np.empty(n_nodes, dtype=np.int16)
    for node in range(n_nodes - 1, -1, -1):
        path[node] = cell_index
        if node:
            prev_cell = int(back_pos[node, rate_index, cell_index])
            prev_rate = int(back_rate[node, rate_index, cell_index])
            cell_index, rate_index = prev_cell, prev_rate
    return cells[path]

def predict_e2_ratehmm_public_residuals(hw: pd.DataFrame, tw: pd.DataFrame) -> dict[int, float]:
    """Predict public rate VITERBI E2 residuals using legal inference columns.

    Args:
        hw: Horizontal well frame containing MD, Z, GR, and TVT_input.
        tw: Typewell frame containing TVT and GR.

    Returns:
        Mapping from raw horizontal row index to TVT residual from last known.
        An empty mapping signals that carry forward must be used.
    """
    required_hw = {"MD", "Z", "GR", "TVT_input"}
    required_tw = {"TVT", "GR"}
    if not required_hw.issubset(hw.columns) or not required_tw.issubset(tw.columns):
        return {}
    known = hw["TVT_input"].notna().to_numpy()
    if not known.any() or known.all():
        return {}
    last_known_index = int(np.flatnonzero(known)[-1])
    pin = float(hw["TVT_input"].to_numpy(float)[last_known_index])
    md0 = float(hw["MD"].to_numpy(float)[last_known_index])
    z0 = float(hw["Z"].to_numpy(float)[last_known_index])

    tw_clean = tw[["TVT", "GR"]].dropna().sort_values("TVT")
    tw_tvt = tw_clean["TVT"].to_numpy(float)
    tw_gr = tw_clean["GR"].to_numpy(float)
    if len(tw_tvt) < 50:
        return {}
    unique = np.concatenate([[True], np.diff(tw_tvt) > 1e-9])
    tw_tvt = tw_tvt[unique]
    tw_gr = e2_smooth_gr(tw_gr[unique])

    gr_all = hw["GR"].to_numpy(float)
    tvt_input = hw["TVT_input"].to_numpy(float)
    calibration_rows = known & np.isfinite(gr_all) & np.isfinite(tvt_input)
    known_reference = np.interp(
        tvt_input[calibration_rows], tw_tvt, tw_gr, left=np.nan, right=np.nan
    )
    valid_calibration = np.isfinite(known_reference)
    if int(valid_calibration.sum()) < 20:
        return {}
    x = gr_all[calibration_rows][valid_calibration]
    if np.std(x) < 1e-6:
        return {}
    design = np.vstack([x, np.ones_like(x)]).T
    alpha, beta = np.linalg.lstsq(design, known_reference[valid_calibration], rcond=None)[0]

    eval_index = np.flatnonzero(~known)
    md_all = hw["MD"].to_numpy(float)[eval_index]
    z_all = hw["Z"].to_numpy(float)[eval_index]
    cal_all = alpha * gr_all[eval_index] + beta
    fit = np.isfinite(md_all) & np.isfinite(z_all) & np.isfinite(gr_all[eval_index])
    md = md_all[fit]
    z = z_all[fit]
    cal = e2_smooth_gr(cal_all[fit])
    if len(md) < 200:
        return {}

    windows = list(e2_iter_windows(md))
    if len(windows) < E2_MIN_EMIT_NODES:
        return {}
    centroids = np.array(sorted(centroid for centroid, _, _ in windows))
    node_md = np.unique(np.concatenate([[md0], centroids]))
    node_z = np.interp(node_md, np.r_[md0, md], np.r_[z0, z])
    pin_node = int(np.argmin(np.abs(node_md - md0)))
    cells = pin + np.arange(-E2_GRID_HALF, E2_GRID_HALF + 1e-9, E2_GRID_STEP)
    score = np.zeros((len(node_md), len(cells)))
    for centroid, selected, width in windows:
        sampled = np.flatnonzero(selected)[::E2_ROW_SUB]
        md_window = md[sampled]
        z_window = z[sampled]
        cal_window = cal[sampled]
        finite = np.isfinite(cal_window)
        md_window = md_window[finite]
        z_window = z_window[finite]
        cal_window = cal_window[finite]
        if len(cal_window) < 20:
            continue
        dmd = md_window - float(md_window.mean())
        neg_dz = -(z_window - float(z_window.mean()))
        n_eff = max(width / E2_GR_DECORR_FT, 1.0)
        node = int(np.argmin(np.abs(node_md - centroid)))
        score[node] += n_eff * e2_ncc_profile(
            cal_window, dmd, neg_dz, tw_tvt, tw_gr, cells
        )

    pin_cell = int(np.argmin(np.abs(cells - pin)))
    score[pin_node] = -1e9
    score[pin_node, pin_cell] = 0.0
    known_index = np.flatnonzero(known)
    tail_index = known_index[-30:]
    tail_md = hw["MD"].to_numpy(float)[tail_index]
    tail_z = hw["Z"].to_numpy(float)[tail_index]
    tail_tvt = hw["TVT_input"].to_numpy(float)[tail_index]
    dm_tail = np.diff(tail_md)
    valid_tail = dm_tail > 0
    if int(valid_tail.sum()) >= 3:
        initial_rate = float(np.median(
            (np.diff(tail_tvt)[valid_tail] + np.diff(tail_z)[valid_tail]) / dm_tail[valid_tail]
        ))
    else:
        initial_rate = 0.0
    track = rate_viterbi(score, node_md, node_z, cells, initial_rate)
    # The OOF emitter decodes its score only on rows with finite GR and Z, then
    # interpolates that track back to every finite MD eval row. Preserve that
    # distinction. A missing GR row still receives a legal path estimate.
    output_rows = np.isfinite(md_all)
    predicted = np.interp(md_all[output_rows], node_md, track)
    return {
        int(row): float(value - pin)
        for row, value in zip(eval_index[output_rows], predicted)
    }

# %% [markdown]
# ## 7. Strong PF predictor (pf12_beam2) + multi-model divergence residual stack
#
# Three divergence-augmented residual models are evaluated on their own training
# feature order. Their averaged residual predictions are combined by the Ridge
# meta model, then blended with the live particle filter residual.

# %%
# --- Editable stack/PF blend weights -----------------------------------
# Tuned on the full 773-well OOF for THIS meta (the median divergence pair +
# realmlp NN) under SG + U projection. The basin is flat: 0.76 -> 8.7300,
# 0.78 -> 8.7281, 0.80 -> 8.7281, 0.82 -> 8.7298, a 0.0019 ft spread against a
# 0.136 ft LB nondeterminism floor, so 0.76 is taken and not re-tuned. The 0.74
# above was tuned for the DIV3 stack this kernel no longer runs.
W_STACK, W_PF = 0.88, 0.12
PF_N_SEEDS = 128
PF_N_PARTICLES = 600
print(f"Blend weights: W_STACK={W_STACK}  W_PF={W_PF}  PF seeds={PF_N_SEEDS}")

# Strong PF predictor. The visible-well contact branch is intentionally disabled.
pf_pred_map = {}  # (well_id, hw_row_index) -> selector absolute TVT
pf_estimator_maps = {
    "pf_selector": {},
    "pf_pf12_beam2": {},
    "pf_beam_oof": {},
    "pf_median": {},
    "pf_scale5": {},
    "pf_scale8": {},
}
# The CNN rides along in the PF loop: it needs exactly the `hw`/`tw` frames already being
# read, so this costs one forward pass per well and no extra file IO.

# --- GRU sequence backbone live inference --------------------------------
# seqalt_gru_psr4avg9: the MEAN of THREE seed replicates of the arm the retune kernel shipped.
# Seq1DNet with its dilated-TCN horizontal encoder swapped for a bidirectional GRU, trained with
# the COMPOUND recipe: --ps-resample 4 --gr-filter 9 --w-shape 2.0 --w-global 0.25 --epochs 160,
# over the otherwise frozen --drift-mode resid --w-local 5.0 --w-smooth 0.0. Seeds 42, 7 and 1337;
# SageMaker jobs rogii-seqalt-gru-fall-{s42-2026-08-03-17-32-28-966,
# s7-2026-08-03-21-22-44-666, s1337-2026-08-03-21-22-51-729}. Every hyperparameter except --seed is
# identical across the three, checked against each job's own metrics.json by
# scripts/stage_seqalt_gru_multiseed_2026_08_04.py rather than assumed.
#
# Averaging is the lever, not the recipe. The resampling axis is saturated: psr2 at 160, psr4 at
# 160, psr8 at 80 and psr4 at 240 span 0.015 ft against the null, inside this family's own 0.059
# same-recipe SEED spread. So the seed is what is left, and averaging removes the draw instead of
# betting on which draw was lucky. Standalone 9.3799 against 10.0939 (s42), 10.0617 (s7) and
# 10.2221 (s1337); pairwise seed error correlations 0.804, 0.771, 0.788. Under the retuned
# post-processing cell the seven-base path goes 7.58714 -> 7.42651 uncorrected.
#
# FIFTEEN checkpoints ship, not five, and the loader below is unchanged. load_fold_models RECURSES
# from the dataset root and predict_well_drift averages every model it is handed, so 3 seeds x 5
# folds under one root, one subdirectory per seed so no filename collides, IS the seed-averaged
# base. scripts/stage_seqalt_gru_multiseed_2026_08_04.py asserts the shipped loader returns 15.
# The OOF this was banked from uses the other convention, each well predicted by the mean over the
# three seeds of its OWNING fold, which is the same convention cnn_1d_v1_avg3 ships under.
#
# The architecture is IDENTICAL to seqalt_gru_v1_s42: train_seq_alt.build_model reads
# only dim, heads, backbone, enc-layers, enc-dropout, resid-scale, drift-mode and
# max_len, and w-shape and w-global enter compute_loss alone and ps-resample only chooses
# the training Dataset class inside train_fold, both after build_model has already run. So the loader below is unchanged, but the
# INPUT is NOT: --gr-filter 9 sets Seq1DConfig.gr_filter, the Savitzky-Golay window that
# build_seq1d_sample applies to GR before the encoder sees it, against the dataclass default of
# 50 that the two earlier GRU ports shipped. It is therefore written out explicitly in
# load_seqalt and cross-checked there against the training job own metrics.json, because a
# defaulted config would feed the network a differently smoothed log than it was trained on:
# the kernel would load strictly, run to completion and be quietly wrong.
#
# The prediction path is the CNN's own predict_well_drift, re-exported rather than
# copied. Verified against the banked OOF by
# scripts/verify_seqalt_gru_psr4avg9_inference_2026_08_04.py, reading only the six columns a
# hidden test well exposes.
SEQALT_PKG_MARKER = "seq_alt/kernel_infer.py"


def find_seqalt_root() -> Path:
    """Locate the dataset holding the seq_alt package and its fold checkpoints."""
    candidates = [
        Path("/kaggle/input/rogii-seqalt-gru-psr4avg9"),
        Path("/kaggle/input/datasets/wguesdon/rogii-seqalt-gru-psr4avg9"),
        Path.cwd() / "kaggle_datasets" / "rogii-seqalt-gru-psr4avg9",
        Path.cwd().parent / "kaggle_datasets" / "rogii-seqalt-gru-psr4avg9",
    ]
    for root in candidates:
        if (root / SEQALT_PKG_MARKER).is_file():
            return root.resolve()
    kaggle_root = Path("/kaggle/input")
    if kaggle_root.is_dir():
        for match in kaggle_root.rglob(SEQALT_PKG_MARKER):
            return match.parent.parent.resolve()
    raise FileNotFoundError(f"Could not locate {SEQALT_PKG_MARKER}.")


def load_seqalt() -> tuple[object, list, object]:
    """Import the shipped seq_alt package and load every fold checkpoint.

    Returns:
        Tuple ``(config, fold_models, predict_well_drift)``.
    """
    root = find_seqalt_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # seq_alt.model imports cnn_1d.model, so the cnn1d root must already be importable.
    find_cnn1d_root()
    from cnn_1d.dataset import Seq1DConfig
    from seq_alt.kernel_infer import load_fold_models, predict_well_drift

    # This arm was NOT trained at the Seq1DConfig defaults: gr_filter=9.
    # gr_filter reaches build_seq1d_sample and changes the model's INPUT rather than
    # only its training loss, so a defaulted config here would feed the network a
    # differently prepared log than it was trained on. It loads strictly, runs to
    # completion and is wrong. Written out explicitly and cross-checked below against
    # the config the training job itself transmitted.
    cfg = Seq1DConfig(gr_filter=9)
    models = load_fold_models(root, n_features=cfg.n_features, max_len=cfg.seq_len)
    trained = sorted(root.rglob("metrics.json"))
    want_cfg = {'gr_filter': 9}
    if trained:
        hp = json.loads(trained[0].read_text()).get("hyperparameters", {})
        for _f, _want in want_cfg.items():
            _got = hp.get(_f.replace("_", "-"))
            if _got is not None and type(_want)(_got) != _want:
                raise RuntimeError(
                    f"seqalt config {_f}={_want} but the shipped checkpoints were "
                    f"trained with {_f}={_got}")
        print(f"[seqalt-gru] trained config cross-checked against {trained[0]}")
    else:
        print("[seqalt-gru] WARNING: no metrics.json beside the checkpoints; the "
              "non-default config could not be cross-checked")
    print(f"[seqalt-gru] root {root}; {len(models)} fold models; seq_len {cfg.seq_len}; non-default {want_cfg}")
    return cfg, models, predict_well_drift


CNN1D_CFG, CNN1D_MODELS, cnn1d_predict_well = load_cnn1d()
SEQALT_CFG, SEQALT_MODELS, seqalt_predict_well = load_seqalt()
cnn1d_map: dict[tuple[str, int], float] = {}
seqalt_map: dict[tuple[str, int], float] = {}
e2_map: dict[tuple[str, int], float] = {}
ratehmm_map: dict[tuple[str, int], float] = {}
ratehmm_public_map: dict[tuple[str, int], float] = {}
ratecoupled_map: dict[tuple[str, int], float] = {}

t_pf = time.time()
for k, path in enumerate(test_paths, start=1):
    wid = path.name.split("__")[0]
    hw = pd.read_csv(path)
    tw = pd.read_csv(path.parent / f"{wid}__typewell.csv")
    ev = hw["TVT_input"].isna().to_numpy()
    ridx = np.where(ev)[0]
    if ridx.size == 0:
        continue
    for ri, drift in cnn1d_predict_well(CNN1D_MODELS, hw, tw, wid, cfg=CNN1D_CFG).items():
        cnn1d_map[(wid, int(ri))] = float(drift)
    for ri, drift in seqalt_predict_well(SEQALT_MODELS, hw, tw, wid, cfg=SEQALT_CFG).items():
        seqalt_map[(wid, int(ri))] = float(drift)
    legal = [c for c in ["MD", "X", "Y", "Z", "GR", "TVT_input"] if c in hw.columns]
    for ri, drift in predict_e2_residuals(hw[legal].copy(), tw).items():
        e2_map[(wid, int(ri))] = float(drift)
    for ri, drift in predict_e2_ratehmm_posterior_residuals(hw[legal].copy(), tw).items():
        ratehmm_map[(wid, int(ri))] = float(drift)
    for ri, drift in predict_e2_ratehmm_public_residuals(hw[legal].copy(), tw).items():
        ratehmm_public_map[(wid, int(ri))] = float(drift)
    for ri, drift in predict_e2_ratecoupled_residuals(hw[legal].copy(), tw).items():
        ratecoupled_map[(wid, int(ri))] = float(drift)
    np.random.seed(42)
    pf_by_scale = run_pf_lik_ensemble_scales(
        hw[legal].copy(), tw, n_particles=PF_N_PARTICLES, n_seeds=PF_N_SEEDS
    )
    tvt_beam = run_beam_ensemble(hw[legal].copy(), tw)
    last_known = hw["TVT_input"].dropna()
    last_known_tvt = float(last_known.iloc[-1]) if len(last_known) else float(np.nanmean(pf_by_scale["pf_mean"]))
    _, variant, _, _ = selector_well_code(hw[legal].copy())
    pred_full = apply_selector_variant(variant, pf_by_scale, tvt_beam, last_known_tvt)
    pf12_beam2 = 0.8 * pf_by_scale["pf_scale_12"] + 0.2 * tvt_beam
    pf_median = np.median(
        np.vstack([
            pf_by_scale["pf_scale_3"],
            pf_by_scale["pf_scale_5"],
            pf_by_scale["pf_scale_8"],
            pf_by_scale["pf_scale_12"],
        ]),
        axis=0,
    )
    for ri in ridx:
        key = (wid, int(ri))
        pf_pred_map[key] = float(pred_full[ri])
        pf_estimator_maps["pf_selector"][key] = float(pred_full[ri])
        pf_estimator_maps["pf_pf12_beam2"][key] = float(pf12_beam2[ri])
        pf_estimator_maps["pf_beam_oof"][key] = float(tvt_beam[ri])
        pf_estimator_maps["pf_median"][key] = float(pf_median[ri])
        pf_estimator_maps["pf_scale5"][key] = float(pf_by_scale["pf_scale_5"][ri])
        pf_estimator_maps["pf_scale8"][key] = float(pf_by_scale["pf_scale_8"][ri])
    if k % 10 == 0:
        print(f"  PF {k}/{len(test_paths)} wells ({time.time() - t_pf:.0f}s)", flush=True)
print(f"PF pass done in {time.time() - t_pf:.0f}s; {len(pf_pred_map)} eval rows")

# %%
last_known_test = test_df["last_known_tvt"].to_numpy(dtype=np.float64)
pf_pred_aligned = np.array([
    pf_pred_map.get((str(w), int(r)), np.nan)
    for w, r in zip(test_df["well_id"].to_numpy(), test_df["row_index"].to_numpy())
], dtype=np.float64)
n_miss = int(np.sum(~np.isfinite(pf_pred_aligned)))
if n_miss:
    print(f"WARNING: {n_miss} test rows missing a PF prediction; falling back to last_known.")
    pf_pred_aligned = np.where(np.isfinite(pf_pred_aligned), pf_pred_aligned, last_known_test)

pf_resid = pf_pred_aligned - last_known_test

# Build live divergence features. All estimator columns are residuals to
# last_known_tvt, matching scripts/build_divergence_features.py.
def _align_abs_map(name: str) -> np.ndarray:
    vals = np.array([
        pf_estimator_maps[name].get((str(w), int(r)), np.nan)
        for w, r in zip(test_df["well_id"].to_numpy(), test_df["row_index"].to_numpy())
    ], dtype=np.float64)
    return np.where(np.isfinite(vals), vals, last_known_test) - last_known_test


est_df = pd.DataFrame({
    "pf_selector": _align_abs_map("pf_selector"),
    "pf_pf12_beam2": _align_abs_map("pf_pf12_beam2"),
    "pf_beam_oof": _align_abs_map("pf_beam_oof"),
    "pf_median": _align_abs_map("pf_median"),
    "pf_scale5": _align_abs_map("pf_scale5"),
    "pf_scale8": _align_abs_map("pf_scale8"),
    "pf_ancc": test_df["pf_ancc"].to_numpy(dtype=np.float64) - last_known_test,
    "pf_z": test_df["pf_z"].to_numpy(dtype=np.float64) - last_known_test,
    "plane_buda": test_df["tvtF_BUDA"].to_numpy(dtype=np.float64) - last_known_test,
    "plane_egfdl": test_df["tvtF_EGFDL"].to_numpy(dtype=np.float64) - last_known_test,
    "dense50": test_df["tvt_dense50_d"].to_numpy(dtype=np.float64),
    "dense": test_df["tvt_dense_d"].to_numpy(dtype=np.float64),
    "beam_mean": test_df["beam_mean_d"].to_numpy(dtype=np.float64),
    "beam_cons": test_df["beam_cons_d"].to_numpy(dtype=np.float64),
})
div_df = add_divergence_columns(est_df)
if list(div_df.columns) != list(DIVERGENCE_COLUMNS):
    raise RuntimeError("Divergence column order mismatch.")
test_df_div = pd.concat([test_df.reset_index(drop=True), div_df.reset_index(drop=True)], axis=1)
print(f"Divergence frame: {div_df.shape}; finite={np.isfinite(div_df.to_numpy()).all()}")

model_residuals: dict[str, np.ndarray] = {}
for model_name in MODEL_NAMES:
    feature_names = json.loads((DIV_ARTIFACTS_ROOT / f"feature_names_{model_name}.json").read_text())
    X_test = select_feature_columns(test_df_div, feature_names)
    fitted_models = joblib.load(DIV_ARTIFACTS_ROOT / f"models_{model_name}.pkl")
    if not isinstance(fitted_models, list) or len(fitted_models) == 0:
        raise TypeError(f"models_{model_name}.pkl must contain a non-empty list of fitted models.")
    residual = np.zeros(n_test, dtype=np.float64)
    for model in fitted_models:
        residual += np.asarray(model.predict(X_test), dtype=np.float64)
    residual /= len(fitted_models)
    model_residuals[model_name] = residual
    print(f"[{model_name}] residual mean {residual.mean():+.4f} std {residual.std():.4f}")
    del fitted_models, X_test
    gc.collect()

# realmlp NN residual (genuinely different family; see predict_realmlp_residual).
model_residuals["realmlp_v1_s42"] = predict_realmlp_residual(test_df_div)

# CNN-1D whole-well residual, gathered during the PF pass above. A well the sample builder
# rejects contributes no key and falls back to 0.0, i.e. carry-forward for that row, which is
# the same convention the PF alignment uses a few cells up.
cnn1d_aligned = np.array([
    cnn1d_map.get((str(w), int(r)), np.nan)
    for w, r in zip(test_df["well_id"].to_numpy(), test_df["row_index"].to_numpy())
], dtype=np.float64)
n_cnn_miss = int(np.sum(~np.isfinite(cnn1d_aligned)))
if n_cnn_miss:
    print(f"WARNING: {n_cnn_miss} test rows have no CNN-1D prediction; using 0.0 residual.")
    cnn1d_aligned = np.where(np.isfinite(cnn1d_aligned), cnn1d_aligned, 0.0)
model_residuals["cnn_1d_v1_avg3"] = cnn1d_aligned

seqalt_aligned = np.array([
    seqalt_map.get((str(w), int(r)), np.nan)
    for w, r in zip(test_df["well_id"].to_numpy(), test_df["row_index"].to_numpy())
], dtype=np.float64)
n_seqalt_miss = int(np.count_nonzero(~np.isfinite(seqalt_aligned)))
if n_seqalt_miss:
    print(f"WARNING: {n_seqalt_miss} test rows have no GRU prediction; using 0.0 residual.")
    seqalt_aligned = np.where(np.isfinite(seqalt_aligned), seqalt_aligned, 0.0)
model_residuals["seqalt_gru_psr4avg9"] = seqalt_aligned
print(
    f"[seqalt-gru] residual mean {seqalt_aligned.mean():+.4f} "
    f"std {seqalt_aligned.std():.4f} "
    f"covered {len(seqalt_map)}/{len(seqalt_aligned)}"
)
print(f"[cnn1d] residual mean {cnn1d_aligned.mean():+.4f} std {cnn1d_aligned.std():.4f}")

e2_aligned = np.array([
    e2_map.get((str(w), int(r)), 0.0)
    for w, r in zip(test_df["well_id"].to_numpy(), test_df["row_index"].to_numpy())
], dtype=np.float64)
model_residuals["untried19_lam0.01_sg5"] = e2_aligned
print(
    f"[e2] residual mean {e2_aligned.mean():+.4f} std {e2_aligned.std():.4f} "
    f"covered {len(e2_map)}/{len(e2_aligned)}"
)

ratehmm_aligned = np.array([
    ratehmm_map.get((str(w), int(r)), 0.0)
    for w, r in zip(test_df["well_id"].to_numpy(), test_df["row_index"].to_numpy())
], dtype=np.float64)
model_residuals["untried19_ratehmm_posterior_sg5"] = ratehmm_aligned

ratehmm_public_aligned = np.array([
    ratehmm_public_map.get((str(w), int(r)), 0.0)
    for w, r in zip(test_df["well_id"].to_numpy(), test_df["row_index"].to_numpy())
], dtype=np.float64)
model_residuals["untried19_ratehmm_public_sg5"] = ratehmm_public_aligned
print(
    f"[ratehmm-public] residual mean {ratehmm_public_aligned.mean():+.4f} "
    f"std {ratehmm_public_aligned.std():.4f} "
    f"covered {len(ratehmm_public_map)}/{len(ratehmm_public_aligned)}"
)
print(
    f"[ratehmm] residual mean {ratehmm_aligned.mean():+.4f} std {ratehmm_aligned.std():.4f} "
    f"covered {len(ratehmm_map)}/{len(ratehmm_aligned)}"
)

ratecoupled_aligned = np.array([
    ratecoupled_map.get((str(w), int(r)), 0.0)
    for w, r in zip(test_df["well_id"].to_numpy(), test_df["row_index"].to_numpy())
], dtype=np.float64)
print(
    f"[ratecoupled] residual mean {ratecoupled_aligned.mean():+.4f} "
    f"std {ratecoupled_aligned.std():.4f} covered {len(ratecoupled_map)}/{len(ratecoupled_aligned)}"
)
test_df_div["untried19_e2_delta"] = e2_aligned
test_df_div["untried19_ratehmm_posterior_delta"] = ratehmm_aligned
test_df_div["untried19_ratecoupled_delta"] = ratecoupled_aligned
ratecoupled_features = json.loads(
    (RATECOUPLED_ARTIFACTS_ROOT / RATECOUPLED_FEATURE_NAME).read_text()
)
X_ratecoupled = select_feature_columns(test_df_div, ratecoupled_features)
ratecoupled_models = joblib.load(RATECOUPLED_ARTIFACTS_ROOT / RATECOUPLED_MODEL_NAME)
if not isinstance(ratecoupled_models, list) or len(ratecoupled_models) != 5:
    raise TypeError("Rate coupled artifact must contain five fold LightGBM models.")
ratecoupled_residual = np.mean(
    [np.asarray(model.predict(X_ratecoupled), dtype=np.float64) for model in ratecoupled_models],
    axis=0,
)
model_residuals["lgbmede2ratecoupledfeature_2026_08_01"] = ratecoupled_residual
print(
    f"[ratecoupled-gbdt] residual mean {ratecoupled_residual.mean():+.4f} "
    f"std {ratecoupled_residual.std():.4f}"
)

# 7-base residual-space Ridge meta for the existing kernel bases plus the verified E2
# matcher. Coefficients are a positive Ridge with alpha 1.0 fit on all 3,783,989 OOF
# rows. `measure_e2_portable_stack_2026_07_30.py` reproduces this full fit at 8.4752
# and the outer fold protocol at 8.5105. The kernel uses the full fit coefficients,
# as every previous submission kernel does.
RIDGE_MODELS = [
    "lgbmede2ratecoupledfeature_2026_08_01",
    "realmlp_v1_s42",
    "cnn_1d_v1_avg3",
    "untried19_lam0.01_sg5",
    "untried19_ratehmm_posterior_sg5",
    "untried19_ratehmm_public_sg5",
    "seqalt_gru_psr4avg9"
]


# NINE-SEED AVERAGE, SINGLE-PATH. The GRU column is the mean of nine draws of the
# shipped recipe cmp_psr4_gr9_w2_e160 rather than three, and this kernel carries NO
# estimate-space datum averaging. It is the 55231323 pipeline with six more seeds.
# Ramped pooled 7.267606 -> 7.211347, -0.0563 ft. Its reason to exist is that every
# line of its inference code has already returned a public score (6.731), so it is the
# candidate to reach for if the multipath inference code turns out to misbehave on the
# rerun. On CV alone the multipath nine-seed kernel is better, 7.159317.
RIDGE_COEF = np.array(
    [
        0.5484621704458936,
        0.09262411471290159,
        0.0,
        0.022625893807875976,
        0.0,
        0.0162329583640758,
        0.48343294272251075,
    ],
    dtype=np.float64,
)


RIDGE_INTERCEPT = -0.20737934191414675


missing_bases = [name for name in RIDGE_MODELS if name not in model_residuals]
if missing_bases:
    raise KeyError(f"Missing residual bases for ridge stack: {missing_bases}")
stack_matrix = np.column_stack([model_residuals[name] for name in RIDGE_MODELS])
stack_residual = stack_matrix @ RIDGE_COEF + RIDGE_INTERCEPT
print(f"[ridge] 7-base intercept={RIDGE_INTERCEPT:+.4f} coef={RIDGE_COEF.tolist()}")
print(f"[ridge] residual mean {stack_residual.mean():+.4f} std {stack_residual.std():.4f}")

final_residual = W_STACK * stack_residual + W_PF * pf_resid
prediction_tvt = (last_known_test + final_residual).astype(np.float32)
print(f"Stack-only TVT mean {(last_known_test + stack_residual).mean():.2f}")
print(f"Blend  TVT mean {prediction_tvt.mean():.2f}  std {prediction_tvt.std():.2f}")

if APPLY_SG_SMOOTHING:
    smoothed = prediction_tvt.astype(np.float64).copy()
    for wid, idx in test_df.groupby("well_id", sort=False).groups.items():
        loc = np.asarray(list(idx), dtype=int)
        order = np.argsort(test_df.iloc[loc]["MD"].to_numpy(dtype=np.float64))
        ordered_loc = loc[order]
        vals = smoothed[ordered_loc]
        if len(vals) > SG_POLY + 1:
            win = min(SG_WINDOW, len(vals) if len(vals) % 2 == 1 else len(vals) - 1)
            if win > SG_POLY:
                smoothed[ordered_loc] = savgol_filter(vals, win, SG_POLY)
    prediction_tvt = smoothed.astype(np.float32)
    print(f"SG-smoothed final TVT mean {prediction_tvt.mean():.2f} std {prediction_tvt.std():.2f}")

if APPLY_U_PROJECTION:
    prediction_tvt, projection_report = project_prediction_by_well(
        prediction=prediction_tvt,
        frame=test_df,
        test_dir=TEST_DIR,
        degree=U_PROJECTION_DEGREE,
        blend_weight=U_PROJECTION_BLEND,
        robust_iters=U_PROJECTION_ROBUST_ITERS,
        robust_c=U_PROJECTION_ROBUST_C,
    )
    projection_report.to_csv("projection_report.csv", index=False)
    if len(projection_report):
        ok_count = int(projection_report["projected"].fillna(False).astype(bool).sum())
        ok_report = projection_report[projection_report["projected"].fillna(False).astype(bool)]
        mean_adj = float(ok_report["mean_abs_adjustment"].mean()) if len(ok_report) else float("nan")
    else:
        ok_count = 0
        mean_adj = float("nan")
    print(
        f"U projection projected {ok_count}/{len(projection_report)} wells; "
        f"mean_abs_adjustment={mean_adj:.3f}; "
        f"final mean {prediction_tvt.mean():.2f} std {prediction_tvt.std():.2f}"
    )


# %% [markdown]
# ## 7a. Trust-gated datum module, INLINED verbatim from src/trust_datum.py
#
# Generated by scripts/build_setk_keeponly_kernel_2026_08_03.py. Do not hand-edit.

# %%
"""Trust-gated typewell datum correction: ONE implementation, used by CV and by inference.

This module exists so the measured artifact and the shipped artifact cannot drift apart. Both
`AGENTS.md` traps about kernel ports were caused by an inference path that differed from the
path whose number justified shipping it, so nothing here is reimplemented anywhere else.

The mechanism, and why it is legal at test time
-----------------------------------------------
`notes/emission_certificate_2026_08_02.md` establishes that the supplied typewell, read at a
candidate TVT and fitted affinely to the observed GR, is a sharp discriminator: at the true
path its datum argmax lands within 1 ft on 76 percent of wells against 0.001 contrast for a
rolled typewell. Inside a short MD window the path error is close to a constant, so the window's
profile over rigid shifts carries a local offset estimate. Most windows are wrong, but the
confident ones are right about four times as often as chance, and confidence is readable
WITHOUT the label from how isolated the profile's peak is.

Every input is exposed by the test set: the well's own GR and MD, the pipeline's own predicted
path, and the supplied typewell. No truth, no other well, no training-set lookup.

The absolute threshold, which is a correctness fix and not a tuning choice
-------------------------------------------------------------------------
The CV probes gated windows at a QUANTILE of the isolation statistic over the whole population
of windows. That is not deployable: at inference the population is a different, much smaller set
of wells, so the same quantile maps to a different cut and the estimator is not even
well-defined per well. `ISOLATION_THRESHOLD` is therefore an ABSOLUTE cut fitted on train and
shipped as a constant, and `scripts/probe_trust_datum_absolute_2026_08_02.py` re-measures the
whole estimator under it rather than assuming the substitution is free.
"""



GRID_STEP = 0.5
# SMOOTH is 1, i.e. no smoothing, and that is a measured choice rather than a default. It was 5,
# a 2.5 ft boxcar, inherited from the probe this module grew out of. `AGENTS.md` records the GR
# correlation length as 1 to 2 ft, so a 2.5 ft boxcar smooths across the entire informative
# scale. `reports/analyse_reference_resolution_2026_08_02.json` measures the cost: the top-trust
# window hit rate at L=400 runs 0.379 at 0.5 ft against 0.337 at 2.5 ft and 0.229 at 4.5 ft,
# with the ROLLED control flat at 0.108 to 0.121 at every resolution. End to end the change moves
# the fully cross-fitted estimator from correlation 0.2000 to 0.2718 and CV 8.1326 to 8.0435.
SMOOTH = 1
MAX_SHIFT = 16.0
SHIFT_STEP = 0.25
ISOLATION_FT = 4.0
MIN_WINDOW_ROWS = 60

# The shipped configuration. Refrozen 2026-08-02 at the fine reference resolution, from the
# cross-fitted sweep in `scripts/probe_trust_datum_absolute_2026_08_02.py`, whose modal pick was
# the `mid` length set at trust quantile 0.7 with temperature 50 and prior sd 2 in four of five
# folds. The thresholds are the ABSOLUTE isolation values that quantile corresponds to over all
# 773 train wells, one per window length, so a test well's correction depends only on its own
# data and never on which other wells are scored beside it.
SHIPPED_LENGTHS: tuple[float, ...] = (300.0, 400.0, 600.0, 800.0)
SHIPPED_THRESHOLDS: dict[float, float] = {
    300.0: 0.03486111760139465,
    400.0: 0.043377211689949034,
    600.0: 0.052142107486724834,
    800.0: 0.05934260189533233,
}
SHIPPED_TEMP = 50.0
SHIPPED_SIGMA = 2.0
SHIPPED_LAMBDA = 0.875


def shipped_datum_shift(md: np.ndarray, gr: np.ndarray, pred: np.ndarray,
                        tw_tvt: np.ndarray, tw_gr: np.ndarray) -> float:
    """The frozen configuration, which is what a kernel calls.

    Each window length is gated at its own absolute isolation threshold and contributes one
    estimate; those are averaged and shrunk by the cross-fitted global scalar.

    Args:
        md: Measured depth of the eval rows, ascending.
        gr: Observed GR of the eval rows.
        pred: Predicted TVT of the eval rows.
        tw_tvt: Typewell TVT.
        tw_gr: Typewell GR.

    Returns:
        The shift in feet to ADD to ``pred``, zero when no window is trustworthy.
    """
    per_length = []
    for length in SHIPPED_LENGTHS:
        v = datum_shift(md, gr, pred, tw_tvt, tw_gr, (length,),
                        SHIPPED_THRESHOLDS[length], SHIPPED_TEMP, SHIPPED_SIGMA, 1.0)
        if v != 0.0:
            per_length.append(v)
    if not per_length:
        return 0.0
    return SHIPPED_LAMBDA * float(np.mean(per_length))


def build_reference(tvt: np.ndarray, gr: np.ndarray, step: float = GRID_STEP,
                    smooth: int = SMOOTH) -> tuple[np.ndarray, np.ndarray]:
    """Resample a log onto a uniform TVT grid to form a reference.

    Args:
        tvt: Reference TVT per row.
        gr: Reference GR per row.
        step: Grid spacing in feet.
        smooth: Boxcar width in grid cells; 1 disables smoothing.

    Returns:
        Tuple ``(grid_tvt, grid_gr)``; empty arrays when the log is unusable.
    """
    good = np.isfinite(tvt) & np.isfinite(gr)
    if good.sum() < 30:
        return np.empty(0), np.empty(0)
    t, g = tvt[good], gr[good]
    lo, hi = t.min(), t.max()
    if hi - lo < 20.0:
        return np.empty(0), np.empty(0)
    idx = np.clip(np.round((t - lo) / step).astype(np.int64), 0, None)
    n = idx.max() + 1
    cnt = np.bincount(idx, minlength=n).astype(np.float64)
    tot = np.bincount(idx, weights=g, minlength=n)
    grid = lo + np.arange(n) * step
    filled = cnt > 0
    if filled.sum() < 20:
        return np.empty(0), np.empty(0)
    vals = np.interp(grid, grid[filled], tot[filled] / cnt[filled])
    if smooth > 1:
        vals = np.convolve(vals, np.ones(smooth) / smooth, mode="same")
    return grid, vals


def window_profile(pred: np.ndarray, gr: np.ndarray, grid: np.ndarray, vals: np.ndarray,
                   shifts: np.ndarray) -> np.ndarray | None:
    """Affine R squared of the reference against GR over rigid shifts of the path.

    The row set is every row that stays inside the reference support for every shift on the
    grid, so all shifts are scored on identical rows and the comparison is a likelihood ratio
    rather than a count of surviving rows.

    Args:
        pred: Predicted TVT for the window's rows.
        gr: Observed GR for those rows.
        grid: Reference TVT grid, ascending and uniform.
        vals: Reference GR on that grid.
        shifts: Candidate shifts in feet.

    Returns:
        R squared per shift, or None when the window is unusable.
    """
    lo, hi, step = grid[0], grid[-1], grid[1] - grid[0]
    dev = float(np.abs(shifts).max())
    keep = np.isfinite(pred) & np.isfinite(gr) & (pred - dev >= lo) & (pred + dev <= hi)
    if keep.sum() < MIN_WINDOW_ROWS:
        return None
    p, y = pred[keep], gr[keep]
    yc = y - y.mean()
    syy = float(yc @ yc)
    if syy < 1e-9:
        return None
    pos = (p[None, :] + shifts[:, None] - lo) / step
    i0 = np.clip(pos.astype(np.int64), 0, len(vals) - 2)
    w = pos - i0
    ref = vals[i0] * (1.0 - w) + vals[i0 + 1] * w
    ref -= ref.mean(axis=1, keepdims=True)
    num = ref @ yc
    den = np.einsum("ij,ij->i", ref, ref)
    return np.where(den > 1e-12, num * num / (den * syy), np.nan)


def peak_isolation(prof: np.ndarray, shifts: np.ndarray,
                   isolation_ft: float = ISOLATION_FT) -> float:
    """Gap between a profile's peak and the best competing peak further than a radius.

    A single dominant peak is a confident match; two comparable peaks are an alias pair and
    the window should not be trusted.

    Args:
        prof: R squared per shift.
        shifts: Candidate shifts in feet.
        isolation_ft: Exclusion radius around the peak, in feet.

    Returns:
        The isolation gap.
    """
    p = np.nan_to_num(prof, nan=-1.0)
    j = int(p.argmax())
    away = np.abs(shifts - shifts[j]) > isolation_ft
    return float(p[j] - np.where(away, p, -np.inf).max())


def posterior_mean(prof: np.ndarray, shifts: np.ndarray, temp: float,
                   sigma: float) -> float:
    """Posterior mean shift for one window under a Gaussian prior.

    Args:
        prof: R squared per shift.
        shifts: Candidate shifts in feet.
        temp: Likelihood temperature.
        sigma: Prior standard deviation in feet.

    Returns:
        The posterior mean shift.
    """
    ll = -np.log(1.0 - np.clip(prof, -0.999, 0.999))
    ll = np.where(np.isfinite(ll), ll, 0.0)
    logp = temp * ll - (shifts ** 2) / (2 * sigma ** 2)
    logp -= logp.max()
    p = np.exp(logp)
    return float(p @ shifts / p.sum())


def datum_shift(md: np.ndarray, gr: np.ndarray, pred: np.ndarray,
                tw_tvt: np.ndarray, tw_gr: np.ndarray, lengths: tuple[float, ...],
                threshold: float, temp: float, sigma: float, lam: float,
                profile_dtype: type | None = None) -> float:
    """The whole estimator for ONE well: a single rigid shift to add to the path.

    Args:
        md: Measured depth of the eval rows, ascending.
        gr: Observed GR of the eval rows.
        pred: Predicted TVT of the eval rows.
        tw_tvt: Typewell TVT.
        tw_gr: Typewell GR.
        lengths: Window lengths in feet; each contributes one estimate, then averaged.
        threshold: ABSOLUTE isolation cut; windows below it are discarded.
        temp: Likelihood temperature.
        sigma: Prior standard deviation in feet.
        lam: Global shrinkage applied to the aggregated estimate.
        profile_dtype: Round each window profile to this dtype before reading it. Used only
            by the verifier, to reproduce the float32 precision the training harness banked
            its profiles at, so a residual mismatch cannot hide behind storage rounding.

    Returns:
        The shift in feet, zero when no window is trustworthy.
    """
    grid, vals = build_reference(tw_tvt, tw_gr)
    if len(grid) < 20 or len(md) < MIN_WINDOW_ROWS:
        return 0.0
    shifts = np.arange(-MAX_SHIFT, MAX_SHIFT + 1e-9, SHIFT_STEP)
    per_length = []
    for length in lengths:
        num, den = 0.0, 0.0
        for a in np.arange(md.min(), md.max() - length * 0.5, length * 0.5):
            m = (md >= a) & (md < a + length)
            if m.sum() < MIN_WINDOW_ROWS:
                continue
            prof = window_profile(pred[m], gr[m], grid, vals, shifts)
            if prof is None or not np.isfinite(prof).any():
                continue
            if profile_dtype is not None:
                prof = prof.astype(profile_dtype).astype(np.float64)
            iso = peak_isolation(prof, shifts)
            if not np.isfinite(iso) or iso < threshold:
                continue
            w = max(iso, 1e-6)
            num += w * posterior_mean(prof, shifts, temp, sigma)
            den += w
        if den > 0:
            per_length.append(num / den)
    if not per_length:
        return 0.0
    return lam * float(np.mean(per_length))

# %% [markdown]
# ## 7b. Trust-gated typewell datum correction
#
# The correction is applied as a RAMP, not a constant. The per-well constant and slope of our
# error correlate at +0.7640, so the error is a ramp: the prediction anchors at last_known and
# drifts away roughly linearly, and a constant correction over-corrects near the anchor and
# under-corrects far from it. On THIS path, with its own profile bank rebuilt around it:
# uncorrected 7.42651, constant correction 7.30668, ramp with the coordinate and both
# constants re-chosen inside every fold 7.27039, mu positive in 5 of 5 folds. Frozen at
# lambda 0.3, mu 1.0 by the leave-one-fold-out median rule and priced by it at 7.27317;
# the pooled figure at the frozen pair is 7.26761 but that has seen all 773 wells and is
# not quoted. reports/freeze_ramp_psr4avg3_2026_08_04.json.

#
# Measured 2026-08-02. Window matches of the predicted path against the supplied typewell,
# gated on peak isolation and collapsed to one per-well datum, move the deployed protocol from
# CV 8.2251 to 8.1326 with every choice cross-fitted, 5 of 5 folds improving. Controls with the
# same selection budget: rolled typewell +0.0760, across-well shuffle +0.0062, cross-fitted
# global scalar +0.0158. Reads only this well's MD, GR, predicted path and its own typewell.
# notes/emission_certificate_2026_08_02.md.

# %%
_trust_shifts = []
_pred64 = np.asarray(prediction_tvt, dtype=np.float64).copy()
for _wid, _gidx in test_df.groupby("well_id", sort=False).groups.items():
    _loc = np.asarray(list(_gidx), dtype=int)
    _ordered = test_df.iloc[_loc].sort_values("MD")
    _oloc = _ordered.index.to_numpy(dtype=int)
    _tw_path = TEST_DIR / f"{_wid}__typewell.csv"
    if not _tw_path.exists():
        _trust_shifts.append({"well_id": _wid, "shift_ft": 0.0, "reason": "no_typewell"})
        continue
    _tw = pd.read_csv(_tw_path, usecols=["TVT", "GR"])
    _md = _ordered["MD"].to_numpy(dtype=np.float64)
    _gr = _ordered["GR"].to_numpy(dtype=np.float64)
    _p = _pred64[_oloc]
    _good = np.isfinite(_md) & np.isfinite(_gr) & np.isfinite(_p)
    if int(_good.sum()) < MIN_WINDOW_ROWS:
        _trust_shifts.append({"well_id": _wid, "shift_ft": 0.0, "reason": "too_few_rows"})
        continue
    _s = shipped_datum_shift(
        _md[_good], _gr[_good], _p[_good],
        _tw["TVT"].to_numpy(dtype=np.float64), _tw["GR"].to_numpy(dtype=np.float64))
    _ramp_lambda, _ramp_mu = 0.3, 1.0
    _x = (_md - float(np.min(_md))) / 5000.0
    _scale = _ramp_lambda + _ramp_mu * _x
    _pred64[_oloc] = _pred64[_oloc] + _s * _scale
    _trust_shifts.append({"well_id": _wid, "shift_ft": float(_s), "applied_min": float(_s * _scale.min()), "applied_max": float(_s * _scale.max()), "reason": "ok"})
prediction_tvt = _pred64.astype(np.float32)
_tr = pd.DataFrame(_trust_shifts)
_tr.to_csv("trust_datum_report.csv", index=False)
_nz = _tr[_tr["reason"] == "ok"]
print(f"Trust datum: corrected {len(_nz)}/{len(_tr)} wells; "
      f"mean |shift| {_nz['shift_ft'].abs().mean() if len(_nz) else float('nan'):.3f} ft; "
      f"max |shift| {_nz['shift_ft'].abs().max() if len(_nz) else float('nan'):.3f} ft")
print(f"Final TVT mean {prediction_tvt.mean():.2f} std {prediction_tvt.std():.2f}")

# %% [markdown]
# ## 8. Write submission

# %%
test_pred_df = pd.DataFrame({
    "id": test_df["well_id"].astype(str) + "_" + test_df["row_index"].astype(int).astype(str),
    "tvt": prediction_tvt.astype(np.float64),
})
sample_sub_full = pd.read_csv(SAMPLE_SUB_PATH)[["id"]]
submission = sample_sub_full.merge(test_pred_df, on="id", how="left")
n_missing = int(submission["tvt"].isna().sum())
if n_missing > 0:
    fallback = float(np.nanmedian(test_pred_df["tvt"]))
    print(f"WARNING: {n_missing} sample rows missing; filling median {fallback:.3f}.")
    submission["tvt"] = submission["tvt"].fillna(fallback)
submission.to_csv("submission.csv", index=False)
print(f"Wrote submission.csv ({len(submission)} rows)")
print(submission.head())
