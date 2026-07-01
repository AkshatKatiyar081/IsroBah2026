"""
05_build_tabular.py — XGBoost Tabular Feature Engineering
==========================================================
Objective 1 | Step 5 of 5 (Feature Engineering)

PURPOSE:
    Build structured tabular feature matrices for the XGBoost residual
    corrector model, aligned to the CNN-patched station-day samples.

PIPELINE (3 sub-steps):

    Step 1 → BASE TABULAR FEATURES (phase3_build_tabular):
        - Reads cpcb_xgboost_features.csv (merged CPCB + ERA5 + station metadata)
        - Aligns rows to Phase 2 QC survivors (same station + date pairs)
        - Extracts: lat, lon, elevation, doy, wind_direction, t2m_squared,
          ventilation_index, solar_radiation, BLH, and previous-day lags
        - Fills NaN lags with training column means
        - Saves tabular_means.json and ceiling_clips.json for inference
        - Output: artifacts/X_tab_{train,val}.npy  shape (N, 12)
        - Output: artifacts/y_{train,val}.npy       shape (N, 6)

    Step 2 → EXTENDED TABULAR WITH GAP LAGS (phase3b / phase3c):
        - Loads the CNN GAP embeddings (64-dim spatial context vectors)
        - Appends previous day's GAP embedding as lag features
        - Builds the full XGBoost feature matrix: GAP(64) + CNN_pred(1) + tab(12) = 77
        - Output: artifacts/X_tab_extended_with_lags_{train,val}_{pollutant}.npy

    Step 3 → RESIDUAL TARGETS:
        - Computes CNN residuals per pollutant: y_residual = y_true - cnn_pred
        - For PM2.5 and PM10, uses relative residual: (y_true - cnn_pred) / (y_true + 1)
        - Stores them ready for XGBoost training

NOTE:
    The artifacts/ directory is already present with pre-built arrays.
    Re-run this script only if you have retrained the CNNs or changed cpcb_master.csv.

USAGE:
    python 05_build_tabular.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# Ensure channels.py is importable from the same directory
# ─────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from channels import (TABULAR_FEATURES, POLLUTANT_CSV_COLS, POLLUTANTS,
                      N_TABULAR, GAP_DIM, N_XGB_FEATS)

# ─────────────────────────────────────────────────────────────────
# PATHS — all relative to the objective_1 folder
# ─────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
DATA_DIR     = BASE_DIR / "data"
PATCH_DIR    = BASE_DIR / "cnn_patches"
ARTIFACT_DIR = BASE_DIR / "artifacts"

FEATURES_CSV = DATA_DIR / "cpcb_xgboost_features.csv"


# ==============================================================
# STEP 1: BASE TABULAR FEATURES
# ==============================================================

def run_step1_base_tabular():
    """
    Step 1 — Build base tabular feature matrices aligned to QC-passed patches.

    Features (12 total):
        lat, lon, elevation, doy,
        lag_PM25_D1, lag_PM25_D2, lag_AOD_D1, lag_NO2_D1, lag_CO_D1,
        ventilation_index, wind_direction, t2m_squared

    Also saves:
        y_train.npy, y_val.npy       — 6-pollutant target arrays
        tabular_means.json           — training column means for inference gap fill
        ceiling_clips.json           — 2× training max for inference clipping
        y_column_order.json          — maps pollutant name → column index in y
    """
    print(f"\n{'='*60}")
    print(f"  STEP 1 — Base Tabular Features")
    print(f"{'='*60}\n")

    if not FEATURES_CSV.exists():
        print(f"[ERROR] {FEATURES_CSV} not found. Run 01_process_cpcb.py first.")
        return

    features_df = pd.read_csv(FEATURES_CSV)
    meta_train  = pd.read_csv(PATCH_DIR / "metadata_train_qc.csv")
    meta_val    = pd.read_csv(PATCH_DIR / "metadata_val_qc.csv")

    print(f"  Features CSV shape: {features_df.shape}")
    print(f"  Meta train (QC survivors): {len(meta_train)}")
    print(f"  Meta val   (QC survivors): {len(meta_val)}")

    # Normalize date formats for join
    for df in [features_df, meta_train, meta_val]:
        df["date"] = pd.to_datetime(df["date"], format="mixed").dt.strftime("%Y-%m-%d")

    def align_split(features_df, meta_df):
        """Left-merge to align features to QC patch survivors."""
        join_keys = ["station", "date"]
        merged = meta_df[join_keys].merge(features_df, on=join_keys, how="left")
        return merged

    aligned_train = align_split(features_df, meta_train)
    aligned_val   = align_split(features_df, meta_val)

    assert len(aligned_train) == len(meta_train), \
        f"Alignment mismatch: {len(aligned_train)} vs {len(meta_train)}"
    assert len(aligned_val)   == len(meta_val), \
        f"Alignment mismatch: {len(aligned_val)} vs {len(meta_val)}"

    print(f"\n  Aligned train: {len(aligned_train)} rows")
    print(f"  Aligned val  : {len(aligned_val)} rows")

    # Check which tabular features actually exist in the CSV
    available_features = [f for f in TABULAR_FEATURES if f in aligned_train.columns]
    missing_features   = [f for f in TABULAR_FEATURES if f not in aligned_train.columns]
    if missing_features:
        print(f"\n  [WARN] Missing features (will be filled with 0): {missing_features}")

    # Extract tabular feature arrays
    X_tab_train = aligned_train[available_features].values.astype(np.float32)
    X_tab_val   = aligned_val[available_features].values.astype(np.float32)

    # Fill NaN lags with training column means (never use val statistics)
    tab_means = np.nanmean(X_tab_train, axis=0)
    for i in range(X_tab_train.shape[1]):
        X_tab_train[np.isnan(X_tab_train[:, i]), i] = tab_means[i]
        X_tab_val[np.isnan(X_tab_val[:, i]), i]     = tab_means[i]

    # Save tabular means and ceiling clips for inference
    tabular_means_dict = {col: float(tab_means[j])
                          for j, col in enumerate(available_features)}
    with open(ARTIFACT_DIR / "tabular_means.json", "w") as f:
        json.dump(tabular_means_dict, f, indent=2)

    # Ceiling clips: 2× training maximum per feature (prevents runaway inference)
    ceiling_clips = {col: float(np.nanmax(X_tab_train[:, j]) * 2)
                     for j, col in enumerate(available_features)}
    with open(ARTIFACT_DIR / "ceiling_clips.json", "w") as f:
        json.dump(ceiling_clips, f, indent=2)

    # Extract target arrays (6 pollutants)
    y_col_order = {}
    y_cols = []
    for i, pollutant in enumerate(POLLUTANTS):
        col = POLLUTANT_CSV_COLS[pollutant]
        y_cols.append(col)
        y_col_order[pollutant] = i

    y_train = aligned_train[y_cols].values.astype(np.float32)
    y_val   = aligned_val[y_cols].values.astype(np.float32)

    # Save arrays
    np.save(ARTIFACT_DIR / "X_tab_train.npy", X_tab_train)
    np.save(ARTIFACT_DIR / "X_tab_val.npy",   X_tab_val)
    np.save(ARTIFACT_DIR / "y_train.npy",      y_train)
    np.save(ARTIFACT_DIR / "y_val.npy",        y_val)

    with open(ARTIFACT_DIR / "y_column_order.json", "w") as f:
        json.dump(y_col_order, f, indent=2)

    print(f"\n  X_tab_train: {X_tab_train.shape}")
    print(f"  X_tab_val  : {X_tab_val.shape}")
    print(f"  y_train    : {y_train.shape}")
    print(f"  y_val      : {y_val.shape}")
    print(f"\n  STEP 1 DONE\n")


# ==============================================================
# STEP 2: EXTENDED TABULAR WITH GAP LAGS
# ==============================================================

def run_step2_gap_lags():
    """
    Step 2 — Extend tabular features with CNN GAP embedding lag columns.

    For each pollutant, builds the full XGBoost input matrix:
        [GAP_lag1 (64)] + [CNN_pred (1)] + [X_tab (12)] = 77 features

    The GAP lag represents spatial context from the previous day's
    CNN embedding, giving XGBoost historical spatial memory.
    """
    print(f"\n{'='*60}")
    print(f"  STEP 2 — Extended Tabular + GAP Lag Features")
    print(f"{'='*60}\n")

    X_tab_train = np.load(ARTIFACT_DIR / "X_tab_train.npy")
    X_tab_val   = np.load(ARTIFACT_DIR / "X_tab_val.npy")
    y_train_all = np.load(ARTIFACT_DIR / "y_train.npy")
    y_val_all   = np.load(ARTIFACT_DIR / "y_val.npy")

    with open(ARTIFACT_DIR / "y_column_order.json") as f:
        y_col_order = json.load(f)
    with open(ARTIFACT_DIR / "gap_active_flags.json") as f:
        gap_flags = json.load(f)

    print(f"  GAP active flags: {gap_flags}\n")

    # Feature column names for SHAP interpretability
    gap_names = [f"GAP_{i}" for i in range(GAP_DIM)]
    col_names  = gap_names + ["cnn_pred"] + TABULAR_FEATURES

    for pollutant in POLLUTANTS:
        safe     = pollutant.replace(".", "")
        suffix   = "_17x17" if pollutant == "CO" else ""
        col_idx  = y_col_order[pollutant]
        gap_flag = gap_flags.get(pollutant, False)

        # Load CNN predictions and GAP embeddings for this pollutant
        cnn_pred_tr   = np.load(ARTIFACT_DIR / f"cnn_pred_train_{safe}{suffix}.npy")
        cnn_pred_vl   = np.load(ARTIFACT_DIR / f"cnn_pred_val_{safe}{suffix}.npy")
        gap_tr_all    = np.load(ARTIFACT_DIR / f"gap_train_{safe}{suffix}.npy")
        gap_vl_all    = np.load(ARTIFACT_DIR / f"gap_val_{safe}{suffix}.npy")
        valid_idx_tr  = np.load(ARTIFACT_DIR / f"valid_idx_train_{safe}{suffix}.npy")
        valid_idx_vl  = np.load(ARTIFACT_DIR / f"valid_idx_val_{safe}{suffix}.npy")

        # Subset to valid (non-NaN target) samples
        X_tr = X_tab_train[valid_idx_tr]
        X_vl = X_tab_val[valid_idx_vl]
        y_tr = y_train_all[valid_idx_tr, col_idx]
        y_vl = y_val_all[valid_idx_vl, col_idx]

        if gap_flag:
            # GAP lag: shift GAP embeddings by 1 day (use zero vector for day 0)
            gap_tr_lag = np.roll(gap_tr_all[valid_idx_tr], shift=1, axis=0)
            gap_tr_lag[0] = 0.0   # no history on first sample

            gap_vl_lag = np.roll(gap_vl_all[valid_idx_vl], shift=1, axis=0)
            gap_vl_lag[0] = 0.0

            # Full feature matrix: [GAP_lag | CNN_pred | tabular]
            X_full_tr = np.hstack([gap_tr_lag, cnn_pred_tr[:, np.newaxis], X_tr])
            X_full_vl = np.hstack([gap_vl_lag, cnn_pred_vl[:, np.newaxis], X_vl])
        else:
            # Degenerate GAP — use zero GAP to avoid noise
            zeros_tr  = np.zeros((len(X_tr), GAP_DIM), dtype=np.float32)
            zeros_vl  = np.zeros((len(X_vl), GAP_DIM), dtype=np.float32)
            X_full_tr = np.hstack([zeros_tr, cnn_pred_tr[:, np.newaxis], X_tr])
            X_full_vl = np.hstack([zeros_vl, cnn_pred_vl[:, np.newaxis], X_vl])

        np.save(ARTIFACT_DIR / f"X_tab_extended_with_lags_train_{safe}.npy", X_full_tr.astype(np.float32))
        np.save(ARTIFACT_DIR / f"X_tab_extended_with_lags_val_{safe}.npy",   X_full_vl.astype(np.float32))

        print(f"  [{pollutant}] Train: {X_full_tr.shape}, Val: {X_full_vl.shape}")

    # Save feature column names (used by SHAP for interpretability)
    with open(ARTIFACT_DIR / "extended_tab_column_names_with_lags.json", "w") as f:
        json.dump(col_names, f, indent=2)

    print(f"\n  STEP 2 DONE\n")


# ==============================================================
# ENTRY POINT
# ==============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  05_build_tabular.py — XGBoost Feature Engineering")
    print("=" * 60)

    if not (ARTIFACT_DIR / "X_tab_train.npy").exists():
        run_step1_base_tabular()
    else:
        print("\n[INFO] Base tabular arrays found in artifacts/. Skipping Step 1.")
        print("       Delete artifacts/X_tab_train.npy to re-run.")

    if not (ARTIFACT_DIR / "gap_active_flags.json").exists():
        print("\n[WARN] gap_active_flags.json not found. Run run_model.py Phase 6 (CNN training) first.")
    else:
        run_step2_gap_lags()

    print(f"\n{'='*60}")
    print(f"  05_build_tabular.py COMPLETE")
    print(f"{'='*60}")
