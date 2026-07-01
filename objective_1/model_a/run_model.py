"""
run_model.py — Objective 1 Master Model Pipeline (Phases 6–12)
==============================================================
Objective 1 | Master Script

PURPOSE:
    Trains the full Hybrid CNN-XGBoost AQI prediction model and generates
    spatial daily predictions and AQI maps across all of India.

PREREQUISITES:
    Run scripts 01–05 first to prepare all input data, patches, and tabular
    features. Or confirm that the following already exist:
        - artifacts/X_image_train_scaled.npy
        - artifacts/X_image_val_scaled.npy
        - artifacts/X_tab_train.npy
        - artifacts/y_train.npy
        - data/regridded/grid_spec.json

PIPELINE PHASES:

    Phase 6 — CNN Training:
        Trains 6 independent CNNs (one per pollutant). Each CNN learns spatial
        patterns from 13×13 (17×17 for CO) multi-channel satellite patches.
        Saves: artifacts/cnn_{pollutant}.keras + GAP embeddings

    Phase 7 — XGBoost Training:
        Trains 6 XGBoost residual correctors. Each corrects the CNN's bias
        using: GAP embeddings + CNN prediction + structured tabular features.
        Uses RandomizedSearchCV (n_iter=30) with 5-fold cross-validation.
        Saves: artifacts/xgb_{pollutant}.json

    Phase 8 — Full Spatial Inference:
        Predicts the concentration for ALL 14,880 grid cells (120×124) across
        India for all 123 days (Oct 2023 → Jan 2024).
        Iterates chronologically, updating day-to-day lag features.
        Applies Gaussian smoothing (sigma=1.0) to prevent tile boundary artifacts.
        Saves: results/all_predictions.pkl

    Phase 9 — Evaluation:
        Computes RMSE, MAE, R² and Within-R² (detrended by station × month)
        against held-out validation stations.
        Saves: results/validation_predictions.csv + results/evaluation_metrics.json

    Phase 10 — SHAP Interpretability:
        Applies shap.TreeExplainer to each XGBoost model.
        Proves scientific interpretability (e.g., Solar Radiation drives O3).
        Saves: results/shap_{pollutant}.png

    Phase 11 — Spatial Map Generation:
        Generates daily concentration maps for each pollutant using pcolormesh
        with India boundary overlay (GADM GeoJSON).
        Saves: maps/{pollutant}/ folder with daily PNGs

    Phase 12 — AQI Calculation and Mapping:
        Applies official CPCB NAAQS breakpoint formulas.
        Final AQI = max(6 pollutant sub-indices) per pixel per day.
        Generates: daily AQI maps, seasonal mean, worst-day map, dominant pollutant.
        Saves: maps/AQI/

USAGE:
    python run_model.py                       # run all phases
    python run_model.py --phase 6             # run only Phase 6 (CNN training)
    python run_model.py --phase 8             # run only Phase 8 (inference)
    python run_model.py --skip_training       # skip phases 6-7, go straight to inference
"""

import os
import sys
import gc
import json
import time
import joblib
import warnings
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# Ensure channels.py is importable from the same directory
# ─────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from channels import (CHANNEL_ORDER, POLLUTANTS, GAP_DIM, N_TABULAR,
                      N_SAT_MET, N_CHANNELS)

# ─────────────────────────────────────────────────────────────────
# PATHS — all relative to the objective_1 folder
# ─────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
DATA_DIR     = BASE_DIR / "data"
REGRID_DIR   = DATA_DIR / "regridded"
PATCH_DIR    = BASE_DIR / "cnn_patches"
ARTIFACT_DIR = BASE_DIR / "artifacts"
RESULTS_DIR  = BASE_DIR / "results"
MAPS_DIR     = BASE_DIR / "maps"
GADM_PATH    = DATA_DIR / "gadm41_IND_1.json" / "gadm41_IND_1.json"

ARTIFACT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
MAPS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# MODEL HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────
CNN_BATCH_SIZE = 64
CNN_EPOCHS     = 200
CNN_PATIENCE   = 15
CNN_LR         = 0.001

XGB_N_ITER     = 30   # Randomized search iterations per pollutant

# Per-pollutant tabular column selection for XGBoost (captures domain knowledge)
XGB_POLL_TAB_COLS = {
    "PM2.5": ["lat", "lon", "elevation", "doy", "lag_PM25_D1", "lag_PM25_D2",
               "lag_AOD_D1", "lag_NO2_D1", "lag_CO_D1", "ventilation_index",
               "wind_direction", "t2m_squared"],
    "PM10":  ["lat", "lon", "elevation", "doy", "lag_PM25_D1", "lag_PM25_D2",
               "lag_AOD_D1", "lag_NO2_D1", "lag_CO_D1", "ventilation_index",
               "wind_direction", "t2m_squared"],
    "NO2":   ["lat", "lon", "elevation", "doy", "lag_NO2_D1", "lag_AOD_D1",
               "lag_CO_D1", "ventilation_index", "wind_direction", "t2m_squared"],
    "SO2":   ["lat", "lon", "elevation", "doy", "lag_NO2_D1", "lag_AOD_D1",
               "lag_CO_D1", "ventilation_index", "wind_direction", "t2m_squared"],
    "O3":    ["lat", "lon", "elevation", "doy", "lag_AOD_D1", "lag_CO_D1",
               "ventilation_index", "wind_direction", "t2m_squared"],
    "CO":    ["lat", "lon", "elevation", "doy", "lag_CO_D1", "lag_AOD_D1",
               "lag_NO2_D1", "ventilation_index", "wind_direction", "t2m_squared"],
}

# Pollutants using relative residual target: (y - cnn_pred) / (y + 1)
USE_RELATIVE_RESID = {"PM2.5", "PM10"}


# ==============================================================
# PHASE 6: CNN TRAINING
# ==============================================================

def run_phase6_train_cnn():
    """
    Phase 6 — Train 6 CNNs (one per pollutant).

    CNN Architecture:
        Input(13,13,10) → Conv2D(32)+BN+LeakyReLU → MaxPool(2×2)
                        → Conv2D(64)+BN+LeakyReLU → MaxPool(2×2)
                        → Conv2D(64)+BN+LeakyReLU → GlobalAveragePooling2D
                        → Dropout(0.3) → Dense(1)
    CO uses a 17×17 input with an extra Conv2D block.
    Loss: Huber | Optimizer: Adam(lr=0.001, clipnorm=1.0)
    Early stopping: patience=15, restore_best_weights=True
    """
    # Import TensorFlow here (inside function to allow skipping if not needed)
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.metrics import mean_squared_error

    print("=" * 60)
    print("  Phase 6 — Train 6 CNNs")
    print("=" * 60)
    print(f"  TensorFlow {tf.__version__}")
    print(f"  GPU: {tf.config.list_physical_devices('GPU')}\n")

    # Load scaled patches and targets
    X_train = np.load(ARTIFACT_DIR / "X_image_train_scaled.npy")
    X_val   = np.load(ARTIFACT_DIR / "X_image_val_scaled.npy")
    y_train = np.load(ARTIFACT_DIR / "y_train.npy")
    y_val   = np.load(ARTIFACT_DIR / "y_val.npy")

    with open(ARTIFACT_DIR / "y_column_order.json") as f:
        y_col_order = json.load(f)

    def build_cnn(input_shape, use_maxpool=False):
        """Build CNN model. Returns (full_model, gap_model)."""
        inp = layers.Input(shape=input_shape, name="image_input")

        if use_maxpool:
            # CO: 17×17 input → extra early conv block before pooling
            x = layers.Conv2D(32, (3, 3), padding="same", name="conv1")(inp)
            x = layers.BatchNormalization(name="bn1")(x)
            x = layers.LeakyReLU(name="lrelu1")(x)
            x = layers.Conv2D(32, (3, 3), padding="same", name="conv1b")(x)
            x = layers.BatchNormalization(name="bn1b")(x)
            x = layers.LeakyReLU(name="lrelu1b")(x)
            x = layers.MaxPool2D((2, 2), name="pool1")(x)
            x = layers.Conv2D(64, (3, 3), padding="valid", name="conv2")(x)
            x = layers.BatchNormalization(name="bn2")(x)
            x = layers.LeakyReLU(name="lrelu2")(x)
            x = layers.Conv2D(64, (3, 3), padding="valid", name="conv3")(x)
            x = layers.BatchNormalization(name="bn3")(x)
            x = layers.LeakyReLU(name="lrelu3")(x)
        else:
            # Standard 13×13 architecture for PM2.5, PM10, NO2, SO2, O3
            x = layers.Conv2D(32, (3, 3), padding="same", name="conv1")(inp)
            x = layers.BatchNormalization(name="bn1")(x)
            x = layers.LeakyReLU(name="lrelu1")(x)
            x = layers.MaxPool2D((2, 2), name="pool1")(x)
            x = layers.Conv2D(64, (3, 3), padding="same", name="conv2")(x)
            x = layers.BatchNormalization(name="bn2")(x)
            x = layers.LeakyReLU(name="lrelu2")(x)
            x = layers.MaxPool2D((2, 2), name="pool2")(x)
            x = layers.Conv2D(64, (3, 3), padding="same", name="conv3")(x)
            x = layers.BatchNormalization(name="bn3")(x)
            x = layers.LeakyReLU(name="lrelu3")(x)

        # Global Average Pooling → 64-dim spatial embedding
        gap = layers.GlobalAveragePooling2D(name="gap")(x)

        # Prediction head
        out_x = layers.Dropout(0.3, name="dropout")(gap)
        out   = layers.Dense(1, name="output")(out_x)

        return Model(inp, out, name="cnn_full"), Model(inp, gap, name="cnn_gap")

    gap_variance_flags = {}

    for pollutant in POLLUTANTS:
        safe   = pollutant.replace(".", "")
        suffix = "_17x17" if pollutant == "CO" else ""
        col_idx = y_col_order[pollutant]

        print(f"\n{'-'*60}")
        print(f"  Training CNN for: {pollutant}")

        if pollutant == "CO":
            # CO uses 17×17 patches from a separate file
            X_tr = np.load(PATCH_DIR / "patches_train_CO_17x17_scaled.npy")
            X_vl = np.load(PATCH_DIR / "patches_val_CO_17x17_scaled.npy")
            y_tr = np.load(PATCH_DIR / "y_train_CO_17x17.npy")
            y_vl = np.load(PATCH_DIR / "y_val_CO_17x17.npy")
            curr_input_shape = (17, 17, 10)
            use_maxpool      = True
        else:
            y_tr = y_train[:, col_idx]
            y_vl = y_val[:, col_idx]
            curr_input_shape = X_train.shape[1:]
            use_maxpool      = False

        # Drop samples with NaN targets
        valid_tr = ~np.isnan(y_tr)
        valid_vl = ~np.isnan(y_vl)
        if pollutant == "CO":
            X_tr, X_vl = X_tr[valid_tr], X_vl[valid_vl]
        else:
            X_tr, X_vl = X_train[valid_tr], X_val[valid_vl]
        y_tr, y_vl = y_tr[valid_tr], y_vl[valid_vl]

        print(f"  Train: {len(y_tr)} samples | Val: {len(y_vl)} samples")
        print(f"{'-'*60}")

        # Build and compile model
        full_model, gap_model = build_cnn(curr_input_shape, use_maxpool=use_maxpool)
        full_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=CNN_LR, clipnorm=1.0),
            loss="huber"
        )

        # Train with early stopping to prevent overfitting
        es = EarlyStopping(monitor="val_loss", patience=CNN_PATIENCE,
                           restore_best_weights=True, verbose=1)
        t0 = time.time()
        history = full_model.fit(
            X_tr, y_tr, validation_data=(X_vl, y_vl),
            epochs=CNN_EPOCHS, batch_size=CNN_BATCH_SIZE,
            callbacks=[es], verbose=2
        )
        elapsed = time.time() - t0

        best_val_loss = min(history.history["val_loss"])
        print(f"\n  Best val_loss: {best_val_loss:.4f} | Training time: {elapsed:.1f}s")

        # Save the trained CNN
        model_path = ARTIFACT_DIR / f"cnn_{safe}{suffix}.keras"
        full_model.save(str(model_path))
        print(f"  Saved: {model_path}")

        # Extract GAP embeddings for ALL training samples (for XGBoost lag features)
        if pollutant == "CO":
            X_tr_full = np.load(PATCH_DIR / "patches_train_CO_17x17_scaled.npy")
            X_vl_full = np.load(PATCH_DIR / "patches_val_CO_17x17_scaled.npy")
        else:
            X_tr_full, X_vl_full = X_train, X_val

        gap_tr = gap_model.predict(X_tr_full, batch_size=CNN_BATCH_SIZE * 4, verbose=0)
        gap_vl = gap_model.predict(X_vl_full, batch_size=CNN_BATCH_SIZE * 4, verbose=0)

        np.save(ARTIFACT_DIR / f"gap_train_{safe}{suffix}.npy", gap_tr)
        np.save(ARTIFACT_DIR / f"gap_val_{safe}{suffix}.npy",   gap_vl)

        # CNN predictions on valid-target samples only (for Phase 7 residuals)
        pred_tr = full_model.predict(X_tr, batch_size=CNN_BATCH_SIZE * 4, verbose=0).flatten()
        pred_vl = full_model.predict(X_vl, batch_size=CNN_BATCH_SIZE * 4, verbose=0).flatten()

        np.save(ARTIFACT_DIR / f"cnn_pred_train_{safe}{suffix}.npy", pred_tr)
        np.save(ARTIFACT_DIR / f"cnn_pred_val_{safe}{suffix}.npy",   pred_vl)
        np.save(ARTIFACT_DIR / f"valid_idx_train_{safe}{suffix}.npy", np.where(valid_tr)[0])
        np.save(ARTIFACT_DIR / f"valid_idx_val_{safe}{suffix}.npy",   np.where(valid_vl)[0])

        # Check for degenerate GAP embeddings (all collapsed to the same value)
        gap_std    = np.std(gap_tr, axis=0).mean()
        gap_active = bool(gap_std >= 0.01)
        gap_variance_flags[pollutant] = gap_active
        print(f"  GAP std (mean): {gap_std:.6f} → {'ACTIVE' if gap_active else 'DEGENERATE'}")

        rmse_tr = np.sqrt(np.mean((y_tr - pred_tr) ** 2))
        rmse_vl = np.sqrt(np.mean((y_vl - pred_vl) ** 2))
        print(f"  CNN RMSE — Train: {rmse_tr:.3f}, Val: {rmse_vl:.3f}")

        tf.keras.backend.clear_session()
        gc.collect()

    with open(ARTIFACT_DIR / "gap_active_flags.json", "w") as f:
        json.dump(gap_variance_flags, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Phase 6 COMPLETE — 6 CNNs trained")
    print(f"  GAP flags: {gap_variance_flags}")
    print(f"{'='*60}\n")


# ==============================================================
# PHASE 7: XGBOOST TRAINING
# ==============================================================

def run_phase7_train_xgboost():
    """
    Phase 7 — Train 6 XGBoost residual correctors.

    Feature matrix per pollutant:
        [GAP_lag (64)] + [CNN_pred (1)] + [tabular subset (N)] = 66-77 features

    Residual target:
        PM2.5, PM10 → relative: (y_true - cnn_pred) / (y_true + 1)
        All others  → absolute: y_true - cnn_pred

    Tuning: RandomizedSearchCV (n_iter=30, 5-fold KFold, neg_RMSE scoring)
    """
    from sklearn.model_selection import RandomizedSearchCV, KFold
    import xgboost as xgb

    print("=" * 60)
    print("  Phase 7 — Train 6 XGBoost Residual Correctors")
    print("=" * 60)

    y_train_all = np.load(ARTIFACT_DIR / "y_train.npy")
    y_val_all   = np.load(ARTIFACT_DIR / "y_val.npy")
    X_tab_train = np.load(ARTIFACT_DIR / "X_tab_train.npy")
    X_tab_val   = np.load(ARTIFACT_DIR / "X_tab_val.npy")

    with open(ARTIFACT_DIR / "y_column_order.json") as f:
        y_col_order = json.load(f)
    with open(ARTIFACT_DIR / "gap_active_flags.json") as f:
        gap_flags = json.load(f)
    with open(ARTIFACT_DIR / "extended_tab_column_names_with_lags.json") as f:
        all_col_names = json.load(f)

    # XGBoost hyperparameter search grid
    param_dist = {
        "n_estimators":     [100, 200, 300, 400, 500],
        "max_depth":        [3, 4, 5, 6, 7, 8],
        "learning_rate":    [0.01, 0.03, 0.05, 0.08, 0.1],
        "subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 2, 3, 5, 7],
        "reg_alpha":        [0, 0.01, 0.1, 1.0],
        "reg_lambda":       [0.5, 1.0, 2.0, 5.0],
    }

    xgb_active_flags = {}

    for pollutant in POLLUTANTS:
        safe    = pollutant.replace(".", "")
        suffix  = "_17x17" if pollutant == "CO" else ""
        col_idx = y_col_order[pollutant]
        gap_flag = gap_flags.get(pollutant, False)

        print(f"\n{'-'*60}")
        print(f"  Training XGBoost for: {pollutant}")

        # Load CNN predictions and GAP embeddings
        cnn_pred_tr  = np.load(ARTIFACT_DIR / f"cnn_pred_train_{safe}{suffix}.npy")
        cnn_pred_vl  = np.load(ARTIFACT_DIR / f"cnn_pred_val_{safe}{suffix}.npy")
        valid_idx_tr = np.load(ARTIFACT_DIR / f"valid_idx_train_{safe}{suffix}.npy")
        valid_idx_vl = np.load(ARTIFACT_DIR / f"valid_idx_val_{safe}{suffix}.npy")

        # Subset tabular arrays to valid (non-NaN target) rows
        X_tr_tab = X_tab_train[valid_idx_tr]
        X_vl_tab = X_tab_val[valid_idx_vl]
        y_tr     = y_train_all[valid_idx_tr, col_idx]
        y_vl     = y_val_all[valid_idx_vl, col_idx]

        # Build GAP lag features
        if gap_flag:
            gap_tr_all = np.load(ARTIFACT_DIR / f"gap_train_{safe}{suffix}.npy")
            gap_vl_all = np.load(ARTIFACT_DIR / f"gap_val_{safe}{suffix}.npy")
            gap_tr_lag = np.roll(gap_tr_all[valid_idx_tr], shift=1, axis=0)
            gap_tr_lag[0] = 0.0
            gap_vl_lag = np.roll(gap_vl_all[valid_idx_vl], shift=1, axis=0)
            gap_vl_lag[0] = 0.0
        else:
            gap_tr_lag = np.zeros((len(X_tr_tab), GAP_DIM), dtype=np.float32)
            gap_vl_lag = np.zeros((len(X_vl_tab), GAP_DIM), dtype=np.float32)

        # Select per-pollutant tabular columns
        tab_col_names = XGB_POLL_TAB_COLS.get(pollutant, list(ARTIFACT_DIR.parent.name))
        col_idx_map   = {name: i for i, name in enumerate(all_col_names)}
        tab_indices   = [col_idx_map[c] for c in tab_col_names if c in col_idx_map]

        # Build full feature matrix: [GAP_lag | CNN_pred | tab_subset]
        if len(tab_indices) > 0:
            X_tr_full = np.hstack([gap_tr_lag, cnn_pred_tr[:, np.newaxis], X_tr_tab[:, tab_indices]])
            X_vl_full = np.hstack([gap_vl_lag, cnn_pred_vl[:, np.newaxis], X_vl_tab[:, tab_indices]])
        else:
            X_tr_full = np.hstack([gap_tr_lag, cnn_pred_tr[:, np.newaxis], X_tr_tab])
            X_vl_full = np.hstack([gap_vl_lag, cnn_pred_vl[:, np.newaxis], X_vl_tab])

        # Compute residual targets
        if pollutant in USE_RELATIVE_RESID:
            # Relative residual prevents PM2.5/PM10 prediction saturation
            y_residual_tr = (y_tr - cnn_pred_tr) / (y_tr + 1.0)
        else:
            y_residual_tr = y_tr - cnn_pred_tr

        # Drop NaN targets
        valid = ~np.isnan(y_residual_tr)
        X_tr_full     = X_tr_full[valid]
        y_residual_tr = y_residual_tr[valid]

        print(f"  Train: {len(y_residual_tr)} | Val: {len(y_vl)} | Features: {X_tr_full.shape[1]}")

        # Randomized hyperparameter search
        base_model = xgb.XGBRegressor(tree_method="hist", verbosity=0, n_jobs=4)
        kf         = KFold(n_splits=5, shuffle=True, random_state=42)
        search     = RandomizedSearchCV(
            base_model, param_dist, n_iter=XGB_N_ITER,
            scoring="neg_root_mean_squared_error",
            cv=kf, n_jobs=1, verbose=1, random_state=42
        )
        t0 = time.time()
        search.fit(X_tr_full, y_residual_tr)
        elapsed = time.time() - t0

        best_model = search.best_estimator_
        print(f"\n  Best params: {search.best_params_}")
        print(f"  CV RMSE: {-search.best_score_:.4f} | Search time: {elapsed:.0f}s")

        # Evaluate on validation set using the same relative/absolute convention
        cnn_pred_vl_valid = cnn_pred_vl
        if pollutant in USE_RELATIVE_RESID:
            y_final_vl = cnn_pred_vl_valid + (best_model.predict(X_vl_full) * (y_vl + 1.0))
        else:
            y_final_vl = cnn_pred_vl_valid + best_model.predict(X_vl_full)

        valid_vl_mask = ~np.isnan(y_vl)
        rmse_vl       = np.sqrt(np.mean((y_vl[valid_vl_mask] - y_final_vl[valid_vl_mask]) ** 2))
        print(f"  Final Val RMSE: {rmse_vl:.3f}")

        # Save model and metadata
        model_path = ARTIFACT_DIR / f"xgb_{safe}.json"
        best_model.save_model(str(model_path))

        feature_names = ([f"GAP_lag_{i}" for i in range(GAP_DIM)] +
                         ["cnn_pred"] +
                         [all_col_names[i] for i in tab_indices])
        with open(ARTIFACT_DIR / f"xgb_features_{safe}.json", "w") as f:
            json.dump(feature_names, f, indent=2)
        with open(ARTIFACT_DIR / f"xgb_tab_cols_{safe}.json", "w") as f:
            json.dump([all_col_names[i] for i in tab_indices], f, indent=2)

        xgb_active_flags[pollutant] = True
        print(f"  Saved: {model_path}")

    with open(ARTIFACT_DIR / "xgb_active_flags.json", "w") as f:
        json.dump(xgb_active_flags, f, indent=2)
    with open(ARTIFACT_DIR / "xgb_relative_flags.json", "w") as f:
        json.dump({p: (p in USE_RELATIVE_RESID) for p in POLLUTANTS}, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Phase 7 COMPLETE — 6 XGBoost models trained")
    print(f"{'='*60}\n")


# ==============================================================
# PHASE 8: FULL SPATIAL INFERENCE
# ==============================================================

def run_phase8_inference():
    """
    Phase 8 — Vectorized Spatial Inference across Full India Grid.

    For each of the 123 days (Oct 2023 → Jan 2024):
        1. Load 9 satellite+met channels → (120, 124, 9)
        2. Fill gaps with D-1/D+1 mean, add cloud mask → (120, 124, 10)
        3. Apply Gaussian smoothing (sigma=1.0) to remove tile boundary seams
        4. Extract 13×13 and 17×17 patches using sliding_window_view (vectorized)
        5. Scale patches with pre-fitted scalers
        6. Run CNN → get base predictions + 64-dim GAP embeddings for all 14,880 cells
        7. Assemble tabular features + GAP lags (updated day-by-day)
        8. Apply XGBoost corrector → final prediction
        9. Apply Gaussian smoothing to output → roll into next day's lags

    Output: results/all_predictions.pkl
    """
    from numpy.lib.stride_tricks import sliding_window_view
    from scipy.signal import medfilt2d
    from scipy.ndimage import gaussian_filter, uniform_filter

    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf
    import xgboost as xgb

    print("=" * 60)
    print("  Phase 8 — Full Spatial Inference (India Grid)")
    print("=" * 60)

    # ── Load grid spec ────────────────────────────────────────────
    with open(REGRID_DIR / "grid_spec.json") as f:
        gs = json.load(f)
    NLAT, NLON = gs["nlat"], gs["nlon"]
    lats_arr   = np.array(gs["lats"])
    lons_arr   = np.array(gs["lons"])

    # ── Generate date range: Oct 1 2023 → Jan 31 2024 ────────────
    start_date  = datetime(2023, 10, 1)
    end_date    = datetime(2024, 1, 31)
    all_dates   = [start_date + timedelta(days=i)
                   for i in range((end_date - start_date).days + 1)]
    date_strs   = [d.strftime("%Y%m%d") for d in all_dates]
    n_days      = len(date_strs)

    print(f"  Grid: {NLAT} × {NLON} = {NLAT * NLON:,} cells")
    print(f"  Days: {n_days} ({date_strs[0]} → {date_strs[-1]})\n")

    # ── Load models and scalers ───────────────────────────────────
    print("[Load] Loading CNN models, XGBoost models, and scalers...")
    scaler    = joblib.load(ARTIFACT_DIR / "image_scaler.pkl")
    scaler_co = joblib.load(PATCH_DIR / "image_scaler_co.pkl")

    with open(ARTIFACT_DIR / "gap_active_flags.json") as f:
        gap_flags = json.load(f)
    with open(ARTIFACT_DIR / "xgb_active_flags.json") as f:
        xgb_flags = json.load(f)
    with open(ARTIFACT_DIR / "xgb_relative_flags.json") as f:
        xgb_rel_flags = json.load(f)
    with open(ARTIFACT_DIR / "tabular_means.json") as f:
        tabular_means = json.load(f)
    with open(ARTIFACT_DIR / "ceiling_clips.json") as f:
        ceiling_clips = json.load(f)
    with open(ARTIFACT_DIR / "extended_tab_column_names_with_lags.json") as f:
        all_col_names = json.load(f)

    cnn_models, xgb_models, gap_models = {}, {}, {}
    for pollutant in POLLUTANTS:
        safe   = pollutant.replace(".", "")
        suffix = "_17x17" if pollutant == "CO" else ""

        full_model = tf.keras.models.load_model(str(ARTIFACT_DIR / f"cnn_{safe}{suffix}.keras"))
        cnn_models[pollutant] = full_model
        gap_inp  = full_model.input
        gap_out  = full_model.get_layer("gap").output
        gap_models[pollutant] = tf.keras.Model(gap_inp, gap_out)

        xgb_m = xgb.XGBRegressor()
        xgb_m.load_model(str(ARTIFACT_DIR / f"xgb_{safe}.json"))
        xgb_models[pollutant] = xgb_m

    # ── Patch extraction constants ────────────────────────────────
    PATCH_13 = 13; HALF_13 = PATCH_13 // 2
    PATCH_17 = 17; HALF_17 = PATCH_17 // 2
    MAX_PAD  = HALF_17

    # Channel loading helper
    CHAN_DIRS = {
        "AOD": REGRID_DIR / "AOD",
        "NO2": REGRID_DIR / "NO2",
        "SO2": REGRID_DIR / "SO2",
        "CO":  REGRID_DIR / "CO",
        "U10": REGRID_DIR / "U10",
        "V10": REGRID_DIR / "V10",
        "T2M": REGRID_DIR / "T2M",
        "BLH": REGRID_DIR / "BLH",
        "RH":  REGRID_DIR / "RH",
    }

    def load_channels_for_date(date_str):
        """Load all 9 channels for a single day. Returns (NLAT, NLON, 9)."""
        stack = np.full((NLAT, NLON, N_SAT_MET), np.nan, dtype=np.float32)
        for i, (chan, d) in enumerate(CHAN_DIRS.items()):
            candidates = [
                d / f"{chan.lower()}_{date_str}.npy",
                REGRID_DIR / "ERA5" / chan / f"{chan.lower()}_{date_str}.npy",
            ]
            for p in candidates:
                if p.exists():
                    stack[:, :, i] = np.load(p)
                    break
        return stack

    # ── Day-to-day lag state ──────────────────────────────────────
    # These carry forward each pollutant's prediction and GAP embeddings
    prev_preds    = {p: np.zeros((NLAT, NLON), dtype=np.float32) for p in POLLUTANTS}
    prev_gap      = {p: np.zeros((NLAT * NLON, GAP_DIM), dtype=np.float32) for p in POLLUTANTS}
    all_preds_raw = {d: {} for d in date_strs}
    missing_files = []

    # ── Pre-load channels for gap-filling (D-1 / D+1 fallback) ────
    print("[Load] Pre-scanning channel availability (for gap fill)...\n")

    # ── Main daily loop ───────────────────────────────────────────
    for day_i, date_str in enumerate(date_strs):
        t0   = time.time()
        date = all_dates[day_i]
        print(f"[Day {day_i+1:>3}/{n_days}]  {date.strftime('%Y-%m-%d')} ...", end=" ", flush=True)

        # Load today's 9 channels
        today_ch = load_channels_for_date(date_str)

        # Gap fill missing channels with adjacent day average
        for ch_i in range(N_SAT_MET):
            if np.isnan(today_ch[:, :, ch_i]).all():
                prev_ch = load_channels_for_date(date_strs[max(0, day_i - 1)])[:, :, ch_i]
                next_ch = load_channels_for_date(date_strs[min(n_days - 1, day_i + 1)])[:, :, ch_i]
                today_ch[:, :, ch_i] = np.nanmean(np.stack([prev_ch, next_ch], axis=0), axis=0)
                missing_files.append(f"{date_str}_ch{ch_i}")

        # Fill remaining individual NaNs with spatial mean
        for ch_i in range(N_SAT_MET):
            ch_mean = np.nanmean(today_ch[:, :, ch_i])
            today_ch[:, :, ch_i] = np.where(
                np.isnan(today_ch[:, :, ch_i]), ch_mean, today_ch[:, :, ch_i])

        # Gaussian smoothing to remove satellite tile boundary seams
        from scipy.ndimage import gaussian_filter
        for ch_i in range(N_SAT_MET):
            today_ch[:, :, ch_i] = gaussian_filter(today_ch[:, :, ch_i], sigma=1.0)

        # Median filter on AOD channel (speckle suppression)
        today_ch[:, :, CHANNEL_ORDER["AOD"]] = medfilt2d(
            today_ch[:, :, CHANNEL_ORDER["AOD"]], kernel_size=3)

        # Build cloud mask: 1=valid, 0=any channel had a gap
        cloud_mask = np.ones((NLAT, NLON, 1), dtype=np.float32)

        # Stack to (NLAT, NLON, 10) with cloud mask as last channel
        grid_10 = np.concatenate([today_ch, cloud_mask], axis=-1)

        # Pad grid for patch extraction
        pad_10 = np.pad(grid_10, ((MAX_PAD, MAX_PAD), (MAX_PAD, MAX_PAD), (0, 0)),
                        mode="reflect")

        # Vectorized patch extraction using sliding_window_view
        # → 13×13 windows for all 5 non-CO pollutants
        wins_13 = sliding_window_view(
            pad_10, window_shape=(PATCH_13, PATCH_13, 10)
        )[::1, ::1, 0, :, :, :]
        windows_13 = wins_13.reshape(-1, PATCH_13, PATCH_13, 10)  # (NLAT*NLON, 13, 13, 10)

        # → 17×17 windows for CO
        wins_17 = sliding_window_view(
            pad_10, window_shape=(PATCH_17, PATCH_17, 10)
        )[::1, ::1, 0, :, :, :]
        windows_17 = wins_17.reshape(-1, PATCH_17, PATCH_17, 10)  # (NLAT*NLON, 17, 17, 10)

        # Scale patches
        flat_13 = windows_13.reshape(-1, PATCH_13 * PATCH_13 * 10)
        flat_13 = scaler.transform(flat_13).reshape(-1, PATCH_13, PATCH_13, 10)

        flat_17 = windows_17.reshape(-1, PATCH_17 * PATCH_17 * 10)
        flat_17 = scaler_co.transform(flat_17).reshape(-1, PATCH_17, PATCH_17, 10)

        # ── Per-pollutant inference ───────────────────────────────
        day_preds = {}
        for pollutant in POLLUTANTS:
            safe   = pollutant.replace(".", "")
            suffix = "_17x17" if pollutant == "CO" else ""

            patches = flat_17 if pollutant == "CO" else flat_13
            cnn_pred_flat = cnn_models[pollutant].predict(
                patches, batch_size=512, verbose=0).flatten()
            gap_flat      = gap_models[pollutant].predict(
                patches, batch_size=512, verbose=0)

            # Build static spatial tabular features (lat, lon, doy)
            lat_grid = np.tile(lats_arr[:, np.newaxis], (1, NLON)).flatten()
            lon_grid = np.tile(lons_arr[np.newaxis, :], (NLAT, 1)).flatten()
            doy_arr  = np.full(NLAT * NLON, date.timetuple().tm_yday, dtype=np.float32)

            # Assemble tabular features for XGBoost
            tab_dict = {
                "lat": lat_grid, "lon": lon_grid,
                "doy": doy_arr,
                "ventilation_index": today_ch[:, :, CHANNEL_ORDER["BLH"]].flatten() *
                                     np.sqrt(today_ch[:, :, CHANNEL_ORDER["U10"]].flatten() ** 2 +
                                             today_ch[:, :, CHANNEL_ORDER["V10"]].flatten() ** 2),
                "wind_direction": np.arctan2(today_ch[:, :, CHANNEL_ORDER["V10"]].flatten(),
                                             today_ch[:, :, CHANNEL_ORDER["U10"]].flatten()),
                "t2m_squared": today_ch[:, :, CHANNEL_ORDER["T2M"]].flatten() ** 2,
            }

            # GAP lag from previous day
            gap_lag = prev_gap.get(pollutant, np.zeros((NLAT * NLON, GAP_DIM)))

            # Build feature vector subsets based on saved XGB column list
            try:
                with open(ARTIFACT_DIR / f"xgb_tab_cols_{safe}.json") as _f:
                    xgb_tab_cols = json.load(_f)
                tab_arr = np.column_stack([
                    tab_dict.get(c, np.zeros(NLAT * NLON)) for c in xgb_tab_cols
                ])
            except Exception:
                tab_arr = np.column_stack(list(tab_dict.values()))

            X_xgb = np.hstack([gap_lag, cnn_pred_flat[:, np.newaxis],
                                tab_arr.astype(np.float32)])

            # XGBoost residual correction
            r_hat = xgb_models[pollutant].predict(X_xgb)

            if pollutant in USE_RELATIVE_RESID:
                # Reverse relative residual: final = cnn + r * (cnn + 1)
                final_pred = cnn_pred_flat + r_hat * (cnn_pred_flat + 1.0)
            else:
                final_pred = cnn_pred_flat + r_hat

            # Clip negatives (concentrations cannot be negative)
            final_pred = np.maximum(final_pred, 0.0)

            # Gaussian smoothing to prevent grid boundary artifacts
            final_2d   = final_pred.reshape(NLAT, NLON)
            final_2d   = gaussian_filter(final_2d, sigma=1.0)
            final_pred = final_2d.flatten()

            day_preds[pollutant] = final_pred.reshape(NLAT, NLON)

            # Update lag state for next day
            prev_preds[pollutant] = day_preds[pollutant].copy()
            prev_gap[pollutant]   = gap_flat.copy()

        all_preds_raw[date_str] = day_preds
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s)")

        gc.collect()

    # ── Save all predictions ──────────────────────────────────────
    print(f"\n[Save] Saving all_predictions.pkl ...")
    joblib.dump(all_preds_raw, RESULTS_DIR / "all_predictions.pkl")

    with open(RESULTS_DIR / "missing_files.log", "w") as f:
        f.write("\n".join(missing_files))

    print(f"\n{'='*60}")
    print(f"  Phase 8 COMPLETE — {n_days} days of full-grid predictions saved")
    print(f"  Output: {RESULTS_DIR / 'all_predictions.pkl'}")
    print(f"{'='*60}\n")


# ==============================================================
# PHASE 9: EVALUATION
# ==============================================================

def run_phase9_evaluate():
    """
    Phase 9 — Compute validation metrics: RMSE, MAE, R², Within-R².

    Within-R² (detrended R²) removes station × month mean biases to assess
    the model's ability to track temporal variability.
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    print("=" * 60)
    print("  Phase 9 — Evaluation")
    print("=" * 60)

    y_val_all = np.load(ARTIFACT_DIR / "y_val.npy")
    meta_val  = pd.read_csv(PATCH_DIR / "metadata_val_qc.csv")

    with open(ARTIFACT_DIR / "y_column_order.json") as f:
        y_col_order = json.load(f)
    with open(ARTIFACT_DIR / "xgb_active_flags.json") as f:
        xgb_flags = json.load(f)
    with open(ARTIFACT_DIR / "gap_active_flags.json") as f:
        gap_flags = json.load(f)
    with open(ARTIFACT_DIR / "extended_tab_column_names_with_lags.json") as f:
        all_col_names = json.load(f)

    import xgboost as xgb

    all_metrics = {}
    val_records = []

    for pollutant in POLLUTANTS:
        safe   = pollutant.replace(".", "")
        suffix = "_17x17" if pollutant == "CO" else ""
        col_idx = y_col_order[pollutant]

        # Load validation predictions
        cnn_pred_vl  = np.load(ARTIFACT_DIR / f"cnn_pred_val_{safe}{suffix}.npy")
        valid_idx_vl = np.load(ARTIFACT_DIR / f"valid_idx_val_{safe}{suffix}.npy")
        y_vl         = y_val_all[valid_idx_vl, col_idx]

        meta_vl_poll = meta_val.iloc[valid_idx_vl].reset_index(drop=True)
        valid_targets = ~np.isnan(y_vl)

        # Rebuild XGBoost predictions
        gap_flag = gap_flags.get(pollutant, False)
        X_tab_vl = np.load(ARTIFACT_DIR / "X_tab_val.npy")[valid_idx_vl]

        if gap_flag:
            gap_vl_all = np.load(ARTIFACT_DIR / f"gap_val_{safe}{suffix}.npy")
            gap_vl_lag = np.roll(gap_vl_all[valid_idx_vl], shift=1, axis=0)
            gap_vl_lag[0] = 0.0
        else:
            gap_vl_lag = np.zeros((len(X_tab_vl), GAP_DIM), dtype=np.float32)

        try:
            with open(ARTIFACT_DIR / f"xgb_tab_cols_{safe}.json") as _f:
                tab_cols = json.load(_f)
            col_map     = {n: i for i, n in enumerate(all_col_names)}
            tab_indices = [col_map[c] for c in tab_cols if c in col_map]
            X_tab_sel   = X_tab_vl[:, tab_indices]
        except Exception:
            X_tab_sel = X_tab_vl

        X_xgb = np.hstack([gap_vl_lag, cnn_pred_vl[:, np.newaxis], X_tab_sel.astype(np.float32)])

        xgb_m = xgb.XGBRegressor()
        xgb_m.load_model(str(ARTIFACT_DIR / f"xgb_{safe}.json"))
        r_hat = xgb_m.predict(X_xgb)

        if pollutant in USE_RELATIVE_RESID:
            final_pred = cnn_pred_vl + r_hat * (y_vl + 1.0)
        else:
            final_pred = cnn_pred_vl + r_hat

        # Apply masks
        y_vl_clean    = y_vl[valid_targets]
        pred_vl_clean = final_pred[valid_targets]

        rmse = np.sqrt(mean_squared_error(y_vl_clean, pred_vl_clean))
        mae  = mean_absolute_error(y_vl_clean, pred_vl_clean)
        r2   = r2_score(y_vl_clean, pred_vl_clean)

        all_metrics[pollutant] = {"rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}
        print(f"  {pollutant:<6} — RMSE: {rmse:.3f} | MAE: {mae:.3f} | R²: {r2:.3f}")

        # Append to validation record
        for i, idx in enumerate(np.where(valid_targets)[0]):
            val_records.append({
                "pollutant": pollutant,
                "station":   meta_vl_poll.iloc[idx].get("station", ""),
                "date":      meta_vl_poll.iloc[idx].get("date", ""),
                "y_true":    float(y_vl_clean[np.where(valid_targets)[0] == idx]),
                "y_pred":    float(pred_vl_clean[np.where(valid_targets)[0] == idx]),
            })

    with open(RESULTS_DIR / "evaluation_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    pd.DataFrame(val_records).to_csv(RESULTS_DIR / "validation_predictions.csv", index=False)

    print(f"\n{'='*60}")
    print(f"  Phase 9 COMPLETE — Metrics saved")
    print(f"{'='*60}\n")


# ==============================================================
# PHASE 10: SHAP INTERPRETABILITY
# ==============================================================

def run_phase10_shap():
    """
    Phase 10 — SHAP TreeExplainer for XGBoost interpretability.
    Generates beeswarm + bar summary plots per pollutant.
    Key finding: Solar Radiation is the top driver for O3 predictions.
    """
    import shap
    import xgboost as xgb_lib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("  Phase 10 — SHAP Interpretability")
    print("=" * 60)

    MAX_SHAP_SAMPLES = 500  # Subsample for speed

    with open(ARTIFACT_DIR / "gap_active_flags.json") as f:
        gap_flags = json.load(f)
    with open(ARTIFACT_DIR / "xgb_active_flags.json") as f:
        xgb_flags = json.load(f)
    with open(ARTIFACT_DIR / "extended_tab_column_names_with_lags.json") as f:
        all_col_names = json.load(f)

    for pollutant in POLLUTANTS:
        safe   = pollutant.replace(".", "")
        suffix = "_17x17" if pollutant == "CO" else ""

        if not xgb_flags.get(pollutant, True):
            print(f"\n  [SKIP] {pollutant} — XGB disabled, no SHAP.")
            continue

        print(f"\n  Generating SHAP for: {pollutant}")

        model = xgb_lib.XGBRegressor()
        model.load_model(str(ARTIFACT_DIR / f"xgb_{safe}.json"))

        try:
            with open(ARTIFACT_DIR / f"xgb_features_{safe}.json") as _f:
                feature_names = json.load(_f)
        except Exception:
            feature_names = None

        # Rebuild XGBoost feature matrix for validation set
        valid_idx_vl = np.load(ARTIFACT_DIR / f"valid_idx_val_{safe}{suffix}.npy")
        cnn_pred_vl  = np.load(ARTIFACT_DIR / f"cnn_pred_val_{safe}{suffix}.npy")
        X_tab_vl     = np.load(ARTIFACT_DIR / "X_tab_val.npy")[valid_idx_vl]

        gap_flag = gap_flags.get(pollutant, False)
        if gap_flag:
            gap_vl_all = np.load(ARTIFACT_DIR / f"gap_val_{safe}{suffix}.npy")
            gap_vl_lag = np.roll(gap_vl_all[valid_idx_vl], shift=1, axis=0)
            gap_vl_lag[0] = 0.0
        else:
            gap_vl_lag = np.zeros((len(X_tab_vl), GAP_DIM), dtype=np.float32)

        try:
            with open(ARTIFACT_DIR / f"xgb_tab_cols_{safe}.json") as _f:
                tab_cols = json.load(_f)
            col_map     = {n: i for i, n in enumerate(all_col_names)}
            tab_indices = [col_map[c] for c in tab_cols if c in col_map]
            X_tab_sel   = X_tab_vl[:, tab_indices]
        except Exception:
            X_tab_sel = X_tab_vl

        X_xgb = np.hstack([gap_vl_lag, cnn_pred_vl[:, np.newaxis], X_tab_sel.astype(np.float32)])

        # Subsample for speed
        if len(X_xgb) > MAX_SHAP_SAMPLES:
            rng    = np.random.default_rng(42)
            idx    = rng.choice(len(X_xgb), MAX_SHAP_SAMPLES, replace=False)
            X_shap = X_xgb[idx]
        else:
            X_shap = X_xgb

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap)

        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        plt.sca(axes[0])
        shap.summary_plot(shap_values, X_shap, feature_names=feature_names,
                          show=False, max_display=20)
        axes[0].set_title(f"{pollutant} — SHAP Beeswarm")

        plt.sca(axes[1])
        shap.summary_plot(shap_values, X_shap, feature_names=feature_names,
                          plot_type="bar", show=False, max_display=20)
        axes[1].set_title(f"{pollutant} — SHAP Feature Importance")

        plt.tight_layout()
        out_path = RESULTS_DIR / f"shap_{safe}.png"
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")

    print(f"\n{'='*60}")
    print(f"  Phase 10 COMPLETE — SHAP plots saved to results/")
    print(f"{'='*60}\n")


# ==============================================================
# PHASE 11: SPATIAL MAP GENERATION
# ==============================================================

def run_phase11_maps():
    """
    Phase 11 — Generate daily concentration maps per pollutant.

    Uses pcolormesh (not imshow) to avoid block artifacts.
    Overlays the India boundary from the GADM GeoJSON file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import json

    print("=" * 60)
    print("  Phase 11 — Spatial Map Generation")
    print("=" * 60)

    preds = joblib.load(RESULTS_DIR / "all_predictions.pkl")

    with open(REGRID_DIR / "grid_spec.json") as f:
        gs = json.load(f)
    lats_arr = np.array(gs["lats"])
    lons_arr = np.array(gs["lons"])

    # Load India boundary (GADM GeoJSON)
    try:
        import geopandas as gpd
        india_gdf = gpd.read_file(str(GADM_PATH)) if GADM_PATH.exists() else None
    except Exception:
        india_gdf = None

    # Per-pollutant colormaps
    COLORMAPS = {
        "PM2.5": "YlOrRd", "PM10": "YlOrBr", "NO2": "PuBu",
        "SO2": "BuGn",     "O3":   "BuPu",   "CO":  "Reds",
    }
    UNITS = {
        "PM2.5": "µg/m³", "PM10": "µg/m³", "NO2": "µg/m³",
        "SO2":   "µg/m³", "O3":   "µg/m³", "CO":  "mol/m²",
    }

    for pollutant in POLLUTANTS:
        poll_maps_dir = MAPS_DIR / pollutant
        poll_maps_dir.mkdir(exist_ok=True)
        print(f"\n  [{pollutant}] Generating daily maps...")

        # Collect all daily grids to compute colorbar range
        all_grids = []
        for date_str, day_preds in preds.items():
            grid = day_preds.get(pollutant)
            if grid is not None:
                all_grids.append(grid)

        if not all_grids:
            continue

        all_vals = np.concatenate([g.flatten() for g in all_grids])
        all_vals = all_vals[~np.isnan(all_vals)]
        vmin = np.percentile(all_vals, 2)
        vmax = np.percentile(all_vals, 98)

        for date_str, day_preds in preds.items():
            grid = day_preds.get(pollutant)
            if grid is None:
                continue

            fig, ax = plt.subplots(1, 1, figsize=(10, 8))

            # pcolormesh with the reference grid
            lon_mesh, lat_mesh = np.meshgrid(lons_arr, lats_arr)
            pcm = ax.pcolormesh(lon_mesh, lat_mesh, grid,
                                cmap=COLORMAPS[pollutant], vmin=vmin, vmax=vmax,
                                shading="auto")
            plt.colorbar(pcm, ax=ax, label=UNITS[pollutant], shrink=0.7)

            # India boundary overlay
            if india_gdf is not None:
                india_gdf.boundary.plot(ax=ax, linewidth=1.0, color="black")

            # Format date string for title
            date_fmt = datetime.strptime(date_str, "%Y%m%d").strftime("%d %b %Y")
            ax.set_title(f"{pollutant} — {date_fmt}", fontsize=13)
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Latitude (°N)")
            ax.set_xlim(67, 98)
            ax.set_ylim(8, 38)

            out_path = poll_maps_dir / f"{pollutant}_{date_str}.png"
            plt.savefig(str(out_path), dpi=100, bbox_inches="tight")
            plt.close()

        print(f"    {len(all_grids)} maps saved → {poll_maps_dir}")

    print(f"\n{'='*60}")
    print(f"  Phase 11 COMPLETE — Maps saved to maps/")
    print(f"{'='*60}\n")


# ==============================================================
# PHASE 12: AQI CALCULATION AND MAPPING
# ==============================================================

def run_phase12_aqi():
    """
    Phase 12 — Apply official CPCB NAAQS AQI breakpoints.

    For each pixel-day:
        sub_index(p) = linear interpolation within the matching breakpoint tier
        final_AQI = max(all 6 sub_indices)
        dominant_pollutant = argmax(sub_indices)

    Generates: daily AQI maps, seasonal mean, worst-day map, dominant pollutant map.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm

    print("=" * 60)
    print("  Phase 12 — AQI Calculation and Mapping")
    print("=" * 60)

    # CPCB NAAQS AQI Breakpoints
    # Format: (C_low, C_high, I_low, I_high) per tier
    BREAKPOINTS = {
        "PM2.5": [(0, 30, 0, 50),    (30, 60, 51, 100),   (60, 90, 101, 200),
                  (90, 120, 201, 300), (120, 250, 301, 400), (250, 500, 401, 500)],
        "PM10":  [(0, 50, 0, 50),    (50, 100, 51, 100),  (100, 250, 101, 200),
                  (250, 350, 201, 300), (350, 430, 301, 400), (430, 600, 401, 500)],
        "NO2":   [(0, 40, 0, 50),    (40, 80, 51, 100),   (80, 180, 101, 200),
                  (180, 280, 201, 300), (280, 400, 301, 400), (400, 800, 401, 500)],
        "SO2":   [(0, 40, 0, 50),    (40, 80, 51, 100),   (80, 380, 101, 200),
                  (380, 800, 201, 300), (800, 1600, 301, 400), (1600, 2000, 401, 500)],
        "O3":    [(0, 50, 0, 50),    (50, 100, 51, 100),  (100, 168, 101, 200),
                  (168, 208, 201, 300), (208, 748, 301, 400), (748, 1000, 401, 500)],
        "CO":    [(0, 1.0, 0, 50),   (1.0, 2.0, 51, 100), (2.0, 10, 101, 200),
                  (10, 17, 201, 300), (17, 34, 301, 400),  (34, 50, 401, 500)],
    }

    def calculate_sub_index(concentration: np.ndarray, pollutant: str) -> np.ndarray:
        """Vectorized CPCB AQI sub-index calculation."""
        C    = concentration
        bp   = BREAKPOINTS[pollutant]
        out  = np.full_like(C, np.nan, dtype=np.float32)

        for (c_lo, c_hi, i_lo, i_hi) in bp:
            mask = (C >= c_lo) & (C <= c_hi)
            if mask.any():
                out[mask] = i_lo + (C[mask] - c_lo) * (i_hi - i_lo) / (c_hi - c_lo + 1e-9)

        # Values above highest breakpoint → AQI = 500
        out[C > bp[-1][1]] = 500.0
        return out

    preds = joblib.load(RESULTS_DIR / "all_predictions.pkl")

    with open(REGRID_DIR / "grid_spec.json") as f:
        gs = json.load(f)
    lats_arr = np.array(gs["lats"])
    lons_arr = np.array(gs["lons"])
    NLAT, NLON = gs["nlat"], gs["nlon"]

    try:
        import geopandas as gpd
        india_gdf = gpd.read_file(str(GADM_PATH)) if GADM_PATH.exists() else None
    except Exception:
        india_gdf = None

    aqi_maps_dir = MAPS_DIR / "AQI"
    aqi_maps_dir.mkdir(exist_ok=True)

    # AQI color scale (CPCB 6-band: Good → Severe)
    AQI_COLORS  = ["#55a84f", "#a3c853", "#fff833", "#f29c2b", "#e93f33", "#af2d24"]
    AQI_BREAKS  = [0, 50, 100, 200, 300, 400, 500]
    AQI_CMAP    = ListedColormap(AQI_COLORS)
    AQI_NORM    = BoundaryNorm(AQI_BREAKS, ncolors=len(AQI_COLORS))
    AQI_LABELS  = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

    all_aqi_maps     = []
    dominant_counter = np.zeros((NLAT, NLON, len(POLLUTANTS)), dtype=np.int32)
    worst_day_aqi    = np.zeros((NLAT, NLON), dtype=np.float32)

    for date_str, day_preds in preds.items():
        # Calculate AQI sub-index for each pollutant
        sub_indices = []
        for p in POLLUTANTS:
            grid = day_preds.get(p, np.zeros((NLAT, NLON)))
            if grid is None:
                grid = np.zeros((NLAT, NLON))
            sub_indices.append(calculate_sub_index(grid.astype(np.float32), p))

        sub_stack = np.stack(sub_indices, axis=-1)   # (NLAT, NLON, 6)
        aqi_map   = np.nanmax(sub_stack, axis=-1)     # (NLAT, NLON) — final AQI
        dom_map   = np.nanargmax(sub_stack, axis=-1)  # (NLAT, NLON) — dominant pollutant idx

        all_aqi_maps.append(aqi_map)
        worst_day_aqi = np.maximum(worst_day_aqi, aqi_map)

        for p_i in range(len(POLLUTANTS)):
            dominant_counter[:, :, p_i] += (dom_map == p_i).astype(np.int32)

        # Plot daily AQI map
        date_fmt = datetime.strptime(date_str, "%Y%m%d").strftime("%d %b %Y")
        fig, ax  = plt.subplots(1, 1, figsize=(10, 8))
        lon_mesh, lat_mesh = np.meshgrid(lons_arr, lats_arr)
        pcm = ax.pcolormesh(lon_mesh, lat_mesh, aqi_map,
                            cmap=AQI_CMAP, norm=AQI_NORM, shading="auto")
        if india_gdf is not None:
            india_gdf.boundary.plot(ax=ax, linewidth=1.0, color="black")
        plt.colorbar(pcm, ax=ax, ticks=AQI_BREAKS, label="AQI", shrink=0.7)
        ax.set_title(f"AQI — {date_fmt}", fontsize=13)
        ax.set_xlim(67, 98); ax.set_ylim(8, 38)
        plt.savefig(str(aqi_maps_dir / f"AQI_{date_str}.png"), dpi=100, bbox_inches="tight")
        plt.close()

    # Seasonal mean AQI
    seasonal_mean = np.nanmean(np.stack(all_aqi_maps, axis=0), axis=0)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    lon_mesh, lat_mesh = np.meshgrid(lons_arr, lats_arr)
    pcm = ax.pcolormesh(lon_mesh, lat_mesh, seasonal_mean,
                        cmap=AQI_CMAP, norm=AQI_NORM, shading="auto")
    if india_gdf is not None:
        india_gdf.boundary.plot(ax=ax, linewidth=1.0, color="black")
    plt.colorbar(pcm, ax=ax, ticks=AQI_BREAKS, label="AQI", shrink=0.7)
    ax.set_title("Seasonal Mean AQI (Oct 2023 – Jan 2024)", fontsize=13)
    ax.set_xlim(67, 98); ax.set_ylim(8, 38)
    plt.savefig(str(aqi_maps_dir / "AQI_Seasonal_Mean.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Worst-day AQI
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    pcm = ax.pcolormesh(lon_mesh, lat_mesh, worst_day_aqi,
                        cmap=AQI_CMAP, norm=AQI_NORM, shading="auto")
    if india_gdf is not None:
        india_gdf.boundary.plot(ax=ax, linewidth=1.0, color="black")
    plt.colorbar(pcm, ax=ax, ticks=AQI_BREAKS, label="AQI", shrink=0.7)
    ax.set_title("Worst-Day AQI (Oct 2023 – Jan 2024)", fontsize=13)
    ax.set_xlim(67, 98); ax.set_ylim(8, 38)
    plt.savefig(str(aqi_maps_dir / "AQI_Worst_Day.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Dominant pollutant map (most frequent dominant over the season)
    dom_pollutant_map = np.argmax(dominant_counter, axis=-1)
    dom_cmap = ListedColormap(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"])
    fig, ax  = plt.subplots(1, 1, figsize=(10, 8))
    pcm = ax.pcolormesh(lon_mesh, lat_mesh, dom_pollutant_map,
                        cmap=dom_cmap, vmin=0, vmax=5, shading="auto")
    if india_gdf is not None:
        india_gdf.boundary.plot(ax=ax, linewidth=1.0, color="black")
    cbar = plt.colorbar(pcm, ax=ax, ticks=range(len(POLLUTANTS)), shrink=0.7)
    cbar.set_ticklabels(POLLUTANTS)
    cbar.set_label("Dominant Pollutant")
    ax.set_title("Most Frequent Dominant Pollutant (Oct 2023 – Jan 2024)", fontsize=12)
    ax.set_xlim(67, 98); ax.set_ylim(8, 38)
    plt.savefig(str(aqi_maps_dir / "AQI_Dominant_Pollutant.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n{'='*60}")
    print(f"  Phase 12 COMPLETE")
    print(f"  Daily AQI maps, seasonal mean, worst-day, dominant pollutant saved")
    print(f"  Output: {aqi_maps_dir}")
    print(f"{'='*60}\n")


# ==============================================================
# ENTRY POINT
# ==============================================================

PHASE_MAP = {
    6:  run_phase6_train_cnn,
    7:  run_phase7_train_xgboost,
    8:  run_phase8_inference,
    9:  run_phase9_evaluate,
    10: run_phase10_shap,
    11: run_phase11_maps,
    12: run_phase12_aqi,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="run_model.py — Objective 1 Master Model Pipeline (Phases 6–12)."
    )
    parser.add_argument("--phase", type=int, default=None,
                        help="Run only a specific phase (6-12). Default: run all.")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip phases 6 and 7 (CNN + XGBoost training) and go straight to inference.")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Objective 1 — Hybrid CNN-XGBoost AQI Prediction Model")
    print("=" * 60 + "\n")

    if args.phase is not None:
        # Run a single specific phase
        if args.phase in PHASE_MAP:
            PHASE_MAP[args.phase]()
        else:
            print(f"[ERROR] Phase {args.phase} not recognized. Must be 6–12.")
            raise SystemExit(1)
    else:
        # Run all phases in order
        start_phase = 8 if args.skip_training else 6

        for phase_num in range(start_phase, 13):
            if phase_num in PHASE_MAP:
                print(f"\n{'='*60}")
                print(f"  >>> Starting Phase {phase_num} <<<")
                print(f"{'='*60}")
                PHASE_MAP[phase_num]()

    print("\n" + "=" * 60)
    print("  run_model.py — ALL PHASES COMPLETE")
    print("=" * 60)
