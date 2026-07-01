"""
04_extract_patches.py — CNN Image Patch Extraction, QC, and Scaling
=====================================================================
Objective 1 | Step 4 of 5 (Patch Engineering)

PURPOSE:
    For every station×day in the CPCB ground truth, extract a spatial
    13×13 pixel patch (CO uses 17×17) from the 9 satellite+met channels,
    apply quality control, and normalize with StandardScaler.

PIPELINE (3 sub-steps):

    Step 1 → EXTRACT PATCHES (extract_cnn_patches):
        - Reads the shared regridded numpy arrays (AOD, NO2, SO2, CO, U10,
          V10, T2M, BLH, RH) for each day.
        - Centers a 13×13 patch on each station's grid location.
        - CO uses an extended 17×17 patch for better spatial context.
        - Output: cnn_patches/X_image_{train,val}.npy  shape (N, 13, 13, 9)
        - Output: cnn_patches/metadata_{train,val}.csv

    Step 2 → IMAGE QC (phase2_image_qc):
        - Drops samples with > 30% NaN across all 9 channels.
        - Fills 100%-NaN channels with the global training mean.
        - Fills remaining NaNs with a 3×3 spatial median filter.
        - Applies a 2D median filter on AOD channel (speckle removal).
        - Appends a cloud/missingness binary mask as channel 10.
        - Output: cnn_patches/X_image_{train,val}_qc.npy  shape (N, 13, 13, 10)

    Step 3 → SCALING (phase5_scaling):
        - Fits a StandardScaler on flattened training patches.
        - Transforms both train and val.
        - Output: artifacts/X_image_{train,val}_scaled.npy
        - Output: artifacts/image_scaler.pkl

NOTE:
    The processed cnn_patches/ and artifacts/ directories are already present
    in this repository. Re-run this script only if you change the input data.

USAGE:
    python 04_extract_patches.py
"""

import os
import sys
import json
import glob
import warnings
import numpy as np
import pandas as pd
import scipy.ndimage
import joblib
from pathlib import Path
from scipy.signal import medfilt2d
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# Ensure channels.py is importable from the same directory
# ─────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from channels import CHANNEL_ORDER, N_SAT_MET, MEDIAN_FILTER_CHANNELS, N_CHANNELS

# ─────────────────────────────────────────────────────────────────
# PATHS — all relative to the objective_1 folder
# ─────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
DATA_DIR     = BASE_DIR / "data"
REGRID_DIR   = DATA_DIR / "regridded"
PATCH_DIR    = BASE_DIR / "cnn_patches"
ARTIFACT_DIR = BASE_DIR / "artifacts"
MASTER_CSV   = DATA_DIR / "cpcb_master.csv"

PATCH_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────
PATCH_SIZE_13 = 13    # standard patch size for all pollutants except CO
PATCH_SIZE_17 = 17    # extended patch size for CO (better spatial context)
NAN_THRESHOLD = 0.30  # drop samples with > 30% NaN across all channels
AOD_IDX       = CHANNEL_ORDER["AOD"]   # channel index 0

# Channel folder names in data/regridded/
CHANNEL_FOLDERS = {
    "AOD": "AOD",
    "NO2": "NO2",
    "SO2": "SO2",
    "CO":  "CO",
    "U10": "U10",
    "V10": "V10",
    "T2M": "T2M",
    "BLH": "BLH",
    "RH":  "RH",
}


# ==============================================================
# STEP 1: EXTRACT CNN PATCHES
# ==============================================================

def load_daily_channels(date_str: str) -> np.ndarray | None:
    """
    Load all 9 satellite+met channels for a given date.
    Returns a float32 array of shape (NLAT, NLON, 9), or None if all missing.
    """
    with open(REGRID_DIR / "grid_spec.json") as f:
        gs = json.load(f)
    nlat, nlon = gs["nlat"], gs["nlon"]

    stack = np.full((nlat, nlon, N_SAT_MET), np.nan, dtype=np.float32)

    for i, (chan_name, folder) in enumerate(CHANNEL_FOLDERS.items()):
        # Try standard naming: {channel_lower}_{YYYYMMDD}.npy
        # ERA5 channels are in a subfolder matching the variable name
        candidates = [
            REGRID_DIR / folder / f"{chan_name.lower()}_{date_str}.npy",
            REGRID_DIR / "ERA5" / folder / f"{chan_name.lower()}_{date_str}.npy",
        ]
        for path in candidates:
            if path.exists():
                stack[:, :, i] = np.load(path)
                break

    return stack


def get_patch(grid: np.ndarray, lat_idx: int, lon_idx: int, size: int) -> np.ndarray:
    """
    Extract a (size × size × C) patch centered on (lat_idx, lon_idx).
    Clamps to grid boundaries — no circular wrap.
    """
    half = size // 2
    nlat, nlon, c = grid.shape

    r0 = max(0, lat_idx - half)
    r1 = min(nlat, lat_idx + half + 1)
    c0 = max(0, lon_idx - half)
    c1 = min(nlon, lon_idx + half + 1)

    patch = grid[r0:r1, c0:c1, :]    # may be smaller at boundary

    # Zero-pad to the target size if at a boundary
    if patch.shape[0] != size or patch.shape[1] != size:
        out = np.full((size, size, c), np.nan, dtype=np.float32)
        pr  = size - patch.shape[0]
        pc  = size - patch.shape[1]
        out[pr // 2: pr // 2 + patch.shape[0],
            pc // 2: pc // 2 + patch.shape[1], :] = patch
        return out
    return patch


def run_step1_extract():
    """Step 1 — Extract 13×13 (and 17×17 for CO) spatial patches per station-day."""
    if not MASTER_CSV.exists():
        print(f"[ERROR] cpcb_master.csv not found at {MASTER_CSV}")
        return

    print(f"\n{'='*60}")
    print(f"  STEP 1 — Extract CNN Patches")
    print(f"{'='*60}\n")

    master = pd.read_csv(MASTER_CSV)
    master["date_parsed"] = pd.to_datetime(master["date"], format="mixed", dayfirst=True).dt.normalize()

    # Load grid spec for lat/lon → row/col mapping
    with open(REGRID_DIR / "grid_spec.json") as f:
        gs = json.load(f)

    lats_arr = np.array(gs["lats"])    # N→S
    lons_arr = np.array(gs["lons"])    # W→E

    def nearest_idx(arr, val):
        return int(np.argmin(np.abs(arr - val)))

    for split in ["train", "val"]:
        split_mask = master["split"] == split if "split" in master.columns else pd.Series([True] * len(master))
        df = master[split_mask].copy()

        patches_13, patches_17 = [], []
        meta_rows = []

        for _, row in df.iterrows():
            date_str = row["date_parsed"].strftime("%Y%m%d")
            grid     = load_daily_channels(date_str)

            lat_idx = nearest_idx(lats_arr, row["lat"])
            lon_idx = nearest_idx(lons_arr, row["lon"])

            patch_13 = get_patch(grid, lat_idx, lon_idx, PATCH_SIZE_13)
            patch_17 = get_patch(grid, lat_idx, lon_idx, PATCH_SIZE_17)

            patches_13.append(patch_13)
            patches_17.append(patch_17)
            meta_rows.append(row)

        if patches_13:
            X_13 = np.stack(patches_13).astype(np.float32)
            X_17 = np.stack(patches_17).astype(np.float32)
            meta = pd.DataFrame(meta_rows)

            np.save(PATCH_DIR / f"X_image_{split}.npy",              X_13)
            np.save(PATCH_DIR / f"patches_{split}_CO_17x17.npy",     X_17)
            meta.to_csv(PATCH_DIR / f"metadata_{split}.csv", index=False)
            print(f"  [{split}] {X_13.shape} patches extracted")

    print(f"\n  STEP 1 DONE\n")


# ==============================================================
# STEP 2: IMAGE QC — Cloud Masking, NaN Fill, Median Filter
# ==============================================================

def qc_patches(X_raw: np.ndarray, global_means: np.ndarray | None = None) -> tuple:
    """
    Apply quality control to a patch array:
      1. Record original NaN mask (1=valid, 0=missing)
      2. Drop samples with > NAN_THRESHOLD NaN fraction
      3. Fill 100%-NaN channels with global training mean
      4. Spatial 3×3 median fill for remaining NaNs
      5. 2D median filter on AOD channel (speckle suppression)
      6. Append cloud mask as channel 10
    Returns (X_qc, keep_mask, global_channel_means)
    """
    N, H, W, C = X_raw.shape

    # Step 1: Build cloud/missingness mask BEFORE any filling
    # cloud_mask[i, r, c] = 1 if all channels valid at that pixel, else 0
    nan_any   = np.isnan(X_raw).any(axis=-1)      # (N, H, W) — True where ANY channel NaN
    cloud_mask = (~nan_any).astype(np.float32)     # 1=valid, 0=imputed

    # Step 2: Drop samples with > 30% NaN across all 9 channels
    nan_frac = np.isnan(X_raw).mean(axis=(1, 2, 3))   # (N,)
    keep     = nan_frac <= NAN_THRESHOLD
    X_keep   = X_raw[keep].copy()
    cloud_keep = cloud_mask[keep]

    print(f"    Dropped {(~keep).sum()} samples with NaN > {NAN_THRESHOLD*100:.0f}%")

    # Step 3: Compute global training channel means (on training set only)
    if global_means is None:
        global_means = np.nanmean(X_keep.reshape(-1, C), axis=0)

    # Step 4: Fill 100%-NaN channels with global training mean
    for i in range(len(X_keep)):
        for c_idx in range(C):
            if np.isnan(X_keep[i, :, :, c_idx]).all():
                X_keep[i, :, :, c_idx] = global_means[c_idx]

    # Step 5: Spatial 3×3 median fill for remaining NaNs
    for i in range(len(X_keep)):
        for c_idx in range(C):
            ch = X_keep[i, :, :, c_idx]
            if np.isnan(ch).any():
                nan_locs = np.isnan(ch)
                ch_filled = scipy.ndimage.generic_filter(
                    np.where(nan_locs, 0.0, ch),
                    function=np.nanmedian, size=3
                )
                ch[nan_locs] = ch_filled[nan_locs]
                X_keep[i, :, :, c_idx] = ch

    # Step 6: 2D median filter on AOD channel only (INSAT speckle noise)
    for i in range(len(X_keep)):
        X_keep[i, :, :, AOD_IDX] = medfilt2d(X_keep[i, :, :, AOD_IDX], kernel_size=3)

    # Step 7: Append cloud mask as channel 10 → shape (N, H, W, 10)
    X_qc = np.concatenate([X_keep, cloud_keep[:, :, :, np.newaxis]], axis=-1)

    return X_qc, keep, global_means


def run_step2_qc():
    """Step 2 — Apply image QC to the extracted patches."""
    print(f"\n{'='*60}")
    print(f"  STEP 2 — Image QC")
    print(f"{'='*60}\n")

    X_train_raw = np.load(PATCH_DIR / "X_image_train.npy")
    X_val_raw   = np.load(PATCH_DIR / "X_image_val.npy")
    meta_train  = pd.read_csv(PATCH_DIR / "metadata_train.csv")
    meta_val    = pd.read_csv(PATCH_DIR / "metadata_val.csv")

    print(f"  Train: {X_train_raw.shape}, Val: {X_val_raw.shape}")

    print("\n  [Train QC]")
    X_train_qc, keep_tr, global_means = qc_patches(X_train_raw, global_means=None)
    print("\n  [Val QC]")
    X_val_qc,   keep_vl, _            = qc_patches(X_val_raw,   global_means=global_means)

    # Save QC outputs
    np.save(PATCH_DIR / "X_image_train_qc.npy", X_train_qc)
    np.save(PATCH_DIR / "X_image_val_qc.npy",   X_val_qc)
    np.save(PATCH_DIR / "global_channel_means.npy", global_means)

    meta_train[keep_tr].to_csv(PATCH_DIR / "metadata_train_qc.csv", index=False)
    meta_val[keep_vl].to_csv(PATCH_DIR / "metadata_val_qc.csv", index=False)

    print(f"\n  STEP 2 DONE")
    print(f"  Train QC: {X_train_raw.shape} → {X_train_qc.shape}")
    print(f"  Val   QC: {X_val_raw.shape}   → {X_val_qc.shape}")


# ==============================================================
# STEP 3: SCALE PATCHES WITH StandardScaler
# ==============================================================

def run_step3_scale():
    """Step 3 — StandardScaler normalization of all patch channels."""
    print(f"\n{'='*60}")
    print(f"  STEP 3 — Image Scaling (StandardScaler)")
    print(f"{'='*60}\n")

    X_train = np.load(PATCH_DIR / "X_image_train_qc.npy")
    X_val   = np.load(PATCH_DIR / "X_image_val_qc.npy")

    N_train, H, W, C = X_train.shape
    N_val             = X_val.shape[0]
    flat_dim          = H * W * C    # 13×13×10 = 1690

    # Flatten, fit on train only, transform both
    X_train_flat   = X_train.reshape(N_train, flat_dim)
    X_val_flat     = X_val.reshape(N_val, flat_dim)

    print("  Fitting StandardScaler on training data...")
    scaler         = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(N_train, H, W, C).astype(np.float32)
    X_val_scaled   = scaler.transform(X_val_flat).reshape(N_val, H, W, C).astype(np.float32)

    # Save to artifacts/
    np.save(ARTIFACT_DIR / "X_image_train_scaled.npy", X_train_scaled)
    np.save(ARTIFACT_DIR / "X_image_val_scaled.npy",   X_val_scaled)
    joblib.dump(scaler, ARTIFACT_DIR / "image_scaler.pkl")

    print(f"\n  STEP 3 DONE")
    print(f"  Train scaled: {X_train_scaled.shape} | mean={X_train_scaled.mean():.4f}")
    print(f"  Val   scaled: {X_val_scaled.shape}   | mean={X_val_scaled.mean():.4f}")
    print(f"  Scaler saved → artifacts/image_scaler.pkl")


# ==============================================================
# ENTRY POINT
# ==============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  04_extract_patches.py — Patch Extraction, QC, and Scaling")
    print("=" * 60)

    # Only run extraction if raw patches don't already exist
    if not (PATCH_DIR / "X_image_train.npy").exists():
        print("\n[INFO] Raw patches not found. Running Step 1 (extraction)...")
        run_step1_extract()
    else:
        print("\n[INFO] Raw patches found in cnn_patches/. Skipping Step 1 (extraction).")

    # Only run QC if QC patches don't already exist
    if not (PATCH_DIR / "X_image_train_qc.npy").exists():
        print("\n[INFO] QC patches not found. Running Step 2 (QC)...")
        run_step2_qc()
    else:
        print("[INFO] QC patches found in cnn_patches/. Skipping Step 2 (QC).")

    # Only scale if scaled artifacts don't already exist
    if not (ARTIFACT_DIR / "X_image_train_scaled.npy").exists():
        print("\n[INFO] Scaled patches not found. Running Step 3 (scaling)...")
        run_step3_scale()
    else:
        print("[INFO] Scaled patches found in artifacts/. Skipping Step 3 (scaling).")

    print(f"\n{'='*60}")
    print(f"  04_extract_patches.py COMPLETE")
    print(f"{'='*60}")
