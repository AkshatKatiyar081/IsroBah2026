"""
03_process_era5.py — ERA5 Meteorological Data Pipeline
=======================================================
Objective 1 | Step 3 of 5 (Data Source: ERA5 Reanalysis)

PURPOSE:
    Regrid and aggregate ERA5 hourly reanalysis NetCDF data into daily
    .npy grids on the shared 0.25° reference grid.

DATA SOURCE:
    ERA5 hourly NetCDF files (Copernicus Climate Data Store download).
    Variables: u10, v10, t2m, d2m, blh, sp, tp, msdwswrf (solar radiation)

PROCESSING STEPS:
    1. Load all ERA5 .nc files via xarray
    2. Resample to daily mean (instantaneous fields)
    3. Derive Relative Humidity (RH) from t2m and d2m using the Magnus formula
    4. Interpolate to the shared 0.25° grid_spec
    5. Export each variable as individual daily .npy files

OUTPUT (saved to data/regridded/):
    data/regridded/ERA5/U10/u10_YYYYMMDD.npy  — shape (120, 124)
    data/regridded/ERA5/V10/v10_YYYYMMDD.npy
    data/regridded/ERA5/T2M/t2m_YYYYMMDD.npy
    data/regridded/ERA5/BLH/blh_YYYYMMDD.npy
    data/regridded/ERA5/RH/rh_YYYYMMDD.npy
    etc.

NOTE:
    The data/regridded/ folder is already present in this repository.
    Re-running this script is only necessary if you have new ERA5 downloads.

USAGE:
    python 03_process_era5.py --era5_dir /path/to/ERA5/nc_files
    python 03_process_era5.py --era5_dir D:/data/ERA5
"""

import os
import json
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# PATHS — all relative to the objective_1 folder
# ─────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / "data"
REGRID_DIR = DATA_DIR / "regridded"
OUT_DIR    = REGRID_DIR / "ERA5"

# Variables to export as individual daily .npy grids
VARS_TO_EXPORT = ["u10", "v10", "t2m", "d2m", "blh", "sp", "tp", "msdwswrf", "rh"]


def run_era5_regrid(era5_dir: str):
    """
    Load ERA5 NetCDF files, resample to daily, compute RH,
    interpolate to the shared 0.25° grid, and export as .npy files.
    """
    era5_path = Path(era5_dir)

    # ── Load grid spec ────────────────────────────────────────────
    grid_spec_path = REGRID_DIR / "grid_spec.json"
    if not grid_spec_path.exists():
        print(f"[ERROR] grid_spec.json not found at {grid_spec_path}")
        print("        Run 02_process_satellite.py first to generate it.")
        return

    with open(grid_spec_path) as f:
        grid_spec = json.load(f)

    target_lats = xr.DataArray(grid_spec["lats"], dims="latitude")
    target_lons = xr.DataArray(grid_spec["lons"], dims="longitude")

    # ── Find all NetCDF files in the ERA5 directory ──────────────
    nc_files = sorted(glob.glob(str(era5_path / "**" / "*.nc"), recursive=True))
    if not nc_files:
        nc_files = sorted(glob.glob(str(era5_path / "*.nc")))

    if not nc_files:
        print(f"[ERROR] No .nc files found in: {era5_path}")
        return

    print(f"\n{'='*60}")
    print(f"  ERA5 Regrid Pipeline")
    print(f"  Source : {era5_path}")
    print(f"  Files  : {len(nc_files)} NetCDF files found")
    print(f"  Output : {OUT_DIR}")
    print(f"{'='*60}\n")

    # ── Load all ERA5 files as a combined dataset ─────────────────
    print("[1] Loading ERA5 datasets...")
    ds = xr.open_mfdataset(nc_files, combine="by_coords")
    print(f"    Dataset variables: {list(ds.data_vars)}")
    print(f"    Time range: {str(ds.time.values[0])[:10]} → {str(ds.time.values[-1])[:10]}")

    # ── Daily aggregation: resample all fields to daily mean ──────
    print("[2] Resampling to daily mean...")
    time_dim = "valid_time" if "valid_time" in ds.dims else "time"
    ds_daily = ds.resample({time_dim: "1D"}).mean()

    # ── Derive Relative Humidity from Magnus formula ──────────────
    # RH = 100 * exp(17.625*Td/(243.04+Td) - 17.625*T/(243.04+T))
    # where T and Td are in degrees Celsius
    print("[3] Computing Relative Humidity (RH) from t2m and d2m...")
    T_celsius  = ds_daily["t2m"] - 273.15    # Kelvin → Celsius
    Td_celsius = ds_daily["d2m"] - 273.15
    ds_daily["rh"] = 100.0 * np.exp(
        (17.625 * Td_celsius) / (243.04 + Td_celsius) -
        (17.625 * T_celsius)  / (243.04 + T_celsius)
    )
    ds_daily["rh"].attrs["units"] = "percent"
    ds_daily["rh"].attrs["long_name"] = "Relative Humidity (derived)"

    # ── Interpolate to the shared 0.25° India grid ────────────────
    print("[4] Interpolating to 0.25° India grid...")
    ds_interp = ds_daily.interp(latitude=target_lats, longitude=target_lons, method="linear")

    # ── Export each variable as daily .npy files ──────────────────
    print("[5] Exporting daily .npy grids...\n")
    time_values = ds_interp[time_dim].values

    for var in VARS_TO_EXPORT:
        if var not in ds_interp:
            print(f"  [SKIP] Variable '{var}' not in dataset — skipping.")
            continue

        # Each variable gets its own subfolder
        var_dir = OUT_DIR / var.upper()
        var_dir.mkdir(parents=True, exist_ok=True)

        done, skipped = 0, 0
        for i, t in enumerate(time_values):
            dt_str   = pd.to_datetime(t).strftime("%Y%m%d")
            out_path = var_dir / f"{var.lower()}_{dt_str}.npy"

            if out_path.exists():
                done += 1
                continue

            try:
                grid = ds_interp[var].isel({time_dim: i}).values.astype(np.float32)
                np.save(out_path, grid)
                done += 1
            except Exception as e:
                print(f"    [ERROR] {var} {dt_str}: {e}")
                skipped += 1

        print(f"  [{var.upper():>8}] Exported {done} days, skipped {skipped}")

    print(f"\n{'='*60}")
    print(f"  ERA5 DONE — Daily grids saved to: {OUT_DIR}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="03_process_era5.py — Regrid ERA5 NetCDF to daily 0.25° .npy grids."
    )
    parser.add_argument(
        "--era5_dir", type=str, required=True,
        help="Path to the folder containing ERA5 .nc files (can be nested)."
    )
    args = parser.parse_args()
    run_era5_regrid(args.era5_dir)
