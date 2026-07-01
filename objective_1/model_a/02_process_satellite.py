"""
02_process_satellite.py — Satellite Data Preprocessing Pipeline
================================================================
Objective 1 | Step 2 of 5 (Data Sources: Sentinel-5P + INSAT-3D)

PURPOSE:
    Regrid all satellite data from native high-resolution swath/tile format
    to a common 0.25° × 0.25° reference grid covering India.

DATA SOURCES:
    1. Sentinel-5P TROPOMI (NO2, SO2, CO)
       - Daily .tif files at ~3.5 km resolution
       - Named: S5P_NO2_YYYYMMDD.tif, etc.

    2. INSAT-3D AOD (Aerosol Optical Depth)
       - Twice-daily .tif files at ~4 km resolution
       - Named: 3DIMG_DDMMMYYYY_HHMM_..._AOD.tif

REGRIDDING METHOD:
    Block-mean spatial aggregation:
    - For each 0.25° cell, average all source pixels whose centre falls
      within that cell, ignoring nodata values.
    - Cells with no valid source pixels receive NaN.
    - For AOD: all daily time slots are stacked and nan-mean averaged.

REFERENCE GRID (India bounding box):
    Lat: 8.0° N – 38.0° N  (120 cells N→S)
    Lon: 67.0° E – 98.0° E (124 cells W→E)
    Resolution: 0.25°

OUTPUT (saved to data/regridded/):
    data/regridded/NO2/no2_YYYYMMDD.npy  — shape (120, 124)
    data/regridded/SO2/so2_YYYYMMDD.npy
    data/regridded/CO/co_YYYYMMDD.npy
    data/regridded/AOD/aod_YYYYMMDD.npy
    data/regridded/grid_spec.json         — lat/lon axis metadata

NOTE:
    The regridded/ folder is already present in this repository. Re-running
    this script is only necessary if you have new raw satellite files to process.

USAGE:
    python 02_process_satellite.py
    python 02_process_satellite.py --sentinel_dir /path/to/Sentinel-5p --aod_dir /path/to/INSAT_AOD
"""

import re
import json
import warnings
import argparse
import numpy as np
import rasterio
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

# ─────────────────────────────────────────────────────────────────
# PATHS — all relative to the objective_1 folder
# ─────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
DATA_DIR     = BASE_DIR / "data"
OUT_DIR      = DATA_DIR / "regridded"

# ─────────────────────────────────────────────────────────────────
# 0.25-DEGREE REFERENCE GRID (India bounding box)
# ─────────────────────────────────────────────────────────────────
LAT_MIN, LAT_MAX = 8.0,  38.0    # south → north
LON_MIN, LON_MAX = 67.0, 98.0    # west  → east
RES  = 0.25

# Grid cell centres
LATS = np.arange(LAT_MAX - RES / 2, LAT_MIN - RES / 2, -RES)   # N→S, shape (120,)
LONS = np.arange(LON_MIN + RES / 2, LON_MAX + RES / 2,  RES)   # W→E, shape (124,)
NLAT, NLON = len(LATS), len(LONS)

AOD_NODATA = -999.0


# ─────────────────────────────────────────────────────────────────
# HELPER: Save grid specification JSON
# ─────────────────────────────────────────────────────────────────

def save_grid_spec():
    """Save the shared 0.25° grid specification so all downstream scripts can load it."""
    spec = {
        "lat_min": LAT_MIN, "lat_max": LAT_MAX,
        "lon_min": LON_MIN, "lon_max": LON_MAX,
        "resolution_deg": RES,
        "nlat": NLAT, "nlon": NLON,
        "lats": LATS.tolist(),
        "lons": LONS.tolist(),
        "description": "0.25-deg grid, N→S lats, W→E lons",
    }
    spec_path = OUT_DIR / "grid_spec.json"
    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"[grid_spec] Saved to {spec_path}")


# ─────────────────────────────────────────────────────────────────
# HELPER: Regrid a single .tif file to the 0.25° grid
# ─────────────────────────────────────────────────────────────────

def regrid_tif(src_path: Path, nodata_val=None) -> np.ndarray:
    """
    Read a single .tif, aggregate pixels into 0.25° cells via block-mean.
    Returns float32 array of shape (NLAT, NLON), NaN where no valid data.
    """
    with rasterio.open(src_path) as src:
        data      = src.read(1).astype(np.float64)
        transform = src.transform
        nd        = src.nodata if nodata_val is None else nodata_val

    # Mask nodata values
    if nd is not None:
        data[data == nd] = np.nan

    # Compute pixel centre coordinates
    rows, cols = np.indices(data.shape)
    px_lons    = transform.c + (cols + 0.5) * transform.a
    px_lats    = transform.f + (rows + 0.5) * transform.e   # e is negative for N→S

    # Only keep pixels within India bounding box
    in_box = ((px_lats >= LAT_MIN) & (px_lats <= LAT_MAX) &
              (px_lons >= LON_MIN) & (px_lons <= LON_MAX))

    out = np.full((NLAT, NLON), np.nan, dtype=np.float64)

    if not in_box.any():
        return out.astype(np.float32)

    valid_vals = data[in_box]
    valid_lats = px_lats[in_box]
    valid_lons = px_lons[in_box]

    # Map pixel lat/lon → grid row/col indices
    row_idx = np.clip(np.floor((LAT_MAX - valid_lats) / RES).astype(int), 0, NLAT - 1)
    col_idx = np.clip(np.floor((valid_lons - LON_MIN) / RES).astype(int), 0, NLON - 1)

    # Block-mean aggregation (nan-safe)
    sums   = np.zeros((NLAT, NLON), dtype=np.float64)
    counts = np.zeros((NLAT, NLON), dtype=np.float64)

    good = ~np.isnan(valid_vals)
    np.add.at(sums,   (row_idx[good], col_idx[good]), valid_vals[good])
    np.add.at(counts, (row_idx[good], col_idx[good]), 1)

    mask       = counts > 0
    out[mask]  = sums[mask] / counts[mask]

    return out.astype(np.float32)


# ─────────────────────────────────────────────────────────────────
# DATE PARSERS
# ─────────────────────────────────────────────────────────────────

def parse_s5p_date(filename: str) -> str | None:
    """Extract YYYYMMDD from S5P_NO2_20231001.tif style filenames."""
    m = re.search(r'(\d{8})', filename)
    return m.group(1) if m else None


def parse_aod_date(filename: str) -> str | None:
    """
    Extract YYYYMMDD from 3DIMG_01DEC2023_0800_..._AOD.tif style filenames.
    Handles month abbreviations: JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC.
    """
    m = re.search(r'(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{4})',
                  filename, re.IGNORECASE)
    if not m:
        return None
    MONTHS = {"JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
              "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
              "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12"}
    day, mon, yr = m.group(1), m.group(2).upper(), m.group(3)
    return f"{yr}{MONTHS[mon]}{day}"


# ─────────────────────────────────────────────────────────────────
# STEP 1: REGRID SENTINEL-5P (NO2, SO2, CO)
# ─────────────────────────────────────────────────────────────────

def regrid_sentinel(sentinel_dir: Path):
    """
    Regrid Sentinel-5P TROPOMI .tif files for NO2, SO2, and CO.
    Expects sentinel_dir/{NO2,SO2,CO}/*.tif structure.
    """
    print(f"\n{'='*60}")
    print(f"  STEP 1 — Regridding Sentinel-5P (NO2, SO2, CO)")
    print(f"  Source : {sentinel_dir}")
    print(f"{'='*60}")

    PRODUCTS = {"NO2": sentinel_dir / "NO2",
                "SO2": sentinel_dir / "SO2",
                "CO":  sentinel_dir / "CO"}

    for product, folder in PRODUCTS.items():
        out_folder = OUT_DIR / product
        out_folder.mkdir(parents=True, exist_ok=True)

        tif_files = sorted(folder.glob("*.tif"))
        print(f"\n  [{product}] Found {len(tif_files)} .tif files in {folder}")

        done, skipped = 0, 0
        for tif in tif_files:
            date_str = parse_s5p_date(tif.stem)
            if date_str is None:
                print(f"    [WARN] Cannot parse date from {tif.name}")
                skipped += 1
                continue

            out_path = out_folder / f"{product.lower()}_{date_str}.npy"
            if out_path.exists():
                done += 1
                continue  # Already processed — skip

            try:
                grid    = regrid_tif(tif)
                nan_pct = np.isnan(grid).mean() * 100
                np.save(out_path, grid)
                print(f"    [OK] {tif.name} → shape {grid.shape}, NaN={nan_pct:.1f}%")
                done += 1
            except Exception as e:
                print(f"    [ERROR] {tif.name}: {e}")
                skipped += 1

        print(f"  [{product}] Done: {done}, Skipped: {skipped}")


# ─────────────────────────────────────────────────────────────────
# STEP 2: REGRID INSAT-3D AOD
# ─────────────────────────────────────────────────────────────────

def regrid_aod(aod_dir: Path):
    """
    Regrid INSAT-3D AOD .tif files.
    Multiple daily slots (0800, 0830) are nan-mean averaged into one daily array.
    """
    print(f"\n{'='*60}")
    print(f"  STEP 2 — Regridding INSAT-3D AOD")
    print(f"  Source : {aod_dir}")
    print(f"{'='*60}")

    out_folder = OUT_DIR / "AOD"
    out_folder.mkdir(parents=True, exist_ok=True)

    tif_files  = sorted(aod_dir.glob("3DIMG_*_AOD.tif"))
    print(f"\n  [AOD] Found {len(tif_files)} .tif files")

    # Group all time slots by date — then daily-average them
    date_files = defaultdict(list)
    for tif in tif_files:
        date_str = parse_aod_date(tif.stem)
        if date_str:
            date_files[date_str].append(tif)
        else:
            print(f"    [WARN] Cannot parse date from {tif.name}")

    done, skipped = 0, 0
    for date_str, files in sorted(date_files.items()):
        out_path = out_folder / f"aod_{date_str}.npy"
        if out_path.exists():
            done += 1
            continue

        try:
            grids = [regrid_tif(f, nodata_val=AOD_NODATA) for f in files]
            # Stack all slots and take nan-mean → daily representative AOD
            daily   = np.nanmean(np.stack(grids, axis=0), axis=0).astype(np.float32)
            nan_pct = np.isnan(daily).mean() * 100
            np.save(out_path, daily)
            print(f"    [OK] {date_str} ({len(files)} slots) → shape {daily.shape}, NaN={nan_pct:.1f}%")
            done += 1
        except Exception as e:
            print(f"    [ERROR] {date_str}: {e}")
            skipped += 1

    print(f"\n  [AOD] Done: {done}, Skipped: {skipped}")


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="02_process_satellite.py — Regrid Sentinel-5P and INSAT-3D to 0.25° grid."
    )
    parser.add_argument("--sentinel_dir", type=str, default=None,
                        help="Path to Sentinel-5P folder containing NO2/, SO2/, CO/ subdirs.")
    parser.add_argument("--aod_dir",      type=str, default=None,
                        help="Path to INSAT-3D folder containing 3DIMG_*_AOD.tif files.")
    args = parser.parse_args()

    if args.sentinel_dir is None and args.aod_dir is None:
        print("[ERROR] Provide at least one of --sentinel_dir or --aod_dir.")
        print("        Example: python 02_process_satellite.py")
        print("                     --sentinel_dir D:/raw/Sentinel-5p")
        print("                     --aod_dir D:/raw/INSAT_AOD")
        raise SystemExit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_grid_spec()

    if args.sentinel_dir:
        regrid_sentinel(Path(args.sentinel_dir))

    if args.aod_dir:
        regrid_aod(Path(args.aod_dir))

    print(f"\n{'='*60}")
    print(f"  ALL DONE — Regridded arrays saved to: {OUT_DIR}")
    print(f"{'='*60}")
