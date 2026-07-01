"""
01_process_cpcb.py — CPCB Ground Station Data Pipeline
=======================================================
Objective 1 | Step 1 of 5 (Data Source: CPCB)

PURPOSE:
    Processes raw CPCB hourly station CSV files into clean, daily-aggregated
    ground-truth targets used for training and validation.

DATA SOURCE:
    Raw hourly CPCB CSVs — organized by station — stored in a nested folder.
    The raw data covers 6 pollutants: PM2.5, PM10, NO2, SO2, O3, CO.

PIPELINE (4 sub-steps):

    Step 1 → FILTER (filter_cpcb_data):
        - Scans raw CPCB CSVs recursively
        - Keeps Oct/Nov/Dec 2023 and Jan 2024 rows only
        - Merges 2023+2024 files for the same station into one file
        - Output: data/cpcb_filtered_data/

    Step 2 → FILL HOURLY GAPS (fill_hourly):
        - Enforces a complete hourly DatetimeIndex (makes implicit gaps explicit)
        - Fills gaps via time-based linear interpolation
        - Adds a <col>_filled flag column (0=real, 1=interpolated)
        - Output: data/cpcb_v3_data/

    Step 3 → AGGREGATE DAILY (aggregate_daily):
        - PM2.5, PM10, NO2, SO2 → 24-hour mean (min 16 valid readings)
        - O3 → max of 8-hour rolling mean
        - CO → 8-hour rolling max + 24-hour mean
        - Output: data/cpcb_daily_data/

    Step 4 → PARSE & MERGE STATION LIST (parse_station_list):
        - Merges station metadata (lat, lon, elevation) with daily ground truth
        - Output: data/cpcb_master.csv

NOTE:
    This script requires the RAW CPCB hourly CSVs (not included in this repo
    due to size). If you already have cpcb_daily_data/ and cpcb_master.csv in
    data/, you can skip this script entirely — those outputs are already present.

USAGE:
    python 01_process_cpcb.py --raw_input /path/to/raw/cpcb_data
"""

import os
import re
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────
# PATHS — all relative to the objective_1 folder
# ─────────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).resolve().parent
DATA_DIR         = BASE_DIR / "data"
FILTERED_DIR     = DATA_DIR / "cpcb_filtered_data"
FILLED_DIR       = DATA_DIR / "cpcb_v3_data"
DAILY_DIR        = DATA_DIR / "cpcb_daily_data"
MASTER_CSV       = DATA_DIR / "cpcb_master.csv"
ERA5_CSV         = DATA_DIR / "era5_daily_stations.csv"
STATION_COORDS   = DATA_DIR / "station_coordinates.csv"

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
FILTER_2023_MONTHS = [10, 11, 12]    # Oct, Nov, Dec
FILTER_2024_MONTHS = [1]             # Jan only
TIMESTAMP_COL      = "Timestamp"
POLLUTANT_COLS     = ["PM2.5", "PM10", "O3", "SO2", "NO2", "CO"]
YEAR_PATTERN       = re.compile(r'(20\d{2})')
MIN_HOURS_MEAN     = 16              # minimum valid hourly readings for 24-hr mean
ROLLING_WINDOW     = 8              # hours for O3 and CO rolling mean
FLAG_SUFFIX        = "_filled"


# ==============================================================
# STEP 1: FILTER RAW CPCB CSVs
# ==============================================================

def get_year_from_filename(filename: str) -> int | None:
    """Extract 4-digit year from filename. Returns None if not found."""
    match = YEAR_PATTERN.search(filename)
    return int(match.group(1)) if match else None


def get_station_key(filename: str) -> str:
    """Remove year from filename to create a station-level matching key."""
    return YEAR_PATTERN.sub('', filename, count=1)


def merged_filename(filename_2023: str) -> str:
    """Replace year tag with range label for merged files."""
    return YEAR_PATTERN.sub('2023_Oct_to_2024_Jan', filename_2023, count=1)


def filtered_filename(original_filename: str, year: int) -> str:
    """Filename for a single-year filtered file (no matching counterpart found)."""
    stem   = Path(original_filename).stem
    ext    = Path(original_filename).suffix
    suffix = {2023: "_Oct_Nov_Dec", 2024: "_Jan"}.get(year, "_filtered")
    return f"{stem}{suffix}{ext}"


def load_and_filter_raw(filepath: str, year: int) -> pd.DataFrame | None:
    """Load one CPCB CSV and keep only the target months."""
    try:
        df = pd.read_csv(filepath, parse_dates=[TIMESTAMP_COL])
    except Exception as e:
        print(f"  [ERROR] Could not read '{filepath}': {e}")
        return None

    if TIMESTAMP_COL not in df.columns:
        print(f"  [ERROR] '{TIMESTAMP_COL}' column not found — skipping.")
        return None

    months   = FILTER_2023_MONTHS if year == 2023 else FILTER_2024_MONTHS
    filtered = df[df[TIMESTAMP_COL].dt.month.isin(months)].copy()

    if filtered.empty:
        print(f"  [WARN]  No rows matched target months in '{os.path.basename(filepath)}'.")
    else:
        print(f"  [OK]    {len(filtered):>6} rows kept from '{os.path.basename(filepath)}'")
    return filtered


def run_step1_filter(raw_input_root: str):
    """Step 1 — Filter & merge raw CPCB CSVs by date window."""
    input_root  = Path(raw_input_root).resolve()
    output_root = FILTERED_DIR

    if not input_root.exists():
        print(f"[FATAL] Raw CPCB input folder not found: {input_root}")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  STEP 1 — Filter Raw CPCB CSVs")
    print(f"  Input  : {input_root}")
    print(f"  Output : {output_root}")
    print(f"{'='*60}\n")

    # Catalogue every CSV indexed by (relative_folder, station_key, year)
    catalogue = defaultdict(lambda: defaultdict(dict))
    for dirpath, _, filenames in os.walk(input_root):
        for fname in sorted(filenames):
            if not fname.lower().endswith('.csv'):
                continue
            year = get_year_from_filename(fname)
            if year not in (2023, 2024):
                continue
            rel_folder  = Path(dirpath).relative_to(input_root)
            station_key = get_station_key(fname)
            catalogue[rel_folder][station_key][year] = Path(dirpath) / fname

    total_written, total_skipped = 0, 0
    for rel_folder, stations in sorted(catalogue.items()):
        out_folder = output_root / rel_folder
        out_folder.mkdir(parents=True, exist_ok=True)

        for station_key, year_map in sorted(stations.items()):
            has_2023, has_2024 = 2023 in year_map, 2024 in year_map

            if has_2023 and has_2024:
                # Both years — filter each and merge chronologically
                df_2023 = load_and_filter_raw(str(year_map[2023]), 2023)
                df_2024 = load_and_filter_raw(str(year_map[2024]), 2024)
                frames  = [f for f in [df_2023, df_2024] if f is not None and not f.empty]
                if not frames:
                    total_skipped += 1
                    continue
                merged    = pd.concat(frames, ignore_index=True).sort_values(TIMESTAMP_COL).reset_index(drop=True)
                out_fname = merged_filename(year_map[2023].name)
                merged.to_csv(out_folder / out_fname, index=False)
                total_written += 1
            elif has_2023:
                df = load_and_filter_raw(str(year_map[2023]), 2023)
                if df is None or df.empty:
                    total_skipped += 1
                    continue
                df.to_csv(out_folder / filtered_filename(year_map[2023].name, 2023), index=False)
                total_written += 1
            elif has_2024:
                df = load_and_filter_raw(str(year_map[2024]), 2024)
                if df is None or df.empty:
                    total_skipped += 1
                    continue
                df.to_csv(out_folder / filtered_filename(year_map[2024].name, 2024), index=False)
                total_written += 1

    print(f"\n{'='*60}")
    print(f"  STEP 1 DONE — {total_written} written, {total_skipped} skipped")
    print(f"{'='*60}\n")


# ==============================================================
# STEP 2: FILL HOURLY GAPS
# ==============================================================

def load_hourly(path: Path) -> pd.DataFrame | None:
    """Read CSV, parse Timestamp, reindex to complete hourly DatetimeIndex."""
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"    [ERROR] Cannot read: {e}")
        return None

    if TIMESTAMP_COL not in df.columns:
        print(f"    [ERROR] '{TIMESTAMP_COL}' column missing — skipping.")
        return None

    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df = df.set_index(TIMESTAMP_COL).sort_index()

    present_cols = [c for c in POLLUTANT_COLS if c in df.columns]
    if not present_cols:
        return None
    df = df[present_cols]

    # Reindex to expose implicit gaps as explicit NaNs
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq="1h")
    df = df.reindex(full_index)
    df.index.name = TIMESTAMP_COL
    return df


def fill_and_flag(df: pd.DataFrame) -> tuple:
    """Fill gaps with linear interpolation and add _filled flag columns."""
    was_missing = df.isna()

    # Time-based linear interpolation — safe because max gap <= 8 hrs (filtered upstream)
    filled = df.interpolate(method="time", limit_direction="both").ffill().bfill()

    out_cols     = {}
    total_filled = 0
    for col in df.columns:
        out_cols[col]             = filled[col]
        out_cols[f"{col}_filled"] = was_missing[col].astype(int)
        total_filled             += int(was_missing[col].sum())

    out_df = pd.DataFrame(out_cols, index=df.index)
    out_df.index.name = TIMESTAMP_COL
    return out_df, {"total_hours": len(df), "total_filled_hours": total_filled}


def run_step2_fill():
    """Step 2 — Fill hourly gaps via linear interpolation."""
    input_root  = FILTERED_DIR
    output_root = FILLED_DIR

    if not input_root.exists():
        print(f"[SKIP] Step 2 skipped — input not found: {input_root}")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  STEP 2 — Fill Hourly Gaps")
    print(f"  Input  : {input_root}")
    print(f"  Output : {output_root}")
    print(f"{'='*60}\n")

    total_written, total_skipped, total_filled_hours = 0, 0, 0
    for dirpath, _, filenames in os.walk(input_root):
        for fname in sorted(f for f in filenames if f.lower().endswith(".csv")):
            src = Path(dirpath) / fname
            rel = Path(dirpath).relative_to(input_root)
            dst = output_root / rel / fname
            dst.parent.mkdir(parents=True, exist_ok=True)

            print(f"  Processing: {fname}")
            df = load_hourly(src)
            if df is None:
                total_skipped += 1
                continue

            out_df, stats = fill_and_flag(df)
            out_df.to_csv(dst)
            total_filled_hours += stats["total_filled_hours"]
            total_written      += 1
            print(f"    [OK]  {stats['total_hours']} hrs, {stats['total_filled_hours']} filled")

    print(f"\n{'='*60}")
    print(f"  STEP 2 DONE — {total_written} files, {total_filled_hours} hours filled total")
    print(f"{'='*60}\n")


# ==============================================================
# STEP 3: AGGREGATE TO DAILY (NAAQS standards)
# ==============================================================

def aggregate_file(src_path: Path) -> pd.DataFrame | None:
    """Aggregate one hourly CSV to daily following CPCB NAAQS standards."""
    try:
        df = pd.read_csv(src_path, parse_dates=[TIMESTAMP_COL])
    except Exception as e:
        print(f"    [ERROR] Cannot read '{src_path.name}': {e}")
        return None

    if TIMESTAMP_COL not in df.columns:
        return None

    df = df.set_index(TIMESTAMP_COL).sort_index()

    available_mean_cols = [c for c in ["PM2.5", "PM10", "NO2", "SO2"] if c in df.columns]
    flag_cols           = [c for c in df.columns if c.endswith(FLAG_SUFFIX)]
    has_o3              = "O3" in df.columns
    has_co              = "CO" in df.columns

    # Revert interpolated (filled) hours to NaN so they don't corrupt 16-hr minimum
    for col in available_mean_cols + (["O3"] if has_o3 else []) + (["CO"] if has_co else []):
        flag_col = f"{col}{FLAG_SUFFIX}"
        if flag_col in df.columns:
            df.loc[df[flag_col] == 1, col] = np.nan

    daily_parts = []

    # PM2.5, PM10, NO2, SO2 → 24-hr mean (min 16 valid hourly readings)
    if available_mean_cols:
        safe_mean  = lambda x: x.dropna().mean() if x.dropna().count() >= MIN_HOURS_MEAN else np.nan
        mean_daily = df[available_mean_cols].resample("D").agg(safe_mean)
        daily_parts.append(mean_daily)

    # O3 → max of 8-hr rolling mean (official CPCB metric)
    if has_o3:
        o3_rolling = df["O3"].rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean()
        daily_parts.append(o3_rolling.resample("D").max().rename("O3_8hr_max"))

    # CO → 8-hr rolling max + 24-hr mean (both used for AQI)
    if has_co:
        co_rolling   = df["CO"].rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW).mean()
        safe_co_mean = lambda x: x.dropna().mean() if x.dropna().count() >= MIN_HOURS_MEAN else np.nan
        daily_parts.append(co_rolling.resample("D").max().rename("CO_8hr_max"))
        daily_parts.append(df["CO"].resample("D").agg(safe_co_mean).rename("CO_24hr_mean"))

    # Flag columns → hours interpolated per day
    if flag_cols:
        flag_daily = df[flag_cols].resample("D").sum().rename(
            columns={c: c.replace(FLAG_SUFFIX, "_hrs_interpolated") for c in flag_cols})
        daily_parts.append(flag_daily)

    if not daily_parts:
        return None

    daily = pd.concat(daily_parts, axis=1)
    daily.index.name = "Date"
    daily.index      = daily.index.strftime("%Y-%m-%d")
    return daily


def run_step3_aggregate():
    """Step 3 — Aggregate hourly CSVs to NAAQS-compliant daily values."""
    input_root  = FILLED_DIR
    output_root = DAILY_DIR

    if not input_root.exists():
        print(f"[SKIP] Step 3 skipped — input not found: {input_root}")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  STEP 3 — Daily Aggregation")
    print(f"  Input  : {input_root}")
    print(f"  Output : {output_root}")
    print(f"{'='*60}\n")

    total_written, total_skipped = 0, 0
    for dirpath, _, filenames in os.walk(input_root):
        for fname in sorted(f for f in filenames if f.lower().endswith(".csv")):
            src = Path(dirpath) / fname
            rel = Path(dirpath).relative_to(input_root)
            dst = output_root / rel / fname
            dst.parent.mkdir(parents=True, exist_ok=True)

            print(f"  Processing: {fname}")
            daily = aggregate_file(src)
            if daily is None or daily.empty:
                total_skipped += 1
                continue
            daily.to_csv(dst)
            print(f"    [OK]  {len(daily)} daily rows")
            total_written += 1

    print(f"\n{'='*60}")
    print(f"  STEP 3 DONE — {total_written} files written, {total_skipped} skipped")
    print(f"{'='*60}\n")


# ==============================================================
# STEP 4: MERGE STATION METADATA (coordinates + ERA5 features)
# ==============================================================

def run_step4_merge_master():
    """
    Step 4 — Merge ERA5 meteorological features and compute Ventilation Index.
    Reads cpcb_master.csv + era5_daily_stations.csv and saves updated cpcb_master.csv.

    Ventilation Index: VI = BLH × sqrt(u10² + v10²)
    """
    print(f"\n{'='*60}")
    print(f"  STEP 4 — Merge ERA5 into cpcb_master.csv")
    print(f"{'='*60}\n")

    if not MASTER_CSV.exists():
        print(f"[SKIP] cpcb_master.csv not found at {MASTER_CSV} — skipping.")
        return
    if not ERA5_CSV.exists():
        print(f"[SKIP] era5_daily_stations.csv not found at {ERA5_CSV} — skipping.")
        return

    master = pd.read_csv(MASTER_CSV)
    era5   = pd.read_csv(ERA5_CSV)

    print(f"  cpcb_master  : {master.shape[0]:,} rows x {master.shape[1]} cols")
    print(f"  era5_daily   : {era5.shape[0]:,} rows x {era5.shape[1]} cols")

    # Map ERA5 integer station IDs (0-97) to alphabetically sorted station names
    unique_stations = sorted(master["station"].dropna().unique())
    id_to_station   = {i: name for i, name in enumerate(unique_stations)}

    era5["station_name"] = era5["station"].map(id_to_station)

    # Compute Ventilation Index: BLH × wind_speed
    era5["wind_speed"] = np.sqrt(era5["u10"] ** 2 + era5["v10"] ** 2)
    era5["vi"]         = era5["blh"] * era5["wind_speed"]

    # Normalize date formats before joining
    master["date_parsed"] = pd.to_datetime(master["date"], format="mixed", dayfirst=True).dt.normalize()
    era5["date_parsed"]   = pd.to_datetime(era5["date"],   format="mixed", dayfirst=True).dt.normalize()

    ERA5_COLS   = ["date_parsed", "station_name", "t2m", "blh", "u10", "v10",
                   "sp", "tp", "msdwswrf", "rh", "wind_speed", "vi"]
    era5_merge  = era5[ERA5_COLS].copy()

    before_rows = len(master)
    merged = master.merge(era5_merge, left_on=["date_parsed", "station"],
                          right_on=["date_parsed", "station_name"], how="left")
    merged.drop(columns=["date_parsed", "station_name"], errors="ignore", inplace=True)

    assert len(merged) == before_rows, "Row count changed after merge!"

    merged.to_csv(MASTER_CSV, index=False)
    vi_coverage = 100 * merged["vi"].notna().sum() / len(merged)
    print(f"  VI coverage: {vi_coverage:.1f}%")
    print(f"\n  [OK] Saved updated cpcb_master.csv → {MASTER_CSV}")
    print(f"       Final shape: {merged.shape[0]:,} rows x {merged.shape[1]} cols")

    print(f"\n{'='*60}")
    print(f"  STEP 4 DONE")
    print(f"{'='*60}\n")


# ==============================================================
# ENTRY POINT
# ==============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="01_process_cpcb.py — Full CPCB ground truth pipeline."
    )
    parser.add_argument(
        "--raw_input", "-i",
        default=None,
        help=(
            "Path to folder containing raw hourly CPCB CSVs. "
            "Required only for Steps 1-3. If data/cpcb_daily_data/ already "
            "exists, Step 4 can run standalone."
        )
    )
    args = parser.parse_args()

    if args.raw_input:
        run_step1_filter(args.raw_input)
        run_step2_fill()
        run_step3_aggregate()
    else:
        print("[INFO] --raw_input not provided. Running Step 4 only (merge ERA5 + VI).")
        print("[INFO] Steps 1-3 require raw hourly CPCB CSVs.")

    run_step4_merge_master()
    print("\nAll CPCB processing steps complete.")
