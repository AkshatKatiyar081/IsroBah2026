# Objective 1 — Hybrid CNN-XGBoost AQI Prediction Model

## Overview

This folder is a self-contained, fully reproducible implementation of a **Hybrid CNN-XGBoost model** for predicting 6 air pollutant concentrations across all of India at 0.25° spatial resolution (120×124 grid).

**Predicted Pollutants:** PM2.5, PM10, NO2, SO2, O3, CO  
**Spatial Coverage:** Full India grid (8°N–38°N, 67°E–98°E) at 0.25° resolution  
**Temporal Coverage:** October 2023 – January 2024 (123 days)

---

## Model Architecture

```
Multi-source Inputs (3 satellite/met data streams)
        ↓
    13×13 Image Patches (10 channels)
        ↓
    CNN (Conv2D + BN + LeakyReLU → GAP → Dense)  × 6 pollutants
        ↓
    CNN Prediction + 64-dim GAP Embedding
        ↓
    XGBoost Residual Corrector (GAP + CNN_pred + tabular features)
        ↓
    Final Prediction = CNN_pred + XGBoost_residual
        ↓
    Full-Grid Inference (120×124 cells, 123 days) + Gaussian Smoothing
        ↓
    AQI Calculation (CPCB NAAQS Breakpoints)
```

### CNN Architecture (per pollutant)
- Input: 13×13×10 patch (CO: 17×17×10)
- Conv2D(32) + BatchNorm + LeakyReLU → MaxPool
- Conv2D(64) + BatchNorm + LeakyReLU → MaxPool
- Conv2D(64) + BatchNorm + LeakyReLU → GlobalAveragePooling2D (64-dim)
- Dropout(0.3) → Dense(1)
- Loss: Huber | Optimizer: Adam(lr=0.001, clipnorm=1.0)

### XGBoost Architecture (per pollutant)
- Feature matrix: GAP_lag(64) + CNN_pred(1) + tabular(N) = 66–77 features
- PM2.5, PM10: relative residual target — `(y - cnn_pred) / (y + 1)`
- Others: absolute residual — `y - cnn_pred`
- Hyperparameter tuning: RandomizedSearchCV (n_iter=30, 5-fold)

---

## 10 Input Channels

| # | Channel | Source         | Description |
|---|---------|----------------|-------------|
| 0 | AOD     | INSAT-3D       | Aerosol Optical Depth (daily mean of all slots) |
| 1 | NO2     | Sentinel-5P    | Tropospheric NO2 column (mol/m²) |
| 2 | SO2     | Sentinel-5P    | SO2 column (mol/m²) |
| 3 | CO      | Sentinel-5P    | CO column (mol/m²) |
| 4 | U10     | ERA5           | U-component wind at 10m (m/s) |
| 5 | V10     | ERA5           | V-component wind at 10m (m/s) |
| 6 | T2M     | ERA5           | Temperature at 2m (K) |
| 7 | BLH     | ERA5           | Boundary Layer Height (m) |
| 8 | RH      | ERA5           | Relative Humidity (%, derived from d2m+t2m) |
| 9 | MASK    | Derived        | Cloud/missingness binary mask (0=imputed, 1=valid) |

---

## Folder Structure

```
objective_1/
├── channels.py              ← Single source of truth for channel/pollutant config
├── 01_process_cpcb.py       ← CPCB hourly → filtered → gap-filled → daily aggregation
├── 02_process_satellite.py  ← Sentinel-5P + INSAT-3D regrid to 0.25°
├── 03_process_era5.py       ← ERA5 NetCDF → daily 0.25° .npy grids + RH derivation
├── 04_extract_patches.py    ← Patch extraction, QC (cloud mask, NaN fill), scaling
├── 05_build_tabular.py      ← XGBoost tabular features + GAP lag construction
├── run_model.py             ← Master script: Phases 6–12 (CNN→XGBoost→Inference→AQI)
│
├── data/
│   ├── regridded/           ← 0.25° daily .npy grids (NO2, SO2, CO, AOD, ERA5)
│   │   ├── grid_spec.json   ← Shared grid specification (lat/lon axes)
│   │   ├── NO2/             ← no2_YYYYMMDD.npy
│   │   ├── SO2/
│   │   ├── CO/
│   │   ├── AOD/
│   │   └── ERA5/            ← u10, v10, t2m, blh, rh ... per day
│   ├── cpcb_master.csv      ← Daily station ground truth + ERA5 features merged
│   ├── cpcb_xgboost_features.csv  ← Lag features + tabular features
│   ├── cpcb_daily_data/     ← Per-station daily CSVs (aggregated)
│   └── gadm41_IND_1.json/   ← India boundary GeoJSON (for map overlays)
│
├── cnn_patches/
│   ├── X_image_train.npy           ← Raw patches (N_train, 13, 13, 9)
│   ├── X_image_train_qc.npy        ← After QC + cloud mask (N_train, 13, 13, 10)
│   ├── patches_train_CO_17x17_scaled.npy  ← CO extended patches
│   └── metadata_train_qc.csv       ← Station, date, lat, lon per sample
│
├── artifacts/
│   ├── cnn_{pollutant}.keras        ← Trained CNN models (6 total)
│   ├── xgb_{pollutant}.json         ← Trained XGBoost models (6 total)
│   ├── X_image_train_scaled.npy     ← Scaled patches (model input)
│   ├── X_tab_train.npy              ← Tabular features
│   ├── y_train.npy                  ← Ground truth targets (N, 6)
│   ├── gap_train_{pollutant}.npy    ← 64-dim GAP embeddings
│   ├── image_scaler.pkl             ← Fitted StandardScaler (for inference)
│   ├── tabular_means.json           ← Training means (for inference gap fill)
│   ├── ceiling_clips.json           ← 2× training max (for inference clipping)
│   └── y_column_order.json          ← Pollutant → column index mapping
│
├── results/
│   ├── all_predictions.pkl          ← Full India grid predictions (dict[date → {pollutant: grid}])
│   ├── validation_predictions.csv   ← Per-station validation predictions
│   ├── evaluation_metrics.json      ← RMSE, MAE, R² per pollutant
│   └── shap_{pollutant}.png         ← SHAP beeswarm + importance plots
│
└── maps/
    ├── {pollutant}/                  ← Daily concentration maps (PNG)
    └── AQI/
        ├── AQI_{date}.png            ← Daily AQI maps
        ├── AQI_Seasonal_Mean.png     ← Seasonal average AQI
        ├── AQI_Worst_Day.png         ← Worst-day AQI across season
        └── AQI_Dominant_Pollutant.png  ← Most frequent AQI driver per pixel
```

---

## How to Run

### Quick Start (if all data is already processed)

```bash
# Step 1: Navigate to the objective_1 folder
cd "d:/data processing/objective_1"

# Step 2: Train the model (Phases 6-7: CNN + XGBoost) and run inference + mapping (Phases 8-12)
python run_model.py

# Or run a specific phase only:
python run_model.py --phase 8     # Inference only
python run_model.py --phase 9     # Evaluation only
python run_model.py --phase 12    # AQI maps only

# Skip CNN/XGBoost training (if models already in artifacts/):
python run_model.py --skip_training
```

### Full Pipeline (from raw data)

```bash
# Step 1: Process CPCB ground truth (requires raw CPCB hourly CSVs)
python 01_process_cpcb.py --raw_input /path/to/raw/cpcb_data

# Step 2: Regrid satellite data (requires Sentinel-5P + INSAT-3D raw TIFs)
python 02_process_satellite.py --sentinel_dir /path/to/Sentinel-5p --aod_dir /path/to/INSAT

# Step 3: Regrid ERA5 data (requires ERA5 NetCDF files)
python 03_process_era5.py --era5_dir /path/to/ERA5

# Step 4: Extract CNN patches, apply QC, and scale
python 04_extract_patches.py

# Step 5: Build tabular features for XGBoost
python 05_build_tabular.py

# Step 6: Train model + inference + maps
python run_model.py
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Separate CNN per pollutant | Different pollutants have different spatial signatures |
| CO uses 17×17 patches | CO has longer-range transport, needs wider spatial context |
| Relative residual for PM2.5/PM10 | Prevents saturation at high concentrations |
| GAP lag features in XGBoost | Spatial memory from previous day improves temporal tracking |
| Gaussian smoothing in inference | Prevents tile boundary seams in the final maps |
| `channels.py` single source of truth | Changes to channel ordering update everywhere automatically |

---

## Model Performance (Validation Set)

Results are saved to `results/evaluation_metrics.json` after running Phase 9.

| Pollutant | RMSE | MAE | R² |
|-----------|------|-----|----|
| PM2.5     | —    | —   | —  |
| PM10      | —    | —   | —  |
| NO2       | —    | —   | —  |
| SO2       | —    | —   | —  |
| O3        | —    | —   | —  |
| CO        | —    | —   | —  |

*Run `python run_model.py --phase 9` to populate this table.*

---

## Dependencies

```
tensorflow >= 2.13
xgboost >= 2.0
scikit-learn
numpy
pandas
rasterio
xarray
netCDF4
scipy
shap
matplotlib
geopandas
joblib
```

Install all dependencies:
```bash
pip install tensorflow xgboost scikit-learn numpy pandas rasterio xarray netCDF4 scipy shap matplotlib geopandas joblib
```
