# ISRO Hackathon 2026 — Team Airheads

This repository contains the complete implementation for our submission to the **ISRO Hackathon 2026**. Our work tackles complex, large-scale atmospheric problems using state-of-the-art satellite data processing, machine learning, and physics-informed neural networks. The repository is structured into two main objectives:

---

## 🌍 Objective 1: High-Resolution 6-Pollutant AQI Prediction
**Goal:** Predict a comprehensive 6-pollutant Air Quality Index (PM2.5, PM10, NO₂, SO₂, O₃, CO) at a high spatial resolution (0.25° grid) over India by fusing Sentinel-5P satellite retrievals, ERA5 meteorological reanalysis, and CPCB ground network data.

### 🧠 The Two-Model Strategy
We tackled this problem by developing two parallel architectural philosophies:

#### Model A (Implemented & Evaluated)
A fast, robust **Hybrid CNN + XGBoost Pipeline** designed for our initial 4-month data window (Oct 2023 – Jan 2024).
- **CNN:** Extracts spatial embeddings (13x13 patches) from satellite raster data using Global Average Pooling (GAP).
- **XGBoost:** Acts as a residual corrector, integrating the CNN embeddings with tabular meteorological data (BLH, solar radiation, wind).
- **Performance:** Achieved Pearson correlations of **0.81–0.88** for primary pollutants (PM2.5, PM10, NO₂) using a strict geographic block-split to prevent spatial data leakage.

#### Model B (The Path Forward: Physics-Informed ConvLSTM)
While Model A is highly accurate, it lacks temporal memory and physical constraints. We designed **Model B**, a physics-constrained **ConvLSTM** architecture, to address this:
- **Temporal Memory:** Uses a 14-day lookback window to track the movement of pollution plumes across the subcontinent.
- **Physics-Informed Loss:** Encodes an **Advection-Diffusion PDE** directly into the loss function, forcing the neural network to obey the laws of atmospheric transport and wind dispersion.
- **Status:** Architecture is fully built and tested for structural correctness (see `AQI_Objective1_Physics_Informed_ConvLSTM.ipynb` in the technical report). It is ready for training once a massive 4-year data stack (2020-2023) is fully acquired.

*See `objective_1/README.md` and `objective_1/artifacts/ISRO_Hackathon_Objective1_Report.docx` for full details.*

---

## 🔥 Objective 2: HCHO Source Apportionment & Transport Pathways
**Goal:** Trace the origins and transport pathways of Formaldehyde (HCHO) anomalies over the Indo-Gangetic Plain during the severe pollution month of November 2023.

### 🔬 Methodology & Pipeline
We built an automated, end-to-end data processing and clustering pipeline (`run_objective_2.py`) to analyze HCHO spikes and correlate them with regional events:
- **Spatial Clustering (DBSCAN):** Identified dense HCHO hotspots over Punjab, Haryana, and Delhi.
- **Fire Correlation & Lag Analysis:** Correlated HCHO anomalies with MODIS/VIIRS active fire counts, establishing a clear 2-day transport lag between stubble burning events and HCHO spikes in downstream urban centers.
- **Trajectory Mapping:** Integrated wind field overlays and trajectory frequencies to visualize the precise atmospheric pathways carrying secondary organic aerosols.

### 📊 Interactive Dashboard
We developed a backend API (`dashboard_api.py`) to serve these findings dynamically. It provides real-time access to the processed HCHO correlation matrices, temporal lags, and spatial clusters, enabling policymakers to visually explore the cause-and-effect relationship between agricultural fires and urban formaldehyde levels.

*See `objective_2/README.md` for execution instructions and pipeline details.*

---

## 🛠️ Data Sources
- **ESA Sentinel-5P (TROPOMI):** NO₂, SO₂, CO, HCHO, AOD.
- **ECMWF ERA5:** Meteorological variables (Wind, BLH, Temperature, Solar Radiation).
- **ISRO/MOSDAC INSAT-3D:** Complementary Aerosol Optical Depth (AOD).
- **MoEFCC CPCB Network:** Ground-truth validation targets for the 6 pollutants.