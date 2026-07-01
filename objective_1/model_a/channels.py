# channels.py
# ===========
# Single source of truth for all channel indices, units, and pollutant targets.
# Import this in every script — change here, it updates everywhere.
# Used by: 04_extract_patches.py, 05_build_tabular.py, run_model.py

CHANNEL_ORDER = {
    'AOD'  : 0,   # INSAT-3D -- Aerosol Optical Depth
    'NO2'  : 1,   # Sentinel-5P TROPOMI -- NO2 tropospheric column
    'SO2'  : 2,   # Sentinel-5P TROPOMI -- SO2 column
    'CO'   : 3,   # Sentinel-5P TROPOMI -- CO column
    'U10'  : 4,   # ERA5 -- U-component wind at 10m
    'V10'  : 5,   # ERA5 -- V-component wind at 10m
    'T2M'  : 6,   # ERA5 -- Temperature at 2m
    'BLH'  : 7,   # ERA5 -- Boundary Layer Height
    'RH'   : 8,   # ERA5 -- Relative Humidity (derived from d2m, t2m via Magnus formula)
    'MASK' : 9,   # Derived -- cloud/missingness binary mask (added in Phase 2 / 04_extract_patches.py)
}

N_CHANNELS  = len(CHANNEL_ORDER)   # = 10
N_SAT_MET   = N_CHANNELS - 1       # = 9  (excludes mask channel)

# Native units (for sanity checks during extraction)
CHANNEL_UNITS = {
    'AOD'  : 'dimensionless (0-5)',
    'NO2'  : 'mol/m2  (~1e-5 range)',
    'SO2'  : 'mol/m2  (~1e-3 range)',
    'CO'   : 'mol/m2  (~0.01-0.1 range)',
    'U10'  : 'm/s',
    'V10'  : 'm/s',
    'T2M'  : 'K  (273-320 range)',
    'BLH'  : 'm  (100-3000 range)',
    'RH'   : '%  (0-100)',
    'MASK' : 'binary (0 or 1)',
}

# Channels sourced from Sentinel-5P (will have cloud/orbit gaps)
SENTINEL_CHANNELS = ['NO2', 'SO2', 'CO']

# Channels that receive 2D median filter (INSAT speckle noise)
MEDIAN_FILTER_CHANNELS = ['AOD']

# Pollutant prediction targets (6 independent models)
POLLUTANTS = ['PM2.5', 'PM10', 'NO2', 'SO2', 'O3', 'CO']

# Confirmed CPCB daily file column names
CPCB_DAILY_COLS = {
    'PM2.5' : 'PM2.5',
    'PM10'  : 'PM10',
    'NO2'   : 'NO2',
    'SO2'   : 'SO2',
    'O3'    : 'O3_8hr_max',
    'CO'    : 'CO_24hr_mean',
}

# Column names in cpcb_xgboost_features.csv that map to y targets
POLLUTANT_CSV_COLS = {
    'PM2.5' : 'pm25',
    'PM10'  : 'pm10',
    'NO2'   : 'no2',
    'SO2'   : 'so2',
    'O3'    : 'o3',
    'CO'    : 'co',
}

# The 12 tabular features fed into XGBoost (in exact order)
TABULAR_FEATURES = [
    'lat',
    'lon',
    'elevation',
    'doy',
    'lag_PM25_D1',
    'lag_PM25_D2',
    'lag_AOD_D1',
    'lag_NO2_D1',
    'lag_CO_D1',
    'ventilation_index',
    'wind_direction',   # arctan2(v10, u10) -- helps NO2 directional dispersion
    't2m_squared',      # T2M^2 -- captures nonlinear O3-temperature relationship
]
N_TABULAR = len(TABULAR_FEATURES)   # = 12

# XGBoost feature matrix width: GAP(64) + CNN_pred(1) + X_tabular(12) = 77
GAP_DIM     = 64
N_XGB_FEATS = GAP_DIM + 1 + N_TABULAR   # = 77

if __name__ == '__main__':
    print('Channel registry loaded successfully.')
    print(f'  N_CHANNELS  : {N_CHANNELS}')
    print(f'  N_SAT_MET   : {N_SAT_MET}')
    print(f'  N_TABULAR   : {N_TABULAR}')
    print(f'  N_XGB_FEATS : {N_XGB_FEATS}')
    print(f'  POLLUTANTS  : {POLLUTANTS}')
    print(f'  CHANNEL_ORDER:')
    for name, idx in CHANNEL_ORDER.items():
        print(f'    [{idx}] {name:<5}  {CHANNEL_UNITS[name]}')
