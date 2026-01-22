#!/usr/bin/env python3
"""
Prepare Alzhanov dataset with multiple meteorological variables for multi-task learning.

Loads the basin list and adds:
- precipitation (total_precipitation_sum)
- temperature (temperature_2m_mean)

These will be used as auxiliary prediction targets for multi-task learning.
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Load existing dataset
print("Loading existing dataset...")
df_base = pd.read_csv('data/caravan_alzhanov_147basins_with_uba.csv')
print(f"Loaded {len(df_base):,} rows, {df_base['gauge_id'].nunique()} basins")

# Add meteorological columns
df_base['precipitation_mm_per_day'] = float('nan')
df_base['temperature_2m_mean'] = float('nan')

# Paths to timeseries data
csv_dirs = {
    'camels': 'data/Caravan-csv/timeseries/csv/camels',
    'camelsaus': 'data/Caravan-csv/timeseries/csv/camelsaus',
    'camelsbr': 'data/Caravan-csv/timeseries/csv/camelsbr',
    'camelscl': 'data/Caravan-csv/timeseries/csv/camelscl',
    'camelsgb': 'data/Caravan-csv/timeseries/csv/camelsgb',
}

# Load meteorology for each basin
basins = df_base['gauge_id'].unique()
print(f"\nLoading meteorology for {len(basins)} basins...")

success_count = 0
missing_count = 0

for basin in tqdm(basins):
    # Handle Uba River separately (from NetCDF)
    if basin == 'ubakz_99999999':
        import netCDF4 as nc
        ds = nc.Dataset('Uba_local_data/ubakz_99999999.nc')

        # NetCDF has integer dates (0, 1, 2...), need to create date range
        # From CSV, we know Uba data starts at 1995-01-01 and has 9465 days
        n_days = len(ds.variables['date'][:])
        uba_dates = pd.date_range(start='1995-01-01', periods=n_days, freq='D')
        uba_precip = ds.variables['total_precipitation_sum'][:]
        uba_temp = ds.variables['temperature_2m_mean'][:]
        ds.close()

        # Create lookup dataframe
        df_uba_meteo = pd.DataFrame({
            'date': uba_dates,
            'precipitation_mm_per_day': uba_precip,
            'temperature_2m_mean': uba_temp,
        })

        # Match with dataframe
        df_uba = df_base[df_base['gauge_id'] == basin].copy()
        df_uba['date'] = pd.to_datetime(df_uba['date'])
        df_uba = df_uba.merge(df_uba_meteo, on='date', how='left', suffixes=('', '_new'))

        # Update main dataframe
        df_base.loc[df_base['gauge_id'] == basin, 'precipitation_mm_per_day'] = \
            df_uba['precipitation_mm_per_day_new'].values
        df_base.loc[df_base['gauge_id'] == basin, 'temperature_2m_mean'] = \
            df_uba['temperature_2m_mean_new'].values

        success_count += 1
        continue

    # Determine which directory
    prefix = basin.split('_')[0]
    if prefix not in csv_dirs:
        print(f"Warning: Unknown prefix {prefix} for basin {basin}")
        missing_count += 1
        continue

    csv_path = Path(csv_dirs[prefix]) / f"{basin}.csv"

    if not csv_path.exists():
        print(f"Warning: Missing CSV for basin {basin}")
        missing_count += 1
        continue

    # Load basin CSV
    try:
        df_basin = pd.read_csv(csv_path)

        # Check if temperature column exists
        if 'temperature_2m_mean' not in df_basin.columns:
            print(f"Warning: {basin} missing temperature_2m_mean column")
            missing_count += 1
            continue

        df_basin['date'] = pd.to_datetime(df_basin['date'])

        # Merge meteorological variables
        df_merge = df_base[df_base['gauge_id'] == basin].copy()
        df_merge['date'] = pd.to_datetime(df_merge['date'])
        df_merge = df_merge.merge(
            df_basin[['date', 'total_precipitation_sum', 'temperature_2m_mean']],
            on='date',
            how='left'
        )

        # Update main dataframe
        # After merge, check which columns exist
        precip_col = 'total_precipitation_sum'
        temp_col = 'temperature_2m_mean' if 'temperature_2m_mean' in df_merge.columns else 'temperature_2m_mean_y'

        df_base.loc[df_base['gauge_id'] == basin, 'precipitation_mm_per_day'] = \
            df_merge[precip_col].values
        df_base.loc[df_base['gauge_id'] == basin, 'temperature_2m_mean'] = \
            df_merge[temp_col].values

        success_count += 1

    except Exception as e:
        print(f"Error loading {basin}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        missing_count += 1

print(f"\nLoading complete:")
print(f"  Success: {success_count} basins")
print(f"  Missing: {missing_count} basins")

# Check meteorological statistics
print(f"\nPrecipitation statistics:")
print(f"  Min: {df_base['precipitation_mm_per_day'].min():.3f} mm/day")
print(f"  Median: {df_base['precipitation_mm_per_day'].median():.3f} mm/day")
print(f"  Max: {df_base['precipitation_mm_per_day'].max():.3f} mm/day")
print(f"  NaN count: {df_base['precipitation_mm_per_day'].isna().sum():,} ({100*df_base['precipitation_mm_per_day'].isna().sum()/len(df_base):.1f}%)")

print(f"\nTemperature statistics:")
print(f"  Min: {df_base['temperature_2m_mean'].min():.3f} °C")
print(f"  Median: {df_base['temperature_2m_mean'].median():.3f} °C")
print(f"  Max: {df_base['temperature_2m_mean'].max():.3f} °C")
print(f"  NaN count: {df_base['temperature_2m_mean'].isna().sum():,} ({100*df_base['temperature_2m_mean'].isna().sum()/len(df_base):.1f}%)")

# Save enhanced dataset
output_path = 'data/caravan_alzhanov_147basins_multitask.csv'
df_base.to_csv(output_path, index=False)
print(f"\nSaved multi-task dataset: {output_path}")
print(f"  Total rows: {len(df_base):,}")
print(f"  Columns: {list(df_base.columns)}")
