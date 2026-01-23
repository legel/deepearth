#!/usr/bin/env python3
"""
Prepare Alzhanov dataset with meteorological INPUT features.

Adds:
- precipitation (total_precipitation_sum)
- temperature (temperature_2m_mean)
- snow (snow_depth_water_equivalent_mean)

These will be used as INPUT features (along with x,y,z,t) to predict streamflow.
Based on Lance's feedback: use P, T, Snow as inputs, not as multi-task outputs.
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Load existing dataset
print("Loading existing dataset...")
df_base = pd.read_csv('data/caravan_alzhanov_147basins_with_uba.csv')
print(f"Loaded {len(df_base):,} rows, {df_base['gauge_id'].nunique()} basins")

# Add meteorological input columns
df_base['precipitation_mm_per_day'] = float('nan')
df_base['temperature_2m_mean'] = float('nan')
df_base['snow_depth_water_equivalent_mean'] = float('nan')

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
print(f"\nLoading P, T, Snow for {len(basins)} basins...")

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
        uba_snow = ds.variables['snow_depth_water_equivalent_mean'][:]
        ds.close()

        # Create lookup dataframe
        df_uba_meteo = pd.DataFrame({
            'date': uba_dates,
            'precipitation_mm_per_day': uba_precip,
            'temperature_2m_mean': uba_temp,
            'snow_depth_water_equivalent_mean': uba_snow,
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
        df_base.loc[df_base['gauge_id'] == basin, 'snow_depth_water_equivalent_mean'] = \
            df_uba['snow_depth_water_equivalent_mean_new'].values

        success_count += 1
        continue

    # Determine which directory
    prefix = basin.split('_')[0]
    if prefix not in csv_dirs:
        print(f"Warning: Unknown prefix {prefix} for basin {basin}")
        missing_count += 1
        continue

    # Load basin CSV
    csv_path = Path(csv_dirs[prefix]) / f"{basin}.csv"
    if not csv_path.exists():
        print(f"Warning: CSV not found for {basin} at {csv_path}")
        missing_count += 1
        continue

    try:
        df_basin = pd.read_csv(csv_path)

        # Select only needed columns
        df_basin = df_basin[['date', 'total_precipitation_sum', 'temperature_2m_mean',
                              'snow_depth_water_equivalent_mean']]
        df_basin.rename(columns={
            'total_precipitation_sum': 'precipitation_mm_per_day',
        }, inplace=True)

        # Merge with main dataframe
        df_subset = df_base[df_base['gauge_id'] == basin].copy()
        df_subset = df_subset.merge(df_basin, on='date', how='left', suffixes=('', '_new'))

        # Update main dataframe
        df_base.loc[df_base['gauge_id'] == basin, 'precipitation_mm_per_day'] = \
            df_subset['precipitation_mm_per_day_new'].values
        df_base.loc[df_base['gauge_id'] == basin, 'temperature_2m_mean'] = \
            df_subset['temperature_2m_mean_new'].values
        df_base.loc[df_base['gauge_id'] == basin, 'snow_depth_water_equivalent_mean'] = \
            df_subset['snow_depth_water_equivalent_mean_new'].values

        success_count += 1

    except Exception as e:
        print(f"Error loading {basin}: {e}")
        missing_count += 1
        continue

print(f"\nSuccessfully loaded: {success_count} basins")
print(f"Missing/errors: {missing_count} basins")

# Check for missing values
print("\nMissing value statistics:")
print(f"  Precipitation: {df_base['precipitation_mm_per_day'].isna().sum():,} / {len(df_base):,} ({100*df_base['precipitation_mm_per_day'].isna().sum()/len(df_base):.1f}%)")
print(f"  Temperature: {df_base['temperature_2m_mean'].isna().sum():,} / {len(df_base):,} ({100*df_base['temperature_2m_mean'].isna().sum()/len(df_base):.1f}%)")
print(f"  Snow: {df_base['snow_depth_water_equivalent_mean'].isna().sum():,} / {len(df_base):,} ({100*df_base['snow_depth_water_equivalent_mean'].isna().sum()/len(df_base):.1f}%)")
print(f"  Streamflow: {df_base['streamflow_mm_per_day'].isna().sum():,} / {len(df_base):,} ({100*df_base['streamflow_mm_per_day'].isna().sum()/len(df_base):.1f}%)")

# Remove rows with missing inputs or outputs
print("\nRemoving rows with missing values...")
df_clean = df_base.dropna(subset=['precipitation_mm_per_day', 'temperature_2m_mean',
                                    'snow_depth_water_equivalent_mean', 'streamflow_mm_per_day'])
print(f"Kept {len(df_clean):,} / {len(df_base):,} rows ({100*len(df_clean)/len(df_base):.1f}%)")

# Print statistics
print("\nInput feature statistics:")
print(f"  Precipitation: mean={df_clean['precipitation_mm_per_day'].mean():.2f} mm/day, "
      f"std={df_clean['precipitation_mm_per_day'].std():.2f}, "
      f"min={df_clean['precipitation_mm_per_day'].min():.2f}, "
      f"max={df_clean['precipitation_mm_per_day'].max():.2f}")
print(f"  Temperature: mean={df_clean['temperature_2m_mean'].mean():.2f} °C, "
      f"std={df_clean['temperature_2m_mean'].std():.2f}, "
      f"min={df_clean['temperature_2m_mean'].min():.2f}, "
      f"max={df_clean['temperature_2m_mean'].max():.2f}")
print(f"  Snow: mean={df_clean['snow_depth_water_equivalent_mean'].mean():.2f} mm, "
      f"std={df_clean['snow_depth_water_equivalent_mean'].std():.2f}, "
      f"min={df_clean['snow_depth_water_equivalent_mean'].min():.2f}, "
      f"max={df_clean['snow_depth_water_equivalent_mean'].max():.2f}")
print(f"  Streamflow: mean={df_clean['streamflow_mm_per_day'].mean():.2f} mm/day, "
      f"std={df_clean['streamflow_mm_per_day'].std():.2f}, "
      f"min={df_clean['streamflow_mm_per_day'].min():.2f}, "
      f"max={df_clean['streamflow_mm_per_day'].max():.2f}")

# Save
output_path = 'data/caravan_alzhanov_147basins_inputs.csv'
df_clean.to_csv(output_path, index=False)
print(f"\nSaved to {output_path}")
print(f"  Basins: {df_clean['gauge_id'].nunique()}")
print(f"  Observations: {len(df_clean):,}")
