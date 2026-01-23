"""
Caravan benchmark constants.

Streamflow prediction using Earth4D positional encoding.
"""

# Dataset configuration
CARAVAN_ZENODO_URL = "https://zenodo.org/records/15530022/files/Caravan-csv.tar.gz"
DEFAULT_CARAVAN_CSV_PATH = "./data/caravan_full.csv"  # Full dataset (12K basins, 135M obs)

# Normalization constants
# Based on log-transform: log(Q + 1) where Q is in mm/day
# Typical streamflow ranges from 0.01 to 100 mm/day
# log(100 + 1) ≈ 4.62
MAX_LOG_STREAMFLOW = 5.0  # Conservative upper bound for log(Q+1)

# Spatial extent (global)
LAT_MIN = -60.0
LAT_MAX = 75.0
LON_MIN = -180.0
LON_MAX = 180.0

# Temporal extent
YEAR_MIN = 1950
YEAR_MAX = 2023

# Basin selection criteria for lightweight benchmark
MIN_RECORD_YEARS = 20  # Minimum years of data
TARGET_N_BASINS = 300  # Target number of basins for quick experiment
MAX_N_BASINS = 500     # Maximum if we need more data

# Train/test temporal split
# Use temporal split: train on earlier years, test on later years
# This tests generalization to future streamflow prediction
TRAIN_END_YEAR = 2015  # Train: 1950-2015
TEST_START_YEAR = 2016  # Test: 2016-2023
