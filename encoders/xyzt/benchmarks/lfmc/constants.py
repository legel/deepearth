"""
LFMC-specific constants following allenai/lfmc approach.
"""

import random

# LFMC value normalization (99.9th percentile)
MAX_LFMC_VALUE = 302

# AI2 official data URL
AI2_LFMC_CSV_URL = "https://raw.githubusercontent.com/allenai/lfmc/refs/heads/main/data/labels/lfmc_data_conus.csv"
DEFAULT_AI2_CSV_PATH = "data/labels/lfmc_data_conus.csv"

# AI2 default fold configuration (70-15-15 split)
# Uses seed 42 for reproducibility, matching allenai/lfmc
DEFAULT_NUM_FOLDS = 100
_rng = random.Random(42)
_folds = _rng.sample(range(DEFAULT_NUM_FOLDS), 30)
DEFAULT_VALIDATION_FOLDS = frozenset(_folds[:15])
DEFAULT_TEST_FOLDS = frozenset(_folds[15:])
