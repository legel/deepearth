"""
Caravan benchmark for Earth4D streamflow prediction.

A global hydrology benchmark testing whether Earth4D can predict streamflow
from (x,y,z,t) coordinates alone, analogous to the LFMC benchmark.
"""

from .constants import *
from .data import *
from .model import *

__all__ = [
    'CaravanDataset',
    'StreamflowModel',
    'compute_streamflow_metrics',
]
