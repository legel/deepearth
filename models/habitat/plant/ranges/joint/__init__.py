"""The joint range model: shared representations of a location (environment, landscape field, place), a
Brownian-motion phylogenetic prior on every species vector, MaxEnt's point-process objective, a learned calibration
penalty, and a transform-coded map store.

Modules: ``prepare`` (training points, plot sets and target-group background from the per-species products),
``scope`` (the mapped region), ``tree`` (Newick, path matrix), ``model`` (JointRangeModel), ``field`` (Entropy3D ring
harmonics of the raw 240 m field, with its CUDA kernel in ``kernels/``), ``place`` (SINR's coordinate network),
``geodesy`` (WGS 84 to the CONUS Albers grid exactly as PROJ), ``climate_fill`` and ``shoreline`` (locations without
climate take the nearest climate), ``data`` (training points, plots, standardization), ``train`` (every training stage
and plot evaluation), ``cache`` (the shared features of a fixed representation), ``zero_shot`` (species without
records), ``store`` (map store build), ``reader`` (map store reader: needs only numpy and zstandard).
Method: docs/joint_model.md.
"""
from .data import JointData, Standardizer
from .model import JointRangeModel
from .reader import Store
from .train import TrainConfig, train
from .tree import Tree, parse_newick, path_matrix
from .zero_shot import ZeroShot, infer_species, l3_ecoregions

__all__ = ["JointData", "JointRangeModel", "Standardizer", "Store", "TrainConfig", "Tree", "ZeroShot",
           "infer_species", "l3_ecoregions", "parse_newick", "path_matrix", "train"]
