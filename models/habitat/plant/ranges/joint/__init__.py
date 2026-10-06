"""The joint range model: one environment network shared by all species, a Brownian-motion phylogenetic prior on the
species' niche vectors, MaxEnt's point-process objective, and a transform-coded map store.

Modules: ``prepare`` (training points and plot sets from the per-species products), ``scope`` (the mapped region),
``tree`` (Newick, path matrix), ``model`` (JointRangeModel), ``data`` (training points, plots, standardization),
``train`` (training and plot evaluation), ``zero_shot`` (species without records), ``store`` (map store build),
``reader`` (map store reader: needs only numpy and zstandard). Method: docs/joint_model.md.
"""
from .data import JointData, Standardizer
from .model import JointRangeModel
from .reader import Store
from .train import TrainConfig, train
from .tree import Tree, parse_newick, path_matrix
from .zero_shot import ZeroShot, infer_species, l3_ecoregions

__all__ = ["JointData", "JointRangeModel", "Standardizer", "Store", "TrainConfig", "Tree", "ZeroShot",
           "infer_species", "l3_ecoregions", "parse_newick", "path_matrix", "train"]
