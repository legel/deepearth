"""Surface-soil properties (pH, clay, sand, organic carbon, USDA texture class) for any site, with the benchmark that
chose the sources. See README.md."""
from .site_layers import layers, fit
from .texture import texture_class, CLASSES

__all__ = ["layers", "fit", "texture_class", "CLASSES"]
