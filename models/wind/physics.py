"""Physical constants and the aerodynamic parameters of the 46 surface classes.

Import-time side-effect free. The class table is transcribed from the ecological ontology
(`ontology_web.json`: `z0_m`, `cd`, `poro`, `LAI`, `closed`). A class is solid when the
ontology closes it or its porosity is 0.1 or less; every other class is a porous drag volume.
"""

from dataclasses import dataclass
from typing import Dict, Final

KAPPA: Final = 0.4
"""von Karman constant."""

M_S_PER_KNOT: Final = 0.514444
"""Metres per second per knot; ASOS reports speed in knots."""

ASOS_ANEMOMETER_M: Final = 10.0
"""Standard ASOS anemometer height [m]."""

NU_AIR: Final = 1.5e-5
"""Kinematic viscosity of air [m^2/s], the floor of the eddy viscosity."""

GROUND_CLASS: Final = "turf_grass"
"""Class assumed for ground that no raster describes: under canopy, outside the parcel."""


@dataclass(frozen=True)
class AeroClass:
    """Aerodynamic parameters of one surface class.

    Attributes:
        z0_m: Roughness length [m].
        cd: Drag coefficient.
        poro: Porosity, 0 closed to 1 open.
        lai: Leaf area index [m^2/m^2].
        solid: True where the class blocks flow.
    """

    z0_m: float
    cd: float
    poro: float
    lai: float
    solid: bool


CLASSES: Dict[str, AeroClass] = {
    "antenna_satellite": AeroClass(0.055, 0.125, 0.35, 0.0, False),
    "asphalt_pavement": AeroClass(0.003, 0.01, 0.01, 0.0, True),
    "balcony": AeroClass(0.03, 0.2, 0.8, 0.0, False),
    "bell_tower": AeroClass(0.3, 0.5, 0.05, 0.0, True),
    "bench": AeroClass(0.03, 0.75, 0.7, 0.0, False),
    "bicycle": AeroClass(0.003, 0.2, 0.85, 0.0, False),
    "bicycle_rack": AeroClass(0.0125, 0.3, 0.85, 0.0, False),
    "brick_paving": AeroClass(0.0065, 0.015, 0.1, 0.0, True),
    "column_archway": AeroClass(0.06, 0.35, 0.06, 0.0, True),
    "concrete_pavement": AeroClass(0.00175, 0.0085, 0.01, 0.0, True),
    "construction_temporary": AeroClass(0.11, 1.0, 0.15, 0.0, False),
    "dirt_soil": AeroClass(0.0125, 0.0225, 0.01, 0.0, True),
    "door_entrance": AeroClass(0.0125, 0.2, 0.01, 0.0, True),
    "eucalyptus_tree": AeroClass(1.0, 0.3, 0.875, 4.0, False),
    "facade_glass": AeroClass(0.003, 0.1, 0.0, 0.0, True),
    "facade_masonry": AeroClass(0.03, 0.2, 0.03, 0.0, True),
    "fence_gate": AeroClass(0.03, 0.85, 0.15, 0.0, False),
    "gravel_surface": AeroClass(0.03, 0.03, 0.01, 0.0, True),
    "hedge_shrub": AeroClass(0.65, 0.2, 0.4, 6.0, False),
    "overhead_power_line": AeroClass(0.003, 0.75, 0.95, 0.0, False),
    "palm_tree": AeroClass(1.25, 0.175, 0.5, 3.0, False),
    "pavement_generic": AeroClass(0.00175, 0.0075, 0.025, 0.0, True),
    "pavement_marking": AeroClass(0.00125, 0.0075, 0.01, 0.0, True),
    "pedestrian": AeroClass(0.2, 1.0, 0.0, 0.0, True),
    "pine_tree": AeroClass(1.0, 0.3, 0.875, 5.0, False),
    "planting_bed": AeroClass(0.175, 0.0125, 0.01, 2.0, True),
    "pole": AeroClass(0.055, 0.75, 0.0, 0.0, True),
    "railing": AeroClass(0.0125, 0.6, 0.75, 0.0, False),
    "roof_appurtenance": AeroClass(0.0055, 0.125, 0.025, 0.0, True),
    "roof_glazing": AeroClass(0.00055, 0.03, 0.0, 0.0, True),
    "roof_sealed": AeroClass(0.0011, 0.0075, 0.005, 0.0, True),
    "roof_tile": AeroClass(0.003, 0.015, 0.025, 0.0, True),
    "roof_vegetated": AeroClass(0.1, 0.2, 0.01, 3.5, True),
    "rooftop_mechanical": AeroClass(0.055, 0.2, 0.025, 0.0, True),
    "sign_panel": AeroClass(0.0055, 0.75, 0.0, 0.0, True),
    "solar_panel": AeroClass(0.00275, 0.03, 0.0, 0.0, True),
    "statue_fountain": AeroClass(0.055, 1.05, 0.025, 0.0, True),
    "stone_paving": AeroClass(0.006, 0.0175, 0.03, 0.0, True),
    "synthetic_sports_surface": AeroClass(0.003, 0.015, 0.01, 0.0, True),
    "trash_receptacle": AeroClass(0.03, 0.75, 0.0, 0.0, True),
    "tree_canopy": AeroClass(2.0, 0.2, 0.4, 5.5, False),
    "tree_trunk": AeroClass(0.03, 0.6, 0.0, 0.0, True),
    "turf_grass": AeroClass(0.06, 0.0065, 0.01, 3.5, True),
    "vehicle": AeroClass(0.003, 0.4, 0.0, 0.0, True),
    "window": AeroClass(0.003, 0.1, 0.0, 0.0, True),
    "wooden_deck": AeroClass(0.006, 0.0225, 0.2, 0.0, False),
}
"""Per-class parameters, keyed by ontology `class_id`."""


def drag_density(cls: AeroClass, height_m: float, dx: float) -> float:
    """Volumetric drag coefficient cd * a [1/m] for a porous column.

    Args:
        cls: The column's class.
        height_m: Column height above its ground [m].
        dx: Cell size [m], the element scale of foliage-free open structures.

    Returns:
        cd * a, with a = LAI / height for foliage and (1 - poro) / dx otherwise; 0 when solid.
    """
    if cls.solid or height_m <= 0.0:
        return 0.0
    area_density = cls.lai / height_m if cls.lai > 0.0 else (1.0 - cls.poro) / dx
    return cls.cd * area_density
