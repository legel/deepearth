"""The variables of the tower dataset: names, units, physical ranges, and where each source keeps the measured value.

Every variable is stored under one name in FLUXNET units (the FP standard, Pastorello et al. 2020). A source column
counts as measured only through the rule given for it here:

- ONEFlux (FLUXNET Shuttle v1.3): `*_F_MDS` with `*_F_MDS_QC == 0`, `NEE_VUT_REF` with `NEE_VUT_REF_QC == 0`, or a
  raw column that ONEFlux never fills (PA, P, WS, WD, USTAR, RH, NETRAD, PPFD_IN, PPFD_OUT, SW_OUT, LW_OUT).
- FLUXNET-CH4 V1.0: the unfilled columns (FCH4, NEE, LE, H, TA, ...). The `_F`, `_F_ANNOPTLM` and `_F_RANDUNC`
  columns are fills and are never read.
- AmeriFlux BASE: the unqualified or positional column (FCH4, FCH4_1_1_1, ...). A column with an `_F` qualifier is a
  PI gap-fill and is never read.
- ICOS ETC L2 Fluxes: the cleaned columns (FCH4, NEE, LE, H). `*_UNCLEANED` columns are never read.

Physical ranges are the AmeriFlux FP-standard variable limits (https://amfcdn.lbl.gov/api/v1/fp_var?limits=True,
retrieved 2026-10-26), widened by a tolerance for radiation sensors whose night offset reads a few W m-2 below zero.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

FLUX, MET, SOIL = "flux", "met", "soil"


@dataclass(frozen=True)
class Var:
    """One dataset variable."""
    name: str
    unit: str
    kind: str
    lo: float
    hi: float
    description: str
    oneflux: Optional[Tuple[str, Optional[str]]] = None
    """(value column, QC column) in the ONEFlux FLUXMET HH file; QC None means a raw column ONEFlux does not fill."""
    tol: float = 0.0
    """Values in [lo - tol, lo) or (hi, hi + tol] are kept: sensor offsets, not physical errors."""
    stuck_steps: Optional[int] = 12
    """A run of identical values this long (in half-hours) or longer is a stuck sensor; None disables the check."""
    stuck_exempt: Tuple[float, ...] = ()
    """Values that may repeat legitimately (zero VPD and 100 % humidity in fog, 0 deg C soil while it freezes or
    thaws)."""
    exempt_atol: float = 1e-9
    """How close to an exempt value a run must be to be kept."""
    dark_repeats: bool = False
    """Night-time repeats are legitimate (a shortwave or PAR sensor reads a constant offset in darkness); a run that
    reaches into daylight is still stuck."""
    spike: bool = False
    """Apply the Papale et al. (2006) double-difference spike test."""
    era5: Optional[str] = None
    """Matching column in the ONEFlux ERA5 HH file."""
    profile: bool = False
    """Several sensors per site (TS_1, TS_2, ...); stored as NAME_1 ... NAME_k."""


VARIABLES: Dict[str, Var] = {v.name: v for v in [
    # Fluxes (targets).
    Var("NEE", "umolCO2 m-2 s-1", FLUX, -100, 100,
        "Net ecosystem exchange of CO2, turbulent flux plus storage, u*-filtered at the ONEFlux VUT reference "
        "threshold; positive is release to the atmosphere", ("NEE_VUT_REF", "NEE_VUT_REF_QC"), spike=True),
    Var("LE", "W m-2", FLUX, -450, 900, "Latent heat turbulent flux; positive is upward",
        ("LE_F_MDS", "LE_F_MDS_QC"), spike=True),
    Var("H", "W m-2", FLUX, -450, 900, "Sensible heat turbulent flux; positive is upward",
        ("H_F_MDS", "H_F_MDS_QC"), spike=True),
    Var("FCH4", "nmolCH4 m-2 s-1", FLUX, -500, 4000,
        "Methane turbulent flux (no storage correction); positive is emission", None, spike=True),
    Var("G", "W m-2", FLUX, -250, 400, "Soil heat flux; positive is into the soil", ("G_F_MDS", "G_F_MDS_QC"),
        stuck_steps=24),
    # Meteorology.
    Var("TA", "deg C", MET, -50, 50, "Air temperature", ("TA_F_MDS", "TA_F_MDS_QC"), era5="TA_ERA"),
    Var("SW_IN", "W m-2", MET, 0, 1300, "Incoming shortwave radiation", ("SW_IN_F_MDS", "SW_IN_F_MDS_QC"),
        tol=10.0, dark_repeats=True, era5="SW_IN_ERA"),
    Var("SW_OUT", "W m-2", MET, 0, 800, "Outgoing shortwave radiation", ("SW_OUT", None), tol=10.0,
        dark_repeats=True),
    Var("LW_IN", "W m-2", MET, 50, 600, "Incoming longwave radiation", ("LW_IN_F_MDS", "LW_IN_F_MDS_QC"),
        era5="LW_IN_ERA"),
    Var("LW_OUT", "W m-2", MET, 100, 750, "Outgoing longwave radiation", ("LW_OUT", None)),
    Var("NETRAD", "W m-2", MET, -200, 1100, "Net radiation", ("NETRAD", None)),
    Var("PPFD_IN", "umolPhoton m-2 s-1", MET, 0, 2400, "Incoming photosynthetic photon flux density",
        ("PPFD_IN", None), tol=20.0, dark_repeats=True),
    Var("PPFD_OUT", "umolPhoton m-2 s-1", MET, 0, 2000, "Outgoing photosynthetic photon flux density",
        ("PPFD_OUT", None), tol=20.0, dark_repeats=True),
    Var("VPD", "hPa", MET, 0, 80, "Vapour pressure deficit", ("VPD_F_MDS", "VPD_F_MDS_QC"), stuck_exempt=(0.0,),
        stuck_steps=24, era5="VPD_ERA"),
    Var("RH", "%", MET, 0, 100, "Relative humidity", ("RH", None), tol=2.0, stuck_exempt=(100.0,), stuck_steps=24),
    Var("PA", "kPa", MET, 60, 105, "Atmospheric pressure", ("PA", None), stuck_steps=48, era5="PA_ERA"),
    Var("P", "mm", MET, 0, 50, "Precipitation in the time step", ("P", None), stuck_steps=None, era5="P_ERA"),
    Var("WS", "m s-1", MET, 0, 40, "Wind speed", ("WS", None), era5="WS_ERA"),
    Var("WD", "decimal degrees", MET, 0, 360, "Wind direction", ("WD", None)),
    Var("USTAR", "m s-1", MET, 0, 8, "Friction velocity", ("USTAR", None)),
    Var("CO2", "umolCO2 mol-1", MET, 150, 1200, "CO2 mole fraction", ("CO2_F_MDS", "CO2_F_MDS_QC")),
    # Soil and water.
    Var("TS", "deg C", SOIL, -40, 65, "Soil temperature, sensors ordered shallow to deep",
        ("TS_F_MDS_{k}", "TS_F_MDS_{k}_QC"), stuck_steps=96, stuck_exempt=(0.0,), exempt_atol=0.5, profile=True),
    Var("SWC", "%", SOIL, 0, 100, "Volumetric soil water content, sensors ordered shallow to deep",
        ("SWC_F_MDS_{k}", "SWC_F_MDS_{k}_QC"), stuck_steps=None, profile=True),
    Var("WTD", "m", SOIL, -10, 10, "Water table depth below the surface (negative is above)", None,
        stuck_steps=None),
]}
"""Stuck-sensor thresholds: 12 half-hours (6 h) for fast signals, 24 (12 h) for VPD, RH and G, 48 (24 h) for
pressure (many sites report it at 0.1 kPa, which can hold for hours), 96 (2 d) for soil temperature. Soil water content and water table change slowly enough that long constant runs are physical, so they
are not tested; precipitation is mostly zero."""

ERA5_COLUMNS = ("TA_ERA", "SW_IN_ERA", "LW_IN_ERA", "VPD_ERA", "PA_ERA", "P_ERA", "WS_ERA")
"""The ONEFlux ERA5 HH file: ERA5 hourly, downscaled to the site by linear regression on the site's measured
half-hours (Vuichard and Papale 2015). A reanalysis modality, never a measurement."""

UNCERTAINTY = {"NEE": "NEE_VUT_REF_RANDUNC", "LE": "LE_RANDUNC", "H": "H_RANDUNC"}
"""ONEFlux random-uncertainty columns, kept where the flux itself is measured (Pastorello et al. 2020)."""

MAX_PROFILE = 5
"""At most this many TS or SWC sensors per site are kept (the ONEFlux FLUXMET file carries up to about 10)."""


def columns(name: str, k: int) -> Tuple[str, Optional[str]]:
    """The ONEFlux (value, QC) columns of profile sensor k (1-based) of `name`."""
    val, qc = VARIABLES[name].oneflux
    return val.format(k=k), (qc.format(k=k) if qc else None)


def base(name: str) -> str:
    """The registry name of a stored column: TS_2 -> TS, NEE -> NEE."""
    head, _, tail = name.rpartition("_")
    return head if tail.isdigit() and head in VARIABLES else name
