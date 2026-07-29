"""Leakage-resistant, CPU-first science gate for Earth4D on Globe-LFMC 2.0.

This module deliberately starts with data and baselines, not a neural model.  It
defines which LFMC claims are identifiable under four evaluation designs and
provides train-only reference models that an Earth4D experiment must beat.

The official random split is retained as an interpolation diagnostic. A future
claim requires at least the temporal split; a future-at-new-sites claim requires
at least the spatiotemporal split. Neither split alone proves autoregression.

Example:

    python3 -m autoresearch.programs.spacetime.science_gate \
        --download --json-out data/lfmc/earth4d_science_gate_dev.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


OFFICIAL_LFMC_COMMIT = "65e84f4f697645b023bcdee6576fb3b114d187df"
OFFICIAL_LFMC_URL = (
    "https://raw.githubusercontent.com/allenai/lfmc/"
    + OFFICIAL_LFMC_COMMIT
    + "/data/labels/lfmc_data_conus.csv"
)
OFFICIAL_LFMC_SHA256 = (
    "b44bf99dab6a136fd6e66df3ba8f91f7614f2cc436bdef14a924b640a9bce3a6"
)
DEFAULT_DATA_PATH = Path("data/lfmc/lfmc_data_conus.csv")
MAX_LFMC_VALUE = 302.0
DEFAULT_TRAIN_END = pd.Timestamp("2020-12-31")
DEFAULT_VALIDATION_END = pd.Timestamp("2021-12-31")
PARTITIONS = ("train", "validation", "test")
DESIGNS = ("random", "spatial", "temporal", "spatiotemporal")
ROLLING_DESIGNS = ("temporal", "spatiotemporal")
ROLLING_HORIZONS_DAYS = (30, 90, 180)
ROLLING_TOLERANCE_DAYS = 7
VISIT_KEYS = ("site_name", "species_collected", "sampling_date")

REQUIRED_COLUMNS = {
    "site_name",
    "latitude",
    "longitude",
    "sampling_date",
    "lfmc_value",
    "species_collected",
    "elevation",
    "random_split",
    "spatial_split",
}

CLAIM_SCOPE = {
    "random": "interpolation diagnostic only (not a future forecast)",
    "spatial": "transfer to held-out sites, without a future-time claim",
    "temporal": "future-time prediction at sites that may have appeared in training",
    "spatiotemporal": "future-time prediction at held-out sites",
}


class ScienceGateError(ValueError):
    """Raised when a split cannot support its stated scientific claim."""


def download_official_lfmc(
    destination: Path = DEFAULT_DATA_PATH,
    url: str = OFFICIAL_LFMC_URL,
) -> Path:
    """Download the official AllenAI LFMC label CSV atomically."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination

    handle, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    os.close(handle)
    temporary_path = Path(temporary_name)
    try:
        urllib.request.urlretrieve(url, str(temporary_path))
        temporary_path.replace(destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return destination


def _require_columns(frame: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ScienceGateError(
            "LFMC CSV is missing required official columns: " + ", ".join(missing)
        )


def prepare_lfmc_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the published benchmark filter and validate gate-critical fields.

    The benchmark filter is exactly LFMC in ``[0, 302]`` plus non-null latitude,
    longitude, and elevation.  Date, site, species, and split labels are then
    validated rather than silently discarded because absent values would make
    the corresponding scientific split undefined.
    """

    _require_columns(frame)
    work = frame.copy()
    raw_rows = len(work)

    for column in ("lfmc_value", "latitude", "longitude", "elevation"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work["sampling_date"] = pd.to_datetime(work["sampling_date"], errors="coerce")

    valid = (
        work["lfmc_value"].between(0.0, MAX_LFMC_VALUE, inclusive="both")
        & work["latitude"].notna()
        & work["longitude"].notna()
        & work["elevation"].notna()
    )
    work = work.loc[valid].copy()
    work["_source_row"] = work.index.astype(np.int64)
    work.reset_index(drop=True, inplace=True)

    critical = (
        "sampling_date",
        "site_name",
        "species_collected",
        "random_split",
        "spatial_split",
    )
    missing_counts = {
        column: int(work[column].isna().sum())
        for column in critical
        if work[column].isna().any()
    }
    if missing_counts:
        detail = ", ".join(
            "{}={}".format(column, count)
            for column, count in sorted(missing_counts.items())
        )
        raise ScienceGateError(
            "Rows surviving the LFMC benchmark filter have missing "
            "gate-critical values: " + detail
        )

    for column in ("random_split", "spatial_split"):
        work[column] = work[column].astype(str).str.strip().str.lower()
        unexpected = sorted(set(work[column]).difference(PARTITIONS))
        if unexpected:
            raise ScienceGateError(
                "{} contains unexpected labels: {}".format(column, unexpected)
            )

    work["site_name"] = work["site_name"].astype(str)
    work["species_collected"] = work["species_collected"].astype(str)
    work["_month"] = work["sampling_date"].dt.month.astype(np.int8)
    work.attrs["raw_rows"] = raw_rows
    work.attrs["filtered_rows"] = len(work)
    work.attrs["removed_rows"] = raw_rows - len(work)
    return work


def load_lfmc_csv(path: Path) -> pd.DataFrame:
    """Load and validate an official-format LFMC CSV."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    return prepare_lfmc_frame(pd.read_csv(path))


def verify_official_hash(path: Path) -> Tuple[str, bool]:
    """Reject silent source drift for the default pinned LFMC artifact."""

    digest = _sha256(path)
    official_verified = digest == OFFICIAL_LFMC_SHA256
    if Path(path).resolve() == DEFAULT_DATA_PATH.resolve() and not official_verified:
        raise ScienceGateError(
            "Default LFMC CSV sha256 {} does not match pinned source {}".format(
                digest, OFFICIAL_LFMC_SHA256
            )
        )
    return digest, official_verified


def make_split_masks(
    frame: pd.DataFrame,
    design: str,
    train_end: pd.Timestamp = DEFAULT_TRAIN_END,
    validation_end: pd.Timestamp = DEFAULT_VALIDATION_END,
) -> Dict[str, pd.Series]:
    """Return disjoint train/validation/test masks for one claim design."""

    if design not in DESIGNS:
        raise ScienceGateError(
            "Unknown design {!r}; choose one of {}".format(design, DESIGNS)
        )
    train_end = pd.Timestamp(train_end)
    validation_end = pd.Timestamp(validation_end)
    if train_end >= validation_end:
        raise ScienceGateError("train_end must be earlier than validation_end")

    if design in ("random", "spatial"):
        column = design + "_split"
        masks = {partition: frame[column].eq(partition) for partition in PARTITIONS}
    else:
        dates = frame["sampling_date"]
        time_masks = {
            "train": dates.le(train_end),
            "validation": dates.gt(train_end) & dates.le(validation_end),
            "test": dates.gt(validation_end),
        }
        if design == "temporal":
            masks = time_masks
        else:
            masks = {
                partition: time_masks[partition]
                & frame["spatial_split"].eq(partition)
                for partition in PARTITIONS
            }
    return {name: mask.astype(bool) for name, mask in masks.items()}


def collapse_lfmc_visits(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated site/species/date measurements to one visit.

    LFMC is averaged within a visit so repeated measurements cannot inflate a
    longitudinal evaluation. Replicate count, sample standard deviation, and
    range remain attached for measurement-noise auditing. Pair construction
    never reads these LFMC-derived columns.
    """

    required = set(VISIT_KEYS).union(
        {
            "lfmc_value",
            "latitude",
            "longitude",
            "elevation",
            "random_split",
            "spatial_split",
        }
    )
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ScienceGateError(
            "Cannot construct LFMC visits; missing columns: " + ", ".join(missing)
        )

    spatial_per_site = frame.groupby(
        "site_name", observed=True
    )["spatial_split"].nunique()
    if bool(spatial_per_site.gt(1).any()):
        raise ScienceGateError(
            "A site has multiple spatial_split labels; rolling site identity "
            "would be ambiguous"
        )
    coordinate_per_site = (
        frame.loc[:, ["site_name", "latitude", "longitude"]]
        .drop_duplicates()
        .groupby("site_name", observed=True)
        .size()
    )
    if bool(coordinate_per_site.gt(1).any()):
        raise ScienceGateError(
            "A site maps to multiple coordinates; rolling site identity would "
            "be ambiguous"
        )

    grouped = frame.groupby(list(VISIT_KEYS), observed=True, sort=True)
    visits = grouped.agg(
        lfmc_value=("lfmc_value", "mean"),
        replicate_count=("lfmc_value", "size"),
        lfmc_replicate_std=("lfmc_value", "std"),
        _lfmc_min=("lfmc_value", "min"),
        _lfmc_max=("lfmc_value", "max"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        elevation=("elevation", "mean"),
        random_split=("random_split", "first"),
        spatial_split=("spatial_split", "first"),
    ).reset_index()
    visits["lfmc_replicate_std"] = (
        visits["lfmc_replicate_std"].fillna(0.0).astype(float)
    )
    visits["lfmc_replicate_range"] = (
        visits.pop("_lfmc_max") - visits.pop("_lfmc_min")
    ).astype(float)
    visits["replicate_count"] = visits["replicate_count"].astype(np.int64)
    visits["_month"] = visits["sampling_date"].dt.month.astype(np.int8)
    visits["_visit_id"] = np.arange(len(visits), dtype=np.int64)

    visit_random_labels = grouped["random_split"].nunique()
    visits.attrs["source_rows"] = int(len(frame))
    visits.attrs["unique_visits"] = int(len(visits))
    visits.attrs["duplicate_visits"] = int(visits["replicate_count"].gt(1).sum())
    visits.attrs["rows_in_duplicate_visits"] = int(
        visits.loc[visits["replicate_count"].gt(1), "replicate_count"].sum()
    )
    visits.attrs["random_split_mixed_visits"] = int(
        visit_random_labels.gt(1).sum()
    )
    return visits


def build_rolling_origin_pairs(
    visits: pd.DataFrame,
    horizon_days: int,
    tolerance_days: int = ROLLING_TOLERANCE_DAYS,
) -> pd.DataFrame:
    """Build deterministic, metadata-only rolling-origin visit pairs.

    The origin is from the exact same site/species series, is strictly earlier
    than the target, and is closest to ``horizon_days`` within the inclusive
    tolerance. An equal-distance tie selects the more recent origin (the
    smaller realized lag). No LFMC value participates in pair selection.
    """

    horizon_days = int(horizon_days)
    tolerance_days = int(tolerance_days)
    if horizon_days <= 0:
        raise ScienceGateError("horizon_days must be positive")
    if tolerance_days < 0 or tolerance_days >= horizon_days:
        raise ScienceGateError(
            "tolerance_days must be non-negative and smaller than horizon_days"
        )
    required = set(VISIT_KEYS).union({"_visit_id"})
    missing = sorted(required.difference(visits.columns))
    if missing:
        raise ScienceGateError(
            "Cannot construct rolling pairs; missing visit columns: "
            + ", ".join(missing)
        )

    target_ids = []
    origin_ids = []
    target_dates = []
    origin_dates = []
    realized_lags = []
    sites = []
    species = []

    for (site, species_name), series in visits.groupby(
        ["site_name", "species_collected"], observed=True, sort=True
    ):
        ordered = series.sort_values(
            ["sampling_date", "_visit_id"], kind="mergesort"
        )
        ids = ordered["_visit_id"].to_numpy(dtype=np.int64)
        dates = (
            ordered["sampling_date"]
            .to_numpy(dtype="datetime64[D]")
            .astype(np.int64)
        )
        for target_position in range(1, len(ordered)):
            earlier_dates = dates[:target_position]
            desired_origin = dates[target_position] - horizon_days
            insertion = int(np.searchsorted(earlier_dates, desired_origin))
            candidate_positions = []
            if insertion < target_position:
                candidate_positions.append(insertion)
            if insertion > 0:
                candidate_positions.append(insertion - 1)
            if not candidate_positions:
                continue

            best_position = min(
                candidate_positions,
                key=lambda position: (
                    abs(
                        int(dates[target_position] - dates[position])
                        - horizon_days
                    ),
                    int(dates[target_position] - dates[position]),
                ),
            )
            lag = int(dates[target_position] - dates[best_position])
            if not (
                horizon_days - tolerance_days
                <= lag
                <= horizon_days + tolerance_days
            ):
                continue

            target_ids.append(int(ids[target_position]))
            origin_ids.append(int(ids[best_position]))
            target_dates.append(pd.Timestamp(dates[target_position], unit="D"))
            origin_dates.append(pd.Timestamp(dates[best_position], unit="D"))
            realized_lags.append(lag)
            sites.append(site)
            species.append(species_name)

    return pd.DataFrame(
        {
            "target_visit_id": np.asarray(target_ids, dtype=np.int64),
            "origin_visit_id": np.asarray(origin_ids, dtype=np.int64),
            "site_name": sites,
            "species_collected": species,
            "target_date": pd.to_datetime(target_dates),
            "origin_date": pd.to_datetime(origin_dates),
            "lag_days": np.asarray(realized_lags, dtype=np.int64),
        }
    )


@dataclass(frozen=True)
class RollingPairBaselineSuite:
    """Causal observed-state baselines fit at one forecast horizon."""

    mean_change: Optional[float]
    train_pairs: int

    def predict(self, origin_lfmc: Iterable[float]) -> Dict[str, np.ndarray]:
        origin = np.asarray(list(origin_lfmc), dtype=float)
        predictions = {"observed_state_persistence": origin.copy()}
        if self.mean_change is not None:
            predictions["train_mean_change"] = np.clip(
                origin + self.mean_change, 0.0, MAX_LFMC_VALUE
            )
        return predictions


def _pair_lfmc(
    visits: pd.DataFrame,
    pairs: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resolve pair outcomes after metadata-only pair membership is frozen."""

    values = visits.set_index("_visit_id")["lfmc_value"]
    origin = values.reindex(pairs["origin_visit_id"]).to_numpy(dtype=float)
    target = values.reindex(pairs["target_visit_id"]).to_numpy(dtype=float)
    if not (np.isfinite(origin).all() and np.isfinite(target).all()):
        raise ScienceGateError("Rolling pair LFMC lookup produced missing values")
    return origin, target


def fit_rolling_pair_baselines(
    visits: pd.DataFrame,
    pairs: pd.DataFrame,
    train_target_visit_ids: Iterable[int],
) -> RollingPairBaselineSuite:
    """Fit the horizon-specific correction from training pairs only."""

    train_ids = set(int(value) for value in train_target_visit_ids)
    training_pairs = pairs.loc[pairs["target_visit_id"].isin(train_ids)]
    if training_pairs.empty:
        return RollingPairBaselineSuite(mean_change=None, train_pairs=0)
    origin, target = _pair_lfmc(visits, training_pairs)
    return RollingPairBaselineSuite(
        mean_change=float(np.mean(target - origin)),
        train_pairs=int(len(training_pairs)),
    )


def evaluate_rolling_pair_baselines(
    suite: RollingPairBaselineSuite,
    visits: pd.DataFrame,
    pairs: pd.DataFrame,
) -> Dict[str, Any]:
    """Evaluate fixed causal baselines on an already-selected pair set."""

    if pairs.empty:
        return {
            "pairs": 0,
            "observed_state_persistence": None,
            "train_mean_change": None,
        }
    origin, target = _pair_lfmc(visits, pairs)
    predictions = suite.predict(origin)
    result = {
        "pairs": int(len(pairs)),
        "observed_state_persistence": regression_metrics(
            target, predictions["observed_state_persistence"]
        ),
        "train_mean_change": None,
    }
    if "train_mean_change" in predictions:
        result["train_mean_change"] = regression_metrics(
            target, predictions["train_mean_change"]
        )
    return result


def _pair_membership_digest(pairs: pd.DataFrame) -> str:
    columns = ("target_visit_id", "origin_visit_id", "lag_days")
    membership = pairs.loc[:, columns].to_numpy(dtype="<i8", copy=True)
    return hashlib.sha256(membership.tobytes()).hexdigest()


def _replicate_audit(
    visits: pd.DataFrame,
    mask: pd.Series,
    include_lfmc_dispersion: bool,
) -> Dict[str, Any]:
    selected = visits.loc[mask]
    repeated = selected.loc[selected["replicate_count"].gt(1)]
    result = {
        "visits": int(len(selected)),
        "source_rows": int(selected["replicate_count"].sum()),
        "duplicate_visits": int(len(repeated)),
        "rows_in_duplicate_visits": int(repeated["replicate_count"].sum()),
        "max_replicates_per_visit": (
            int(selected["replicate_count"].max()) if len(selected) else 0
        ),
        "lfmc_dispersion_reported": bool(include_lfmc_dispersion),
    }
    if include_lfmc_dispersion:
        ranges = repeated["lfmc_replicate_range"]
        standard_deviations = repeated["lfmc_replicate_std"]
        result["lfmc_replicate_dispersion"] = {
            "visits": int(len(repeated)),
            "median_range": float(ranges.median()) if len(ranges) else None,
            "p95_range": (
                float(ranges.quantile(0.95)) if len(ranges) else None
            ),
            "median_sample_std": (
                float(standard_deviations.median())
                if len(standard_deviations)
                else None
            ),
        }
    return result


def audit_rolling_origin_pairs(
    visits: pd.DataFrame,
    design: str,
    train_end: pd.Timestamp = DEFAULT_TRAIN_END,
    validation_end: pd.Timestamp = DEFAULT_VALIDATION_END,
    open_test: bool = False,
    horizons_days: Sequence[int] = ROLLING_HORIZONS_DAYS,
    tolerance_days: int = ROLLING_TOLERANCE_DAYS,
) -> Dict[str, Any]:
    """Audit causal pair availability and fixed observed-state baselines."""

    if design not in ROLLING_DESIGNS:
        raise ScienceGateError(
            "Rolling-origin pairs require temporal or spatiotemporal design"
        )
    masks = make_split_masks(visits, design, train_end, validation_end)
    assert_split_integrity(visits, masks, design)

    horizons = {}
    for horizon in horizons_days:
        pairs = build_rolling_origin_pairs(visits, horizon, tolerance_days)
        target_ids_by_partition = {
            partition: set(
                visits.loc[
                    masks[partition], "_visit_id"
                ].to_numpy(dtype=np.int64)
            )
            for partition in PARTITIONS
        }
        baseline_suite = fit_rolling_pair_baselines(
            visits, pairs, target_ids_by_partition["train"]
        )
        partitions = {}
        for partition in PARTITIONS:
            selected = pairs.loc[
                pairs["target_visit_id"].isin(
                    target_ids_by_partition[partition]
                )
            ]
            target_visits = int(masks[partition].sum())
            partitions[partition] = {
                "target_visits": target_visits,
                "pairs": int(len(selected)),
                "coverage": (
                    float(len(selected) / target_visits)
                    if target_visits
                    else 0.0
                ),
                "unique_target_sites": int(selected["site_name"].nunique()),
                "unique_target_series": int(
                    selected.loc[
                        :, ["site_name", "species_collected"]
                    ].drop_duplicates().shape[0]
                ),
                "pair_membership_sha256": _pair_membership_digest(selected),
                "median_realized_lag_days": (
                    float(selected["lag_days"].median())
                    if len(selected)
                    else None
                ),
                "realized_lag_days": {
                    "min": (
                        int(selected["lag_days"].min()) if len(selected) else None
                    ),
                    "p10": (
                        float(selected["lag_days"].quantile(0.10))
                        if len(selected)
                        else None
                    ),
                    "median": (
                        float(selected["lag_days"].median())
                        if len(selected)
                        else None
                    ),
                    "p90": (
                        float(selected["lag_days"].quantile(0.90))
                        if len(selected)
                        else None
                    ),
                    "max": (
                        int(selected["lag_days"].max()) if len(selected) else None
                    ),
                },
                "origin_date_range": {
                    "min": (
                        selected["origin_date"].min().date().isoformat()
                        if len(selected)
                        else None
                    ),
                    "max": (
                        selected["origin_date"].max().date().isoformat()
                        if len(selected)
                        else None
                    ),
                },
                "target_date_range": {
                    "min": (
                        selected["target_date"].min().date().isoformat()
                        if len(selected)
                        else None
                    ),
                    "max": (
                        selected["target_date"].max().date().isoformat()
                        if len(selected)
                        else None
                    ),
                },
            }
        evaluated_partitions = (
            ("validation", "test") if open_test else ("validation",)
        )
        causal_baselines = {
            "evidence_level": (
                "causal observed-state baselines; not Earth4D evidence"
            ),
            "fit": {
                "partition": "train",
                "train_pairs": baseline_suite.train_pairs,
                "mean_change": baseline_suite.mean_change,
                "mean_change_clipped_to_lfmc_range": True,
            },
            "metrics": {},
        }
        for partition in evaluated_partitions:
            selected = pairs.loc[
                pairs["target_visit_id"].isin(
                    target_ids_by_partition[partition]
                )
            ]
            causal_baselines["metrics"][partition] = (
                evaluate_rolling_pair_baselines(
                    baseline_suite, visits, selected
                )
            )
        horizons[str(int(horizon))] = {
            "horizon_days": int(horizon),
            "tolerance_days": int(tolerance_days),
            "partitions": partitions,
            "causal_baselines": causal_baselines,
        }

    replicate_partitions = {
        partition: _replicate_audit(
            visits,
            masks[partition],
            include_lfmc_dispersion=(
                partition != "test" or bool(open_test)
            ),
        )
        for partition in PARTITIONS
    }
    return {
        "evidence_level": "pair-availability audit; not Earth4D evidence",
        "earth4d_evaluated": False,
        "design": design,
        "evaluation_semantics": (
            "rolling new-site-with-history"
            if design == "spatiotemporal"
            else "rolling same-series history"
        ),
        "sequential_release": (
            "Process visits in date order. Reveal a validation/test LFMC value "
            "only after all forecasts targeting that timestamp are scored; it "
            "may then become observed origin state, but never updates weights."
        ),
        "origin_policy": {
            "series_keys": ["site_name", "species_collected"],
            "strictly_earlier": True,
            "closest_to_nominal_horizon": True,
            "tie_break": "more recent origin (smaller realized lag)",
            "pair_partition": "target visit only",
            "selection_uses_lfmc": False,
        },
        "visit_collapse": {
            "keys": list(VISIT_KEYS),
            "lfmc_aggregation": "mean",
            "source_rows": int(visits.attrs.get("source_rows", 0)),
            "unique_visits": int(visits.attrs.get("unique_visits", len(visits))),
            "duplicate_visits": int(visits.attrs.get("duplicate_visits", 0)),
            "rows_in_duplicate_visits": int(
                visits.attrs.get("rows_in_duplicate_visits", 0)
            ),
            "random_split_mixed_visits": int(
                visits.attrs.get("random_split_mixed_visits", 0)
            ),
            "partitions": replicate_partitions,
        },
        "horizons": horizons,
    }


def _set_overlap(
    frame: pd.DataFrame,
    left_mask: pd.Series,
    right_mask: pd.Series,
    columns: Sequence[str],
) -> int:
    if len(columns) == 1:
        left = set(frame.loc[left_mask, columns[0]].tolist())
        right = set(frame.loc[right_mask, columns[0]].tolist())
    else:
        left = set(
            map(tuple, frame.loc[left_mask, list(columns)].itertuples(index=False, name=None))
        )
        right = set(
            map(tuple, frame.loc[right_mask, list(columns)].itertuples(index=False, name=None))
        )
    return len(left.intersection(right))


def _date_range(frame: pd.DataFrame, mask: pd.Series) -> Dict[str, Optional[str]]:
    dates = frame.loc[mask, "sampling_date"]
    if dates.empty:
        return {"min": None, "max": None}
    return {
        "min": dates.min().date().isoformat(),
        "max": dates.max().date().isoformat(),
    }


def _membership_digest(mask: pd.Series) -> str:
    """Hash ordered filtered-row positions assigned to one partition."""
    positions = np.flatnonzero(mask.to_numpy(dtype=bool)).astype("<i8", copy=False)
    return hashlib.sha256(positions.tobytes()).hexdigest()


def audit_split(
    frame: pd.DataFrame,
    masks: Mapping[str, pd.Series],
    design: str,
) -> Dict[str, Any]:
    """Describe overlap and the claim-defining integrity properties."""

    for partition in PARTITIONS:
        if partition not in masks:
            raise ScienceGateError("Missing {} mask".format(partition))
        if len(masks[partition]) != len(frame):
            raise ScienceGateError("{} mask length does not match data".format(partition))

    pairs = (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    )
    row_overlaps = {
        "{}__{}".format(left, right): int((masks[left] & masks[right]).sum())
        for left, right in pairs
    }
    site_overlaps = {
        "{}__{}".format(left, right): _set_overlap(
            frame, masks[left], masks[right], ("site_name",)
        )
        for left, right in pairs
    }
    coordinate_overlaps = {
        "{}__{}".format(left, right): _set_overlap(
            frame, masks[left], masks[right], ("latitude", "longitude")
        )
        for left, right in pairs
    }
    date_overlaps = {
        "{}__{}".format(left, right): _set_overlap(
            frame, masks[left], masks[right], ("sampling_date",)
        )
        for left, right in pairs
    }

    chronological = None
    if design in ("temporal", "spatiotemporal"):
        chronological = bool(
            frame.loc[masks["train"], "sampling_date"].max()
            < frame.loc[masks["validation"], "sampling_date"].min()
            and frame.loc[masks["validation"], "sampling_date"].max()
            < frame.loc[masks["test"], "sampling_date"].min()
        )

    site_disjoint = None
    coordinate_disjoint = None
    if design in ("spatial", "spatiotemporal"):
        site_disjoint = all(value == 0 for value in site_overlaps.values())
        coordinate_disjoint = all(
            value == 0 for value in coordinate_overlaps.values()
        )

    assigned = masks["train"] | masks["validation"] | masks["test"]
    return {
        "counts": {
            partition: int(masks[partition].sum()) for partition in PARTITIONS
        },
        "membership_sha256": {
            partition: _membership_digest(masks[partition])
            for partition in PARTITIONS
        },
        "excluded_rows": int((~assigned).sum()),
        "date_ranges": {
            partition: _date_range(frame, masks[partition])
            for partition in PARTITIONS
        },
        "row_overlaps": row_overlaps,
        "site_overlaps": site_overlaps,
        "coordinate_overlaps": coordinate_overlaps,
        "date_overlaps": date_overlaps,
        "row_disjoint": all(value == 0 for value in row_overlaps.values()),
        "chronological": chronological,
        "site_disjoint": site_disjoint,
        "coordinate_disjoint": coordinate_disjoint,
    }


def assert_split_integrity(
    frame: pd.DataFrame,
    masks: Mapping[str, pd.Series],
    design: str,
) -> Dict[str, Any]:
    """Raise unless the split supports the claim associated with ``design``."""

    audit = audit_split(frame, masks, design)
    empty = [
        partition
        for partition, count in audit["counts"].items()
        if count == 0
    ]
    if empty:
        raise ScienceGateError(
            "{} split has empty partitions: {}".format(design, ", ".join(empty))
        )
    if not audit["row_disjoint"]:
        raise ScienceGateError("{} split assigns rows more than once".format(design))
    if design in ("temporal", "spatiotemporal") and not audit["chronological"]:
        raise ScienceGateError("{} split is not strictly chronological".format(design))
    if design in ("spatial", "spatiotemporal") and not audit["site_disjoint"]:
        raise ScienceGateError("{} split reuses sites across partitions".format(design))
    if (
        design in ("spatial", "spatiotemporal")
        and not audit["coordinate_disjoint"]
    ):
        raise ScienceGateError(
            "{} split reuses coordinates across partitions".format(design)
        )
    return audit


def _group_mean(frame: pd.DataFrame, keys: Sequence[str]) -> pd.Series:
    return frame.groupby(list(keys), observed=True)["lfmc_value"].mean()


def _lookup(
    mapping: pd.Series,
    frame: pd.DataFrame,
    keys: Sequence[str],
) -> np.ndarray:
    if len(keys) == 1:
        values = mapping.reindex(frame[keys[0]].to_numpy()).to_numpy(dtype=float)
    else:
        index = pd.MultiIndex.from_frame(frame.loc[:, list(keys)])
        values = mapping.reindex(index).to_numpy(dtype=float)
    return values


def _fill(primary: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(primary), primary, fallback)


@dataclass
class BaselineSuite:
    """Train-only LFMC reference models.

    No method accepts labels from validation or test data.  ``predict`` uses only
    covariates in the supplied rows and statistics stored by ``fit_baselines``.
    """

    global_mean: float
    species_mean: pd.Series
    species_month_mean: pd.Series
    site_species_month_mean: pd.Series
    persistence_mean: pd.Series
    persistence_date: pd.Series
    forecast_origin: pd.Timestamp

    def predict_with_coverage(
        self, frame: pd.DataFrame
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        n = len(frame)
        global_prediction = np.full(n, self.global_mean, dtype=float)

        species_exact = _lookup(
            self.species_mean, frame, ("species_collected",)
        )
        species_prediction = _fill(species_exact, global_prediction)

        species_month_exact = _lookup(
            self.species_month_mean,
            frame,
            ("species_collected", "_month"),
        )
        species_month_prediction = _fill(
            species_month_exact, species_prediction
        )

        site_species_month_exact = _lookup(
            self.site_species_month_mean,
            frame,
            ("site_name", "species_collected", "_month"),
        )
        site_species_month_prediction = _fill(
            site_species_month_exact, species_month_prediction
        )

        persistence_exact = _lookup(
            self.persistence_mean,
            frame,
            ("site_name", "species_collected"),
        )
        persistence_prediction = _fill(
            persistence_exact, species_month_prediction
        )

        predictions = {
            "global_mean": global_prediction,
            "species_mean": species_prediction,
            "species_month_mean": species_month_prediction,
            "site_species_month_mean": site_species_month_prediction,
            "fixed_origin_persistence": persistence_prediction,
        }
        exact = {
            "global_mean": np.ones(n, dtype=bool),
            "species_mean": np.isfinite(species_exact),
            "species_month_mean": np.isfinite(species_month_exact),
            "site_species_month_mean": np.isfinite(site_species_month_exact),
            "fixed_origin_persistence": np.isfinite(persistence_exact),
        }
        coverage = {
            name: float(mask.mean()) if n else 0.0
            for name, mask in exact.items()
        }
        return predictions, coverage

    def predict(self, frame: pd.DataFrame) -> Dict[str, np.ndarray]:
        predictions, _ = self.predict_with_coverage(frame)
        return predictions


def fit_baselines(train: pd.DataFrame) -> BaselineSuite:
    """Fit all reference baselines using training rows only."""

    if train.empty:
        raise ScienceGateError("Cannot fit LFMC baselines on an empty training set")
    if train["lfmc_value"].isna().any():
        raise ScienceGateError("Training LFMC labels contain missing values")

    latest_dates = train.groupby(
        ["site_name", "species_collected"], observed=True
    )["sampling_date"].transform("max")
    latest = train.loc[train["sampling_date"].eq(latest_dates)]
    persistence_mean = _group_mean(
        latest, ("site_name", "species_collected")
    )
    persistence_date = latest.groupby(
        ["site_name", "species_collected"], observed=True
    )["sampling_date"].max()

    return BaselineSuite(
        global_mean=float(train["lfmc_value"].mean()),
        species_mean=_group_mean(train, ("species_collected",)),
        species_month_mean=_group_mean(
            train, ("species_collected", "_month")
        ),
        site_species_month_mean=_group_mean(
            train, ("site_name", "species_collected", "_month")
        ),
        persistence_mean=persistence_mean,
        persistence_date=persistence_date,
        forecast_origin=pd.Timestamp(train["sampling_date"].max()),
    )


def regression_metrics(
    truth: Iterable[float], prediction: Iterable[float]
) -> Dict[str, Any]:
    """Return LFMC percentage-point MAE/RMSE and coefficient of determination."""

    truth_array = np.asarray(list(truth), dtype=float)
    prediction_array = np.asarray(list(prediction), dtype=float)
    valid = np.isfinite(truth_array) & np.isfinite(prediction_array)
    truth_array = truth_array[valid]
    prediction_array = prediction_array[valid]
    if not len(truth_array):
        raise ScienceGateError("No finite rows available for metric computation")
    residual = truth_array - prediction_array
    denominator = float(np.square(truth_array - truth_array.mean()).sum())
    r2 = None
    if denominator > 0.0:
        r2 = float(1.0 - np.square(residual).sum() / denominator)
    return {
        "n": int(len(truth_array)),
        "mae": float(np.abs(residual).mean()),
        "rmse": float(math.sqrt(np.square(residual).mean())),
        "r2": r2,
    }


def evaluate_baselines(
    suite: BaselineSuite,
    evaluation: pd.DataFrame,
) -> Dict[str, Dict[str, Any]]:
    predictions, coverage = suite.predict_with_coverage(evaluation)
    result = {}
    for name, prediction in predictions.items():
        metrics = regression_metrics(evaluation["lfmc_value"], prediction)
        metrics["exact_group_coverage"] = coverage[name]
        result[name] = metrics
    return result


def run_design(
    frame: pd.DataFrame,
    design: str,
    train_end: pd.Timestamp = DEFAULT_TRAIN_END,
    validation_end: pd.Timestamp = DEFAULT_VALIDATION_END,
    open_test: bool = False,
    rolling_visits: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Audit one design and evaluate its fixed non-Earth4D baselines."""

    masks = make_split_masks(frame, design, train_end, validation_end)
    audit = assert_split_integrity(frame, masks, design)
    train = frame.loc[masks["train"]]
    suite = fit_baselines(train)
    evaluation_partitions = ("validation", "test") if open_test else ("validation",)
    evaluations = {
        partition: evaluate_baselines(suite, frame.loc[masks[partition]])
        for partition in evaluation_partitions
    }

    test_dates = frame.loc[masks["test"], "sampling_date"]
    test_strictly_after_origin = bool(
        len(test_dates) and test_dates.min() > suite.forecast_origin
    )
    result = {
        "claim_scope": CLAIM_SCOPE[design],
        "audit": audit,
        "forecast_origin": suite.forecast_origin.date().isoformat(),
        "test_strictly_after_forecast_origin": test_strictly_after_origin,
        "baselines": evaluations,
    }
    if design in ROLLING_DESIGNS:
        if rolling_visits is None:
            rolling_visits = collapse_lfmc_visits(frame)
        result["rolling_origin"] = audit_rolling_origin_pairs(
            rolling_visits,
            design,
            train_end=train_end,
            validation_end=validation_end,
            open_test=open_test,
        )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_gate(
    path: Path,
    designs: Sequence[str] = DESIGNS,
    train_end: pd.Timestamp = DEFAULT_TRAIN_END,
    validation_end: pd.Timestamp = DEFAULT_VALIDATION_END,
    open_test: bool = False,
    frozen_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run selected LFMC designs and return a JSON-serializable result."""

    frozen_id = None if frozen_id is None else str(frozen_id).strip()
    if open_test and not frozen_id:
        raise ScienceGateError(
            "Opening test requires a non-empty frozen_id identifying the locked run"
        )
    digest, official_verified = verify_official_hash(path)
    frame = load_lfmc_csv(path)
    rolling_visits = (
        collapse_lfmc_visits(frame)
        if any(design in ROLLING_DESIGNS for design in designs)
        else None
    )
    return {
        "schema_version": 2,
        "evidence_level": "data-and-baseline audit",
        "earth4d_evaluated": False,
        "dataset": {
            "path": str(Path(path).resolve()),
            "official_url": OFFICIAL_LFMC_URL if official_verified else None,
            "official_commit": (
                OFFICIAL_LFMC_COMMIT if official_verified else None
            ),
            "official_hash_verified": official_verified,
            "sha256": digest,
            "raw_rows": int(frame.attrs["raw_rows"]),
            "filtered_rows": int(frame.attrs["filtered_rows"]),
            "removed_rows": int(frame.attrs["removed_rows"]),
            "filter": (
                "0 <= lfmc_value <= 302 and non-null latitude, "
                "longitude, elevation"
            ),
        },
        "policy": {
            "train_end_inclusive": pd.Timestamp(train_end).date().isoformat(),
            "validation_end_inclusive": (
                pd.Timestamp(validation_end).date().isoformat()
            ),
            "selection": (
                "fit on train, select on validation, evaluate frozen model "
                "once on test"
            ),
            "test_opened": bool(open_test),
            "frozen_id": frozen_id if open_test else None,
            "gate_code_sha256": _sha256(Path(__file__)),
            "lock_scope": (
                "procedural identifier plus data/code/split hashes; external "
                "append-only storage must enforce one-time model evaluation"
            ),
            "rolling_origin": (
                "metadata-only pair membership is visible in development; "
                "targets are scored sequentially, and test LFMC metrics remain "
                "locked unless test_opened=true"
            ),
        },
        "designs": {
            design: run_design(
                frame,
                design,
                train_end,
                validation_end,
                open_test=open_test,
                rolling_visits=rolling_visits,
            )
            for design in designs
        },
    }


def format_report(result: Mapping[str, Any]) -> str:
    """Render a compact human-readable companion to the JSON artifact."""

    dataset = result["dataset"]
    lines = [
        "Globe-LFMC 2.0 science gate",
        "rows: {filtered_rows:,}/{raw_rows:,} after benchmark filter; "
        "sha256={sha256}".format(**dataset),
    ]
    for design, record in result["designs"].items():
        audit = record["audit"]
        counts = audit["counts"]
        lines.append(
            "\n{}: {} | train={:,} validation={:,} test={:,} excluded={:,}".format(
                design,
                record["claim_scope"],
                counts["train"],
                counts["validation"],
                counts["test"],
                audit["excluded_rows"],
            )
        )
        integrity = "row_disjoint={}".format(audit["row_disjoint"])
        if audit["chronological"] is not None:
            integrity += " chronological={}".format(audit["chronological"])
        if audit["site_disjoint"] is not None:
            integrity += " site_disjoint={}".format(audit["site_disjoint"])
            integrity += " coordinate_disjoint={}".format(
                audit["coordinate_disjoint"]
            )
        lines.append("  " + integrity)
        partition = "test" if "test" in record["baselines"] else "validation"
        lines.append("  reported_partition={}".format(partition))
        for name, metrics in record["baselines"][partition].items():
            r2 = "NA" if metrics["r2"] is None else "{:.3f}".format(metrics["r2"])
            lines.append(
                "  {:27s} MAE={:6.2f} RMSE={:6.2f} R2={:>7s} exact={:5.1%}".format(
                    name,
                    metrics["mae"],
                    metrics["rmse"],
                    r2,
                    metrics["exact_group_coverage"],
                )
            )
        if "rolling_origin" in record:
            rolling = record["rolling_origin"]
            lines.append(
                "  rolling_origin={} [pair availability; not Earth4D evidence]".format(
                    rolling["evaluation_semantics"]
                )
            )
            visit_collapse = rolling["visit_collapse"]
            lines.append(
                "    visits={:,} from rows={:,}; duplicate_visits={:,}".format(
                    visit_collapse["unique_visits"],
                    visit_collapse["source_rows"],
                    visit_collapse["duplicate_visits"],
                )
            )
            validation_replicates = visit_collapse["partitions"]["validation"]
            validation_dispersion = validation_replicates.get(
                "lfmc_replicate_dispersion"
            )
            if validation_dispersion is not None:
                lines.append(
                    "    validation duplicate LFMC median_range={}".format(
                        validation_dispersion["median_range"]
                    )
                )
            for horizon, horizon_record in rolling["horizons"].items():
                pair_partitions = horizon_record["partitions"]
                lines.append(
                    "    {}d +/-{}d pairs: "
                    "train={:,}/{:,} ({:.1%}) "
                    "validation={:,}/{:,} ({:.1%}) "
                    "test={:,}/{:,} ({:.1%})".format(
                        horizon,
                        horizon_record["tolerance_days"],
                        pair_partitions["train"]["pairs"],
                        pair_partitions["train"]["target_visits"],
                        pair_partitions["train"]["coverage"],
                        pair_partitions["validation"]["pairs"],
                        pair_partitions["validation"]["target_visits"],
                        pair_partitions["validation"]["coverage"],
                        pair_partitions["test"]["pairs"],
                        pair_partitions["test"]["target_visits"],
                        pair_partitions["test"]["coverage"],
                    )
                )
                lines.append(
                    "      test_sites={:,} median_lag_days={}".format(
                        pair_partitions["test"]["unique_target_sites"],
                        pair_partitions["test"]["median_realized_lag_days"],
                    )
                )
                metrics_by_partition = horizon_record[
                    "causal_baselines"
                ]["metrics"]
                metric_partition = (
                    "test" if "test" in metrics_by_partition else "validation"
                )
                persistence = metrics_by_partition[metric_partition][
                    "observed_state_persistence"
                ]
                if persistence is not None:
                    lines.append(
                        "      {} causal persistence MAE={:.2f} RMSE={:.2f}".format(
                            metric_partition,
                            persistence["mae"],
                            persistence["rmse"],
                        )
                    )
    return "\n".join(lines)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Official-format LFMC CSV (default: %(default)s)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the official CSV when --data does not exist",
    )
    parser.add_argument(
        "--design",
        action="append",
        choices=DESIGNS,
        help="Design to run; repeat as needed (default: all)",
    )
    parser.add_argument(
        "--train-end",
        default=DEFAULT_TRAIN_END.date().isoformat(),
        help="Inclusive final training date",
    )
    parser.add_argument(
        "--validation-end",
        default=DEFAULT_VALIDATION_END.date().isoformat(),
        help="Inclusive final validation date",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional path for the machine-readable result",
    )
    parser.add_argument(
        "--open-test",
        action="store_true",
        help=(
            "Explicitly evaluate the locked test partition; omit during "
            "development and model selection"
        ),
    )
    parser.add_argument(
        "--frozen-id",
        help=(
            "Immutable run/config identifier required with --open-test; ignored "
            "for development-mode validation reports"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if not args.data.exists():
        if not args.download:
            raise SystemExit(
                "{} does not exist; pass --download to fetch the official CSV".format(
                    args.data
                )
            )
        download_official_lfmc(args.data)

    designs = tuple(args.design) if args.design else DESIGNS
    result = run_gate(
        args.data,
        designs=designs,
        train_end=pd.Timestamp(args.train_end),
        validation_end=pd.Timestamp(args.validation_end),
        open_test=args.open_test,
        frozen_id=args.frozen_id,
    )
    print(format_report(result), flush=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.write("\n")
        print("\nJSON: {}".format(args.json_out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
