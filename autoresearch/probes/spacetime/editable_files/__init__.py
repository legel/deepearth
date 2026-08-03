"""Public Earth4D autoresearch candidate surface.

The fixed probe imports only this module. Internal scientific modules may be
renamed freely without teaching the probe their layout.
"""
from deepearth.autoresearch.probes.spacetime.editable_files.earth4d import Earth4D
from deepearth.autoresearch.probes.spacetime.editable_files.lib.candidate_data import (
    CAPABILITY_CONFIG,
    CHANNELS,
    CONFIG,
    REPAIRED,
    apply_capability_config,
)
from deepearth.autoresearch.probes.spacetime.editable_files.lib.candidate_data import (
    load_dated_gbif_support,
    load_env,
    load_historical_gbif_support,
    load_obs,
    load_vision,
)
from deepearth.autoresearch.probes.spacetime.editable_files.lib.probe_training import (
    GROUP_DRO_TEMPERATURE,
    evaluate_candidate,
    train_candidate,
)
from deepearth.autoresearch.probes.spacetime.editable_files.lib.dyntargets import (
    cooccur_routing,
    sdm_presence,
    sdm_presence_hard,
)
from deepearth.autoresearch.probes.spacetime.editable_files.lib.phenology import run_phenology_all
from deepearth.autoresearch.probes.spacetime.editable_files.lib.recurrence import (
    DEFAULT_TIME_HORIZON,
    normalize_forecast_time,
    normalize_time_from_train,
    phenology_feature_set,
    phenology_mode,
    nearest_dated_conspecific,
    run_recurrence,
    run_recurrence_timecond,
    strict_spatiotemporal_masks,
)


def build_candidate_encoder(config=None, **overrides) -> Earth4D:
    """Build the current Earth4D candidate from its editable experiment profile."""
    cfg = CONFIG if config is None else config
    options = dict(
        verbose=False,
        spatial_levels=cfg["spatial_levels"],
        temporal_levels=cfg["temporal_levels"],
        spatial_log2_hashmap_size=cfg["log2_hashmap"],
        temporal_log2_hashmap_size=cfg["log2_hashmap"],
        freq_log_scale_init=-2.5,
        fourier_features=cfg["fourier"],
        fourier_scale=cfg["fourier_scale"],
        time_harmonics=cfg["time_harmonics"],
        spatial_cline=cfg["spatial_cline"],
        cline_scale=cfg["cline_scale"],
        nystrom=cfg["nystrom"],
        drop_spatiotemporal=cfg["drop_spatiotemporal"],
        tile=cfg["tile"],
        tile_offsets=cfg["tile_offsets"],
        coordinate_system=("geographic" if cfg["geographic"] else "ecef"),
        enable_relative=True,
    )
    options.update(overrides)
    return Earth4D(**options)
