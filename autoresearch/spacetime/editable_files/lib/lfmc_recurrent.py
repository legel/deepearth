"""Recurrent LFMC experiment: the registered Earth4D evidence run.

UNVERIFIED DRAFT -- not deployed, not run. Guards must pass and the draft must be independently
reviewed before any science is claimed or launched. TEST PARTITION REMAINS CLOSED.

REGISTERED ENDPOINT
  primary   : 90-day LFMC MAE under joint rolling held-site-with-history (design=spatiotemporal)
  secondary : 30-day, powered
  Validation only.

WHAT THE MODEL MAY CONSUME
  Only OBSERVED ORIGIN STATE plus leak-free metadata (coordinates, elevation, sampling date,
  species, realized lag). The 90-day prediction is produced RECURSIVELY: one 30-day step model is
  applied three times and is fed ITS OWN prediction at each step; it never re-reads an observed
  value after the origin. A pointwise regressor on target coordinates is NOT this experiment.

COORDINATES (defect 1)
  Earth4D receives PHYSICAL lat/lon/elev and a physical day index, with an explicit train-fitted
  GeoAdaptiveRange. Normalizing to [0,1] and handing that to the default ECEF path would have made
  the encoder interpret unit-cube numbers as metres. The range's time span is extended past the
  full validation horizon, because the hash grid returns identical features beyond its representable
  time range -- which would silently erase the forecast axis.

TRAINING (defect 2)
  Encoder features are computed INSIDE the training loop with gradients enabled, so the registered
  encoder is trained end-to-end within the equal per-arm budget. Precomputing them under no_grad
  froze the hash table and could not establish learned-Earth4D science.

ARM PARITY (defect 3)
  Every arm projects its encoding through an adapter to a COMMON WIDTH and shares one head
  architecture, so head parameter counts and initialization are matched and asserted.

CONTROLS (defect 4)
  persistence, seasonal_climatology, raw, generic_pe, matched_mlp, propagator_no_earth4d,
  no_history, shuffled_history, time_reversal, future_sentinel. None may be collapsed or omitted
  for a confirmatory artifact.

INFERENCE (defect 5)
  >=5 matched seeds, site-by-year block bootstrap, lower 95% CI of the improvement > 0, point gain
  >= 5%, MAE/RMSE/R2 reported, and the registered no-regression floor.

PROVENANCE (defect 6)
  Code/data/split/config/seed hashes and an append-only, hash-chained per-arm ledger. A plain
  overwriteable JSON is discovery-only and cannot carry a confirmatory claim.

PROTOCOL MINIMUMS (defect 7)
  Fewer than 5 seeds, under 600 s/arm, a missing primary/secondary horizon, or an omitted required
  arm downgrades the artifact to discovery-only; it can never be emitted as confirmatory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from deepearth.autoresearch.spacetime.editable_files.lib.science_gate import (
    DEFAULT_DATA_PATH,
    DEFAULT_TRAIN_END,
    DEFAULT_VALIDATION_END,
    assert_split_integrity,
    build_rolling_origin_pairs,
    collapse_lfmc_visits,
    evaluate_rolling_pair_baselines,
    fit_rolling_pair_baselines,
    load_lfmc_csv,
    make_split_masks,
    prepare_lfmc_frame,
    regression_metrics,
)

DESIGN = "spatiotemporal"           # joint rolling held-site-with-history
STEP_DAYS = 30
PRIMARY_HORIZON = 90
SECONDARY_HORIZON = 30
MIN_SEEDS = 5
MIN_SECONDS = 600.0
MIN_POINT_GAIN = 0.05               # >= 5% MAE improvement
COMMON_WIDTH = 128                  # every arm's encoding is adapted to this width

# Learned arms (trained through run_arm). persistence/seasonal_climatology are computed
# analytically from TRAIN pairs and carry no learned parameters.
LEARNED_ARMS = (
    "earth4d",
    "raw",
    "generic_pe",
    "matched_mlp",
    "propagator_no_earth4d",
    "no_history",
    "shuffled_history",
    "time_reversal",
    "future_sentinel",
)
ANALYTIC_ARMS = ("persistence", "seasonal_climatology")
REQUIRED_ARMS = tuple(sorted(set(LEARNED_ARMS) | set(ANALYTIC_ARMS)))


class ProtocolError(RuntimeError):
    """Raised when a run cannot satisfy the registered protocol."""


class FutureAccessError(RuntimeError):
    """Raised when a model path touches a value it is not allowed to see."""


# --------------------------------------------------------------------------- provenance (6)
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _split_digest(visits: pd.DataFrame, masks: Dict[str, pd.Series]) -> str:
    parts = {
        name: sorted(visits.loc[mask, "_visit_id"].to_numpy(np.int64).tolist())
        for name, mask in masks.items()
        if name != "test"          # the test partition is never materialized here
    }
    return _sha256_obj(parts)


class LedgerWriter:
    """Append-only, hash-chained per-arm outcomes. Each record signs the previous one."""

    def __init__(self, path: Path, provenance: Dict):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._prev = self._tail_signature()
        self.append({"record_type": "provenance", **provenance})

    def _tail_signature(self) -> str:
        if not self.path.exists():
            return "genesis"
        last = ""
        for line in self.path.read_text().splitlines():
            if line.strip():
                last = line
        return json.loads(last)["signature"] if last else "genesis"

    def append(self, record: Dict) -> str:
        body = dict(record)
        body["prev_signature"] = self._prev
        body["signature"] = hashlib.sha256(
            (self._prev + json.dumps(body, sort_keys=True, default=str)).encode()
        ).hexdigest()
        with open(self.path, "a") as fh:      # append-only; existing lines are never rewritten
            fh.write(json.dumps(body, sort_keys=True, default=str) + "\n")
        self._prev = body["signature"]
        return body["signature"]


# --------------------------------------------------------------------------- data plumbing
def _day_index(dates) -> np.ndarray:
    series = dates if isinstance(dates, pd.Series) else pd.Series(dates)
    return (series.to_numpy(dtype="datetime64[D]")
            - np.datetime64("1970-01-01")).astype(np.float64)


def _pair_table(visits: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    v = visits.set_index("_visit_id")

    def col(name, ids):
        return v[name].reindex(ids).to_numpy()

    o = pairs["origin_visit_id"]
    t = pairs["target_visit_id"]
    out = pd.DataFrame({
        "origin_lfmc": col("lfmc_value", o).astype(float),
        "target_lfmc": col("lfmc_value", t).astype(float),
        "lat": col("latitude", o).astype(float),
        "lon": col("longitude", o).astype(float),
        "elev": col("elevation", o).astype(float),
        "origin_day": _day_index(v["sampling_date"].reindex(o)),
        "target_day": _day_index(v["sampling_date"].reindex(t)),
        "lag_days": pairs["lag_days"].to_numpy(float),
        "site_name": col("site_name", t),
        "species_collected": col("species_collected", t),
        "target_month": v["_month"].reindex(t).to_numpy(),
    })
    out["target_year"] = pd.Series(
        pd.to_datetime(v["sampling_date"].reindex(t).to_numpy())).dt.year.to_numpy()
    return out.dropna(subset=["origin_lfmc", "target_lfmc", "lat", "lon", "elev"]).reset_index(
        drop=True)


def _coords(tab: pd.DataFrame, day: np.ndarray) -> np.ndarray:
    """PHYSICAL coordinates: degrees, degrees, metres, day index. No unit-cube rescaling."""
    return np.stack([tab["lat"].to_numpy(float), tab["lon"].to_numpy(float),
                     tab["elev"].to_numpy(float), np.asarray(day, dtype=float)], 1)


# --------------------------------------------------------------------------- encoders (1,2,3)
def default_earth4d_factory(train_coords: np.ndarray, horizon_days: int, device,
                            time_max_day: Optional[float] = None):
    """Build Earth4D over an explicit TRAIN-fitted geographic range with horizon headroom.

    Imported lazily so CPU-only guard tests can collect this module without CUDA.
    """
    from deepearth.encoders.spacetime.earth4d import Earth4D, GeoAdaptiveRange

    lat, lon, elev, day = (train_coords[:, i] for i in range(4))
    pad = 0.05
    span_lat = max(lat.max() - lat.min(), 1e-3)
    span_lon = max(lon.max() - lon.min(), 1e-3)
    span_elev = max(elev.max() - elev.min(), 1.0)
    # Time must reach past the FULL validation horizon: beyond its range the hash grid returns
    # identical features for every future date, which would erase the forecast axis entirely.
    geo_range = GeoAdaptiveRange(
        lat_min=float(lat.min() - pad * span_lat), lat_max=float(lat.max() + pad * span_lat),
        lon_min=float(lon.min() - pad * span_lon), lon_max=float(lon.max() + pad * span_lon),
        elev_min=float(elev.min() - pad * span_elev), elev_max=float(elev.max() + pad * span_elev),
        time_min=float(day.min()),
        # The domain must cover the WHOLE registered validation window. A train_max + k*horizon pad
        # left 359/812 (90d) and 371/1262 (30d) validation targets outside the hash's temporal
        # range, where it returns identical features -- erasing the forecast axis for ~44%/29% of
        # the evaluation sets. The bound comes from DEFAULT_VALIDATION_END, a split-definition
        # constant, so no observed validation row is consulted.
        time_max=float(time_max_day if time_max_day is not None
                       else day.max() + 2.0 * horizon_days + 1.0),
        buffer_fraction=0.0, mode="adaptive",
    )
    return Earth4D(verbose=False, coordinate_system="geographic", geo_range=geo_range,
                   spatial_levels=16, temporal_levels=16,
                   spatial_log2_hashmap_size=20, temporal_log2_hashmap_size=20).to(device)


class PositionalArm(nn.Module):
    """Encodes PHYSICAL (lat, lon, elev, day) for one arm, then adapts to the common width.

    Earth4D is trained end-to-end: encode() runs with gradients so the hash table receives them.
    """

    def __init__(self, arm: str, train_coords: np.ndarray, device, seed: int, horizon_days: int,
                 common_width: int = COMMON_WIDTH,
                 encoder_factory: Optional[Callable] = None,
                 time_max_day: Optional[float] = None):
        super().__init__()
        self.arm = arm
        self.device = device
        self.encoder = None
        kind = _arm_encoder_kind(arm)
        if kind == "earth4d":
            factory = encoder_factory or default_earth4d_factory
            try:
                self.encoder = factory(train_coords, horizon_days, device,
                                       time_max_day=time_max_day)
            except TypeError:                      # injected stubs may take the 3-arg form
                self.encoder = factory(train_coords, horizon_days, device)
            with torch.no_grad():
                probe = torch.zeros(2, 4, device=device)
                probe[:, 0] = float(train_coords[0, 0]); probe[:, 1] = float(train_coords[0, 1])
                probe[:, 2] = float(train_coords[0, 2]); probe[:, 3] = float(train_coords[0, 3])
                raw_dim = int(self.encoder(probe).shape[1])
        else:
            # raw and RFF share one normalization fitted on TRAIN coordinates only
            self._lo = train_coords.min(0)
            self._span = np.maximum(train_coords.max(0) - train_coords.min(0), 1e-6)
            # same registered bound for the non-encoder arms, so every arm sees one time domain
            if time_max_day is not None:
                self._span[3] = max(float(time_max_day) - float(self._lo[3]), 1e-6)
            else:
                self._span[3] = self._span[3] + 2.0 * horizon_days
            if kind == "rff":
                rng = np.random.default_rng(seed)
                self._B = rng.normal(0.0, 4.0, size=(4, 128)).astype(np.float32)
                raw_dim = 256
            else:
                raw_dim = 4
        self.raw_dim = raw_dim
        self.adapter = nn.Linear(raw_dim, common_width).to(device)
        self.out_dim = common_width

    def _encode_raw(self, coords: np.ndarray) -> torch.Tensor:
        kind = _arm_encoder_kind(self.arm)
        if kind == "earth4d":
            if self.encoder is None:
                raise RuntimeError("earth4d arm requires an encoder")
            c = torch.tensor(coords.astype(np.float32), device=self.device)
            return self.encoder(c)                     # gradients flow into the hash table
        n = ((coords - self._lo) / self._span).astype(np.float32)
        if kind == "rff":
            proj = 2.0 * np.pi * (n @ self._B)
            return torch.tensor(np.concatenate([np.sin(proj), np.cos(proj)], 1).astype(np.float32),
                                device=self.device)
        return torch.tensor(n, device=self.device)

    def forward(self, coords: np.ndarray) -> torch.Tensor:
        return self.adapter(self._encode_raw(coords))


def _arm_encoder_kind(arm: str) -> str:
    if arm in ("earth4d", "no_history", "shuffled_history", "time_reversal", "future_sentinel"):
        return "earth4d"                                # controls isolate STATE, not the encoder
    if arm in ("generic_pe", "matched_mlp"):
        return "rff"
    return "raw"                                        # raw, propagator_no_earth4d


class StepModel(nn.Module):
    """One 30-day step: (adapted origin coord, adapted step coord, state) -> LFMC change."""

    def __init__(self, pos_dim: int, state_dim: int, hidden: int = 256):
        super().__init__()
        self.state_dim = state_dim
        # State is projected to a fixed width so head parameter counts match across arms even when
        # an arm withholds state (defect 3).
        self.state_proj = nn.Linear(max(state_dim, 1), 16)
        self.net = nn.Sequential(
            nn.Linear(2 * pos_dim + 16, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, p_origin, p_step, state):
        if self.state_dim == 0:
            state = torch.zeros(p_origin.shape[0], 1, device=p_origin.device)
        return self.net(torch.cat([p_origin, p_step, self.state_proj(state)], -1)).squeeze(-1)


def head_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.net.parameters()) + \
        sum(p.numel() for p in model.state_proj.parameters())


# --------------------------------------------------------------------------- controls (4)
@dataclass
class ArmSpec:
    name: str
    use_state: bool = True
    shuffle_state: bool = False       # shuffled-history control
    reverse_time: bool = False        # time-reversal control
    sentinel: bool = False            # future-sentinel canary
    analytic: bool = False


ARM_SPECS: Dict[str, ArmSpec] = {
    "earth4d": ArmSpec("earth4d"),
    "raw": ArmSpec("raw"),
    "generic_pe": ArmSpec("generic_pe"),
    "matched_mlp": ArmSpec("matched_mlp"),
    "propagator_no_earth4d": ArmSpec("propagator_no_earth4d"),
    "no_history": ArmSpec("no_history", use_state=False),
    "shuffled_history": ArmSpec("shuffled_history", shuffle_state=True),
    "time_reversal": ArmSpec("time_reversal", reverse_time=True),
    "future_sentinel": ArmSpec("future_sentinel", sentinel=True),
    "persistence": ArmSpec("persistence", analytic=True),
    "seasonal_climatology": ArmSpec("seasonal_climatology", analytic=True),
}

SENTINEL = -987654.0                  # any leakage of a future value shows up as this magnitude


def _state_tensor(values: np.ndarray, mu: float, sd: float, spec: ArmSpec, device,
                  rng: np.random.Generator) -> torch.Tensor:
    if not spec.use_state:
        return torch.zeros(len(values), 0, device=device)
    v = np.asarray(values, dtype=np.float64)
    if spec.shuffle_state:
        v = v[rng.permutation(len(v))]     # destroys the pairing between state and target
    return torch.tensor(((v - mu) / sd).astype(np.float32), device=device)[:, None]


# --------------------------------------------------------------------------- statistics (5)
def block_bootstrap_ci(arm_err: np.ndarray, ref_err: np.ndarray, blocks: np.ndarray,
                       n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Site-by-year block bootstrap of the MAE improvement (ref - arm). Returns (lo, hi, point)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(blocks)
    idx_by_block = {b: np.flatnonzero(blocks == b) for b in uniq}
    point = float(ref_err.mean() - arm_err.mean())
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        take = np.concatenate([idx_by_block[b] for b in chosen])
        draws[i] = ref_err[take].mean() - arm_err[take].mean()
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), point


# --------------------------------------------------------------------------- the experiment
def _calibrate_steps(model, pos, opt, train_tab, o_days, t_days, y, mu, sd, spec, device, rng,
                     seconds: float, n: int, bs: int, probe_steps: int = 8) -> int:
    """Measure this arm's throughput, then convert the wall-clock budget into a step count."""
    if n <= 0 or seconds <= 0:
        return 0
    t0 = time.time()
    for _ in range(probe_steps):
        sel = rng.integers(0, n, size=min(bs, n))
        sub = train_tab.iloc[sel]
        loss = nn.functional.l1_loss(
            model(pos(_coords(sub, o_days[sel])), pos(_coords(sub, t_days[sel])),
                  _state_tensor(sub["origin_lfmc"].to_numpy(), mu, sd, spec, device, rng)),
            y[torch.tensor(sel, device=device)])
        opt.zero_grad(); loss.backward(); opt.step()
    rate = probe_steps / max(time.time() - t0, 1e-9)
    return max(int(seconds * rate), 1)


def run_arm(arm: str, train_tab: pd.DataFrame, eval_tabs: Dict[int, pd.DataFrame], seed: int,
            seconds: float, device, horizon_days: int, lr: float = 1e-3, steps: Optional[int] = None,
            encoder_factory: Optional[Callable] = None,
            step_model_factory: Optional[Callable] = None,
            positional_factory: Optional[Callable] = None,
            state_source: Optional[Callable] = None,
            time_max_day: Optional[float] = None) -> Dict:
    """Train the 30-day step model for `seconds`, then evaluate recursively at each horizon.

    `state_source(step_index, current_prediction, model_view)` supplies the state fed to each step.
    The registered default returns the model's OWN previous prediction, which is what makes this
    autoregressive. It is injectable so a guard can substitute a deliberately leaky source and prove
    the future-sentinel canary actually fires -- a canary on a channel no leak can reach is
    worthless.
    """
    spec = ARM_SPECS[arm]
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    train_coords = _coords(train_tab, train_tab["origin_day"].to_numpy())
    pos_factory = positional_factory or PositionalArm
    try:
        pos = pos_factory(arm, train_coords, device, seed, horizon_days,
                          encoder_factory=encoder_factory, time_max_day=time_max_day)
    except TypeError:
        pos = pos_factory(arm, train_coords, device, seed, horizon_days,
                          encoder_factory=encoder_factory)
    make_head = step_model_factory or StepModel
    model = make_head(pos.out_dim, 1 if spec.use_state else 0).to(device)

    params = list(model.parameters()) + list(pos.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    mu = float(train_tab["origin_lfmc"].mean())
    sd = float(train_tab["origin_lfmc"].std()) or 1.0
    train_origin_mean = mu                      # the no-history rollout's starting value
    o_days = train_tab["origin_day"].to_numpy()
    t_days = train_tab["target_day"].to_numpy()
    if spec.reverse_time:
        o_days, t_days = t_days, o_days          # control: run the step backwards in time
    y = torch.tensor((train_tab["target_lfmc"].to_numpy()
                      - train_tab["origin_lfmc"].to_numpy()).astype(np.float32), device=device)

    # A raw wall-clock stop makes the run NON-DETERMINISTIC at fixed seed (two runs take different
    # step counts and diverge). So the budget DERIVES a step count -- measured once on this arm --
    # and the step count is what the run is pinned to. `steps` may be passed explicitly for exact
    # reproduction; the realized seconds are recorded either way so budget parity stays auditable.
    n, bs = len(train_tab), 1024
    t0 = time.time()
    if steps is None:
        steps = _calibrate_steps(model, pos, opt, train_tab, o_days, t_days, y, mu, sd, spec,
                                 device, rng, seconds, n, bs)
    executed = 0
    for _ in range(steps):
        if n <= 0:
            break
        sel = rng.integers(0, n, size=min(bs, n))
        sub = train_tab.iloc[sel]
        # features are computed INSIDE the loop so gradients reach the encoder (defect 2)
        p_o = pos(_coords(sub, o_days[sel]))
        p_s = pos(_coords(sub, t_days[sel]))
        st = _state_tensor(sub["origin_lfmc"].to_numpy(), mu, sd, spec, device, rng)
        loss = nn.functional.l1_loss(model(p_o, p_s, st), y[torch.tensor(sel, device=device)])
        opt.zero_grad(); loss.backward(); opt.step()
        executed += 1

    out = {"train_steps": executed, "planned_steps": int(steps), "head_params": head_parameter_count(model),
           "encoder_params": sum(p.numel() for p in pos.parameters()),
           "seconds": float(time.time() - t0), "horizons": {}}
    model.eval()
    for horizon, tab in eval_tabs.items():
        scoring_truth = tab["target_lfmc"].to_numpy().astype(np.float64)
        model_view = tab.copy()
        if spec.sentinel:
            # canary: the model path sees a sentinel instead of the target value; if any prediction
            # depends on it, the sentinel's magnitude propagates and scoring raises.
            model_view["target_lfmc"] = SENTINEL
        with torch.no_grad():
            # A no-history arm must not smuggle the observed origin in as the rollout's starting
            # value: withholding only the state VECTOR still leaks it through `cur`. Start from a
            # TRAIN-derived constant instead.
            cur = (np.full(len(model_view), train_origin_mean, dtype=np.float64)
                   if not spec.use_state
                   else model_view["origin_lfmc"].to_numpy().astype(np.float64))
            day = model_view["origin_day"].to_numpy().astype(np.float64)
            tgt_day = model_view["target_day"].to_numpy().astype(np.float64)
            n_steps = max(int(round(horizon / STEP_DAYS)), 1)
            source = state_source or (lambda k, current, view: current)
            for k in range(n_steps):
                nxt = tgt_day if k == n_steps - 1 else np.minimum(day + STEP_DAYS, tgt_day)
                a_day, b_day = (nxt, day) if spec.reverse_time else (day, nxt)
                p_o = pos(_coords(model_view, a_day))
                p_s = pos(_coords(model_view, b_day))
                fed = np.asarray(source(k, cur, model_view), dtype=np.float64)
                if spec.sentinel and np.any(np.abs(fed) > abs(SENTINEL) / 2):
                    raise FutureAccessError(
                        f"arm {arm!r} fed a sentinel-poisoned value into step {k}: a future "
                        "observation reached the recursion")
                st = _state_tensor(fed, mu, sd, spec, device, rng)
                cur = cur + model(p_o, p_s, st).cpu().numpy().astype(np.float64)
                day = nxt
        if spec.sentinel and np.any(np.abs(cur) > abs(SENTINEL) / 2):
            raise FutureAccessError(
                f"arm {arm!r} produced sentinel-scale output at {horizon}d: a future value reached "
                "the model path")
        m = regression_metrics(scoring_truth, cur)
        m["errors"] = np.abs(scoring_truth - cur)
        m["predictions"] = cur.copy()
        out["horizons"][horizon] = m
    return out


def analytic_arm(arm: str, train_tab: pd.DataFrame, eval_tabs: Dict[int, pd.DataFrame],
                 visits: pd.DataFrame, pairs_by_h: Dict[int, pd.DataFrame],
                 train_ids: set) -> Dict:
    """persistence and seasonal_climatology -- fitted on TRAIN pairs only, no learned parameters."""
    out = {"train_steps": 0, "head_params": 0, "encoder_params": 0, "horizons": {}}
    clim = (train_tab.groupby(["species_collected", "target_month"])["target_lfmc"].mean()
            if arm == "seasonal_climatology" else None)
    global_mean = float(train_tab["target_lfmc"].mean()) if clim is not None else 0.0
    for horizon, tab in eval_tabs.items():
        truth = tab["target_lfmc"].to_numpy().astype(np.float64)
        if arm == "persistence":
            pred = tab["origin_lfmc"].to_numpy().astype(np.float64)
        else:
            lookup = {} if clim is None else clim.to_dict()
            keys = list(zip(tab["species_collected"], tab["target_month"]))
            pred = np.array([lookup.get(k, global_mean) for k in keys], dtype=np.float64)
        m = regression_metrics(truth, pred)
        m["errors"] = np.abs(truth - pred)
        m["predictions"] = pred.copy()
        out["horizons"][horizon] = m
    return out


def protocol_violations(seeds: int, seconds: float, horizons: Sequence[int],
                        arms: Sequence[str]) -> List[str]:
    """Defect 7: any of these forbids a confirmatory artifact."""
    v = []
    if seeds < MIN_SEEDS:
        v.append(f"seeds={seeds} < required {MIN_SEEDS}")
    if seconds < MIN_SECONDS:
        v.append(f"seconds_per_arm={seconds} < required {MIN_SECONDS}")
    for h in (PRIMARY_HORIZON, SECONDARY_HORIZON):
        if h not in horizons:
            v.append(f"missing registered horizon {h}d")
    missing = sorted(set(REQUIRED_ARMS).difference(set(arms)))
    if missing:
        v.append("missing required arms: " + ", ".join(missing))
    return v


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    ap.add_argument("--seeds", type=int, default=MIN_SEEDS)
    ap.add_argument("--seconds", type=float, default=MIN_SECONDS)
    ap.add_argument("--horizons", type=int, nargs="+", default=[PRIMARY_HORIZON, SECONDARY_HORIZON])
    ap.add_argument("--arms", nargs="+", default=list(REQUIRED_ARMS))
    ap.add_argument("--comparator", default="persistence")
    ap.add_argument("--ledger", type=Path, default=Path("autoresearch/spacetime/data/lfmc/earth4d_recurrent_ledger.jsonl"))
    ap.add_argument("--json-out", type=Path, default=Path("autoresearch/spacetime/data/lfmc/earth4d_recurrent.json"))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")

    violations = protocol_violations(a.seeds, a.seconds, a.horizons, a.arms)
    status = "discovery-only" if violations else "confirmatory-eligible"

    frame = prepare_lfmc_frame(load_lfmc_csv(a.data))
    visits = collapse_lfmc_visits(frame)
    masks = make_split_masks(visits, DESIGN, DEFAULT_TRAIN_END, DEFAULT_VALIDATION_END)
    assert_split_integrity(visits, masks, DESIGN)
    train_ids = set(visits.loc[masks["train"], "_visit_id"].to_numpy(np.int64).tolist())
    val_ids = set(visits.loc[masks["validation"], "_visit_id"].to_numpy(np.int64).tolist())

    step_pairs = build_rolling_origin_pairs(visits, STEP_DAYS)
    train_tab = _pair_table(visits, step_pairs.loc[step_pairs["target_visit_id"].isin(train_ids)])
    eval_tabs, pairs_by_h, gate_null = {}, {}, {}
    for horizon in a.horizons:
        hp = build_rolling_origin_pairs(visits, horizon)
        pairs_by_h[horizon] = hp
        eval_tabs[horizon] = _pair_table(visits, hp.loc[hp["target_visit_id"].isin(val_ids)])
        suite = fit_rolling_pair_baselines(visits, hp, train_ids)
        gate_null[horizon] = evaluate_rolling_pair_baselines(
            suite, visits, hp.loc[hp["target_visit_id"].isin(val_ids)])

    provenance = {
        "code_sha256": _sha256_file(Path(__file__)),
        "data_sha256": _sha256_file(a.data),
        "split_digest": _split_digest(visits, masks),
        "config_sha256": _sha256_obj(vars(a)),
        "seeds": list(range(a.seeds)),
        "design": DESIGN, "step_days": STEP_DAYS, "partition": "validation",
        "test_opened": False, "status": status, "protocol_violations": violations,
    }
    ledger = LedgerWriter(a.ledger, provenance)

    print(f"=== LFMC RECURRENT [{status}] design={DESIGN} step={STEP_DAYS}d "
          f"train_pairs={len(train_tab)} eval={{h: len(t) for h, t in eval_tabs.items()}} "
          f"seeds={a.seeds}x{a.seconds:.0f}s TEST CLOSED ===", flush=True)
    for v in violations:
        print(f"  !! PROTOCOL: {v}", flush=True)

    per_arm_errors: Dict[str, Dict[int, np.ndarray]] = {}
    results: Dict[str, Dict] = {}
    for arm in a.arms:
        spec = ARM_SPECS[arm]
        seed_runs = []
        if spec.analytic:
            seed_runs.append(analytic_arm(arm, train_tab, eval_tabs, visits, pairs_by_h,
                                          train_ids))
        else:
            for seed in range(a.seeds):
                seed_runs.append(run_arm(arm, train_tab, eval_tabs, seed, a.seconds, device,
                                         max(a.horizons)))
        summary = {}
        for h in a.horizons:
            maes = [r["horizons"][h]["mae"] for r in seed_runs]
            summary[str(h)] = {
                "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
                "rmse_mean": float(np.mean([r["horizons"][h]["rmse"] for r in seed_runs])),
                "r2_mean": float(np.mean([r["horizons"][h]["r2"] for r in seed_runs])),
                "per_seed_mae": [float(x) for x in maes],
                "head_params": seed_runs[0]["head_params"],
                "encoder_params": seed_runs[0]["encoder_params"],
            }
            per_arm_errors.setdefault(arm, {})[h] = np.mean(
                [r["horizons"][h]["errors"] for r in seed_runs], axis=0)
        results[arm] = summary
        ledger.append({"record_type": "arm", "arm": arm, "summary": summary})
        for h in a.horizons:
            s = summary[str(h)]
            print(f"  {arm:22} {h:3d}d MAE {s['mae_mean']:6.2f} +/- {s['mae_std']:.2f} "
                  f"RMSE {s['rmse_mean']:6.2f} R2 {s['r2_mean']:+.3f}", flush=True)

    # inference vs the registered comparator (defect 5)
    inference = {}
    ref = a.comparator
    if ref in per_arm_errors:
        for arm in a.arms:
            if arm == ref:
                continue
            inference[arm] = {}
            for h in a.horizons:
                tab = eval_tabs[h]
                blocks = (tab["site_name"].astype(str) + "|" + tab["target_year"].astype(str)).to_numpy()
                lo, hi, point = block_bootstrap_ci(per_arm_errors[arm][h],
                                                   per_arm_errors[ref][h], blocks)
                ref_mae = float(per_arm_errors[ref][h].mean())
                gain = point / ref_mae if ref_mae else 0.0
                passes = (lo > 0) and (gain >= MIN_POINT_GAIN) and not violations
                inference[arm][str(h)] = {
                    "improvement_mae": point, "ci95_low": lo, "ci95_high": hi,
                    "relative_gain": gain, "passes_registered_bar": bool(passes),
                    "comparator": ref, "blocks": int(len(np.unique(blocks))),
                }
                ledger.append({"record_type": "inference", "arm": arm, "horizon": h,
                               **inference[arm][str(h)]})

    out = {"status": status, "protocol_violations": violations, "provenance": provenance,
           "gate_null": gate_null, "arms": results, "inference": inference,
           "eval_pairs": {h: int(len(t)) for h, t in eval_tabs.items()}}
    a.json_out.parent.mkdir(parents=True, exist_ok=True)
    a.json_out.write_text(json.dumps(out, indent=1, default=str))
    print(f"JSON: {a.json_out}   LEDGER: {a.ledger}", flush=True)
    return out


if __name__ == "__main__":
    main()
