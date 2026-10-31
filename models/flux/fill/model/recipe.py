"""Every modeling component of FluxFill in one place: its flag, what it does, where it comes from.

`OPTIONS` are the options of the network and of training that follow a published method; each is off by default, and with
all of them off the network computes the same function as before they existed. `COMPONENTS` lists every component that
the ablation study measures, the established ones and these, with the flags that remove or add it. train.py builds its
command line from `OPTIONS`, train.py and method.py build the network with `net_kwargs`, and ablate.py runs `COMPONENTS`
against one recipe, so a result in the ablation table names exactly one change.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

# The flags of the network as trained (the recipe of v4-d320-48k), without its size and schedule.
FINAL = ("--flow --memory --mem-residual --gated-base --gate-scale --img-nll --edge-continue --year-anomaly --sinr-place 0 "
         "--fast-memory --fused-adam --lazy-sync --imagery-sites all")


@dataclass(frozen=True)
class Option:
    flag: str
    arg: Dict
    source: str


OPTIONS: Tuple[Option, ...] = (
    Option("--var-mean", dict(action="store_true", help="mean loss per variable, then across variables"),
           "4M (Mizrahi et al. 2023), ml-4m fourm/models/fm.py"),
    Option("--mot", dict(default="", choices=["", "ffn", "all"], help="weights per modality: the MLP, or every weight"),
           "Mixture-of-Transformers (Liang et al. 2024); BAGEL modeling/bagel/qwen2_navit.py, Qwen2MoTDecoderLayer"),
    Option("--schedule", dict(default="time,var", help="layer kinds of one repeat: time, var, time_var, var_time"),
           "Timestral config block_schedule; timestral/layers.py TransformerBlock"),
    Option("--attn-gate", dict(action="store_true", help="attention output times sigmoid(linear(inputs))"),
           "Timestral layers.py Attention, sdpa_sigmoid"),
    Option("--rope-partial", dict(type=float, default=1.0, help="share of each head rotated by RoPE (Timestral: 0.5)"),
           "Timestral layers.py SinCosRoPE"),
    Option("--rmsnorm", dict(action="store_true", help="RMSNorm for LayerNorm"), "Timestral layers.py RMSNorm"),
    Option("--swiglu", dict(action="store_true", help="SwiGLU MLP (hidden 1024/384 of d) for GELU"),
           "Timestral layers.py FusedSwiGLU"),
    Option("--quantile-head", dict(action="store_true", help="21 monotone quantiles and the pinball loss for the Student-t head"),
           "Timestral model.py QuantileHead, monotone_quantiles; training loss not published, pinball assumed"),
    Option("--asinh", dict(action="store_true", help="values are asinh of the standardised value"),
           "Timestral scaler.py apply_affine_asinh_masked (its causal per-patch location and scale are not used)"),
    Option("--fgn-noise", dict(type=int, default=0, help="FGN: noise dimension (32); replaces the head and its loss"),
           "FGN (Alet et al. 2025); WeatherNext weathernext2/fgn.py crps_loss, utils/dense.py LinearNormConditioning"),
    Option("--plr", dict(action="store_true", help="periodic embeddings of the value channel"),
           "Gorishniy et al. 2022; rtdl_num_embeddings _Periodic, lite form"),
    Option("--time-enc", dict(default="", choices=["", "prithvi", "sinr"], help="positional encoder of absolute time"),
           "prithvi: terratorch prithvi_mae.py TemporalEncoder; sinr: SINR's network (Cole et al. 2023) on time"),
    Option("--time-octaves", dict(type=int, default=3, help="octaves of the time of year for --time-enc sinr"),
           "NeRF positional encoding (Mildenhall et al. 2020)"),
    Option("--solar", dict(action="store_true", help="solar altitude and top-of-atmosphere radiation per half-hour"),
           "WeatherNext utils/solar_radiation.py"),
    Option("--bottleneck", dict(type=int, default=0, help="width of the token embedding every head reads through; 0 is off"),
           "net.Bottleneck; AlphaEarth Foundations uses a 64-dimensional embedding (Brown et al. 2025)"),
    Option("--hh-read", dict(type=int, default=0, help="width of the read of values visible at each half-hour; 0 is off"),
           "hypernetwork read-out, Ha et al. 2017; net.HalfHourRead"),
    Option("--token-conv", dict(type=int, default=0, help="width of convolutions over a day-token's half-hours; 0 is off"),
           "net.TokenConv"),
    Option("--off", dict(nargs="*", default=[], help="base components to remove: rope time_attn var_attn day_emb doy site "
                                                     "prov_emb var_emb mask_token"), "ablation"),
    Option("--ema-std", dict(type=float, default=0.0, help="power-function EMA of the weights with this relative std"),
           "EDM2 (Karras et al. 2024), training/phema.py PowerFunctionEMA; evaluation and best.pt use the average"),
)

_NET = ("d", "depth", "heads", "dropout", "flow_width", "flow_depth", "memory", "mem_residual", "img_nll", "components",
        "local_base", "gated_base", "gate_scale", "edge_continue", "memory_skew", "driver_anomaly", "scale_anchor",
        "year_anomaly", "sinr_place", "sinr_replace", "scale_anchor_fch4", "schedule", "off", "attn_gate", "rope_partial",
        "rmsnorm", "swiglu", "mot", "quantile_head", "fgn_noise", "plr", "time_enc", "time_octaves", "solar", "hh_read",
        "token_conv", "bottleneck")


def add_arguments(ap) -> None:
    for o in OPTIONS:
        ap.add_argument(o.flag, **o.arg)


def net_kwargs(cfg: Dict) -> Dict:
    """FluxFillNet arguments from a run's arguments or a saved config; a key the config lacks takes the network's default."""
    kw = {k: cfg[k] for k in _NET if cfg.get(k) is not None and k in cfg}
    kw["use_imagery"] = not cfg.get("no_imagery", False)
    kw["memory_skew_fluxes_only"] = cfg.get("memory_skew_fluxes", False)
    return kw


@dataclass(frozen=True)
class Component:
    key: str
    label: str
    remove: str = ""              # flags dropped from FINAL
    add: str = ""                 # flags added to FINAL
    group: str = "input"


COMPONENTS: Tuple[Component, ...] = (
    # inputs
    Component("memory", "Site memory", remove="--memory --mem-residual --gated-base --gate-scale --edge-continue --year-anomaly --fast-memory"),
    Component("imagery", "Sentinel-2 / Clay", add="--no-imagery"),
    Component("site", "Location (site features)", add="--off site"),
    Component("sinr", "SINR location encoder", remove="--sinr-place 0"),
    Component("doy", "Day-of-year harmonics", add="--off doy"),
    Component("day_emb", "Day-of-window embedding", add="--off day_emb"),
    Component("var_emb", "Variable embedding", add="--off var_emb"),
    Component("prov_emb", "Provenance embedding", add="--off prov_emb"),
    Component("mask_token", "Mask token", add="--off mask_token"),
    # backbone
    Component("time_attn", "Time attention", add="--off time_attn", group="backbone"),
    Component("var_attn", "Variable attention", add="--off var_attn", group="backbone"),
    Component("rope", "Rotary day positions", add="--off rope", group="backbone"),
    # anchors and heads
    Component("gated", "Gated local and memory anchors", remove="--gated-base --gate-scale --edge-continue --year-anomaly", group="head"),
    Component("gate_scale", "Evidence-scaled spread", remove="--gate-scale", group="head"),
    Component("edge", "Edge continuation", remove="--edge-continue", group="head"),
    Component("year_anom", "Year anomaly", remove="--year-anomaly", group="head"),
    Component("flow", "Flow head (auxiliary)", remove="--flow", group="head"),
    Component("img_nll", "Imagery likelihood", remove="--img-nll", group="head"),
    # additions from the literature
    Component("var_mean", "Per-variable loss (4M)", add="--var-mean", group="add"),
    Component("mot_ffn", "Per-modality MLPs (MoT)", add="--mot ffn", group="add"),
    Component("mot_all", "Per-modality weights (MoT)", add="--mot all", group="add"),
    Component("gate", "Attention gate (Timestral)", add="--attn-gate", group="add"),
    Component("rope_half", "Partial RoPE (Timestral)", add="--rope-partial 0.5", group="add"),
    Component("rmsnorm", "RMSNorm (Timestral)", add="--rmsnorm", group="add"),
    Component("swiglu", "SwiGLU (Timestral)", add="--swiglu", group="add"),
    Component("tttv", "Time-heavy schedule (Timestral)", add="--schedule time,time,time,time_var --depth 3", group="add"),
    Component("quantile", "Quantile head (Timestral)", add="--quantile-head", remove="--gated-base --gate-scale --edge-continue --year-anomaly", group="add"),
    Component("asinh", "Asinh scaling (Timestral)", add="--asinh", group="add"),
    Component("fgn", "Fair CRPS and shared noise (FGN)", add="--fgn-noise 32", remove="--flow --gated-base --gate-scale --edge-continue --year-anomaly", group="add"),
    Component("plr", "Periodic value embeddings (PLR)", add="--plr", group="add"),
    Component("time_prithvi", "Time encoder (Prithvi)", add="--time-enc prithvi", group="add"),
    Component("time_sinr", "Time encoder (SINR form)", add="--time-enc sinr", group="add"),
    Component("solar", "Solar geometry", add="--solar", group="add"),
    Component("ema", "Weight EMA (EDM2)", add="--ema-std 0.05", group="add"),
)
