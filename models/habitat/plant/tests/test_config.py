"""Configuration: every path resolves under the data root, which the environment can move."""
import json

from ranges import config


def test_paths_resolve_under_the_data_root(tmp_path, monkeypatch):
    f = tmp_path / "run.json"
    f.write_text(json.dumps({"data_root": "data", "x": "work/a"}))
    monkeypatch.delenv(config.ENV, raising=False)
    cfg = config.load(f)
    assert cfg.root == config.MODEL_DIR / "data" and cfg.path(cfg["x"]) == config.MODEL_DIR / "data/work/a"
    assert cfg.path("/abs/p").as_posix() == "/abs/p" and cfg.path(None) is None
    monkeypatch.setenv(config.ENV, str(tmp_path))
    assert config.load(f).path("work/a") == tmp_path / "work/a"


def test_the_published_configuration_is_complete():
    cfg = config.load()
    for section in ("sources", "species", "per_species", "joint", "scope", "store", "geo", "evaluation"):
        assert section in cfg, section
    from ranges.joint.train import TrainConfig
    tc = TrainConfig.from_dict(cfg["joint"]["train"])
    assert (tc.width, tc.depth, tc.steps, tc.selection) == (256, 3, 21000, "dev_joint_paired")
    assert cfg["store"]["delta"] == 3.2 and cfg["scope"]["region"] == "CONUS" and len(cfg["scope"]["region_l3"]) == 49


def test_the_stages_of_the_full_model_are_complete():
    """Every stage parses; the representation starts from the base, the species stage keeps the representation's
    networks fixed with the same pathways, and the store maps a configured run."""
    cfg = config.load()
    from ranges.joint.train import TrainConfig
    st = cfg["joint"]["stages"]
    tc = {k: TrainConfig.from_dict(v["train"]) for k, v in st.items()}
    assert st["representation"]["init"] == "base" and st["species"]["shared_from"] == "representation"
    rep, spc = tc["representation"], tc["species"]
    assert spc.freeze_shared and not rep.freeze_shared and spc.target_group and spc.shoreline_records
    for k in ("width", "depth", "place_dim", "place_hidden", "place_blocks", "field", "field_dim", "field_heads",
              "field_layers", "field_radii", "field_angles", "field_orders", "calibration_penalty"):
        assert getattr(rep, k) == getattr(spc, k), k
    assert (rep.place_dim, rep.field_angles, rep.continental_background, spc.steps) == (256, 8, 512, 6000)
    assert cfg["store"]["model"] in ("environment", *st)
    assert len(cfg["joint"]["field"]["channels"]) == 12
