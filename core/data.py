"""Data adapters: load observations from a source into the arrays a training run needs.

An adapter reads a specific data source and exposes a uniform interface: per-observation variable values, a mask
of which variables are present, coordinates, a spatial hold-out split, and a nearest-neighbor index. Register a
new source by adding an adapter and listing its name in a config.
"""
from __future__ import annotations
import csv, glob
from pathlib import Path
from typing import Callable, Dict
import numpy as np
import torch
from scipy.spatial import cKDTree

_REGISTRY: Dict[str, Callable] = {}


def register(name: str):
    def wrap(cls):
        _REGISTRY[name] = cls
        return cls
    return wrap


def build(name: str, **kwargs):
    return _REGISTRY[name](**kwargs)


def _normalize(a):
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)


@register("california")
class California:
    """California observations. Each carries coordinates, two ground-image representations, a categorical
    identity, an evolutionary-position vector, and several categorical descriptors. Builds a spatial hold-out
    split (whole 0.5 degree cells) and a nearest-neighbor index."""

    _traits = ["plant_type", "growth_rate", "seasonality", "sun", "water", "soil_drainage", "ease_of_care", "form"]

    # attributes that fully define the assembled dataset (so a prepared cache can restore it without the ~2.7 min
    # glob + KD-tree neighbor build). Tensors are saved on CPU and moved to ``device`` on load.
    _PREPARED_KEYS = ("n", "n_classes", "reference_latitude_deg", "n_neighbors", "holdout", "time_axis", "_time_km",
                      "dims", "trait_classes", "group_names", "binomial", "_tip_labels", "train", "test",
                      "_train_bool", "time_span_days", "time_cut",
                      "lat", "lon", "elev", "cls", "dino", "bio", "phylo", "traits", "coords", "class_group",
                      "species_text", "neighbors", "gbifID")

    def __init__(self, cache_dir: str, n_neighbors: int = 24, device: str = "cuda", holdout_fraction: float = 1 / 6,
                 holdout: str = "spatial", subset: dict | None = None, time_axis: bool = False,
                 meta_path: str | None = None, time_km: float = 50.0, prepared: str | None = None):
        if prepared and Path(prepared).exists():           # fast path: restore the assembled dataset from a cache
            self._load_prepared(prepared, device); return
        cache = Path(cache_dir); dev = self.device = device; self.n_neighbors = n_neighbors
        self.reference_latitude_deg = 37.0
        # Observation time (event date). When off, the time coordinate is a constant 0 (space-only, the historical
        # default). When on, each observation's date is normalized to [0, 1] over the dataset's time span and placed
        # in coords[:, 3], activating the Earth4D temporal axis (absolute) and the neighbor time-offset (relative).
        self.time_axis = time_axis
        self._time_km = float(time_km)                     # normalized-time -> km weight for the neighbor KD-tree
        self._cache = cache
        self._meta_path = meta_path or self._find_meta(cache)

        vocab = np.load(cache / "gbif_vocab.npz", allow_pickle=True)
        phylo = _normalize(vocab["E1"].astype(np.float32))
        self.n_classes = len(vocab["global_idx"])
        gid, cls, lat, lon, dino, bio = [], [], [], [], [], []
        for f in sorted(glob.glob(str(cache / "gbif_tokens" / "*.npz"))):
            d = np.load(f)
            gid.append(d["gbifID"]); cls.append(d["species_local"]); lat.append(d["lat"]); lon.append(d["lon"])
            dino.append(d["dino"]); bio.append(d["bio"])
        gid = np.concatenate(gid); cls = np.concatenate(cls).astype(np.int64)
        lat = np.concatenate(lat).astype(np.float32); lon = np.concatenate(lon).astype(np.float32)
        dino = _normalize(np.concatenate(dino).astype(np.float32)); bio = _normalize(np.concatenate(bio).astype(np.float32))
        elev = np.zeros(len(gid), np.float32)
        if (cache / "gbif_elev.npz").exists():
            ge = np.load(cache / "gbif_elev.npz"); em = dict(zip(ge["gbifID"].tolist(), ge["elev"].tolist()))
            elev = np.array([em.get(int(g), 0.0) for g in gid], np.float32)
        rows = list(csv.DictReader(open(cache / "derived/species_index.csv")))
        self._tip_labels = [rows[i]["tip_label"] for i in vocab["global_idx"]]   # Newick leaf label per species
        groups = np.array([rows[i]["family"] for i in vocab["global_idx"]])
        self.group_names = sorted(set(groups.tolist())); gmap = {g: i for i, g in enumerate(self.group_names)}
        self.class_group = torch.tensor([gmap[g] for g in groups], device=dev)
        z = np.load(cache / "derived/traits_syn.npz", allow_pickle=True)
        self.trait_classes = {t: int(len(z[f"catvocab_{t}"])) for t in self._traits}
        traits = np.stack([z[f"cat_{t}"][vocab["global_idx"]] for t in self._traits], 1)

        self.n = len(gid)
        self.lat = torch.tensor(lat, device=dev); self.lon = torch.tensor(lon, device=dev)
        self.elev = torch.tensor(elev, device=dev); self.cls = torch.tensor(cls, device=dev)
        self.dino = torch.tensor(dino, device=dev); self.bio = torch.tensor(bio, device=dev)
        self.phylo = torch.tensor(phylo, device=dev); self.traits = torch.tensor(traits, device=dev)
        self.species_text = None                          # BioCLIP text embedding per species (inductive placement)
        if (cache / "bioclip_text_emb.npy").exists():
            te = _normalize(np.load(cache / "bioclip_text_emb.npy")[vocab["global_idx"]].astype(np.float32))
            self.species_text = torch.tensor(te, device=dev)
        tnorm = self._load_event_time(gid) if self.time_axis else np.zeros_like(lat)
        self.coords = torch.tensor(np.stack([lat, lon, elev, tnorm], 1), device=dev)
        self.dims = {"vision_dino": dino.shape[1], "vision_bio": bio.shape[1], "phylo": phylo.shape[1]}

        # Optional subset (a fast-benchmark lever): keep only observations inside a bounding box or belonging to a
        # set of families, applied BEFORE the split + neighbor index so both build on the reduced set. Per-observation
        # arrays are reindexed to the kept rows; per-species arrays (phylo, traits, species_text, class_group) are
        # indexed through ``cls`` and so remain full-length and correct.
        if subset:
            gid, cls, lat, lon, elev = self._apply_subset(subset, gid, cls, lat, lon, elev, dev)
        self.gbifID = gid                                  # per-observation GBIF id (post-subset order), for I/O bundles
        self.binomial = vocab["binomial"]                  # species-local index -> binomial name (e.g. "Quercus agrifolia")

        # hold-out split. "spatial" hides whole 0.5 degree cells (transfer to unseen places); "phylo" hides whole
        # families (transfer to unseen clades). Whichever is chosen, its members never appear in training.
        self.holdout = holdout
        rng = np.random.default_rng(0)
        if holdout == "temporal":
            # forecasting split: the LATEST fraction of observations (by event time) is held out, so the test set
            # lies strictly in the future of every training observation. Requires time_axis=True.
            if not self.time_axis:
                raise ValueError("holdout='temporal' requires time_axis=True")
            tnorm = self.coords[:, 3].cpu().numpy()
            cut = np.quantile(tnorm, 1.0 - holdout_fraction)
            self.test = np.where(tnorm >= cut)[0]
            self.time_cut = float(cut)
        elif holdout == "phylo":
            obs_family = self.class_group.cpu().numpy()[cls]
            families = np.unique(obs_family); rng.shuffle(families)   # only families actually present (post-subset)
            held = families[: max(1, int(round(len(families) * holdout_fraction)))]
            self.test = np.where(np.isin(obs_family, held))[0]
        else:
            cell = (np.floor(lat / 0.5).astype(np.int64) * 10007 + np.floor(lon / 0.5).astype(np.int64))
            cells = np.unique(cell); rng.shuffle(cells)
            self.test = np.where(np.isin(cell, cells[: max(1, int(len(cells) * holdout_fraction))]))[0]
        self.train = np.setdiff1d(np.arange(self.n), self.test)
        self.train_index = torch.tensor(self.train, device=dev)
        self._train_bool = np.zeros(self.n, bool); self._train_bool[self.train] = True   # for train-only normalization
        self._build_neighbors()
        self.extra = {}                                    # extra continuous modalities keyed by name
        self._load_modalities(cache, gid, dev)
        self.tree = self._load_tree(cache)                 # the dated phylogeny as message-passing buffers (or None)
        if prepared:                                       # persist the assembled dataset for instant reload
            self._save_prepared(prepared)

    def _save_prepared(self, path: str) -> None:
        """Pickle the fully-assembled dataset (tensors on CPU) so a later run restores it in ~1s instead of rebuilding
        the glob + KD-tree neighbor index. Includes the extra modalities and the parsed tree buffers."""
        blob = {}
        for k in self._PREPARED_KEYS:
            v = getattr(self, k, None)
            blob[k] = v.detach().cpu() if torch.is_tensor(v) else v
        blob["extra"] = {n: (t.cpu(), h.cpu(), d) for n, (t, h, d) in self.extra.items()}
        blob["tree"] = self.tree
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(blob, path)

    def _load_prepared(self, path: str, device: str) -> None:
        """Restore a dataset saved by :meth:`_save_prepared`, moving tensors to ``device``."""
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.device = device
        for k, v in blob.items():
            if k in ("extra", "tree"):
                continue
            setattr(self, k, v.to(device) if torch.is_tensor(v) else v)
        self.extra = {n: (t.to(device), h.to(device), d) for n, (t, h, d) in blob["extra"].items()}
        self.tree = blob["tree"]
        self.train_index = torch.tensor(self.train, device=device)     # derived: train row indices as a device tensor

    @staticmethod
    def _find_meta(cache: Path):
        """Locate observations_meta.parquet (carries eventDate per gbifID) across machines/layouts."""
        cands = [cache / "observations_meta.parquet",
                 cache.parent / "deepearth_gbif" / "observations_meta.parquet",
                 Path.home() / "deepearth/data/deepearth_gbif/observations_meta.parquet",
                 Path("/home/photon/4tb/deepearth_gbif/observations_meta.parquet")]
        for c in cands:
            if c.exists():
                return str(c)
        return str(cands[0])

    def _load_event_time(self, gid):
        """Per-observation event time normalized to [0, 1] over the dataset span, aligned to ``gid``.

        Reads ``eventDate`` (falling back to year/month/day) from the observations metadata parquet, keyed by
        gbifID. Times are converted to days-since-epoch, then min-max normalized so the earliest observation is 0
        and the latest is 1 — the units the Earth4D absolute temporal axis expects (it maps [0,1] -> [-1,1]).
        Observations with no parseable date take the dataset's median time (a neutral value). ``self.time_span_days``
        records the physical span so a relative time window can be set in the same normalized units
        (1.0 == the whole span; e.g. 1 day == 1/span)."""
        # Prefer a lightweight precomputed sidecar (gbifID -> days-since-epoch), so machines without a parquet
        # engine still get the time axis. Fall back to reading the parquet with pandas.
        sidecar = Path(self._cache) / "gbif_eventtime.npz"
        if sidecar.exists():
            z = np.load(sidecar)
            lut_gid, lut_days = z["gbifID"], z["days"].astype(np.float64)
            order = np.argsort(lut_gid); lut_gid, lut_days = lut_gid[order], lut_days[order]
            pos = np.searchsorted(lut_gid, gid).clip(max=len(lut_gid) - 1)
            hit = lut_gid[pos] == gid
            days = np.where(hit, lut_days[pos], np.nan)
        else:
            import pandas as pd
            meta = pd.read_parquet(self._meta_path, columns=["gbifID", "eventDate", "year", "month", "day"])
            meta = meta.drop_duplicates("gbifID").set_index("gbifID")
            sub = meta.reindex(gid)
            dt = pd.to_datetime(sub["eventDate"], errors="coerce", utc=True)
            ymd = pd.to_datetime(dict(year=sub["year"].fillna(0).astype(int),          # fallback for unparsed dates
                                      month=sub["month"].clip(1, 12).fillna(1).astype(int),
                                      day=sub["day"].clip(1, 28).fillna(1).astype(int)),
                                 errors="coerce", utc=True)
            dt = dt.fillna(ymd)
            days = (dt.view("int64").to_numpy().astype(np.float64)) / (1e9 * 86400.0)   # ns -> days since 1970
        days[~np.isfinite(days)] = np.nan
        valid = np.isfinite(days)
        if valid.sum() == 0:
            raise ValueError(f"time_axis=True but no parseable dates in {self._meta_path}")
        tmin, tmax = np.nanmin(days), np.nanmax(days)
        self.time_span_days = float(tmax - tmin) if tmax > tmin else 1.0
        med = np.nanmedian(days)
        days[~valid] = med
        tnorm = (days - tmin) / self.time_span_days
        self._n_dated = int(valid.sum())
        return np.clip(tnorm, 0.0, 1.0).astype(np.float32)

    def _load_tree(self, cache):
        """Parse the dated Newick tree and align it to the model's species, for tree-structured message passing.

        Returns the buffer dict from :func:`phylo_tree.build_tree_buffers` (leaves in model-species order), or
        ``None`` if the tree file is absent. The buffers are static topology; the species-graph operator registers
        them as tensors. Independent of the observation subset/split (the tree is over the full species set)."""
        nwk = cache / "ca_subtree.dated.nwk"
        if not nwk.exists():
            return None
        from deepearth.encoders.biological.phylogenomic import build_tree_buffers
        return build_tree_buffers(str(nwk), self._tip_labels)

    def _apply_subset(self, subset, gid, cls, lat, lon, elev, dev):
        """Restrict to a bounding box (``{"bbox": [lat0, lat1, lon0, lon1]}``) and/or a set of families
        (``{"families": [...]}``, matched through ``class_group``/``group_names``). Reindexes every per-observation
        array (numpy locals + the ``self.*`` tensors, plus any already-loaded ``self.extra`` modalities) to the kept
        rows and resets ``self.n``; returns the reindexed numpy locals the split + modality loader still consume."""
        keep = np.ones(len(gid), bool)
        if subset.get("bbox") is not None:
            lat0, lat1, lon0, lon1 = subset["bbox"]
            keep &= (lat >= min(lat0, lat1)) & (lat <= max(lat0, lat1))
            keep &= (lon >= min(lon0, lon1)) & (lon <= max(lon0, lon1))
        if subset.get("families") is not None:
            gmap = {g: i for i, g in enumerate(self.group_names)}
            want = [gmap[f] for f in subset["families"] if f in gmap]
            obs_family = self.class_group.cpu().numpy()[cls]
            keep &= np.isin(obs_family, want)
        idx = np.where(keep)[0]
        if len(idx) == 0:
            raise ValueError(f"subset {subset} kept 0 of {len(gid)} observations")
        ti = torch.tensor(idx, device=dev)
        gid, cls, lat, lon, elev = gid[idx], cls[idx], lat[idx], lon[idx], elev[idx]
        for a in ("lat", "lon", "elev", "cls", "dino", "bio", "coords"):
            setattr(self, a, getattr(self, a)[ti])
        if getattr(self, "extra", None):
            self.extra = {n: (v[ti], h[ti], d) for n, (v, h, d) in self.extra.items()}
        self.n = len(gid)
        return gid, cls, lat, lon, elev

    def _add_modality(self, name, ids, rows, gid, dev, zscore=False, normalize=False, valid=None):
        """Align a feature matrix (keyed by its own gbifID ``ids``) to the observation order and store it with a
        presence mask. ``zscore`` standardizes per channel (for physical series); ``normalize`` unit-scales each
        row (for embedding vectors); ``valid`` is an optional per-source-row mask of which rows carry real data."""
        rows = np.nan_to_num(rows.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if valid is None:
            valid = np.ones(len(ids), bool)
        order = np.argsort(ids); ids, rows, valid = ids[order], rows[order], valid[order]
        pos = np.searchsorted(ids, gid).clip(max=len(ids) - 1); have = (ids[pos] == gid) & valid[pos]
        arr = np.zeros((len(gid), rows.shape[1]), np.float32); arr[have] = rows[pos[have]]
        if zscore:
            fit = have & getattr(self, "_train_bool", np.ones(len(gid), bool))   # normalize on train only (no test-stat leak)
            m, s = arr[fit].mean(0), arr[fit].std(0) + 1e-6; arr = np.clip((arr - m) / s, -10.0, 10.0)
        if normalize and have.any():
            arr[have] = arr[have] / (np.linalg.norm(arr[have], axis=1, keepdims=True) + 1e-9)
        self.extra[name] = (torch.tensor(arr, device=dev), torch.tensor(have, device=dev), rows.shape[1])

    def _load_modalities(self, cache, gid, dev):
        """Load every extra modality present in the cache, each aligned to the observations by gbifID."""
        dm = sorted(glob.glob(str(cache / "gbif_daymet_tokens" / "*.npz")))          # Daymet: 180 days x 7 vars
        if dm:
            ids = np.concatenate([np.load(f)["gbifID"] for f in dm])
            rows = np.concatenate([np.load(f)["daymet"].reshape(len(np.load(f)["gbifID"]), -1) for f in dm])
            self._add_modality("climate", ids, rows, gid, dev, zscore=True)
        nf = sorted(glob.glob(str(cache / "gbif_naip_tokens" / "*.npz")))            # NAIP DINOv3-SAT493M (RGB, IR)
        if nf:
            ids = np.concatenate([np.load(f)["gbifID"] for f in nf])
            for key, name in (("rgb_pool", "naip_rgb"), ("ir_pool", "naip_ir")):
                rows = np.concatenate([np.load(f)[key] for f in nf])
                self._add_modality(name, ids, rows, gid, dev, normalize=True)
        clay = cache / "gbif_clay_tokens.npz"                                        # Clay 1.5 Sentinel-2
        if clay.exists():
            z = np.load(clay)
            self._add_modality("clay", z["gbifID"], z["clay"], gid, dev, normalize=True,
                               valid=z["has_clay"] if "has_clay" in z else None)
        soil = cache / "gbif_soil_tokens.npz"                                        # SSURGO soil properties (9)
        if soil.exists():
            z = np.load(soil)
            self._add_modality("soil", z["gbifID"], z["soil"], gid, dev, zscore=True, valid=z["has_soil"])

    def _frame(self, idx):
        lat = self.lat.cpu().numpy()[idx]; lon = self.lon.cpu().numpy()[idx]; elev = self.elev.cpu().numpy()[idx]
        # Neighbor selection is spatial by default; with the time axis on, a modest ``time_km`` weight makes it
        # spatio-temporal (space still dominates, but recency breaks ties) so neighbors carry real time offsets.
        t = self.coords[:, 3].cpu().numpy()[idx] * self._time_km if self.time_axis else np.zeros(len(idx), np.float32)
        return np.stack([lat * 111.0, lon * 111.0 * np.cos(np.radians(self.reference_latitude_deg)), elev / 50.0,
                         t], 1)

    def _build_neighbors(self):
        tree = cKDTree(self._frame(self.train))
        # Exclude self by GLOBAL index, not KDTree column 0: with duplicate coordinates a train obs's own row can land
        # past column 0, leaking its own identity/features into its context. Query a few extra, push any self entry to
        # the end (stable sort on the self-mask), then keep the first n_neighbors.
        _, a = tree.query(self._frame(self.train), k=self.n_neighbors + 4); cand = self.train[a]
        is_self = cand == self.train[:, None]
        cand = np.take_along_axis(cand, np.argsort(is_self, axis=1, kind="stable"), axis=1)
        nn_tr = cand[:, : self.n_neighbors]
        _, b = tree.query(self._frame(self.test), k=self.n_neighbors); nn_te = self.train[b]   # test uses train-only tree: no self
        nbr = np.zeros((self.n, self.n_neighbors), np.int64); nbr[self.train] = nn_tr; nbr[self.test] = nn_te
        self.neighbors = torch.tensor(nbr, device=self.device)

    def variable_dims(self):
        """Widths for the config's variables, filled from the data (vector dims, class counts, trait descriptors)."""
        d = {**self.dims, "identity_classes": self.n_classes, "trait_classes": self.trait_classes}
        for name, (_, _, dim) in self.extra.items():
            d[name] = dim
        return d

    def memory(self, size: int = 4096):
        """A memory bank for experience replay: anchor observations keyed by their neighborhood's habitat
        signature (mean neighbor vision), with the anchors' own features. Returns ``(key [M, dim], features)``."""
        m = min(size, len(self.train))
        anchors = self.train_index[torch.randint(0, len(self.train_index), (m,), device=self.device)]
        key = self.dino[self.neighbors[anchors]].mean(1)
        key = key / key.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return key, {"vision_dino": self.dino[anchors]}

    def batch(self, idx):
        """Return one batch: variable values, observed masks, query and neighbor coordinates, the coordinates in
        each vector subspace (here the biological one), and the neighbors' own feature values."""
        ci = self.neighbors[idx]
        values = {"vision_dino": self.dino[idx], "vision_bio": self.bio[idx], "identity": self.cls[idx],
                  "phylo": self.phylo[self.cls[idx]]}
        observed = {n: torch.ones(len(idx), dtype=torch.bool, device=self.device) for n in values}
        for k, t in enumerate(self.trait_classes):
            values[t] = (self.traits[self.cls[idx], k] - 1).clamp(0)
            observed[t] = self.traits[self.cls[idx], k] > 0
        for name, (vals, have, _) in self.extra.items():
            values[name] = vals[idx]; observed[name] = have[idx]
        manifold_positions = {"biological": self.phylo[self.cls[ci]]}   # neighbors' known positions only
        neighbor_values = {"identity": self.cls[ci], "vision_dino": self.dino[ci]}
        return values, observed, self.coords[idx], self.coords[ci], manifold_positions, neighbor_values
