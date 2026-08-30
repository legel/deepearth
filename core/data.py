"""Production data for the California world model."""
from __future__ import annotations

import csv
import glob
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

def build(name: str, **kwargs):
    if name != "california":
        raise ValueError(f"unknown data source: {name}")
    return California(**kwargs)


def _normalize(a):
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)


class California:
    """California observations aligned by GBIF ID with held-out neighbors."""

    _traits = ["plant_type", "growth_rate", "seasonality", "sun", "water", "soil_drainage", "ease_of_care", "form"]

    _PREPARED_KEYS = ("n", "n_classes", "reference_latitude_deg", "n_neighbors", "holdout", "time_axis", "_time_km",
                      "dims", "trait_classes", "group_names", "binomial", "_tip_labels", "train", "test",
                      "_train_bool", "time_span_days", "time_cut",
                      "lat", "lon", "elev", "cls", "dino", "bio", "phylo", "traits", "coords", "class_group",
                      "species_text", "neighbors", "gbifID",
                      "lfmc", "lfmc_valid", "flower", "flower_valid", "myco", "myco_valid",
                      "obs_month", "month_tnorm", "species_peak_month")

    def __init__(self, cache_dir: str, n_neighbors: int = 24, device: str = "cuda", holdout_fraction: float = 1 / 6,
                 holdout: str = "spatial", subset: dict | None = None, time_axis: bool = False,
                 meta_path: str | None = None, time_km: float = 50.0, prepared: str | None = None,
                 clay_v2: bool = False):
        self._clay_v2 = clay_v2
        if prepared and Path(prepared).exists():
            self._load_prepared(prepared, device)
            return
        cache = Path(cache_dir)
        dev = self.device = device
        self.n_neighbors = n_neighbors
        self.reference_latitude_deg = 37.0
        self.time_axis = time_axis
        self._time_km = float(time_km)
        self._cache = cache
        self._meta_path = meta_path or self._find_meta(cache)

        gid, cls, lat, lon, elev, vocab = self._load_observations(cache, dev)

        if subset:
            gid, cls, lat, lon, elev = self._apply_subset(subset, gid, cls, lat, lon, elev, dev)
        self.gbifID = gid
        self.binomial = vocab["binomial"]

        self._split(cls, lat, lon, holdout, holdout_fraction, dev)
        self._load_scientific_data(cache, gid, dev)
        if prepared:
            self._save_prepared(prepared)

    def _load_observations(self, cache, dev):
        vocab = np.load(cache / "gbif_vocab.npz", allow_pickle=True)
        phylo = _normalize(vocab["E1"].astype(np.float32))
        self.n_classes = len(vocab["global_idx"])
        chunks = [
            np.load(path) for path in sorted(
                glob.glob(str(cache / "gbif_tokens" / "*.npz"))
            )
        ]
        arrays = {
            name: np.concatenate([chunk[name] for chunk in chunks])
            for name in ("gbifID", "species_local", "lat", "lon", "dino", "bio")
        }
        gid = arrays["gbifID"]
        cls = arrays["species_local"].astype(np.int64)
        lat, lon = (arrays[name].astype(np.float32) for name in ("lat", "lon"))
        dino, bio = (
            _normalize(arrays[name].astype(np.float32))
            for name in ("dino", "bio")
        )
        elev = np.zeros(len(gid), np.float32)
        elevation = cache / "gbif_elev.npz"
        if elevation.exists():
            values = np.load(elevation)
            lookup = dict(zip(values["gbifID"].tolist(), values["elev"].tolist()))
            elev = np.array([lookup.get(int(item), 0) for item in gid], np.float32)

        rows = list(csv.DictReader(open(cache / "derived/species_index.csv")))
        indices = vocab["global_idx"]
        self._tip_labels = [rows[index]["tip_label"] for index in indices]
        groups = np.array([rows[index]["family"] for index in indices])
        self.group_names = sorted(set(groups.tolist()))
        group_index = {name: index for index, name in enumerate(self.group_names)}
        self.class_group = torch.tensor(
            [group_index[name] for name in groups], device=dev
        )
        trait_data = np.load(
            cache / "derived/traits_syn.npz", allow_pickle=True
        )
        self.trait_classes = {
            name: len(trait_data[f"catvocab_{name}"]) for name in self._traits
        }
        traits = np.stack([
            trait_data[f"cat_{name}"][indices] for name in self._traits
        ], 1)

        self.n = len(gid)
        for name, values in (
            ("lat", lat), ("lon", lon), ("elev", elev), ("cls", cls),
            ("dino", dino), ("bio", bio), ("phylo", phylo),
            ("traits", traits),
        ):
            setattr(self, name, torch.tensor(values, device=dev))
        self.has_vision = self.dino.abs().sum(-1) > 1e-6
        self.species_text = self._load_species_text(cache, indices, dev)
        time = self._load_event_time(gid) if self.time_axis else np.zeros_like(lat)
        self.coords = torch.tensor(
            np.stack((lat, lon, elev, time), 1), device=dev
        )
        self.dims = {
            "vision_dino": dino.shape[1], "vision_bio": bio.shape[1],
            "phylo": phylo.shape[1],
        }
        return gid, cls, lat, lon, elev, vocab

    @staticmethod
    def _load_species_text(cache, indices, dev):
        taxon = cache / "bioclip_taxon_text_emb.npy"
        if taxon.exists():
            values = np.load(taxon).astype(np.float32)
        else:
            path = cache / "bioclip_text_emb.npy"
            if not path.exists():
                return None
            values = _normalize(np.load(path)[indices].astype(np.float32))
        return torch.tensor(values, device=dev)

    def _split(self, cls, lat, lon, holdout, fraction, dev):
        self.holdout = holdout
        rng = np.random.default_rng(0)
        if holdout == "temporal":
            if not self.time_axis:
                raise ValueError("holdout='temporal' requires time_axis=True")
            time = self.coords[:, 3].cpu().numpy()
            self.time_cut = float(np.quantile(time, 1 - fraction))
            self.test = np.where(time >= self.time_cut)[0]
        elif holdout == "phylo":
            family = self.class_group.cpu().numpy()[cls]
            groups = np.unique(family)
            rng.shuffle(groups)
            held = groups[:max(1, round(len(groups) * fraction))]
            self.test = np.where(np.isin(family, held))[0]
        else:
            cell = (
                np.floor(lat / 0.5).astype(np.int64) * 10007
                + np.floor(lon / 0.5).astype(np.int64)
            )
            cells = np.unique(cell)
            rng.shuffle(cells)
            held = cells[:max(1, int(len(cells) * fraction))]
            self.test = np.where(np.isin(cell, held))[0]
        self.train = np.setdiff1d(np.arange(self.n), self.test)
        self.train_index = torch.tensor(self.train, device=dev)
        self._train_bool = np.zeros(self.n, bool)
        self._train_bool[self.train] = True
        self._build_neighbors()

    def _load_scientific_data(self, cache, gid, dev):
        self.extra = {}
        self._load_modalities(cache, gid, dev)
        self._load_pollinator(cache, dev)
        lfmc = cache / "gbif_lfmc.npz"
        if lfmc.exists():
            values = np.load(lfmc)
            self.lfmc = torch.tensor(values["lfmc"], device=dev)
            self.lfmc_valid = torch.tensor(values["has_lfmc"], device=dev)
        myco = cache / "gbif_mycorrhiza.npz"
        if myco.exists():
            values = np.load(myco, allow_pickle=True)
            self.myco = torch.tensor(values["myco"].astype(np.int64), device=dev)
            self.myco_valid = torch.tensor(values["has_myco"], device=dev)
            self.myco_classes = list(values["classes"])
        self._load_flowering(cache, gid, dev)
        self.tree = self._load_tree(cache)
        self.lca_tree, self.lca_tip_row = self._load_tree_lca(cache)

    def _load_flowering(self, cache, gid, dev):
        path = cache / "gbif_flower_all.npz"
        if not path.exists():
            return
        data = np.load(path)
        lookup = dict(zip(data["gbifID"].astype(int), data["flower"].astype(float)))
        flower = np.array([lookup.get(int(item), 0) for item in gid], np.float32)
        valid = np.array([int(item) in lookup for item in gid], np.float32)
        self.flower = torch.tensor(flower, device=dev)
        self.flower_valid = torch.tensor(valid, device=dev)
        if not hasattr(self, "obs_month"):
            return
        classes = self.cls.cpu().numpy()
        observed = valid > 0.5
        peak = np.full(self.n_classes, -1, np.int64)
        for species in np.unique(classes[observed]):
            rows = observed & (classes == species)
            if rows.sum() < 8:
                continue
            rates = np.array([
                flower[rows & (self.obs_month == month)].mean()
                if (rows & (self.obs_month == month)).any() else -1
                for month in range(12)
            ])
            if (rates >= 0).sum() >= 3:
                peak[species] = rates.argmax()
        self.species_peak_month = torch.tensor(peak, device=dev)

    def _save_prepared(self, path: str) -> None:
        """Pickle the assembled dataset (tensors on CPU, plus extra modalities and tree buffers) for fast reload."""
        blob = {}
        for k in self._PREPARED_KEYS:
            v = getattr(self, k, None)
            blob[k] = v.detach().cpu() if torch.is_tensor(v) else v
        blob["extra"] = {n: (t.cpu(), h.cpu(), d) for n, (t, h, d) in self.extra.items()}
        for k in ("poll_idx", "poll_frq", "poll_valid", "n_pollinators"):
            if hasattr(self, k):
                v = getattr(self, k)
                blob[k] = v.cpu() if torch.is_tensor(v) else v
        blob["tree"] = self.tree
        blob["lca_tree"] = self.lca_tree
        blob["lca_tip_row"] = self.lca_tip_row
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(blob, path)

    def _load_prepared(self, path: str, device: str) -> None:
        """Restore a dataset saved by :meth:`_save_prepared`, moving tensors to ``device``."""
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.device = device
        for k, v in blob.items():
            if k in ("extra", "tree", "lca_tree", "lca_tip_row"):
                continue
            setattr(self, k, v.to(device) if torch.is_tensor(v) else v)
        self.extra = {n: (t.to(device), h.to(device), d) for n, (t, h, d) in blob["extra"].items()}
        self.tree = blob["tree"]
        self.lca_tree = blob.get("lca_tree")
        self.lca_tip_row = blob["lca_tip_row"].to(device) if torch.is_tensor(blob.get("lca_tip_row")) else None
        self.train_index = torch.tensor(self.train, device=device)
        self.has_vision = self.dino.abs().sum(-1) > 1e-6

    @staticmethod
    def _find_meta(cache: Path):
        """Locate observations_meta.parquet (carries eventDate per gbifID) across machines/layouts."""
        cands = [cache / "observations_meta.parquet",
                 cache.parent / "deepearth_gbif" / "observations_meta.parquet",
                 Path.home() / "deepearth/data/deepearth_gbif/observations_meta.parquet"]
        for c in cands:
            if c.exists():
                return str(c)
        return str(cands[0])

    def _load_event_time(self, gid):
        """Per-observation event time min-max normalized to [0,1] over the dataset span, aligned to ``gid``.

        Reads ``eventDate`` (falling back to year/month/day) keyed by gbifID; unparseable dates take the median.
        ``self.time_span_days`` records the physical span so a relative time window uses the same normalized units."""
        sidecar = Path(self._cache) / "gbif_eventtime.npz"
        if sidecar.exists():
            z = np.load(sidecar)
            lut_gid, lut_days = z["gbifID"], z["days"].astype(np.float64)
            order = np.argsort(lut_gid)
            lut_gid, lut_days = lut_gid[order], lut_days[order]
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
        tnorm = np.clip((days - tmin) / self.time_span_days, 0.0, 1.0).astype(np.float32)
        import pandas as pd
        self.obs_month = pd.to_datetime(days, unit="D", origin="unix").month.to_numpy().astype(np.int64) - 1   # per-obs calendar month 0-11
        self.month_tnorm = np.array([float(tnorm[self.obs_month == m].mean()) if (self.obs_month == m).any()   # data-driven normalized-time anchor per month
                                     else (m + 0.5) / 12 for m in range(12)], np.float32)
        self._n_dated = int(valid.sum())
        return tnorm

    def _load_tree_lca(self, cache):
        """Tree buffers over the IN-TREE tips only, plus ``tip_row`` (species-local vocab index of each tip, in the
        same order as the tree's tips) — for ``operator='latent-clade'`` (rule 29). Out-of-tree species (synthetic
        labels absent from the Newick) are filtered out here; the operator covers them by clade cross-attention."""
        nwk = cache / "ca_subtree.dated.nwk"
        if not nwk.exists():
            return None, None
        import re, torch as _t
        from deepearth.encoders.biological.phylogenomic import build_tree_buffers
        toks = set(re.findall(r"[^(),:;\s]+", open(nwk).read()))
        pairs = [(i, tl) for i, tl in enumerate(self._tip_labels) if tl in toks]   # (species-local idx, tip_label)
        if not pairs:
            return None, None
        tree = build_tree_buffers(str(nwk), [tl for _, tl in pairs])
        return tree, _t.tensor([i for i, _ in pairs], dtype=_t.long)

    def _load_tree(self, cache):
        """Parse the dated Newick tree aligned to the model's species (leaves in species order) for message passing;
        returns :func:`build_tree_buffers`' static-topology dict, or ``None`` if absent. Independent of subset/split."""
        nwk = cache / "ca_subtree.dated.nwk"
        if not nwk.exists():
            return None
        from deepearth.encoders.biological.phylogenomic import build_tree_buffers
        try:
            return build_tree_buffers(str(nwk), self._tip_labels)
        except KeyError as e:
            print(f"tree operator unavailable ({e}); ou-attention uses the E1 distance, unaffected", flush=True)
            return None

    def _apply_subset(self, subset, gid, cls, lat, lon, elev, dev):
        """Restrict to a bbox (``{"bbox": [lat0,lat1,lon0,lon1]}``) and/or families (``{"families": [...]}``),
        reindexing every per-observation array (numpy locals, ``self.*`` tensors, ``self.extra``) and resetting
        ``self.n``; returns the reindexed numpy locals the split + modality loader still consume."""
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
        """Align a feature matrix (keyed by its own gbifID ``ids``) to observation order and store it with a presence
        mask. ``zscore`` standardizes per channel; ``normalize`` unit-scales each row; ``valid`` marks real-data rows."""
        rows = np.nan_to_num(rows.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if valid is None:
            valid = np.ones(len(ids), bool)
        order = np.argsort(ids)
        ids, rows, valid = ids[order], rows[order], valid[order]
        pos = np.searchsorted(ids, gid).clip(max=len(ids) - 1)
        have = (ids[pos] == gid) & valid[pos]
        arr = np.zeros((len(gid), rows.shape[1]), np.float32)
        arr[have] = rows[pos[have]]
        if zscore:
            fit = have & getattr(self, "_train_bool", np.ones(len(gid), bool))
            mean, std = arr[fit].mean(0), arr[fit].std(0) + 1e-6
            arr = np.clip((arr - mean) / std, -10, 10)
        if normalize and have.any():
            arr[have] = arr[have] / (np.linalg.norm(arr[have], axis=1, keepdims=True) + 1e-9)
        self.extra[name] = (torch.tensor(arr, device=dev), torch.tensor(have, device=dev), rows.shape[1])

    def _load_modalities(self, cache, gid, dev):
        dm = sorted(glob.glob(str(cache / "gbif_daymet_tokens" / "*.npz")))
        if dm:
            chunks = [np.load(path) for path in dm]
            ids = np.concatenate([chunk["gbifID"] for chunk in chunks])
            rows = np.concatenate([
                chunk["daymet"].reshape(len(chunk["gbifID"]), -1)
                for chunk in chunks
            ])
            self._add_modality("climate", ids, rows, gid, dev, zscore=True)

        nf = sorted(glob.glob(str(cache / "gbif_naip_tokens" / "*.npz")))
        if nf:
            chunks = [np.load(path) for path in nf]
            ids = np.concatenate([chunk["gbifID"] for chunk in chunks])
            for key, name in (("rgb_pool", "naip_rgb"), ("ir_pool", "naip_ir")):
                rows = np.concatenate([chunk[key] for chunk in chunks])
                self._add_modality(name, ids, rows, gid, dev, normalize=True)

        clay = cache / "gbif_clay_tokens.npz"
        if getattr(self, "_clay_v2", False) and (cache / "gbif_clay_v2_tokens.npz").exists():
            clay = cache / "gbif_clay_v2_tokens.npz"
        if clay.exists():
            z = np.load(clay)
            valid = z["has_clay"] if "has_clay" in z else None
            self._add_modality(
                "clay", z["gbifID"], z["clay"], gid, dev,
                normalize=True, valid=valid,
            )

        files = (
            ("soil", "gbif_soil_tokens.npz"),
            ("topo", "gbif_topo_tokens.npz"),
            ("chm", "gbif_chm_tokens.npz"),
            ("hydro", "gbif_hydro_tokens.npz"),
            ("worldclim", "gbif_worldclim_tokens.npz"),
            ("phenology", "gbif_phenology_tokens.npz"),
        )
        for name, filename in files:
            path = cache / filename
            if not path.exists():
                continue
            z = np.load(path)
            self._add_modality(
                name, z["gbifID"], z[name], gid, dev, zscore=True,
                valid=z[f"has_{name}"],
            )

        alphaearth = cache / "gbif_alphaearth_tokens.npz"
        if alphaearth.exists():
            z = np.load(alphaearth)
            values = np.asarray(z["ae"], dtype=np.float32)
            valid = np.isfinite(values[:, 0])
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            self._add_modality(
                "alphaearth", z["gbifID"], values, gid, dev,
                zscore=True, valid=valid,
            )

    @staticmethod
    def _norm_binom(s):
        parts = str(s).split()
        if len(parts) >= 2:
            return f"{parts[0]} {parts[1]}".lower()
        return str(s).strip().lower()

    def _load_pollinator(self, cache, dev, topk=40):
        """Per-species-class marginal pollinator distribution (GloBI): class -> top-k pollinator vocab indices + freqs.
        Bridges the model's species (self.binomial) to the pollinator file's plant_idx by normalized binomial."""
        pf = cache / "gbif_pollinator_dist.npz"
        if not pf.exists():
            return
        z = np.load(pf)
        pidx, pfrq, npoll = z["marg_poll_idx"], z["marg_poll_frq"], z["marg_npoll"]
        pt = cache / "pollinator_taxon_text_emb.npy"
        self.pollinator_text = torch.tensor(np.load(pt).astype(np.float32), device=dev) if pt.exists() else None
        self.n_pollinators = max(int(pidx.max()) + 1, self.pollinator_text.shape[0] if self.pollinator_text is not None else 0)
        b2p = {}
        for r in csv.DictReader(open(cache / "derived/species_index.csv")):
            b2p[self._norm_binom(r["binomial"])] = int(r["idx"])
        am = cache / "pollinator_animal_mask.npy"
        animal = np.load(am) if am.exists() else np.ones(int(pidx.max()) + 1, bool)
        K = min(topk, pidx.shape[1])
        ci = np.zeros((self.n_classes, K), np.int64)
        cf = np.zeros((self.n_classes, K), np.float32)
        cv = np.zeros(self.n_classes, bool)
        for c in range(self.n_classes):
            p = b2p.get(self._norm_binom(self.binomial[c]), -1)
            if 0 <= p < len(npoll) and npoll[p] > 0:
                idx = pidx[p, :K]
                f = pfrq[p, :K].astype(np.float32)
                f[~animal[idx.clip(0, len(animal) - 1)]] = 0.0
                s = f.sum()
                if s > 0:
                    ci[c] = idx
                    cf[c] = f / s
                    cv[c] = True
        self.poll_idx = torch.tensor(ci, device=dev)
        self.poll_frq = torch.tensor(cf, device=dev)
        self.poll_valid = torch.tensor(cv, device=dev)
        print(f"pollinator loaded: {int(cv.sum())}/{self.n_classes} species have GloBI pollinators; vocab {self.n_pollinators}", flush=True)

    def _frame(self, idx):
        lat = self.lat.cpu().numpy()[idx]
        lon = self.lon.cpu().numpy()[idx]
        elev = self.elev.cpu().numpy()[idx]
        t = self.coords[:, 3].cpu().numpy()[idx] * self._time_km if self.time_axis else np.zeros(len(idx), np.float32)
        return np.stack([lat * 111.0, lon * 111.0 * np.cos(np.radians(self.reference_latitude_deg)), elev / 50.0,
                         t], 1)

    def _build_neighbors(self):
        tree = cKDTree(self._frame(self.train))
        _, a = tree.query(self._frame(self.train), k=self.n_neighbors + 4)
        cand = self.train[a]
        is_self = cand == self.train[:, None]
        cand = np.take_along_axis(cand, np.argsort(is_self, axis=1, kind="stable"), axis=1)
        nn_tr = cand[:, : self.n_neighbors]
        _, b = tree.query(self._frame(self.test), k=self.n_neighbors)
        nn_te = self.train[b]
        nbr = np.zeros((self.n, self.n_neighbors), np.int64)
        nbr[self.train] = nn_tr
        nbr[self.test] = nn_te
        self.neighbors = torch.tensor(nbr, device=self.device)

    def variable_dims(self):
        """Widths for the config's variables, filled from the data (vector dims, class counts, trait descriptors)."""
        d = {**self.dims, "identity_classes": self.n_classes, "trait_classes": self.trait_classes}
        for name, (_, _, dim) in self.extra.items():
            d[name] = dim
        return d

    def batch(self, idx):
        """Return one batch: variable values, observed masks, query and neighbor coordinates, the coordinates in
        each vector subspace (here the biological one), and the neighbors' own feature values."""
        ci = self.neighbors[idx]
        values = {"vision_dino": self.dino[idx], "vision_bio": self.bio[idx], "identity": self.cls[idx],
                  "phylo": self.phylo[self.cls[idx]]}
        observed = {n: torch.ones(len(idx), dtype=torch.bool, device=self.device) for n in values}
        observed["vision_dino"] = self.has_vision[idx]
        observed["vision_bio"] = self.has_vision[idx]
        for k, t in enumerate(self.trait_classes):
            values[t] = (self.traits[self.cls[idx], k] - 1).clamp(0)
            observed[t] = self.traits[self.cls[idx], k] > 0
        for name, (vals, have, _) in self.extra.items():
            values[name] = vals[idx]
            observed[name] = have[idx]
        if hasattr(self, "sdist_idx"):                                    # local species distribution (KL/community aux loss)
            values["_sdist_idx"] = self.sdist_idx[idx]
            values["_sdist_frq"] = self.sdist_frq[idx]
        if hasattr(self, "poll_idx"):                                     # per-species pollinator distribution (GloBI aux loss)
            c = self.cls[idx]
            values["_poll_idx"] = self.poll_idx[c]
            values["_poll_frq"] = self.poll_frq[c]
            values["_poll_valid"] = self.poll_valid[c]
        if hasattr(self, "lfmc"):                                         # per-species live fuel moisture (B34 aux head)
            c = self.cls[idx]
            values["_lfmc"] = self.lfmc[c]
            values["_lfmc_valid"] = self.lfmc_valid[c]
        if hasattr(self, "myco"):                                         # per-species mycorrhizal type (B42 symbiosis head)
            c = self.cls[idx]
            values["_myco"] = self.myco[c]
            values["_myco_valid"] = self.myco_valid[c]
        if hasattr(self, "flower"):                                       # per-observation flowering label (B26 phenology head)
            values["_flower"] = self.flower[idx]
            values["_flower_valid"] = self.flower_valid[idx]
        manifold_positions = {"biological": self.phylo[self.cls[ci]]}   # neighbors' known positions only
        neighbor_values = {"identity": self.cls[ci], "vision_dino": self.dino[ci]}
        return values, observed, self.coords[idx], self.coords[ci], manifold_positions, neighbor_values
