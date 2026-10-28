"""Where every tower file comes from, how it is fetched, and the provenance record kept for it.

Sources (all CC-BY-4.0):
- FLUXNET Shuttle, ONEFlux v1.3 products of AmeriFlux, ICOS (with the European Fluxes Database, ChinaFlux,
  JapanFlux, KoFlux, SAEON) and TERN. Listed and fetched with the official `fluxnet-shuttle` library
  (github.com/fluxnet/shuttle); AmeriFlux files on ftp.fluxdata.org, ICOS files through the Carbon Portal licence
  acceptance, TERN files on its public THREDDS server.
- FLUXNET-CH4 V1.0 (Delwiche et al. 2021) and AmeriFlux BASE-BADM, through the AmeriFlux data-download API with a
  registered account.
- ICOS ETC L2 Fluxes, through the Carbon Portal licence acceptance (data.icos-cp.eu/licence_accept).

Every downloaded file gets a provenance row: source, site, URL, version, download time (UTC), size and SHA-256. The
raw archive may be deleted after processing; the row stays, so the file can be fetched and verified again.
"""

import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

AMF_API = "https://amfcdn.lbl.gov/api/v1"
AMF_USER = {"user_id": os.environ.get("AMERIFLUX_USER_ID", ""), "user_email": os.environ.get("AMERIFLUX_USER_EMAIL", "")}
"""A registered AmeriFlux account (environment variables AMERIFLUX_USER_ID and AMERIFLUX_USER_EMAIL); the API records
it with every download."""
BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
"""ftp.fluxdata.org answers 403 to clients without a browser User-Agent."""
ICOS_LICENCE = "https://data.icos-cp.eu/licence_accept?ids=%5B%22{id}%22%5D"

SHUTTLE, FLUXNET_CH4, AMF_BASE, ICOS_L2 = "shuttle", "fluxnet_ch4", "amf_base", "icos_l2"
SOURCE_CODES = {SHUTTLE: 1, FLUXNET_CH4: 2, AMF_BASE: 3, ICOS_L2: 4}
"""The per-step `<VAR>_src` code of the stored tables; 0 means no measured value."""
SOURCE_LABELS = {
    SHUTTLE: "FLUXNET Shuttle, ONEFlux v1.3 FLUXMET HH",
    FLUXNET_CH4: "FLUXNET-CH4 Community Product V1.0 HH (Delwiche et al. 2021)",
    AMF_BASE: "AmeriFlux BASE (site-submitted, AmeriFlux QA/QC, not gap-filled)",
    ICOS_L2: "ICOS ETC L2 Fluxes",
}


@dataclass
class Provenance:
    source: str
    site_id: str
    file_name: str
    url: str
    version: str
    downloaded_utc: str
    bytes: int
    sha256: str
    kept: bool
    """Whether the raw archive is still on disk under the raw directory."""


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def curl(url: str, dest: Path, cookies: Optional[Path] = None, retries: int = 3) -> None:
    """Fetch with curl and a browser User-Agent (python urllib gets 403 from ftp.fluxdata.org), following
    redirects; a cookie jar carries the ICOS licence acceptance."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["curl", "-sS", "-L", "--fail", "--http1.1", "-A", BROWSER_UA, "-o", str(dest), "--retry", str(retries),
           "--retry-delay", "5", "--max-time", "3600"]
    if cookies is not None:
        cmd += ["-c", str(cookies), "-b", str(cookies)]
    subprocess.run(cmd + [url], check=True)


def amf_download_urls(site_ids: Iterable[str], product: str, variant: Optional[str] = None,
                      intended_use: str = "other_research") -> Dict[str, str]:
    """{site_id: url} from the AmeriFlux data-download API. `product` is "FLUXNET-CH4" (no variant) or
    "BASE-BADM" (variant "FULLSET"). The request records the CC-BY-4.0 policy agreement."""
    body = {**AMF_USER, "data_product": product, "data_policy": "CCBY4.0", "site_ids": list(site_ids),
            "intended_use": intended_use, "description": "FluxFill tower dataset (UC Berkeley)",
            "agree_policy": True}
    if variant:
        body["data_variant"] = variant
    out = subprocess.run(["curl", "-sS", "--fail", "-A", BROWSER_UA, "-H", "Content-Type: application/json",
                          "-X", "POST", "-d", json.dumps(body), f"{AMF_API}/data_download"],
                         check=True, capture_output=True, text=True).stdout
    res = json.loads(out)
    urls = {}
    for f in res.get("data_urls", []):
        urls[f["site_id"]] = f["url"]
    return urls


def amf_availability(network: str, product: str) -> List[Dict]:
    """GET data_availability/<AmeriFlux|FLUXNET>/<product>/CCBY4.0: [{SITE_ID, URL, publish_years}]."""
    out = subprocess.run(["curl", "-sS", "--fail", "-A", BROWSER_UA,
                          f"{AMF_API}/data_availability/{network}/{product}/CCBY4.0"],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def shuttle_download(snapshot_csv: Path, site_id: str, out_dir: Path) -> Path:
    """One site's ONEFlux zip through the official FLUXNET Shuttle library (fluxnet_shuttle.download)."""
    from fluxnet_shuttle import download  # imported here: only the download step needs the library
    out_dir.mkdir(parents=True, exist_ok=True)
    files = download(site_ids=[site_id], snapshot_file=str(snapshot_csv), output_dir=str(out_dir))
    if not files:
        raise RuntimeError(f"shuttle download returned no file for {site_id}")
    return Path(files[0])


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def record(path: Path, source: str, site_id: str, url: str, version: str, kept: bool,
           downloaded_utc: Optional[str] = None) -> Provenance:
    return Provenance(source, site_id, path.name, url, version, downloaded_utc or utc_now(),
                      os.path.getsize(path), sha256(path), kept)


def append_provenance(table: Path, rows: Iterable[Provenance]) -> None:
    """Append rows to a JSON-lines provenance log (one object per file)."""
    table.parent.mkdir(parents=True, exist_ok=True)
    with open(table, "a") as f:
        for r in rows:
            f.write(json.dumps(asdict(r)) + "\n")


def read_provenance(table: Path) -> List[Dict]:
    if not table.exists():
        return []
    with open(table) as f:
        return [json.loads(line) for line in f if line.strip()]
