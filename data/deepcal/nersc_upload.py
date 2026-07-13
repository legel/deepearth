"""Publish deepcal_data.zip to NERSC CFS (served at portal.nersc.gov/cfs/m5239/deepcal/) via SFAPI, in resumable
150 MB chunks. Build the zip first with package_dataset.py; prepare.py then auto-downloads it on any machine.

  python -m deepearth.data.deepcal.package_dataset --out $DEEPCAL_ZIP   # build (post-recompute)
  python -m deepearth.data.deepcal.nersc_upload                         # publish

Credentials (SFAPI, from iris.nersc.gov) are read from $SUPERFACILITY_DIR (default ~/.superfacility):
clientid.txt (the current client id) + the private key ($SFAPI_KEY, default legel.pem). They expire — if auth
fails, create a fresh client and update BOTH files. Paths/creds are env-configurable so this is machine-portable.
"""
import io, os, subprocess, time
from pathlib import Path

SF = Path(os.environ.get("SUPERFACILITY_DIR", str(Path.home() / ".superfacility")))
ZIP = Path(os.environ.get("DEEPCAL_ZIP", "/home/photon/4tb/deepcal_data.zip"))
PARTS = ZIP.parent / "parts"
CHUNK = 150 * 1024 * 1024                       # 150 MB parts (under the SFAPI per-file limit)
REMOTE = os.environ.get("DEEPCAL_REMOTE", "/global/cfs/cdirs/m5239/www/deepcal")
KEY = os.environ.get("SFAPI_KEY", "legel.pem")


def main():
    from sfapi_client import Client
    if not ZIP.exists():
        raise SystemExit(f"{ZIP} not found — build it first: python -m deepearth.data.deepcal.package_dataset --out {ZIP}")
    c = Client((SF / "clientid.txt").read_text().strip(), (SF / KEY).read_text())
    pm = c.compute("perlmutter")
    pm.run(f"mkdir -p {REMOTE}/parts")
    PARTS.mkdir(exist_ok=True)
    if not any(PARTS.glob("part_*")):
        print("splitting zip..."); subprocess.run(["split", "-b", str(CHUNK), str(ZIP), str(PARTS / "part_")], check=True)
    parts = sorted(PARTS.glob("part_*"))
    print(f"{len(parts)} parts, {ZIP.stat().st_size / 1e9:.2f} GB total")
    existing = {l.split()[-1] for l in pm.run(f"ls {REMOTE}/parts 2>/dev/null").splitlines() if l}
    [d] = pm.ls(f"{REMOTE}/parts", directory=True)
    for i, p in enumerate(parts):
        if p.name in existing:
            print(f"[{i+1}/{len(parts)}] {p.name} already present"); continue
        for attempt in range(4):                                       # resumable: skips uploaded parts, retries transient failures
            try:
                b = io.BytesIO(p.read_bytes()); b.filename = p.name
                d.upload(b); print(f"[{i+1}/{len(parts)}] {p.name} uploaded"); break
            except Exception as e:
                print(f"[{i+1}/{len(parts)}] {p.name} attempt {attempt+1} failed: {repr(e)[:120]}"); time.sleep(3)
        else:
            print(f"ABORT: {p.name} failed 4x"); return
    print("reassembling on NERSC...")
    print(pm.run(f"cd {REMOTE} && cat parts/part_* > deepcal_data.zip && rm -rf parts && ls -la deepcal_data.zip"))
    local = ZIP.stat().st_size
    remote = int(pm.run(f"stat -c%s {REMOTE}/deepcal_data.zip").strip())
    print(f"local {local} vs remote {remote} -> {'MATCH' if local == remote else 'MISMATCH'}")
    subprocess.run(["rm", "-rf", str(PARTS)])
    print("URL: https://portal.nersc.gov/cfs/m5239/deepcal/deepcal_data.zip")


if __name__ == "__main__":
    main()
