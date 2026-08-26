"""
Resumable, retrying LAZ tile downloader — USGS TNM LPC products
==================================================================
Fills a real gap found 2026-07-27: build_lidar_pointcloud.py's load_points_in_bbox() only
READS existing .laz files in lidar/data/raw/ — despite earlier notes claiming
that script "downloads 6 LAZ tiles on first run," there is (and was) no download logic
anywhere in this codebase. The original 6 tiles were fetched via some ad hoc, unsaved process.
This script is the actual, permanent, reusable fetch step that was missing.

Built with resume + retry specifically because a plain requests.get(stream=True) already failed
once on a much smaller file (an 8.28GB reference dataset died at 67%/4hrs from a
dropped connection, with no way to resume other than starting over from zero). For a 9.51GB,
25-tile fetch, that failure mode is a real risk, not a hypothetical one:
  - Each tile resumes from its own existing partial-file byte offset via an HTTP Range request,
    rather than restarting the whole tile (let alone the whole batch) from zero.
  - Retries with exponential backoff on any connection error, up to MAX_RETRIES times per tile.
  - Verifies final size against the TNM API's own reported sizeInBytes before considering a
    tile done — catches silent truncation a bare "did the request not raise" check would miss.
  - A 416 (Range Not Satisfiable) response is treated as "already fully downloaded", not an
    error, since that's what a resume-complete server correctly returns.

Usage:
    python3 lidar/download_laz_tiles.py --lat 28.690514 --lon -81.287539 --radius_km 2.99
"""
import os, sys, time, argparse
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
os.makedirs(RAW_DIR, exist_ok=True)

MAX_RETRIES = 6
CHUNK_SIZE = 1024 * 1024  # 1 MB
CONNECT_TIMEOUT = 15
READ_TIMEOUT = 30  # per-chunk stall timeout -- without this a hung connection never triggers
                    # the retry loop, it just sits there indefinitely


def bbox_from_center(lat, lon, radius_km):
    import math
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * math.cos(math.radians(lat)))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def query_tnm_products(bbox_wsen):
    r = requests.get("https://tnmaccess.nationalmap.gov/api/v1/products", params={
        "bbox": ",".join(f"{v:.6f}" for v in bbox_wsen),
        "datasets": "Lidar Point Cloud (LPC)", "prodFormats": "LAZ",
    }, timeout=30)
    r.raise_for_status()
    items = r.json().get("items", [])
    return [{"title": it["title"], "url": it["downloadURL"], "size": it["sizeInBytes"]}
            for it in items]


def download_resumable(url, dest_path, expected_size=None, max_retries=MAX_RETRIES):
    """Download with HTTP-Range resume + exponential-backoff retry. Returns True on success."""
    for attempt in range(1, max_retries + 1):
        existing_size = os.path.getsize(dest_path) if os.path.exists(dest_path) else 0
        if expected_size and existing_size >= expected_size:
            print(f"    already complete ({existing_size/1e6:.1f} MB)")
            return True

        headers = {"Range": f"bytes={existing_size}-"} if existing_size > 0 else {}
        mode = "ab" if existing_size > 0 else "wb"
        if existing_size > 0:
            print(f"    resuming from byte {existing_size:,} ({existing_size/1e6:.1f} MB) "
                  f"[attempt {attempt}/{max_retries}]")
        else:
            print(f"    starting download [attempt {attempt}/{max_retries}]")

        try:
            with requests.get(url, headers=headers, stream=True,
                               timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)) as r:
                if r.status_code == 416:  # Range Not Satisfiable -- server confirms file is
                                            # already fully present, not an error
                    print("    server confirms file already complete (416)")
                    return True
                r.raise_for_status()
                with open(dest_path, mode) as f:
                    downloaded = existing_size
                    last_print = downloaded
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        if downloaded - last_print > 50 * 1024 * 1024:  # print every ~50MB
                            print(f"      {downloaded/1e6:.0f} MB "
                                  + (f"/ {expected_size/1e6:.0f} MB" if expected_size else ""))
                            last_print = downloaded

            final_size = os.path.getsize(dest_path)
            if expected_size and abs(final_size - expected_size) > 4096:
                raise IOError(f"size mismatch after download: got {final_size:,}, "
                               f"expected {expected_size:,}")
            print(f"    done ({final_size/1e6:.1f} MB)")
            return True

        except (requests.exceptions.RequestException, IOError) as e:
            wait = min(2 ** attempt, 60)
            print(f"    attempt {attempt}/{max_retries} failed: {e} -- retrying in {wait}s")
            if attempt < max_retries:
                time.sleep(wait)

    print(f"    FAILED after {max_retries} attempts: {os.path.basename(dest_path)}")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--radius_km", type=float, required=True)
    args = ap.parse_args()

    bbox = bbox_from_center(args.lat, args.lon, args.radius_km)
    print(f"Querying TNM for LPC/LAZ products in bbox {bbox} …")
    items = query_tnm_products(bbox)
    total_gb = sum(it["size"] for it in items) / 1e9
    print(f"{len(items)} tiles found, {total_gb:.2f} GB total\n")

    n_ok, n_fail = 0, 0
    for i, item in enumerate(items, 1):
        fname = os.path.basename(item["url"])
        dest = os.path.join(RAW_DIR, fname)
        print(f"[{i}/{len(items)}] {fname}  ({item['size']/1e6:.1f} MB)")
        ok = download_resumable(item["url"], dest, expected_size=item["size"])
        n_ok += ok
        n_fail += not ok

    print(f"\nDone: {n_ok}/{len(items)} tiles OK, {n_fail} failed")
    if n_fail:
        print("Re-run this exact command to retry only the failed/incomplete tiles "
              "(already-complete ones are skipped instantly via the size check above).")
        sys.exit(1)


if __name__ == "__main__":
    main()
