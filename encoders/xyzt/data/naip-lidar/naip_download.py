#!/usr/bin/env python3
"""
Production NAIP-3DEP data acquisition system

One-stop script for downloading and parsing NAIP+CHM data by geographic location.

Features:
- Auto-downloads and caches 22M chip metadata (files.csv) on first use
- Converts lat/lon to UTM zones automatically
- Downloads only relevant tar files for your location
- Extracts and parses chips into Earth4D format
- Tracks chip IDs for proper train/test splitting

Usage:
    # Download 10x10 region around ASU campus
    python naip_download.py --lat 33.42 --lon -111.93 --grid-size 10 --output data/asu

    # Use named location
    python naip_download.py --location Stanford_CA --grid-size 10 --output data/stanford
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import rasterio
from rasterio.warp import transform as rio_transform

from naip_utils import (
    latlon_to_utm,
    get_download_urls_for_location,
    TARGET_LOCATIONS
)


# Global configuration
METADATA_URL = "http://rangeland.ntsg.umt.edu/data/rap/chm-naip/files.csv"
BASE_URL = "http://rangeland.ntsg.umt.edu/data/rap/chm-naip"
METADATA_CACHE = Path.home() / ".cache" / "naip" / "files.csv"


class NAIPDownloader:
    """Production NAIP downloader with automatic metadata caching"""

    def __init__(self, output_dir: str, cache_dir: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Cache directory for metadata and tar files
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = self.output_dir / ".cache"
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        self.metadata_path = self.cache_dir / "files.csv"
        self.metadata_df = None

    def ensure_metadata(self):
        """Download and cache files.csv if not present"""
        if self.metadata_path.exists():
            print(f"✓ Using cached metadata: {self.metadata_path}")
            print(f"  Size: {self.metadata_path.stat().st_size / 1024**3:.2f} GB")
            return

        print(f"Downloading NAIP metadata index (3.2 GB, ~30-40 min one-time download)...")
        print(f"This will be cached at: {self.metadata_path}")
        print(f"Future runs will use the cached version.\n")

        # Download with resume capability
        cmd = [
            'wget',
            '-c',  # Resume
            '-q', '--show-progress',  # Quiet with progress
            '-O', str(self.metadata_path),
            METADATA_URL
        ]

        try:
            subprocess.run(cmd, check=True, timeout=3600)
            print(f"\n✓ Metadata downloaded and cached")
        except subprocess.TimeoutExpired:
            print("\n⚠ Download timed out. Run again to resume.")
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Download failed: {e}")
            sys.exit(1)

    def load_metadata(self):
        """Load metadata CSV into pandas DataFrame"""
        if self.metadata_df is not None:
            return self.metadata_df

        print(f"Loading metadata index...")
        # CSV columns: chm, naip, utm_zone, x, y, chm_date, naip_date, land_cover, us_l3code, partition
        self.metadata_df = pd.read_csv(self.metadata_path)
        print(f"✓ Loaded {len(self.metadata_df):,} chip records")
        return self.metadata_df

    def find_chips_near_location(
        self,
        lat: float,
        lon: float,
        radius_km: float = 10.0
    ) -> pd.DataFrame:
        """
        Find all chips within radius of a lat/lon location

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            radius_km: Search radius in kilometers

        Returns:
            DataFrame of matching chips with metadata
        """
        # Convert to UTM
        zone, easting, northing = latlon_to_utm(lat, lon)

        # Load metadata
        df = self.load_metadata()

        # Filter by UTM zone
        df_zone = df[df['utm_zone'] == zone].copy()

        if len(df_zone) == 0:
            print(f"⚠ No chips found in UTM zone {zone}")
            return pd.DataFrame()

        # Calculate distance from target point
        radius_m = radius_km * 1000
        df_zone['distance'] = np.sqrt(
            (df_zone['x'] - easting)**2 +
            (df_zone['y'] - northing)**2
        )

        # Filter by radius
        nearby = df_zone[df_zone['distance'] <= radius_m].copy()
        nearby = nearby.sort_values('distance')

        print(f"Found {len(nearby):,} chips within {radius_km} km of ({lat:.4f}, {lon:.4f})")
        print(f"  UTM Zone: {zone}, Easting: {easting:.0f}, Northing: {northing:.0f}")

        return nearby

    def download_and_extract_tar(
        self,
        zone: int,
        tar_filename: str,
        data_type: str  # 'chm' or 'naip'
    ) -> Path:
        """Download and extract a tar file, return extraction directory"""
        url = f"{BASE_URL}/{data_type}/{zone}/{tar_filename}"
        tar_path = self.cache_dir / data_type / str(zone) / tar_filename
        extract_dir = self.output_dir / data_type / str(zone) / tar_filename.replace('.tar', '')

        # Create directories
        tar_path.parent.mkdir(parents=True, exist_ok=True)
        extract_dir.mkdir(parents=True, exist_ok=True)

        # Download if not cached
        if not tar_path.exists():
            print(f"\n  Downloading {data_type}/{zone}/{tar_filename}...")
            cmd = ['wget', '-c', '--show-progress', '-O', str(tar_path), url]
            try:
                subprocess.run(cmd, check=True, timeout=3600)
            except:
                tar_path.unlink(missing_ok=True)
                raise
        else:
            print(f"  ✓ Using cached {data_type}/{zone}/{tar_filename}")

        # Extract if not already extracted
        if not any(extract_dir.iterdir()):
            print(f"  Extracting {tar_filename}...")
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(extract_dir)

        return extract_dir

    def download_for_location(
        self,
        lat: float,
        lon: float,
        max_chips: Optional[int] = None,
        radius_km: float = 10.0
    ) -> List[Dict]:
        """
        Download all data needed for a location

        Returns:
            List of chip metadata dicts with local file paths
        """
        # Find relevant chips
        chips_df = self.find_chips_near_location(lat, lon, radius_km)

        if len(chips_df) == 0:
            print("No chips found!")
            return []

        if max_chips:
            chips_df = chips_df.head(max_chips)
            print(f"Limiting to {max_chips} closest chips")

        # Get unique tar files needed
        tar_files = set()
        for _, row in chips_df.iterrows():
            # CHM path format: chm/zone/y-prefix/zone_x_y_date.tif
            # Tar file format: zone-y-prefix.tar
            chm_path = row['chm']
            path_parts = chm_path.split('/')
            zone = row['utm_zone']
            y_prefix = path_parts[2]  # e.g., '381' from chm/10/381/...
            tar_file = f"{zone}-{y_prefix}.tar"
            tar_files.add((zone, tar_file))

        print(f"\nNeed to download {len(tar_files)} unique tar files")

        # Download and extract all tar files
        extracted_dirs = {}
        for zone, tar_filename in tqdm(list(tar_files), desc="Downloading tars"):
            try:
                chm_dir = self.download_and_extract_tar(zone, tar_filename, 'chm')
                naip_dir = self.download_and_extract_tar(zone, tar_filename, 'naip')
                extracted_dirs[(zone, tar_filename)] = {
                    'chm': chm_dir,
                    'naip': naip_dir
                }
            except Exception as e:
                print(f"  ✗ Failed to download {tar_filename}: {e}")
                continue

        # Build list of chip file paths
        chip_records = []
        for _, row in chips_df.iterrows():
            chm_path_str = row['chm']
            naip_path_str = row['naip']
            path_parts = chm_path_str.split('/')
            zone = row['utm_zone']
            y_prefix = path_parts[2]
            tar_file = f"{zone}-{y_prefix}.tar"
            chm_filename = path_parts[3]
            naip_filename = naip_path_str.split('/')[-1]

            if (zone, tar_file) not in extracted_dirs:
                continue

            chm_path = extracted_dirs[(zone, tar_file)]['chm'] / chm_filename
            naip_path = extracted_dirs[(zone, tar_file)]['naip'] / naip_filename

            if chm_path.exists() and naip_path.exists():
                chip_records.append({
                    'chip_id': chm_filename.replace('.tif', ''),
                    'chm_path': str(chm_path),
                    'naip_path': str(naip_path),
                    'utm_zone': zone,
                    'x': row['x'],
                    'y': row['y'],
                    'chm_date': row['chm_date'],
                    'naip_date': row['naip_date'],
                    'land_cover': row['land_cover'],
                    'ecoregion': row['us_l3code'],
                    'distance': row['distance'],
                    'partition': row['partition']
                })

        print(f"\n✓ Successfully located {len(chip_records)} chips")
        return chip_records


def parse_chip(chm_path: str, naip_path: str) -> torch.Tensor:
    """
    Parse a CHM+NAIP pair into (N, 7) tensor: [lat, lon, elev, time, r, g, b]

    Note: CHM values are scaled by 100 in the file, so divide by 100 to get meters
    """
    # Read CHM (elevation)
    with rasterio.open(chm_path) as src:
        elevation = src.read(1).astype(np.float32) / 100.0  # Scale to meters
        chm_transform = src.transform
        chm_crs = src.crs
        height, width = elevation.shape

    # Read NAIP (RGB)
    with rasterio.open(naip_path) as src:
        naip_data = src.read()  # 5 bands: R, G, B, NIR, mask
        r_band = naip_data[0].astype(np.float32) / 255.0
        g_band = naip_data[1].astype(np.float32) / 255.0
        b_band = naip_data[2].astype(np.float32) / 255.0

        # Resample to match CHM if needed
        if r_band.shape != elevation.shape:
            from rasterio.warp import reproject, Resampling
            r_resampled = np.zeros(elevation.shape, dtype=np.float32)
            g_resampled = np.zeros(elevation.shape, dtype=np.float32)
            b_resampled = np.zeros(elevation.shape, dtype=np.float32)

            for band, resampled in [(r_band, r_resampled), (g_band, g_resampled), (b_band, b_resampled)]:
                reproject(
                    source=band,
                    destination=resampled,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=chm_transform,
                    dst_crs=chm_crs,
                    resampling=Resampling.bilinear
                )
            r_band, g_band, b_band = r_resampled, g_resampled, b_resampled

    # Generate lat/lon grid
    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    xs, ys = chm_transform * (cols.flatten() + 0.5, rows.flatten() + 0.5)

    # Convert to WGS84 lat/lon
    lons, lats = rio_transform(chm_crs, 'EPSG:4326', xs, ys)

    # Extract timestamp from filename (format: zone_x_y_YYYY-MM-DD.tif)
    date_str = Path(chm_path).stem.split('_')[-1]
    timestamp = datetime.strptime(date_str, '%Y-%m-%d').timestamp()

    # Stack into (N, 7)
    xyztrgb = np.stack([
        np.array(lats).flatten().astype(np.float32),
        np.array(lons).flatten().astype(np.float32),
        elevation.flatten(),
        np.full(height * width, timestamp, dtype=np.float32),
        r_band.flatten(),
        g_band.flatten(),
        b_band.flatten(),
    ], axis=1)

    return torch.from_numpy(xyztrgb)


def main():
    parser = argparse.ArgumentParser(
        description='Production NAIP data downloader',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download data near ASU campus
  python naip_download.py --lat 33.42 --lon -111.93 --max-chips 100 --output data/asu

  # Use named location
  python naip_download.py --location Stanford_CA --radius 5 --output data/stanford

  # Download and parse in one step
  python naip_download.py --location Boulder_CO --max-chips 200 --parse --output data/boulder

Available locations: ASU_Tempe, Boulder_CO, NYC_HighLine, Stanford_CA, Miami_SouthBeach
        """
    )

    location_group = parser.add_mutually_exclusive_group(required=True)
    location_group.add_argument('--location', choices=list(TARGET_LOCATIONS.keys()))
    location_group.add_argument('--lat', type=float)

    parser.add_argument('--lon', type=float, help='Longitude (required with --lat)')
    parser.add_argument('--radius', type=float, default=10.0,
                       help='Search radius in km (default: 10)')
    parser.add_argument('--max-chips', type=int,
                       help='Max number of chips to download')
    parser.add_argument('--output', '-o', required=True,
                       help='Output directory')
    parser.add_argument('--parse', action='store_true',
                       help='Parse chips into .pt file after download')
    parser.add_argument('--cache-dir',
                       help='Cache directory for metadata and tars')

    args = parser.parse_args()

    # Validate lat/lon
    if args.lat is not None and args.lon is None:
        parser.error("--lat requires --lon")
    if args.lon is not None and args.lat is None:
        parser.error("--lon requires --lat")

    # Get location
    if args.location:
        lat, lon = TARGET_LOCATIONS[args.location]
        location_name = args.location
    else:
        lat, lon = args.lat, args.lon
        location_name = f"{lat:.4f}_{lon:.4f}"

    print(f"{'='*70}")
    print(f"NAIP Data Download: {location_name}")
    print(f"Location: ({lat:.6f}, {lon:.6f})")
    print(f"Search radius: {args.radius} km")
    print(f"{'='*70}\n")

    # Initialize downloader
    downloader = NAIPDownloader(args.output, args.cache_dir)

    # Ensure metadata is available (auto-download if needed)
    downloader.ensure_metadata()

    # Download chips
    chip_records = downloader.download_for_location(
        lat, lon,
        max_chips=args.max_chips,
        radius_km=args.radius
    )

    if len(chip_records) == 0:
        print("No chips downloaded. Exiting.")
        return

    # Save chip metadata
    metadata_path = Path(args.output) / 'chip_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(chip_records, f, indent=2)
    print(f"\n✓ Chip metadata saved: {metadata_path}")

    # Parse if requested
    if args.parse:
        print(f"\nParsing {len(chip_records)} chips...")
        all_tensors = []
        chip_ids = []
        chip_sizes = []

        for record in tqdm(chip_records, desc="Parsing chips"):
            try:
                tensor = parse_chip(record['chm_path'], record['naip_path'])
                all_tensors.append(tensor)
                chip_ids.append(record['chip_id'])
                chip_sizes.append(len(tensor))
            except Exception as e:
                print(f"  ✗ Failed to parse {record['chip_id']}: {e}")
                continue

        if len(all_tensors) == 0:
            print("No chips successfully parsed!")
            return

        # Concatenate all data
        data = torch.cat(all_tensors, dim=0)

        # Save parsed data with metadata for chip-based splitting
        output_path = Path(args.output) / 'parsed_xyztrgb.pt'
        torch.save({
            'data': data,
            'chip_ids': chip_ids,
            'chip_sizes': chip_sizes,  # Number of points per chip
            'chip_metadata': chip_records,
            'columns': ['lat', 'lon', 'elevation', 'timestamp', 'r', 'g', 'b'],
            'n_chips': len(chip_ids),
            'n_points': len(data),
        }, output_path)

        print(f"\n{'='*70}")
        print(f"✓ Parsing complete!")
        print(f"  Output: {output_path}")
        print(f"  Chips: {len(chip_ids)}")
        print(f"  Points: {len(data):,}")
        print(f"  Size: {output_path.stat().st_size / 1024**2:.1f} MB")
        print(f"{'='*70}")


if __name__ == '__main__':
    main()
