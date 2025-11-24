#!/usr/bin/env python3
"""
Download NAIP/CHM data for specific geographic locations

Features:
- Convert lat/lon to UTM zones automatically
- Download data for single location or region (NxN grid)
- Support for all 5 target locations
- Checkpoint and resume capability
- Parallel downloads
"""

import os
import sys
import json
import argparse
from pathlib import Path
import subprocess
from datetime import datetime
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from naip_utils import (
    get_download_urls_for_location,
    get_unique_tar_files_for_region,
    TARGET_LOCATIONS
)


class LocationBasedDownloader:
    def __init__(self, output_dir, checkpoint_file='download_checkpoint.json'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Create subdirectories for CHM and NAIP
        self.chm_dir = self.output_dir / 'chm'
        self.naip_dir = self.output_dir / 'naip'
        self.chm_dir.mkdir(exist_ok=True, parents=True)
        self.naip_dir.mkdir(exist_ok=True, parents=True)

        self.checkpoint_file = self.output_dir / checkpoint_file
        self.checkpoint = self.load_checkpoint()

    def load_checkpoint(self):
        """Load download progress from checkpoint file"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {
            'completed': [],
            'failed': [],
            'last_updated': None
        }

    def save_checkpoint(self):
        """Save download progress to checkpoint file"""
        self.checkpoint['last_updated'] = datetime.now().isoformat()
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.checkpoint, f, indent=2)

    def is_downloaded(self, filepath):
        """Check if file is already downloaded and valid"""
        if not filepath.exists():
            return False

        # Check if file size is reasonable (> 1KB)
        if filepath.stat().st_size < 1024:
            return False

        # Check if in completed list
        if str(filepath) in self.checkpoint['completed']:
            return True

        return False

    def download_file(self, url, output_path, file_type='CHM'):
        """Download a single file with progress"""
        output_path = Path(output_path)

        # Skip if already downloaded
        if self.is_downloaded(output_path):
            return {
                'status': 'skipped',
                'filename': output_path.name,
                'size': output_path.stat().st_size,
                'type': file_type
            }

        try:
            # Use wget for robust downloading with resume capability
            cmd = [
                'wget',
                '-c',  # Continue partial downloads
                '-q',  # Quiet mode
                '--show-progress',  # But show progress bar
                '-O', str(output_path),
                url
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

            if result.returncode == 0:
                file_size = output_path.stat().st_size
                if file_size > 1024:  # At least 1KB
                    self.checkpoint['completed'].append(str(output_path))
                    self.save_checkpoint()
                    return {
                        'status': 'success',
                        'filename': output_path.name,
                        'size': file_size,
                        'type': file_type
                    }
                else:
                    output_path.unlink()  # Remove invalid file
                    return {'status': 'failed', 'filename': output_path.name, 'error': 'File too small'}
            else:
                # Check if it's a 404 error (file doesn't exist on server)
                if '404' in result.stderr or 'Not Found' in result.stderr:
                    return {'status': 'not_found', 'filename': output_path.name, 'error': '404 Not Found'}
                return {'status': 'failed', 'filename': output_path.name, 'error': result.stderr}

        except subprocess.TimeoutExpired:
            return {'status': 'failed', 'filename': output_path.name, 'error': 'Timeout'}
        except Exception as e:
            return {'status': 'failed', 'filename': output_path.name, 'error': str(e)}

    def check_url_exists(self, url):
        """Check if URL exists on server"""
        try:
            response = requests.head(url, timeout=10)
            return response.status_code == 200
        except:
            return False

    def download_tar_file(self, tar_filename, zone, download_chm=True, download_naip=True):
        """Download CHM and/or NAIP tar file for a specific zone"""
        base_url = "http://rangeland.ntsg.umt.edu/data/rap/chm-naip"
        results = []

        if download_chm:
            chm_url = f"{base_url}/chm/{zone}/{tar_filename}"
            zone_dir = self.chm_dir / str(zone)
            zone_dir.mkdir(exist_ok=True, parents=True)
            output_path = zone_dir / tar_filename
            result = self.download_file(chm_url, output_path, 'CHM')
            results.append(result)

        if download_naip:
            naip_url = f"{base_url}/naip/{zone}/{tar_filename}"
            zone_dir = self.naip_dir / str(zone)
            zone_dir.mkdir(exist_ok=True, parents=True)
            output_path = zone_dir / tar_filename
            result = self.download_file(naip_url, output_path, 'NAIP')
            results.append(result)

        return results

    def download_for_location(
        self,
        lat,
        lon,
        grid_size=1,
        spacing_meters=1000.0,
        download_chm=True,
        download_naip=True,
        max_workers=4
    ):
        """
        Download data for a location or region

        Args:
            lat: Center latitude
            lon: Center longitude
            grid_size: Grid size (1 for single point, 10 for 10x10 region)
            spacing_meters: Spacing between grid points
            download_chm: Download CHM files
            download_naip: Download NAIP files
            max_workers: Parallel download workers
        """
        print(f"\n{'='*70}")
        print(f"Downloading data for location: ({lat:.6f}, {lon:.6f})")
        print(f"Grid size: {grid_size}x{grid_size}")
        print(f"Spacing: {spacing_meters}m")
        print(f"{'='*70}\n")

        # Get unique tar files needed for this region
        tar_files = get_unique_tar_files_for_region(lat, lon, grid_size, spacing_meters)

        print(f"Identified {len(tar_files)} unique tar files to download")
        for tar_name, chm_url, naip_url, zone in tar_files:
            print(f"  Zone {zone}: {tar_name}")
        print()

        # Download files
        total_files = len(tar_files) * (int(download_chm) + int(download_naip))
        completed = 0
        skipped = 0
        failed = 0
        not_found = 0
        total_size = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            future_to_tar = {
                executor.submit(
                    self.download_tar_file,
                    tar_name,
                    zone,
                    download_chm,
                    download_naip
                ): (tar_name, zone)
                for tar_name, chm_url, naip_url, zone in tar_files
            }

            # Process completed downloads
            with tqdm(total=total_files, desc="Downloading") as pbar:
                for future in as_completed(future_to_tar):
                    tar_name, zone = future_to_tar[future]
                    try:
                        results = future.result()
                        for result in results:
                            if result['status'] == 'success':
                                completed += 1
                                total_size += result['size']
                                pbar.set_postfix({
                                    'Zone': zone,
                                    'Size': f"{total_size / 1024**3:.2f} GB"
                                })
                            elif result['status'] == 'skipped':
                                skipped += 1
                                total_size += result['size']
                            elif result['status'] == 'not_found':
                                not_found += 1
                            else:
                                failed += 1
                                self.checkpoint['failed'].append(result['filename'])
                                self.save_checkpoint()

                            pbar.update(1)

                    except Exception as e:
                        print(f"\nError downloading {tar_name}: {e}")
                        failed += 1
                        pbar.update(1)

        # Summary
        print("\n" + "="*70)
        print("DOWNLOAD SUMMARY")
        print("="*70)
        print(f"Total files attempted: {total_files}")
        print(f"Successfully downloaded: {completed}")
        print(f"Skipped (already exists): {skipped}")
        print(f"Not found on server: {not_found}")
        print(f"Failed: {failed}")
        print(f"Total size: {total_size / 1024**3:.2f} GB")
        print(f"Output directory: {self.output_dir}")
        print("="*70)

        return {
            'completed': completed,
            'skipped': skipped,
            'not_found': not_found,
            'failed': failed,
            'total_size': total_size
        }


def main():
    parser = argparse.ArgumentParser(
        description='Download NAIP/CHM data for specific geographic locations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download data for ASU campus (single point)
  python download_naip_by_location.py --location ASU_Tempe --output data/asu

  # Download 10x10 region around Boulder
  python download_naip_by_location.py --location Boulder_CO --grid-size 10 --output data/boulder

  # Download for custom lat/lon
  python download_naip_by_location.py --lat 33.42 --lon -111.93 --grid-size 10 --output data/custom

  # Download all 5 target locations
  python download_naip_by_location.py --all-targets --grid-size 10 --output data/all_targets

Available named locations:
  ASU_Tempe, Boulder_CO, NYC_HighLine, Stanford_CA, Miami_SouthBeach
        """
    )

    location_group = parser.add_mutually_exclusive_group(required=True)
    location_group.add_argument('--location', choices=list(TARGET_LOCATIONS.keys()),
                               help='Named location to download')
    location_group.add_argument('--lat', type=float,
                               help='Latitude (use with --lon)')
    location_group.add_argument('--all-targets', action='store_true',
                               help='Download all 5 target locations')

    parser.add_argument('--lon', type=float,
                       help='Longitude (use with --lat)')
    parser.add_argument('--grid-size', type=int, default=1,
                       help='Grid size (1=single point, 10=10x10 region, etc)')
    parser.add_argument('--spacing', type=float, default=1000.0,
                       help='Grid spacing in meters (default: 1000)')
    parser.add_argument('--output', '-o', required=True,
                       help='Output directory')
    parser.add_argument('--workers', '-w', type=int, default=4,
                       help='Number of parallel download workers (default: 4)')
    parser.add_argument('--chm-only', action='store_true',
                       help='Download only CHM (elevation) files')
    parser.add_argument('--naip-only', action='store_true',
                       help='Download only NAIP (RGB) files')

    args = parser.parse_args()

    # Validate lat/lon pair
    if args.lat is not None and args.lon is None:
        parser.error("--lat requires --lon")
    if args.lon is not None and args.lat is None:
        parser.error("--lon requires --lat")

    # Determine what to download
    download_chm = not args.naip_only
    download_naip = not args.chm_only

    # Get locations to download
    locations = []
    if args.all_targets:
        locations = [(name, lat, lon) for name, (lat, lon) in TARGET_LOCATIONS.items()]
    elif args.location:
        lat, lon = TARGET_LOCATIONS[args.location]
        locations = [(args.location, lat, lon)]
    else:
        locations = [('custom', args.lat, args.lon)]

    # Download for each location
    for name, lat, lon in locations:
        print(f"\n{'='*70}")
        print(f"Processing: {name}")
        print(f"{'='*70}")

        # Create location-specific output directory if multiple locations
        if len(locations) > 1:
            output_dir = Path(args.output) / name
        else:
            output_dir = Path(args.output)

        downloader = LocationBasedDownloader(output_dir)
        downloader.download_for_location(
            lat=lat,
            lon=lon,
            grid_size=args.grid_size,
            spacing_meters=args.spacing,
            download_chm=download_chm,
            download_naip=download_naip,
            max_workers=args.workers
        )


if __name__ == '__main__':
    main()
