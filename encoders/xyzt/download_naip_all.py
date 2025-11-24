#!/usr/bin/env python3
"""
Download all NAIP/CHM data with checkpoint and resume capability

Features:
- Downloads both CHM (elevation) and NAIP (RGB) tar files
- Skips already downloaded files
- Saves progress to checkpoint file
- Can resume after interruption
- Parallel downloads for speed
- Validates file sizes
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


# Base URLs
CHM_BASE_URL = "http://rangeland.ntsg.umt.edu/data/rap/chm-naip/chm/15/"
NAIP_BASE_URL = "http://rangeland.ntsg.umt.edu/data/rap/chm-naip/naip/15/"

# Zone range (covers entire US)
# Based on the server listing, zones range from 15-316 to 15-4xx
MIN_ZONE = 316
MAX_ZONE = 450  # Adjust based on actual availability


class NAIPDownloader:
    def __init__(self, output_dir, checkpoint_file='download_checkpoint.json'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
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
            'in_progress': [],
            'last_updated': None
        }

    def save_checkpoint(self):
        """Save download progress to checkpoint file"""
        self.checkpoint['last_updated'] = datetime.now().isoformat()
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.checkpoint, f, indent=2)

    def is_downloaded(self, filename):
        """Check if file is already downloaded and valid"""
        filepath = self.output_dir / filename
        if not filepath.exists():
            return False

        # Check if file size is reasonable (> 1KB)
        if filepath.stat().st_size < 1024:
            return False

        # Check if in completed list
        if filename in self.checkpoint['completed']:
            return True

        return False

    def download_file(self, url, filename, file_type='CHM'):
        """Download a single file with progress"""
        filepath = self.output_dir / filename

        # Skip if already downloaded
        if self.is_downloaded(filename):
            return {'status': 'skipped', 'filename': filename, 'size': filepath.stat().st_size}

        try:
            # Use wget for robust downloading with resume capability
            cmd = [
                'wget',
                '-c',  # Continue partial downloads
                '-q',  # Quiet mode
                '--show-progress',  # But show progress bar
                '-O', str(filepath),
                url
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

            if result.returncode == 0:
                file_size = filepath.stat().st_size
                if file_size > 1024:  # At least 1KB
                    self.checkpoint['completed'].append(filename)
                    self.save_checkpoint()
                    return {
                        'status': 'success',
                        'filename': filename,
                        'size': file_size,
                        'type': file_type
                    }
                else:
                    filepath.unlink()  # Remove invalid file
                    return {'status': 'failed', 'filename': filename, 'error': 'File too small'}
            else:
                return {'status': 'failed', 'filename': filename, 'error': result.stderr}

        except subprocess.TimeoutExpired:
            return {'status': 'failed', 'filename': filename, 'error': 'Timeout'}
        except Exception as e:
            return {'status': 'failed', 'filename': filename, 'error': str(e)}

    def check_zone_exists(self, zone):
        """Check if zone exists on server"""
        chm_filename = f"15-{zone}.tar"
        chm_url = CHM_BASE_URL + chm_filename

        try:
            response = requests.head(chm_url, timeout=10)
            return response.status_code == 200
        except:
            return False

    def get_available_zones(self, min_zone, max_zone, check_server=True):
        """Get list of available zones"""
        print("Checking available zones on server...")
        available = []

        if not check_server:
            # Return all zones without checking
            return list(range(min_zone, max_zone + 1))

        # Check each zone (can be slow)
        for zone in tqdm(range(min_zone, max_zone + 1), desc="Checking zones"):
            if self.check_zone_exists(zone):
                available.append(zone)

        print(f"Found {len(available)} available zones")
        return available

    def download_zone(self, zone, download_chm=True, download_naip=True):
        """Download both CHM and NAIP files for a zone"""
        results = []

        if download_chm:
            chm_filename = f"15-{zone}.tar"
            chm_url = CHM_BASE_URL + chm_filename
            result = self.download_file(chm_url, chm_filename, 'CHM')
            results.append(result)

        if download_naip:
            naip_filename = f"15-{zone}.tar"
            naip_url = NAIP_BASE_URL + naip_filename
            result = self.download_file(naip_url, naip_filename, 'NAIP')
            results.append(result)

        return results

    def download_all(self, zones, max_workers=4, download_chm=True, download_naip=True):
        """Download all zones with parallel workers"""
        print(f"\nDownloading {len(zones)} zones...")
        print(f"Output directory: {self.output_dir}")
        print(f"Parallel workers: {max_workers}")
        print(f"CHM: {download_chm}, NAIP: {download_naip}")
        print()

        total_files = len(zones) * (int(download_chm) + int(download_naip))
        completed = 0
        skipped = 0
        failed = 0
        total_size = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            future_to_zone = {
                executor.submit(self.download_zone, zone, download_chm, download_naip): zone
                for zone in zones
            }

            # Process completed downloads
            with tqdm(total=total_files, desc="Overall Progress") as pbar:
                for future in as_completed(future_to_zone):
                    zone = future_to_zone[future]
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
                            else:
                                failed += 1
                                self.checkpoint['failed'].append(result['filename'])
                                self.save_checkpoint()

                            pbar.update(1)

                    except Exception as e:
                        print(f"\nError downloading zone {zone}: {e}")
                        failed += 1
                        pbar.update(1)

        # Final summary
        print("\n" + "="*70)
        print("DOWNLOAD SUMMARY")
        print("="*70)
        print(f"Total files: {total_files}")
        print(f"Completed: {completed}")
        print(f"Skipped (already downloaded): {skipped}")
        print(f"Failed: {failed}")
        print(f"Total size: {total_size / 1024**3:.2f} GB")
        print(f"Output directory: {self.output_dir}")
        print("="*70)

        if failed > 0:
            print(f"\n⚠️  {failed} files failed to download")
            print(f"Failed files are logged in: {self.checkpoint_file}")
            print("You can re-run this script to retry failed downloads")


def main():
    parser = argparse.ArgumentParser(
        description='Download NAIP/CHM data with checkpoint and resume',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download zones 316-325 (10 zones)
  python download_naip_all.py --zones 316-325 --output data/naip_all

  # Download specific zones
  python download_naip_all.py --zones 316,320,325,330 --output data/naip_selected

  # Download only CHM (elevation) files
  python download_naip_all.py --zones 316-350 --chm-only --output data/chm_only

  # Download only NAIP (RGB) files
  python download_naip_all.py --zones 316-350 --naip-only --output data/naip_only

  # Resume interrupted download
  python download_naip_all.py --zones 316-400 --output data/naip_all
  (Will skip already downloaded files automatically)

  # Fast download with more workers
  python download_naip_all.py --zones 316-400 --workers 8 --output data/naip_all
        """
    )

    parser.add_argument('--zones', required=True,
                       help='Zones to download (e.g., "316-325" or "316,320,325")')
    parser.add_argument('--output', '-o', default='data/naip_all',
                       help='Output directory (default: data/naip_all)')
    parser.add_argument('--workers', '-w', type=int, default=4,
                       help='Number of parallel download workers (default: 4)')
    parser.add_argument('--chm-only', action='store_true',
                       help='Download only CHM (elevation) files')
    parser.add_argument('--naip-only', action='store_true',
                       help='Download only NAIP (RGB) files')
    parser.add_argument('--check-server', action='store_true',
                       help='Check server for available zones (slow but accurate)')

    args = parser.parse_args()

    # Parse zones
    if '-' in args.zones:
        start, end = map(int, args.zones.split('-'))
        zones = list(range(start, end + 1))
    elif ',' in args.zones:
        zones = [int(z.strip()) for z in args.zones.split(',')]
    else:
        zones = [int(args.zones)]

    # Determine what to download
    download_chm = not args.naip_only
    download_naip = not args.chm_only

    # Create downloader
    downloader = NAIPDownloader(args.output)

    # Filter available zones if requested
    if args.check_server:
        zones = downloader.get_available_zones(min(zones), max(zones), check_server=True)
        if not zones:
            print("No available zones found!")
            return

    # Download
    downloader.download_all(
        zones,
        max_workers=args.workers,
        download_chm=download_chm,
        download_naip=download_naip
    )


if __name__ == '__main__':
    main()
