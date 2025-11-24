#!/usr/bin/env python3
"""
Parse paired CHM (elevation) + NAIP (RGB) GeoTIFF files into Earth4D format

The dataset provides two types of files:
- CHM (Canopy Height Model): Single-band elevation data (256x256)
- NAIP (RGB imagery): 5-band imagery (R,G,B,NIR,mask) (427x427)

We need to:
1. Match CHM and NAIP files by coordinates
2. Resample NAIP to match CHM resolution (256x256)
3. Extract (x,y,z,t,r,g,b) for each pixel
"""

import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
import os
import re


def parse_filename_coords(filename: str) -> Tuple[str, str, str, str]:
    """
    Parse NAIP/CHM filename to extract coordinates and dates

    CHM format: 15_XXXXXX_YYYYYYY_DATE.tif
    NAIP format: 15_XXXXXX_YYYYYYY_DATE1_DATE2.tif

    Returns: (level, x_coord, y_coord, date1, date2_optional)
    """
    basename = os.path.basename(filename).replace('.tif', '')
    parts = basename.split('_')

    if len(parts) == 4:
        # CHM file
        return parts[0], parts[1], parts[2], parts[3], None
    elif len(parts) == 5:
        # NAIP file
        return parts[0], parts[1], parts[2], parts[3], parts[4]
    else:
        raise ValueError(f"Unexpected filename format: {filename}")


def find_matching_naip(chm_path: str, naip_dir: str) -> Optional[str]:
    """
    Find matching NAIP RGB file for a CHM elevation file

    Args:
        chm_path: Path to CHM file (e.g., "15_206698_3166153_2019-01-25.tif")
        naip_dir: Directory containing NAIP files

    Returns:
        Path to matching NAIP file, or None if not found
    """
    level, x_coord, y_coord, date1, _ = parse_filename_coords(chm_path)

    # Look for NAIP file with same coordinates
    naip_dir = Path(naip_dir)
    pattern = f"{level}_{x_coord}_{y_coord}_{date1}_*.tif"

    matches = list(naip_dir.glob(pattern))

    if len(matches) > 0:
        return str(matches[0])
    else:
        return None


def parse_paired_geotiffs(
    chm_path: str,
    naip_path: str,
    normalize_rgb: bool = True,
    verbose: bool = False
) -> Tuple[torch.Tensor, Dict]:
    """
    Parse paired CHM (elevation) and NAIP (RGB) files into Earth4D format

    Args:
        chm_path: Path to CHM elevation file (256x256, single-band)
        naip_path: Path to NAIP RGB file (427x427, 5-band)
        normalize_rgb: Normalize RGB to [0,1]
        verbose: Print progress

    Returns:
        Tuple of:
            - tensor: Shape (N, 7) with [lat, lon, elevation, timestamp, r, g, b]
            - metadata: Dictionary with file information
    """
    if verbose:
        print(f"Parsing CHM: {os.path.basename(chm_path)}")
        print(f"      NAIP: {os.path.basename(naip_path)}")

    # Read CHM elevation data
    with rasterio.open(chm_path) as chm_src:
        elevation = chm_src.read(1)  # Single band
        chm_transform = chm_src.transform
        chm_crs = chm_src.crs
        chm_shape = elevation.shape

    # Read NAIP RGB data
    with rasterio.open(naip_path) as naip_src:
        naip_data = naip_src.read()  # All bands (5: R,G,B,NIR,mask)
        naip_crs = naip_src.crs

        # Extract RGB bands (typically bands 1-3)
        r_band = naip_data[0]
        g_band = naip_data[1]
        b_band = naip_data[2]

        # Resample NAIP RGB to match CHM resolution (427x427 → 256x256)
        r_resampled = np.zeros(chm_shape, dtype=r_band.dtype)
        g_resampled = np.zeros(chm_shape, dtype=g_band.dtype)
        b_resampled = np.zeros(chm_shape, dtype=b_band.dtype)

        reproject(
            source=r_band,
            destination=r_resampled,
            src_transform=naip_src.transform,
            src_crs=naip_crs,
            dst_transform=chm_transform,
            dst_crs=chm_crs,
            resampling=Resampling.bilinear
        )

        reproject(
            source=g_band,
            destination=g_resampled,
            src_transform=naip_src.transform,
            src_crs=naip_crs,
            dst_transform=chm_transform,
            dst_crs=chm_crs,
            resampling=Resampling.bilinear
        )

        reproject(
            source=b_band,
            destination=b_resampled,
            src_transform=naip_src.transform,
            src_crs=naip_crs,
            dst_transform=chm_transform,
            dst_crs=chm_crs,
            resampling=Resampling.bilinear
        )

    # Generate lat/lon grid for each pixel
    height, width = chm_shape
    rows, cols = np.meshgrid(
        np.arange(height) + 0.5,
        np.arange(width) + 0.5,
        indexing='ij'
    )

    # Convert pixel coordinates to geographic coordinates
    from rasterio.transform import xy as transform_xy
    lons_list, lats_list = transform_xy(chm_transform, rows.flatten(), cols.flatten())
    lons = np.array(lons_list).flatten()
    lats = np.array(lats_list).flatten()

    # Convert to WGS84 if needed
    if chm_crs.to_epsg() != 4326:
        from rasterio.warp import transform
        lons_transformed, lats_transformed = transform(chm_crs, 'EPSG:4326', lons, lats)
        lons = np.array(lons_transformed).flatten()
        lats = np.array(lats_transformed).flatten()

    # Extract timestamp from CHM filename
    _, _, _, date_str, _ = parse_filename_coords(chm_path)
    timestamp = datetime.strptime(date_str, '%Y-%m-%d').timestamp()

    # Flatten all arrays
    elevation_flat = elevation.flatten().astype(np.float32)
    r_flat = r_resampled.flatten().astype(np.float32)
    g_flat = g_resampled.flatten().astype(np.float32)
    b_flat = b_resampled.flatten().astype(np.float32)

    # Normalize RGB if requested
    if normalize_rgb:
        r_flat = r_flat / 255.0
        g_flat = g_flat / 255.0
        b_flat = b_flat / 255.0

    # Create timestamp array
    t = np.full(len(lats), timestamp, dtype=np.float32)

    # Stack into (N, 7) array
    xyztrgb = np.stack([
        lats.astype(np.float32),  # x = latitude
        lons.astype(np.float32),  # y = longitude
        elevation_flat,           # z = elevation
        t,                        # t = timestamp
        r_flat,                   # r
        g_flat,                   # g
        b_flat,                   # b
    ], axis=1)

    # Convert to PyTorch tensor
    tensor = torch.from_numpy(xyztrgb)

    # Metadata
    metadata = {
        'chm_file': os.path.basename(chm_path),
        'naip_file': os.path.basename(naip_path),
        'n_points': len(tensor),
        'timestamp': timestamp,
        'date': date_str,
        'lat_range': (lats.min(), lats.max()),
        'lon_range': (lons.min(), lons.max()),
        'elevation_range': (elevation_flat.min(), elevation_flat.max()),
    }

    if verbose:
        print(f"  Extracted {len(tensor):,} points")
        print(f"  Lat: [{lats.min():.6f}, {lats.max():.6f}]")
        print(f"  Lon: [{lons.min():.6f}, {lons.max():.6f}]")
        print(f"  Elevation: [{elevation_flat.min():.1f}, {elevation_flat.max():.1f}] m")
        print(f"  RGB ranges: R[{r_flat.min():.3f},{r_flat.max():.3f}] "
              f"G[{g_flat.min():.3f},{g_flat.max():.3f}] "
              f"B[{b_flat.min():.3f},{b_flat.max():.3f}]")

    return tensor, metadata


def parse_directory_pairs(
    chm_dir: str,
    naip_dir: str,
    output_path: str,
    max_files: Optional[int] = None,
    verbose: bool = True
) -> Dict:
    """
    Parse all CHM/NAIP file pairs in directories

    Args:
        chm_dir: Directory with CHM elevation files
        naip_dir: Directory with NAIP RGB files
        output_path: Output .pt file path
        max_files: Max number of file pairs to process
        verbose: Print progress

    Returns:
        Dataset metadata dictionary
    """
    chm_dir = Path(chm_dir)
    naip_dir = Path(naip_dir)

    chm_files = sorted(chm_dir.glob('*.tif'))

    if max_files:
        chm_files = chm_files[:max_files]

    if verbose:
        print(f"Found {len(chm_files)} CHM files")
        print(f"Matching with NAIP RGB files...")

    all_tensors = []
    all_metadata = []
    matched = 0

    for i, chm_path in enumerate(chm_files):
        # Find matching NAIP file
        naip_path = find_matching_naip(str(chm_path), naip_dir)

        if naip_path is None:
            if verbose:
                print(f"  Warning: No NAIP match for {chm_path.name}")
            continue

        if verbose and (i % 10 == 0):
            print(f"  Progress: {matched}/{len(chm_files)} pairs processed")

        try:
            tensor, meta = parse_paired_geotiffs(
                str(chm_path),
                naip_path,
                verbose=False
            )
            all_tensors.append(tensor)
            all_metadata.append(meta)
            matched += 1
        except Exception as e:
            if verbose:
                print(f"  Error processing {chm_path.name}: {e}")
            continue

    if len(all_tensors) == 0:
        raise ValueError("No file pairs were successfully processed!")

    # Concatenate all tensors
    dataset = torch.cat(all_tensors, dim=0)

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'data': dataset,
        'metadata': all_metadata,
        'n_files': len(all_tensors),
        'n_points': len(dataset),
        'columns': ['lat', 'lon', 'elevation', 'timestamp', 'r', 'g', 'b'],
    }, output_path)

    if verbose:
        print(f"\nDataset saved to {output_path}")
        print(f"  Total points: {len(dataset):,}")
        print(f"  Total file pairs: {len(all_tensors)}")
        print(f"  File size: {output_path.stat().st_size / 1024**2:.2f} MB")

    return {
        'n_files': len(all_tensors),
        'n_points': len(dataset),
        'output_path': str(output_path),
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Parse paired CHM + NAIP files into Earth4D format'
    )
    parser.add_argument('--chm-dir', required=True, help='Directory with CHM elevation files')
    parser.add_argument('--naip-dir', required=True, help='Directory with NAIP RGB files')
    parser.add_argument('--output', '-o', required=True, help='Output .pt file path')
    parser.add_argument('--max-files', type=int, help='Max file pairs to process')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    parse_directory_pairs(
        args.chm_dir,
        args.naip_dir,
        args.output,
        max_files=args.max_files,
        verbose=args.verbose
    )
