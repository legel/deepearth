#!/usr/bin/env python3
"""
Utility functions for NAIP-3DEP data acquisition

Provides tools to:
- Convert lat/lon to UTM coordinates and zone
- Map coordinates to correct tar file URLs
- Download data for any geographic location
"""

import math
from typing import Tuple, List, Optional
from pathlib import Path


def latlon_to_utm_zone(lat: float, lon: float) -> int:
    """
    Convert latitude/longitude to UTM zone number

    Args:
        lat: Latitude in degrees (-90 to 90)
        lon: Longitude in degrees (-180 to 180)

    Returns:
        UTM zone number (1-60)

    Reference:
        https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system
    """
    # UTM zones are 6 degrees wide, starting at -180 longitude
    # Zone 1 is centered at -177, Zone 2 at -171, etc.
    zone = int((lon + 180) / 6) + 1

    # Special zones for Norway and Svalbard
    if lat >= 56.0 and lat < 64.0 and lon >= 3.0 and lon < 12.0:
        zone = 32
    elif lat >= 72.0 and lat < 84.0:
        if lon >= 0.0 and lon < 9.0:
            zone = 31
        elif lon >= 9.0 and lon < 21.0:
            zone = 33
        elif lon >= 21.0 and lon < 33.0:
            zone = 35
        elif lon >= 33.0 and lon < 42.0:
            zone = 37

    return zone


def latlon_to_utm(lat: float, lon: float) -> Tuple[int, float, float]:
    """
    Convert latitude/longitude to UTM coordinates (simplified)

    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees

    Returns:
        Tuple of (zone, easting, northing)

    Note:
        This is a simplified conversion. For production use, consider
        using the pyproj or utm library for more accurate results.
    """
    zone = latlon_to_utm_zone(lat, lon)

    # Convert to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    # UTM projection parameters (WGS84)
    a = 6378137.0  # Earth equatorial radius
    f = 1 / 298.257223563  # Flattening
    k0 = 0.9996  # Scale factor

    # Calculate central meridian
    lon0 = (zone - 1) * 6 - 180 + 3  # Central meridian of zone
    lon0_rad = math.radians(lon0)

    # Simplified UTM calculation (Karney 2011 formulas would be more accurate)
    e2 = 2 * f - f * f  # Eccentricity squared
    e = math.sqrt(e2)

    N = a / math.sqrt(1 - e2 * math.sin(lat_rad) ** 2)
    T = math.tan(lat_rad) ** 2
    C = e2 * math.cos(lat_rad) ** 2 / (1 - e2)
    A = (lon_rad - lon0_rad) * math.cos(lat_rad)

    M = a * ((1 - e2/4 - 3*e2*e2/64 - 5*e2*e2*e2/256) * lat_rad
             - (3*e2/8 + 3*e2*e2/32 + 45*e2*e2*e2/1024) * math.sin(2*lat_rad)
             + (15*e2*e2/256 + 45*e2*e2*e2/1024) * math.sin(4*lat_rad)
             - (35*e2*e2*e2/3072) * math.sin(6*lat_rad))

    easting = k0 * N * (A + (1 - T + C) * A**3 / 6
                        + (5 - 18*T + T**2 + 72*C - 58*e2) * A**5 / 120) + 500000.0

    northing = k0 * (M + N * math.tan(lat_rad) * (A**2 / 2
                     + (5 - T + 9*C + 4*C**2) * A**4 / 24
                     + (61 - 58*T + T**2 + 600*C - 330*e2) * A**6 / 720))

    if lat < 0:
        northing += 10000000.0  # False northing for southern hemisphere

    return zone, easting, northing


def get_tar_filename_for_location(lat: float, lon: float) -> Tuple[str, str, int, int, int]:
    """
    Get the tar filename containing data for a given lat/lon location

    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees

    Returns:
        Tuple of:
            - CHM tar filename (e.g., "15-316.tar")
            - NAIP tar filename (e.g., "15-316.tar")
            - UTM zone
            - Easting (rounded to integer)
            - Northing (rounded to integer)

    Example:
        >>> get_tar_filename_for_location(29.7604, -95.3698)  # Houston
        ('15-327.tar', '15-327.tar', 15, 271646, 3290632)
    """
    zone, easting, northing = latlon_to_utm(lat, lon)

    # Extract first 3 digits of northing (in kilometers)
    northing_km = int(northing) // 1000
    northing_prefix = northing_km

    # Tar files are named: {zone}-{northing_prefix}.tar
    tar_name = f"{zone}-{northing_prefix}.tar"

    return tar_name, tar_name, zone, int(easting), int(northing)


def get_download_urls_for_location(lat: float, lon: float) -> Tuple[str, str, dict]:
    """
    Get download URLs for CHM and NAIP data at a location

    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees

    Returns:
        Tuple of:
            - CHM download URL
            - NAIP download URL
            - Metadata dict with zone, easting, northing, etc.

    Example:
        >>> chm_url, naip_url, meta = get_download_urls_for_location(29.7604, -95.3698)
        >>> print(chm_url)
        http://rangeland.ntsg.umt.edu/data/rap/chm-naip/chm/15/15-327.tar
    """
    tar_name, _, zone, easting, northing = get_tar_filename_for_location(lat, lon)

    base_url = "http://rangeland.ntsg.umt.edu/data/rap/chm-naip"
    chm_url = f"{base_url}/chm/{zone}/{tar_name}"
    naip_url = f"{base_url}/naip/{zone}/{tar_name}"

    metadata = {
        'lat': lat,
        'lon': lon,
        'utm_zone': zone,
        'easting': easting,
        'northing': northing,
        'tar_filename': tar_name,
        'chm_url': chm_url,
        'naip_url': naip_url,
    }

    return chm_url, naip_url, metadata


def get_surrounding_locations(
    lat: float,
    lon: float,
    grid_size: int = 10,
    spacing_meters: float = 1000.0
) -> List[Tuple[float, float]]:
    """
    Get a grid of lat/lon points around a center location

    Args:
        lat: Center latitude in degrees
        lon: Center longitude in degrees
        grid_size: Size of grid (e.g., 10 for 10x10 grid)
        spacing_meters: Spacing between points in meters

    Returns:
        List of (lat, lon) tuples covering the grid

    Note:
        This is an approximation. For precise geodesic calculations,
        use geopy or pyproj.
    """
    # Approximate meters per degree at this latitude
    meters_per_lat = 111320.0  # Relatively constant
    meters_per_lon = 111320.0 * math.cos(math.radians(lat))

    # Convert spacing to degrees
    lat_spacing = spacing_meters / meters_per_lat
    lon_spacing = spacing_meters / meters_per_lon

    # Generate grid centered on location
    locations = []
    half_size = grid_size / 2.0

    for i in range(grid_size):
        for j in range(grid_size):
            # Offset from center
            lat_offset = (i - half_size + 0.5) * lat_spacing
            lon_offset = (j - half_size + 0.5) * lon_spacing

            grid_lat = lat + lat_offset
            grid_lon = lon + lon_offset

            locations.append((grid_lat, grid_lon))

    return locations


def get_unique_tar_files_for_region(
    lat: float,
    lon: float,
    grid_size: int = 10,
    spacing_meters: float = 1000.0
) -> List[Tuple[str, str, int]]:
    """
    Get unique tar files needed to cover a region

    Args:
        lat: Center latitude
        lon: Center longitude
        grid_size: Grid size (e.g., 10 for 10x10)
        spacing_meters: Spacing between points

    Returns:
        List of tuples: (tar_filename, chm_url, naip_url, zone)
    """
    locations = get_surrounding_locations(lat, lon, grid_size, spacing_meters)

    # Collect unique tar files
    tar_files = {}

    for loc_lat, loc_lon in locations:
        try:
            chm_url, naip_url, meta = get_download_urls_for_location(loc_lat, loc_lon)
            tar_name = meta['tar_filename']
            zone = meta['utm_zone']

            if tar_name not in tar_files:
                tar_files[tar_name] = (tar_name, chm_url, naip_url, zone)
        except Exception as e:
            print(f"Warning: Could not process ({loc_lat}, {loc_lon}): {e}")
            continue

    return list(tar_files.values())


# Test locations from requirements
TARGET_LOCATIONS = {
    'ASU_Tempe': (33.42, -111.93),          # UTM Zone 12
    'Boulder_CO': (40.01, -105.27),         # UTM Zone 13
    'NYC_HighLine': (40.75, -74.00),        # UTM Zone 18
    'Stanford_CA': (37.43, -122.17),        # UTM Zone 10
    'Miami_SouthBeach': (25.76, -80.13),    # UTM Zone 17
}


if __name__ == '__main__':
    """Test the utility functions"""

    print("="*70)
    print("NAIP Utility Functions Test")
    print("="*70)

    for name, (lat, lon) in TARGET_LOCATIONS.items():
        print(f"\n{name}: ({lat}, {lon})")

        # Get UTM zone
        zone = latlon_to_utm_zone(lat, lon)
        print(f"  UTM Zone: {zone}")

        # Get UTM coordinates
        zone, easting, northing = latlon_to_utm(lat, lon)
        print(f"  UTM Coords: Zone {zone}, E {easting:.1f}, N {northing:.1f}")

        # Get tar file
        chm_tar, naip_tar, zone, e, n = get_tar_filename_for_location(lat, lon)
        print(f"  Tar file: {chm_tar}")

        # Get URLs
        chm_url, naip_url, meta = get_download_urls_for_location(lat, lon)
        print(f"  CHM URL: {chm_url}")
        print(f"  NAIP URL: {naip_url}")

        # Get unique tar files for 10x10 region
        tar_files = get_unique_tar_files_for_region(lat, lon, grid_size=10, spacing_meters=1000)
        print(f"  Tar files for 10x10 region: {len(tar_files)} unique files")

    print("\n" + "="*70)
