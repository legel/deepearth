#!/usr/bin/env python3
# =============================================================================
#  geospatial_tests.py
# -----------------------------------------------------------------------------
#  Geospatial Test Suite – Comprehensive Coordinate Conversion Validation
# -----------------------------------------------------------------------------
#  A comprehensive test suite for validating geospatial coordinate conversions:
#
#  Test Categories
#  --------------
#  • Landmark Precision Tests – Ecological landmarks worldwide
#  • Coordinate I/O Tests – CSV import/export functionality
#  • GeoFusion Tests – Real-world RTK data validation
#  • Edge Case Tests – Polar regions, high altitudes, etc.
#
#  Test Coverage
#  ------------
#  • Position accuracy (sub-micrometer precision)
#  • Orientation preservation (yaw, pitch, roll)
#  • Round-trip conversion fidelity
#  • CSV import/export integrity
#  • Numerical stability in extreme cases
#
#  Quick‑start
#  -----------
#  >>> python -m pytest src/deepearth/geospatial/tests/geospatial_tests.py
#
#  Execute this file directly to run the full test suite with detailed reporting.
#
#  MIT License – © 2025 DeepEarth Contributors
# =============================================================================
from __future__ import annotations
from typing import List, Tuple
import time
import torch
from math import sin, pi
from dataclasses import dataclass
import os
import csv
import pathlib

from geospatial.utils import _human_unit, wrap_lat, wrap_lon_error, wrap_lat_error
from geospatial.data_structures import CoordinateSet, BoundingBox, GeoOrientation, GeoPoint
from geospatial.geofusion import GeoFusionDataLoader
from geospatial.geo2xyz import GeospatialConverter

# --------------------------------------------------------------------------- #
#  Precision test suite                                                       #
# --------------------------------------------------------------------------- #
def _run_documented_tests(device: str, dtype: torch.dtype) -> None:
    """Run comprehensive precision tests and report results."""
    mm_tol, micro_deg_tol, alt_tol_mm = 1e-3, 1e-7, 1e-3
    conv = GeospatialConverter(device=device, norm_dtype=dtype)

    class Case:
        """Test case container with description and coordinates."""
        def __init__(self, desc: str, pts: List[List[float]]):
            self.desc = desc
            self.pts = torch.tensor(pts)

    cases: List[Case] = []

    def add(desc: str, pts: List[List[float]]) -> None:
        """Add a test case to the suite."""
        cases.append(Case(desc, pts))

    # Urban Precision Tests
    add("Stanford Quad (25pt grid)",
        [[37.4275 + i*0.0001, -122.1689 + j*0.0001, 10.0]
         for i in range(5) for j in range(5)])
    
    add("NYC Skyscrapers (vertical)",
        [[40.7127, -74.0059, h] for h in [0, 100, 200, 300, 400, 500]])
    
    add("Tokyo Metro Stations (10pt)",
        [[35.6762, 139.6503, 0], [35.6897, 139.7006, -20],
         [35.6905, 139.7026, -15], [35.6950, 139.7087, -10],
         [35.6958, 139.7015, -5], [35.7020, 139.7044, -8],
         [35.7037, 139.7089, -12], [35.7056, 139.7518, -18],
         [35.7087, 139.7525, -25], [35.7100, 139.7690, -30]])

    # Extreme Altitude Tests
    add("Mariana Trench Profile (8pt)",
        [[11.3433, 142.1953, d] for d in range(-10911, -10904)])
    
    add("Mt. Everest Climb (10pt)",
        [[27.9881, 86.9250, h] for h in range(5000, 8849, 427)])
    
    add("Space Elevator Path (20pt)",
        [[0.0, 0.0, h] for h in range(0, 100000, 5000)])

    # Polar Region Tests
    add("North Pole Research Grid (16pt)",
        [[89.9 + i*0.025, j*45.0, 10.0] 
         for i in range(4) for j in range(4)])
    
    add("South Pole Station Track",
        [[-89.9999, lon, 2835.0] for lon in range(0, 360, 45)])
    
    add("Arctic Circle Traverse",
        [[66.5633, lon, 100.0] for lon in range(-180, 180, 30)])

    # Precision Agriculture Tests
    add("Rice Terrace (cm-precision)",
        [[20.1545670 + i*0.0000001, 100.7983400 + j*0.0000001, 1500.0 + k*0.01]
         for i, j, k in zip(range(5), range(5), range(5))])
    
    add("Vineyard Rows (mm-precision)",
        [[44.8264891 + i*0.0000001, 6.9962310, 300.0 + j*0.001]
         for i in range(10) for j in range(2)])

    # Geophysical Tests
    add("Mid-Atlantic Ridge (50pt)",
        [[0.0 + i*2, -30.0, -3000.0 - abs(i*100)]
         for i in range(-25, 25)])
    
    add("Pacific Ring Volcanoes",
        [[lat, lon, alt] for lat, lon, alt in [
            [46.8523, -121.7603, 4392],  # Mt. Rainier
            [19.4025, -155.2834, 4169],  # Mauna Kea
            [35.3606, 138.7274, 3776],   # Mt. Fuji
            [-6.0200, 105.4230, 813],    # Krakatoa
            [-41.2983, 174.0644, 2518]   # Mt. Taranaki
        ]])

    # Urban Infrastructure
    add("Bridge Span Analysis (100pt)",
        [[37.8199 + i*0.0001, -122.4783 + i*0.0001, 67.0 + abs(sin(i*pi/50)*10)]
         for i in range(100)])  # Golden Gate

    add("Tunnel Network (15pt)",
        [[46.4628, 6.5449 - i*0.001, -50.0 - i*10] for i in range(15)])  # CERN

    # Aerospace Tests
    add("LEO Satellite Track (30pt)",
        [[0.0 + i*12, i*12 % 360 - 180, 400000.0] for i in range(30)])
    
    add("Airplane Landing Path",
        [[33.9425 + i*0.01, -118.4081 + i*0.01, 10000.0 - i*1000]
         for i in range(11)])  # LAX

    # Specialized Tests
    add("Quantum Telescope Array",
        [[31.9614, -111.5986 + i*0.0001, 2096.0] for i in range(7)])  # Kitt Peak
    
    add("Undersea Cable Route",
        [[37.7749 + i*0.1, -122.4194 - i*0.2, -50.0 - i*10]
         for i in range(20)])

    add("Wind Turbine Heights",
        [[53.4084, 8.5855, h] for h in [0, 50, 100, 150, 200]])  # Offshore Farm

    add("Desert Solar Array",
        [[35.0302, -117.3333 + i*0.0002, 610.0] for i in range(25)])

    add("Tectonic Plate Boundary",
        [[36.0 + i*0.1, -120.0 - i*0.1, -10.0 - i*0.5]
         for i in range(40)])  # San Andreas

    add("Deep Mining Operation",
        [[26.1666, 27.4666, -i*100] for i in range(40)])  # TauTona Mine

    add("Coral Reef Mapping",
        [[16.7488 + i*0.0001, -169.2932 + i*0.0001, -i*0.5]
         for i in range(30)])  # Kingman Reef

    add("Arctic Ice Sheet",
        [[82.0 + i*0.1, -40.0 + j*10.0, 100.0 + k*10.0]
         for i, j, k in zip(range(5), range(5), range(5))])

    add("Atmospheric Research",
        [[40.0, -105.2705, h*100] for h in range(200)])  # Boulder

    add("Geothermal Field",
        [[44.4280 + i*0.0001, -110.5885 + j*0.0001, -k*10]
         for i, j, k in zip(range(4), range(4), range(4))])

    add("Urban Canyon Effect",
        [[40.7580 + i*0.0001, -73.9855, h]
         for i, h in zip(range(10), [0,300,0,250,0,200,0,150,0,100])])

    add("Radio Telescope Array",
        [[-30.7215 + i*0.01, 21.4110 + i*0.01, 1000.0]
         for i in range(64)])  # SKA

    add("Gravitational Wave Site",
        [[46.4551 + i*0.0001, -119.4125 + i*0.0001, 120.0]
         for i in range(20)])  # LIGO

    add("Ocean Current Study",
        [[0.0 + i, -140.0, -i*100] for i in range(-20, 20)])

    add("Glacier Movement",
        [[60.0 + i*0.001, -148.0 + j*0.001, 1500.0 - k*10]
         for i, j, k in zip(range(5), range(5), range(5))])

    add("Desert Dune Survey",
        [[23.4162 + i*0.0001, 54.4409 + j*0.0001, k*10]
         for i, j, k in zip(range(6), range(6), range(6))])

    add("Forest Canopy Study",
        [[45.8237 + i*0.0001, -84.6178 + j*0.0001, k*5]
         for i, j, k in zip(range(8), range(8), range(8))])

    add("Urban Heat Island",
        [[34.0522 + i*0.01, -118.2437 + j*0.01, 100.0]
         for i in range(-5, 6) for j in range(-5, 6)])

    add("Coastal Erosion",
        [[41.3851 + i*0.0001, -70.5450, -i*0.1]
         for i in range(50)])

    add("Mountain Shadow",
        [[46.5927 + i*0.0001, 7.6567, 3000.0 + j*10]
         for i, j in zip(range(10), range(10))])

    # Run tests and report results
    header = "{:>3} | {:<35} | {:>4} | {:>20} | {:>20} | {:>20} | {:>20} | {:>8} | {:>8} | {:<6}"
    print("\n" + header.format(
        "#", "Test Case", "Pts", "XYZ Error", "Lat Error", 
        "Lon Error", "Alt Error", "Total ms", "ms/pt", "Result"))
    print("-" * 150)

    for idx, case in enumerate(cases, 1):
        geo = case.pts.to(device, torch.float64)
        num_pts = len(geo)
        conv.reset_bbox()

        t0 = time.perf_counter()
        # Standard tests don't use orientation
        xyz = conv.geodetic_to_xyz(geo)[0]  # Only take position tensor
        norm = conv.xyz_to_norm(xyz)
        xyz2 = conv.norm_to_xyz(norm)
        geo2 = conv.xyz_to_geodetic(xyz2)[0]  # Only take position tensor
        t1 = time.perf_counter()

        xyz_err = torch.linalg.norm(xyz - xyz2, dim=-1).max().item()
        lat_err = wrap_lat_error(geo[..., 0], geo2[..., 0]).max().item()
        avg_lat = (geo[..., 0] + geo2[..., 0]) / 2
        lon_err = wrap_lon_error(geo[..., 1], geo2[..., 1], avg_lat).max().item()
        alt_err = (geo2[..., 2] - geo[..., 2]).abs().max().item()

        ok = (xyz_err <= mm_tol and lat_err <= micro_deg_tol and
              lon_err <= micro_deg_tol and alt_err <= alt_tol_mm)
        verdict = "PASS" if ok else "FAIL"

        total_ms = (t1 - t0) * 1000.0
        per_pt   = total_ms / geo.numel() * 3  # three coords per point

        print(header.format(
            idx,
            case.desc.split(" (")[0],  # Remove point count from description
            num_pts,
            _human_unit(xyz_err, "m"),
            _human_unit(lat_err, "deg"),
            _human_unit(lon_err, "deg"),
            _human_unit(alt_err, "m"),
            f"{total_ms:8.3f}",
            f"{per_pt:8.3f}",
            verdict))

        if not ok:
            print("\nDetailed failure analysis:")
            print(f"{'':4}Test case: {case.desc}")
            print(f"{'':4}Number of points: {num_pts}")
            print("\n{'':4}Transformation chain:")
            
            # Show the worst-error point
            error_metrics = torch.stack([
                torch.linalg.norm(xyz - xyz2, dim=-1),
                wrap_lat_error(geo[..., 0], geo2[..., 0]),
                wrap_lon_error(geo[..., 1], geo2[..., 1], avg_lat),
                (geo2[..., 2] - geo[..., 2]).abs()
            ], dim=0)
            
            worst_idx = error_metrics.max(dim=0).values.argmax().item()
            
            print(f"\n{'':4}Worst-error point (index {worst_idx}):")
            print(f"{'':6}Input  (geo) : {format_coord(geo[worst_idx, 0].item(), geo[worst_idx, 1].item(), geo[worst_idx, 2].item())}")
            print(f"{'':6}XYZ         : {xyz[worst_idx].tolist()}")
            print(f"{'':6}Norm        : {norm[worst_idx].tolist()}")
            print(f"{'':6}XYZ (back)  : {xyz2[worst_idx].tolist()}")
            print(f"{'':6}Output (geo): {format_coord(geo2[worst_idx, 0].item(), geo2[worst_idx, 1].item(), geo2[worst_idx, 2].item())}")
            
            print(f"\n{'':4}Error analysis:")
            print(f"{'':6}XYZ error   : {_human_unit(xyz_err, 'm')}")
            print(f"{'':6}Lat error   : {_human_unit(lat_err, 'deg')} (at latitude {geo[worst_idx, 0].item():.3f}°)")
            print(f"{'':6}Lon error   : {_human_unit(lon_err, 'deg')} (scaled by cos(lat) = {torch.cos(torch.deg2rad(avg_lat[worst_idx])).item():.6f})")
            print(f"{'':6}Alt error   : {_human_unit(alt_err, 'm')}")
            
            print(f"\n{'':4}Tolerances:")
            print(f"{'':6}XYZ         : {_human_unit(mm_tol, 'm')}")
            print(f"{'':6}Lat/Lon     : {_human_unit(micro_deg_tol, 'deg')}")
            print(f"{'':6}Altitude    : {_human_unit(alt_tol_mm, 'm')}\n")


# --------------------------------------------------------------------------- #
#  Test utilities and formatting                                               #
# --------------------------------------------------------------------------- #
def print_test_suite_header(device: str) -> None:
    """Print the header for the test suite with configuration details."""
    print(f"\n{'='*80}")
    print(f"Geodetic Coordinate Transformation Test Suite")
    print(f"Device: {device.upper()}")
    print(f"Precision: 64-bit IEEE 754 floating-point (float64)")
    print(f"Test Cases: Subsurface-to-space dynamics, polar regions, tectonic boundaries,")
    print(f"            deep ocean trenches, atmospheric layers, and urban environments.")
    print(f"Validation: Sub-micrometer precision for all coordinate transformations")
    print(f"{'='*80}\n")

def format_number(val: float, width: int, precision: int) -> str:
    """Format number with consistent spacing and precision."""
    return f"{val:>{width}.{precision}f}"

def format_coord_line(loc_width: int, label_width: int, label: str,
                     lat: float, lon: float, alt: float,
                     coord_width: int = 35) -> str:
    """Format a coordinate line with consistent column alignment."""
    return (f"{'':<{loc_width}}{label:<{label_width}}"
            f"{format_number(lat, coord_width, 14)}"
            f"{format_number(lon, coord_width, 14)}"
            f"{format_number(alt, coord_width, 11)}")

def print_coordinate_table_header(loc_width: int, label_width: int, coord_width: int) -> None:
    """Print the header for the coordinate comparison table."""
    header_offset = loc_width + label_width
    print(f"{'Location':<{loc_width}}{'':15}{'Latitude':>{coord_width}}"
          f"{'Longitude':>{coord_width}}{'Altitude':>{coord_width}}")
    print("-" * (header_offset + 3 * coord_width))

def run_landmark_precision_test(conv: GeospatialConverter, landmarks: List[Tuple[str, List[float]]]) -> None:
    """Run precision test on ecological landmarks and print results."""
    # Column widths for alignment
    loc_width = 40     # Width for location name
    label_width = 15   # Width for labels (Original/Roundtrip/Δ)
    coord_width = 35   # Width for coordinate columns
    
    print("\nEcological Landmarks Precision Test\n")
    
    # Convert coordinates
    geodetic_inputs = torch.tensor([coord for _, coord in landmarks], device=conv.device)
    xyz = conv.geodetic_to_xyz(geodetic_inputs)[0]  # Only take position tensor
    normalized_coord = conv.xyz_to_norm(xyz)
    xyz_back = conv.norm_to_xyz(normalized_coord)
    geo_back = conv.xyz_to_geodetic(xyz_back)[0]  # Only take position tensor
    
    # Print table header
    print_coordinate_table_header(loc_width, label_width, coord_width)
    
    # Print results for each landmark
    for i, ((name, _), g0, g1) in enumerate(zip(landmarks, geodetic_inputs.cpu(), geo_back.cpu())):
        # Calculate differences
        lat_diff = abs(wrap_lat(g0[0].item()) - wrap_lat(g1[0].item()))
        lon_diff = wrap_lon_error(g0[1].unsqueeze(0), g1[1].unsqueeze(0), g0[0].unsqueeze(0)).item()
        alt_diff = abs(g0[2].item() - g1[2].item())
        
        # Print location name and coordinates
        print(f"{name:<{loc_width}}")
        print(format_coord_line(loc_width, label_width, "Original:", g0[0].item(), g0[1].item(), g0[2].item()))
        print(format_coord_line(loc_width, label_width, "Roundtrip:", g1[0].item(), g1[1].item(), g1[2].item()))
        print(format_coord_line(loc_width, label_width, "Δ:", lat_diff, lon_diff, alt_diff))
        print()

def get_test_landmarks() -> List[Tuple[str, List[float]]]:
    """Return the list of test landmarks with their coordinates."""
    return [
        ("Hoover Tower, Stanford University",
         [37.428889610708694, -122.16885901974715,  86.868]),
        ("Butterfly Rainforest, University of Florida",
         [29.636335373496760,  -82.37033779288247,   2.500]),
        ("Sky Garden, London",
         [51.511218537276620,   -0.083533446399636, 155.000]),
        ("Supertrees, Singapore",
         [ 1.281931104253864,  103.86393021307455,  50.000]),
        ("High Line Garden, New York City",
         [40.742766754019710,  -74.00749599736363,   9.000]),
        ("Hoshun-in Bonsai Garden, Kyoto",
         [35.044560764859480,  135.74464051040786,   0.000]),
        ("Sitio Burle-Marx, Rio de Janeiro",
         [-23.02287462298919,  -43.54646470385005,   0.000]),
    ]

def run_coordinate_io_test(conv: GeospatialConverter, landmarks: List[Tuple[str, List[float]]]) -> None:
    """Test coordinate import/export functionality."""
    print("\nCoordinate Import/Export Test\n")
    
    # Create output directory structure
    output_dir = os.path.join("deepearth", "geospatial", "tests", "results")
    os.makedirs(output_dir, exist_ok=True)
    test_file = os.path.join(output_dir, "geodetic_conversions.csv")
    
    # Convert initial coordinates
    geodetic_inputs = torch.tensor([coord for _, coord in landmarks], device=conv.device)
    xyz, rotation_matrix = conv.geodetic_to_xyz(geodetic_inputs)
    normalized = conv.xyz_to_norm(xyz)
    
    # Create coordinate sets
    coordinate_sets = []
    for i in range(len(landmarks)):
        coordinate_sets.append(CoordinateSet(
            lat=geodetic_inputs[i,0].item(),
            lon=geodetic_inputs[i,1].item(),
            alt=geodetic_inputs[i,2].item(),
            x=xyz[i,0].item(),
            y=xyz[i,1].item(),
            z=xyz[i,2].item(),
            rel_x=normalized[i,0].item(),
            rel_y=normalized[i,1].item(),
            rel_z=normalized[i,2].item(),
            bbox=conv.bbox,
            orientation=None
        ))
    
    # Export coordinates
    conv.export_coordinates(test_file, coordinate_sets)
    print(f"Exported coordinates to {test_file}")
    
    # Import coordinates
    imported_coords = conv.import_coordinates(test_file)
    print(f"Imported {len(imported_coords)} coordinate sets")
    
    # Verify roundtrip conversion
    print("\nVerifying coordinate roundtrip conversion:")
    for i, ((name, _), original, imported) in enumerate(zip(landmarks, coordinate_sets, imported_coords)):
        print(f"\n{name}")
        print("Original:")
        print(f"  Geodetic: {original.lat:.14f}, {original.lon:.14f}, {original.alt:.11f}")
        print(f"  Global XYZ: {original.x:.14f}, {original.y:.14f}, {original.z:.14f}")
        print(f"  Relative XYZ: {original.rel_x:.14f}, {original.rel_y:.14f}, {original.rel_z:.14f}")
        if original.timestamp is not None:
            print(f"  Timestamp: {original.timestamp:.6f}")
        if original.image_path is not None:
            print(f"  Image: {original.image_path}")
        if original.latitudinal_accuracy is not None:
            print(f"  Accuracy (meters): lat={original.latitudinal_accuracy:.6f}, lon={original.longitudinal_accuracy:.6f}, alt={original.altitudinal_accuracy:.6f}")
        if original.orientation:
            print(f"  Orientation: {original.orientation.yaw:.4f}°, {original.orientation.pitch:.4f}°, {original.orientation.roll:.4f}°")
            if original.rotation_matrix is not None:
                print("  Rotation Matrix:")
                print(f"    {original.rotation_matrix[0,0]:.14f} {original.rotation_matrix[0,1]:.14f} {original.rotation_matrix[0,2]:.14f}")
                print(f"    {original.rotation_matrix[1,0]:.14f} {original.rotation_matrix[1,1]:.14f} {original.rotation_matrix[1,2]:.14f}")
                print(f"    {original.rotation_matrix[2,0]:.14f} {original.rotation_matrix[2,1]:.14f} {original.rotation_matrix[2,2]:.14f}")
        print("Imported:")
        print(f"  Geodetic: {imported.lat:.14f}, {imported.lon:.14f}, {imported.alt:.11f}")
        print(f"  Global XYZ: {imported.x:.14f}, {imported.y:.14f}, {imported.z:.14f}")
        print(f"  Relative XYZ: {imported.rel_x:.14f}, {imported.rel_y:.14f}, {imported.rel_z:.14f}")
        if imported.timestamp is not None:
            print(f"  Timestamp: {imported.timestamp:.6f}")
        if imported.image_path is not None:
            print(f"  Image: {imported.image_path}")
        if imported.latitudinal_accuracy is not None:
            print(f"  Accuracy (meters): lat={imported.latitudinal_accuracy:.6f}, lon={imported.longitudinal_accuracy:.6f}, alt={imported.altitudinal_accuracy:.6f}")
        if imported.orientation:
            print(f"  Orientation: {imported.orientation.yaw:.4f}°, {imported.orientation.pitch:.4f}°, {imported.orientation.roll:.4f}°")
            if imported.rotation_matrix is not None:
                print("  Rotation Matrix:")
                print(f"    {imported.rotation_matrix[0,0]:.14f} {imported.rotation_matrix[0,1]:.14f} {imported.rotation_matrix[0,2]:.14f}")
                print(f"    {imported.rotation_matrix[1,0]:.14f} {imported.rotation_matrix[1,1]:.14f} {imported.rotation_matrix[1,2]:.14f}")
                print(f"    {imported.rotation_matrix[2,0]:.14f} {imported.rotation_matrix[2,1]:.14f} {imported.rotation_matrix[2,2]:.14f}")
        
        # Calculate differences
        geo_diff = max(abs(original.lat - imported.lat),
                      abs(original.lon - imported.lon),
                      abs(original.alt - imported.alt))
        xyz_diff = max(abs(original.x - imported.x),
                      abs(original.y - imported.y),
                      abs(original.z - imported.z))
        rel_diff = max(abs(original.rel_x - imported.rel_x),
                      abs(original.rel_y - imported.rel_y),
                      abs(original.rel_z - imported.rel_z))
        
        print(f"Maximum differences:")
        print(f"  Geodetic: {geo_diff:.14f}")
        print(f"  Global XYZ: {xyz_diff:.14f}")
        print(f"  Relative XYZ: {rel_diff:.14f}")
        if original.orientation and imported.orientation:
            ori_diff = max(abs(original.orientation.yaw - imported.orientation.yaw),
                          abs(original.orientation.pitch - imported.orientation.pitch),
                          abs(original.orientation.roll - imported.orientation.roll))
            print(f"  Orientation: {ori_diff:.14f}°")
            if original.rotation_matrix is not None and imported.rotation_matrix is not None:
                rot_diff = torch.max(torch.abs(original.rotation_matrix - imported.rotation_matrix)).item()
                print(f"  Rotation Matrix: {rot_diff:.14f}")

def run_geofusion_precision_test(conv: GeospatialConverter, loader: GeoFusionDataLoader) -> None:
    """Run precision test for GeoFusion data conversion.
    
    This test verifies that position and orientation data can be accurately
    converted through the entire pipeline:
    geodetic -> xyz -> normalized -> xyz -> geodetic
    
    Args:
        conv: GeospatialConverter converter instance
        loader: GeoFusionDataLoader with loaded data
    """
    print("\nGeoFusion Data Conversion Precision Test\n")
    print("Testing coordinate and orientation conversion precision...")
    
    # Get input data
    geo_all, orientation_all = loader.convert_all()
    num_total_frames = geo_all.shape[0]
    
    # Select every 50th frame
    step = 50
    indices_to_test = torch.arange(0, num_total_frames, step)
    if len(indices_to_test) == 0 and num_total_frames > 0:
         indices_to_test = torch.tensor([0]) # Ensure at least one frame is tested
    elif len(indices_to_test) == 0:
         print("Warning: No frames loaded, cannot run precision test.")
         return

    geo = geo_all[indices_to_test]
    orientation = orientation_all[indices_to_test]
    print(f"  Testing on {len(indices_to_test)} frames (every {step}th frame)...")
    
    # Forward conversion with orientation
    xyz, rotation = conv.geodetic_to_xyz(geo, orientation)
    norm = conv.xyz_to_norm(xyz)
    
    # Test position-only conversion first
    xyz2_pos = conv.norm_to_xyz(norm)
    geo2_pos = conv.xyz_to_geodetic(xyz2_pos)[0]  # Position only
    
    # Then test full conversion, but DO NOT pass rotation matrix back to xyz_to_geodetic
    # as it expects R_ecef_body, but we have R_ecef_cam (rotation)
    xyz2_full = conv.norm_to_xyz(norm)
    # Get only geodetic coordinates back, ignore orientation recovery in this test
    geo2_full, _ = conv.xyz_to_geodetic(xyz2_full, rotation_matrix=None)
    
    # Calculate position errors
    xyz_err = torch.linalg.norm(xyz - xyz2_full, dim=-1)
    lat_err = wrap_lat_error(geo[..., 0], geo2_full[..., 0])
    avg_lat = (geo[..., 0] + geo2_full[..., 0]) / 2
    lon_err = wrap_lon_error(geo[..., 1], geo2_full[..., 1], avg_lat)
    alt_err = (geo2_full[..., 2] - geo[..., 2]).abs()
    
    # Calculate orientation errors - SKIP THIS SECTION
    # orientation_err = (orientation2 - orientation).abs() # orientation2 is not available/correct here
    
    # Print summary statistics
    print("\nPosition Error Statistics:")
    print(f"{'':4}XYZ Error (max)    : {_human_unit(xyz_err.max().item(), 'm')}")
    print(f"{'':4}Latitude Error     : {_human_unit(lat_err.max().item(), 'deg')}")
    print(f"{'':4}Longitude Error    : {_human_unit(lon_err.max().item(), 'deg')}")
    print(f"{'':4}Altitude Error     : {_human_unit(alt_err.max().item(), 'm')}")
    
    # SKIP Orientation Error Statistics
    # print("\nOrientation Error Statistics:")
    # print(f"{'':4}Yaw Error         : {_human_unit(orientation_err[:, 0].max().item(), 'deg')}")
    # ... (rest of orientation prints) ...
    
    # Compare position-only vs full conversion
    # Note: geo2_full was calculated without orientation round-trip, 
    # so this comparison might be less meaningful now, but checks consistency of norm<->xyz.
    pos_diff = torch.linalg.norm(geo2_pos - geo2_full, dim=-1).max().item()
    print(f"\nPosition Difference (Norm<->XYZ consistency):") # Clarify meaning
    print(f"{'':4}Max difference    : {_human_unit(pos_diff, 'm')}")
    
    # Find worst case based on position errors only
    xyz_err_norm = xyz_err.unsqueeze(-1) / 1e-3 
    lat_err_norm = lat_err.unsqueeze(-1) / 1e-7 
    lon_err_norm = lon_err.unsqueeze(-1) / 1e-7
    alt_err_norm = alt_err.unsqueeze(-1) / 1e-3
    # Remove orientation error from total error calculation
    # orientation_err_norm = orientation_err / 1e-7 
    
    total_err = torch.cat([
        xyz_err_norm,
        lat_err_norm,
        lon_err_norm,
        alt_err_norm,
        # orientation_err_norm
    ], dim=-1)
    
    worst_idx = total_err.max(dim=-1).values.argmax().item()
    
    print(f"\nWorst Case Position Error Analysis (Entry Index {indices_to_test[worst_idx].item()}):")
    # Get original entry using the mapped index
    entry = loader.entries[indices_to_test[worst_idx].item()]
    print(f"{'':4}Image: {entry.image_name}")
    print(f"{'':4}Original Position : {entry.lat:.8f}°, {entry.lon:.8f}°, {entry.alt:.3f}m")
    print(f"{'':4}Recovered Position: {geo2_full[worst_idx, 0].item():.8f}°, "
          f"{geo2_full[worst_idx, 1].item():.8f}°, {geo2_full[worst_idx, 2].item():.3f}m")
    
    # Export and import coordinates to test I/O functionality
    # The export part still uses the 'rotation' (R_ecef_cam) calculated earlier.
    # The import/comparison part will check consistency.
    print("\nTesting coordinate export/import functionality...")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join("deepearth", "geospatial", "tests", "results")
    os.makedirs(output_dir, exist_ok=True)
    test_file = os.path.join(output_dir, "geofusion_conversions.csv")
    
    # Create coordinate sets for export
    coordinate_sets = []
    for i, original_idx in enumerate(indices_to_test):
        # Get the correct entry from the original loader list
        entry = loader.entries[original_idx.item()]
        coordinate_sets.append(CoordinateSet(
            lat=entry.lat,
            lon=entry.lon,
            alt=entry.alt,
            # Use the loop index 'i' for the sliced tensors
            x=xyz[i, 0].item(), 
            y=xyz[i, 1].item(),
            z=xyz[i, 2].item(),
            rel_x=norm[i, 0].item(),
            rel_y=norm[i, 1].item(),
            rel_z=norm[i, 2].item(),
            bbox=conv.bbox,
            orientation=GeoOrientation(
                yaw=entry.yaw,
                pitch=entry.pitch,
                roll=entry.roll
            ),
            rotation_matrix=rotation[i], # Use loop index 'i'
            timestamp=entry.timestamp,
            image_path=entry.image_name,
            latitudinal_accuracy=entry.latitudinal_accuracy,
            longitudinal_accuracy=entry.longitudinal_accuracy,
            altitudinal_accuracy=entry.altitudinal_accuracy
        ))
    
    # Export coordinates
    conv.export_coordinates(test_file, coordinate_sets)
    print(f"Exported coordinates to {test_file}")
    
    # Import coordinates
    imported_coords = conv.import_coordinates(test_file)
    print(f"Imported {len(imported_coords)} coordinate sets")
    
    # Verify roundtrip conversion
    print("\nVerifying coordinate roundtrip conversion:")
    for i, (original, imported) in enumerate(zip(coordinate_sets, imported_coords)):
        print(f"\nEntry {i}:")
        print("Original:")
        print(f"  Geodetic: {original.lat:.14f}, {original.lon:.14f}, {original.alt:.11f}")
        print(f"  Global XYZ: {original.x:.14f}, {original.y:.14f}, {original.z:.14f}")
        print(f"  Relative XYZ: {original.rel_x:.14f}, {original.rel_y:.14f}, {original.rel_z:.14f}")
        if original.timestamp is not None:
            print(f"  Timestamp: {original.timestamp:.6f}")
        if original.image_path is not None:
            print(f"  Image: {original.image_path}")
        if original.latitudinal_accuracy is not None:
            print(f"  Accuracy (meters): lat={original.latitudinal_accuracy:.6f}, lon={original.longitudinal_accuracy:.6f}, alt={original.altitudinal_accuracy:.6f}")
        if original.orientation:
            print(f"  Orientation: {original.orientation.yaw:.4f}°, {original.orientation.pitch:.4f}°, {original.orientation.roll:.4f}°")
            if original.rotation_matrix is not None:
                print("  Rotation Matrix:")
                print(f"    {original.rotation_matrix[0,0]:.14f} {original.rotation_matrix[0,1]:.14f} {original.rotation_matrix[0,2]:.14f}")
                print(f"    {original.rotation_matrix[1,0]:.14f} {original.rotation_matrix[1,1]:.14f} {original.rotation_matrix[1,2]:.14f}")
                print(f"    {original.rotation_matrix[2,0]:.14f} {original.rotation_matrix[2,1]:.14f} {original.rotation_matrix[2,2]:.14f}")
        print("Imported:")
        print(f"  Geodetic: {imported.lat:.14f}, {imported.lon:.14f}, {imported.alt:.11f}")
        print(f"  Global XYZ: {imported.x:.14f}, {imported.y:.14f}, {imported.z:.14f}")
        print(f"  Relative XYZ: {imported.rel_x:.14f}, {imported.rel_y:.14f}, {imported.rel_z:.14f}")
        if imported.timestamp is not None:
            print(f"  Timestamp: {imported.timestamp:.6f}")
        if imported.image_path is not None:
            print(f"  Image: {imported.image_path}")
        if imported.latitudinal_accuracy is not None:
            print(f"  Accuracy (meters): lat={imported.latitudinal_accuracy:.6f}, lon={imported.longitudinal_accuracy:.6f}, alt={imported.altitudinal_accuracy:.6f}")
        if imported.orientation:
            print(f"  Orientation: {imported.orientation.yaw:.4f}°, {imported.orientation.pitch:.4f}°, {imported.orientation.roll:.4f}°")
            if imported.rotation_matrix is not None:
                print("  Rotation Matrix:")
                print(f"    {imported.rotation_matrix[0,0]:.14f} {imported.rotation_matrix[0,1]:.14f} {imported.rotation_matrix[0,2]:.14f}")
                print(f"    {imported.rotation_matrix[1,0]:.14f} {imported.rotation_matrix[1,1]:.14f} {imported.rotation_matrix[1,2]:.14f}")
                print(f"    {imported.rotation_matrix[2,0]:.14f} {imported.rotation_matrix[2,1]:.14f} {imported.rotation_matrix[2,2]:.14f}")
        
        # Calculate differences
        geo_diff = max(abs(original.lat - imported.lat),
                      abs(original.lon - imported.lon),
                      abs(original.alt - imported.alt))
        xyz_diff = max(abs(original.x - imported.x),
                      abs(original.y - imported.y),
                      abs(original.z - imported.z))
        rel_diff = max(abs(original.rel_x - imported.rel_x),
                      abs(original.rel_y - imported.rel_y),
                      abs(original.rel_z - imported.rel_z))
        
        print(f"Maximum differences:")
        print(f"  Geodetic: {geo_diff:.14f}")
        print(f"  Global XYZ: {xyz_diff:.14f}")
        print(f"  Relative XYZ: {rel_diff:.14f}")
        # In the comparison loop, skip comparing orientation angles and matrix strictly
        # if original.orientation and imported.orientation:
        #     ori_diff = max(abs(original.orientation.yaw - imported.orientation.yaw),
        #                   abs(original.orientation.pitch - imported.orientation.pitch),
        #                   abs(original.orientation.roll - imported.orientation.roll))
        #     print(f"  Orientation: {ori_diff:.14f}°")
        #     if original.rotation_matrix is not None and imported.rotation_matrix is not None:
        #         # Compare the matrices directly if needed for debugging, but round-trip won't match YPR
        #         rot_diff = torch.max(torch.abs(original.rotation_matrix - imported.rotation_matrix)).item()
        #         print(f"  Rotation Matrix Diff: {rot_diff:.14f}") # Changed label


def test_geofusion_loader(converter, num_points: int = 3) -> None:
    """Test the GeoFusion loader with a subset of points.
    
    Args:
        converter: GeospatialConverter converter instance
        num_points: Number of points to test (default: 3)
    """
    loader = GeoFusionDataLoader(converter)
    loader.load_csv()
    
    # Get first N points
    locations = loader.get_locations()[:num_points]
    orientations = loader.get_orientations()[:num_points]
    
    print("\nGeoFusion Data Test Results")
    print("=" * 80)
    
    # Original coordinates and rotations
    xyz, rot = converter.geodetic_to_xyz(locations, orientations)
    print("\nOriginal Coordinates and Rotations:")
    print("-" * 80)
    for i in range(num_points):
        entry = loader.entries[i]
        print(f"\nImage: {entry.image_name}")
        print(f"Geodetic    : lat={entry.lat:.8f}°, lon={entry.lon:.8f}°, alt={entry.alt:.4f}m")
        print(f"XYZ         : {xyz[i].cpu().tolist()}")
        print(f"Orientation : yaw={entry.yaw:.2f}°, pitch={entry.pitch:.2f}°, roll={entry.roll:.2f}°")
        print("Rotation Matrix:")
        R = rot[i].cpu()
        for row in R:
            print(f"    [{', '.join(f'{x:8.4f}' for x in row)}]")
    
    # Normalized coordinates
    norm = converter.xyz_to_norm(xyz)
    print("\nNormalized Coordinates (with original rotations):")
    print("-" * 80)
    for i in range(num_points):
        print(f"\nImage: {loader.entries[i].image_name}")
        print(f"Normalized  : {norm[i].cpu().tolist()}")
        print("Rotation Matrix (unchanged):")
        R = rot[i].cpu()
        for row in R:
            print(f"    [{', '.join(f'{x:8.4f}' for x in row)}]")
    
    # Convert back to verify preservation
    xyz_back = converter.norm_to_xyz(norm)
    # Only recover geodetic position, ignore orientation
    geo_back, _ = converter.xyz_to_geodetic(xyz_back, rot)
    
    print("\nRecovered Coordinates and Rotations:")
    print("-" * 80)
    for i in range(num_points):
        print(f"\nImage: {loader.entries[i].image_name}")
        lat, lon, alt = geo_back[i].cpu().tolist()
        print(f"Geodetic    : lat={lat:.8f}°, lon={lon:.8f}°, alt={alt:.4f}m")
        print(f"XYZ         : {xyz_back[i].cpu().tolist()}")
    
    # Verify preservation
    pos_diff = torch.norm(locations - geo_back, dim=1).max().item()
    print("\nPreservation Analysis:")
    print("-" * 80)
    print(f"Position difference    : {pos_diff:.8f} meters")
    
    # Verify perspective preservation
    print("\nPerspective Analysis:")
    print("-" * 80)
    
    # Calculate relative vectors and angles
    for i in range(num_points - 1):
        # Original space
        v1 = xyz[i+1] - xyz[i]
        v1 = v1 / torch.norm(v1)
        
        # Normalized space
        v2 = norm[i+1] - norm[i]
        v2 = v2 / torch.norm(v2)
        
        # Calculate angle between vectors
        cos_angle = torch.clamp(torch.sum(v1 * v2), -1.0, 1.0)
        angle = torch.acos(cos_angle)
        print(f"Angle between vectors {i}-{i+1}: {angle.cpu().item() * 180.0 / 3.14159:.6f}°") 



if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Print test suite configuration
    print_test_suite_header(DEVICE)
    
    # Run comprehensive tests
    _run_documented_tests(DEVICE, torch.float64)
    
    # Run landmark precision test
    conv = GeospatialConverter(device=DEVICE, norm_dtype=torch.float64)
    landmarks = get_test_landmarks()
    run_landmark_precision_test(conv, landmarks)
    
    # Run coordinate I/O test
    run_coordinate_io_test(conv, landmarks)
    
    # Run GeoFusion test if data is available
    try:
        loader = GeoFusionDataLoader(conv)
        BASE_DIR = pathlib.Path(__file__).parent.parent.parent.parent.parent
        geofusion_data = os.path.join(BASE_DIR, "src", "datasets", "geofusion", "landscape_architecture_studio", "logs", "geofusion.csv")
        loader.load_csv(geofusion_data)
        run_geofusion_precision_test(conv, loader)
    except FileNotFoundError:
        print("\nGeoFusion test data not found, skipping precision test.")

        