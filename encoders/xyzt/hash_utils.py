"""
Hash encoding utilities for Earth4D.

Contains:
- String hash encode/decode (Earth4DHash)
- Prefix tree for YOHO optimization
- Active level detection
"""

import torch
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

# Constants for hash encoding
BASE32_ALPHABET = "0123456789bcdefghjkmnpqrstuvwxyz"
HASH_PRIME_1 = 1
HASH_PRIME_2 = 2654435761
HASH_PRIME_3 = 805459861
BITS_PER_DIMENSION = 48
BITS_PER_LEVEL = 5
MAX_HASH_VALUE = (1 << BITS_PER_DIMENSION) - 1

# Physical constants for coordinate ranges
ELEVATION_MIN_M = 0.0
ELEVATION_MAX_M = 40075000.0  # ~40,075 km
TIME_RANGE_MICROSECONDS = 900 * 365.25 * 24 * 3600 * 1_000_000  # 900 years


def encode_dimension_to_hash(value: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """
    Encode a dimension to 48-bit hierarchical hash.

    Args:
        value: Tensor of dimension values
        min_val: Minimum of range
        max_val: Maximum of range

    Returns:
        Tensor of 48-bit integer hashes (as int64)
    """
    # Normalize to [0, 1)
    normalized = (value - min_val) / (max_val - min_val)
    normalized = torch.clamp(normalized, 0.0, 1.0 - 1e-15)
    # Scale to 48-bit integer
    return (normalized * MAX_HASH_VALUE).long()


def extract_prefix(hash_val: torch.Tensor, level: int) -> torch.Tensor:
    """
    Extract prefix bits at a given level.

    Level 1: top 5 bits (shift by 43)
    Level 2: top 10 bits (shift by 38)
    ...
    Level 10: all 48 bits (shift by 0, but only 48 bits used)

    Args:
        hash_val: 48-bit hash values
        level: Level 1-10

    Returns:
        Prefix values at that level
    """
    shift = BITS_PER_DIMENSION - (level * BITS_PER_LEVEL)
    shift = max(0, shift)  # Level 10 uses all bits
    return hash_val >> shift


def spatial_hash_3tuple(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, table_size: int) -> torch.Tensor:
    """
    Compute spatial hash index for 3-tuple using prime multiplication.

    Args:
        a, b, c: Prefix values at current level
        table_size: Size of hash table (M)

    Returns:
        Hash indices in [0, M)
    """
    combined = (a * HASH_PRIME_1) ^ (b * HASH_PRIME_2) ^ (c * HASH_PRIME_3)
    return combined % table_size


class Earth4DHash:
    """
    Human-readable 4D spatiotemporal hash encoding.

    Format: LLLLLLLLLL-OOOOOOOOOO-EEEEEEEEEE-TTTTTTTTTT (43 chars with dashes)
    Each dimension: 10 base32 characters = 50 bits capacity (48 bits used)

    Example: "t8dg5x2n4k-9jm3p7q1hz-0w8c2e4g6h-5b1d3f7k9m"

    Compatible with GeoHash conventions for interoperability.
    """

    CHARS_PER_DIM = 10
    BITS_PER_CHAR = 5
    BITS_PER_DIM = 48  # Actual precision bits

    def __init__(self, adaptive_range: Optional[Any] = None):
        """
        Initialize hash encoder.

        Args:
            adaptive_range: If provided, use adaptive normalization. Otherwise use global ranges.
        """
        self.adaptive_range = adaptive_range
        self._decode_map = {c: i for i, c in enumerate(BASE32_ALPHABET)}

    def encode(self, lat: float, lon: float, elev: float, time_us: int) -> str:
        """
        Encode coordinates to 43-character hash string.

        Args:
            lat: Latitude in degrees [-90, 90]
            lon: Longitude in degrees [-180, 180]
            elev: Elevation from Earth center in meters
            time_us: Microseconds since 1600-01-01T00:00:00Z

        Returns:
            Hash string: "LLLLLLLLLL-OOOOOOOOOO-EEEEEEEEEE-TTTTTTTTTT"
        """
        if self.adaptive_range:
            lat_range = self.adaptive_range.get_effective_range('lat')
            lon_range = self.adaptive_range.get_effective_range('lon')
            elev_range = self.adaptive_range.get_effective_range('elev')
            time_range = self.adaptive_range.get_effective_range('time')
        else:
            lat_range = (-90.0, 90.0)
            lon_range = (-180.0, 180.0)
            elev_range = (ELEVATION_MIN_M, ELEVATION_MAX_M)
            time_range = (0, TIME_RANGE_MICROSECONDS)

        lat_int = self._encode_dimension(lat, *lat_range)
        lon_int = self._encode_dimension(lon, *lon_range)
        elev_int = self._encode_dimension(elev, *elev_range)
        time_int = self._encode_dimension(float(time_us), *time_range)

        return (f"{self._int_to_base32(lat_int)}-"
                f"{self._int_to_base32(lon_int)}-"
                f"{self._int_to_base32(elev_int)}-"
                f"{self._int_to_base32(time_int)}")

    def decode(self, hash_str: str) -> Tuple[float, float, float, int]:
        """
        Decode hash string back to coordinates.

        Args:
            hash_str: 43-character hash string

        Returns:
            Tuple of (lat, lon, elev, time_us)
        """
        parts = hash_str.split('-')
        if len(parts) != 4:
            raise ValueError(f"Invalid hash format: expected 4 parts separated by '-', got {len(parts)}")

        if self.adaptive_range:
            lat_range = self.adaptive_range.get_effective_range('lat')
            lon_range = self.adaptive_range.get_effective_range('lon')
            elev_range = self.adaptive_range.get_effective_range('elev')
            time_range = self.adaptive_range.get_effective_range('time')
        else:
            lat_range = (-90.0, 90.0)
            lon_range = (-180.0, 180.0)
            elev_range = (ELEVATION_MIN_M, ELEVATION_MAX_M)
            time_range = (0, TIME_RANGE_MICROSECONDS)

        lat = self._decode_dimension(self._base32_to_int(parts[0]), *lat_range)
        lon = self._decode_dimension(self._base32_to_int(parts[1]), *lon_range)
        elev = self._decode_dimension(self._base32_to_int(parts[2]), *elev_range)
        time_us = int(self._decode_dimension(self._base32_to_int(parts[3]), *time_range))

        return lat, lon, elev, time_us

    def truncate(self, hash_str: str, level: int) -> str:
        """
        Truncate hash to specified level (1-10 chars per dimension).

        Args:
            hash_str: Full 43-character hash string
            level: Number of characters to keep per dimension (1-10)

        Returns:
            Truncated hash string
        """
        if not 1 <= level <= 10:
            raise ValueError(f"Level must be 1-10, got {level}")
        parts = hash_str.split('-')
        return '-'.join(p[:level] for p in parts)

    def _encode_dimension(self, value: float, min_val: float, max_val: float) -> int:
        """Encode single dimension to 48-bit integer."""
        if max_val == min_val:
            return 0
        normalized = (value - min_val) / (max_val - min_val)
        normalized = max(0.0, min(1.0 - 1e-15, normalized))
        return int(normalized * ((1 << 48) - 1))

    def _decode_dimension(self, val: int, min_val: float, max_val: float) -> float:
        """Decode 48-bit integer to dimension value."""
        normalized = val / ((1 << 48) - 1)
        return normalized * (max_val - min_val) + min_val

    def _int_to_base32(self, val: int, num_chars: int = 10) -> str:
        """Convert 48-bit integer to 10-character base32 string."""
        val = val << 2  # Left-align 48 bits in 50-bit field
        chars = []
        for _ in range(num_chars):
            chars.append(BASE32_ALPHABET[(val >> 45) & 0x1F])
            val = (val << 5) & ((1 << 50) - 1)
        return ''.join(chars)

    def _base32_to_int(self, s: str) -> int:
        """Convert base32 string to 48-bit integer."""
        val = 0
        for c in s.lower():
            if c not in self._decode_map:
                raise ValueError(f"Invalid base32 character: '{c}'")
            val = (val << 5) | self._decode_map[c]
        return val >> 2  # Right-shift to get 48 bits from 50


class PrefixTreeIndex:
    """
    Index structure for YOHO (You Only Hash Once) optimization.

    For spatiotemporally coherent data, many coordinates share prefixes at coarse levels.
    This class identifies unique prefix tuples and provides inverse mapping for scatter.

    Example: 100K samples from one weather station for one day
    - Level 1-3: All share same prefix -> 1 unique tuple
    - Level 4: ~4 unique tuples (4 time periods)
    - Level 5: ~100 unique tuples
    Speedup: 100K lookups -> ~100 lookups + scatter = 100-1000x faster
    """

    def __init__(self, num_levels: int = 24, bits_per_level: int = 5):
        self.num_levels = num_levels
        self.bits_per_level = bits_per_level
        self.grids = ['xyz', 'xyt', 'yzt', 'xzt']

    def build(
        self,
        coords_normalized: torch.Tensor,  # (N, 4) normalized to [0, 1]
        base_resolution: int,
        growth_factor: float,
        active_levels: Tuple[int, int],  # (min_level, max_level) 1-indexed
        device: torch.device,
    ) -> Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Build prefix tree index for active levels.

        Args:
            coords_normalized: (N, 4) tensor normalized to [0, 1]
            base_resolution: Base grid resolution
            growth_factor: Resolution multiplier between levels
            active_levels: (min_level, max_level) tuple, 1-indexed
            device: Target device for tensors

        Returns:
            Dictionary mapping level -> grid_name -> (unique_tuples, inverse_idx)
            - unique_tuples: (K, 3) tensor of unique cell coordinates
            - inverse_idx: (N,) tensor mapping each input to its unique tuple index
        """
        lat, lon, elev, time = coords_normalized.T  # Each (N,)

        index = {}

        for level in range(active_levels[0], active_levels[1] + 1):
            resolution = int(base_resolution * (growth_factor ** (level - 1)))

            # Quantize to grid cells at this level
            lat_cell = (lat * resolution).floor().long()
            lon_cell = (lon * resolution).floor().long()
            elev_cell = (elev * resolution).floor().long()
            time_cell = (time * resolution).floor().long()

            index[level] = {}

            # Grid xyz: (lat, lon, elev)
            index[level]['xyz'] = self._unique_tuples(
                torch.stack([lat_cell, lon_cell, elev_cell], dim=1), device)

            # Grid xyt: (lat, lon, time)
            index[level]['xyt'] = self._unique_tuples(
                torch.stack([lat_cell, lon_cell, time_cell], dim=1), device)

            # Grid yzt: (lon, elev, time)
            index[level]['yzt'] = self._unique_tuples(
                torch.stack([lon_cell, elev_cell, time_cell], dim=1), device)

            # Grid xzt: (lat, elev, time)
            index[level]['xzt'] = self._unique_tuples(
                torch.stack([lat_cell, elev_cell, time_cell], dim=1), device)

        return index

    def _unique_tuples(
        self,
        tuples: torch.Tensor,  # (N, 3) int64
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find unique 3-tuples and inverse mapping."""
        unique, inverse = torch.unique(tuples, dim=0, return_inverse=True)
        return unique.to(device), inverse.to(device)

    def get_stats(self, index: Dict) -> Dict[int, Dict[str, int]]:
        """
        Get statistics about unique tuples per level/grid.

        Returns:
            Dictionary mapping level -> grid_name -> number of unique tuples
        """
        stats = {}
        for level, grids in index.items():
            stats[level] = {
                grid: unique.shape[0] for grid, (unique, _) in grids.items()
            }
        return stats


@dataclass
class ActiveLevelInfo:
    """Information about which encoder levels are active for current data."""
    total_levels: int
    min_active_level: int
    max_active_level: int
    shared_prefix_levels: int  # Levels where all data shares same cell
    skipped_suffix_levels: int  # Levels beyond max_precision_level

    def __str__(self):
        return (f"Levels {self.min_active_level}-{self.max_active_level} of {self.total_levels} "
                f"({self.shared_prefix_levels} shared prefix, {self.skipped_suffix_levels} skipped suffix)")


def detect_active_levels(
    coords_normalized: torch.Tensor,  # (N, 4) in [0, 1]
    base_resolution: int,
    growth_factor: float,
    total_levels: int,
    max_precision_level: Optional[int] = None,
) -> ActiveLevelInfo:
    """
    Detect which levels have meaningful entropy for the given data.

    Identifies:
    - Shared prefix levels: All coordinates map to same cell (skip - no information)
    - Suffix levels: Beyond max_precision_level (skip - unnecessary precision)

    Args:
        coords_normalized: (N, 4) tensor normalized to [0, 1]
        base_resolution: Base grid resolution
        growth_factor: Resolution multiplier between levels
        total_levels: Total number of levels in encoder
        max_precision_level: Optional cap on finest level

    Returns:
        ActiveLevelInfo with detected level bounds
    """
    if max_precision_level is None:
        max_precision_level = total_levels

    # Find first divergent level (where coordinates differ)
    min_active = 1
    for level in range(1, total_levels + 1):
        resolution = int(base_resolution * (growth_factor ** (level - 1)))
        grid_coords = (coords_normalized * resolution).floor().long()

        # Check if any dimension has more than one unique cell
        has_entropy = False
        for d in range(coords_normalized.shape[1]):
            if grid_coords[:, d].unique().numel() > 1:
                has_entropy = True
                break

        if has_entropy:
            min_active = level
            break

    max_active = min(max_precision_level, total_levels)

    return ActiveLevelInfo(
        total_levels=total_levels,
        min_active_level=min_active,
        max_active_level=max_active,
        shared_prefix_levels=min_active - 1,
        skipped_suffix_levels=total_levels - max_active,
    )
