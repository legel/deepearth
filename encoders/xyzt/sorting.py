"""
Spatiotemporal sorting utilities for improved data locality.

These functions pre-sort data by 4D Morton code to improve cache locality
during hash encoding operations.
"""

import torch
import numpy as np
from typing import Optional, Tuple


def morton_encode_4d(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, t: torch.Tensor,
                     bits_per_dim: int = 16) -> torch.Tensor:
    """
    Compute 4D Morton code (Z-order curve) by interleaving bits from x, y, z, t.

    Morton codes preserve locality: points close in 4D space have similar codes.
    This maximizes warp-level cell sharing across all 4 Earth4D grids (xyz, xyt, yzt, xzt).

    Args:
        x, y, z, t: Quantized coordinate values (should be integers in [0, 2^bits_per_dim))
        bits_per_dim: Bits per dimension (default 16, giving 64-bit output)

    Returns:
        Morton codes as int64 tensor
    """
    # Ensure integer type
    x = x.long()
    y = y.long()
    z = z.long()
    t = t.long()

    # Interleave bits: for each bit position, place x,y,z,t bits consecutively
    # Output bit pattern: t0 z0 y0 x0 t1 z1 y1 x1 t2 z2 y2 x2 ...
    result = torch.zeros_like(x, dtype=torch.int64)

    for i in range(bits_per_dim):
        # Extract bit i from each dimension
        x_bit = (x >> i) & 1
        y_bit = (y >> i) & 1
        z_bit = (z >> i) & 1
        t_bit = (t >> i) & 1

        # Place in interleaved position
        out_pos = i * 4
        result |= (x_bit << out_pos)
        result |= (y_bit << (out_pos + 1))
        result |= (z_bit << (out_pos + 2))
        result |= (t_bit << (out_pos + 3))

    return result


def spatiotemporal_sort_indices(
    coords: torch.Tensor,
    resolution: int = 65536,
    coord_ranges: Optional[Tuple[Tuple[float, float], ...]] = None
) -> torch.Tensor:
    """
    Compute sort indices for spatiotemporal data to improve cache locality.

    Sorts points by 4D Morton code so that spatiotemporally nearby points are
    adjacent in memory. This improves cache efficiency for hash grid lookups.

    Args:
        coords: (N, 4) tensor of [x, y, z, t] coordinates (normalized or raw)
        resolution: Quantization resolution per dimension (default 65536 = 2^16)
        coord_ranges: Optional ((x_min, x_max), (y_min, y_max), (z_min, z_max), (t_min, t_max))
                     If None, ranges are computed from data

    Returns:
        sort_indices: (N,) int64 tensor of indices that sort coords by Morton code

    Example:
        >>> coords = torch.rand(100000, 4)  # x, y, z, t
        >>> sort_idx = spatiotemporal_sort_indices(coords)
        >>> sorted_coords = coords[sort_idx]
        >>> # Now consecutive points are spatiotemporally close
    """
    device = coords.device
    N = coords.shape[0]

    # Compute ranges if not provided
    if coord_ranges is None:
        mins = coords.min(dim=0).values
        maxs = coords.max(dim=0).values
        # Add small epsilon to avoid division by zero
        ranges = maxs - mins + 1e-8
    else:
        mins = torch.tensor([r[0] for r in coord_ranges], device=device)
        maxs = torch.tensor([r[1] for r in coord_ranges], device=device)
        ranges = maxs - mins + 1e-8

    # Normalize to [0, 1] then quantize to [0, resolution-1]
    normalized = (coords - mins) / ranges
    quantized = (normalized * (resolution - 1)).clamp(0, resolution - 1).long()

    # Compute 4D Morton codes
    bits_per_dim = int(np.log2(resolution))
    morton_codes = morton_encode_4d(
        quantized[:, 0], quantized[:, 1],
        quantized[:, 2], quantized[:, 3],
        bits_per_dim=bits_per_dim
    )

    # Sort by Morton code
    sort_indices = torch.argsort(morton_codes)

    return sort_indices


def spatiotemporal_sort(
    coords: torch.Tensor,
    *arrays: torch.Tensor,
    resolution: int = 65536,
    coord_ranges: Optional[Tuple[Tuple[float, float], ...]] = None
) -> Tuple[torch.Tensor, ...]:
    """
    Sort coordinates and associated arrays by spatiotemporal locality.

    Convenience wrapper that sorts coords and any number of associated arrays
    (targets, species indices, etc.) by the same Morton-code ordering.

    Args:
        coords: (N, 4) tensor of [x, y, z, t] coordinates
        *arrays: Additional tensors to sort with the same indices (must have N rows)
        resolution: Quantization resolution (default 65536)
        coord_ranges: Optional coordinate ranges for normalization

    Returns:
        Tuple of (sorted_coords, sorted_array1, sorted_array2, ...)

    Example:
        >>> coords = torch.rand(100000, 4)
        >>> targets = torch.rand(100000)
        >>> species = torch.randint(0, 100, (100000,))
        >>> sorted_coords, sorted_targets, sorted_species = spatiotemporal_sort(
        ...     coords, targets, species
        ... )
    """
    sort_indices = spatiotemporal_sort_indices(coords, resolution, coord_ranges)

    results = [coords[sort_indices]]
    for arr in arrays:
        results.append(arr[sort_indices])

    return tuple(results)


def block_shuffle_indices(
    n_samples: int,
    block_size: int = 1024,
    device: torch.device = None
) -> torch.Tensor:
    """
    Generate indices for block-wise shuffling that preserves local structure.

    Instead of fully random shuffling (which destroys spatial locality),
    this shuffles blocks of consecutive samples. Combined with spatiotemporal
    sorting, this maintains cache locality while providing training randomness.

    Args:
        n_samples: Total number of samples
        block_size: Size of blocks to keep together (default 1024)
        device: Device for output tensor

    Returns:
        Permutation indices that shuffle at block level

    Example:
        >>> # After spatiotemporal sorting, use block shuffle instead of randperm
        >>> sort_idx = spatiotemporal_sort_indices(coords)
        >>> sorted_coords = coords[sort_idx]
        >>>
        >>> # Each epoch, shuffle blocks (not individual samples)
        >>> block_perm = block_shuffle_indices(len(coords), block_size=1024)
        >>> epoch_coords = sorted_coords[block_perm]
    """
    n_blocks = (n_samples + block_size - 1) // block_size

    # Shuffle block order
    block_perm = torch.randperm(n_blocks, device=device)

    # Build full permutation
    indices = []
    for block_idx in block_perm:
        start = block_idx * block_size
        end = min(start + block_size, n_samples)
        indices.append(torch.arange(start, end, device=device))

    return torch.cat(indices)
