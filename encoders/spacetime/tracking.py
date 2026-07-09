"""
Collision tracking for Earth4D hash encoder analysis.
"""

import torch
from pathlib import Path
from typing import Dict


def init(model, max_examples: int) -> Dict:
    """Initialize collision tracking tensors."""
    spatial_levels = model.xyz_encoder.num_levels
    temporal_levels = model.xyt_encoder.num_levels

    return {
        'xyz': {'collision_indices': torch.zeros((max_examples, spatial_levels), dtype=torch.int32),
                'max_tracked_examples': max_examples, 'example_offset': 0},
        'xyt': {'collision_indices': torch.zeros((max_examples, temporal_levels), dtype=torch.int32),
                'max_tracked_examples': max_examples, 'example_offset': 0},
        'yzt': {'collision_indices': torch.zeros((max_examples, temporal_levels), dtype=torch.int32),
                'max_tracked_examples': max_examples, 'example_offset': 0},
        'xzt': {'collision_indices': torch.zeros((max_examples, temporal_levels), dtype=torch.int32),
                'max_tracked_examples': max_examples, 'example_offset': 0},
        'coordinates': {
            'original': torch.zeros((max_examples, 4), dtype=torch.float32),
            'normalized': torch.zeros((max_examples, 4), dtype=torch.float32),
            'count': 0
        }
    }


def move(data: Dict, *args, **kwargs) -> Dict:
    """Move collision tracking tensors to specified device."""
    if data is None:
        return data
    for grid_name in ['xyz', 'xyt', 'yzt', 'xzt']:
        if grid_name in data:
            data[grid_name]['collision_indices'] = data[grid_name]['collision_indices'].to(*args, **kwargs)
    if 'coordinates' in data:
        data['coordinates']['original'] = data['coordinates']['original'].to(*args, **kwargs)
        data['coordinates']['normalized'] = data['coordinates']['normalized'].to(*args, **kwargs)
    return data


def save_coords(data: Dict, coords: torch.Tensor, norm_coords: torch.Tensor, max_examples: int) -> int:
    """Save coordinates for tracking during forward pass. Returns example offset."""
    current_count = data['coordinates']['count']
    if current_count >= max_examples:
        return current_count
    batch_size = coords.shape[0]
    save_count = min(batch_size, max_examples - current_count)
    if save_count > 0:
        data['coordinates']['original'][current_count:current_count+save_count] = coords[:save_count]
        data['coordinates']['normalized'][current_count:current_count+save_count] = norm_coords[:save_count]
        data['coordinates']['count'] += save_count
    return current_count


def set_offsets(data: Dict, offset: int):
    """Set example offset for all grids before encoding."""
    for grid_name in ['xyz', 'xyt', 'yzt', 'xzt']:
        data[grid_name]['example_offset'] = offset


def export(data: Dict, encoder, max_examples: int, output_dir: str = "collision_analysis", fmt: str = 'csv') -> Dict:
    """Export collision tracking data for analysis."""
    import pandas as pd
    import json

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tracked_count = data['coordinates']['count']
    if tracked_count == 0:
        raise RuntimeError("No collision data tracked yet")

    print(f"Exporting collision data for {tracked_count} examples...")

    original = data['coordinates']['original'][:tracked_count].cpu().numpy()
    normalized = data['coordinates']['normalized'][:tracked_count].cpu().numpy()

    df_data = {
        'latitude': original[:, 0], 'longitude': original[:, 1],
        'elevation_m': original[:, 2], 'time_original': original[:, 3],
        'x_normalized': normalized[:, 0], 'y_normalized': normalized[:, 1],
        'z_normalized': normalized[:, 2], 'time_normalized': normalized[:, 3]
    }

    for grid_name in ['xyz', 'xyt', 'yzt', 'xzt']:
        indices = data[grid_name]['collision_indices'][:tracked_count].cpu().numpy()
        for level in range(indices.shape[1]):
            df_data[f"{grid_name}_level_{level:02d}_index"] = indices[:, level]

    if fmt == 'pt':
        tensor_data = {
            'coordinates': {'original': torch.from_numpy(original), 'normalized': torch.from_numpy(normalized)},
            'hash_indices': {g: data[g]['collision_indices'][:tracked_count].cpu() for g in ['xyz', 'xyt', 'yzt', 'xzt']}
        }
        data_path = output_path / "collision_data.pt"
        torch.save(tensor_data, data_path)
    else:
        csv_path = output_path / "earth4d_collision_data.csv"
        pd.DataFrame(df_data).to_csv(csv_path, index=False)

    metadata = {
        'spatial_levels': encoder.xyz_encoder.num_levels,
        'temporal_levels': encoder.xyt_encoder.num_levels,
        'tracked_examples': tracked_count,
        'format': fmt
    }
    json_path = output_path / "metadata.json"
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Export complete ({fmt} format)")
    return {'tracked_examples': tracked_count, 'format': fmt, 'output_dir': str(output_path)}
