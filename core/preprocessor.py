# deepearth/core/preprocessor.py
"""
DeepEarth Data Preprocessing Pipeline
═════════════════════════════════════

This module transforms raw Earth observation data into optimized tensor
representations for neural processing. The pipeline handles:

1. Coordinate system transformations (geographic → neural-friendly)
2. Temporal decomposition into cyclic components  
3. Modality-specific encoding and metadata extraction
4. Efficient tensor packing and caching

Data Flow:
    CSV/DataFrame → Column Detection → Coordinate Transform
           ↓               ↓                ↓
    Metadata Encoding → Tensor Packing → Cache Storage
           ↓               ↓                ↓
        UMAP Indices → Context Sampling → DataLoader
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import json
import hashlib
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from datetime import datetime
import pyproj
from tqdm import tqdm


class CoordinateTransformer:
    """
    Transform between Earth coordinate systems and neural representations.
    
    Geographic coordinates (latitude, longitude, elevation) are intuitive
    for humans but problematic for neural networks due to discontinuities
    (e.g., longitude wrapping at ±180°). This transformer provides smooth,
    continuous representations suitable for gradient-based learning.
    
    Supported transformations:
    - WGS84 (lat/lon) → ECEF (Earth-Centered, Earth-Fixed)
    - ECEF → Unit sphere (normalized for neural processing)
    - Temporal → Cyclic components (time of day, season, era)
    """
    
    def __init__(self, time_range: Tuple[int, int] = (1900, 2100)):
        """
        Initialize coordinate transformer.
        
        Args:
            time_range: (start_year, end_year) for temporal normalization
        """
        self.time_range = time_range
        
        # Initialize projection systems
        self.wgs84 = pyproj.CRS('EPSG:4326')  # Standard lat/lon
        self.ecef = pyproj.CRS('EPSG:4978')   # Earth-Centered, Earth-Fixed
        self.transformer = pyproj.Transformer.from_crs(
            self.wgs84, self.ecef, always_xy=True
        )
        
        print(f"[CoordinateTransformer] Initialized with time range {time_range}")
    
    def latlon_to_ecef(self, lat: float, lon: float, elev: float) -> Tuple[float, float, float]:
        """
        Convert geographic coordinates to Cartesian ECEF.
        
        ECEF provides a continuous 3D representation without the
        discontinuities present in latitude/longitude systems.
        
        Args:
            lat: Latitude in degrees [-90, 90]
            lon: Longitude in degrees [-180, 180]
            elev: Elevation in meters above sea level
            
        Returns:
            (x, y, z) in ECEF coordinate system
        """
        x, y, z = self.transformer.transform(lon, lat, elev)
        return x, y, z
    
    def normalize_ecef(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """
        Normalize ECEF coordinates to unit sphere.
        
        Projects Earth positions onto a unit sphere centered at Earth's
        center, preserving angular relationships while providing bounded
        inputs for neural networks.
        
        Args:
            x, y, z: ECEF coordinates in meters
            
        Returns:
            (x, y, z) normalized to unit sphere
        """
        # Compute distance from Earth's center
        radius = torch.sqrt(x**2 + y**2 + z**2)
        
        # Avoid division by zero
        radius = torch.clamp(radius, min=1e-8)
        
        # Project to unit sphere
        x_norm = x / radius
        y_norm = y / radius
        z_norm = z / radius
        
        return x_norm, y_norm, z_norm
    
    def normalize_time(self, timestamp: Any) -> float:
        """
        Map timestamp to normalized interval [0, 1].

        Args:
            timestamp: Time as string (ISO format), datetime object, or numeric year

        Returns:
            Normalized time in [0, 1] within configured range
        """
        if isinstance(timestamp, str):
            # Parse ISO format datetime
            dt = datetime.fromisoformat(timestamp)
            # Convert to fractional year
            year = dt.year + dt.timetuple().tm_yday / 365.0
        elif isinstance(timestamp, datetime):
            # Direct datetime object
            year = timestamp.year + timestamp.timetuple().tm_yday / 365.0
        elif isinstance(timestamp, (int, float)):
            year = float(timestamp)
        else:
            raise ValueError(f"Unsupported timestamp type: {type(timestamp)}")
        
        # Linear normalization within configured range
        normalized = (year - self.time_range[0]) / (self.time_range[1] - self.time_range[0])
        
        # Clamp to valid range
        return torch.clamp(torch.tensor(normalized), 0.0, 1.0).item()
    
    def time_to_components(self, timestamp: Any) -> Tuple[float, float, float]:
        """
        Decompose time into cyclic components for better learning.
        
        Neural networks struggle with raw timestamps due to their
        unbounded growth and lack of periodicity. This decomposition
        captures natural cycles in Earth systems.
        
        Args:
            timestamp: Time as string or numeric
            
        Returns:
            (time_of_day, time_of_year, time_of_history) all in [0, 1]
        """
        if isinstance(timestamp, str):
            dt = datetime.fromisoformat(timestamp)
        elif isinstance(timestamp, datetime):
            dt = timestamp
        else:
            # Convert normalized time back to datetime for decomposition
            year = self.time_range[0] + timestamp * (self.time_range[1] - self.time_range[0])
            dt = datetime(int(year), 1, 1)  # Simplified - could be enhanced
        
        # Time of day: captures diurnal cycles (0=midnight, 0.5=noon)
        seconds_since_midnight = dt.hour * 3600 + dt.minute * 60 + dt.second
        time_of_day = seconds_since_midnight / 86400.0
        
        # Time of year: captures seasonal cycles (0=Jan 1, 0.5=July 1)
        day_of_year = dt.timetuple().tm_yday
        time_of_year = day_of_year / 365.0
        
        # Time in history: captures long-term trends
        time_of_history = self.normalize_time(timestamp)
        
        return time_of_day, time_of_year, time_of_history


class DatasetPreprocessor:
    """
    Transform raw CSV data into optimized tensor representations.
    
    This preprocessor handles the complexity of heterogeneous Earth
    observation data, automatically detecting data formats, applying
    appropriate transformations, and caching results for efficiency.
    
    Processing pipeline:
    1. Column detection and semantic mapping
    2. Coordinate system transformations
    3. Metadata encoding (dataset, modality, encoder)
    4. Tensor packing and indexing
    5. Cache management for fast reloading
    """
    
    def __init__(self, config):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: DeepEarthConfig instance
        """
        self.config = config
        self.coord_transformer = CoordinateTransformer(config.time_range)
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Mapping dictionaries for categorical encoding
        self.dataset_map = {}   # dataset_name -> integer_id
        self.modality_map = {}  # modality_name -> integer_id  
        self.encoder_map = {}   # encoder_name -> integer_id
        
        print(f"[DatasetPreprocessor] Initialized with cache at {self.cache_dir}")
    
    def get_cache_hash(self, csv_path: str) -> str:
        """
        Generate unique fingerprint for dataset + configuration.
        
        The cache is invalidated when either the data or relevant
        configuration parameters change, ensuring consistency.
        
        Args:
            csv_path: Path to input CSV file
            
        Returns:
            16-character hash string
        """
        # Hash the file contents
        with open(csv_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        # Include configuration parameters that affect preprocessing
        config_str = json.dumps({
            'file_hash': file_hash,
            'universal_dim': self.config.universal_dim,
            'spacetime_dim': self.config.spacetime_dim,
            'coordinate_system': self.config.coordinate_system,
            'time_range': self.config.time_range,
            'spatial_range': self.config.spatial_range,
            'modalities': {
                name: mod.__dict__ 
                for name, mod in self.config.modalities.items()
            }
        }, sort_keys=True)
        
        # Generate combined hash
        combined_hash = hashlib.md5(config_str.encode()).hexdigest()[:16]
        
        print(f"[DatasetPreprocessor] Cache hash for {csv_path}: {combined_hash}")
        
        return combined_hash
    
    def detect_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Intelligently detect and map DataFrame columns to semantic roles.
        
        This function uses pattern matching to identify spatiotemporal
        coordinates, metadata fields, and data columns without requiring
        strict column naming conventions.
        
        Column detection patterns:
        - Spatial: x/lat/latitude, y/lon/longitude, z/elev/elevation
        - Temporal: t/time/timestamp/date/datetime
        - Metadata: dataset, modality, encoder
        - Data: data_file + data_index OR remaining numeric columns
        
        Args:
            df: Input DataFrame to analyze
            
        Returns:
            Dictionary mapping semantic roles to column names
        """
        print("\n[DatasetPreprocessor] Detecting column mappings...")
        
        columns = {}
        col_names = df.columns.tolist()
        
        # ─────────────────────────────────────────────────────────
        # Detect spatiotemporal coordinates
        # ─────────────────────────────────────────────────────────
        
        coord_patterns = {
            'x': ['x', 'lat', 'latitude'],
            'y': ['y', 'lon', 'longitude', 'lng'],
            'z': ['z', 'elev', 'elevation', 'altitude', 'height', 'depth'],
            't': ['t', 'time', 'timestamp', 'date', 'datetime', 'when']
        }
        
        spatial_dims_found = 0
        
        for coord, patterns in coord_patterns.items():
            for col in col_names:
                if col.lower() in patterns:
                    columns[coord] = col
                    if coord in ['x', 'y', 'z']:
                        spatial_dims_found += 1
                    print(f"  → {coord} coordinate: '{col}'")
                    break
        
        # Determine spatial dimensionality
        if 'x' in columns and 'y' not in columns:
            self.config.spatial_dims = 1
            print(f"  → Detected 1D spatial data")
        elif 'x' in columns and 'y' in columns and 'z' not in columns:
            self.config.spatial_dims = 2
            print(f"  → Detected 2D spatial data")
        else:
            self.config.spatial_dims = 3
            print(f"  → Detected 3D spatial data")
        
        # ─────────────────────────────────────────────────────────
        # Detect metadata columns
        # ─────────────────────────────────────────────────────────
        
        for key in ['dataset', 'modality', 'encoder']:
            for col in col_names:
                if col.lower() == key:
                    columns[key] = col
                    print(f"  → {key}: '{col}'")
                    break
        
        # ─────────────────────────────────────────────────────────
        # Detect data columns (file references or direct values)
        # ─────────────────────────────────────────────────────────
        
        if 'data_file' in col_names:
            columns['data_file'] = 'data_file'
            print(f"  → Data from files: 'data_file'")
            
            if 'data_index' in col_names:
                columns['data_index'] = 'data_index'
                print(f"  → Data index: 'data_index'")
        else:
            # Direct data values in CSV
            used_cols = set(columns.values())
            data_cols = [col for col in col_names if col not in used_cols]
            
            if data_cols:
                columns['data'] = data_cols
                print(f"  → Data columns: {data_cols[:5]}{'...' if len(data_cols) > 5 else ''}")
        
        # ─────────────────────────────────────────────────────────
        # Detect modality-specific position columns
        # ─────────────────────────────────────────────────────────
        
        if 'modality_position' in col_names:
            columns['modality_position'] = 'modality_position'
            print(f"  → Modality positions: 'modality_position'")
        
        # Check for modality-specific position columns
        for modality_name in self.config.modalities.keys():
            pos_col = f"{modality_name}_position"
            if pos_col in col_names:
                if 'modality_positions' not in columns:
                    columns['modality_positions'] = {}
                columns['modality_positions'][modality_name] = pos_col
                print(f"  → {modality_name} positions: '{pos_col}'")
        
        print(f"\n[DatasetPreprocessor] Column detection complete")
        
        return columns
    
    def process_dataframe(self, df: pd.DataFrame, columns: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform DataFrame into structured tensors for neural processing.
        
        This method orchestrates the complete transformation pipeline,
        handling coordinate transformations, categorical encoding, and
        efficient tensor packing. Progress is reported throughout for
        large datasets.
        
        Processing steps:
        1. Extract and transform spatiotemporal coordinates
        2. Encode categorical metadata (dataset, modality, encoder)
        3. Load or extract observation data
        4. Pack into efficient tensor structures
        5. Generate indices for fast lookup
        
        Args:
            df: Input DataFrame with Earth observations
            columns: Column mapping from detect_columns()
            
        Returns:
            Dictionary of preprocessed tensors and metadata
        """
        n_samples = len(df)
        print(f"\n[DatasetPreprocessor] Processing {n_samples:,} observations...")
        
        # ═══════════════════════════════════════════════════════════
        # Initialize accumulators
        # ═══════════════════════════════════════════════════════════
        
        xyzt_list = []              # Spatiotemporal coordinates
        time_components_list = []   # Decomposed time components
        dataset_modality_encoder_list = []  # Metadata indices
        encoded_data = {}           # Grouped by encoder
        encoder_indices = []        # Encoder ID for each sample
        modality_positions = {}     # Position within modality
        
        # ═══════════════════════════════════════════════════════════
        # Process each observation
        # ═══════════════════════════════════════════════════════════
        
        for idx, row in tqdm(df.iterrows(), total=n_samples, 
                            desc="Processing observations", 
                            unit="obs"):
            
            # ───────────────────────────────────────────────────────
            # Extract spatial coordinates
            # ───────────────────────────────────────────────────────
            
            x = float(row[columns['x']]) if 'x' in columns else 0.0
            y = float(row[columns['y']]) if 'y' in columns else 0.0
            z = float(row[columns['z']]) if 'z' in columns else 0.0
            t = row[columns['t']] if 't' in columns else 0.0
            
            # Apply coordinate transformation if needed
            if self.config.coordinate_system == 'ecef' and self.config.spatial_dims == 3:
                # Transform geographic to ECEF
                x_ecef, y_ecef, z_ecef = self.coord_transformer.latlon_to_ecef(x, y, z)
                # Normalize to unit sphere
                x, y, z = self.coord_transformer.normalize_ecef(
                    torch.tensor(x_ecef), 
                    torch.tensor(y_ecef), 
                    torch.tensor(z_ecef)
                )
                x, y, z = x.item(), y.item(), z.item()
            
            # ───────────────────────────────────────────────────────
            # Process temporal coordinate
            # ───────────────────────────────────────────────────────
            
            t_norm = self.coord_transformer.normalize_time(t) if 't' in columns else 0.0
            t_day, t_year, t_hist = self.coord_transformer.time_to_components(t) if 't' in columns else (0, 0, 0)
            
            xyzt_list.append([x, y, z, t_norm])
            time_components_list.append([t_day, t_year, t_hist])
            
            # ───────────────────────────────────────────────────────
            # Process metadata (dataset, modality, encoder)
            # ───────────────────────────────────────────────────────
            
            dataset = str(row[columns['dataset']]) if 'dataset' in columns else 'unknown'
            modality = str(row[columns['modality']]) if 'modality' in columns else 'unknown'
            encoder = str(row[columns['encoder']]) if 'encoder' in columns else 'unknown'
            
            # Create integer mappings for efficient embedding lookup
            if dataset not in self.dataset_map:
                self.dataset_map[dataset] = len(self.dataset_map)
                print(f"  New dataset: '{dataset}' → ID {self.dataset_map[dataset]}")
                
            if modality not in self.modality_map:
                self.modality_map[modality] = len(self.modality_map)
                print(f"  New modality: '{modality}' → ID {self.modality_map[modality]}")
                
            if encoder not in self.encoder_map:
                self.encoder_map[encoder] = len(self.encoder_map)
                print(f"  New encoder: '{encoder}' → ID {self.encoder_map[encoder]}")
            
            dataset_idx = self.dataset_map[dataset]
            modality_idx = self.modality_map[modality]
            encoder_idx = self.encoder_map[encoder]
            
            dataset_modality_encoder_list.append([dataset_idx, modality_idx, encoder_idx])
            encoder_indices.append(encoder_idx)
            
            # ───────────────────────────────────────────────────────
            # Process encoded observation data
            # ───────────────────────────────────────────────────────
            
            if 'data_file' in columns:
                # Load from external file
                data_file = row[columns['data_file']]
                data_idx = int(row[columns['data_index']]) if 'data_index' in columns else 0
                
                # Support for different file formats
                if data_file.endswith('.pt'):
                    # PyTorch tensor file
                    data_tensor = torch.load(data_file)
                    if data_tensor.dim() > 1:
                        data_values = data_tensor[data_idx]
                    else:
                        data_values = data_tensor
                elif data_file.endswith('.npy'):
                    # NumPy array file
                    import numpy as np
                    data_array = np.load(data_file)
                    if data_array.ndim > 1:
                        data_values = torch.from_numpy(data_array[data_idx])
                    else:
                        data_values = torch.from_numpy(data_array)
                else:
                    raise ValueError(f"Unsupported data file format: {data_file}")
                
                # Ensure tensor format
                if not isinstance(data_values, torch.Tensor):
                    data_values = torch.tensor(data_values, dtype=torch.float32)
                    
            else:
                # Direct data values from CSV columns
                data_values = []
                for col in columns['data']:
                    val = row[col]
                    if isinstance(val, str):
                        # Handle space or comma-separated values
                        data_values.extend([float(v) for v in val.replace(',', ' ').split()])
                    else:
                        data_values.append(float(val))
                
                data_values = torch.tensor(data_values, dtype=torch.float32)
            
            # Store by encoder for grouped processing
            if encoder_idx not in encoded_data:
                encoded_data[encoder_idx] = []
            encoded_data[encoder_idx].append(data_values)
            
            # ───────────────────────────────────────────────────────
            # Process modality-specific positions if present
            # ───────────────────────────────────────────────────────
            
            if 'modality_position' in columns and pd.notna(row[columns['modality_position']]):
                pos = row[columns['modality_position']]
                if isinstance(pos, str):
                    pos_values = [float(v) for v in pos.split()]
                else:
                    pos_values = [float(pos)]
                modality_positions[idx] = torch.tensor(pos_values)
            
            # Check modality-specific position columns
            if 'modality_positions' in columns and modality in columns['modality_positions']:
                pos_col = columns['modality_positions'][modality]
                if pd.notna(row[pos_col]):
                    pos = row[pos_col]
                    if isinstance(pos, str):
                        pos_values = [float(v) for v in pos.split()]
                    else:
                        pos_values = [float(pos)]
                    modality_positions[idx] = torch.tensor(pos_values)
            
            # Report progress periodically
            if (idx + 1) % 10000 == 0:
                print(f"  Processed {idx + 1:,} / {n_samples:,} observations")
        
        # ═══════════════════════════════════════════════════════════
        # Convert to tensors
        # ═══════════════════════════════════════════════════════════
        
        print(f"\n[DatasetPreprocessor] Converting to tensors...")
        
        xyzt_tensor = torch.tensor(xyzt_list, dtype=torch.float32)
        time_components_tensor = torch.tensor(time_components_list, dtype=torch.float32)
        dataset_modality_encoder_tensor = torch.tensor(
            dataset_modality_encoder_list, dtype=torch.int16
        )
        
        # ═══════════════════════════════════════════════════════════
        # Pack encoded data by encoder
        # ═══════════════════════════════════════════════════════════
        
        print(f"\n[DatasetPreprocessor] Packing encoded data...")
        
        encoded_tensors = {}
        encoded_file_indices = []
        encoded_row_indices = []
        
        for encoder_idx in sorted(encoded_data.keys()):
            # Stack all data for this encoder
            encoder_data_list = encoded_data[encoder_idx]
            
            # Handle variable dimensions
            max_dim = max(d.numel() for d in encoder_data_list)
            padded_data = []
            
            for data in encoder_data_list:
                if data.numel() < max_dim:
                    # Pad if needed
                    padding = torch.zeros(max_dim - data.numel())
                    padded = torch.cat([data.flatten(), padding])
                else:
                    padded = data.flatten()[:max_dim]
                padded_data.append(padded)
            
            encoded_tensors[encoder_idx] = torch.stack(padded_data)
            
            # Build lookup indices
            for i, row_encoder_idx in enumerate(encoder_indices):
                if row_encoder_idx == encoder_idx:
                    encoded_file_indices.append(encoder_idx)
                    # Count how many samples for this encoder we've seen
                    count = sum(1 for e in encoder_indices[:i] if e == encoder_idx)
                    encoded_row_indices.append(count)
            
            encoder_name = self.encoder_map.get(encoder_idx, f"encoder_{encoder_idx}")
            print(f"  Encoder {encoder_idx} ({encoder_name}): "
                  f"{len(encoder_data_list)} samples, "
                  f"{encoded_tensors[encoder_idx].shape[-1]} dims")
        
        # ═══════════════════════════════════════════════════════════
        # Assemble final result
        # ═══════════════════════════════════════════════════════════
        
        result = {
            'xyzt': xyzt_tensor,
            'time_components': time_components_tensor,
            'dataset_modality_encoder': dataset_modality_encoder_tensor,
            'encoded_data': encoded_tensors,
            'encoded_file_indices': torch.tensor(encoded_file_indices, dtype=torch.int16),
            'encoded_row_indices': torch.tensor(encoded_row_indices, dtype=torch.int64),
            'dataset_map': self.dataset_map,
            'modality_map': self.modality_map,
            'encoder_map': self.encoder_map,
            'n_samples': n_samples,
            'spatial_dims': self.config.spatial_dims
        }
        
        if modality_positions:
            result['modality_positions'] = modality_positions
            print(f"  Modality positions: {len(modality_positions)} samples")
        
        print(f"\n[DatasetPreprocessor] Processing complete!")
        print(f"  Total samples: {n_samples:,}")
        print(f"  Datasets: {len(self.dataset_map)}")
        print(f"  Modalities: {len(self.modality_map)}")
        print(f"  Encoders: {len(self.encoder_map)}")
        
        return result
    
    def save_cache(self, data: Dict[str, Any], cache_dir: Path):
        """
        Save preprocessed data to cache for fast reloading.
        
        Cache structure:
        cache_dir/
        ├── metadata.json          # Maps and configuration
        ├── xyzt.pt               # Spatiotemporal coordinates
        ├── time_components.pt    # Decomposed time
        ├── dataset_modality_encoder.pt  # Metadata indices
        ├── encoded_0.pt          # Data for encoder 0
        ├── encoded_1.pt          # Data for encoder 1
        └── ...
        """
        print(f"\n[DatasetPreprocessor] Saving cache to {cache_dir}")
        
        # Save tensor files
        torch.save(data['xyzt'], cache_dir / 'xyzt.pt')
        torch.save(data['time_components'], cache_dir / 'time_components.pt')
        torch.save(data['dataset_modality_encoder'], cache_dir / 'dataset_modality_encoder.pt')
        torch.save(data['encoded_file_indices'], cache_dir / 'encoded_file_indices.pt')
        torch.save(data['encoded_row_indices'], cache_dir / 'encoded_row_indices.pt')
        
        # Save encoded data by encoder
        for encoder_idx, tensor in data['encoded_data'].items():
            torch.save(tensor, cache_dir / f'encoded_{encoder_idx}.pt')
        
        # Save metadata
        metadata = {
            'dataset_map': data['dataset_map'],
            'modality_map': data['modality_map'],
            'encoder_map': data['encoder_map'],
            'n_samples': data['n_samples'],
            'spatial_dims': data['spatial_dims']
        }
        
        if 'modality_positions' in data:
            torch.save(data['modality_positions'], cache_dir / 'modality_positions.pt')
            metadata['has_modality_positions'] = True
        
        with open(cache_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[DatasetPreprocessor] Cache saved successfully")
    
    def load_cache(self, cache_dir: Path) -> Dict[str, Any]:
        """Load preprocessed data from cache."""
        print(f"\n[DatasetPreprocessor] Loading cache from {cache_dir}")
        
        # Load metadata
        with open(cache_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Load tensors
        result = {
            'xyzt': torch.load(cache_dir / 'xyzt.pt'),
            'time_components': torch.load(cache_dir / 'time_components.pt'),
            'dataset_modality_encoder': torch.load(cache_dir / 'dataset_modality_encoder.pt'),
            'encoded_file_indices': torch.load(cache_dir / 'encoded_file_indices.pt'),
            'encoded_row_indices': torch.load(cache_dir / 'encoded_row_indices.pt'),
            **metadata
        }
        
        # Load encoded data
        result['encoded_data'] = {}
        for encoder_idx in range(len(metadata['encoder_map'])):
            file_path = cache_dir / f'encoded_{encoder_idx}.pt'
            if file_path.exists():
                result['encoded_data'][encoder_idx] = torch.load(file_path)
        
        # Load modality positions if present
        if metadata.get('has_modality_positions'):
            result['modality_positions'] = torch.load(cache_dir / 'modality_positions.pt')
        
        print(f"[DatasetPreprocessor] Cache loaded successfully")
        print(f"  Samples: {result['n_samples']:,}")
        print(f"  Encoders: {len(result['encoded_data'])}")
        
        return result
    
    def process_csv(self, csv_path: str) -> Dict[str, Any]:
        """
        Main entry point for CSV processing with caching.
        
        Checks for existing cache to avoid redundant computation,
        falling back to full processing only when necessary.
        """
        cache_hash = self.get_cache_hash(csv_path)
        cache_subdir = self.cache_dir / cache_hash
        
        # Use cache if available and not regenerating
        if cache_subdir.exists() and not self.config.regenerate_cache:
            print(f"\n[DatasetPreprocessor] Found existing cache")
            return self.load_cache(cache_subdir)
        
        print(f"\n[DatasetPreprocessor] No cache found, processing CSV")
        cache_subdir.mkdir(parents=True, exist_ok=True)
        
        # Full processing pipeline
        df = pd.read_csv(csv_path)
        print(f"[DatasetPreprocessor] Loaded {len(df):,} rows from {csv_path}")
        
        columns = self.detect_columns(df)
        processed_data = self.process_dataframe(df, columns)
        self.save_cache(processed_data, cache_subdir)
        
        return processed_data
