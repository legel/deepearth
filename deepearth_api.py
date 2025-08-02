"""
Simple, Clean API for DeepEarth
One-line data source registration and inference
"""

import torch
import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any
from pathlib import Path
import rasterio
import pandas as pd
import xarray as xr

from core.data_registry import create_deepearth_with_registry


class DeepEarth:
    """
    Simple API for DeepEarth model
    
    Example:
        >>> # Initialize
        >>> earth = DeepEarth()
        >>> 
        >>> # Register data sources (one line each!)
        >>> earth.register("soil_ph", shape=(1,), type="value")
        >>> earth.register("landsat", shape=(11, 64, 64), type="satellite")
        >>> earth.register("temperature", shape=(24,), type="timeseries")
        >>> 
        >>> # Predict
        >>> result = earth.predict(
        ...     location=(37.7749, -122.4194),  # San Francisco
        ...     time="2024-01-15",
        ...     data={
        ...         "soil_ph": 6.5,
        ...         "landsat": landsat_image,
        ...         "temperature": temp_history
        ...     }
        ... )
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        """Initialize DeepEarth model"""
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🌍 Initializing DeepEarth on {self.device}")
        
        # Create model with registry
        self.model = create_deepearth_with_registry()
        
        # Load pretrained weights if provided
        if model_path:
            self.load(model_path)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Quick type mappings
        self.type_mappings = {
            'value': 'vector',
            'values': 'vector',
            'satellite': 'raster',
            'image': 'image',
            'raster': 'raster',
            'timeseries': 'time_series',
            'series': 'time_series',
            'vector': 'vector',
            'table': 'vector'
        }
        
        print("✓ DeepEarth ready!")
    
    def register(
        self,
        name: str,
        shape: Union[tuple, int],
        type: str = "auto",
        **kwargs
    ) -> None:
        """
        Register a new data source (one line!)
        
        Args:
            name: Name for this data source
            shape: Data shape (single int for scalars)
            type: Data type hint ('value', 'satellite', 'timeseries', etc.)
            **kwargs: Additional options
            
        Examples:
            >>> earth.register("elevation", shape=1, type="value")
            >>> earth.register("sentinel2", shape=(13, 64, 64), type="satellite")
            >>> earth.register("weather", shape=(168, 5), type="timeseries")
        """
        # Handle single values
        if isinstance(shape, int):
            shape = (shape,)
        
        # Map friendly type names
        input_type = self.type_mappings.get(type.lower(), 'vector')
        
        # Smart defaults based on shape
        if type == "auto":
            if len(shape) == 1:
                input_type = 'vector'
            elif len(shape) == 2:
                input_type = 'time_series' if shape[0] > shape[1] else 'vector'
            elif len(shape) == 3:
                input_type = 'raster'
        
        # Register with model
        self.model.data_registry.register_data_source(
            name=name,
            input_type=input_type,
            input_shape=shape,
            **kwargs
        )
    
    def predict(
        self,
        location: Union[Tuple[float, float], np.ndarray],
        time: Union[str, float, np.datetime64] = None,
        data: Dict[str, Any] = None,
        return_tokens: bool = False
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Make prediction for a location and time
        
        Args:
            location: (lat, lon) tuple or (x, y, z) array
            time: Time as string, float (0-1), or datetime
            data: Dict of data source names to values
            return_tokens: Return all tokens instead of just fused representation
            
        Returns:
            Prediction embedding or dict of embeddings
        """
        # Prepare coordinates
        xyzt = self._prepare_coordinates(location, time)
        
        # Prepare data inputs
        inputs = {'xyzt': xyzt}
        
        if data:
            # Convert data to tensors
            tensor_data = {}
            for name, value in data.items():
                tensor_data[name] = self._to_tensor(value, name)
            
            inputs['additional_modalities'] = tensor_data
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs, return_intermediates=return_tokens)
        
        # Return results
        if return_tokens:
            results = {
                'fused': outputs['fused_representation'].cpu().numpy(),
                'tokens': {
                    name: tokens.cpu().numpy()
                    for name, tokens in outputs.get('modality_tokens', {}).items()
                }
            }
            return results
        else:
            return outputs['fused_representation'].cpu().numpy()
    
    def predict_batch(
        self,
        locations: List[Tuple[float, float]],
        times: List[Any],
        data: Dict[str, List[Any]]
    ) -> np.ndarray:
        """Batch prediction for multiple locations/times"""
        batch_size = len(locations)
        
        # Prepare coordinates
        xyzt_list = [
            self._prepare_coordinates(loc, time)
            for loc, time in zip(locations, times)
        ]
        xyzt = torch.cat(xyzt_list, dim=0)
        
        # Prepare batch data
        inputs = {'xyzt': xyzt}
        
        if data:
            tensor_data = {}
            for name, values in data.items():
                tensors = [self._to_tensor(v, name) for v in values]
                tensor_data[name] = torch.stack(tensors)
            
            inputs['additional_modalities'] = tensor_data
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return outputs['fused_representation'].cpu().numpy()
    
    def load_from_file(self, name: str, filepath: str) -> Any:
        """
        Load data from file based on registered type
        
        Supports:
        - GeoTIFF/raster files (.tif, .tiff)
        - CSV files (.csv)
        - NetCDF files (.nc)
        - NumPy files (.npy, .npz)
        """
        filepath = Path(filepath)
        
        if filepath.suffix in ['.tif', '.tiff']:
            # Load raster
            with rasterio.open(filepath) as src:
                data = src.read()
                return data
        
        elif filepath.suffix == '.csv':
            # Load CSV
            df = pd.read_csv(filepath)
            return df.values
        
        elif filepath.suffix == '.nc':
            # Load NetCDF
            ds = xr.open_dataset(filepath)
            return ds.to_array().values
        
        elif filepath.suffix in ['.npy', '.npz']:
            # Load NumPy
            return np.load(filepath)
        
        else:
            raise ValueError(f"Unsupported file type: {filepath.suffix}")
    
    def _prepare_coordinates(
        self,
        location: Union[Tuple[float, float], np.ndarray],
        time: Any
    ) -> torch.Tensor:
        """Prepare spatiotemporal coordinates"""
        # Handle location
        if isinstance(location, (tuple, list)):
            lat, lon = location
            x = lon / 180.0  # Normalize
            y = lat / 90.0
            z = 0.0  # Sea level
        else:
            x, y, z = location[:3]
            x = x / 180.0
            y = y / 90.0
        
        # Handle time
        if time is None:
            t = 0.5  # Default to middle of range
        elif isinstance(time, str):
            # Parse datetime string (simplified)
            t = 0.5  # Would implement proper parsing
        elif isinstance(time, (int, float)):
            t = float(time)
        else:
            t = 0.5
        
        return torch.tensor([[x, y, z, t]], dtype=torch.float32, device=self.device)
    
    def _to_tensor(self, value: Any, name: str) -> torch.Tensor:
        """Convert various input types to tensor"""
        # Get expected shape
        if name in self.model.data_registry.data_sources:
            expected_shape = self.model.data_registry.data_sources[name].input_shape
        else:
            expected_shape = None
        
        # Handle different input types
        if isinstance(value, torch.Tensor):
            tensor = value
        elif isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value).float()
        elif isinstance(value, (int, float)):
            tensor = torch.tensor([value], dtype=torch.float32)
        elif isinstance(value, list):
            tensor = torch.tensor(value, dtype=torch.float32)
        else:
            raise ValueError(f"Cannot convert {type(value)} to tensor")
        
        # Add batch dimension if needed
        if expected_shape and len(tensor.shape) == len(expected_shape):
            tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
    
    def save(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'data_sources': self.model.data_registry.data_sources
        }, path)
        print(f"✓ Model saved to {path}")
    
    def load(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Model loaded from {path}")
    
    def list_sources(self) -> List[str]:
        """List all registered data sources"""
        return self.model.data_registry.list_sources()
    
    def info(self, name: str) -> Dict[str, Any]:
        """Get info about a data source"""
        return self.model.data_registry.get_info(name)


# Even simpler functional API
_global_model = None

def init(model_path: Optional[str] = None) -> DeepEarth:
    """Initialize global DeepEarth model"""
    global _global_model
    _global_model = DeepEarth(model_path)
    return _global_model

def register(name: str, shape: Union[tuple, int], type: str =
