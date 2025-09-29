# deepearth/core/inference.py
"""
DeepEarth Inference Module
═════════════════════════

Production inference engine for trained DeepEarth models with support for:
- Batch processing
- Multiple query formats (CSV, DataFrame, dict)
- Flexible masking specifications
- Modality-aware predictions
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import json
from tqdm import tqdm

from deepearth.core.config import DeepEarthConfig
from deepearth.core.perceiver import DeepEarthPerceiver
from deepearth.core.preprocessor import DatasetPreprocessor


class DeepEarthInference:
    """
    Inference engine for trained DeepEarth models.
    
    Provides flexible interfaces for:
    - Single and batch predictions
    - Spatiotemporal queries
    - Cross-modal inference
    - Uncertainty quantification
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        compile_model: bool = False
    ):
        """
        Initialize inference engine.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device for inference (default: auto-detect)
            compile_model: Whether to compile model for speed
        """
        print(f"\n{'='*70}")
        print(f"DeepEarth Inference Engine")
        print(f"{'='*70}")
        
        # Auto-detect device if not specified
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = torch.device(device)
        print(f"Device: {self.device}")
        
        # ═══════════════════════════════════════════════════════════
        # Load checkpoint
        # ═══════════════════════════════════════════════════════════
        
        print(f"\nLoading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract configuration
        self.config = checkpoint['config']
        print(f"  Model configuration loaded")
        
        # ═══════════════════════════════════════════════════════════
        # Load metadata
        # ═══════════════════════════════════════════════════════════
        
        checkpoint_dir = Path(checkpoint_path).parent.parent
        metadata_path = checkpoint_dir / 'metadata.json'
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.encoder_configs = metadata.get('encoder_configs', {})
                print(f"  Metadata loaded")
        else:
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        # ═══════════════════════════════════════════════════════════
        # Initialize model
        # ═══════════════════════════════════════════════════════════
        
        print(f"\nInitializing model...")
        self.model = DeepEarthPerceiver(self.config, self.encoder_configs)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Compile if requested
        if compile_model and hasattr(torch, 'compile'):
            print(f"  Compiling model for inference...")
            self.model = torch.compile(self.model, mode='reduce-overhead')
        
        # ═══════════════════════════════════════════════════════════
        # Initialize preprocessor
        # ═══════════════════════════════════════════════════════════
        
        self.preprocessor = DatasetPreprocessor(self.config)
        
        print(f"\n{'='*70}")
        print(f"Inference engine ready")
        print(f"{'='*70}\n")
    
    def query(
        self,
        query_data: Union[str, pd.DataFrame, Dict],
        mask_spec: Optional[Union[str, pd.DataFrame, Dict]] = None,
        batch_size: int = 32,
        return_latents: bool = False
    ) -> Dict:
        """
        Run inference query.
        
        Args:
            query_data: Input data (CSV path, DataFrame, or dict)
            mask_spec: Masking specification (what to predict)
            batch_size: Batch size for processing
            return_latents: Whether to return latent representations
            
        Returns:
            Dictionary with predictions and metadata
        """
        print(f"\n[Inference] Processing query...")
        
        # ───────────────────────────────────────────────────────
        # Process input data
        # ───────────────────────────────────────────────────────
        
        if isinstance(query_data, str):
            print(f"  Loading data from: {query_data}")
            df = pd.read_csv(query_data)
            data = self.preprocessor.process_dataframe(
                df,
                self.preprocessor.detect_columns(df)
            )
        elif isinstance(query_data, pd.DataFrame):
            print(f"  Processing DataFrame ({len(query_data)} rows)")
            data = self.preprocessor.process_dataframe(
                query_data,
                self.preprocessor.detect_columns(query_data)
            )
        elif isinstance(query_data, dict):
            print(f"  Using preprocessed data dictionary")
            data = query_data
        else:
            raise ValueError(f"Unsupported query_data type: {type(query_data)}")
        
        # ───────────────────────────────────────────────────────
        # Process mask specification
        # ───────────────────────────────────────────────────────
        
        masks = self._process_mask_spec(mask_spec, data['n_samples'])
        
        # ───────────────────────────────────────────────────────
        # Run batch inference
        # ───────────────────────────────────────────────────────
        
        all_predictions = []
        all_latents = []
        
        n_samples = data['n_samples']
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        print(f"\n  Running inference on {n_samples:,} samples in {n_batches} batches...")
        
        with torch.no_grad():
            for batch_start in tqdm(range(0, n_samples, batch_size), desc="Inference"):
                batch_end = min(batch_start + batch_size, n_samples)
                
                # Prepare batch
                batch = self._prepare_batch(data, batch_start, batch_end)
                
                # Prepare masks for this batch
                batch_masks = {}
                for key, mask in masks.items():
                    if isinstance(mask, bool):
                        batch_masks[key] = torch.full(
                            (1, batch_end - batch_start),
                            mask,
                            dtype=torch.bool,
                            device=self.device
                        )
                    else:
                        batch_masks[key] = mask[batch_start:batch_end].unsqueeze(0)
                
                # Run model
                outputs = self.model(
                    batch,
                    mask=batch_masks,
                    return_latents=return_latents,
                    inference_mode=True
                )
                
                all_predictions.append(outputs.get('predictions', {}))
                
                if return_latents and 'latents' in outputs:
                    all_latents.append(outputs['latents'])
        
        # ───────────────────────────────────────────────────────
        # Aggregate results
        # ───────────────────────────────────────────────────────
        
        results = self._aggregate_predictions(all_predictions, data)
        
        if return_latents and all_latents:
            results['latents'] = torch.cat(all_latents, dim=0)
        
        # Add metadata
        results['metadata'] = {
            'n_samples': n_samples,
            'dataset_map': data.get('dataset_map', {}),
            'modality_map': data.get('modality_map', {}),
            'encoder_map': data.get('encoder_map', {})
        }
        
        print(f"\n[Inference] Complete")
        
        return results
    
    def predict_at_location(
        self,
        coordinates: Union[Dict, List[Dict]],
        modality: str,
        encoder: str = None,
        dataset: str = 'query',
        return_latents: bool = False
    ) -> Dict:
        """
        Predict observations at specific spatiotemporal locations.
        
        Args:
            coordinates: Single or multiple coordinate dicts
                {'x': float, 'y': float, 'z': float, 't': time}
            modality: Target modality name
            encoder: Encoder name (auto-detect if None)
            dataset: Dataset name
            return_latents: Whether to return latent representations
            
        Returns:
            Predictions dictionary
        """
        # Handle single coordinate
        if isinstance(coordinates, dict):
            coordinates = [coordinates]
        
        print(f"\n[Predict] {len(coordinates)} locations for {modality}")
        
        # Create query DataFrame
        rows = []
        for coord in coordinates:
            row = {
                'x': coord.get('x', coord.get('lat', 0)),
                'y': coord.get('y', coord.get('lon', 0)),
                'z': coord.get('z', coord.get('elev', 0)),
                't': coord.get('t', coord.get('time', 0)),
                'dataset': dataset,
                'modality': modality,
                'encoder': encoder or f'{modality}_encoder',
                'data': 0  # Placeholder
            }
            rows.append(row)
        
        query_df = pd.DataFrame(rows)
        
        # Set mask to predict data
        mask_spec = {
            'spacetime': False,
            'data': True,  # Predict data from location
            'dataset': False,
            'modality': False,
            'encoder': False
        }
        
        # Run inference
        results = self.query(query_df, mask_spec, return_latents=return_latents)
        
        return results
    
    def cross_modal_inference(
        self,
        source_data: Union[str, pd.DataFrame],
        source_modality: str,
        target_modality: str,
        return_latents: bool = False
    ) -> Dict:
        """
        Perform cross-modal inference.
        
        Given observations in one modality, predict corresponding
        observations in another modality at the same spacetime locations.
        
        Args:
            source_data: Source modality observations
            source_modality: Name of source modality
            target_modality: Name of target modality
            return_latents: Whether to return latent representations
            
        Returns:
            Cross-modal predictions
        """
        print(f"\n[Cross-Modal] {source_modality} → {target_modality}")
        
        # Load source data
        if isinstance(source_data, str):
            source_df = pd.read_csv(source_data)
        else:
            source_df = source_data
        
        # Modify to target modality
        target_df = source_df.copy()
        target_df['modality'] = target_modality
        
        # Mask configuration: predict target data
        mask_spec = {
            'spacetime': False,
            'data': True,
            'dataset': False,
            'modality': False,
            'encoder': False
        }
        
        # Run inference
        results = self.query(target_df, mask_spec, return_latents=return_latents)
        
        return results
    
    def _process_mask_spec(
        self,
        mask_spec: Optional[Union[str, pd.DataFrame, Dict]],
        n_samples: int
    ) -> Dict[str, torch.Tensor]:
        """Process mask specification into tensor format."""
        if mask_spec is None:
            # Default: predict data from spacetime
            masks = {
                'spacetime': False,
                'data': True,
                'dataset': False,
                'modality': False,
                'encoder': False
            }
        elif isinstance(mask_spec, str):
            # Load from CSV
            mask_df = pd.read_csv(mask_spec)
            masks = {}
            for component in ['spacetime', 'data', 'dataset', 'modality', 'encoder']:
                if component in mask_df.columns:
                    masks[component] = torch.tensor(
                        mask_df[component].values[:n_samples],
                        dtype=torch.bool,
                        device=self.device
                    )
                else:
                    masks[component] = torch.zeros(n_samples, dtype=torch.bool, device=self.device)
        elif isinstance(mask_spec, pd.DataFrame):
            masks = {}
            for component in ['spacetime', 'data', 'dataset', 'modality', 'encoder']:
                if component in mask_spec.columns:
                    masks[component] = torch.tensor(
                        mask_spec[component].values[:n_samples],
                        dtype=torch.bool,
                        device=self.device
                    )
                else:
                    masks[component] = torch.zeros(n_samples, dtype=torch.bool, device=self.device)
        elif isinstance(mask_spec, dict):
            masks = {}
            for component in ['spacetime', 'data', 'dataset', 'modality', 'encoder']:
                if component in mask_spec:
                    if isinstance(mask_spec[component], bool):
                        masks[component] = mask_spec[component]
                    else:
                        masks[component] = torch.tensor(
                            mask_spec[component],
                            dtype=torch.bool,
                            device=self.device
                        )
                else:
                    masks[component] = False
        else:
            raise ValueError(f"Unsupported mask_spec type: {type(mask_spec)}")
        
        # Report masking configuration
        mask_summary = {k: v.float().mean().item() if torch.is_tensor(v) else float(v) 
                       for k, v in masks.items()}
        print(f"  Mask configuration: {mask_summary}")
        
        return masks
    
    def _prepare_batch(self, data: Dict, start: int, end: int) -> Dict:
        """Prepare batch for inference."""
        batch_size = end - start
        
        batch = {
            'xyzt': data['xyzt'][start:end].unsqueeze(0),
            'dataset_modality_encoder': data['dataset_modality_encoder'][start:end].unsqueeze(0)
        }
        
        # Process encoded data
        encoded = []
        for i in range(start, end):
            encoder_id = data['encoded_file_indices'][i].item()
            row_idx = data['encoded_row_indices'][i].item()
            encoded.append(data['encoded_data'][encoder_id][row_idx])
        
        batch['encoded_data'] = torch.stack(encoded).unsqueeze(0)
        
        # Add modality positions if available
        if 'modality_positions' in data:
            positions = {}
            for i in range(start, end):
                if i in data['modality_positions']:
                    positions[i - start] = data['modality_positions'][i]
            if positions:
                batch['modality_positions'] = positions
        
        # Move to device
        batch = {
            k: v.to(self.device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
        
        return batch
    
    def _aggregate_predictions(
        self,
        predictions_list: List[Dict],
        data: Dict
    ) -> Dict:
        """Aggregate batch predictions."""
        results = {}
        
        # Aggregate spacetime predictions
        if any('spacetime' in p for p in predictions_list):
            spacetime_preds = []
            for p in predictions_list:
                if 'spacetime' in p:
                    spacetime_preds.append(p['spacetime'])
            if spacetime_preds:
                results['spacetime'] = torch.cat(spacetime_preds, dim=0).cpu().numpy()
        
        # Aggregate modality predictions
        if any('data_modality' in p for p in predictions_list):
            modality_preds = {}
            for p in predictions_list:
                if 'data_modality' in p:
                    modality_preds.update(p['data_modality'])
            if modality_preds:
                results['data_modality'] = modality_preds
        
        return results


def main():
    """Command-line interface for DeepEarth inference."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='DeepEarth inference engine',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--checkpoint', required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--query', required=True,
                       help='Query data (CSV path or JSON)')
    parser.add_argument('--mask', default=None,
                       help='Mask specification (CSV or JSON)')
    parser.add_argument('--output', default=None,
                       help='Output file path')
    parser.add_argument('--device', default=None,
                       help='Device (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--compile', action='store_true',
                       help='Compile model for speed')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    engine = DeepEarthInference(
        args.checkpoint,
        device=args.device,
        compile_model=args.compile
    )
    
    # Run query
    results = engine.query(
        args.query,
        args.mask,
        batch_size=args.batch_size
    )
    
    # Save or print results
    if args.output:
        # Determine output format from extension
        output_path = Path(args.output)
        
        if output_path.suffix == '.json':
            with open(output_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                json_results = {}
                for key, value in results.items():
                    if isinstance(value, np.ndarray):
                        json_results[key] = value.tolist()
                    else:
                        json_results[key] = value
                json.dump(json_results, f, indent=2, default=str)
        elif output_path.suffix == '.pt':
            torch.save(results, output_path)
        else:
            # Default to pickle for complex objects
            import pickle
            with open(output_path, 'wb') as f:
                pickle.dump(results, f)
        
        print(f"\nResults saved to: {output_path}")
    else:
        # Print summary
        print("\n" + "="*70)
        print("Inference Results")
        print("="*70)
        
        for key, value in results.items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for k, v in value.items():
                    if isinstance(v, (list, np.ndarray)):
                        print(f"  {k}: shape={getattr(v, 'shape', len(v))}")
                    else:
                        print(f"  {k}: {v}")
            elif isinstance(value, (list, np.ndarray)):
                print(f"{key}: shape={getattr(value, 'shape', len(value))}")
            else:
                print(f"{key}: {value}")


if __name__ == '__main__':
    main()
