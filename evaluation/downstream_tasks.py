"""
Evaluation framework for DeepEarth on various downstream tasks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class DownstreamTaskHead(nn.Module):
    """Task-specific head for downstream applications"""
    
    def __init__(self, input_dim: int, task_type: str, output_dim: int):
        super().__init__()
        self.task_type = task_type
        
        if task_type == 'regression':
            self.head = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.LayerNorm(input_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(input_dim // 2, output_dim)
            )
        elif task_type == 'classification':
            self.head = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.LayerNorm(input_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(input_dim // 2, output_dim)
            )
        elif task_type == 'segmentation':
            # For pixel-level predictions
            self.head = nn.Sequential(
                nn.ConvTranspose2d(input_dim, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.GELU(),
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.GELU(),
                nn.Conv2d(128, output_dim, 1)
            )
    
    def forward(self, x):
        return self.head(x)


class DeepEarthEvaluator:
    """Comprehensive evaluation framework for DeepEarth"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Task-specific heads will be added dynamically
        self.task_heads = {}
    
    def add_task(self, task_name: str, task_type: str, output_dim: int):
        """Add a new downstream task"""
        self.task_heads[task_name] = DownstreamTaskHead(
            self.model.config.hidden_dim,
            task_type,
            output_dim
        ).to(self.device)
    
    def extract_features(self, dataloader: DataLoader) -> Dict[str, torch.Tensor]:
        """Extract features using frozen DeepEarth backbone"""
        features = {
            'embeddings': [],
            'spatial_pools': [],
            'modality_tokens': {},
            'metadata': []
        }
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features"):
                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Prepare language input if present
                language_input = None
                if 'input_ids' in batch:
                    language_input = {
                        'input_ids': batch['input_ids'],
                        'attention_mask': batch.get('attention_mask', torch.ones_like(batch['input_ids']))
                    }
                
                # Forward through DeepEarth
                outputs = self.model(
                    xyzt=batch['xyzt'],
                    vision_input=batch.get('images'),
                    language_input=language_input,
                    additional_modalities=batch.get('modalities'),
                    return_intermediates=True
                )
                
                features['embeddings'].append(outputs['all_tokens'].cpu())
                features['spatial_pools'].append(outputs['fused_representation'].cpu())
                
                # Collect modality-specific tokens
                for modality, tokens in outputs.get('modality_tokens', {}).items():
                    if modality not in features['modality_tokens']:
                        features['modality_tokens'][modality] = []
                    features['modality_tokens'][modality].append(tokens.cpu())
                
                features['metadata'].extend(batch.get('metadata', []))
        
        # Concatenate features
        features['embeddings'] = torch.cat(features['embeddings'], dim=0)
        features['spatial_pools'] = torch.cat(features['spatial_pools'], dim=0)
        
        for modality in features['modality_tokens']:
            features['modality_tokens'][modality] = torch.cat(
                features['modality_tokens'][modality], dim=0
            )
        
        return features
    
    def evaluate_task(
        self,
        task_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: int = 10,
        learning_rate: float = 1e-3
    ) -> Dict[str, float]:
        """Fine-tune and evaluate on a specific downstream task"""
        
        if task_name not in self.task_heads:
            raise ValueError(f"Task {task_name} not registered")
        
        task_head = self.task_heads[task_name]
        optimizer = torch.optim.AdamW(task_head.parameters(), lr=learning_rate)
        
        # Training loop
        for epoch in range(num_epochs):
            task_head.train()
            train_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                # Extract features
                with torch.no_grad():
                    # Prepare language input
                    language_input = None
                    if 'input_ids' in batch:
                        language_input = {
                            'input_ids': batch['input_ids'].to(self.device),
                            'attention_mask': batch.get('attention_mask', torch.ones_like(batch['input_ids'])).to(self.device)
                        }
                    
                    outputs = self.model(
                        xyzt=batch['xyzt'].to(self.device),
                        vision_input=batch.get('images', torch.tensor([])).to(self.device) if 'images' in batch else None,
                        language_input=language_input,
                        additional_modalities={k: v.to(self.device) for k, v in batch.get('modalities', {}).items()},
                        return_intermediates=True
                    )
                    features = outputs['fused_representation']
                
                # Task-specific forward
                predictions = task_head(features)
                targets = batch['targets'].to(self.device)
                
                # Compute loss
                if task_head.task_type == 'regression':
                    loss = F.mse_loss(predictions, targets)
                elif task_head.task_type == 'classification':
                    loss = F.cross_entropy(predictions, targets)
                else:
                    loss = F.binary_cross_entropy_with_logits(predictions, targets)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            val_metrics = self._evaluate_metrics(task_head, val_loader, task_name)
            print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, "
                  f"Val Metrics: {val_metrics}")
        
        # Final test evaluation
        test_metrics = self._evaluate_metrics(task_head, test_loader, task_name)
        
        return test_metrics
    
    def _evaluate_metrics(
        self,
        task_head: nn.Module,
        dataloader: DataLoader,
        task_name: str
    ) -> Dict[str, float]:
        """Compute task-specific metrics"""
        task_head.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Extract features
                language_input = None
                if 'input_ids' in batch:
                    language_input = {
                        'input_ids': batch['input_ids'].to(self.device),
                        'attention_mask': batch.get('attention_mask', torch.ones_like(batch['input_ids'])).to(self.device)
                    }
                
                outputs = self.model(
                    xyzt=batch['xyzt'].to(self.device),
                    vision_input=batch.get('images', torch.tensor([])).to(self.device) if 'images' in batch else None,
                    language_input=language_input,
                    additional_modalities={k: v.to(self.device) for k, v in batch.get('modalities', {}).items()},
                    return_intermediates=True
                )
                features = outputs['fused_representation']
                
                # Predictions
                predictions = task_head(features)
                targets = batch['targets']
                
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        # Concatenate
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Compute metrics
        metrics = {}
        
        if task_head.task_type == 'regression':
            predictions_np = all_predictions.numpy()
            targets_np = all_targets.numpy()
            
            metrics['mse'] = mean_squared_error(targets_np, predictions_np)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['r2'] = r2_score(targets_np, predictions_np)
            metrics['mae'] = np.mean(np.abs(targets_np - predictions_np))
            
        elif task_head.task_type == 'classification':
            predictions_np = all_predictions.argmax(dim=1).numpy()
            targets_np = all_targets.numpy()
            
            metrics['accuracy'] = accuracy_score(targets_np, predictions_np)
            metrics['f1_macro'] = f1_score(targets_np, predictions_np, average='macro')
            metrics['f1_weighted'] = f1_score(targets_np, predictions_np, average='weighted')
        
        return metrics
    
    def benchmark_all_tasks(self) -> Dict[str, Dict[str, float]]:
        """Run evaluation on all standard benchmarks"""
        
        results = {}
        
        # Task 1: Temperature Prediction (Regression)
        print("\n=== Temperature Prediction ===")
        self.add_task('temperature_prediction', 'regression', output_dim=1)
        # Load temperature prediction dataset
        # results['temperature'] = self.evaluate_task('temperature_prediction', ...)
        
        # Task 2: Land Cover Classification
        print("\n=== Land Cover Classification ===")
        self.add_task('land_cover', 'classification', output_dim=10)
        # Load land cover dataset
        # results['land_cover'] = self.evaluate_task('land_cover', ...)
        
        # Task 3: Species Distribution Modeling
        print("\n=== Species Distribution ===")
        self.add_task('species_distribution', 'classification', output_dim=100)
        # Load species dataset
        # results['species'] = self.evaluate_task('species_distribution', ...)
        
        # Task 4: Wildfire Risk Prediction
        print("\n=== Wildfire Risk ===")
        self.add_task('wildfire_risk', 'regression', output_dim=1)
        # Load wildfire dataset
        # results['wildfire'] = self.evaluate_task('wildfire_risk', ...)
        
        # Task 5: Crop Yield Prediction
        print("\n=== Crop Yield ===")
        self.add_task('crop_yield', 'regression', output_dim=1)
        # Load crop yield dataset
        # results['crop_yield'] = self.evaluate_task('crop_yield', ...)
        
        return results
    
    def visualize_embeddings(self, features: Dict[str, torch.Tensor], save_path: str = None):
        """Visualize learned embeddings using t-SNE/UMAP"""
        from sklearn.manifold import TSNE
        import umap
        
        embeddings = features['spatial_pools'].numpy()
        
        # t-SNE visualization
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings[:1000])  # Subsample for speed
        
        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)
        plt.title("t-SNE of DeepEarth Spatial Embeddings")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def attention_analysis(self, sample_batch: Dict[str, torch.Tensor]):
        """Analyze attention patterns in the model"""
        # Hook into attention layers
        attention_maps = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                attention_maps[name] = output.detach()
            return hook
        
        # Register hooks on attention layers
        hooks = []
        for name, module in self.model.named_modules():
            if 'attention' in name.lower():
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Forward pass
        with torch.no_grad():
            language_input = None
            if 'input_ids' in sample_batch:
                language_input = {
                    'input_ids': sample_batch['input_ids'].to(self.device),
                    'attention_mask': sample_batch.get('attention_mask', torch.ones_like(sample_batch['input_ids'])).to(self.device)
                }
            
            outputs = self.model(
                xyzt=sample_batch['xyzt'].to(self.device),
                vision_input=sample_batch.get('images', torch.tensor([])).to(self.device) if 'images' in sample_batch else None,
                language_input=language_input,
                additional_modalities={k: v.to(self.device) for k, v in sample_batch.get('modalities', {}).items()},
                return_intermediates=True
            )
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Visualize attention patterns
        for name, attn in attention_maps.items():
            if len(attn.shape) == 4:  # Multi-head attention
                # Average over heads
                attn_avg = attn.mean(dim=1)
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(attn_avg[0].cpu().numpy(), cmap='Blues')
                plt.title(f"Attention Pattern: {name}")
                plt.show()
        
        return attention_maps


class SpatiotemporalMetrics:
    """Specialized metrics for spatiotemporal predictions"""
    
    @staticmethod
    def spatial_autocorrelation(predictions: np.ndarray, coordinates: np.ndarray) -> float:
        """Compute Moran's I for spatial autocorrelation"""
        from pysal.lib import weights
        from pysal.explore import esda
        
        # Create spatial weights
        w = weights.KNN.from_array(coordinates, k=8)
        
        # Compute Moran's I
        moran = esda.Moran(predictions.flatten(), w)
        
        return moran.I
    
    @staticmethod
    def temporal_consistency(predictions: np.ndarray, timestamps: np.ndarray) -> float:
        """Measure temporal smoothness of predictions"""
        # Sort by time
        time_order = np.argsort(timestamps)
        sorted_preds = predictions[time_order]
        
        # Compute temporal differences
        temporal_diffs = np.diff(sorted_preds, axis=0)
        
        # Measure consistency (lower is more consistent)
        consistency = np.mean(np.abs(temporal_diffs))
        
        return consistency
    
    @staticmethod
    def spatiotemporal_rmse(
        predictions: np.ndarray,
        targets: np.ndarray,
        coordinates: np.ndarray,
        timestamps: np.ndarray,
        spatial_bins: int = 10,
        temporal_bins: int = 10
    ) -> Dict[str, float]:
        """Compute RMSE across spatial and temporal bins"""
        results = {}
        
        # Spatial binning
        lat_bins = np.linspace(coordinates[:, 1].min(), coordinates[:, 1].max(), spatial_bins)
        lon_bins = np.linspace(coordinates[:, 0].min(), coordinates[:, 0].max(), spatial_bins)
        
        for i in range(spatial_bins - 1):
            for j in range(spatial_bins - 1):
                mask = (
                    (coordinates[:, 1] >= lat_bins[i]) & 
                    (coordinates[:, 1] < lat_bins[i+1]) &
                    (coordinates[:, 0] >= lon_bins[j]) & 
                    (coordinates[:, 0] < lon_bins[j+1])
                )
                
                if mask.sum() > 0:
                    spatial_rmse = np.sqrt(mean_squared_error(
                        targets[mask],
                        predictions[mask]
                    ))
                    results[f'spatial_bin_{i}_{j}'] = spatial_rmse
        
        # Temporal binning
        time_bins = np.linspace(timestamps.min(), timestamps.max(), temporal_bins)
        
        for i in range(temporal_bins - 1):
            mask = (timestamps >= time_bins[i]) & (timestamps < time_bins[i+1])
            
            if mask.sum() > 0:
                temporal_rmse = np.sqrt(mean_squared_error(
                    targets[mask],
                    predictions[mask]
                ))
                results[f'temporal_bin_{i}'] = temporal_rmse
        
        return results


# Example usage
if __name__ == "__main__":
    from models.deepearth_integrated import create_integrated_deepearth
    
    # Load pretrained model
    model = create_deepearth_model()
    # model.load_state_dict(torch.load('checkpoints/best_model.pt'))
    
    # Create evaluator
    evaluator = DeepEarthEvaluator(model)
    
    # Run benchmark
    results = evaluator.benchmark_all_tasks()
    print("Benchmark Results:", results)
