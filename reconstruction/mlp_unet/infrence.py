#!/usr/bin/env python3
"""
Fixed inference script that loads species list from checkpoint
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
try:
    from umap import UMAP
    has_umap = True
except ImportError:
    print("Warning: UMAP not installed. Install with: pip install umap-learn")
    has_umap = False
import seaborn as sns
from PIL import Image
import os
from typing import List, Optional

from model import BimodalMLPUNet
from dataset import CentralFloridaPlantsDataModule


class BimodalInference:
    """Inference and visualization for trained MLP U-Net"""
    
    def __init__(self, 
                 checkpoint_path: str,
                 species_list: List[str] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = device
        
        # Load checkpoint first to get species list
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Use species list from checkpoint if not provided
        if species_list is None:
            if 'species_list' in checkpoint:
                species_list = checkpoint['species_list']
                print(f"Loaded species list from checkpoint: {species_list}")
            else:
                raise ValueError("No species list found in checkpoint and none provided")
        
        self.species_list = species_list
        
        # Load model with correct species list
        self.model = BimodalMLPUNet(
            species_list=species_list,
            embedding_dim=2048,
            hidden_dim=512,
            bottleneck_dim=128
        ).to(device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Loaded model from {checkpoint_path}")
        print(f"Model accuracy: {checkpoint.get('accuracy', 'N/A')}")
        
    def predict_species_from_image(self, image_path: str, top_k: int = 3):
        """Predict species from a single image"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Transform
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.model.predict_species(image_tensor, mask_ratio=0.0, top_k=top_k)
            
        return predictions[0]  # Return first batch element
    
    def find_similar_images(self, species_name: str, test_loader, top_k: int = 5):
        """Find images most similar to a given species"""
        if species_name not in self.species_list:
            raise ValueError(f"Species {species_name} not in model vocabulary")
            
        # Get species embedding
        with torch.no_grad():
            species_features = self.model.get_image_features_from_species([species_name])
            
        # Collect all image embeddings from test set
        all_embeddings = []
        all_paths = []
        
        for images, species_names in test_loader:
            images = images.to(self.device)
            with torch.no_grad():
                embeddings = self.model.image_encoder(images)
                all_embeddings.append(embeddings)
                
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        # Compute distances
        distances = torch.cdist(species_features, all_embeddings).squeeze(0)
        
        # Get top-k closest
        _, top_indices = torch.topk(distances, k=top_k, largest=False)
        
        return top_indices.cpu().numpy()
    
    def visualize_embedding_space(self, data_module: CentralFloridaPlantsDataModule, 
                                 method: str = 'tsne', n_samples: int = 200):
        """Visualize the embedding space using UMAP or t-SNE"""
        
        if method == 'umap' and not has_umap:
            print("UMAP not available, falling back to t-SNE")
            method = 'tsne'
        
        # Collect embeddings
        image_embeddings = []
        species_embeddings = []
        labels = []
        
        test_loader = data_module.test_dataloader()
        
        print("Collecting embeddings...")
        with torch.no_grad():
            # Get image embeddings
            sample_count = 0
            for images, species_names in test_loader:
                if sample_count >= n_samples:
                    break
                    
                images = images.to(self.device)
                embeddings = self.model.image_encoder(images)
                
                for i in range(len(species_names)):
                    if sample_count >= n_samples:
                        break
                    image_embeddings.append(embeddings[i].cpu().numpy())
                    labels.append(species_names[i])
                    sample_count += 1
                    
            # Get species embeddings
            for species in self.species_list:
                emb = self.model.species_embeddings.get_embedding([species])
                species_embeddings.append(emb.squeeze(0).cpu().numpy())
                
        # Stack embeddings
        image_embeddings = np.stack(image_embeddings)
        species_embeddings = np.stack(species_embeddings)
        
        print(f"Reducing dimensions using {method}...")
        # Reduce dimensions
        if method == 'umap':
            reducer = UMAP(n_components=2, random_state=42)
        else:  # tsne
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(image_embeddings)-1))
            
        # Fit on combined embeddings
        all_embeddings = np.vstack([image_embeddings, species_embeddings])
        reduced = reducer.fit_transform(all_embeddings)
        
        # Split back
        image_reduced = reduced[:len(image_embeddings)]
        species_reduced = reduced[len(image_embeddings):]
        
        # Plot
        plt.figure(figsize=(12, 10))
        
        # Create color map
        unique_species = list(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_species)))
        color_map = {sp: colors[i] for i, sp in enumerate(unique_species)}
        
        # Plot image embeddings
        for species in unique_species:
            mask = [l == species for l in labels]
            species_points = image_reduced[mask]
            if len(species_points) > 0:
                plt.scatter(species_points[:, 0], species_points[:, 1], 
                           c=[color_map[species]], label=f"{species} (images)", 
                           alpha=0.6, s=30)
            
        # Plot species embeddings
        for i, species in enumerate(self.species_list):
            if species in color_map:
                plt.scatter(species_reduced[i, 0], species_reduced[i, 1], 
                           c=[color_map[species]], marker='*', s=500, 
                           edgecolors='black', linewidth=2,
                           label=f"{species} (embedding)")
            
        plt.title(f"Embedding Space Visualization ({method.upper()})")
        plt.xlabel(f"{method} 1")
        plt.ylabel(f"{method} 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'embedding_space_{method}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def visualize_reconstruction_quality(self, data_module: CentralFloridaPlantsDataModule):
        """Visualize reconstruction quality with different mask ratios"""
        
        # Get a sample batch
        test_loader = data_module.test_dataloader()
        images, species_names = next(iter(test_loader))
        
        # Use only up to 8 images
        num_samples = min(8, len(images))
        images = images[:num_samples].to(self.device)
        species_names = species_names[:num_samples]
        
        mask_ratios = [0.0, 0.3, 0.5, 0.7, 0.9]
        
        fig, axes = plt.subplots(len(mask_ratios), 3, figsize=(12, 15))
        fig.suptitle("Reconstruction Quality vs Mask Ratio", fontsize=16)
        
        with torch.no_grad():
            for i, mask_ratio in enumerate(mask_ratios):
                # Forward pass with masking
                reconstructed, original_emb, masks = self.model.forward_image_to_species(
                    images, mask_ratio
                )
                
                # Get target embeddings
                target_emb = self.model.species_embeddings.get_embedding(species_names)
                
                # Calculate distances
                orig_distances = torch.norm(original_emb - target_emb, dim=1)
                recon_distances = torch.norm(reconstructed - target_emb, dim=1)
                
                # Visualize mask pattern (on embedding)
                mask_pattern = masks[0].cpu().numpy()
                axes[i, 0].imshow(mask_pattern.reshape(64, 32), cmap='binary')
                axes[i, 0].set_title(f"Mask (ratio={mask_ratio})")
                axes[i, 0].axis('off')
                
                # Plot distance comparison
                x = np.arange(len(species_names))
                width = 0.35
                
                axes[i, 1].bar(x - width/2, orig_distances.cpu(), width, 
                              label='Original', alpha=0.8)
                axes[i, 1].bar(x + width/2, recon_distances.cpu(), width, 
                              label='Reconstructed', alpha=0.8)
                axes[i, 1].set_ylabel('L2 Distance')
                axes[i, 1].set_title(f'Distance to Target (mask={mask_ratio})')
                axes[i, 1].legend()
                axes[i, 1].set_xticks(x)
                axes[i, 1].set_xticklabels([s[:10] for s in species_names], rotation=45)
                
                # Predict species and show accuracy
                predictions = self.model.predict_species(images, mask_ratio=mask_ratio, top_k=1)
                correct = sum([pred[0] == true for pred, true in zip(predictions, species_names)])
                accuracy = correct / len(species_names)
                
                axes[i, 2].text(0.5, 0.5, f"Accuracy: {accuracy:.2%}\n{correct}/{len(species_names)} correct", 
                               ha='center', va='center', fontsize=14)
                axes[i, 2].set_title(f"Prediction Accuracy")
                axes[i, 2].axis('off')
                
        plt.tight_layout()
        plt.savefig('reconstruction_quality.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def analyze_species_relationships(self):
        """Analyze relationships between species in embedding space"""
        
        with torch.no_grad():
            # Get all species embeddings
            all_species_emb = self.model.species_embeddings.get_all_embeddings()
            
            # Compute pairwise distances
            distances = torch.cdist(all_species_emb, all_species_emb)
            distances_np = distances.cpu().numpy()
            
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(distances_np, 
                    xticklabels=self.species_list,
                    yticklabels=self.species_list,
                    cmap='viridis_r',
                    annot=True,
                    fmt='.2f')
        plt.title("Species Embedding Distances")
        plt.tight_layout()
        plt.savefig('species_distances.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Find most similar species pairs
        print("\nMost similar species pairs:")
        # Set diagonal to infinity to ignore self-similarity
        np.fill_diagonal(distances_np, np.inf)
        
        num_pairs = min(5, len(self.species_list) * (len(self.species_list) - 1) // 2)
        for _ in range(num_pairs):
            min_idx = np.unravel_index(np.argmin(distances_np), distances_np.shape)
            species1 = self.species_list[min_idx[0]]
            species2 = self.species_list[min_idx[1]]
            distance = distances_np[min_idx]
            print(f"{species1} <-> {species2}: {distance:.4f}")
            distances_np[min_idx] = np.inf  # Mark as processed
            distances_np[min_idx[1], min_idx[0]] = np.inf  # Mark symmetric entry


def main():
    """Example usage"""
    # Check if checkpoint exists
    checkpoint_path = './checkpoints/best.pth'
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please train the model first using quickstart.py or train.py")
        return
    
    # Load checkpoint to get species list
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    species_list = checkpoint.get('species_list', None)
    
    if species_list is None:
        print("Error: No species list found in checkpoint!")
        return
    
    print(f"Found {len(species_list)} species in checkpoint: {species_list}")
    
    # Create inference object with species from checkpoint
    inference = BimodalInference(
        checkpoint_path=checkpoint_path,
        species_list=species_list
    )
    
    # Create data module with same species
    data_module = CentralFloridaPlantsDataModule(
        root_dir='./data/plants',
        species_list=species_list,
        batch_size=32,
        images_per_species=50,
        test_split=0.2
    )
    
    # Create output directory
    os.makedirs('visualizations', exist_ok=True)
    
    print("\n🎨 Generating visualizations...")
    
    # 1. Visualize embedding space with t-SNE
    print("\n1. Creating t-SNE embedding visualization...")
    try:
        inference.visualize_embedding_space(data_module, method='tsne', n_samples=100)
        print("   ✓ t-SNE visualization saved")
    except Exception as e:
        print(f"   ✗ t-SNE visualization failed: {e}")
    
    # 2. Try UMAP if available
    if has_umap:
        print("\n2. Creating UMAP embedding visualization...")
        try:
            inference.visualize_embedding_space(data_module, method='umap', n_samples=100)
            print("   ✓ UMAP visualization saved")
        except Exception as e:
            print(f"   ✗ UMAP visualization failed: {e}")
    
    # 3. Visualize reconstruction quality
    print("\n3. Creating reconstruction quality visualization...")
    try:
        inference.visualize_reconstruction_quality(data_module)
        print("   ✓ Reconstruction quality visualization saved")
    except Exception as e:
        print(f"   ✗ Reconstruction quality visualization failed: {e}")
    
    # 4. Analyze species relationships
    print("\n4. Analyzing species relationships...")
    try:
        inference.analyze_species_relationships()
        print("   ✓ Species relationship heatmap saved")
    except Exception as e:
        print(f"   ✗ Species relationship analysis failed: {e}")
    
    # 5. Test predictions
    print("\n5. Testing predictions on sample images...")
    test_loader = data_module.test_dataloader()
    
    if len(test_loader) > 0:
        images, true_species = next(iter(test_loader))
        
        # Get predictions for first 5 images
        num_test = min(5, len(images))
        predictions = inference.model.predict_species(
            images[:num_test].to(inference.device), 
            mask_ratio=0.0, 
            top_k=3
        )
        
        print("\n📋 Sample Predictions:")
        print("-" * 60)
        for i in range(num_test):
            print(f"\nImage {i+1}:")
            print(f"  True species: {true_species[i]}")
            print(f"  Predictions:")
            for j, pred in enumerate(predictions[i]):
                print(f"    {j+1}. {pred}")
    
    # 6. Performance summary
    print("\n" + "="*60)
    print("📈 Performance Summary")
    print("="*60)
    
    # Run evaluation on full test set
    print("\nEvaluating on full test set...")
    correct_top1 = 0
    correct_top3 = 0
    total = 0
    
    with torch.no_grad():
        for images, species_names in test_loader:
            images = images.to(inference.device)
            predictions = inference.model.predict_species(images, mask_ratio=0.0, top_k=3)
            
            for i in range(len(species_names)):
                if species_names[i] == predictions[i][0]:
                    correct_top1 += 1
                if species_names[i] in predictions[i]:
                    correct_top3 += 1
                total += 1
    
    if total > 0:
        print(f"\nTest Set Results:")
        print(f"  Total samples: {total}")
        print(f"  Top-1 Accuracy: {100*correct_top1/total:.2f}%")
        print(f"  Top-3 Accuracy: {100*correct_top3/total:.2f}%")
    
    # Move visualizations to the visualizations folder
    import shutil
    viz_files = ['embedding_space_tsne.png', 'embedding_space_umap.png', 
                 'reconstruction_quality.png', 'species_distances.png']
    
    for file in viz_files:
        if os.path.exists(file):
            shutil.move(file, f'visualizations/{file}')
    
    print("\n✅ All visualizations completed!")
    print("📁 Visualizations saved to ./visualizations/")


if __name__ == "__main__":
    main()
