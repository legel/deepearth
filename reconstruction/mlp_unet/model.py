import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Dict, Optional, Tuple
from datasets import load_dataset
from collections import defaultdict


class SimpleImageEncoder(nn.Module):
    """Simple CNN encoder for images using pretrained backbone"""
    def __init__(self, embedding_dim=2048, pretrained=True):
        super().__init__()
        # Use ResNet50 as backbone
        self.backbone = models.resnet50(pretrained=pretrained)
        # Replace final FC layer to output desired embedding dimension
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, embedding_dim)
        
    def forward(self, x):
        return self.backbone(x)  # Output: [batch, embedding_dim]


class DeepSeekEmbeddings(nn.Module):
    """DeepSeek embeddings for species descriptions pulled from dataset"""
    def __init__(self, 
                 species_list: List[str],
                 model_name: str = "deepseek-ai/deepseek-coder-1.3b-base",
                 embedding_dim: int = 2048,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 use_dataset_descriptions: bool = True):
        super().__init__()
        
        self.species_list = species_list
        self.embedding_dim = embedding_dim
        self.device = device
        self.use_dataset_descriptions = use_dataset_descriptions
        
        print(f"Loading DeepSeek model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(device)
        
        # Get DeepSeek embedding dimension
        self.deepseek_dim = self.model.config.hidden_size
        
        # Projection layer to match target embedding dimension
        self.projection = nn.Linear(self.deepseek_dim, embedding_dim)
        
        # Create species to index mapping
        self.species_to_idx = {sp: i for i, sp in enumerate(species_list)}
        self.idx_to_species = {i: sp for sp, i in self.species_to_idx.items()}
        
        # Load descriptions from dataset
        self.species_descriptions = self._load_species_descriptions()
        
        # Cache embeddings for efficiency
        self.cached_embeddings = None
        self._cache_embeddings()
        
    def _load_species_descriptions(self) -> Dict[str, str]:
        """Load species descriptions from the HuggingFace dataset"""
        if not self.use_dataset_descriptions:
            # Simple fallback descriptions
            return {species: f"Plant species: {species}" for species in self.species_list}
        
        print("Loading species descriptions from dataset...")
        descriptions = {}
        
        # Load dataset
        dataset = load_dataset("deepearth/central-florida-native-plants", split="train", streaming=True)
        
        # Collect all information for each species
        species_info = defaultdict(lambda: {
            'common_names': set(),
            'descriptions': [],
            'habitats': set(),
            'characteristics': set(),
            'count': 0
        })
        
        # Scan through dataset to collect information
        for item in dataset:
            if item['taxon_name'] in self.species_list:
                info = species_info[item['taxon_name']]
                info['count'] += 1
                
                # Collect various fields from the dataset
                if 'common_name' in item and item['common_name']:
                    info['common_names'].add(item['common_name'])
                
                if 'description' in item and item['description']:
                    info['descriptions'].append(item['description'])
                
                if 'habitat' in item and item['habitat']:
                    info['habitats'].add(item['habitat'])
                
                if 'characteristics' in item and item['characteristics']:
                    if isinstance(item['characteristics'], list):
                        info['characteristics'].update(item['characteristics'])
                    else:
                        info['characteristics'].add(item['characteristics'])
                
                # Stop if we have enough info for all species
                if all(species_info[sp]['count'] > 0 for sp in self.species_list):
                    if all(species_info[sp]['count'] >= 5 for sp in self.species_list):
                        break
        
        # Create rich descriptions from collected information
        for species in self.species_list:
            info = species_info[species]
            
            parts = [f"Scientific name: {species}"]
            
            if info['common_names']:
                parts.append(f"Common names: {', '.join(list(info['common_names'])[:3])}")
            
            if info['descriptions']:
                # Use the first/most common description
                parts.append(f"Description: {info['descriptions'][0]}")
            
            if info['habitats']:
                parts.append(f"Habitat: {', '.join(list(info['habitats'])[:3])}")
            
            if info['characteristics']:
                parts.append(f"Characteristics: {', '.join(list(info['characteristics'])[:5])}")
            
            # Create final description
            if len(parts) > 1:
                descriptions[species] = ". ".join(parts)
            else:
                # Fallback if no info found
                descriptions[species] = f"Plant species: {species}. Native Florida plant."
            
            print(f"Loaded description for {species}: {len(descriptions[species])} chars")
        
        return descriptions
    
    def _cache_embeddings(self):
        """Pre-compute and cache all species embeddings"""
        print("Caching DeepSeek embeddings for all species...")
        
        all_embeddings = []
        with torch.no_grad():
            for species in self.species_list:
                text = self.species_descriptions[species]
                
                # Tokenize
                inputs = self.tokenizer(text, 
                                      return_tensors="pt", 
                                      padding=True, 
                                      truncation=True,
                                      max_length=512).to(self.device)
                
                # Get DeepSeek embeddings
                outputs = self.model(**inputs)
                
                # Use mean pooling over sequence length
                embeddings = outputs.last_hidden_state.mean(dim=1)  # [1, deepseek_dim]
                
                # Project to target dimension
                projected = self.projection(embeddings)  # [1, embedding_dim]
                
                all_embeddings.append(projected)
        
        self.cached_embeddings = torch.cat(all_embeddings, dim=0)  # [num_species, embedding_dim]
        print(f"Cached embeddings shape: {self.cached_embeddings.shape}")
    
    def get_embedding(self, species_names: List[str]) -> torch.Tensor:
        """Get embeddings for a batch of species names"""
        if isinstance(species_names, str):
            species_names = [species_names]
            
        indices = [self.species_to_idx[name] for name in species_names]
        indices_tensor = torch.tensor(indices, dtype=torch.long, device=self.device)
        
        return self.cached_embeddings[indices_tensor]
    
    def get_all_embeddings(self) -> torch.Tensor:
        """Get all species embeddings"""
        return self.cached_embeddings


class MLPUNet(nn.Module):
    """MLP U-Net for bidirectional embedding reconstruction"""
    def __init__(self, embedding_dim=2048, hidden_dim=512, bottleneck_dim=128):
        super().__init__()
        
        # Encoder path (downsampling)
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.ReLU()
        )
        
        # Decoder path (upsampling)
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
    def forward(self, masked_embedding):
        # Pass through encoder to bottleneck
        encoded = self.encoder(masked_embedding)
        # Reconstruct through decoder
        reconstructed = self.decoder(encoded)
        return reconstructed


class BimodalMLPUNet(nn.Module):
    """Complete bimodal system with optional DeepSeek embeddings"""
    def __init__(self, 
                 species_list: List[str],
                 embedding_dim: int = 2048,
                 hidden_dim: int = 512,
                 bottleneck_dim: int = 128,
                 pretrained_encoder: bool = True,
                 use_deepseek: bool = True,
                 deepseek_model: str = "deepseek-ai/deepseek-coder-1.3b-base"):
        super().__init__()
        
        # Image encoder
        self.image_encoder = SimpleImageEncoder(embedding_dim, pretrained_encoder)
        
        # Species embeddings
        if use_deepseek:
            try:
                self.species_embeddings = DeepSeekEmbeddings(
                    species_list=species_list,
                    model_name=deepseek_model,
                    embedding_dim=embedding_dim,
                    use_dataset_descriptions=True
                )
                # Freeze DeepSeek model to save memory
                for param in self.species_embeddings.model.parameters():
                    param.requires_grad = False
                print("✅ Using DeepSeek embeddings")
            except Exception as e:
                print(f"⚠️ Failed to load DeepSeek: {e}")
                print("Falling back to learnable embeddings")
                self.species_embeddings = LearnableSpeciesEmbeddings(species_list, embedding_dim)
        else:
            self.species_embeddings = LearnableSpeciesEmbeddings(species_list, embedding_dim)
            
        # MLP U-Net
        self.mlp_unet = MLPUNet(embedding_dim, hidden_dim, bottleneck_dim)
        
        self.embedding_dim = embedding_dim
        
    def mask_embedding(self, embedding: torch.Tensor, mask_ratio: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly mask portions of the embedding"""
        batch_size, dim = embedding.shape
        
        # Create random mask
        mask = torch.rand(batch_size, dim, device=embedding.device) > mask_ratio
        
        # Apply mask
        masked_embedding = embedding * mask
        
        return masked_embedding, mask
    
    def forward_image_to_species(self, images: torch.Tensor, mask_ratio: float = 0.5):
        """Forward pass: Image → Species embedding"""
        # Encode images
        image_embeddings = self.image_encoder(images)
        
        # Mask embeddings
        masked_embeddings, masks = self.mask_embedding(image_embeddings, mask_ratio)
        
        # Reconstruct through U-Net
        reconstructed = self.mlp_unet(masked_embeddings)
        
        return reconstructed, image_embeddings, masks
    
    def forward_species_to_image(self, species_names: List[str], mask_ratio: float = 0.5):
        """Forward pass: Species → Image embedding"""
        # Get species embeddings
        species_embeddings = self.species_embeddings.get_embedding(species_names)
        
        # Mask embeddings
        masked_embeddings, masks = self.mask_embedding(species_embeddings, mask_ratio)
        
        # Reconstruct through U-Net
        reconstructed = self.mlp_unet(masked_embeddings)
        
        return reconstructed, species_embeddings, masks
    
    def predict_species(self, images: torch.Tensor, mask_ratio: float = 0.0, top_k: int = 1) -> List[List[str]]:
        """Predict species from images"""
        self.eval()
        with torch.no_grad():
            # Get reconstructed embeddings
            reconstructed, _, _ = self.forward_image_to_species(images, mask_ratio)
            
            # Get all species embeddings
            all_species_emb = self.species_embeddings.get_all_embeddings()
            
            # Compute cosine similarities (better for semantic embeddings)
            reconstructed_norm = F.normalize(reconstructed, p=2, dim=1)
            all_species_norm = F.normalize(all_species_emb, p=2, dim=1)
            
            similarities = torch.mm(reconstructed_norm, all_species_norm.t())  # [batch, num_species]
            
            # Get top-k predictions
            _, top_indices = torch.topk(similarities, k=top_k, dim=1)
            
            # Convert indices to species names
            predictions = []
            for batch_idx in range(top_indices.shape[0]):
                batch_predictions = []
                for k in range(top_k):
                    species_idx = top_indices[batch_idx, k].item()
                    species_name = self.species_embeddings.idx_to_species[species_idx]
                    batch_predictions.append(species_name)
                predictions.append(batch_predictions)
                
        return predictions
    
    def get_image_features_from_species(self, species_names: List[str], mask_ratio: float = 0.0) -> torch.Tensor:
        """Get predicted image features from species names"""
        self.eval()
        with torch.no_grad():
            reconstructed, _, _ = self.forward_species_to_image(species_names, mask_ratio)
        return reconstructed


class LearnableSpeciesEmbeddings(nn.Module):
    """Simple learnable embeddings as fallback"""
    def __init__(self, species_list: List[str], embedding_dim: int = 2048):
        super().__init__()
        self.species_list = species_list
        self.embedding_dim = embedding_dim
        
        # Create learnable embeddings
        self.embeddings = nn.Embedding(len(species_list), embedding_dim)
        nn.init.xavier_uniform_(self.embeddings.weight)
        
        # Create mappings
        self.species_to_idx = {sp: i for i, sp in enumerate(species_list)}
        self.idx_to_species = {i: sp for sp, i in self.species_to_idx.items()}
        
    def get_embedding(self, species_names: List[str]) -> torch.Tensor:
        """Get embeddings for species names"""
        if isinstance(species_names, str):
            species_names = [species_names]
            
        indices = [self.species_to_idx[name] for name in species_names]
        indices_tensor = torch.tensor(indices, dtype=torch.long)
        
        if torch.cuda.is_available():
            indices_tensor = indices_tensor.cuda()
            
        return self.embeddings(indices_tensor)
    
    def get_all_embeddings(self) -> torch.Tensor:
        """Get all species embeddings"""
        all_indices = torch.arange(len(self.species_list))
        if torch.cuda.is_available():
            all_indices = all_indices.cuda()
        return self.embeddings(all_indices)
