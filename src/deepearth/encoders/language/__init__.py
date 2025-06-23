"""
Language Encoders - Extract embeddings from text using Large Language Models
"""

from .client import DeepSeekClient
from .umap_processor import compute_3d_umap_and_clusters

__all__ = ['DeepSeekClient', 'compute_3d_umap_and_clusters']