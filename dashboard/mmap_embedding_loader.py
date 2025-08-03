#!/usr/bin/env python3
"""
Memory-Mapped Embedding Loader for DeepEarth

High-performance loader for vision embeddings using memory-mapped files.
Provides sub-100ms access to 6.4M-dimensional V-JEPA-2 embeddings.

This loader uses:
- Memory-mapped binary files for O(1) access
- SQLite for fast GBIF ID indexing
- Thread-local file handles for concurrent access
- LRU caching for frequently accessed embeddings

Author: DeepEarth Project
License: MIT
"""

import numpy as np
import torch
from pathlib import Path
import sqlite3
import logging
import time
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Any
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)


class MMapEmbeddingLoader:
    """
    High-performance loader for memory-mapped vision embeddings.
    
    This class provides extremely fast access to large vision embeddings
    by using memory-mapped files and SQLite indexing. Thread-safe for
    concurrent web requests.
    """
    
    def __init__(self, embeddings_file: str = "embeddings.mmap", 
                 index_db: str = "embeddings_index.db",
                 cache_size: int = 500):
        """
        Initialize the memory-mapped embedding loader.
        
        Args:
            embeddings_file: Path to memory-mapped binary file
            index_db: Path to SQLite index database
            cache_size: Number of embeddings to cache in memory
        """
        self.embeddings_file = Path(embeddings_file)
        self.index_db = Path(index_db)
        self.cache_size = cache_size
        
        # Embedding specifications (V-JEPA-2)
        self.embedding_size = 6488064  # 8 × 24 × 24 × 1408
        self.embedding_dtype = np.float32
        self.bytes_per_embedding = self.embedding_size * 4  # 4 bytes per float32
        
        # Thread-local storage for file handles
        self._thread_local = threading.local()
        
        # Statistics
        self.stats = defaultdict(int)
        self._lock = threading.Lock()
        
        # Verify files exist
        if not self.embeddings_file.exists():
            raise FileNotFoundError(f"Embeddings file not found: {self.embeddings_file}")
        if not self.index_db.exists():
            raise FileNotFoundError(f"Index database not found: {self.index_db}")
        
        # Initialize cache
        self._init_cache()
        
        logger.info(f"Initialized MMapEmbeddingLoader:")
        logger.info(f"  Embeddings: {self.embeddings_file} ({self.embeddings_file.stat().st_size / 1024**3:.1f} GB)")
        logger.info(f"  Index DB: {self.index_db}")
        logger.info(f"  Cache size: {cache_size} embeddings")
        
    def _init_cache(self):
        """Initialize LRU cache for embeddings"""
        @lru_cache(maxsize=self.cache_size)
        def _cached_load(gbif_id: int) -> Optional[np.ndarray]:
            return self._load_embedding_uncached(gbif_id)
        
        self._cached_load = _cached_load
    
    def _get_thread_resources(self) -> Tuple[Any, sqlite3.Connection]:
        """
        Get thread-local file handle and database connection.
        
        Returns:
            Tuple of (mmap_file, db_connection)
        """
        if not hasattr(self._thread_local, 'mmap_file'):
            # Open memory-mapped file for this thread
            self._thread_local.mmap_file = np.memmap(
                self.embeddings_file, 
                dtype=self.embedding_dtype, 
                mode='r',
                shape=(self.embeddings_file.stat().st_size // 4,)  # Total float32 elements
            )
            
            # Open database connection for this thread
            self._thread_local.db_conn = sqlite3.connect(
                self.index_db,
                check_same_thread=False
            )
            # Enable query optimization
            self._thread_local.db_conn.execute("PRAGMA query_only = ON")
            self._thread_local.db_conn.execute("PRAGMA temp_store = MEMORY")
            
        return self._thread_local.mmap_file, self._thread_local.db_conn
    
    def _load_embedding_uncached(self, gbif_id: int) -> Optional[np.ndarray]:
        """
        Load embedding from memory-mapped file (uncached).
        
        Args:
            gbif_id: GBIF identifier
            
        Returns:
            numpy array or None if not found
        """
        start_time = time.time()
        
        try:
            # Get thread-local resources
            mmap_file, db_conn = self._get_thread_resources()
            
            # Query file offset from SQLite
            cursor = db_conn.cursor()
            cursor.execute(
                "SELECT file_offset FROM embedding_index WHERE gbif_id = ?", 
                (gbif_id,)
            )
            result = cursor.fetchone()
            
            if result is None:
                with self._lock:
                    self.stats['misses'] += 1
                return None
            
            file_offset = result[0]
            
            # Calculate array indices
            start_idx = file_offset // 4  # Convert byte offset to float32 index
            end_idx = start_idx + self.embedding_size
            
            # Extract embedding from memory-mapped file
            embedding = mmap_file[start_idx:end_idx].copy()
            
            # Update statistics
            retrieval_time = (time.time() - start_time) * 1000  # ms
            with self._lock:
                self.stats['hits'] += 1
                self.stats['total_time_ms'] += retrieval_time
                self.stats['retrievals'] += 1
            
            return embedding
            
        except Exception as e:
            with self._lock:
                self.stats['errors'] += 1
                # Rate-limit error logging to prevent flooding
                if self.stats['errors'] <= 5:
                    logger.error(f"Error loading embedding for GBIF {gbif_id}: {e}")
                elif self.stats['errors'] == 6:
                    logger.error(f"Multiple embedding errors detected ({self.stats['errors']} total), suppressing further error logs...")
            return None
    
    def get_vision_embedding(self, gbif_id: int) -> Optional[torch.Tensor]:
        """
        Get vision embedding as PyTorch tensor.
        
        Args:
            gbif_id: GBIF identifier
            
        Returns:
            PyTorch tensor [6488064] or None if not found
        """
        # Try cache first
        embedding = self._cached_load(gbif_id)
        
        if embedding is not None:
            # Convert to PyTorch tensor
            return torch.from_numpy(embedding).float()
        
        return None
    
    def get_vision_embeddings_batch(self, gbif_ids: List[int]) -> Dict[int, torch.Tensor]:
        """
        Get multiple vision embeddings efficiently.
        
        This method is optimized for batch retrieval by:
        1. Single database query for all IDs
        2. Reading from mmap in offset order (better OS page cache usage)
        3. Minimal memory copies using views instead of full copies
        
        Args:
            gbif_ids: List of GBIF identifiers
            
        Returns:
            Dict mapping GBIF ID to embedding tensor
        """
        start_time = time.time()
        results = {}
        
        if not gbif_ids:
            return results
        
        # Get thread-local resources
        mmap_file, db_conn = self._get_thread_resources()
        
        # Single database query for all IDs
        placeholders = ','.join(['?'] * len(gbif_ids))
        cursor = db_conn.cursor()
        cursor.execute(
            f"SELECT gbif_id, file_offset FROM embedding_index "
            f"WHERE gbif_id IN ({placeholders}) ORDER BY file_offset",
            gbif_ids
        )
        
        # Load embeddings in offset order for better cache performance
        offset_data = cursor.fetchall()
        
        for gbif_id, file_offset in offset_data:
            try:
                start_idx = file_offset // 4
                end_idx = start_idx + self.embedding_size
                
                # Use view instead of copy for performance - convert to tensor directly
                embedding_view = mmap_file[start_idx:end_idx]
                results[gbif_id] = torch.from_numpy(embedding_view).float().clone()
                
            except Exception as e:
                logger.error(f"Error loading batch embedding for GBIF {gbif_id}: {e}")
                continue
        
        # Update statistics
        batch_time = (time.time() - start_time) * 1000  # ms
        with self._lock:
            self.stats['batch_retrievals'] += 1
            self.stats['batch_total_time_ms'] += batch_time
            self.stats['batch_total_embeddings'] += len(results)
        
        return results
    
    def get_observation_metadata(self, gbif_id: int) -> Optional[Dict[str, Any]]:
        """
        Get observation metadata from index.
        
        Args:
            gbif_id: GBIF identifier
            
        Returns:
            Dict with metadata or None if not found
        """
        _, db_conn = self._get_thread_resources()
        
        cursor = db_conn.cursor()
        cursor.execute(
            """SELECT gbif_id, taxon_id, taxon_name, latitude, longitude, 
                      year, month, day, hour, has_vision, split, file_idx
               FROM embedding_index WHERE gbif_id = ?""",
            (gbif_id,)
        )
        
        result = cursor.fetchone()
        if result is None:
            return None
        
        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, result))
    
    def search_by_region(self, north: float, south: float, 
                        east: float, west: float, 
                        limit: int = 1000) -> List[int]:
        """
        Find observations within geographic bounds.
        
        Args:
            north, south, east, west: Geographic bounds
            limit: Maximum results
            
        Returns:
            List of GBIF IDs
        """
        _, db_conn = self._get_thread_resources()
        
        cursor = db_conn.cursor()
        cursor.execute(
            """SELECT gbif_id FROM embedding_index 
               WHERE latitude BETWEEN ? AND ?
                 AND longitude BETWEEN ? AND ?
               LIMIT ?""",
            (south, north, west, east, limit)
        )
        
        return [row[0] for row in cursor.fetchall()]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache and performance statistics.
        
        Returns:
            Dict with cache statistics
        """
        cache_info = self._cached_load.cache_info()
        
        with self._lock:
            avg_time = (self.stats['total_time_ms'] / self.stats['retrievals'] 
                       if self.stats['retrievals'] > 0 else 0)
            
            batch_avg_time = (self.stats['batch_total_time_ms'] / self.stats['batch_total_embeddings']
                            if self.stats['batch_total_embeddings'] > 0 else 0)
            
            return {
                'cache_size': cache_info.maxsize,
                'cache_hits': cache_info.hits,
                'cache_misses': cache_info.misses,
                'cache_hit_rate': (cache_info.hits / (cache_info.hits + cache_info.misses) 
                                 if (cache_info.hits + cache_info.misses) > 0 else 0),
                'db_hits': self.stats['hits'],
                'db_misses': self.stats['misses'],
                'errors': self.stats['errors'],
                'avg_retrieval_time_ms': avg_time,
                'batch_avg_time_per_embedding_ms': batch_avg_time,
                'total_retrievals': self.stats['retrievals']
            }
    
    def get_performance_stats(self) -> dict:
        """Get current performance statistics."""
        with self._lock:
            total_requests = self.stats['retrievals']
            error_rate = (self.stats['errors'] / total_requests * 100) if total_requests > 0 else 0
            avg_time = (self.stats['total_time_ms'] / self.stats['hits']) if self.stats['hits'] > 0 else 0
            
            return {
                'total_requests': total_requests,
                'successful_hits': self.stats['hits'],
                'errors': self.stats['errors'],
                'error_rate_percent': error_rate,
                'average_time_ms': avg_time,
                'cache_hits': self.stats.get('cache_hits', 0)
            }
    
    def log_performance_summary(self):
        """Log a performance summary to help with debugging."""
        stats = self.get_performance_stats()
        if stats['total_requests'] > 0:
            logger.info(f"Embedding loader stats: {stats['successful_hits']}/{stats['total_requests']} successful "
                       f"({stats['error_rate_percent']:.1f}% error rate), "
                       f"avg: {stats['average_time_ms']:.1f}ms")
    
    def close(self):
        """Clean up resources"""
        if hasattr(self._thread_local, 'db_conn'):
            self._thread_local.db_conn.close()
        if hasattr(self._thread_local, 'mmap_file'):
            del self._thread_local.mmap_file


class SafeMMapCompatibilityLoader(MMapEmbeddingLoader):
    """
    Compatibility wrapper matching the original interface.
    
    This class maintains backward compatibility with existing code
    while using the new optimized loader internally.
    """
    
    def __init__(self, embeddings_file: str = "embeddings.mmap",
                 index_db: str = "embeddings_index.db"):
        """Initialize with default parameters matching original"""
        super().__init__(embeddings_file, index_db, cache_size=500)
        logger.info("Initialized SafeMMapCompatibilityLoader (compatibility mode)")


# Convenience function for testing
def test_loader():
    """Test the memory-mapped loader with sample queries"""
    loader = MMapEmbeddingLoader()
    
    # Test single retrieval
    print("Testing single embedding retrieval...")
    embedding = loader.get_vision_embedding(1052754321)
    if embedding is not None:
        print(f"✅ Retrieved embedding: shape={embedding.shape}, dtype={embedding.dtype}")
    else:
        print("❌ Embedding not found")
    
    # Test batch retrieval
    print("\nTesting batch retrieval...")
    gbif_ids = [1052754321, 1052754322, 1052754323]
    embeddings = loader.get_vision_embeddings_batch(gbif_ids)
    print(f"✅ Retrieved {len(embeddings)} embeddings")
    
    # Show statistics
    print("\nPerformance statistics:")
    stats = loader.get_cache_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    loader.close()


if __name__ == "__main__":
    test_loader()