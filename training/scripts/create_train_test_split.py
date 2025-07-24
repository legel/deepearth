#!/usr/bin/env python3
"""
Spatial and Temporal Train/Test Split Generator

Creates DeepEarth dataset splits with spatial carve-outs and temporal boundaries.
Implements the strategy defined in the ML training roadmap.

Strategy:
- Train: All years except 2025, excluding spatial test regions
- Test: Year 2025 observations + spatial carve-outs from all years
- Spatial carve-outs: 5 regions of 2×2km, ≥15km apart

Usage:
    cd /home/photon/4tb/deepseek/deepearth/dashboard
    python3 ../training/scripts/create_train_test_split.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import sys
from pathlib import Path
from datetime import datetime
from geopy.distance import geodesic
import seaborn as sns

# Add dashboard to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "dashboard"))

from data_cache import UnifiedDataCache
from services.training_data import get_available_observation_ids, create_observation_id


class SpatialTemporalSplitter:
    """
    🗺️ Spatial and Temporal Dataset Splitter
    
    Creates scientifically rigorous train/test splits for biodiversity ML models.
    Ensures no data leakage through spatial and temporal boundaries.
    """
    
    def __init__(self, cache):
        """
        Initialize splitter with data cache.
        
        Args:
            cache: UnifiedDataCache instance
        """
        self.cache = cache
        self.observations = None
        self.test_regions = []
        self.split_config = {}
        
    def load_data(self):
        """Load observations data for split generation."""
        print("📊 Loading observations data...")
        self.observations = self.cache.load_observations()
        
        # Filter to observations with vision embeddings for ML training
        vision_obs = self.observations[self.observations['has_vision'] == True].copy()
        
        print(f"Total observations: {len(self.observations):,}")
        print(f"Observations with vision: {len(vision_obs):,}")
        print(f"Year range: {vision_obs['year'].min()}-{vision_obs['year'].max()}")
        print(f"Species count: {vision_obs['taxon_name'].nunique()}")
        
        self.observations = vision_obs
        return self.observations
    
    def find_spatial_test_regions(self, num_regions=5, min_distance_km=15, region_size_km=2):
        """
        Find spatially separated test regions across the dataset.
        
        Args:
            num_regions: Number of test regions to create
            min_distance_km: Minimum distance between regions
            region_size_km: Size of each square region (side length)
            
        Returns:
            list: Test region configurations
        """
        print(f"\n🎯 Finding {num_regions} spatial test regions (≥{min_distance_km}km apart)")
        
        # Sample candidate observations across the geographic area
        candidate_obs = self.observations.sample(n=min(1000, len(self.observations)), random_state=42)
        
        test_centers = []
        test_regions = []
        
        for i in range(num_regions):
            best_candidate = None
            best_distance = 0
            
            # Find candidate that maximizes minimum distance to existing centers
            for _, obs in candidate_obs.iterrows():
                candidate_pos = (obs['latitude'], obs['longitude'])
                
                if not test_centers:
                    # First region - any candidate is valid
                    min_dist = float('inf')
                else:
                    # Calculate minimum distance to existing centers
                    distances = [geodesic(candidate_pos, center).kilometers for center in test_centers]
                    min_dist = min(distances)
                
                if min_dist > best_distance and min_dist >= min_distance_km:
                    best_distance = min_dist
                    best_candidate = obs
            
            if best_candidate is None:
                print(f"⚠️  Could not find candidate for region {i+1} with {min_distance_km}km separation")
                print(f"   Reducing minimum distance requirement...")
                min_distance_km *= 0.8  # Reduce requirement
                continue
            
            # Define square region around the selected observation
            center_lat = best_candidate['latitude']
            center_lng = best_candidate['longitude']
            
            # Convert km to approximate degrees (rough conversion for Central Florida)
            lat_offset = (region_size_km / 2) / 111.0  # ~111 km per degree latitude
            lng_offset = (region_size_km / 2) / (111.0 * np.cos(np.radians(center_lat)))  # Adjust for longitude
            
            region = {
                'region_id': i + 1,
                'center_gbif_id': int(best_candidate['gbif_id']),
                'center_species': best_candidate['taxon_name'],
                'center_lat': float(center_lat),
                'center_lng': float(center_lng),
                'bounds': {
                    'north': float(center_lat + lat_offset),
                    'south': float(center_lat - lat_offset),
                    'east': float(center_lng + lng_offset),
                    'west': float(center_lng - lng_offset)
                },
                'size_km': region_size_km
            }
            
            test_centers.append((center_lat, center_lng))
            test_regions.append(region)
            
            print(f"  Region {i+1}: {best_candidate['taxon_name']} at ({center_lat:.4f}, {center_lng:.4f})")
            if i > 0:
                print(f"    Distance from nearest: {best_distance:.1f}km")
        
        self.test_regions = test_regions
        return test_regions
    
    def assign_observations_to_splits(self):
        """
        Assign each observation to train or test based on spatial and temporal rules.
        
        Returns:
            dict: Split assignments and statistics
        """
        print(f"\n📋 Assigning observations to train/test splits...")
        
        # Initialize split assignments
        self.observations['split'] = 'train'  # Default to train
        self.observations['split_reason'] = 'temporal_train'  # Default reason
        
        # Temporal test: exclude 2025 from training
        temporal_test_mask = self.observations['year'] == 2025
        self.observations.loc[temporal_test_mask, 'split'] = 'test'
        self.observations.loc[temporal_test_mask, 'split_reason'] = 'temporal_test'
        
        # Spatial test: observations in test regions (any year)
        for region in self.test_regions:
            bounds = region['bounds']
            spatial_mask = (
                (self.observations['latitude'] >= bounds['south']) &
                (self.observations['latitude'] <= bounds['north']) &
                (self.observations['longitude'] >= bounds['west']) &
                (self.observations['longitude'] <= bounds['east'])
            )
            
            self.observations.loc[spatial_mask, 'split'] = 'test'
            self.observations.loc[spatial_mask, 'split_reason'] = f"spatial_test_region_{region['region_id']}"
        
        # Calculate statistics
        train_count = (self.observations['split'] == 'train').sum()
        test_count = (self.observations['split'] == 'test').sum()
        temporal_test_count = (self.observations['split_reason'] == 'temporal_test').sum()
        spatial_test_count = test_count - temporal_test_count
        
        split_stats = {
            'total_observations': len(self.observations),
            'train_count': int(train_count),
            'test_count': int(test_count),
            'train_percentage': float(train_count / len(self.observations) * 100),
            'test_percentage': float(test_count / len(self.observations) * 100),
            'temporal_test_count': int(temporal_test_count),
            'spatial_test_count': int(spatial_test_count),
            'unique_species_train': int(self.observations[self.observations['split'] == 'train']['taxon_name'].nunique()),
            'unique_species_test': int(self.observations[self.observations['split'] == 'test']['taxon_name'].nunique()),
        }
        
        print(f"Split Statistics:")
        print(f"  Train: {split_stats['train_count']:,} observations ({split_stats['train_percentage']:.1f}%)")
        print(f"  Test:  {split_stats['test_count']:,} observations ({split_stats['test_percentage']:.1f}%)")
        print(f"    Temporal test: {split_stats['temporal_test_count']:,}")
        print(f"    Spatial test:  {split_stats['spatial_test_count']:,}")
        print(f"  Species in train: {split_stats['unique_species_train']}")
        print(f"  Species in test:  {split_stats['unique_species_test']}")
        
        return split_stats
    
    def create_observation_id_mappings(self):
        """
        Create OBSERVATION_ID mappings for all observations.
        
        Returns:
            dict: Mappings between OBSERVATION_ID and metadata
        """
        print(f"\n🏷️  Creating OBSERVATION_ID mappings...")
        
        observation_mappings = {}
        
        for _, row in self.observations.iterrows():
            gbif_id = row['gbif_id']
            
            # For now, assume 1 image per observation
            # TODO: Extend when multi-image support is added
            obs_id = create_observation_id(gbif_id, 1)
            
            observation_mappings[obs_id] = {
                'gbif_id': int(gbif_id),
                'taxon_id': int(row['taxon_id']),
                'taxon_name': row['taxon_name'],
                'image_index': 1,
                'split': row['split'],
                'split_reason': row['split_reason'],
                'latitude': float(row['latitude']),
                'longitude': float(row['longitude']),
                'year': int(row['year']) if pd.notna(row['year']) else None,
                'month': int(row['month']) if pd.notna(row['month']) else None,
                'day': int(row['day']) if pd.notna(row['day']) else None
            }
        
        print(f"Created {len(observation_mappings):,} OBSERVATION_ID mappings")
        return observation_mappings
    
    def generate_split_config(self):
        """
        Generate complete dataset split configuration.
        
        Returns:
            dict: Complete configuration for the train/test split
        """
        print(f"\n📄 Generating split configuration...")
        
        # Calculate spatial bounds for training data
        train_obs = self.observations[self.observations['split'] == 'train']
        
        train_bounds = {
            'north': float(train_obs['latitude'].max()),
            'south': float(train_obs['latitude'].min()), 
            'east': float(train_obs['longitude'].max()),
            'west': float(train_obs['longitude'].min())
        }
        
        # Get split statistics
        split_stats = self.assign_observations_to_splits()
        
        # Create observation mappings
        observation_mappings = self.create_observation_id_mappings()
        
        # Generate configuration
        config = {
            'dataset_info': {
                'dataset_name': 'central-florida-native-plants',
                'version': '0.2.0',
                'split_created': datetime.now().isoformat(),
                'description': 'Spatial and temporal train/test split for DeepEarth ML training'
            },
            'split_strategy': {
                'temporal_split': {
                    'train_years': 'all except 2025',
                    'test_years': [2025],
                    'description': 'Exclude 2025 from training to test temporal generalization'
                },
                'spatial_split': {
                    'num_test_regions': len(self.test_regions),
                    'region_size_km': 10.0,
                    'min_distance_km': 15.0,
                    'description': 'Spatial carve-outs for testing geographic generalization'
                }
            },
            'train_spatial_bounds': train_bounds,
            'test_spatial_regions': self.test_regions,
            'split_statistics': split_stats,
            'observation_mappings': observation_mappings
        }
        
        self.split_config = config
        return config
    
    def visualize_split(self, save_path=None):
        """
        Create elegant visualization of the spatial and temporal split.
        
        Args:
            save_path: Path to save visualization plots
        """
        print(f"\n📊 Creating elegant split visualizations...")
        
        # Set elegant style
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 10,
            'font.family': 'sans-serif',
            'axes.linewidth': 0.8,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'grid.alpha': 0.3
        })
        
        # Define consistent colors
        train_color = '#2E86AB'  # Deep blue
        test_color = '#A23B72'   # Deep red/magenta
        region_color = '#F18F01' # Orange for spatial regions
        
        # Better organized layout with proper spacing
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, height_ratios=[0.1, 1.2, 1], width_ratios=[1.5, 1, 1], 
                             hspace=0.3, wspace=0.3)
        
        # Main title
        title_ax = fig.add_subplot(gs[0, :])
        title_ax.axis('off')
        title_ax.text(0.5, 0.5, 'DeepEarth ML Training Dataset Split', 
                     fontsize=22, fontweight='bold', ha='center', va='center', 
                     color='#2c3e50', transform=title_ax.transAxes)
        
        # 1. Spatial distribution (main plot, larger and better positioned)
        ax1 = fig.add_subplot(gs[1, 0])
        train_obs = self.observations[self.observations['split'] == 'train']
        test_obs = self.observations[self.observations['split'] == 'test']
        
        # Plot training points first (background)
        ax1.scatter(train_obs['longitude'], train_obs['latitude'], 
                   c=train_color, alpha=0.3, s=0.5, label=f'Training ({len(train_obs):,})', 
                   rasterized=True)
        
        # Plot test points on top (more prominent)
        ax1.scatter(test_obs['longitude'], test_obs['latitude'], 
                   c=test_color, alpha=0.8, s=2, label=f'Testing ({len(test_obs):,})',
                   edgecolors='white', linewidths=0.1)
        
        # Draw spatial test regions with better styling
        for i, region in enumerate(self.test_regions):
            bounds = region['bounds']
            
            # Semi-transparent fill
            rect_fill = plt.Rectangle((bounds['west'], bounds['south']), 
                                    bounds['east'] - bounds['west'],
                                    bounds['north'] - bounds['south'],
                                    fill=True, facecolor=region_color, alpha=0.2, 
                                    edgecolor='none', zorder=1)
            ax1.add_patch(rect_fill)
            
            # Bold border
            rect_border = plt.Rectangle((bounds['west'], bounds['south']), 
                                      bounds['east'] - bounds['west'],
                                      bounds['north'] - bounds['south'],
                                      fill=False, edgecolor=region_color, 
                                      linewidth=2.5, linestyle='-', zorder=3)
            ax1.add_patch(rect_border)
            
            # Small, unobtrusive region labels positioned outside the region
            center_lng = (bounds['west'] + bounds['east']) / 2
            center_lat = bounds['north'] + 0.005  # Position above the region
            ax1.text(center_lng, center_lat, f"R{region['region_id']}", 
                    fontsize=8, color=region_color, fontweight='bold', 
                    ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                             edgecolor=region_color, linewidth=1, alpha=0.9))
        
        ax1.set_xlabel('Longitude', fontsize=11, fontweight='medium')
        ax1.set_ylabel('Latitude', fontsize=11, fontweight='medium')
        ax1.set_title('Spatial Distribution & Test Regions (10×10 km)', 
                     fontsize=13, fontweight='bold', pad=15)
        ax1.legend(frameon=True, fancybox=True, shadow=True, loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_aspect('equal', adjustable='box')
        
        # 2. Temporal distribution
        ax2 = fig.add_subplot(gs[1, 1])
        year_counts = self.observations.groupby(['year', 'split']).size().unstack(fill_value=0)
        
        # Ensure consistent color order (train=blue, test=red/magenta)
        if 'train' in year_counts.columns and 'test' in year_counts.columns:
            year_counts = year_counts[['train', 'test']]  # Explicit column order
        
        bars = year_counts.plot(kind='bar', ax=ax2, color=[train_color, test_color], 
                               width=0.7, alpha=0.8)
        ax2.set_xlabel('Year', fontsize=11, fontweight='medium')
        ax2.set_ylabel('Observations', fontsize=11, fontweight='medium')
        ax2.set_title('Temporal Split\n(2025 held out)', 
                     fontsize=13, fontweight='bold', pad=15)
        ax2.legend(['Training', 'Testing'], frameon=True, fancybox=True, shadow=True, fontsize=9)
        ax2.tick_params(axis='x', rotation=45, labelsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Fixed species distribution calculation
        ax3 = fig.add_subplot(gs[1, 2])
        species_split = self.observations.groupby(['taxon_name', 'split']).size().unstack(fill_value=0)
        
        # Fix the species counting logic
        train_only_species = set(species_split[species_split['train'] > 0].index) - set(species_split[species_split['test'] > 0].index)
        test_only_species = set(species_split[species_split['test'] > 0].index) - set(species_split[species_split['train'] > 0].index)
        both_splits_species = set(species_split[species_split['train'] > 0].index) & set(species_split[species_split['test'] > 0].index)
        
        coverage_data = pd.DataFrame({
            'Split': ['Training\nOnly', 'Testing\nOnly', 'Both\nSplits'],
            'Species Count': [len(train_only_species), len(test_only_species), len(both_splits_species)]
        })
        
        bars = ax3.bar(coverage_data['Split'], coverage_data['Species Count'], 
                      color=[train_color, test_color, '#34495E'], alpha=0.8)
        ax3.set_ylabel('Species Count', fontsize=11, fontweight='medium')
        ax3.set_title('Species Coverage\nAcross Splits', 
                     fontsize=13, fontweight='bold', pad=15)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.tick_params(axis='x', labelsize=9)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=10)
        
        # 4. Compact, readable summary statistics
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        # Create a more compact, organized summary table
        total_obs = self.split_config['split_statistics']['total_observations']
        train_count = self.split_config['split_statistics']['train_count'] 
        test_count = self.split_config['split_statistics']['test_count']
        train_pct = self.split_config['split_statistics']['train_percentage']
        test_pct = self.split_config['split_statistics']['test_percentage']
        temporal_test = self.split_config['split_statistics']['temporal_test_count']
        spatial_test = self.split_config['split_statistics']['spatial_test_count']
        
        # Organized summary with better layout
        summary_lines = [
            f"📊 DATASET OVERVIEW: {total_obs:,} total observations across {len(both_splits_species) + len(train_only_species) + len(test_only_species)} species",
            "",
            f"🚂 TRAINING SET ({train_pct:.1f}%): {train_count:,} observations • {len(train_only_species) + len(both_splits_species)} species • Years 2010-2024",
            f"🎯 TESTING SET ({test_pct:.1f}%): {test_count:,} observations • {len(test_only_species) + len(both_splits_species)} species",
            f"    └─ Temporal: {temporal_test:,} obs (2025) • Spatial: {spatial_test:,} obs (5 regions of 10×10 km)",
            "",
            f"🗺️  SPATIAL STRATEGY: 5 test regions, 10×10 km each, ≥15 km apart • Geographic generalization evaluation"
        ]
        
        summary_text = '\n'.join(summary_lines)
        
        ax4.text(0.05, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='#f8f9fa', 
                         edgecolor='#dee2e6', linewidth=1, alpha=0.9))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ Elegant visualization saved to {save_path}")
        else:
            plt.show()
        
        return fig
    
    def save_split_config(self, output_path):
        """
        Save split configuration to JSON file.
        
        Args:
            output_path: Path for output configuration file
        """
        print(f"\n💾 Saving split configuration to {output_path}")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.split_config, f, indent=2)
        
        print(f"✅ Split configuration saved successfully")
        print(f"   Total size: {len(json.dumps(self.split_config)) / 1024:.1f} KB")


def main():
    """Main execution for train/test split generation."""
    print("🎯 DeepEarth Spatial-Temporal Train/Test Split Generator")
    print("="*60)
    
    # Initialize cache
    print("Initializing data cache...")
    try:
        cache = UnifiedDataCache("dataset_config.json")
        print("✅ Cache initialized successfully")
    except Exception as e:
        print(f"❌ Cache initialization failed: {e}")
        print("Make sure to run from the dashboard directory!")
        return 1
    
    # Create splitter and load data
    splitter = SpatialTemporalSplitter(cache)
    splitter.load_data()
    
    # Find spatial test regions
    test_regions = splitter.find_spatial_test_regions(
        num_regions=5,
        min_distance_km=15,
        region_size_km=10
    )
    
    # Generate complete split configuration
    split_config = splitter.generate_split_config()
    
    # Create visualization
    viz_path = Path(__file__).parent.parent / "docs" / "train_test_split_visualization.png"
    viz_path.parent.mkdir(parents=True, exist_ok=True)
    splitter.visualize_split(save_path=viz_path)
    
    # Save configuration
    config_path = Path(__file__).parent.parent / "config" / "central_florida_split.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    splitter.save_split_config(config_path)
    
    print(f"\n🎉 Train/Test Split Generation Complete!")
    print(f"Configuration: {config_path}")
    print(f"Visualization: {viz_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())