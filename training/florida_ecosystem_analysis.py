#!/usr/bin/env python3
"""
Florida Ecosystem Analysis using DeepEarth embeddings

Analyzes plant communities and ecological patterns in Florida
using the learned multimodal embeddings.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import folium
from collections import defaultdict
import json

from deepearth_multimodal_training import MultimodalMaskingModel, DeepEarthDataset


# Florida ecosystem categories based on common species
ECOSYSTEM_INDICATORS = {
    'Wetlands': [
        'Pontederia cordata',      # Pickerelweed
        'Taxodium distichum',      # Bald cypress
        'Sagittaria lancifolia',   # Bulltongue arrowhead
        'Cephalanthus occidentalis' # Buttonbush
    ],
    'Pine Flatwoods': [
        'Pinus palustris',         # Longleaf pine
        'Serenoa repens',          # Saw palmetto
        'Lyonia lucida',           # Fetterbush
        'Carphephorus corymbosus', # Florida paintbrush
        'Lyonia ferruginea'        # Rusty lyonia
    ],
    'Coastal/Maritime': [
        'Sabal palmetto',          # Cabbage palm
        'Phyla nodiflora',         # Turkey tangle fogfruit
        'Ilex cassine',            # Dahoon holly
    ],
    'Hardwood Hammocks': [
        'Magnolia grandiflora',    # Southern magnolia
        'Liquidambar styraciflua', # Sweetgum
        'Acer rubrum',             # Red maple
        'Hamelia patens'           # Firebush
    ],
    'Scrub': [
        'Lyonia lucida',           # Fetterbush
        'Vaccinium myrsinites',    # Shiny blueberry
        'Bejaria racemosa',        # Tarflower
        'Ilex glabra'              # Gallberry
    ]
}


def load_model_and_extract_embeddings(model_path, data_dir, max_samples=None):
    """Load model and extract embeddings for all samples."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model = MultimodalMaskingModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Load dataset
    dataset = DeepEarthDataset(
        data_dir=data_dir,
        max_samples=max_samples,
        cache_embeddings=True
    )
    
    # Extract embeddings
    embeddings = {
        'vision_universal': [],
        'language_universal': [],
        'taxon_names': [],
        'latitudes': [],
        'longitudes': [],
        'gbif_ids': []
    }
    
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    with torch.no_grad():
        for batch in loader:
            vision_emb = batch['vision_embedding'].to(device)
            language_emb = batch['language_embedding'].to(device)
            
            # Get universal embeddings
            vision_universal = model.vision_mlp(vision_emb)
            language_universal = model.language_mlp(language_emb)
            
            embeddings['vision_universal'].append(vision_universal.cpu().numpy())
            embeddings['language_universal'].append(language_universal.cpu().numpy())
            embeddings['taxon_names'].extend(batch['taxon_name'])
            embeddings['latitudes'].extend(batch['latitude'].numpy())
            embeddings['longitudes'].extend(batch['longitude'].numpy())
            embeddings['gbif_ids'].extend(batch['gbif_id'].numpy())
    
    # Concatenate
    embeddings['vision_universal'] = np.vstack(embeddings['vision_universal'])
    embeddings['language_universal'] = np.vstack(embeddings['language_universal'])
    
    return embeddings, dataset


def analyze_ecosystem_clusters(embeddings):
    """Analyze ecosystem clustering based on species composition."""
    # Assign ecosystem labels based on indicator species
    ecosystem_labels = []
    for taxon in embeddings['taxon_names']:
        assigned_ecosystem = 'Other'
        for ecosystem, indicators in ECOSYSTEM_INDICATORS.items():
            if taxon in indicators:
                assigned_ecosystem = ecosystem
                break
        ecosystem_labels.append(assigned_ecosystem)
    
    # Create color map
    ecosystems = list(ECOSYSTEM_INDICATORS.keys()) + ['Other']
    colors = plt.cm.tab10(np.linspace(0, 1, len(ecosystems)))
    color_map = {eco: colors[i] for i, eco in enumerate(ecosystems)}
    
    # PCA visualization
    pca = PCA(n_components=2)
    vision_pca = pca.fit_transform(embeddings['vision_universal'])
    
    plt.figure(figsize=(12, 8))
    
    # Plot by ecosystem
    for ecosystem in ecosystems:
        mask = [label == ecosystem for label in ecosystem_labels]
        if sum(mask) > 0:
            plt.scatter(
                vision_pca[mask, 0], 
                vision_pca[mask, 1],
                c=[color_map[ecosystem]], 
                label=f'{ecosystem} (n={sum(mask)})',
                alpha=0.6,
                s=50
            )
    
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Florida Plant Species Clustering by Ecosystem Type')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('florida_ecosystem_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return ecosystem_labels


def create_geographic_distribution_map(embeddings, ecosystem_labels):
    """Create an interactive map of species distributions."""
    # Create base map centered on Florida
    florida_map = folium.Map(
        location=[27.8, -81.8],  # Center of Florida
        zoom_start=7,
        tiles='OpenStreetMap'
    )
    
    # Color map for ecosystems
    ecosystems = list(set(ecosystem_labels))
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightblue']
    eco_colors = {eco: colors[i % len(colors)] for i, eco in enumerate(ecosystems)}
    
    # Add markers for each observation
    for i in range(len(embeddings['taxon_names'])):
        folium.CircleMarker(
            location=[embeddings['latitudes'][i], embeddings['longitudes'][i]],
            radius=5,
            popup=f"{embeddings['taxon_names'][i]}<br>Ecosystem: {ecosystem_labels[i]}",
            color=eco_colors[ecosystem_labels[i]],
            fill=True,
            fillColor=eco_colors[ecosystem_labels[i]]
        ).add_to(florida_map)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                top: 50px; right: 50px; width: 200px; height: auto;
                background-color: white; z-index:9999; font-size:14px;
                border:2px solid grey; border-radius: 5px; padding: 10px">
    <p style="font-weight: bold;">Ecosystem Types</p>
    '''
    
    for eco, color in eco_colors.items():
        legend_html += f'<p><span style="color: {color};">●</span> {eco}</p>'
    
    legend_html += '</div>'
    florida_map.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    florida_map.save('florida_species_distribution.html')
    print("Saved interactive map to florida_species_distribution.html")


def analyze_species_relationships(embeddings):
    """Analyze relationships between species based on embeddings."""
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Get unique species
    unique_species = list(set(embeddings['taxon_names']))
    species_embeddings = {}
    
    # Average embeddings for each species
    for species in unique_species:
        mask = [t == species for t in embeddings['taxon_names']]
        species_embeddings[species] = embeddings['vision_universal'][mask].mean(axis=0)
    
    # Calculate similarity matrix
    species_list = list(species_embeddings.keys())
    embeddings_matrix = np.vstack([species_embeddings[s] for s in species_list])
    similarity_matrix = cosine_similarity(embeddings_matrix)
    
    # Find most similar species pairs
    similar_pairs = []
    for i in range(len(species_list)):
        for j in range(i+1, len(species_list)):
            similar_pairs.append({
                'species1': species_list[i],
                'species2': species_list[j],
                'similarity': similarity_matrix[i, j]
            })
    
    similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)
    
    print("\nMost Similar Species Pairs (by embedding similarity):")
    for pair in similar_pairs[:10]:
        print(f"  {pair['species1']} ↔ {pair['species2']}: {pair['similarity']:.3f}")
    
    # Create heatmap for top species
    top_species = [s for s in species_list if embeddings['taxon_names'].count(s) >= 50][:20]
    if len(top_species) > 5:
        top_indices = [species_list.index(s) for s in top_species]
        top_similarity = similarity_matrix[np.ix_(top_indices, top_indices)]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            top_similarity,
            xticklabels=top_species,
            yticklabels=top_species,
            cmap='coolwarm',
            center=0.5,
            square=True,
            cbar_kws={'label': 'Cosine Similarity'}
        )
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.title('Species Similarity Matrix (Top Species by Count)')
        plt.tight_layout()
        plt.savefig('species_similarity_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    return similar_pairs


def analyze_habitat_preferences(embeddings, ecosystem_labels):
    """Analyze habitat preferences based on geographic distribution."""
    import numpy as np
    from scipy import stats
    
    # Define Florida regions by latitude/longitude
    regions = {
        'North Florida': lambda lat, lon: lat > 29.5,
        'Central Florida': lambda lat, lon: 27.5 < lat <= 29.5,
        'South Florida': lambda lat, lon: lat <= 27.5,
        'East Coast': lambda lat, lon: lon > -81.5,
        'West Coast': lambda lat, lon: lon < -82.5,
        'Interior': lambda lat, lon: -82.5 <= lon <= -81.5
    }
    
    # Analyze species distribution by region
    species_regions = defaultdict(lambda: defaultdict(int))
    
    for i in range(len(embeddings['taxon_names'])):
        species = embeddings['taxon_names'][i]
        lat, lon = embeddings['latitudes'][i], embeddings['longitudes'][i]
        
        for region, check_func in regions.items():
            if check_func(lat, lon):
                species_regions[species][region] += 1
    
    # Print habitat preferences for common species
    print("\nHabitat Preferences (Top 10 Species):")
    common_species = sorted(set(embeddings['taxon_names']), 
                          key=lambda x: embeddings['taxon_names'].count(x), 
                          reverse=True)[:10]
    
    for species in common_species:
        total_count = embeddings['taxon_names'].count(species)
        print(f"\n{species} (n={total_count}):")
        
        region_prefs = species_regions[species]
        for region, count in sorted(region_prefs.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_count) * 100
            print(f"  {region}: {count} ({percentage:.1f}%)")


def main():
    print("🌴 Florida Ecosystem Analysis using DeepEarth")
    
    # Load model and data
    print("\nLoading model and extracting embeddings...")
    embeddings, dataset = load_model_and_extract_embeddings(
        model_path='models/multimodal_model_best.pth',
        data_dir='../dashboard/huggingface_dataset/hf_download/',
        max_samples=2000  # Limit for faster analysis
    )
    
    print(f"\nAnalyzing {len(embeddings['taxon_names'])} observations")
    print(f"Unique species: {len(set(embeddings['taxon_names']))}")
    
    # Analyze ecosystem clusters
    print("\n1. Analyzing ecosystem clusters...")
    ecosystem_labels = analyze_ecosystem_clusters(embeddings)
    
    # Create geographic distribution map
    print("\n2. Creating geographic distribution map...")
    create_geographic_distribution_map(embeddings, ecosystem_labels)
    
    # Analyze species relationships
    print("\n3. Analyzing species relationships...")
    similar_pairs = analyze_species_relationships(embeddings)
    
    # Analyze habitat preferences
    print("\n4. Analyzing habitat preferences...")
    analyze_habitat_preferences(embeddings, ecosystem_labels)
    
    # Save results
    results = {
        'total_observations': len(embeddings['taxon_names']),
        'unique_species': len(set(embeddings['taxon_names'])),
        'ecosystem_distribution': dict(pd.Series(ecosystem_labels).value_counts()),
        'top_similar_pairs': similar_pairs[:20]
    }
    
    with open('florida_ecosystem_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Analysis complete! Check the generated files:")
    print("  - florida_ecosystem_clusters.png")
    print("  - florida_species_distribution.html")
    print("  - species_similarity_matrix.png")
    print("  - florida_ecosystem_analysis_results.json")


if __name__ == "__main__":
    main()
