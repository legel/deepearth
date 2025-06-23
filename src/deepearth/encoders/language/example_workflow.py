#!/usr/bin/env python3
"""
Example workflow: Extract embeddings and create 3D visualization
"""

from pathlib import Path
import pandas as pd
from client import DeepSeekClient
from umap_processor import compute_3d_umap_and_clusters, create_visualization_config

def main():
    # Configuration
    SERVER_URL = "http://localhost:8888"
    OUTPUT_DIR = Path("example_output")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Sample data - replace with your data
    sample_texts = [
        # Ecology and environment
        "Tropical rainforest ecosystem dynamics and biodiversity",
        "Climate change impacts on coral reef systems",
        "Urban ecology and green infrastructure planning",
        "Wetland restoration and ecosystem services",
        "Mountain ecosystem adaptation to climate change",
        
        # Conservation biology
        "Endangered species habitat conservation strategies",
        "Wildlife corridor design for landscape connectivity",
        "Marine protected area network optimization",
        "Invasive species management in island ecosystems",
        "Genetic diversity in small populations",
        
        # Plant science
        "Photosynthesis efficiency in C4 plants",
        "Root system architecture and nutrient uptake",
        "Plant-pollinator interaction networks",
        "Drought tolerance mechanisms in desert plants",
        "Mycorrhizal symbiosis and forest health",
        
        # Technology and environment
        "Remote sensing for deforestation monitoring",
        "Environmental DNA sampling techniques",
        "Machine learning for species distribution modeling",
        "Drone technology in wildlife surveys",
        "Bioacoustics monitoring of ecosystem health"
    ]
    
    labels = [
        "rainforest", "coral_reef", "urban_ecology", "wetlands", "mountains",
        "endangered", "corridors", "marine_protected", "invasive", "genetics",
        "photosynthesis", "roots", "pollinators", "drought", "mycorrhiza",
        "remote_sensing", "edna", "ml_modeling", "drones", "bioacoustics"
    ]
    
    print("=== DeepSeek Language Embedding Example ===\n")
    
    # Step 1: Connect to server
    print("1. Connecting to DeepSeek server...")
    client = DeepSeekClient(SERVER_URL)
    
    # Check server health
    health = client.health_check()
    print(f"   Server status: {health['status']}")
    print(f"   Model loaded: {health['model_loaded']}")
    print(f"   Embedding dimension: {health['embedding_dim']}")
    
    # Step 2: Extract embeddings
    print("\n2. Extracting embeddings...")
    embeddings_dir = OUTPUT_DIR / "embeddings"
    
    results = client.extract_embeddings_with_tokens(
        sample_texts,
        labels=labels,
        output_dir=embeddings_dir,
        save_format='pt'
    )
    
    print(f"   Processed {results['num_processed']} texts")
    print(f"   Saved to: {results['output_dir']}")
    
    # Show sample token analysis
    print("\n3. Sample tokenization analysis:")
    sample_result = client.tokenize(sample_texts[0])
    print(f"   Text: '{sample_texts[0]}'")
    print(f"   Tokens: {sample_result['num_tokens']}")
    print("   First 5 tokens:")
    for token in sample_result['tokens'][:5]:
        print(f"     [{token['position']}] '{token['token_str']}' (ID: {token['token_id']})")
    
    # Step 3: Compute 3D UMAP projection
    print("\n4. Computing 3D UMAP projection...")
    umap_output = OUTPUT_DIR / "umap_3d_data.json"
    
    umap_data = compute_3d_umap_and_clusters(
        embeddings_dir,
        output_path=umap_output,
        n_neighbors=5,  # Smaller for this small dataset
        min_dist=0.1,
        min_cluster_size=3,  # Smaller for demonstration
        normalize=True
    )
    
    # Step 4: Create visualization config
    print("\n5. Creating visualization configuration...")
    viz_config_path = OUTPUT_DIR / "visualization_config.json"
    
    config = create_visualization_config(
        umap_data,
        title="Environmental Science Concepts",
        subtitle="20 concepts embedded with DeepSeek-V3",
        output_path=viz_config_path
    )
    
    # Step 5: Show how to visualize
    print("\n6. To visualize the results:")
    print(f"   cd ../../visualization/embeddings")
    print(f"   cp {umap_output} .")
    print(f"   cp {viz_config_path} configs/")
    print(f"   python server.py --config configs/visualization_config.json")
    
    # Summary statistics
    print("\n7. Summary:")
    stats = umap_data['stats']
    print(f"   Total points: {stats['total_points']}")
    print(f"   Clusters found: {stats['n_clusters']}")
    print(f"   Noise points: {stats['noise_points']}")
    print(f"   Cluster sizes: {stats['cluster_sizes']}")
    
    # Export summary CSV
    summary_df = pd.DataFrame([
        {
            'label': point['name'],
            'text': sample_texts[i],
            'x': point['x'],
            'y': point['y'], 
            'z': point['z'],
            'cluster': point['cluster']
        }
        for i, point in enumerate(umap_data['points'])
    ])
    
    summary_path = OUTPUT_DIR / "embedding_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n   Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()