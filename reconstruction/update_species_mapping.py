import json
import os

# Your actual folder structure based on the output
species_mapping = {
    # Common name folders
    "american_beautyberry": "Callicarpa americana",
    "beach_sunflower": "Helianthus debilis",
    "black_eyed_susan": "Rudbeckia hirta",
    "blanket_flower": "Gaillardia pulchella",
    "coontie": "Zamia integrifolia",
    "leavenworth_tickseed": "Coreopsis leavenworthii",
    "saw_palmetto": "Serenoa repens",
    "spiderwort": "Tradescantia ohiensis",
    "spotted_beebalm": "Monarda punctata",
    "tropical_sage": "Salvia coccinea",
    
    # Scientific name folders (keep as is)
    "callicarpa_americana": "Callicarpa americana",
    "coreopsis_leavenworthii": "Coreopsis leavenworthii",
    "gaillardia_pulchella": "Gaillardia pulchella",
    "helianthus_debilis": "Helianthus debilis",
    "monarda_punctata": "Monarda punctata",
    "rudbeckia_hirta": "Rudbeckia hirta",
    "salvia_coccinea": "Salvia coccinea",
    "tradescantia_ohiensis": "Tradescantia ohiensis",
    "zamia_integrifolia": "Zamia integrifolia"
}

# Save to the data directory
data_root = "/home/ubuntu/a/deepearth/reconstruction/mlp_unet/data/plants"
mapping_file = os.path.join(data_root, "species_mapping.json")

with open(mapping_file, 'w') as f:
    json.dump(species_mapping, f, indent=2)

print(f"Updated species mapping saved to: {mapping_file}")

# Also create a list of folders to use (avoiding duplicates)
folders_to_use = [
    "american_beautyberry",
    "beach_sunflower", 
    "black_eyed_susan",
    "blanket_flower",
    "coontie",
    "leavenworth_tickseed",
    "saw_palmetto",
    "spiderwort",
    "spotted_beebalm",
    "tropical_sage"
]

with open("recommended_folders.txt", 'w') as f:
    f.write('\n'.join(folders_to_use))

print(f"Recommended folders saved to: recommended_folders.txt")
print(f"Total species to use: {len(folders_to_use)}")
