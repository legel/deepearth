#!/usr/bin/env python3
"""
Force download images for 10 species - deletes existing and re-downloads
"""

import os
import shutil
import subprocess
import sys

# The 10 species we want (with replacements)
FINAL_10_SPECIES = [
    'Helianthus debilis',        # Beach Sunflower
    'Gaillardia pulchella',      # Blanket Flower
    'Coreopsis leavenworthii',   # Leavenworth's Tickseed
    'Rudbeckia hirta',           # Black-eyed Susan
    'Monarda punctata',          # Spotted Beebalm
    'Salvia coccinea',           # Tropical Sage
    'Zamia integrifolia',        # Coontie
    'Tradescantia ohiensis',     # Spiderwort
    'Callicarpa americana',      # American Beautyberry (replacement)
    'Serenoa repens',            # Saw Palmetto (replacement)
]

def main():
    print("🧹 Cleaning up old data...")
    
    # Remove old data directory
    if os.path.exists('./data/plants'):
        response = input("Remove existing data/plants directory? (y/n): ")
        if response.lower() == 'y':
            shutil.rmtree('./data/plants')
            print("✓ Removed old data")
    
    print("\n📥 Downloading fresh images for 10 species...")
    
    # Use the adaptive download script
    cmd = [
        sys.executable, 
        'download_images.py',
        '--species', '10',
        '--images-per-species', '200',
        '--min-images', '30'
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("❌ Download failed!")
        return 1
    
    # Verify downloads
    print("\n✅ Verifying downloads...")
    data_dir = './data/plants'
    
    total = 0
    for species in FINAL_10_SPECIES:
        # Try different folder name formats
        folder_names = [
            species.replace(' ', '_').lower(),
            species.replace(' ', '_'),
            species.lower().replace(' ', '_'),
        ]
        
        # Add known mappings
        if species == 'Helianthus debilis':
            folder_names.append('beach_sunflower')
        elif species == 'Gaillardia pulchella':
            folder_names.append('blanket_flower')
        elif species == 'Callicarpa americana':
            folder_names.append('american_beautyberry')
        elif species == 'Serenoa repens':
            folder_names.append('saw_palmetto')
        
        found = False
        for folder_name in folder_names:
            folder_path = os.path.join(data_dir, folder_name)
            if os.path.exists(folder_path):
                count = len([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
                if count > 0:
                    print(f"  {species}: {count} images ✓")
                    total += count
                    found = True
                    break
        
        if not found:
            print(f"  {species}: NOT FOUND ❌")
    
    print(f"\nTotal images: {total}")
    
    if total == 0:
        print("\n❌ No images were downloaded! Check your internet connection and try again.")
        return 1
    
    print("\n✨ Download complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
