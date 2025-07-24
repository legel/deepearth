#!/usr/bin/env python3
"""
Download only real images from iNaturalist, replacing species that have no images
"""

import os
import requests
from PIL import Image
from io import BytesIO
from datasets import load_dataset
from tqdm import tqdm
import time
import hashlib

# The 10 species we want (primary list)
SPECIES_LIST_PRIMARY = [
    'Helianthus debilis',        # Beach Sunflower
    'Gaillardia pulchella',      # Blanket Flower
    'Coreopsis leavenworthii',   # Leavenworth's Tickseed
    'Rudbeckia hirta',           # Black-eyed Susan
    'Monarda punctata',          # Spotted Beebalm
    'Salvia coccinea',           # Tropical Sage
    'Zamia integrifolia',        # Coontie
    'Tradescantia ohiensis',     # Spiderwort
    'Callicarpa americana',      # American Beautyberry
    'Serenoa repens',            # Saw Palmetto
]

# Alternative species to use as replacements
ALTERNATIVE_SPECIES = [
    'Hamelia patens',            # Firebush
    'Iris virginica',            # Virginia Iris
    'Lonicera sempervirens',     # Coral Honeysuckle
    'Mimosa strigillosa',        # Sunshine Mimosa
    'Psychotria nervosa',        # Wild Coffee
    'Solidago sempervirens',     # Seaside Goldenrod
    'Coreopsis lanceolata',      # Lanceleaf Coreopsis
    'Conradina canescens',       # False Rosemary
    'Sambucus nigra',            # Elderberry
    'Ipomoea imperati',          # Beach Morning Glory
    'Kosteletzkya pentacarpos',  # Seashore Mallow
    'Licania michauxii',         # Gopher Apple
    'Ruellia caroliniensis',     # Wild Petunia
    'Yucca filamentosa',         # Adam's Needle
    'Passiflora incarnata',      # Purple Passionflower
]

FOLDER_NAMES = {}  # Will be populated dynamically


def extract_urls_from_field(url_field):
    """Extract individual URLs from the image_urls field"""
    urls = []
    
    if not url_field:
        return urls
    
    # If it's a list, process each item
    if isinstance(url_field, list):
        for item in url_field:
            if not item:
                continue
            
            # Check if the item contains semicolon-separated URLs
            if ';' in str(item):
                # Split by semicolon
                sub_urls = str(item).split(';')
                urls.extend([u.strip() for u in sub_urls if u.strip()])
            else:
                # Single URL
                urls.append(str(item).strip())
    
    # If it's a string
    elif isinstance(url_field, str):
        if ';' in url_field:
            # Split by semicolon
            sub_urls = url_field.split(';')
            urls.extend([u.strip() for u in sub_urls if u.strip()])
        else:
            urls.append(url_field.strip())
    
    # Filter valid URLs
    valid_urls = []
    for url in urls:
        if url and url.startswith('http') and ('inaturalist' in url or 'amazonaws' in url):
            valid_urls.append(url)
    
    return valid_urls


def count_available_images(dataset, species):
    """Count how many valid image URLs are available for a species"""
    species_data = [ex for ex in dataset if ex['taxon_name'] == species]
    total_urls = 0
    
    for ex in species_data:
        urls = extract_urls_from_field(ex.get('image_urls', []))
        total_urls += len(urls)
    
    return len(species_data), total_urls


def find_replacement_species(dataset, used_species, min_images=50):
    """Find a suitable replacement species with enough images"""
    candidates = []
    
    for alt_species in ALTERNATIVE_SPECIES:
        if alt_species not in used_species:
            entries, urls = count_available_images(dataset, alt_species)
            if urls >= min_images:
                candidates.append((alt_species, entries, urls))
    
    # Sort by number of URLs (descending)
    candidates.sort(key=lambda x: x[2], reverse=True)
    
    if candidates:
        return candidates[0][0]  # Return species name
    
    # If no alternatives found, search the entire dataset
    print("\n🔍 Searching entire dataset for species with enough images...")
    all_species = {}
    
    for ex in dataset:
        species = ex['taxon_name']
        if species not in used_species and 'Florida' not in species:  # Prefer native species
            if species not in all_species:
                all_species[species] = 0
            urls = extract_urls_from_field(ex.get('image_urls', []))
            all_species[species] += len(urls)
    
    # Find species with enough images
    suitable = [(sp, count) for sp, count in all_species.items() if count >= min_images]
    suitable.sort(key=lambda x: x[1], reverse=True)
    
    if suitable:
        return suitable[0][0]
    
    return None


def download_image(url, filepath, timeout=30):
    """Download a single image"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
        }
        
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()
        
        # Download content
        content = BytesIO()
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                content.write(chunk)
        
        content.seek(0)
        
        # Verify it's an image
        img = Image.open(content)
        
        # Convert to RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize if too large
        max_size = 512
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Save
        img.save(filepath, 'JPEG', quality=90)
        return True
        
    except Exception as e:
        return False


def main():
    print("🌿 Real Image Downloader for Central Florida Plants")
    print("=" * 50)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset("deepearth/central-florida-native-plants", split="train")
    print(f"Total dataset size: {len(dataset)} entries")
    
    # Check availability and find replacements
    print("\n🔍 Checking species availability and finding replacements...")
    
    final_species_list = []
    replacements = []
    used_species = set()
    
    for species in SPECIES_LIST_PRIMARY:
        entries, urls = count_available_images(dataset, species)
        print(f"\n{species}:")
        print(f"  Entries: {entries}, URLs: {urls}")
        
        if urls < 30:  # Not enough images
            replacement = find_replacement_species(dataset, used_species)
            if replacement:
                repl_entries, repl_urls = count_available_images(dataset, replacement)
                print(f"  ❌ Too few images! Replacing with: {replacement} ({repl_urls} URLs)")
                final_species_list.append(replacement)
                replacements.append((species, replacement))
                used_species.add(replacement)
            else:
                print(f"  ⚠️  Too few images, but no replacement found. Keeping anyway.")
                final_species_list.append(species)
                used_species.add(species)
        else:
            print(f"  ✅ Sufficient images available")
            final_species_list.append(species)
            used_species.add(species)
    
    # Show final list
    print("\n" + "="*50)
    print("📋 Final species list:")
    for i, species in enumerate(final_species_list, 1):
        print(f"  {i}. {species}")
    
    if replacements:
        print("\n🔄 Replacements made:")
        for old, new in replacements:
            print(f"  {old} → {new}")
    
    # Create folder name mapping
    for species in final_species_list:
        # Create a clean folder name
        folder_name = species.lower().replace(' ', '_').replace("'", '')
        FOLDER_NAMES[species] = folder_name
    
    # Create base directory
    base_dir = './data/plants'
    os.makedirs(base_dir, exist_ok=True)
    
    # Download images
    print("\n" + "="*50)
    print("📥 Starting downloads...")
    
    total_downloaded = 0
    
    for species in final_species_list:
        folder_name = FOLDER_NAMES[species]
        species_dir = os.path.join(base_dir, folder_name)
        os.makedirs(species_dir, exist_ok=True)
        
        # Get all data for this species
        species_data = [ex for ex in dataset if ex['taxon_name'] == species]
        
        print(f"\n📥 Downloading {species}")
        
        # Collect all URLs
        all_urls = []
        for entry in species_data:
            urls = extract_urls_from_field(entry.get('image_urls', []))
            all_urls.extend(urls)
        
        # Remove duplicates
        all_urls = list(set(all_urls))
        print(f"   Found {len(all_urls)} unique URLs")
        
        # Download images (max 200 per species)
        downloaded = 0
        failed = 0
        target = min(200, len(all_urls))
        
        with tqdm(total=target, desc=f"   {species[:30]}") as pbar:
            for i, url in enumerate(all_urls[:target]):
                # Create filename
                url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
                filename = f"{folder_name}_{url_hash}.jpg"
                filepath = os.path.join(species_dir, filename)
                
                # Skip if exists
                if os.path.exists(filepath):
                    downloaded += 1
                    pbar.update(1)
                    continue
                
                # Download
                if download_image(url, filepath):
                    downloaded += 1
                else:
                    failed += 1
                
                pbar.update(1)
                
                # Rate limiting
                if (i + 1) % 10 == 0:
                    time.sleep(0.5)
        
        print(f"   ✓ Downloaded: {downloaded}, Failed: {failed}")
        total_downloaded += downloaded
    
    # Save species list
    species_file = os.path.join(base_dir, 'species_list.txt')
    with open(species_file, 'w') as f:
        for species in final_species_list:
            f.write(f"{species}\n")
    
    # Save folder mapping
    mapping_file = os.path.join(base_dir, 'species_folder_mapping.txt')
    with open(mapping_file, 'w') as f:
        for species in final_species_list:
            folder = FOLDER_NAMES[species]
            f.write(f"{species}|{folder}\n")
    
    # Final summary
    print("\n" + "=" * 50)
    print("📊 Final Summary:")
    print(f"{'Species':<40} {'Folder':<30} {'Images':<10}")
    print("-" * 80)
    
    total_final = 0
    for species in final_species_list:
        folder_name = FOLDER_NAMES[species]
        species_dir = os.path.join(base_dir, folder_name)
        
        if os.path.exists(species_dir):
            count = len([f for f in os.listdir(species_dir) if f.endswith('.jpg')])
            print(f"{species:<40} {folder_name:<30} {count:<10}")
            total_final += count
    
    print("-" * 80)
    print(f"{'Total:':<40} {'':<30} {total_final:<10}")
    
    print(f"\n✅ Downloaded {total_final} real images!")
    print(f"📁 Images saved to: {base_dir}")
    print(f"📄 Species list saved to: {species_file}")
    
    if total_final < 100:
        print("\n⚠️  Warning: Very few images downloaded. Check your internet connection.")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
