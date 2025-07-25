#!/usr/bin/env python3
"""
Fixed download script that handles the actual DeepEarth dataset format
"""

import os
import requests
from PIL import Image
from io import BytesIO
from datasets import load_dataset
from tqdm import tqdm
import time
import hashlib

# The 10 species we want
SPECIES_LIST = [
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

FOLDER_NAMES = {
    'Helianthus debilis': 'beach_sunflower',
    'Gaillardia pulchella': 'blanket_flower',
    'Coreopsis leavenworthii': 'leavenworth_tickseed',
    'Rudbeckia hirta': 'black_eyed_susan',
    'Monarda punctata': 'spotted_beebalm',
    'Salvia coccinea': 'tropical_sage',
    'Zamia integrifolia': 'coontie',
    'Tradescantia ohiensis': 'spiderwort',
    'Callicarpa americana': 'american_beautyberry',
    'Serenoa repens': 'saw_palmetto',
}


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
    print("🌿 DeepEarth Plant Image Downloader - Fixed Format")
    print("=" * 50)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset("deepearth/central-florida-native-plants", split="train")
    print(f"Total dataset size: {len(dataset)} entries")
    
    # Create base directory
    base_dir = './data/plants'
    os.makedirs(base_dir, exist_ok=True)
    
    # Debug: Check data for our species
    print("\n🔍 Checking data availability:")
    for species in SPECIES_LIST:
        species_data = [ex for ex in dataset if ex['taxon_name'] == species]
        total_urls = 0
        for ex in species_data:
            urls = extract_urls_from_field(ex.get('image_urls', []))
            total_urls += len(urls)
        print(f"  {species}: {len(species_data)} entries, ~{total_urls} URLs")
    
    # Process each species
    total_downloaded = 0
    species_with_no_images = []
    
    for species in SPECIES_LIST:
        folder_name = FOLDER_NAMES[species]
        species_dir = os.path.join(base_dir, folder_name)
        os.makedirs(species_dir, exist_ok=True)
        
        # Filter dataset for this species
        species_data = [ex for ex in dataset if ex['taxon_name'] == species]
        
        if not species_data:
            print(f"\n❌ {species}: No data found")
            species_with_no_images.append(species)
            continue
        
        print(f"\n📥 Processing {species}")
        print(f"   Found {len(species_data)} entries")
        
        # Collect all URLs
        all_urls = []
        for entry in species_data:
            urls = extract_urls_from_field(entry.get('image_urls', []))
            all_urls.extend(urls)
        
        # Remove duplicates
        all_urls = list(set(all_urls))
        
        print(f"   Found {len(all_urls)} unique URLs")
        
        if not all_urls:
            print(f"   ⚠️  No valid URLs found")
            species_with_no_images.append(species)
            continue
        
        # Download images (max 200 per species)
        downloaded = 0
        failed = 0
        target = min(200, len(all_urls))
        
        with tqdm(total=target, desc=f"   {species[:30]}") as pbar:
            for i, url in enumerate(all_urls[:target]):
                # Create filename
                url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
                filename = f"{species.replace(' ', '_')}_{url_hash}.jpg"
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
    
    # Handle species with no images
    if species_with_no_images:
        print(f"\n⚠️  {len(species_with_no_images)} species have no images in the dataset")
        print("Creating synthetic images for these species...")
        
        # Import the synthetic generator
        from create_synthetic_data import generate_flower_pattern, SPECIES_COLORS
        import numpy as np
        import colorsys
        
        for species in species_with_no_images:
            folder_name = FOLDER_NAMES[species]
            species_dir = os.path.join(base_dir, folder_name)
            
            print(f"\n🎨 Creating synthetic images for {species}")
            
            # Create 50 synthetic images
            base_hsv = SPECIES_COLORS.get(species, (180, 0.5, 0.7))
            
            for i in range(50):
                img = Image.new('RGB', (224, 224), (255, 255, 255))
                
                # Add variation
                h_var = base_hsv[0] + np.random.uniform(-10, 10)
                s_var = base_hsv[1] + np.random.uniform(-0.1, 0.1)
                v_var = base_hsv[2] + np.random.uniform(-0.1, 0.1)
                
                rgb = colorsys.hsv_to_rgb(h_var/360, s_var, v_var)
                petal_color = tuple(int(c * 255) for c in rgb)
                
                center_rgb = colorsys.hsv_to_rgb(h_var/360, s_var, v_var * 0.7)
                center_color = tuple(int(c * 255) for c in center_rgb)
                
                # Generate pattern
                img = generate_flower_pattern(img, center_color, petal_color, 
                                            SPECIES_LIST.index(species))
                
                # Add noise
                img_array = np.array(img)
                noise = np.random.normal(0, 10, img_array.shape)
                img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
                img = Image.fromarray(img_array)
                
                # Save
                filename = f"{species.replace(' ', '_')}_synthetic_{i:04d}.jpg"
                filepath = os.path.join(species_dir, filename)
                img.save(filepath, 'JPEG', quality=90)
            
            print(f"   ✓ Created 50 synthetic images")
    
    # Save species list
    species_file = os.path.join(base_dir, 'species_list.txt')
    with open(species_file, 'w') as f:
        for species in SPECIES_LIST:
            f.write(f"{species}\n")
    
    # Final summary
    print("\n" + "=" * 50)
    print("📊 Final Summary:")
    print(f"{'Species':<35} {'Folder':<25} {'Images':<10}")
    print("-" * 70)
    
    total_final = 0
    for species in SPECIES_LIST:
        folder_name = FOLDER_NAMES[species]
        species_dir = os.path.join(base_dir, folder_name)
        
        if os.path.exists(species_dir):
            count = len([f for f in os.listdir(species_dir) if f.endswith('.jpg')])
            note = " (synthetic)" if species in species_with_no_images else ""
            print(f"{species:<35} {folder_name:<25} {count:<10}{note}")
            total_final += count
    
    print("-" * 70)
    print(f"{'Total:':<35} {'':<25} {total_final:<10}")
    
    print(f"\n✅ Dataset ready with {total_final} images!")
    print(f"📁 Images saved to: {base_dir}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
