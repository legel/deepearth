#!/usr/bin/env python3
"""
Upload corrected Parquet files to Hugging Face
"""

from huggingface_hub import HfApi
import os

api = HfApi()

# First, delete all old parquet files
print("Deleting old parquet files...")
old_files = [
    'data/train-00000-of-00001.parquet',
    'data/species_summary.parquet',
    'data/train-00000-of-00005.parquet',
    'data/train-00001-of-00005.parquet', 
    'data/train-00002-of-00005.parquet',
    'data/train-00003-of-00005.parquet',
    'data/train-00004-of-00005.parquet'
]

for file in old_files:
    try:
        api.delete_file(
            path_in_repo=file,
            repo_id='deepearth/central_florida_native_plants',
            repo_type='dataset'
        )
        print(f'  Deleted {file}')
    except Exception as e:
        print(f'  Could not delete {file}: {e}')

print("\nUploading new parquet files...")

# Upload all new parquet files
data_files = [f for f in os.listdir('data') if f.endswith('.parquet')]
for file in sorted(data_files):
    print(f'Uploading {file}...')
    api.upload_file(
        path_or_fileobj=f'data/{file}',
        path_in_repo=f'data/{file}',
        repo_id='deepearth/central_florida_native_plants',
        repo_type='dataset'
    )
    print(f'✓ Uploaded {file}')

print(f'\nAll {len(data_files)} parquet files uploaded to Hugging Face!')
print('\nThe dataset viewer should now show:')
print('- species_mean_embedding: Mean of ONLY the species name tokens (7168 dims)')
print('- all_tokens_mean_embedding: Mean of all tokens including prompt (7168 dims)')
print('- is_species_token: Correctly identified species name tokens')
print('- num_species_tokens: Count of tokens in the species name')
print('- Full token-level embeddings for all tokens')