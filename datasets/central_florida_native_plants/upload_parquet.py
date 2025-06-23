#!/usr/bin/env python3
"""
Upload Parquet files to Hugging Face
"""

from huggingface_hub import HfApi
import os

api = HfApi()

# First, delete the old parquet file
try:
    api.delete_file(
        path_in_repo='data/train-00000-of-00001.parquet',
        repo_id='deepearth/central_florida_native_plants',
        repo_type='dataset'
    )
    print('Deleted old parquet file')
except:
    pass

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
print('\nThe dataset viewer should update shortly to show:')
print('- Full 7168-dimensional embeddings for each token')
print('- Full 7168-dimensional mean embeddings')
print('- Token information (position, ID, string, is_species_token)')
print('- Embedding statistics (mean, std, min, max)')