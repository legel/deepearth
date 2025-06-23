#!/usr/bin/env python3
"""
DeepSeek Model Server - Load once, query many times
Provides HTTP endpoints for embeddings, completions, and tokenization
"""

import os
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify
from llama_cpp import Llama
import pandas as pd
from typing import List, Dict, Any
import threading
import queue
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global model instance
model = None
model_lock = threading.Lock()

app = Flask(__name__)

def load_model(model_path=None, n_gpu_layers=0):
    """Load the model once at startup"""
    global model
    logger.info("Loading DeepSeek-V3 model...")
    
    if model_path is None:
        # Default path - adjust based on your setup
        model_path = os.environ.get('DEEPSEEK_MODEL_PATH', 
                                   "models/DeepSeek-V3-0324-UD-Q4_K_XL-00001-of-00009.gguf")
    
    model = Llama(
        model_path=model_path,
        n_ctx=2048,  # Larger context for flexibility
        n_threads=os.cpu_count() or 40,  # Use all available CPUs
        n_gpu_layers=n_gpu_layers,  # GPU layers, 0 for CPU only
        embedding=True,
        verbose=False,
        n_batch=512,
        seed=42,
        use_mmap=True,
        use_mlock=False,
        low_vram=True
    )
    
    logger.info(f"Model loaded successfully! Using {model.n_threads} threads")
    logger.info(f"Embedding dimension: {model.n_embd()}")
    return model

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'embedding_dim': model.n_embd() if model else None,
        'n_threads': model.n_threads if model else None
    })

@app.route('/tokenize', methods=['POST'])
def tokenize():
    """Tokenize text and return token IDs and strings"""
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        with model_lock:
            # Tokenize
            token_ids = model.tokenize(text.encode('utf-8'))
            
            # Decode each token
            tokens_info = []
            for i, token_id in enumerate(token_ids):
                try:
                    token_str = model.detokenize([token_id]).decode('utf-8', errors='replace')
                except:
                    token_str = f"[TOKEN_{token_id}]"
                
                tokens_info.append({
                    'position': i,
                    'token_id': int(token_id),
                    'token_str': token_str,
                    'byte_length': len(token_str.encode('utf-8'))
                })
        
        return jsonify({
            'text': text,
            'num_tokens': len(token_ids),
            'tokens': tokens_info
        })
    
    except Exception as e:
        logger.error(f"Tokenization error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/embed', methods=['POST'])
def embed():
    """Get embeddings for text, with detailed token information"""
    try:
        data = request.json
        text = data.get('text', '')
        return_tokens = data.get('return_tokens', True)
        return_token_embeddings = data.get('return_token_embeddings', False)
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        with model_lock:
            # Get embeddings
            embeddings = model.embed(text)
            embeddings_array = np.array(embeddings)
            
            # Get token information if requested
            tokens_info = []
            if return_tokens:
                token_ids = model.tokenize(text.encode('utf-8'))
                for i, token_id in enumerate(token_ids):
                    try:
                        token_str = model.detokenize([token_id]).decode('utf-8', errors='replace')
                    except:
                        token_str = f"[TOKEN_{token_id}]"
                    
                    tokens_info.append({
                        'position': i,
                        'token_id': int(token_id),
                        'token_str': token_str
                    })
            
            # Prepare response
            response = {
                'text': text,
                'embedding_shape': embeddings_array.shape,
                'num_tokens': len(tokens_info) if tokens_info else embeddings_array.shape[0] if len(embeddings_array.shape) > 1 else 1
            }
            
            # Add mean embedding
            if len(embeddings_array.shape) == 2:
                mean_embedding = np.mean(embeddings_array, axis=0)
                response['mean_embedding'] = mean_embedding.tolist()
                response['mean_embedding_stats'] = {
                    'mean': float(np.mean(mean_embedding)),
                    'std': float(np.std(mean_embedding)),
                    'min': float(np.min(mean_embedding)),
                    'max': float(np.max(mean_embedding))
                }
            else:
                response['mean_embedding'] = embeddings_array.tolist()
            
            # Add token info
            if return_tokens:
                response['tokens'] = tokens_info
            
            # Add individual token embeddings if requested
            if return_token_embeddings and len(embeddings_array.shape) == 2:
                response['token_embeddings'] = embeddings_array.tolist()
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/embed_batch', methods=['POST'])
def embed_batch():
    """Get embeddings for multiple texts efficiently"""
    try:
        data = request.json
        texts = data.get('texts', [])
        save_to_csv = data.get('save_to_csv', False)
        csv_path = data.get('csv_path', 'batch_embeddings.csv')
        
        if not texts:
            return jsonify({'error': 'No texts provided'}), 400
        
        results = []
        all_token_data = []
        
        with model_lock:
            for idx, text in enumerate(texts):
                # Get embeddings
                embeddings = model.embed(text)
                embeddings_array = np.array(embeddings)
                
                # Get tokens
                token_ids = model.tokenize(text.encode('utf-8'))
                
                # Process tokens
                for i, token_id in enumerate(token_ids):
                    try:
                        token_str = model.detokenize([token_id]).decode('utf-8', errors='replace')
                    except:
                        token_str = f"[TOKEN_{token_id}]"
                    
                    # If we have token-level embeddings
                    if len(embeddings_array.shape) == 2 and i < embeddings_array.shape[0]:
                        token_embedding_mean = float(np.mean(embeddings_array[i]))
                        token_embedding_std = float(np.std(embeddings_array[i]))
                    else:
                        token_embedding_mean = None
                        token_embedding_std = None
                    
                    all_token_data.append({
                        'text_idx': idx,
                        'text': text[:50] + '...' if len(text) > 50 else text,
                        'token_position': i,
                        'token_id': int(token_id),
                        'token_str': token_str,
                        'token_embedding_mean': token_embedding_mean,
                        'token_embedding_std': token_embedding_std
                    })
                
                # Compute mean embedding
                if len(embeddings_array.shape) == 2:
                    mean_embedding = np.mean(embeddings_array, axis=0)
                else:
                    mean_embedding = embeddings_array
                
                results.append({
                    'text': text,
                    'num_tokens': len(token_ids),
                    'mean_embedding_shape': mean_embedding.shape,
                    'mean_embedding_stats': {
                        'mean': float(np.mean(mean_embedding)),
                        'std': float(np.std(mean_embedding))
                    }
                })
        
        # Save to CSV if requested
        if save_to_csv and all_token_data:
            df = pd.DataFrame(all_token_data)
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved token data to {csv_path}")
        
        return jsonify({
            'num_texts': len(texts),
            'results': results,
            'token_data_saved': save_to_csv,
            'csv_path': csv_path if save_to_csv else None
        })
    
    except Exception as e:
        logger.error(f"Batch embedding error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/complete', methods=['POST'])
def complete():
    """Generate completion for a prompt"""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        max_tokens = data.get('max_tokens', 100)
        temperature = data.get('temperature', 0.7)
        top_p = data.get('top_p', 0.9)
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        with model_lock:
            # Generate completion
            response = model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                echo=False
            )
            
            completion_text = response['choices'][0]['text']
            
            # Tokenize the completion for analysis
            completion_tokens = model.tokenize(completion_text.encode('utf-8'))
            
            return jsonify({
                'prompt': prompt,
                'completion': completion_text,
                'total_tokens': len(completion_tokens),
                'usage': {
                    'prompt_tokens': len(model.tokenize(prompt.encode('utf-8'))),
                    'completion_tokens': len(completion_tokens),
                    'total_tokens': len(model.tokenize(prompt.encode('utf-8'))) + len(completion_tokens)
                }
            })
    
    except Exception as e:
        logger.error(f"Completion error: {e}")
        return jsonify({'error': str(e)}), 500

def run_server(host='0.0.0.0', port=8888, model_path=None, n_gpu_layers=0):
    """Run the Flask server"""
    logger.info(f"Starting DeepSeek Model Server on {host}:{port}")
    
    # Load model before starting server
    load_model(model_path, n_gpu_layers)
    
    # Run Flask app
    app.run(host=host, port=port, debug=False, threaded=True)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='DeepSeek Model Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8888, help='Port to bind to')
    parser.add_argument('--model-path', type=str, help='Path to GGUF model file')
    parser.add_argument('--gpu-layers', type=int, default=0, help='Number of layers to offload to GPU')
    
    args = parser.parse_args()
    
    # Install Flask if needed
    try:
        import flask
    except ImportError:
        logger.info("Installing Flask...")
        import subprocess
        subprocess.run(["pip", "install", "flask"])
    
    run_server(args.host, args.port, args.model_path, args.gpu_layers)