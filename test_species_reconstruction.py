"""
Test DeepEarth's ability to reconstruct masked language embeddings,
effectively acting as a species classifier from image data.
"""
import torch
import torch.nn.functional as F
import logging

from florida_plants_deepearth import FloridaPlantsProcessor
# FIX: Import the new, full model factory function
from models.deepearth_full_model import create_full_deepearth_model

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_species_reconstruction_test(batch_size: int = 4):
    """
    Performs the masked language reconstruction test.
    """
    logger.info("--- Starting Masked Species Reconstruction Test ---")
    
    # 1. Initialize data processor and the full model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = FloridaPlantsProcessor()
    
    logger.info("Creating full DeepEarth Model in 8-bit mode...")
    # FIX: Create the full model, passing arguments correctly
    model = create_full_deepearth_model(
        language_precision="int8",
        device=device
    )
    model.eval() # Set model to evaluation mode

    # 2. Prepare a data batch
    logger.info(f"Preparing a sample batch of {batch_size} images...")
    indices = list(range(batch_size))
    batch = processor.prepare_batch(batch_indices=indices)
    
    vision_input = batch['vision'].to(device)
    language_input_text = batch['language']

    # 3. Get the "ground truth" language embeddings
    with torch.no_grad():
        logger.info("Encoding original language tokens (ground truth)...")
        language_tokens_original = model.universal_encoder(
            {'language': language_input_text, 'vision': vision_input}
        )['language']

    # 4. Run the model with masked language
    logger.info("Running simulator with masked language input...")
    mask_config = {
        'language': 1.0,  # Mask 100% of the language tokens
        'vision': 0.0      # Do not mask any vision tokens
    }
    
    with torch.no_grad():
        # Pass all inputs to the single model.forward() call
        outputs = model(
            vision=vision_input,
            language=language_input_text,
            mask_config=mask_config
        )

    reconstructed_lang_tokens = outputs['reconstructions']['language']

    # 5. Evaluate the reconstruction
    logger.info("\n--- Evaluating Reconstruction as a Classifier ---")
    
    original_embeddings = language_tokens_original.mean(dim=1)
    reconstructed_embeddings = reconstructed_lang_tokens.mean(dim=1)

    original_embeddings_norm = F.normalize(original_embeddings, p=2, dim=1)
    reconstructed_embeddings_norm = F.normalize(reconstructed_embeddings, p=2, dim=1)

    similarity_matrix = torch.matmul(reconstructed_embeddings_norm, original_embeddings_norm.T)
    predictions = torch.argmax(similarity_matrix, dim=1)

    for i in range(batch_size):
        predicted_idx = predictions[i].item()
        actual_idx = i
        
        actual_species = language_input_text[actual_idx]
        predicted_species = language_input_text[predicted_idx]
        
        reconstruction_quality = similarity_matrix[i, actual_idx].item()
        
        print(f"\nSample {i+1}:")
        print(f"  - Actual Species:    '{actual_species}'")
        print(f"  - Predicted Species:   '{predicted_species}'")
        print(f"  - Reconstruction Quality (Cosine Sim): {reconstruction_quality:.4f}")
        
        if predicted_idx == actual_idx:
            print("  - Result: CORRECT 🎯")
        else:
            print("  - Result: INCORRECT ❌")

if __name__ == "__main__":
    run_species_reconstruction_test()
