"""
UltraHD Image Processing Script

This script processes images using the loaded UltraHD encoder model.
It takes an input image, applies the model, and outputs the processed image.
"""

import os
import sys
import json
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Import the model from load_model.py if it's already loaded
try:
    from load_model import MODEL
except ImportError:
    MODEL = None

def load_model_if_needed(model_type='standard'):
    """Load the model if it's not already loaded"""
    global MODEL
    
    if MODEL is not None:
        return MODEL
    
    # Define model paths based on model type
    if model_type == 'high-quality':
        model_paths = [
            os.path.join('..', '..', '..', 'Python', 'Fine_Tuned_Model_5', 'crystal_clear_denoiser_final.keras'),
        ]
    elif model_type == 'experimental':
        model_paths = [
            os.path.join('..', '..', '..', 'Python', 'Fine_Tuned_Model_5', 'ultrahd_denoiser_best.keras'),
        ]
    else:  # standard model - use any available model
        model_paths = [
            os.path.join('..', '..', '..', 'Python', 'Fine_Tuned_Model_5', 'crystal_clear_denoiser_final.keras'),
            os.path.join('..', '..', '..', 'Python', 'Fine_Tuned_Model_5', 'ultrahd_denoiser_best.keras'),
        ]
    
    # Try to load from any of the specified paths
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                print(f"Loading model from: {model_path}")
                MODEL = load_model(model_path, compile=False)
                return MODEL
            except Exception as e:
                print(f"Error loading model: {e}")
                continue
    
    # If we get here, no model was loaded
    raise ValueError("No valid model found. Please ensure the model files exist in Python/Fine_Tuned_Model_5")

def preprocess_image(image_path, target_size=(384, 384)):
    """Load and preprocess the image for the model"""
    try:
        # Load image
        img = Image.open(image_path)
        
        # Resize if needed
        if img.size != target_size:
            img = img.resize(target_size, Image.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img).astype(np.float32)
        
        # Normalize to [-1, 1] as expected by model
        img_array = img_array / 127.5 - 1.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, img
    
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {e}")

def postprocess_image(processed_array):
    """Convert the processed output to a displayable image"""
    # Remove batch dimension
    processed_array = processed_array[0]
    
    # Convert from [-1, 1] to [0, 255]
    processed_array = ((processed_array + 1) / 2.0 * 255).astype(np.uint8)
    
    # Create PIL image
    processed_image = Image.fromarray(processed_array)
    
    return processed_image

def calculate_metrics(original_img, processed_img):
    """Calculate PSNR and SSIM metrics between original and processed images"""
    # Convert to numpy arrays in [0, 1] range
    original_array = np.array(original_img).astype(np.float32) / 255.0
    processed_array = np.array(processed_img).astype(np.float32) / 255.0
    
    # Calculate PSNR (higher is better)
    psnr = peak_signal_noise_ratio(original_array, processed_array, data_range=1.0)
    
    # Calculate SSIM (higher is better)
    ssim_value = structural_similarity(
        original_array, 
        processed_array, 
        data_range=1.0,
        channel_axis=2 if len(original_array.shape) > 2 else None
    )
    
    return psnr, ssim_value

def process_image(input_path, output_path, model_type='standard'):
    """
    Process an image using the UltraHD encoder model
    
    Args:
        input_path: Path to the input image
        output_path: Path to save the processed image
        model_type: Type of model to use ('standard', 'high-quality', or 'experimental')
    
    Returns:
        Dictionary with processing results
    """
    start_time = time.time()
    
    try:
        print(f"Processing image with model_type: {model_type}")
        print(f"Input path: {input_path}")
        print(f"Output path: {output_path}")
        
        # Load the model if it's not already loaded
        model = load_model_if_needed(model_type)
        
        # Load and preprocess the image
        preprocessed_image, original_img = preprocess_image(input_path)
        
        # Process the image
        processed_array = model.predict(preprocessed_image)
        
        # Convert back to image format
        processed_img = postprocess_image(processed_array)
        
        # Save the processed image
        processed_img.save(output_path)
        
        # Calculate file size
        processed_size = os.path.getsize(output_path)
        
        # Calculate metrics
        psnr, ssim_value = calculate_metrics(original_img, processed_img)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Return results
        result = {
            "success": True,
            "outputPath": output_path,
            "processedSize": processed_size,
            "psnr": psnr,
            "ssim": ssim_value,
            "processingTime": processing_time
        }
        
        return result
    
    except Exception as e:
        print(f"Error in process_image: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "processingTime": time.time() - start_time
        }

def main():
    """Main function to process an image from command line arguments"""
    if len(sys.argv) < 3:
        print(json.dumps({
            "success": False,
            "error": "Not enough arguments. Usage: python process_image.py input_path output_path [model_type]"
        }))
        return
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    model_type = sys.argv[3] if len(sys.argv) > 3 else 'standard'
    
    result = process_image(input_path, output_path, model_type)
    print(json.dumps(result))

if __name__ == "__main__":
    main() 