"""
Load UltraHD Encoder Model Script

This script loads the TensorFlow model used for image encoding/processing.
It makes the model available for subsequent processing requests.
"""

import os
import sys
import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import time

# Configure TensorFlow to use memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using {len(gpus)} GPU(s) with memory growth enabled")
    except RuntimeError as e:
        print(f"Error configuring GPUs: {e}")

# Define model paths - check multiple possible locations
MODEL_PATHS = [
    os.path.join('..', '..', '..', 'Python', 'Fine_Tuned_Model_5', 'crystal_clear_denoiser_final.keras'),
    os.path.join('..', '..', '..', 'Python', 'Fine_Tuned_Model_5', 'ultrahd_denoiser_best.keras'),
]

def load_ultrahd_model():
    """
    Load the UltraHD model from predefined paths
    
    Returns:
        model: Loaded TensorFlow model or None if loading fails
        error: Error message if loading fails, None otherwise
    """
    print("Current working directory:", os.getcwd())
    print("Checking for models in the following paths:")
    for path in MODEL_PATHS:
        print(f" - {os.path.abspath(path)}")
    
    for model_path in MODEL_PATHS:
        try:
            if os.path.exists(model_path):
                print(f"Found model at {model_path}, loading...")
                start_time = time.time()
                model = load_model(model_path, compile=False)
                load_time = time.time() - start_time
                print(f"Model loaded successfully in {load_time:.2f} seconds")
                
                # Print model summary to verify
                model.summary()
                
                return model, None
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
    
    # If no model is found or loaded
    return None, "No valid model found in the specified directories"

def main():
    """Main function to load the model and return status"""
    try:
        model, error = load_ultrahd_model()
        
        if model is not None:
            # Store model info in global variable for reuse
            model_info = {
                "status": "success",
                "input_shape": str(model.input_shape),
                "model_loaded": True
            }
            print(json.dumps(model_info))
            # Keep model in memory for server to use
            global MODEL
            MODEL = model
            return True
        else:
            error_info = {
                "status": "error",
                "message": error,
                "model_loaded": False
            }
            print(json.dumps(error_info))
            return False
    except Exception as e:
        error_info = {
            "status": "error",
            "message": str(e),
            "model_loaded": False
        }
        print(json.dumps(error_info))
        return False

# Try to load the model when this module is imported
MODEL, error = load_ultrahd_model()
if MODEL is None:
    print(f"Warning: Failed to preload model - {error}")
else:
    print("Model successfully preloaded and ready for use")

if __name__ == "__main__":
    main() 