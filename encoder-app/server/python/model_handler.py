#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim
import traceback

# Disable eager execution for better performance with saved models
tf.compat.v1.disable_eager_execution()

# Global variables
model = None
model_path = None

# Configure GPU memory growth to avoid OOM errors
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"Error configuring GPU memory growth: {e}")

def initialize_model(model_path):
    """
    Initialize the model from the given path
    """
    global model
    try:
        # Load model with compile=False to avoid optimizer issues
        model = load_model(model_path, compile=False)
        
        # Basic model information
        input_shape = model.input_shape
        output_shape = model.output_shape
        
        # Force a prediction to initialize all weights
        sample_input = np.zeros((1, input_shape[1], input_shape[2], input_shape[3]), dtype=np.float32)
        _ = model.predict(sample_input)
        
        return {
            "success": True,
            "model_path": model_path,
            "input_shape": input_shape[1:],
            "output_shape": output_shape[1:],
            "message": "Model loaded successfully"
        }
    except Exception as e:
        error_details = traceback.format_exc()
        return {
            "success": False,
            "model_path": model_path,
            "error": str(e),
            "details": error_details,
            "message": "Failed to load model"
        }

def get_model_info(model_path):
    """
    Get information about the model
    """
    global model
    
    try:
        if model is None:
            initialize_model(model_path)
        
        # Gather model information
        input_shape = model.input_shape
        output_shape = model.output_shape
        layers_info = []
        
        for layer in model.layers:
            layers_info.append({
                "name": layer.name,
                "type": layer.__class__.__name__,
                "output_shape": str(layer.output_shape)
            })
        
        # Check for GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        gpu_info = []
        for gpu in gpus:
            gpu_info.append({"name": gpu.name})
        
        return {
            "success": True,
            "model_path": model_path,
            "input_shape": input_shape[1:],
            "output_shape": output_shape[1:],
            "layers_count": len(model.layers),
            "layers": layers_info[:10],  # Limit to 10 layers to avoid too much data
            "gpus": gpu_info,
            "message": "Model info retrieved successfully"
        }
    except Exception as e:
        error_details = traceback.format_exc()
        return {
            "success": False,
            "model_path": model_path,
            "error": str(e),
            "details": error_details,
            "message": "Failed to get model information"
        }

def preprocess_image(image_path):
    """
    Preprocess the image for the model
    """
    try:
        # Load image
        img = Image.open(image_path).convert('RGB')
        
        # Resize to model input dimensions if needed
        input_shape = model.input_shape
        if input_shape[1] is not None and input_shape[2] is not None:
            img = img.resize((input_shape[1], input_shape[2]))
        
        # Convert to numpy and normalize to [-1, 1] range
        img_array = np.array(img).astype(np.float32) / 127.5 - 1.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, img.size
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

def postprocess_image(img_array, size=None):
    """
    Postprocess the model output to an image
    """
    try:
        # Convert from [-1, 1] to [0, 255] range
        img_array = ((img_array + 1) * 127.5).astype(np.uint8)
        
        # Remove batch dimension
        if img_array.shape[0] == 1:
            img_array = img_array[0]
        
        # Create PIL image
        img = Image.fromarray(img_array)
        
        # Resize back to original size if specified
        if size:
            img = img.resize(size)
        
        return img
    except Exception as e:
        raise Exception(f"Error postprocessing image: {str(e)}")

def process_image(input_path, output_path, options_json):
    """
    Process an image with the model
    """
    global model
    
    try:
        options = json.loads(options_json)
        
        # Get noise level from options or use default
        noise_level = options.get('noise_level', 0.5)
        noise_type = options.get('noise_type', 'gaussian')
        
        # Load and preprocess image
        img_array, original_size = preprocess_image(input_path)
        
        # Add noise if specified
        if noise_level > 0:
            if noise_type == 'gaussian':
                noise = np.random.normal(0, noise_level, img_array.shape).astype(np.float32)
                noisy_img = img_array + noise
            elif noise_type == 'salt_pepper':
                noisy_img = img_array.copy()
                # Salt noise
                salt_mask = np.random.random(img_array.shape) < (noise_level / 2)
                noisy_img[salt_mask] = 1.0
                # Pepper noise
                pepper_mask = np.random.random(img_array.shape) < (noise_level / 2)
                noisy_img[pepper_mask] = -1.0
            else:
                # Default to gaussian if type not recognized
                noise = np.random.normal(0, noise_level, img_array.shape).astype(np.float32)
                noisy_img = img_array + noise
                
            # Clip to valid range
            noisy_img = np.clip(noisy_img, -1.0, 1.0)
        else:
            noisy_img = img_array
        
        # Make a prediction
        if model is None:
            raise Exception("Model not initialized")
        
        # Save the noisy image if requested
        if options.get('save_noisy', False):
            noisy_path = os.path.splitext(output_path)[0] + '_noisy' + os.path.splitext(output_path)[1]
            noisy_img_pil = postprocess_image(noisy_img, original_size)
            noisy_img_pil.save(noisy_path)
        
        # Process with model
        output_array = model.predict(noisy_img)
        
        # Postprocess and save output
        output_img = postprocess_image(output_array, original_size)
        output_img.save(output_path)
        
        # Calculate metrics
        metrics = calculate_image_metrics(img_array[0], output_array[0])
        
        return {
            "success": True,
            "input_path": input_path,
            "output_path": output_path,
            "noisy_path": os.path.splitext(output_path)[0] + '_noisy' + os.path.splitext(output_path)[1] if options.get('save_noisy', False) else None,
            "metrics": metrics,
            "message": "Image processed successfully"
        }
    except Exception as e:
        error_details = traceback.format_exc()
        return {
            "success": False,
            "input_path": input_path,
            "output_path": output_path,
            "error": str(e),
            "details": error_details,
            "message": "Failed to process image"
        }

def calculate_image_metrics(original, processed):
    """
    Calculate image quality metrics
    """
    try:
        # Convert from [-1, 1] to [0, 1] range for metric calculation
        original_0_1 = (original + 1) / 2.0
        processed_0_1 = (processed + 1) / 2.0
        
        # Calculate PSNR
        psnr_value = peak_signal_noise_ratio(original_0_1, processed_0_1, data_range=1.0)
        
        # Calculate SSIM
        ssim_value = ssim(original_0_1, processed_0_1, data_range=1.0, channel_axis=2)
        
        # Calculate MSE
        mse = np.mean((original_0_1 - processed_0_1) ** 2)
        
        # Edge preservation (simplified)
        # Convert to grayscale and compute gradients
        original_gray = np.mean(original_0_1, axis=2)
        processed_gray = np.mean(processed_0_1, axis=2)
        
        # Simple edge detection using gradient magnitude
        def gradient_magnitude(img):
            dx = img[:, 1:] - img[:, :-1]
            dy = img[1:, :] - img[:-1, :]
            # Pad to maintain shape
            dx = np.pad(dx, ((0, 0), (0, 1)), mode='constant')
            dy = np.pad(dy, ((0, 1), (0, 0)), mode='constant')
            return np.sqrt(dx**2 + dy**2)
        
        original_edge = gradient_magnitude(original_gray)
        processed_edge = gradient_magnitude(processed_gray)
        
        # Edge correlation
        edge_preservation = np.corrcoef(original_edge.flatten(), processed_edge.flatten())[0, 1]
        
        return {
            "psnr": float(psnr_value),
            "ssim": float(ssim_value),
            "mse": float(mse),
            "edge_preservation": float(edge_preservation)
        }
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return {
            "psnr": 0,
            "ssim": 0,
            "mse": 1,
            "edge_preservation": 0,
            "error": str(e)
        }

def calculate_metrics_between_images(original_path, processed_path):
    """
    Calculate metrics between two images
    """
    try:
        # Load images
        original_img = Image.open(original_path).convert('RGB')
        processed_img = Image.open(processed_path).convert('RGB')
        
        # Ensure same size
        processed_img = processed_img.resize(original_img.size)
        
        # Convert to numpy arrays and normalize to [0, 1]
        original_array = np.array(original_img).astype(np.float32) / 255.0
        processed_array = np.array(processed_img).astype(np.float32) / 255.0
        
        # Calculate PSNR
        psnr_value = peak_signal_noise_ratio(original_array, processed_array, data_range=1.0)
        
        # Calculate SSIM
        ssim_value = ssim(original_array, processed_array, data_range=1.0, channel_axis=2)
        
        # Calculate MSE
        mse = np.mean((original_array - processed_array) ** 2)
        
        # Edge preservation (simplified)
        # Convert to grayscale
        original_gray = np.mean(original_array, axis=2)
        processed_gray = np.mean(processed_array, axis=2)
        
        # Simple edge detection using gradient magnitude
        def gradient_magnitude(img):
            dx = img[:, 1:] - img[:, :-1]
            dy = img[1:, :] - img[:-1, :]
            # Pad to maintain shape
            dx = np.pad(dx, ((0, 0), (0, 1)), mode='constant')
            dy = np.pad(dy, ((0, 1), (0, 0)), mode='constant')
            return np.sqrt(dx**2 + dy**2)
        
        original_edge = gradient_magnitude(original_gray)
        processed_edge = gradient_magnitude(processed_gray)
        
        # Edge correlation
        edge_preservation = np.corrcoef(original_edge.flatten(), processed_edge.flatten())[0, 1]
        
        return {
            "success": True,
            "original_path": original_path,
            "processed_path": processed_path,
            "metrics": {
                "psnr": float(psnr_value),
                "ssim": float(ssim_value),
                "mse": float(mse),
                "edge_preservation": float(edge_preservation)
            },
            "message": "Metrics calculated successfully"
        }
    except Exception as e:
        error_details = traceback.format_exc()
        return {
            "success": False,
            "original_path": original_path,
            "processed_path": processed_path,
            "error": str(e),
            "details": error_details,
            "message": "Failed to calculate metrics"
        }

if __name__ == "__main__":
    # Command line interface
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "message": "No command specified"}))
        sys.exit(1)
    
    command = sys.argv[1]
    
    try:
        if command == "initialize":
            if len(sys.argv) < 3:
                print(json.dumps({"success": False, "message": "No model path specified"}))
                sys.exit(1)
            
            model_path = sys.argv[2]
            result = initialize_model(model_path)
            print(json.dumps(result))
        
        elif command == "info":
            if len(sys.argv) < 3:
                print(json.dumps({"success": False, "message": "No model path specified"}))
                sys.exit(1)
            
            model_path = sys.argv[2]
            result = get_model_info(model_path)
            print(json.dumps(result))
        
        elif command == "process":
            if len(sys.argv) < 4:
                print(json.dumps({"success": False, "message": "Input or output path not specified"}))
                sys.exit(1)
            
            input_path = sys.argv[2]
            output_path = sys.argv[3]
            options_json = "{}" if len(sys.argv) < 5 else sys.argv[4]
            
            result = process_image(input_path, output_path, options_json)
            print(json.dumps(result))
        
        elif command == "metrics":
            if len(sys.argv) < 4:
                print(json.dumps({"success": False, "message": "Original or processed path not specified"}))
                sys.exit(1)
            
            original_path = sys.argv[2]
            processed_path = sys.argv[3]
            
            result = calculate_metrics_between_images(original_path, processed_path)
            print(json.dumps(result))
        
        else:
            print(json.dumps({"success": False, "message": f"Unknown command: {command}"}))
    
    except Exception as e:
        error_details = traceback.format_exc()
        print(json.dumps({
            "success": False,
            "error": str(e),
            "details": error_details,
            "message": "Command execution failed"
        }))
        sys.exit(1) 