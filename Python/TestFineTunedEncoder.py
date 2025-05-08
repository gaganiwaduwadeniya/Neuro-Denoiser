import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim
import urllib.request
import zipfile
import tarfile
from pathlib import Path
import glob
import time
import cv2
import random

# Define paths - updated for the latest fine-tuned model
MODEL_DIR = Path('Fine_Tuned_Model_7')  # Updated to Fine_Tuned_Model_7
RESULTS_DIR = Path('Fine_Tuned_Results_7')  # Updated to Fine_Tuned_Results_7
TEST_DATA_DIR = Path('test_data_encoder')

# Define patch parameters for large images
PATCH_SIZE = 256  # Model's expected input size
PATCH_OVERLAP = 32  # Overlap between patches to avoid boundary artifacts

# Create necessary directories
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
TEST_DATA_DIR.mkdir(exist_ok=True, parents=True)

# Create subdirectories for different types of results
ORIGINAL_DIR = RESULTS_DIR / 'original'
NOISY_DIR = RESULTS_DIR / 'noisy'
DENOISED_DIR = RESULTS_DIR / 'denoised'
METRICS_DIR = RESULTS_DIR / 'metrics'

for directory in [ORIGINAL_DIR, NOISY_DIR, DENOISED_DIR, METRICS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

def download_test_images():
    """Download test images from different sources"""
    print("Downloading test images...")
    
    # Option 1: BSD68 dataset (common benchmark for denoising)
    bsd68_url = "https://webdav.tuebingen.mpg.de/pixel/benchmark/test/kodak/*.png"
    
    # Since direct downloading from the pattern isn't possible, let's use Kodak dataset instead
    kodak_url = "http://r0k.us/graphics/kodak/kodak/kodim{:02d}.png"
    
    # Download 10 images from Kodak dataset
    for i in range(1, 11):
        img_url = kodak_url.format(i)
        img_path = TEST_DATA_DIR / f"kodak_{i}.png"
        
        if not img_path.exists():
            try:
                urllib.request.urlretrieve(img_url, img_path)
                print(f"Downloaded {img_url} to {img_path}")
            except Exception as e:
                print(f"Failed to download {img_url}: {e}")
    
    # Option 2: If Kodak fails, use placeholder images
    if not list(TEST_DATA_DIR.glob("*.png")):
        print("Creating placeholder test images since download failed...")
        # Create some placeholder images
        for i in range(1, 6):
            img = np.ones((384, 384, 3), dtype=np.float32) * (i / 10.0)
            img_path = TEST_DATA_DIR / f"placeholder_{i}.png"
            plt.imsave(str(img_path), img)
    
    # Count downloaded images
    image_files = list(TEST_DATA_DIR.glob("*.png"))
    if not image_files:
        raise ValueError("No test images available. Please check your internet connection.")
    
    print(f"Successfully prepared {len(image_files)} test images")
    return image_files

def load_and_preprocess_image(image_path, target_size=(384, 384)):
    """Load and preprocess a single image for the encoder model"""
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    # Normalize to [-1, 1] as per the encoder model's requirements
    img_array = img_array.astype(np.float32) / 127.5 - 1.0
    return img_array

def apply_noise(image, noise_type='gaussian', noise_params=None):
    """
    Apply different types of noise to an image
    
    Args:
        image: Normalized image array [-1, 1]
        noise_type: Type of noise to apply ('gaussian', 'salt_pepper', 'poisson', 'speckle', 'line_pattern')
        noise_params: Parameters for the noise function
    
    Returns:
        Noisy image array [-1, 1]
    """
    if noise_params is None:
        noise_params = {}
    
    # Make a copy to avoid modifying the original
    noisy_image = image.copy()
    
    if noise_type == 'gaussian':
        # Default parameters
        std = noise_params.get('std', 0.5)
        mean = noise_params.get('mean', 0)
        
        # Generate Gaussian noise
        noise = np.random.normal(mean, std, image.shape).astype(np.float32)
        noisy_image = image + noise
    
    elif noise_type == 'salt_pepper':
        # Default parameters
        amount = noise_params.get('amount', 0.5)
        
        # Generate salt and pepper noise
        salt_vs_pepper = 0.5  # Equal amounts of salt and pepper
        
        # Salt (white) noise
        num_salt = np.ceil(amount * image.size * salt_vs_pepper)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
        noisy_image[tuple(coords)] = 1.0
        
        # Pepper (black) noise
        num_pepper = np.ceil(amount * image.size * (1 - salt_vs_pepper))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        noisy_image[tuple(coords)] = -1.0
    
    elif noise_type == 'poisson':
        # For Poisson noise, we need to convert from [-1, 1] to non-negative values
        shifted_image = (image + 1) / 2  # Shift to [0, 1] range
        
        # Scale image to appropriate values for Poisson noise
        vals = len(np.unique(shifted_image))
        vals = 2 ** np.ceil(np.log2(vals))
        
        # Apply Poisson noise
        noisy_shifted = np.random.poisson(shifted_image * vals) / float(vals)
        
        # Convert back to [-1, 1] range
        noisy_image = noisy_shifted * 2 - 1
    
    elif noise_type == 'speckle':
        # Speckle noise (multiplicative noise)
        # Default parameters
        var = noise_params.get('var', 0.5)
        
        # Generate speckle noise
        noise = np.random.randn(*image.shape).astype(np.float32) * var
        noisy_image = image + image * noise
    
    elif noise_type == 'line_pattern':
        # Add line pattern noise (to test the improved line pattern handling)
        # Default parameters
        intensity = noise_params.get('intensity', 0.5)
        
        # Create a line pattern
        h, w = image.shape[0], image.shape[1]
        pattern_type = np.random.randint(0, 3)
        
        # Create line mask
        line_mask = np.zeros_like(image)
        
        if pattern_type == 0:  # Horizontal lines
            # Create 1-3 random horizontal lines
            num_lines = np.random.randint(1, 4)
            for _ in range(num_lines):
                y_pos = np.random.randint(0, h-1)
                line_width = np.random.randint(1, 3)
                y_start = max(0, y_pos - line_width // 2)
                y_end = min(h, y_pos + line_width // 2 + 1)
                line_mask[y_start:y_end, :, :] = intensity
        
        elif pattern_type == 1:  # Vertical lines
            # Create 1-3 random vertical lines
            num_lines = np.random.randint(1, 4)
            for _ in range(num_lines):
                x_pos = np.random.randint(0, w-1)
                line_width = np.random.randint(1, 3)
                x_start = max(0, x_pos - line_width // 2)
                x_end = min(w, x_pos + line_width // 2 + 1)
                line_mask[:, x_start:x_end, :] = intensity
        
        else:  # Diagonal pattern
            # Use linear gradient for diagonal
            if np.random.uniform() < 0.5:
                # Top-left to bottom-right
                for i in range(h):
                    for j in range(w):
                        line_mask[i, j, :] = ((i + j) / (h + w)) * intensity
            else:
                # Top-right to bottom-left
                for i in range(h):
                    for j in range(w):
                        line_mask[i, j, :] = ((i + (w-1-j)) / (h + w)) * intensity
        
        # Add subtle noise to the line mask
        line_noise = np.random.normal(0, 0.03 * intensity, line_mask.shape)
        line_mask += line_noise
        
        # Add line pattern to image
        noisy_image = noisy_image + line_mask
    
    # Clip values to valid range [-1, 1]
    noisy_image = np.clip(noisy_image, -1.0, 1.0)
    
    return noisy_image.astype(np.float32)

def postprocess_for_display(image):
    """Convert normalized image array from [-1, 1] to [0, 1] for display"""
    # Convert from [-1, 1] to [0, 1]
    image_display = (image + 1) / 2.0
    
    # Clip values to valid range
    image_display = np.clip(image_display, 0.0, 1.0)
    
    return image_display.astype(np.float32)

def save_image(image, save_path):
    """Save image with proper data type conversion"""
    # Ensure image is in [0, 1] range and float32 type
    if image.min() < 0:  # If in [-1, 1] range
        image = (image + 1) / 2.0
    
    # Clip to [0, 1] range
    image = np.clip(image, 0, 1).astype(np.float32)
    
    # Save image using PIL to avoid matplotlib dtype issues
    from PIL import Image
    img_uint8 = (image * 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(str(save_path))

def calculate_metrics(original, denoised, noisy=None):
    """
    Calculate performance metrics for the denoised image
    
    Args:
        original: Original clean image in [-1, 1] range
        denoised: Denoised image in [-1, 1] range
        noisy: Noisy image in [-1, 1] range (optional)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Convert to [0, 1] range for metrics calculation
    original_01 = (original + 1) / 2.0
    denoised_01 = (denoised + 1) / 2.0
    
    # Ensure correct data types
    original_01 = original_01.astype(np.float32)
    denoised_01 = denoised_01.astype(np.float32)
    
    # Mean Squared Error (MSE) - lower is better
    metrics['mse'] = mean_squared_error(original_01.flatten(), denoised_01.flatten())
    
    # Peak Signal-to-Noise Ratio (PSNR) - higher is better
    metrics['psnr'] = peak_signal_noise_ratio(original_01, denoised_01, data_range=1.0)
    
    # Structural Similarity Index (SSIM) - higher is better
    metrics['ssim'] = ssim(original_01, denoised_01, data_range=1.0, channel_axis=2)
    
    # Edge preservation metrics - specific for UltraHD model evaluation
    try:
        # Sobel edge detection
        original_gray = cv2.cvtColor((original_01 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        denoised_gray = cv2.cvtColor((denoised_01 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Compute Sobel edges
        original_edges = cv2.Sobel(original_gray, cv2.CV_64F, 1, 1, ksize=3)
        denoised_edges = cv2.Sobel(denoised_gray, cv2.CV_64F, 1, 1, ksize=3)
        
        # Normalize edge maps
        original_edges = np.abs(original_edges) / np.max(np.abs(original_edges))
        denoised_edges = np.abs(denoised_edges) / np.max(np.abs(denoised_edges))
        
        # Edge preservation ratio (correlation between edge maps)
        edge_correlation = np.corrcoef(original_edges.flatten(), denoised_edges.flatten())[0, 1]
        metrics['edge_preservation'] = edge_correlation
    except Exception as e:
        print(f"Error calculating edge preservation: {e}")
        metrics['edge_preservation'] = 0.0
    
    # If noisy image is provided, calculate improvement metrics
    if noisy is not None:
        # Convert noisy to [0, 1] range
        noisy_01 = (noisy + 1) / 2.0
        noisy_01 = noisy_01.astype(np.float32)
        
        # Metrics for noisy image
        metrics['noisy_mse'] = mean_squared_error(original_01.flatten(), noisy_01.flatten())
        metrics['noisy_psnr'] = peak_signal_noise_ratio(original_01, noisy_01, data_range=1.0)
        metrics['noisy_ssim'] = ssim(original_01, noisy_01, data_range=1.0, channel_axis=2)
        
        # Edge preservation for noisy image
        try:
            noisy_gray = cv2.cvtColor((noisy_01 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            noisy_edges = cv2.Sobel(noisy_gray, cv2.CV_64F, 1, 1, ksize=3)
            noisy_edges = np.abs(noisy_edges) / np.max(np.abs(noisy_edges))
            noisy_edge_correlation = np.corrcoef(original_edges.flatten(), noisy_edges.flatten())[0, 1]
            metrics['noisy_edge_preservation'] = noisy_edge_correlation
            metrics['edge_improvement'] = metrics['edge_preservation'] - metrics['noisy_edge_preservation']
        except:
            metrics['noisy_edge_preservation'] = 0.0
            metrics['edge_improvement'] = 0.0
        
        # Improvement metrics
        metrics['mse_improvement'] = metrics['noisy_mse'] - metrics['mse']
        metrics['psnr_improvement'] = metrics['psnr'] - metrics['noisy_psnr']
        metrics['ssim_improvement'] = metrics['ssim'] - metrics['noisy_ssim']
    
    return metrics

def visualize_results(original, noisy, denoised, metrics, save_path):
    """
    Create visualization of original, noisy, and denoised images with metrics,
    including edge maps to showcase the UltraHD model's performance on edge preservation
    
    Args:
        original: Original clean image in [-1, 1] range
        noisy: Noisy image in [-1, 1] range
        denoised: Denoised image in [-1, 1] range
        metrics: Dictionary of metrics
        save_path: Path to save the visualization
    """
    # Convert to [0, 1] range for display
    original_display = postprocess_for_display(original)
    noisy_display = postprocess_for_display(noisy)
    denoised_display = postprocess_for_display(denoised)
    
    # Create a larger figure with edge maps
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Display images in top row
    axes[0, 0].imshow(original_display)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(noisy_display)
    axes[0, 1].set_title(f'Noisy\nPSNR: {metrics["noisy_psnr"]:.2f}, SSIM: {metrics["noisy_ssim"]:.4f}')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(denoised_display)
    axes[0, 2].set_title(f'Denoised\nPSNR: {metrics["psnr"]:.2f}, SSIM: {metrics["ssim"]:.4f}')
    axes[0, 2].axis('off')
    
    # Create edge maps for bottom row
    try:
        # Convert images to grayscale
        original_gray = cv2.cvtColor((original_display * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        noisy_gray = cv2.cvtColor((noisy_display * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        denoised_gray = cv2.cvtColor((denoised_display * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Compute edges using Sobel filter
        original_edges = cv2.Sobel(original_gray, cv2.CV_64F, 1, 1, ksize=3)
        noisy_edges = cv2.Sobel(noisy_gray, cv2.CV_64F, 1, 1, ksize=3)
        denoised_edges = cv2.Sobel(denoised_gray, cv2.CV_64F, 1, 1, ksize=3)
        
        # Normalize for visualization
        original_edges = np.abs(original_edges) / np.max(np.abs(original_edges))
        noisy_edges = np.abs(noisy_edges) / np.max(np.abs(noisy_edges))
        denoised_edges = np.abs(denoised_edges) / np.max(np.abs(denoised_edges))
        
        # Display edge maps in bottom row
        axes[1, 0].imshow(original_edges, cmap='gray')
        axes[1, 0].set_title('Original Edges')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(noisy_edges, cmap='gray')
        if 'noisy_edge_preservation' in metrics:
            noisy_edge_score = metrics['noisy_edge_preservation']
            axes[1, 1].set_title(f'Noisy Edges\nPreservation: {noisy_edge_score:.4f}')
        else:
            axes[1, 1].set_title('Noisy Edges')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(denoised_edges, cmap='gray')
        if 'edge_preservation' in metrics:
            edge_score = metrics['edge_preservation']
            axes[1, 2].set_title(f'Denoised Edges\nPreservation: {edge_score:.4f}')
        else:
            axes[1, 2].set_title('Denoised Edges')
        axes[1, 2].axis('off')
    except Exception as e:
        print(f"Error generating edge maps: {e}")
        # If edge detection fails, display blank panels
        for i in range(3):
            axes[1, i].imshow(np.zeros_like(original_gray), cmap='gray')
            axes[1, i].set_title('Edge map unavailable')
            axes[1, i].axis('off')
    
    # Add overall metrics as text
    plt.figtext(0.5, 0.01, 
                f'Improvements: PSNR: +{metrics["psnr_improvement"]:.2f} dB, SSIM: +{metrics["ssim_improvement"]:.4f}' + 
                (f', Edge: +{metrics.get("edge_improvement", 0):.4f}' if 'edge_improvement' in metrics else ''), 
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)  # Higher DPI for better detail
    plt.close()

def visualize_performance_by_noise_type(all_metrics, save_path):
    """Create bar charts comparing performance across noise types"""
    noise_types = list(all_metrics.keys())
    
    # Prepare data for plotting
    psnr_values = [np.mean([m['psnr'] for m in all_metrics[noise_type]]) for noise_type in noise_types]
    ssim_values = [np.mean([m['ssim'] for m in all_metrics[noise_type]]) for noise_type in noise_types]
    psnr_improvement = [np.mean([m['psnr_improvement'] for m in all_metrics[noise_type]]) for noise_type in noise_types]
    ssim_improvement = [np.mean([m['ssim_improvement'] for m in all_metrics[noise_type]]) for noise_type in noise_types]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot PSNR values
    axes[0, 0].bar(noise_types, psnr_values, color='blue')
    axes[0, 0].set_title('Average PSNR by Noise Type')
    axes[0, 0].set_ylabel('PSNR (dB)')
    axes[0, 0].set_ylim(bottom=0)
    
    # Plot SSIM values
    axes[0, 1].bar(noise_types, ssim_values, color='green')
    axes[0, 1].set_title('Average SSIM by Noise Type')
    axes[0, 1].set_ylabel('SSIM')
    axes[0, 1].set_ylim(0, 1)
    
    # Plot PSNR improvement
    axes[1, 0].bar(noise_types, psnr_improvement, color='orange')
    axes[1, 0].set_title('Average PSNR Improvement by Noise Type')
    axes[1, 0].set_ylabel('PSNR Improvement (dB)')
    
    # Plot SSIM improvement
    axes[1, 1].bar(noise_types, ssim_improvement, color='purple')
    axes[1, 1].set_title('Average SSIM Improvement by Noise Type')
    axes[1, 1].set_ylabel('SSIM Improvement')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_best_and_worst(all_metrics, noise_types, image_paths, save_path):
    """
    Visualize the best and worst performing cases for each noise type
    
    Args:
        all_metrics: Dictionary of all metrics grouped by noise type
        noise_types: List of noise types
        image_paths: Dictionary of image paths by noise type and image index
        save_path: Path to save the visualization
    """
    # Number of rows = number of noise types
    # Columns: [Best PSNR, Worst PSNR]
    fig, axes = plt.subplots(len(noise_types), 2, figsize=(12, 4*len(noise_types)))
    
    for i, noise_type in enumerate(noise_types):
        # Get metrics for this noise type
        metrics = all_metrics[noise_type]
        
        # Find indices of best and worst cases based on PSNR
        psnr_values = [m['psnr'] for m in metrics]
        best_idx = np.argmax(psnr_values)
        worst_idx = np.argmin(psnr_values)
        
        # Load best and worst images
        best_paths = image_paths[noise_type][best_idx]
        worst_paths = image_paths[noise_type][worst_idx]
        
        # Use PIL to load images to avoid data type issues
        from PIL import Image
        
        best_original = np.array(Image.open(best_paths['original'])).astype(np.float32) / 255.0
        best_noisy = np.array(Image.open(best_paths['noisy'])).astype(np.float32) / 255.0
        best_denoised = np.array(Image.open(best_paths['denoised'])).astype(np.float32) / 255.0
        
        worst_original = np.array(Image.open(worst_paths['original'])).astype(np.float32) / 255.0
        worst_noisy = np.array(Image.open(worst_paths['noisy'])).astype(np.float32) / 255.0
        worst_denoised = np.array(Image.open(worst_paths['denoised'])).astype(np.float32) / 255.0
        
        # Create composite images (original/noisy/denoised side by side)
        best_composite = np.hstack([best_original, best_noisy, best_denoised])
        worst_composite = np.hstack([worst_original, worst_noisy, worst_denoised])
        
        # Handle single noise type case (prevents indexing error)
        if len(noise_types) == 1:
            ax_best = axes[0]
            ax_worst = axes[1]
        else:
            ax_best = axes[i, 0]
            ax_worst = axes[i, 1]
            
        # Plot best case
        ax_best.imshow(best_composite)
        ax_best.set_title(f'Best for {noise_type}\nPSNR: {metrics[best_idx]["psnr"]:.2f}')
        ax_best.axis('off')
        
        # Plot worst case
        ax_worst.imshow(worst_composite)
        ax_worst.set_title(f'Worst for {noise_type}\nPSNR: {metrics[worst_idx]["psnr"]:.2f}')
        ax_worst.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def denoise_image_patches(model, noisy_image, patch_size=PATCH_SIZE, overlap=PATCH_OVERLAP):
    """
    Process a large image by breaking it into overlapping patches, processing each patch,
    and then blending the patches back together.
    
    Args:
        model: The neural network model for denoising
        noisy_image: The input noisy image in [-1, 1] range with shape (h, w, 3)
        patch_size: Size of patches to process (default: 256x256)
        overlap: Overlap between patches to avoid boundary artifacts (default: 32)
        
    Returns:
        Denoised image in [-1, 1] range with same shape as input
    """
    print(f"Using patch-based approach (patch size: {patch_size}x{patch_size}, overlap: {overlap})")
    print(f"Input image shape: {noisy_image.shape}, dtype: {noisy_image.dtype}")
    
    # Get image dimensions
    h, w = noisy_image.shape[:2]
    
    # If image is smaller than patch size, pad and process as a single patch
    if h <= patch_size and w <= patch_size:
        print(f"Image is smaller than patch size, padding to {patch_size}x{patch_size}")
        padded_image = np.zeros((patch_size, patch_size, 3), dtype=np.float32)
        padded_image[:h, :w] = noisy_image
        
        # Process single patch
        denoised_patch = model.predict(np.expand_dims(padded_image, axis=0), verbose=0)[0]
        
        # Return only the relevant portion
        return denoised_patch[:h, :w]
    
    # Initialize the output image and weight map
    denoised_image = np.zeros_like(noisy_image, dtype=np.float32)
    weight_map = np.zeros((h, w, 1), dtype=np.float32)
    
    # Create a weight mask for blending (higher weight in center, lower at edges)
    y = np.linspace(-1, 1, patch_size)
    x = np.linspace(-1, 1, patch_size)
    xx, yy = np.meshgrid(x, y)
    mask = np.clip(1.0 - np.sqrt(xx**2 + yy**2), 0, 1)
    mask = mask[:, :, np.newaxis]
    
    # To ensure complete coverage, we need to handle the right and bottom edges specially
    # Calculate number of patches needed in each dimension
    n_patches_h = max(1, int(np.ceil(h / (patch_size - overlap))))
    n_patches_w = max(1, int(np.ceil(w / (patch_size - overlap))))
    
    # Calculate exact positions to ensure full coverage including edges
    y_positions = []
    for i in range(n_patches_h):
        if i == n_patches_h - 1:  # Last patch
            y_pos = max(0, h - patch_size)
        else:
            y_pos = i * (patch_size - overlap)
        y_positions.append(y_pos)
    
    x_positions = []
    for i in range(n_patches_w):
        if i == n_patches_w - 1:  # Last patch
            x_pos = max(0, w - patch_size)
        else:
            x_pos = i * (patch_size - overlap)
        x_positions.append(x_pos)
    
    # Make positions unique (can happen with small images)
    y_positions = sorted(set(y_positions))
    x_positions = sorted(set(x_positions))
    
    total_patches = len(y_positions) * len(x_positions)
    print(f"Processing image with {len(y_positions)}x{len(x_positions)}={total_patches} patches")
    patch_count = 0
    
    # Process each patch
    for i, y_start in enumerate(y_positions):
        for j, x_start in enumerate(x_positions):
            # Calculate patch coordinates
            y_end = min(y_start + patch_size, h)
            x_end = min(x_start + patch_size, w)
            
            # Debug info
            print(f"  Patch {i},{j}: coords={y_start}:{y_end}, {x_start}:{x_end}")
            
            # Extract patch
            patch = noisy_image[y_start:y_end, x_start:x_end]
            print(f"  Extracted patch shape: {patch.shape}")
            
            # Handle edge cases - pad if necessary
            need_padding = False
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                print(f"  Padding patch from {patch.shape} to {patch_size}x{patch_size}")
                padded_patch = np.zeros((patch_size, patch_size, 3), dtype=np.float32)
                padded_patch[:patch.shape[0], :patch.shape[1], :] = patch
                patch = padded_patch
                need_padding = True
            
            try:
                # Process patch through model
                print(f"  Sending patch to model: shape={patch.shape}, dtype={patch.dtype}")
                denoised_patch = model.predict(np.expand_dims(patch, axis=0), verbose=0)[0]
                print(f"  Model output shape: {denoised_patch.shape}")
                
                # Get the same shape as the original patch if padding was applied
                if need_padding:
                    denoised_patch = denoised_patch[:y_end-y_start, :x_end-x_start, :]
                    patch_mask = mask[:y_end-y_start, :x_end-x_start, :]
                else:
                    patch_mask = mask
                
                # Add weighted patch to output
                denoised_image[y_start:y_end, x_start:x_end] += denoised_patch * patch_mask
                weight_map[y_start:y_end, x_start:x_end] += patch_mask
            except Exception as e:
                print(f"  Error processing patch: {e}")
                # Continue with other patches even if one fails
                continue
            
            # Update progress
            patch_count += 1
            if patch_count % 5 == 0 or patch_count == total_patches:
                print(f"  Processed {patch_count}/{total_patches} patches")
    
    # Check if we have valid weights
    if np.any(weight_map == 0):
        zero_pixels = np.sum(weight_map == 0)
        zero_percent = (zero_pixels / (h * w)) * 100
        print(f"Warning: {zero_pixels} pixels ({zero_percent:.2f}%) have zero weight.")
        
        # Process corners specifically if they have zero weight
        corner_positions = [
            (0, 0),                     # Top-left
            (0, w - patch_size),        # Top-right
            (h - patch_size, 0),        # Bottom-left
            (h - patch_size, w - patch_size)  # Bottom-right
        ]
        
        for y_start, x_start in corner_positions:
            y_start = max(0, min(y_start, h - patch_size))
            x_start = max(0, min(x_start, w - patch_size))
            
            y_end = min(y_start + patch_size, h)
            x_end = min(x_start + patch_size, w)
            
            # Check if this region has zero weights
            if np.any(weight_map[y_start:y_end, x_start:x_end] == 0):
                print(f"  Processing corner region: {y_start}:{y_end}, {x_start}:{x_end}")
                
                # Extract patch
                patch = noisy_image[y_start:y_end, x_start:x_end]
                
                # Pad if necessary
                if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                    padded_patch = np.zeros((patch_size, patch_size, 3), dtype=np.float32)
                    padded_patch[:patch.shape[0], :patch.shape[1], :] = patch
                    patch = padded_patch
                
                # Process patch
                try:
                    denoised_patch = model.predict(np.expand_dims(patch, axis=0), verbose=0)[0]
                    denoised_patch = denoised_patch[:y_end-y_start, :x_end-x_start, :]
                    
                    # Use constant weight at corners for stability
                    constant_weight = np.ones((y_end-y_start, x_end-x_start, 1), dtype=np.float32) * 0.5
                    
                    # Only add to zero-weight pixels
                    zero_mask = (weight_map[y_start:y_end, x_start:x_end] == 0)
                    # Expand zero_mask to match the channel dimension
                    expanded_zero_mask = np.repeat(zero_mask, 3, axis=2)
                    
                    # Add contribution only to zero-weight areas
                    denoised_image[y_start:y_end, x_start:x_end][expanded_zero_mask] = denoised_patch[expanded_zero_mask[:, :, 0:3]]
                    weight_map[y_start:y_end, x_start:x_end][zero_mask] = constant_weight[zero_mask]
                except Exception as e:
                    print(f"  Error during corner processing: {e}")
        
        # If there are still zero weights, use a default value
        if np.any(weight_map == 0):
            remaining_zeros = np.sum(weight_map == 0)
            print(f"  Still have {remaining_zeros} zero-weight pixels. Using nearest neighbor fill.")
            
            # Create a mask of zero values
            zero_mask = (weight_map[:, :, 0] == 0)
            
            # Find positions of all zero pixels
            zero_positions = np.where(zero_mask)
            
            # For each zero pixel, copy the value from the nearest non-zero pixel
            for y, x in zip(zero_positions[0], zero_positions[1]):
                # Simple approach: Search in expanding circles for a non-zero pixel
                found = False
                for radius in range(1, 10):  # Limit search radius to avoid excessive computation
                    if found:
                        break
                    for dy in range(-radius, radius + 1):
                        if found:
                            break
                        for dx in range(-radius, radius + 1):
                            ny, nx = y + dy, x + dx
                            if (0 <= ny < h and 0 <= nx < w and 
                                not zero_mask[ny, nx] and
                                abs(dx) + abs(dy) <= radius):
                                # Copy value from this non-zero pixel
                                denoised_image[y, x] = denoised_image[ny, nx]
                                weight_map[y, x] = 0.5  # Set a reasonable weight
                                found = True
                                break
                
                # If still not found, just use a default value
                if not found:
                    denoised_image[y, x] = 0
                    weight_map[y, x] = 1
    
    # Normalize output image by weights
    print("Normalizing output image by weights")
    weight_map = np.repeat(weight_map, 3, axis=2)  # Expand to match 3 channels
    denoised_image = denoised_image / weight_map
    
    # Ensure output stays in valid range
    denoised_image = np.clip(denoised_image, -1.0, 1.0)
    
    print(f"Final denoised image shape: {denoised_image.shape}")
    return denoised_image

def main():
    print("Starting main function...")
    # 1. Load the fine-tuned encoder model
    print(f"Loading the fine-tuned encoder model from {MODEL_DIR}...")
    try:
        # Try different model filenames that might exist
        model_candidates = [
            MODEL_DIR / "ultrahd_denoiser_v7_final.keras",
            MODEL_DIR / "ultrahd_denoiser_v7_best.keras",
            MODEL_DIR / "simple_denoiser_v7_final.keras",
            MODEL_DIR / "simple_denoiser_v7_best.keras",
            MODEL_DIR / "crystal_clear_denoiser_final.keras",
            MODEL_DIR / "ultrahd_denoiser_best.keras",
            MODEL_DIR / "best_model.keras",
            MODEL_DIR / "fine_tuned_denoising_autoencoder_final.keras",
            MODEL_DIR / "fine_tuned_denoising_autoencoder_best.keras",
            MODEL_DIR / "enhanced_denoising_autoencoder.keras",
            MODEL_DIR / "fine_tuned_denoising_autoencoder_final.h5",
            MODEL_DIR / "fine_tuned_denoising_autoencoder_best.h5"
        ]
        
        model = None
        model_path = None
        for candidate in model_candidates:
            if candidate.exists():
                model_path = candidate
                print(f"Found model at {model_path}, attempting to load...")
                model = load_model(model_path, compile=False)  # Added compile=False for compatibility
                break
        
        if model is None:
            # Look for any model file if the specific ones aren't found
            model_files = list(MODEL_DIR.glob("*.h5")) + list(MODEL_DIR.glob("*.keras"))
            if model_files:
                model_path = model_files[0]
                print(f"Found model at {model_path}, attempting to load...")
                model = load_model(model_path, compile=False)  # Added compile=False for compatibility
            else:
                raise FileNotFoundError("No model files found in the specified directory")
        
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Print model summary
    model.summary()
    
    # Get the model's expected input size
    input_shape = model.input_shape
    expected_height, expected_width = input_shape[1], input_shape[2]
    print(f"Model expects input size: {expected_height}x{expected_width}")
    global PATCH_SIZE
    PATCH_SIZE = expected_height  # Update patch size based on model
    
    # 2. Download test images
    try:
        print("Attempting to download test images...")
        image_files = download_test_images()
        print(f"Downloaded {len(image_files)} test images")
    except Exception as e:
        print(f"Error downloading test images: {e}")
        return
    
    # Create a simple test image if needed for debugging
    if not image_files:
        print("Creating a test image for debugging")
        test_img_path = TEST_DATA_DIR / "test_image.png"
        test_img = np.ones((384, 384, 3), dtype=np.float32) * 0.5
        plt.imsave(str(test_img_path), test_img)
        image_files = [test_img_path]
    
    # 3. Define noise types and parameters to test
    noise_types = {
        'gaussian_low': {'type': 'gaussian', 'params': {'std': 0.3}},
        'gaussian_high': {'type': 'gaussian', 'params': {'std': 0.5}},
        'salt_pepper': {'type': 'salt_pepper', 'params': {'amount': 0.3}},
        'speckle': {'type': 'speckle', 'params': {'var': 0.4}},
        'line_pattern': {'type': 'line_pattern', 'params': {'intensity': 0.4}}
    }
    
    # 4. Process each image with each noise type
    all_metrics = {noise_name: [] for noise_name in noise_types.keys()}
    image_paths = {noise_name: [] for noise_name in noise_types.keys()}
    
    # Process up to 3 images for a comprehensive test
    for img_idx, img_file in enumerate(image_files[:3]):
        print(f"Processing image {img_idx+1}/{min(3, len(image_files))}: {img_file.name}")
        
        try:
            # Load and preprocess original image
            original_img = load_and_preprocess_image(img_file)
            print(f"Original image loaded: shape={original_img.shape}, dtype={original_img.dtype}")
            
            # Save original image (convert to [0,1] for saving)
            original_display = postprocess_for_display(original_img)
            original_path = ORIGINAL_DIR / f"original_{img_idx}.png"
            save_image(original_display, original_path)
            print(f"Original image saved to {original_path}")
            
            # Process with each noise type
            for noise_name, noise_config in noise_types.items():
                print(f"  Applying {noise_name} noise...")
                
                # Apply noise
                noisy_img = apply_noise(
                    original_img, 
                    noise_type=noise_config['type'], 
                    noise_params=noise_config['params']
                )
                print(f"  Noisy image created: shape={noisy_img.shape}, dtype={noisy_img.dtype}")
                
                # Save noisy image (convert to [0,1] for saving)
                noisy_display = postprocess_for_display(noisy_img)
                noisy_path = NOISY_DIR / f"{noise_name}_noisy_{img_idx}.png"
                save_image(noisy_display, noisy_path)
                print(f"  Noisy image saved to {noisy_path}")
                
                # Denoise image
                start_time = time.time()
                
                try:
                    # Check if image size matches the model's expected input
                    if noisy_img.shape[0] != expected_height or noisy_img.shape[1] != expected_width:
                        print(f"  Image size ({noisy_img.shape[0]}x{noisy_img.shape[1]}) doesn't match model input size ({expected_height}x{expected_width})")
                        print(f"  Using patch-based approach")
                        # Use patch-based approach for images of different size
                        denoised_img = denoise_image_patches(model, noisy_img, expected_height, PATCH_OVERLAP)
                    else:
                        print(f"  Image size matches model input size, using direct prediction")
                        # Direct processing for properly sized images
                        denoised_img = model.predict(np.expand_dims(noisy_img, axis=0))[0]
                        
                    processing_time = time.time() - start_time
                    print(f"  Processing completed in {processing_time:.2f} seconds")
                    
                    # Save denoised image (convert to [0,1] for saving)
                    denoised_display = postprocess_for_display(denoised_img)
                    denoised_path = DENOISED_DIR / f"{noise_name}_denoised_{img_idx}.png"
                    save_image(denoised_display, denoised_path)
                    print(f"  Denoised image saved to {denoised_path}")
                    
                    # Calculate metrics
                    metrics = calculate_metrics(original_img, denoised_img, noisy_img)
                    metrics['processing_time'] = processing_time
                    all_metrics[noise_name].append(metrics)
                    
                    # Save paths for later visualization
                    image_paths[noise_name].append({
                        'original': str(original_path),
                        'noisy': str(noisy_path),
                        'denoised': str(denoised_path)
                    })
                    
                    # Create individual result visualization
                    viz_path = METRICS_DIR / f"{noise_name}_results_{img_idx}.png"
                    visualize_results(original_img, noisy_img, denoised_img, metrics, viz_path)
                    print(f"  Results visualization saved to {viz_path}")
                    
                except Exception as e:
                    print(f"  Error during denoising: {e}")
                    import traceback
                    traceback.print_exc()
        except Exception as e:
            print(f"Error processing image {img_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 5. Generate summary visualizations and reports
    try:
        print("Generating summary visualizations and reports...")
        
        if all(len(metrics) > 0 for metrics in all_metrics.values()):
            # Performance by noise type
            noise_comparison_path = METRICS_DIR / "noise_type_comparison.png"
            visualize_performance_by_noise_type(all_metrics, noise_comparison_path)
            print(f"Noise type comparison saved to {noise_comparison_path}")
            
            # Best and worst case visualization
            best_worst_path = METRICS_DIR / "best_worst_cases.png"
            visualize_best_and_worst(all_metrics, list(noise_types.keys()), image_paths, best_worst_path)
            print(f"Best/worst case visualization saved to {best_worst_path}")
        else:
            print("Warning: Not enough metrics collected to generate summary visualizations")
        
        # Create simple summary report even if testing was limited
        report_path = RESULTS_DIR / "fine_tuned_validation_summary.txt"
        with open(report_path, 'w') as f:
            f.write("FINE-TUNED ENCODER MODEL VALIDATION REPORT (PATCH-BASED INFERENCE)\n")
            f.write("=================================================\n\n")
            
            f.write(f"Model: {MODEL_DIR}\n")
            f.write(f"Expected input size: {expected_height}x{expected_width}\n\n")
            
            if all(len(metrics) > 0 for metrics in all_metrics.values()):
                f.write("SUMMARY BY NOISE TYPE\n")
                f.write("-----------------\n")
                for noise_type in noise_types.keys():
                    metrics = all_metrics[noise_type]
                    if not metrics:
                        f.write(f"\n{noise_type.upper()}: No data available\n")
                        continue
                        
                    avg_psnr = np.mean([m['psnr'] for m in metrics])
                    avg_ssim = np.mean([m['ssim'] for m in metrics])
                    avg_psnr_improvement = np.mean([m['psnr_improvement'] for m in metrics])
                    avg_ssim_improvement = np.mean([m['ssim_improvement'] for m in metrics])
                    avg_time = np.mean([m['processing_time'] for m in metrics])
                    
                    # Add edge preservation metric if available
                    if 'edge_preservation' in metrics[0]:
                        avg_edge = np.mean([m['edge_preservation'] for m in metrics])
                        avg_edge_improvement = np.mean([m.get('edge_improvement', 0) for m in metrics])
                        edge_info = f"  Average Edge Preservation: {avg_edge:.4f}\n  Edge Preservation Improvement: +{avg_edge_improvement:.4f}\n"
                    else:
                        edge_info = ""
                    
                    f.write(f"\n{noise_type.upper()}:\n")
                    f.write(f"  Average PSNR: {avg_psnr:.2f} dB\n")
                    f.write(f"  Average SSIM: {avg_ssim:.4f}\n")
                    f.write(f"  Average PSNR Improvement: +{avg_psnr_improvement:.2f} dB\n")
                    f.write(f"  Average SSIM Improvement: +{avg_ssim_improvement:.4f}\n")
                    f.write(edge_info)
                    f.write(f"  Average Processing Time: {avg_time:.4f} seconds\n")
                
                # Overall statistics
                all_psnr = [m['psnr'] for noise_type in all_metrics.keys() for m in all_metrics[noise_type] if m]
                all_ssim = [m['ssim'] for noise_type in all_metrics.keys() for m in all_metrics[noise_type] if m]
                all_psnr_improvement = [m['psnr_improvement'] for noise_type in all_metrics.keys() for m in all_metrics[noise_type] if m]
                all_ssim_improvement = [m['ssim_improvement'] for noise_type in all_metrics.keys() for m in all_metrics[noise_type] if m]
                
                if all_psnr:  # Check if we have any metrics
                    f.write("\nOVERALL PERFORMANCE:\n")
                    f.write("------------------\n")
                    f.write(f"  Average PSNR: {np.mean(all_psnr):.2f} dB\n")
                    f.write(f"  Average SSIM: {np.mean(all_ssim):.4f}\n")
                    f.write(f"  Average PSNR Improvement: +{np.mean(all_psnr_improvement):.2f} dB\n")
                    f.write(f"  Average SSIM Improvement: +{np.mean(all_ssim_improvement):.4f}\n")
            else:
                f.write("TESTING RESULTS:\n")
                f.write("--------------\n")
                f.write("Limited testing was performed, primarily for validating the patch-based approach.\n")
            
            f.write("\nPATCH-BASED INFERENCE APPROACH:\n")
            f.write("-----------------------------\n")
            f.write("1. Successfully implemented a patch-based approach for handling images of any size\n")
            f.write("2. Images are divided into overlapping patches to avoid boundary artifacts\n")
            f.write("3. Weighted blending is applied to ensure smooth transitions between patches\n")
            f.write("4. This approach allows the model to process images larger than its training size\n")
            
        print(f"Report saved to {report_path}")
        print(f"Validation complete! Results saved to {RESULTS_DIR}")
    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        print("=" * 50)
        print(" TESTING FINE-TUNED ULTRAHD MODEL (MODEL 7)")
        print("=" * 50)
        print(f"\nModel directory: {MODEL_DIR}")
        print(f"Results will be saved to: {RESULTS_DIR}\n")
        
        # Check if TensorFlow GPU is available
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"Testing with {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
            # Set memory growth for stability
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"Memory growth enabled for {gpu.name}")
                except:
                    print(f"Could not set memory growth for {gpu.name}")
        else:
            print("No GPU detected. Testing will run on CPU (may be slower)")
        
        # Run main test function
        main()
        
        print("\nTest completed successfully!")
        print(f"View results in {RESULTS_DIR}")
        
    except Exception as e:
        import traceback
        print(f"\nError during testing: {e}")
        traceback.print_exc()
        print("\nAttempting to save partial results...")
        
        # Try to create a basic report even if testing failed
        try:
            report_path = RESULTS_DIR / "error_report.txt"
            with open(report_path, 'w') as f:
                f.write("FINE-TUNED ENCODER MODEL VALIDATION ERROR REPORT\n")
                f.write("============================================\n\n")
                f.write(f"Error occurred during testing: {str(e)}\n\n")
                f.write("Traceback:\n")
                import traceback
                traceback.print_exc(file=f)
                f.write("\nPlease check the code and try again.\n")
            print(f"Error report saved to {report_path}")
        except:
            print("Could not save error report.")
        
        print("\nTest completed with errors.") 