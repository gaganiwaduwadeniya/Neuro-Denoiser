import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model, optimizers, callbacks, regularizers
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from pathlib import Path
import time
import random
import cv2
import gc
import shutil
import tarfile
import urllib.request
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import zipfile
from skimage.util import view_as_windows
from scipy import ndimage

# Memory optimization: Configure TensorFlow memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"Error configuring GPUs: {e}")

# Set random seed for reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Enhanced configuration for higher PSNR (>32dB) and SSIM
BATCH_SIZE = 4  # Increased for better training dynamics
IMG_SIZE = 256  # Patch size for training
NOISE_LEVEL_MIN = 0.02  # Slightly higher minimum noise
NOISE_LEVEL_MAX = 0.5   # Higher maximum noise for better robustness
EPOCHS = 150            # More epochs for better convergence
INITIAL_LEARNING_RATE = 5e-5  # Adjusted learning rate 
MIN_LEARNING_RATE = 1e-7      # Minimum learning rate floor
PATIENCE = 20           # Increased patience for early stopping
MEMORY_LIMIT = 0.9      # Limit memory usage to 90% of available
BASE_DIR = Path("UltraDenoiser_v7")

# Dataset params - improved
MAX_TOTAL_PATCHES = 5000  # Increased patch count for better training
TRAIN_VALIDATION_SPLIT = 0.9  # 90% training, 10% validation
PATCH_STRIDE = 128  # Stride for extracting patches (50% overlap)

# Patch-based denoising parameters
PATCH_SIZE_INFERENCE = 256  # Size of patches for inference
PATCH_OVERLAP = 32  # Overlap between patches to avoid boundary artifacts

# Create necessary directories
os.makedirs(BASE_DIR, exist_ok=True)
MODELS_DIR = BASE_DIR / "models"
SAMPLES_DIR = BASE_DIR / "samples"
LOGS_DIR = BASE_DIR / "logs"
DATASET_DIR = BASE_DIR / "datasets"
BSDS_DIR = DATASET_DIR / "BSDS500"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(BSDS_DIR, exist_ok=True)

# Generate synthetic training data if needed
GENERATE_SYNTHETIC_DATA = True  # Enable synthetic data generation
SYNTHETIC_DATA_SIZE = 500  # Number of synthetic images

# Download BSDS500 dataset if needed
DOWNLOAD_BSDS = True  # Set to True to download BSDS500 dataset

# Find existing model to fine-tune
def find_existing_model():
    """Find the most recent model to fine-tune"""
    model_paths = [
        "Fine_Tuned_Model_6/ultrahd_denoiser_v6_final.keras",
        "Fine_Tuned_Model_6/ultrahd_denoiser_v6_best.keras",
        "Python/Fine_Tuned_Model_6/ultrahd_denoiser_v6_final.keras",
        "Python/Fine_Tuned_Model_6/ultrahd_denoiser_v6_best.keras",
        # Add v5 models
        "Fine_Tuned_Model_5/crystal_clear_denoiser_final.keras",
        "Fine_Tuned_Model_5/ultrahd_denoiser_best.keras",
        "Python/Fine_Tuned_Model_5/crystal_clear_denoiser_final.keras",
        "Python/Fine_Tuned_Model_5/ultrahd_denoiser_best.keras",
        # Add .h5 versions
        "Fine_Tuned_Model_6/ultrahd_denoiser_v6_final.h5",
        "Fine_Tuned_Model_6/ultrahd_denoiser_v6_best.h5",
        "Python/Fine_Tuned_Model_6/ultrahd_denoiser_v6_final.h5",
        "Python/Fine_Tuned_Model_6/ultrahd_denoiser_v6_best.h5",
        "Fine_Tuned_Model_5/crystal_clear_denoiser_final.h5",
        "Fine_Tuned_Model_5/ultrahd_denoiser_best.h5",
        "Python/Fine_Tuned_Model_5/crystal_clear_denoiser_final.h5",
        "Python/Fine_Tuned_Model_5/ultrahd_denoiser_best.h5"
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            print(f"Found existing model at: {path}")
            return path
    
    # Fallback to earlier versions if needed
    for version in range(6, 0, -1):
        path = f"Fine_Tuned_Model_{version}/ultrahd_denoiser_v{version}_final.keras"
        if os.path.exists(path):
            print(f"Found earlier model version at: {path}")
            return path
    
    print("No existing model found. Will create a new model from scratch.")
    return None

class BSDSDatasetHandler:
    """Handler for downloading and preparing the BSDS500 dataset"""
    
    def __init__(self, dataset_dir=BSDS_DIR):
        self.dataset_dir = dataset_dir
        
    def download_and_extract(self):
        """Download and extract the BSDS500 dataset"""
        if self.is_dataset_available():
            print("BSDS500 dataset is already available.")
            return True
        
        print("Downloading BSDS500 dataset...")
        url = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
        try:
            # First, try direct download
            download_path = os.path.join(self.dataset_dir, "BSR_bsds500.tgz")
            urllib.request.urlretrieve(url, download_path)
            
            # Extract the archive
            print("Extracting BSDS500 dataset...")
            with tarfile.open(download_path, "r:gz") as tar:
                # Extract only images folder to save space
                for member in tar.getmembers():
                    if "images" in member.name:
                        tar.extract(member, self.dataset_dir)
            
            # Clean up
            os.remove(download_path)
            print("BSDS500 dataset downloaded and extracted successfully.")
            return True
            
        except Exception as e:
            print(f"Error downloading BSDS500 dataset: {e}")
            
            # Fallback: Try alternative sources or create placeholder
            print("Attempting to download from alternative source...")
            alt_url = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
            
            try:
                download_path = os.path.join(self.dataset_dir, "BSDS300-images.tgz")
                urllib.request.urlretrieve(alt_url, download_path)
                
                with tarfile.open(download_path, "r:gz") as tar:
                    tar.extractall(self.dataset_dir)
                
                os.remove(download_path)
                print("BSDS300 dataset downloaded as fallback.")
                return True
                
            except Exception as alt_error:
                print(f"Error downloading alternative dataset: {alt_error}")
                print("Will use synthetic data and any available images instead.")
                return False
    
    def is_dataset_available(self):
        """Check if the BSDS dataset is already available"""
        # Check for BSDS500 directory structure
        bsds_images_dir = os.path.join(self.dataset_dir, "BSR", "BSDS500", "data", "images")
        alt_images_dir = os.path.join(self.dataset_dir, "BSDS300", "images")
        
        # Check for any images in the dataset directory (direct or subdirectories)
        image_files = []
        for root, dirs, files in os.walk(self.dataset_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(root, file))
                    if len(image_files) >= 10:  # Found enough images
                        return True
        
        return os.path.isdir(bsds_images_dir) or os.path.isdir(alt_images_dir) or len(image_files) >= 10
    
    def get_image_paths(self):
        """Get paths to all images in the BSDS dataset"""
        image_paths = []
        
        # Try BSDS500 directory structure
        bsds_images_dir = os.path.join(self.dataset_dir, "BSR", "BSDS500", "data", "images")
        if os.path.isdir(bsds_images_dir):
            for subset in ["train", "test", "val"]:
                subset_dir = os.path.join(bsds_images_dir, subset)
                if os.path.isdir(subset_dir):
                    for file in os.listdir(subset_dir):
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            image_paths.append(os.path.join(subset_dir, file))
        
        # Try BSDS300 directory structure as fallback
        alt_images_dir = os.path.join(self.dataset_dir, "BSDS300", "images")
        if os.path.isdir(alt_images_dir):
            for subset in ["train", "test"]:
                subset_dir = os.path.join(alt_images_dir, subset)
                if os.path.isdir(subset_dir):
                    for file in os.listdir(subset_dir):
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            image_paths.append(os.path.join(subset_dir, file))
        
        # If still no images found, search entire dataset directory
        if not image_paths:
            for root, dirs, files in os.walk(self.dataset_dir):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_paths.append(os.path.join(root, file))
        
        print(f"Found {len(image_paths)} images in BSDS dataset.")
        return image_paths

class MemoryEfficientDataLoader:
    """Memory-optimized data loader with patch-based approach for denoising training"""
    
    def __init__(self, img_size=IMG_SIZE, patch_stride=PATCH_STRIDE, noise_min=NOISE_LEVEL_MIN, noise_max=NOISE_LEVEL_MAX):
        self.img_size = img_size
        self.patch_stride = patch_stride
        self.noise_min = noise_min
        self.noise_max = noise_max
        
        # Initialize BSDS dataset handler
        self.bsds_handler = BSDSDatasetHandler()
        
        # Download BSDS500 if needed
        if DOWNLOAD_BSDS:
            self.bsds_handler.download_and_extract()
    
    def extract_patches(self, img, patch_size=None, stride=None):
        """Extract patches from an image using sliding window"""
        if patch_size is None:
            patch_size = self.img_size
        if stride is None:
            stride = self.patch_stride
            
        # Get image dimensions
        h, w = img.shape[:2]
        
        # Skip if image is too small
        if h < patch_size or w < patch_size:
            return []
        
        # Extract patches with skimage's view_as_windows
        window_shape = (patch_size, patch_size, 3) if img.ndim == 3 else (patch_size, patch_size)
        step = (stride, stride, 1) if img.ndim == 3 else (stride, stride)
        
        try:
            patches = view_as_windows(img, window_shape, step)
            # Reshape to get a list of patches
            patches_shape = patches.shape
            if img.ndim == 3:
                patches = patches.reshape(patches_shape[0] * patches_shape[1], patch_size, patch_size, 3)
            else:
                patches = patches.reshape(patches_shape[0] * patches_shape[1], patch_size, patch_size)
                
        except ValueError as e:
            print(f"Error extracting patches: {e}")
            # Fallback to manual patch extraction
            patches = []
            for y in range(0, h - patch_size + 1, stride):
                for x in range(0, w - patch_size + 1, stride):
                    patch = img[y:y + patch_size, x:x + patch_size]
                    patches.append(patch)
            
            if patches:
                patches = np.array(patches)
            
        return patches
    
    def load_images(self, limit=MAX_TOTAL_PATCHES):
        """Load and prepare images with memory efficiency and patch extraction"""
        print("Loading images and extracting patches with memory optimization...")
        
        # Collect potential image paths from various sources
        image_paths = []
        
        # First, try to get images from BSDS dataset
        bsds_paths = self.bsds_handler.get_image_paths()
        if bsds_paths:
            print(f"Using {len(bsds_paths)} images from BSDS dataset")
            image_paths.extend(bsds_paths)
        
        # Look in various directories for additional datasets
        data_dirs = [
            "Python/Fine_Tuned_Model_6/data",
            "Fine_Tuned_Model_6/data",
            "Python/Fine_Tuned_Model_5/data",
            "Fine_Tuned_Model_5/data",
            "data",
            "datasets",
            "images",
            "test_data"
        ]
        
        # Add more potential directories to search
        current_dir = os.getcwd()
        parent_dir = os.path.dirname(current_dir)
        
        # Also check in the current and parent directories
        additional_dirs = [
            current_dir,
            parent_dir,
            os.path.join(current_dir, "Python"),
            os.path.join(parent_dir, "Python"),
            os.path.join(current_dir, "data"),
            os.path.join(parent_dir, "data")
        ]
        
        data_dirs.extend(additional_dirs)
        
        # Check if any directory in data_dirs actually exists
        existing_dirs = [d for d in data_dirs if os.path.exists(d) and os.path.isdir(d)]
        
        if existing_dirs:
            print(f"Scanning additional directories for more images...")
            
            for data_dir in existing_dirs:
                for root, _, files in os.walk(data_dir):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                            file_path = os.path.join(root, file)
                            if file_path not in image_paths:  # Avoid duplicates
                                image_paths.append(file_path)
            
            # Shuffle paths to ensure diverse dataset
            random.shuffle(image_paths)
            
            print(f"Found total of {len(image_paths)} potential images from all sources")
        else:
            print("Warning: No additional data directories found beyond BSDS")
        
        # If no images found or insufficient, generate synthetic data
        if len(image_paths) < 10 and GENERATE_SYNTHETIC_DATA:
            print("Insufficient real images found. Generating synthetic data...")
            return self.generate_synthetic_data(num_images=SYNTHETIC_DATA_SIZE)
        
        # Process images and extract patches
        all_patches = []
        skipped_images = 0
        processed_images = 0
        
        for img_path in tqdm(image_paths):
            if len(all_patches) >= limit:
                print(f"Reached patch limit ({limit}). Stopping extraction.")
                break
                
            try:
                # Load and process image
                img = cv2.imread(img_path)
                if img is None:
                    skipped_images += 1
                    continue
                    
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Skip if image is too small
                if img.shape[0] < self.img_size or img.shape[1] < self.img_size:
                    skipped_images += 1
                    continue
                
                # Extract patches using sliding window
                patches = self.extract_patches(img, self.img_size, self.patch_stride)
                
                # Skip if no patches were extracted
                if len(patches) == 0:
                    skipped_images += 1
                    continue
                
                # Convert to float and normalize to [-1, 1]
                patches = patches.astype(np.float32) / 127.5 - 1.0
                
                # Apply random augmentations to some patches (flip/rotate)
                augmented_patches = []
                for patch in patches:
                    # Randomly decide whether to augment
                    if np.random.random() < 0.3:
                        # Random flip (horizontal or vertical)
                        if np.random.random() < 0.5:
                            patch = np.fliplr(patch)
                        else:
                            patch = np.flipud(patch)
                    
                    # Randomly decide whether to rotate
                    if np.random.random() < 0.3:
                        # Random rotation (90, 180, or 270 degrees)
                        k = np.random.randint(1, 4)
                        patch = np.rot90(patch, k)
                    
                    augmented_patches.append(patch)
                
                all_patches.extend(augmented_patches)
                processed_images += 1
                
                # Memory optimization: periodic garbage collection
                if processed_images % 50 == 0:
                    gc.collect()
                    print(f"Processed {processed_images} images, extracted {len(all_patches)} patches so far")
                
            except Exception as e:
                skipped_images += 1
                print(f"Error processing {img_path}: {e}")
                continue
            
            # Limit patches to avoid memory issues
            if len(all_patches) >= limit:
                all_patches = all_patches[:limit]
                break
        
        print(f"Extracted {len(all_patches)} patches from {processed_images} images, skipped {skipped_images} images")
        
        # If insufficient patches were extracted, augment with synthetic data
        if len(all_patches) < 100 and GENERATE_SYNTHETIC_DATA:
            print(f"Only {len(all_patches)} patches extracted from real images. Augmenting with synthetic data...")
            synthetic_patches = self.generate_synthetic_data(num_images=min(500, limit - len(all_patches)))
            all_patches.extend(synthetic_patches[:limit - len(all_patches)])
            print(f"Added synthetic patches. New total: {len(all_patches)}")
        
        # Convert to numpy array
        all_patches = np.array(all_patches, dtype=np.float32)
        
        return all_patches 

    def generate_synthetic_data(self, num_images=500):
        """Generate synthetic images for training when no real data is available"""
        print(f"Generating {num_images} synthetic training images...")
        
        synthetic_patches = []
        
        for i in tqdm(range(num_images)):
            # Generate synthetic clean image types with more variety
            image_type = random.choice(['gradient', 'pattern', 'shape', 'noise', 'fractal'])
            
            if image_type == 'gradient':
                # Random gradient image with improved variety
                direction = random.choice(['horizontal', 'vertical', 'diagonal', 'radial'])
                color1 = np.array([random.random(), random.random(), random.random()], dtype=np.float32) * 2 - 1
                color2 = np.array([random.random(), random.random(), random.random()], dtype=np.float32) * 2 - 1
                
                x = np.linspace(-1, 1, self.img_size, dtype=np.float32)
                y = np.linspace(-1, 1, self.img_size, dtype=np.float32)
                xx, yy = np.meshgrid(x, y)
                
                if direction == 'horizontal':
                    gradient = xx
                elif direction == 'vertical':
                    gradient = yy
                elif direction == 'diagonal':
                    gradient = (xx + yy) / 2
                else:  # radial
                    gradient = np.sqrt(xx**2 + yy**2)
                    # Normalize to [-1, 1]
                    gradient = gradient / np.max(gradient) * 2 - 1
                
                # Normalize gradient to [-1, 1]
                if direction != 'radial':
                    gradient = (gradient - np.min(gradient)) / (np.max(gradient) - np.min(gradient)) * 2 - 1
                
                # Create RGB image
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
                for c in range(3):
                    img[:, :, c] = color1[c] + (color2[c] - color1[c]) * (gradient + 1) / 2
                
            elif image_type == 'pattern':
                # Create pattern image with more variations
                pattern_type = random.choice(['checkerboard', 'stripes', 'circles', 'waves', 'dots'])
                
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
                
                if pattern_type == 'checkerboard':
                    check_size = random.randint(8, 32)
                    xx, yy = np.meshgrid(range(self.img_size), range(self.img_size))
                    pattern = ((xx // check_size) % 2) ^ ((yy // check_size) % 2)
                    for c in range(3):
                        intensity = np.float32(random.uniform(-0.8, 0.8))
                        img[:, :, c] = pattern * intensity
                        
                elif pattern_type == 'stripes':
                    stripe_width = random.randint(4, 20)
                    xx = np.arange(self.img_size)
                    pattern = ((xx // stripe_width) % 2)
                    for c in range(3):
                        intensity = np.float32(random.uniform(-0.8, 0.8))
                        img[:, :, c] = pattern * intensity
                        
                elif pattern_type == 'circles':
                    xx, yy = np.meshgrid(range(self.img_size), range(self.img_size))
                    center_x, center_y = self.img_size // 2, self.img_size // 2
                    dist = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)
                    circle_size = random.randint(10, 30)
                    pattern = ((dist // circle_size) % 2)
                    
                    for c in range(3):
                        intensity = np.float32(random.uniform(-0.8, 0.8))
                        img[:, :, c] = pattern * intensity
                
                elif pattern_type == 'waves':
                    # Sine wave patterns
                    xx, yy = np.meshgrid(range(self.img_size), range(self.img_size))
                    freq = random.uniform(0.01, 0.1)
                    phase = random.uniform(0, 2 * np.pi)
                    
                    pattern = np.sin(xx * freq + phase) * np.cos(yy * freq)
                    pattern = (pattern - np.min(pattern)) / (np.max(pattern) - np.min(pattern)) * 2 - 1
                    
                    for c in range(3):
                        intensity = np.float32(random.uniform(0.3, 1.0))
                        img[:, :, c] = pattern * intensity
                
                else:  # dots
                    num_dots = random.randint(20, 100)
                    for _ in range(num_dots):
                        x, y = random.randint(0, self.img_size-1), random.randint(0, self.img_size-1)
                        radius = random.randint(2, 8)
                        color = [random.uniform(-1, 1) for _ in range(3)]
                        
                        # Create circular dots
                        for dx in range(-radius, radius+1):
                            for dy in range(-radius, radius+1):
                                if dx**2 + dy**2 <= radius**2:
                                    nx, ny = x + dx, y + dy
                                    if 0 <= nx < self.img_size and 0 <= ny < self.img_size:
                                        img[ny, nx] = color
            
            elif image_type == 'shape':
                # Random shapes
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
                color = np.array([random.random(), random.random(), random.random()], dtype=np.float32) * 2 - 1
                
                # Fill with base color
                base_color = np.array([random.random(), random.random(), random.random()], dtype=np.float32) * 2 - 1
                img[:, :, :] = base_color
                
                # Add random shapes
                for _ in range(random.randint(3, 10)):
                    shape_type = random.choice(['rectangle', 'circle', 'triangle', 'ellipse'])
                    color = np.array([random.random(), random.random(), random.random()], dtype=np.float32) * 2 - 1
                    
                    if shape_type == 'rectangle':
                        x1, y1 = random.randint(0, self.img_size-20), random.randint(0, self.img_size-20)
                        width = random.randint(20, self.img_size//2)
                        height = random.randint(20, self.img_size//2)
                        x2, y2 = min(x1 + width, self.img_size-1), min(y1 + height, self.img_size-1)
                        
                        img[y1:y2, x1:x2, :] = color
                        
                    elif shape_type == 'circle':
                        center_x = random.randint(20, self.img_size-20)
                        center_y = random.randint(20, self.img_size-20)
                        radius = random.randint(10, self.img_size//4)
                        
                        yy, xx = np.ogrid[:self.img_size, :self.img_size]
                        dist = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)
                        mask = dist <= radius
                        
                        for c in range(3):
                            img[:, :, c][mask] = color[c]
                    
                    elif shape_type == 'triangle':
                        # Create a random triangle
                        points = np.array([
                            [random.randint(0, self.img_size), random.randint(0, self.img_size)],
                            [random.randint(0, self.img_size), random.randint(0, self.img_size)],
                            [random.randint(0, self.img_size), random.randint(0, self.img_size)]
                        ])
                        
                        # Create mask for triangle
                        mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
                        cv2.fillPoly(mask, [points], 1)
                        
                        for c in range(3):
                            img[:, :, c][mask == 1] = color[c]
                    
                    else:  # ellipse
                        center_x = random.randint(20, self.img_size-20)
                        center_y = random.randint(20, self.img_size-20)
                        width = random.randint(10, self.img_size//3)
                        height = random.randint(10, self.img_size//3)
                        angle = random.randint(0, 180)
                        
                        # Create mask for ellipse
                        mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
                        cv2.ellipse(mask, (center_x, center_y), (width, height), angle, 0, 360, 1, -1)
                        
                        for c in range(3):
                            img[:, :, c][mask == 1] = color[c]
            
            elif image_type == 'noise':
                # Pure generated noise patterns
                noise_type = random.choice(['gaussian', 'perlin', 'binary', 'speckle'])
                
                if noise_type == 'gaussian':
                    # Colored Gaussian noise
                    img = np.random.normal(0, 0.5, (self.img_size, self.img_size, 3)).astype(np.float32)
                    # Blur to create correlations
                    img = cv2.GaussianBlur(img, (5, 5), 0)
                
                elif noise_type == 'perlin':
                    # Generate a crude approximation of Perlin noise
                    small_size = self.img_size // 8
                    small_noise = np.random.normal(0, 1, (small_size, small_size, 3)).astype(np.float32)
                    # Resize to create smooth noise
                    img = cv2.resize(small_noise, (self.img_size, self.img_size))
                    img = img / max(1e-6, np.max(np.abs(img)))  # Normalize to [-1, 1]
                
                elif noise_type == 'binary':
                    # Binary noise pattern
                    img = (np.random.random((self.img_size, self.img_size, 3)) > 0.5).astype(np.float32) * 2 - 1
                
                else:  # speckle
                    # Base image
                    base = np.random.normal(0, 0.2, (self.img_size, self.img_size, 3)).astype(np.float32)
                    # Speckle pattern
                    speckles = (np.random.random((self.img_size, self.img_size)) > 0.995).astype(np.float32)
                    speckle_intensity = np.random.uniform(0.5, 1.0)
                    
                    img = base.copy()
                    for c in range(3):
                        img[:, :, c] += speckles * speckle_intensity
                        
            else:  # fractal
                # Simple fractal-like pattern using recursive subdivision
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
                
                # Start with random corner values
                corners = np.random.uniform(-1, 1, (4, 3)).astype(np.float32)
                
                # Recursive subdivision (simplified)
                def subdivide(x1, y1, x2, y2, depth=0):
                    if depth > 5 or (x2 - x1 < 3) or (y2 - y1 < 3):
                        return
                    
                    # Calculate midpoints
                    mx, my = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    # Set midpoint value with random offset
                    offset = 1.0 / (depth + 1)
                    img[my, mx] = (img[y1, x1] + img[y2, x2]) / 2 + np.random.uniform(-offset, offset, 3)
                    
                    # Recursively subdivide
                    subdivide(x1, y1, mx, my, depth+1)
                    subdivide(mx, y1, x2, my, depth+1)
                    subdivide(x1, my, mx, y2, depth+1)
                    subdivide(mx, my, x2, y2, depth+1)
                
                # Set corner values
                img[0, 0] = corners[0]
                img[0, self.img_size-1] = corners[1]
                img[self.img_size-1, 0] = corners[2]
                img[self.img_size-1, self.img_size-1] = corners[3]
                
                # Initial subdivision
                subdivide(0, 0, self.img_size-1, self.img_size-1)
                
                # Interpolate missing values
                img = cv2.resize(cv2.resize(img, (self.img_size//4, self.img_size//4)), (self.img_size, self.img_size))
            
            # Adjust contrast/brightness randomly
            contrast = np.float32(random.uniform(0.7, 1.3))
            brightness = np.float32(random.uniform(-0.2, 0.2))
            img = img * contrast + brightness
            
            # Apply random blur to some images
            if random.random() < 0.3:
                blur_size = random.choice([3, 5, 7])
                img = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
            
            # Ensure values are within [-1, 1]
            img = np.clip(img, -1.0, 1.0)
            
            # Ensure the image is float32
            img = img.astype(np.float32)
            
            # Extract patches from this synthetic image if it's larger than needed
            if self.img_size > IMG_SIZE:
                patches = self.extract_patches(img, IMG_SIZE)
                if len(patches) > 0:
                    synthetic_patches.extend(patches)
            else:
                synthetic_patches.append(img)
            
            # Memory optimization: periodic garbage collection
            if (i+1) % 100 == 0:
                gc.collect()
        
        # Convert to numpy array with explicit float32 dtype
        synthetic_patches = np.array(synthetic_patches, dtype=np.float32)
        
        print(f"Generated {len(synthetic_patches)} synthetic patches")
        print(f"Data shape: {synthetic_patches.shape}, dtype: {synthetic_patches.dtype}")
        
        # Create a directory to save some samples for inspection
        os.makedirs(f"{BASE_DIR}/synthetic_samples", exist_ok=True)
        
        # Save a few sample synthetic images for inspection
        for i in range(min(5, len(synthetic_patches))):
            # Convert from [-1, 1] to [0, 255] for saving
            sample = ((synthetic_patches[i] + 1) * 127.5).astype(np.uint8)
            cv2.imwrite(f"{BASE_DIR}/synthetic_samples/sample_{i}.png", cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))
        
        return synthetic_patches
    
    def create_tf_dataset(self, patches, batch_size=BATCH_SIZE, is_training=True):
        """Create optimized TensorFlow dataset for training with advanced noise augmentation"""
        # Safety check - ensure patches is not empty
        if len(patches) == 0:
            raise ValueError("No patches available for creating dataset. Check the data loading process.")
        
        # Ensure patches are float32 to avoid type mismatches
        if patches.dtype != np.float32:
            print(f"Converting patches from {patches.dtype} to float32")
            patches = patches.astype(np.float32)
        
        # Memory optimization: use tf.data pipeline for efficient data handling
        
        def add_advanced_noise(clean_img):
            """Add sophisticated noise patterns for better denoising performance"""
            # Ensure input is float32 to prevent type mismatches
            clean_img = tf.cast(clean_img, tf.float32)
            
            # Reshape if needed for adding noise correctly - prevent rank issues
            input_shape = tf.shape(clean_img)
            
            # Reshape the input to 4D if it's not already
            clean_img_4d = tf.reshape(clean_img, [1, input_shape[0], input_shape[1], input_shape[2]])
            
            # Choose from multiple noise types for better robustness
            noise_type = tf.random.uniform(shape=[], minval=0, maxval=5, dtype=tf.int32)
            
            # Base noise level - random but higher than before
            noise_level = tf.random.uniform(
                shape=[], 
                minval=self.noise_min,
                maxval=self.noise_max,
                dtype=tf.float32
            )
            
            # Common Gaussian noise component for all noise types
            base_noise = tf.random.normal(
                shape=tf.shape(clean_img_4d),
                mean=0.0,
                stddev=noise_level,
                dtype=tf.float32
            )
            
            def apply_gaussian_noise():
                # Standard Gaussian noise
                return base_noise
            
            def apply_correlated_noise():
                # Apply spatial correlation to noise for realism
                correlated = tf.nn.avg_pool2d(
                    base_noise,
                    ksize=5,  # Larger kernel for more correlation
                    strides=1,
                    padding='SAME'
                )
                return correlated * 1.5  # Boost intensity
            
            def apply_salt_pepper_noise():
                # Salt and pepper noise
                salt_prob = noise_level * 0.1
                pepper_prob = noise_level * 0.1
                
                # Generate salt mask (bright spots)
                salt_mask = tf.cast(
                    tf.random.uniform(tf.shape(clean_img_4d), dtype=tf.float32) < salt_prob,
                    tf.float32
                )
                
                # Generate pepper mask (dark spots)
                pepper_mask = tf.cast(
                    tf.random.uniform(tf.shape(clean_img_4d), dtype=tf.float32) < pepper_prob,
                    tf.float32
                )
                
                # Apply base noise with salt and pepper
                noisy = base_noise * 0.5  # Reduce base noise
                noisy = noisy + (salt_mask * 2.0) - (pepper_mask * 2.0)
                return noisy
            
            def apply_stripe_noise():
                # Line/stripe pattern noise (horizontal or vertical)
                direction = tf.random.uniform(shape=[], minval=0, maxval=2, dtype=tf.int32)
                
                if direction == 0:  # Horizontal stripes
                    stripe_width = tf.random.uniform(shape=[], minval=1, maxval=5, dtype=tf.int32)
                    stripe_noise = tf.tile(
                        tf.random.normal([1, stripe_width, input_shape[1], input_shape[2]], stddev=noise_level*2),
                        [1, input_shape[0] // stripe_width + 1, 1, 1]
                    )
                    stripe_noise = stripe_noise[:, :input_shape[0], :, :]
                else:  # Vertical stripes
                    stripe_width = tf.random.uniform(shape=[], minval=1, maxval=5, dtype=tf.int32)
                    stripe_noise = tf.tile(
                        tf.random.normal([1, input_shape[0], stripe_width, input_shape[2]], stddev=noise_level*2),
                        [1, 1, input_shape[1] // stripe_width + 1, 1]
                    )
                    stripe_noise = stripe_noise[:, :, :input_shape[1], :]
                
                return base_noise * 0.3 + stripe_noise  # Mix with base noise
            
            def apply_mixed_noise():
                # Combination of multiple noise types
                gaussian = apply_gaussian_noise() * 0.6
                correlated = apply_correlated_noise() * 0.3
                sp_noise = apply_salt_pepper_noise() * 0.1
                return gaussian + correlated + sp_noise
            
            # Apply the selected noise type
            noise_functions = [
                apply_gaussian_noise,      # 0: Gaussian
                apply_correlated_noise,    # 1: Correlated
                apply_salt_pepper_noise,   # 2: Salt & Pepper
                apply_stripe_noise,        # 3: Stripe pattern
                apply_mixed_noise          # 4: Mixed noise
            ]
            
            final_noise = tf.switch_case(
                noise_type,
                branch_fns=noise_functions
            )
            
            # Add noise to image
            noisy_img = clean_img_4d + final_noise
            
            # Add compression artifacts occasionally (20% chance)
            compress_prob = tf.random.uniform(shape=[], dtype=tf.float32)
            
            def apply_compression():
                # Simulate compression by selective blurring and adding blocking artifacts
                # First apply blur
                blurred = tf.nn.avg_pool2d(
                    noisy_img,
                    ksize=2,
                    strides=1,
                    padding='SAME'
                )
                
                # Add blocking artifacts (simulated JPEG blocks)
                block_size = 8  # Standard JPEG block size
                
                # Create block averaging effect
                block_effect = tf.nn.avg_pool2d(
                    noisy_img,
                    ksize=block_size,
                    strides=block_size,
                    padding='SAME'
                )
                block_effect = tf.image.resize(
                    block_effect, 
                    [input_shape[0], input_shape[1]], 
                    method='nearest'
                )
                
                # Mix original, blurred and blocked versions
                compression_ratio = noise_level * 0.7
                block_ratio = noise_level * 0.3
                
                return noisy_img * (1 - compression_ratio - block_ratio) + \
                       blurred * compression_ratio + \
                       block_effect * block_ratio
            
            # Apply compression artifacts conditionally
            noisy_img = tf.cond(
                compress_prob < 0.2,
                lambda: apply_compression(),
                lambda: noisy_img
            )
            
            # Reshape back to 3D
            noisy_img = tf.reshape(noisy_img, input_shape)
            
            # Ensure values stay in valid range
            noisy_img = tf.clip_by_value(noisy_img, -1.0, 1.0)
            clean_img = tf.clip_by_value(clean_img, -1.0, 1.0)
            
            return noisy_img, clean_img
        
        # Create dataset with memory-efficient pipeline
        dataset = tf.data.Dataset.from_tensor_slices(patches)
        
        # Apply noise augmentation with controlled parallelism 
        dataset = dataset.map(
            add_advanced_noise,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Shuffle only for training
        if is_training:
            buffer_size = min(2000, len(patches))
            dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
        
        # Batch and prefetch for performance
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset 

class HybridResidualAttentionModel:
    """Advanced model architecture with hybrid residual connections and attention mechanisms"""
    
    def __init__(self, img_size=IMG_SIZE):
        self.img_size = img_size
        self.model = None
    
    def build_model(self, fine_tune_base=True):
        """Build an enhanced denoising model targeting PSNR > 32dB"""
        print("Building enhanced denoising model with advanced architecture...")
        
        # Load existing model for fine-tuning
        base_model_path = find_existing_model()
        if base_model_path and os.path.exists(base_model_path):
            print(f"Fine-tuning existing model: {base_model_path}")
            try:
                self.model = load_model(base_model_path, compile=False)
                print(f"Loaded model with input shape: {self.model.input_shape}")
                
                # Fix input shape if needed
                if self.model.input_shape[1:3] != (self.img_size, self.img_size):
                    print(f"Rebuilding model for input shape ({self.img_size}, {self.img_size}, 3)")
                    self.model = None
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = None
        
        # Create new model if needed
        if self.model is None:
            print("Creating new model architecture...")
            
            # Input layer
            inputs = layers.Input(shape=(self.img_size, self.img_size, 3))
            
            # Initial feature extraction
            x = layers.Conv2D(64, 3, padding='same')(inputs)
            x = layers.LeakyReLU(0.2)(x)
            
            # Skip connection for original input (critical for PSNR)
            orig_connection = inputs
            
            # === First Stage: Downsample and Extract Features ===
            # Block 1
            x = self._residual_block(x, 64, kernel_size=3)
            block1_output = x
            
            # Block 2 with attention
            x = layers.MaxPooling2D(2)(x)
            x = self._residual_block(x, 128, kernel_size=3)
            x = self._attention_block(x)
            block2_output = x
            
            # Block 3 with advanced attention
            x = layers.MaxPooling2D(2)(x)
            x = self._residual_block(x, 256, kernel_size=3)
            x = self._advanced_attention_block(x)
            
            # === Bottleneck with Dilated Convolutions ===
            # Add dilated convolutions for larger receptive field without downsampling
            x = self._dilated_block(x, 256)
            
            # === Second Stage: Upsample and Refine ===
            # Block 4 with skip connection from Block 2
            x = layers.UpSampling2D(2, interpolation='bilinear')(x)
            x = layers.Concatenate()([x, block2_output])
            x = self._residual_block(x, 128, kernel_size=3)
            
            # Block 5 with skip connection from Block 1
            x = layers.UpSampling2D(2, interpolation='bilinear')(x)
            x = layers.Concatenate()([x, block1_output])
            x = self._residual_block(x, 64, kernel_size=3)
            
            # Final feature integration with enhanced details
            x = layers.Conv2D(64, 3, padding='same')(x)
            x = layers.LeakyReLU(0.2)(x)
            
            # Add detail-enhancing branch
            detail_branch = layers.Conv2D(32, 3, padding='same')(x)
            detail_branch = layers.LeakyReLU(0.2)(detail_branch)
            detail_branch = layers.Conv2D(32, 3, padding='same')(detail_branch)
            detail_branch = layers.LeakyReLU(0.2)(detail_branch)
            detail_branch = layers.Conv2D(3, 3, padding='same')(detail_branch)
            
            # Main branch
            main_branch = layers.Conv2D(3, 3, padding='same')(x)
            
            # Combine branches with original input
            combined = layers.Add()([main_branch, detail_branch, orig_connection])
            
            # Final output refinement
            outputs = layers.Conv2D(3, 3, padding='same', activation='tanh')(combined)
            
            # Create model
            self.model = Model(inputs, outputs, name="ultrahd_denoiser_v7")
        
        # Configure the model for fine-tuning
        if fine_tune_base:
            # Unfreeze all layers for full fine-tuning
            for layer in self.model.layers:
                layer.trainable = True
        
        # Compile with optimized settings for PSNR/SSIM
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE),
            loss=self._psnr_ssim_combined_loss,
            metrics=['mae']  # Track Mean Absolute Error
        )
        
        # Print model summary
        self.model.summary()
        
        return self.model
    
    def _residual_block(self, x, filters, kernel_size=3):
        """Residual block with batch normalization"""
        # Memory-efficient implementation
        skip = x
        
        # If channels don't match, use 1x1 conv to match dimensions
        if x.shape[-1] != filters:
            skip = layers.Conv2D(filters, 1, padding='same')(x)
        
        # First convolution
        x = layers.Conv2D(
            filters, kernel_size, 
            padding='same',
            kernel_regularizer=regularizers.l2(1e-5)
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        # Second convolution
        x = layers.Conv2D(
            filters, kernel_size, 
            padding='same',
            kernel_regularizer=regularizers.l2(1e-5)
        )(x)
        x = layers.BatchNormalization()(x)
        
        # Add skip connection
        x = layers.Add()([x, skip])
        x = layers.LeakyReLU(0.2)(x)
        
        return x
    
    def _attention_block(self, x):
        """Channel and spatial attention mechanism for better feature selection"""
        # Memory-efficient attention implementation
        
        # Channel attention
        avg_pool = layers.GlobalAveragePooling2D()(x)
        avg_pool = layers.Reshape((1, 1, x.shape[-1]))(avg_pool)
        avg_pool = layers.Conv2D(x.shape[-1]//4, 1, activation='relu')(avg_pool)
        avg_pool = layers.Conv2D(x.shape[-1], 1, activation='sigmoid')(avg_pool)
        
        # Apply channel attention
        x = layers.Multiply()([x, avg_pool])
        
        return x
    
    def _advanced_attention_block(self, x):
        """Enhanced attention with both channel and spatial components"""
        # Channel attention
        avg_pool = layers.GlobalAveragePooling2D()(x)
        max_pool = layers.GlobalMaxPooling2D()(x)
        
        avg_pool = layers.Reshape((1, 1, x.shape[-1]))(avg_pool)
        max_pool = layers.Reshape((1, 1, x.shape[-1]))(max_pool)
        
        avg_pool = layers.Conv2D(x.shape[-1]//4, 1, activation='relu')(avg_pool)
        avg_pool = layers.Conv2D(x.shape[-1], 1)(avg_pool)
        
        max_pool = layers.Conv2D(x.shape[-1]//4, 1, activation='relu')(max_pool)
        max_pool = layers.Conv2D(x.shape[-1], 1)(max_pool)
        
        channel_attention = layers.Add()([avg_pool, max_pool])
        channel_attention = layers.Activation('sigmoid')(channel_attention)
        
        # Apply channel attention
        x_channel = layers.Multiply()([x, channel_attention])
        
        # Spatial attention
        avg_spatial = tf.reduce_mean(x_channel, axis=-1, keepdims=True)
        max_spatial = tf.reduce_max(x_channel, axis=-1, keepdims=True)
        spatial_features = layers.Concatenate()([avg_spatial, max_spatial])
        
        spatial_attention = layers.Conv2D(1, 7, padding='same', activation='sigmoid')(spatial_features)
        
        # Apply spatial attention
        x_spatial = layers.Multiply()([x_channel, spatial_attention])
        
        return x_spatial
    
    def _dilated_block(self, x, filters):
        """Dilated convolution block for expanding receptive field"""
        # Parallel dilated convolutions with different rates
        dilate1 = layers.Conv2D(filters//4, 3, padding='same', dilation_rate=1)(x)
        dilate2 = layers.Conv2D(filters//4, 3, padding='same', dilation_rate=2)(x)
        dilate4 = layers.Conv2D(filters//4, 3, padding='same', dilation_rate=4)(x)
        dilate8 = layers.Conv2D(filters//4, 3, padding='same', dilation_rate=8)(x)
        
        # Concatenate all dilated features
        dilated_out = layers.Concatenate()([dilate1, dilate2, dilate4, dilate8])
        dilated_out = layers.BatchNormalization()(dilated_out)
        dilated_out = layers.LeakyReLU(0.2)(dilated_out)
        
        # Add residual connection
        if x.shape[-1] == filters:
            dilated_out = layers.Add()([x, dilated_out])
        
        return dilated_out
    
    def _psnr_ssim_combined_loss(self, y_true, y_pred):
        """Combined loss function optimized for both PSNR and SSIM"""
        # MSE directly affects PSNR
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Add a term that encourages gradient similarity (structure preservation)
        def gradient_loss(y_true, y_pred):
            # Compute image gradients
            y_true_dx = y_true[:, :, 1:, :] - y_true[:, :, :-1, :]
            y_true_dy = y_true[:, 1:, :, :] - y_true[:, :-1, :, :]
            y_pred_dx = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
            y_pred_dy = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]
            
            # Compute gradient loss
            dx_loss = tf.reduce_mean(tf.square(y_true_dx - y_pred_dx))
            dy_loss = tf.reduce_mean(tf.square(y_true_dy - y_pred_dy))
            
            return (dx_loss + dy_loss) / 2.0
        
        # Simplified perceptual loss component to help with SSIM
        def perceptual_loss(y_true, y_pred):
            # Compute local patch correlations (simplified SSIM component)
            y_true_patches = tf.image.extract_patches(
                y_true, 
                sizes=[1, 5, 5, 1], 
                strides=[1, 2, 2, 1], 
                rates=[1, 1, 1, 1], 
                padding='SAME'
            )
            y_pred_patches = tf.image.extract_patches(
                y_pred, 
                sizes=[1, 5, 5, 1], 
                strides=[1, 2, 2, 1], 
                rates=[1, 1, 1, 1], 
                padding='SAME'
            )
            
            # Compute correlation loss
            patch_loss = tf.reduce_mean(tf.square(
                tf.nn.l2_normalize(y_true_patches, axis=-1) - 
                tf.nn.l2_normalize(y_pred_patches, axis=-1)
            ))
            
            return patch_loss
        
        # Weighted combination of losses
        # MSE weight is higher to prioritize PSNR
        gradient_weight = 0.15
        perceptual_weight = 0.1
        
        return 0.75 * mse + gradient_weight * gradient_loss(y_true, y_pred) + perceptual_weight * perceptual_loss(y_true, y_pred)
    
    def denoise_image_patches(self, noisy_image):
        """Denoise an image using patch-based approach for improved quality"""
        # Convert image to range [-1, 1] if needed
        if noisy_image.max() > 1.0:
            noisy_image = noisy_image.astype(np.float32) / 127.5 - 1.0
        
        # Get image dimensions
        h, w = noisy_image.shape[:2]
        
        # Define patch size and overlap
        patch_size = PATCH_SIZE_INFERENCE
        overlap = PATCH_OVERLAP
        
        # Calculate steps for sliding window
        h_steps = max(1, (h - patch_size) // (patch_size - overlap) + 1)
        w_steps = max(1, (w - patch_size) // (patch_size - overlap) + 1)
        
        # Initialize the output image with zeros
        denoised_image = np.zeros_like(noisy_image)
        weight_map = np.zeros((h, w, 1))
        
        # Create a weight mask for blending patches
        # Weight is higher at the center of the patch and lower at the edges
        y = np.linspace(-1, 1, patch_size)
        x = np.linspace(-1, 1, patch_size)
        xx, yy = np.meshgrid(x, y)
        mask = np.clip(1.0 - np.sqrt(xx**2 + yy**2), 0, 1)
        mask = mask[:, :, np.newaxis]
        
        # Process each patch
        print(f"Processing image with patch-based approach ({h_steps}x{w_steps} patches)...")
        total_patches = h_steps * w_steps
        processed = 0
        
        for i in range(h_steps):
            for j in range(w_steps):
                # Calculate patch coordinates
                y_start = min(i * (patch_size - overlap), max(0, h - patch_size))
                x_start = min(j * (patch_size - overlap), max(0, w - patch_size))
                y_end = min(y_start + patch_size, h)
                x_end = min(x_start + patch_size, w)
                
                # Handle edge cases
                current_patch_h = y_end - y_start
                current_patch_w = x_end - x_start
                
                # Extract patch
                patch = noisy_image[y_start:y_end, x_start:x_end]
                
                # Pad if necessary to reach patch_size
                if current_patch_h < patch_size or current_patch_w < patch_size:
                    padded_patch = np.zeros((patch_size, patch_size, noisy_image.shape[2]), dtype=np.float32)
                    padded_patch[:current_patch_h, :current_patch_w] = patch
                    patch = padded_patch
                
                # Denoise patch
                denoised_patch = self.model.predict(np.expand_dims(patch, axis=0), verbose=0)[0]
                
                # Handle potential dimensionality issues
                if len(denoised_patch.shape) < 3:
                    denoised_patch = denoised_patch.reshape(patch.shape)
                
                # Adjust mask for current patch size if needed
                current_mask = mask
                if current_patch_h < patch_size or current_patch_w < patch_size:
                    current_mask = mask[:current_patch_h, :current_patch_w]
                
                # Update the output image with weighted patch
                denoised_image[y_start:y_end, x_start:x_end] += denoised_patch[:current_patch_h, :current_patch_w] * current_mask
                weight_map[y_start:y_end, x_start:x_end] += current_mask
                
                processed += 1
                if processed % 10 == 0 or processed == total_patches:
                    print(f"Processed {processed}/{total_patches} patches")
        
        # Normalize the output image by the weight map
        weight_map = np.repeat(weight_map, noisy_image.shape[2], axis=2)
        denoised_image = np.divide(denoised_image, weight_map, where=weight_map!=0)
        
        # Clip values to valid range
        denoised_image = np.clip(denoised_image, -1.0, 1.0)
        
        return denoised_image
    
    def train(self, train_dataset, val_dataset, epochs=EPOCHS):
        """Train the model with memory-optimized approach"""
        print(f"Training model for {epochs} epochs...")
        
        # Create callbacks
        callbacks_list = [
            # Model checkpoint to save best model
            callbacks.ModelCheckpoint(
                filepath=str(MODELS_DIR / "ultrahd_denoiser_v7_best.keras"),
                monitor='val_loss' if val_dataset else 'loss',
                save_best_only=True,
                verbose=1
            ),
            # Early stopping to prevent overfitting
            callbacks.EarlyStopping(
                monitor='val_loss' if val_dataset else 'loss',
                patience=PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce learning rate when plateauing
            callbacks.ReduceLROnPlateau(
                monitor='val_loss' if val_dataset else 'loss',
                factor=0.5,
                patience=PATIENCE // 3,
                min_lr=MIN_LEARNING_RATE,
                verbose=1
            ),
            # TensorBoard logging
            callbacks.TensorBoard(
                log_dir=str(LOGS_DIR),
                histogram_freq=0,
                write_graph=False  # Disable graph writing to save memory
            ),
            # Memory optimization callback
            self._MemoryOptimization()
        ]
        
        # Train the model
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Save final model
        final_model_path = str(MODELS_DIR / "ultrahd_denoiser_v7_final.keras")
        try:
            self.model.save(final_model_path)
            print(f"Final model saved to {final_model_path}")
        except Exception as e:
            print(f"Error saving model: {e}. Trying alternative format...")
            # Try saving in h5 format if keras format fails
            h5_path = str(MODELS_DIR / "ultrahd_denoiser_v7_final.h5")
            try:
                self.model.save(h5_path)
                print(f"Final model saved to {h5_path}")
            except Exception as e2:
                print(f"Error saving in .h5 format: {e2}")
                # Try one more time with SavedModel format
                saved_model_path = str(MODELS_DIR / "ultrahd_denoiser_v7_saved_model")
                try:
                    self.model.save(saved_model_path)
                    print(f"Model saved in SavedModel format to {saved_model_path}")
                except Exception as e3:
                    print(f"All save attempts failed: {e3}")
        
        # Save training history
        history_dict = history.history
        np.save(str(LOGS_DIR / "training_history.npy"), history_dict)
        
        # Plot training history
        self._plot_training_history(history)
        
        return history
    
    def evaluate(self, test_dataset, noise_levels=[0.1, 0.2, 0.3, 0.5]):
        """Evaluate model performance with focus on PSNR and SSIM metrics"""
        print("Evaluating model performance...")
        
        if not os.path.exists(SAMPLES_DIR):
            os.makedirs(SAMPLES_DIR)
        
        # Get samples from test dataset
        samples = []
        
        # Try-except block to handle potential errors when iterating through the dataset
        try:
            # Take a few sample images from the test dataset
            for x_batch, y_batch in test_dataset.take(3):  # Reduced from 5 to 3 for memory efficiency
                for i in range(min(2, x_batch.shape[0])):
                    samples.append((x_batch[i].numpy(), y_batch[i].numpy()))
                if len(samples) >= 5:  # Limit total samples
                    break
        except Exception as e:
            print(f"Error sampling test dataset: {e}")
            # If we failed to get samples from the dataset, create synthetic test samples
            if len(samples) == 0:
                print("Creating synthetic test samples...")
                data_loader = MemoryEfficientDataLoader(img_size=self.img_size)
                synthetic_images = data_loader.generate_synthetic_data(num_images=5)
                
                for i in range(min(5, len(synthetic_images))):
                    clean = synthetic_images[i]
                    # Add noise manually
                    noise = np.random.normal(0, 0.3, clean.shape)
                    noisy = np.clip(clean + noise, -1.0, 1.0)
                    samples.append((noisy, clean))
        
        # Evaluate on samples
        results = []
        
        for idx, (noisy, clean) in enumerate(samples):
            try:
                # Denoise using the patch-based approach if the image is large enough
                if max(noisy.shape[:2]) > PATCH_SIZE_INFERENCE + PATCH_OVERLAP:
                    print(f"Using patch-based approach for sample {idx}")
                    denoised = self.denoise_image_patches(noisy)
                else:
                    # For smaller images, use direct prediction
                    denoised = self.model.predict(tf.expand_dims(noisy, 0), verbose=0)[0].numpy()
                
                # Convert from [-1, 1] to [0, 1] for metrics calculation
                noisy_0_1 = (noisy + 1) / 2
                clean_0_1 = (clean + 1) / 2
                denoised_0_1 = (denoised + 1) / 2
                
                # Calculate PSNR
                psnr_noisy = peak_signal_noise_ratio(clean_0_1, noisy_0_1)
                psnr_denoised = peak_signal_noise_ratio(clean_0_1, denoised_0_1)
                
                # Calculate SSIM with data_range parameter for float images
                ssim_noisy = structural_similarity(
                    clean_0_1, noisy_0_1, 
                    channel_axis=2,
                    data_range=1.0
                )
                ssim_denoised = structural_similarity(
                    clean_0_1, denoised_0_1, 
                    channel_axis=2,
                    data_range=1.0
                )
                
                # Save results
                results.append({
                    "psnr_noisy": psnr_noisy,
                    "psnr_denoised": psnr_denoised,
                    "psnr_improvement": psnr_denoised - psnr_noisy,
                    "ssim_noisy": ssim_noisy,
                    "ssim_denoised": ssim_denoised,
                    "ssim_improvement": ssim_denoised - ssim_noisy
                })
                
                # Save sample images
                plt.figure(figsize=(15, 5))
                
                # Plot original clean image
                plt.subplot(1, 3, 1)
                plt.imshow(clean_0_1)
                plt.title("Clean")
                plt.axis('off')
                
                # Plot noisy image
                plt.subplot(1, 3, 2)
                plt.imshow(noisy_0_1)
                plt.title(f"Noisy (PSNR: {psnr_noisy:.2f}dB, SSIM: {ssim_noisy:.4f})")
                plt.axis('off')
                
                # Plot denoised image
                plt.subplot(1, 3, 3)
                plt.imshow(denoised_0_1)
                plt.title(f"Denoised (PSNR: {psnr_denoised:.2f}dB, SSIM: {ssim_denoised:.4f})")
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(str(SAMPLES_DIR / f"sample_{idx}.png"))
                plt.close()
            except Exception as e:
                print(f"Error evaluating sample {idx}: {e}")
                continue
        
        # Print average results
        if results:
            avg_psnr_improvement = np.mean([r["psnr_improvement"] for r in results])
            avg_ssim_improvement = np.mean([r["ssim_improvement"] for r in results])
            avg_psnr_denoised = np.mean([r["psnr_denoised"] for r in results])
            avg_ssim_denoised = np.mean([r["ssim_denoised"] for r in results])
            
            print("\nPerformance Metrics:")
            print(f"Average PSNR: {avg_psnr_denoised:.2f}dB (+{avg_psnr_improvement:.2f}dB)")
            print(f"Average SSIM: {avg_ssim_denoised:.4f} (+{avg_ssim_improvement:.4f})")
            print(f"{'='*50}")
            
            # Check if we achieved target metrics
            if avg_psnr_denoised > 32:
                print("🎉 SUCCESS: Target PSNR > 32dB achieved!")
            elif avg_psnr_denoised > 30:
                print("✓ GOOD: PSNR > 30dB achieved, but below 32dB target.")
            else:
                print("⚠️ Target PSNR > 32dB not yet achieved.")
                
            if avg_ssim_denoised > 0.92:
                print("🎉 SUCCESS: Excellent SSIM > 0.92 achieved!")
            elif avg_ssim_denoised > 0.85:
                print("✓ GOOD: Good SSIM > 0.85 achieved.")
            else:
                print("⚠️ SSIM could be improved further.")
        
        return results
    
    def _plot_training_history(self, history):
        """Plot and save training metrics"""
        try:
            history_dict = history.history
            
            plt.figure(figsize=(12, 5))
            
            # Plot training & validation loss
            plt.subplot(1, 2, 1)
            plt.plot(history_dict['loss'], label='Training Loss')
            if 'val_loss' in history_dict:
                plt.plot(history_dict['val_loss'], label='Validation Loss')
            plt.title('Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # Plot MAE
            plt.subplot(1, 2, 2)
            plt.plot(history_dict['mae'], label='Training MAE')
            if 'val_mae' in history_dict:
                plt.plot(history_dict['val_mae'], label='Validation MAE')
            plt.title('Mean Absolute Error')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(str(LOGS_DIR / "training_history.png"))
            plt.close()
            
            print(f"Training history plot saved to {LOGS_DIR / 'training_history.png'}")
        except Exception as e:
            print(f"Error plotting training history: {e}")
    
    class _MemoryOptimization(callbacks.Callback):
        """Custom callback for memory optimization"""
        def on_epoch_end(self, epoch, logs=None):
            # Force garbage collection
            gc.collect()
            
            # Clear tensorflow cached memory
            if tf.__version__.startswith('2'):
                tf.keras.backend.clear_session() 

def main():
    """Main function to run the fine-tuning pipeline"""
    print("="*80)
    print("Starting UltraHD Denoiser v7 fine-tuning pipeline with BSDS dataset and patch-based approach")
    print("="*80)
    start_time = time.time()
    
    # Memory optimization before starting
    gc.collect()
    
    # Set TensorFlow to use float32 by default
    try:
        tf.keras.backend.set_floatx('float32')
        print("Set TensorFlow default float type to float32")
    except Exception as e:
        print(f"Warning: Could not set TensorFlow default float type: {e}")
    
    # Create data loader
    data_loader = MemoryEfficientDataLoader(
        img_size=IMG_SIZE,
        patch_stride=PATCH_STRIDE,
        noise_min=NOISE_LEVEL_MIN,
        noise_max=NOISE_LEVEL_MAX
    )
    
    try:
        # Load and prepare images with patch-based approach
        patches = data_loader.load_images(limit=MAX_TOTAL_PATCHES)
        
        # Debug information
        print(f"Loaded patches info: shape={patches.shape}, dtype={patches.dtype}")
        
        # Verify we have data to work with
        if len(patches) == 0:
            print("Error: No data available for training after data loading process.")
            return
        
        # Make sure the data is float32
        if patches.dtype != np.float32:
            print(f"Converting patches from {patches.dtype} to float32")
            patches = patches.astype(np.float32)
            
        # Check data values
        print(f"Data range: min={patches.min()}, max={patches.max()}, mean={patches.mean()}")
        
        # Split into training and validation
        split_idx = int(len(patches) * TRAIN_VALIDATION_SPLIT)
        
        # Memory optimization: use indices instead of copying data
        indices = np.random.permutation(len(patches))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:] if split_idx < len(patches) else indices[:0]  # Empty if no validation data
        
        train_patches = patches[train_indices]
        val_patches = patches[val_indices] if len(val_indices) > 0 else None
        
        print(f"Training patches: {len(train_patches)}")
        print(f"Validation patches: {len(val_patches) if val_patches is not None else 0}")
        
        # Free memory by deleting the full dataset
        del patches
        gc.collect()
        
        # Ensure we have at least one batch for training
        if len(train_patches) < BATCH_SIZE:
            print(f"Warning: Not enough training samples ({len(train_patches)}) for a single batch (size={BATCH_SIZE}).")
            print("Generating additional synthetic data to complete a batch...")
            
            # Generate additional synthetic samples
            additional_needed = BATCH_SIZE - len(train_patches)
            synthetic_samples = data_loader.generate_synthetic_data(num_images=additional_needed)
            
            # Combine existing with synthetic
            train_patches = np.concatenate([train_patches, synthetic_samples[:additional_needed]])
            print(f"New training set size: {len(train_patches)}")
        
        # Create TensorFlow datasets with advanced noise patterns
        print("Creating training dataset with advanced noise patterns...")
        train_dataset = data_loader.create_tf_dataset(train_patches, batch_size=BATCH_SIZE, is_training=True)
        
        if val_patches is not None and len(val_patches) >= BATCH_SIZE:
            print("Creating validation dataset...")
            val_dataset = data_loader.create_tf_dataset(val_patches, batch_size=BATCH_SIZE, is_training=False)
        else:
            print("Skipping validation dataset creation (insufficient data)")
            val_dataset = None
        
        # Create and build model with enhanced architecture
        model = HybridResidualAttentionModel(img_size=IMG_SIZE)
        model.build_model(fine_tune_base=True)
        
        # Train model with higher parameter settings
        print(f"Starting model training with {EPOCHS} epochs and advanced parameters...")
        model.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            epochs=EPOCHS
        )
        
        # Evaluate model with patch-based approach
        print("Evaluating model with patch-based approach...")
        test_dataset = val_dataset if val_dataset is not None else train_dataset
        results = model.evaluate(test_dataset)
        
        # Save a final copy of the model with version number
        final_model_path = MODELS_DIR / f"ultrahd_denoiser_v7_final_{time.strftime('%Y%m%d')}.keras"
        try:
            model.model.save(str(final_model_path))
            print(f"Final timestamped model saved to {final_model_path}")
        except Exception as e:
            print(f"Error saving timestamped model: {e}")
        
        # Calculate total time
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Total fine-tuning completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
        # Final memory cleanup
        del train_patches
        if val_patches is not None:
            del val_patches
        gc.collect()
        
        # Return confirmation
        print("="*80)
        print(f"UltraHD Denoiser v7 model saved to {MODELS_DIR / 'ultrahd_denoiser_v7_final.keras'}")
        print("Fine-tuning with BSDS dataset and patch-based approach completed successfully!")
        print("="*80)
    
    except Exception as e:
        print(f"Error in training pipeline: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to recover by creating a simplified model if possible
        print("Attempting to create a simplified model as fallback...")
        
        try:
            # Create simplified model architecture
            def create_simple_model(img_size=IMG_SIZE):
                inputs = layers.Input(shape=(img_size, img_size, 3))
                
                # Simple encoder-decoder with skip connections
                # Encoder
                x = layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
                x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
                skip1 = x
                x = layers.MaxPooling2D(2)(x)
                
                x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
                x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
                skip2 = x
                x = layers.MaxPooling2D(2)(x)
                
                # Bottleneck
                x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
                x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
                
                # Decoder
                x = layers.UpSampling2D(2)(x)
                x = layers.Concatenate()([x, skip2])
                x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
                x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
                
                x = layers.UpSampling2D(2)(x)
                x = layers.Concatenate()([x, skip1])
                x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
                x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
                
                # Output
                outputs = layers.Conv2D(3, 3, padding='same')(x)
                
                model = Model(inputs, outputs, name="simple_denoiser_v7")
                model.compile(
                    optimizer=optimizers.Adam(learning_rate=1e-4),
                    loss='mse',
                    metrics=['mae']
                )
                return model
            
            # Generate synthetic data 
            print("Generating synthetic data for simplified model training...")
            synthetic_data = data_loader.generate_synthetic_data(num_images=200)
            
            # Split into train/val
            split = int(len(synthetic_data) * 0.8)
            train_data = synthetic_data[:split]
            val_data = synthetic_data[split:]
            
            # Create simplified datasets
            print("Creating simplified datasets with basic noise...")
            
            # Create a simple pipeline without the complex noise function
            def simple_add_noise(image):
                noise = tf.random.normal(
                    shape=tf.shape(image), 
                    mean=0.0, 
                    stddev=0.2, 
                    dtype=tf.float32
                )
                noisy_image = tf.clip_by_value(image + noise, -1.0, 1.0)
                return noisy_image, image
            
            # Create simplified datasets
            train_ds = tf.data.Dataset.from_tensor_slices(train_data)
            train_ds = train_ds.map(simple_add_noise)
            train_ds = train_ds.batch(BATCH_SIZE)
            
            val_ds = tf.data.Dataset.from_tensor_slices(val_data)
            val_ds = val_ds.map(simple_add_noise)
            val_ds = val_ds.batch(BATCH_SIZE)
            
            # Create and train simple model
            print("Creating and training simplified model...")
            simple_model = create_simple_model()
            
            # Simplified callbacks
            simple_callbacks = [
                callbacks.ModelCheckpoint(
                    filepath=str(MODELS_DIR / "simple_denoiser_v7_best.keras"),
                    save_best_only=True,
                    verbose=1
                ),
                callbacks.EarlyStopping(
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                )
            ]
            
            # Train for fewer epochs
            simple_model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=30,
                callbacks=simple_callbacks,
                verbose=1
            )
            
            # Save model
            simple_model.save(str(MODELS_DIR / "simple_denoiser_v7_final.keras"))
            print(f"Simplified model saved to {MODELS_DIR / 'simple_denoiser_v7_final.keras'}")
            
        except Exception as recovery_error:
            print(f"Recovery failed: {recovery_error}")
            print("Please check the error messages above and fix the issues before trying again.")

if __name__ == "__main__":
    main() 