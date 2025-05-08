import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model, optimizers, callbacks, applications, regularizers
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from pathlib import Path
import urllib.request
import zipfile
import io
import random
import time
from PIL import Image
from tqdm import tqdm
import cv2
import gc
import tarfile
import shutil

# Set number of CPU threads for data processing - Optimal for Kaggle environments
CPU_THREADS = min(6, os.cpu_count() or 4)  # Limited to avoid excessive CPU usage

# Configure TensorFlow threading BEFORE any other TensorFlow operations
try:
    tf.config.threading.set_inter_op_parallelism_threads(CPU_THREADS // 2)
    tf.config.threading.set_intra_op_parallelism_threads(CPU_THREADS)
    print(f"TensorFlow configured to use {CPU_THREADS} CPU threads")
except RuntimeError as e:
    print(f"Warning: Could not set thread parallelism: {e}")
    print("Continuing with default thread settings")

# Configure memory growth - Critical for Kaggle's GPU environment
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # First set memory growth for all GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        print(f"Found {len(gpus)} GPU(s) with memory growth enabled")
        
        # Then configure memory limits as a separate step
        for gpu in gpus:
            # Set memory limit for Kaggle environment to avoid OOM
            # Adjust based on available GPU memory (usually ~16GB on Kaggle's P100)
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=14000)]  # ~14GB limit
            )
            
    except RuntimeError as e:
        print(f"GPU setup error: {e}")

# Check TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Configure multi-GPU strategy with performance optimizations
try:
    strategy = tf.distribute.MirroredStrategy()
    print(f"Using MirroredStrategy with {strategy.num_replicas_in_sync} devices")
    
    # Configure collective operations
    if strategy.num_replicas_in_sync > 1:
        os.environ['TF_COLLECTIVE_OPERATIONS_PROTOCOL'] = 'RING'  # Use ring-reduce for better performance
        print("Configured ring-reduce for multi-GPU communication")
except Exception as e:
    print(f"Error setting up MirroredStrategy: {e}")
    strategy = tf.distribute.get_strategy()
    print("Falling back to default strategy")

# Set random seed for reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# Enhanced configuration parameters for high noise and high resolution - MEMORY OPTIMIZED
BATCH_SIZE = max(1, min(2, strategy.num_replicas_in_sync))  # Reduced batch size to save memory
BASE_IMG_SIZE = 128  # Base image size for training
HIGH_RES_IMG_SIZE = 384  # Reduced from 512 to 384 to lower memory usage
NOISE_LEVEL_MIN = 0.05   # Lower minimum noise level
NOISE_LEVEL_MAX = 0.6   # Higher maximum noise level
EPOCHS = 100  # Reduced epochs to save time/memory
INITIAL_LEARNING_RATE = 1e-4  # Higher initial learning rate
MIN_LEARNING_RATE = 1e-7  # Minimum learning rate
PATIENCE = 15  # Reduced patience to save time
MIN_DELTA = 1e-5
BASE_DIR = Path("crystal_clear_denoiser_v4")  # Updated directory

# Advanced memory optimization for Kaggle's memory constraints
MAX_PATCHES_PER_IMAGE = 3  # Reduced to save memory
MAX_TOTAL_PATCHES = 5000  # Reduced total patches significantly
PATCH_SIZE = HIGH_RES_IMG_SIZE  # Match patch size to high-res size
MEMORY_EFFICIENT_BATCH_SIZE = 10  # Reduced for memory efficiency
GRADIENT_ACCUMULATION_STEPS = 4  # Accumulate gradients to simulate larger batch size

# Set a model path that will be accessible in most environments
try:
    import google.colab
    IN_COLAB = True
    MODEL_PATH = "/content/denoising_autoencoder_base.keras"
except ImportError:
    IN_COLAB = False
    if os.path.exists("/kaggle/input/fine-tuned-model-3/tensorflow2/default/1/advanced_high_noise_denoiser_best.keras"):
        # For Kaggle environment, look for potential pre-trained models
        potential_paths = [
            "/kaggle/input/image-encorder-base-model/tensorflow2/default/1/denoising_autoencoder_best.keras",
            "/kaggle/input/fine-tuned-model-3/advanced_high_noise_denoiser_best.keras",
            "/kaggle/input/fine-tuned-model-3/advanced_high_noise_denoiser_final.keras"
        ]
        MODEL_PATH = next((path for path in potential_paths if os.path.exists(path)), 
                         "/kaggle/input/image-encorder-base-model/tensorflow2/default/1/denoising_autoencoder_best.keras")
    else:
        # Local path
        MODEL_PATH = str(Path(BASE_DIR) / "models" / "base_model.keras")

# Additional datasets for fine-tuning
DIV2K_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
BSDS_URL = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300-images.tgz"
KODAK_URL = "http://r0k.us/graphics/kodak/kodak/kodim{:02d}.png"
SET5_URL = "https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip"
SET14_URL = "https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip" 
URBAN100_URL = "https://uofi.box.com/shared/static/65upg43jjd0a4cwsiqgl6o6ixube6klm.zip"

# Create necessary directories
os.makedirs(BASE_DIR, exist_ok=True)
MODELS_DIR = BASE_DIR / "models"
SAMPLES_DIR = BASE_DIR / "samples"
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

for directory in [MODELS_DIR, SAMPLES_DIR, LOGS_DIR, DATA_DIR, CHECKPOINT_DIR]:
    os.makedirs(directory, exist_ok=True)

print("Configuration complete - Enhanced for crystal-clear denoising")

class OptimizedDataLoader:
    """Memory-optimized data loader for high-resolution image denoising with enhanced noise handling"""
    
    def __init__(self, base_img_size=BASE_IMG_SIZE, high_res_img_size=HIGH_RES_IMG_SIZE, 
                 noise_level_min=NOISE_LEVEL_MIN, noise_level_max=NOISE_LEVEL_MAX, 
                 patch_size=PATCH_SIZE):
        self.base_img_size = base_img_size
        self.high_res_img_size = high_res_img_size
        self.noise_level_min = noise_level_min
        self.noise_level_max = noise_level_max
        self.patch_size = patch_size  # Should be equal to high_res_img_size
        self.data_dir = DATA_DIR
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize dataset statistics for balanced sampling
        self.dataset_stats = {
            'div2k': {'count': 0, 'weight': 1.2},   # High quality DIV2K gets higher weight
            'bsds': {'count': 0, 'weight': 1.0},
            'kodak': {'count': 0, 'weight': 1.5},   # Standard test images - high weight
            'set5': {'count': 0, 'weight': 0.8},
            'set14': {'count': 0, 'weight': 0.8},
            'urban100': {'count': 0, 'weight': 1.1}, # Urban scenes get higher weight
            'synthetic': {'count': 0, 'weight': 0.7} # Synthetic gets lower weight
        }
    
    def download_div2k_subset(self, num_images=300):
        """Download a subset of DIV2K dataset - high quality training images"""
        div2k_dir = self.data_dir / "div2k"
        os.makedirs(div2k_dir, exist_ok=True)
        
        # Check if we already have enough DIV2K images
        existing_images = list(div2k_dir.glob("**/*.png")) + list(div2k_dir.glob("**/*.jpg"))
        if len(existing_images) >= num_images:
            print(f"Using {len(existing_images)} existing DIV2K images")
            self.dataset_stats['div2k']['count'] = len(existing_images[:num_images])
            return [str(img) for img in existing_images[:num_images]]
        
        # Download and extract DIV2K dataset
        zip_path = self.data_dir / "div2k.zip"
        if not zip_path.exists():
            try:
                print(f"Downloading DIV2K dataset from {DIV2K_URL}...")
                urllib.request.urlretrieve(DIV2K_URL, zip_path)
                print(f"DIV2K dataset downloaded to {zip_path}")
            except Exception as e:
                print(f"Error downloading DIV2K: {e}")
                return []
        
        # Extract the zip file
        try:
            print("Extracting DIV2K dataset...")
            # First verify if it's a valid zip file
            try:
                with zipfile.ZipFile(zip_path, 'r') as check_zip:
                    # Just check if it's a valid zip
                    pass
            except zipfile.BadZipFile:
                print(f"Error: {zip_path} is not a valid zip file. Removing and skipping extraction.")
                # Remove corrupt file
                os.remove(zip_path)
                return []
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get all image files in the zip
                image_files = [f for f in zip_ref.namelist() 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg')) 
                              and not f.startswith('__MACOSX')]
                
                # Limit to requested number of images
                selected_files = image_files[:num_images]
                
                # Extract files with progress updates
                for i, file in enumerate(selected_files):
                    zip_ref.extract(file, div2k_dir)
                    if (i+1) % 10 == 0 or i+1 == len(selected_files):
                        print(f"Extracted {i+1}/{len(selected_files)} DIV2K images")
            
            # Find all extracted images (search recursively)
            div2k_images = []
            for ext in ['.png', '.jpg', '.jpeg']:
                div2k_images.extend(list(div2k_dir.glob(f"**/*{ext}")))
            
            div2k_images = [str(img) for img in div2k_images[:num_images]]
            self.dataset_stats['div2k']['count'] = len(div2k_images)
            print(f"Successfully extracted {len(div2k_images)} DIV2K images")
            return div2k_images
            
        except Exception as e:
            print(f"Error extracting DIV2K dataset: {e}")
            return []
    
    def download_bsds_subset(self, num_images=200):
        """Download a subset of Berkeley Segmentation Dataset"""
        bsds_dir = self.data_dir / "bsds"
        os.makedirs(bsds_dir, exist_ok=True)
        
        # Check if we already have enough BSDS images
        existing_images = list(bsds_dir.glob("**/*.jpg"))
        if len(existing_images) >= num_images:
            print(f"Using {len(existing_images)} existing BSDS images")
            self.dataset_stats['bsds']['count'] = len(existing_images[:num_images])
            return [str(img) for img in existing_images[:num_images]]
        
        # Download the BSDS dataset
        tgz_path = self.data_dir / "bsds.tgz"
        if not tgz_path.exists():
            try:
                print(f"Downloading BSDS dataset from {BSDS_URL}...")
                # Use a more robust download method with timeout and retries
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        urllib.request.urlretrieve(BSDS_URL, tgz_path)
                        print(f"BSDS dataset downloaded to {tgz_path}")
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            print(f"Download attempt {attempt+1} failed: {e}. Retrying...")
                            time.sleep(2)  # Wait before retrying
                        else:
                            print(f"Failed to download BSDS dataset after {max_retries} attempts: {e}")
                            return []
            except Exception as e:
                print(f"Error downloading BSDS dataset: {e}")
                return []
        
        # Extract the tgz file
        try:
            print("Extracting BSDS dataset...")
            with tarfile.open(tgz_path, 'r:gz') as tar_ref:
                # Extract only a subset of images to save space
                count = 0
                for member in tar_ref.getmembers():
                    if member.name.lower().endswith('.jpg'):
                        if count >= num_images:
                            break
                        # Extract the file
                        tar_ref.extract(member, bsds_dir)
                        count += 1
                        if count % 30 == 0:
                            print(f"Extracted {count} images so far...")
        
            # Get the paths of extracted images - search recursively
            bsds_images = list(bsds_dir.glob("**/*.jpg"))
            bsds_images = [str(img) for img in bsds_images[:num_images]]
            self.dataset_stats['bsds']['count'] = len(bsds_images)
            print(f"Extracted {len(bsds_images)} BSDS images")
            return bsds_images
            
        except Exception as e:
            print(f"Error extracting BSDS dataset: {e}")
            return []
    
    def download_kodak_subset(self, num_images=24):
        """Download Kodak dataset (high-quality reference images)"""
        kodak_dir = self.data_dir / "kodak"
        os.makedirs(kodak_dir, exist_ok=True)
        
        # Check if we already have enough Kodak images
        existing_images = list(kodak_dir.glob("**/*.png"))
        if len(existing_images) >= num_images:
            print(f"Using {len(existing_images)} existing Kodak images")
            self.dataset_stats['kodak']['count'] = len(existing_images[:num_images])
            return [str(img) for img in existing_images[:num_images]]
        
        # Download Kodak images
        images = []
        for i in range(1, num_images + 1):
            img_path = kodak_dir / f"kodim{i:02d}.png"
            
            if img_path.exists():
                images.append(str(img_path))
                continue
            
            try:
                img_url = KODAK_URL.format(i)
                print(f"Downloading Kodak image {i}/{num_images}...")
                urllib.request.urlretrieve(img_url, img_path)
                images.append(str(img_path))
            except Exception as e:
                print(f"Error downloading Kodak image {i}: {e}")
        
        self.dataset_stats['kodak']['count'] = len(images)
        print(f"Downloaded {len(images)} Kodak images")
        return images
    
    def download_set5_subset(self, num_images=5):
        """Download Set5 dataset"""
        set5_dir = self.data_dir / "set5"
        os.makedirs(set5_dir, exist_ok=True)
        
        # Check if we already have enough Set5 images
        existing_images = list(set5_dir.glob("**/*.png"))
        if len(existing_images) >= num_images:
            print(f"Using {len(existing_images)} existing Set5 images")
            self.dataset_stats['set5']['count'] = len(existing_images[:num_images])
            return [str(img) for img in existing_images[:num_images]]
        
        # Download Set5 dataset
        zip_path = self.data_dir / "set5.zip"
        if not zip_path.exists():
            try:
                print(f"Downloading Set5 dataset from {SET5_URL}...")
                urllib.request.urlretrieve(SET5_URL, zip_path)
                print(f"Set5 dataset downloaded to {zip_path}")
            except Exception as e:
                print(f"Error downloading Set5 dataset: {e}")
                return []
        
        # Extract the zip file
        try:
            print("Extracting Set5 dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(set5_dir)
            
            # Get the paths of extracted images
            set5_images = list(set5_dir.glob("**/*.png"))
            set5_images = [str(img) for img in set5_images[:num_images]]
            self.dataset_stats['set5']['count'] = len(set5_images)
            print(f"Extracted {len(set5_images)} Set5 images")
            return set5_images
            
        except Exception as e:
            print(f"Error extracting Set5 dataset: {e}")
            return []
    
    def download_set14_subset(self, num_images=14):
        """Download Set14 dataset"""
        set14_dir = self.data_dir / "set14"
        os.makedirs(set14_dir, exist_ok=True)
        
        # Check if we already have enough Set14 images
        existing_images = list(set14_dir.glob("**/*.png"))
        if len(existing_images) >= num_images:
            print(f"Using {len(existing_images)} existing Set14 images")
            self.dataset_stats['set14']['count'] = len(existing_images[:num_images])
            return [str(img) for img in existing_images[:num_images]]
        
        # Download Set14 dataset
        zip_path = self.data_dir / "set14.zip"
        if not zip_path.exists():
            try:
                print(f"Downloading Set14 dataset from {SET14_URL}...")
                urllib.request.urlretrieve(SET14_URL, zip_path)
                print(f"Set14 dataset downloaded to {zip_path}")
            except Exception as e:
                print(f"Error downloading Set14 dataset: {e}")
                return []
        
        # Extract the zip file
        try:
            print("Extracting Set14 dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(set14_dir)
            
            # Get the paths of extracted images
            set14_images = list(set14_dir.glob("**/*.png"))
            set14_images = [str(img) for img in set14_images[:num_images]]
            self.dataset_stats['set14']['count'] = len(set14_images)
            print(f"Extracted {len(set14_images)} Set14 images")
            return set14_images
            
        except Exception as e:
            print(f"Error extracting Set14 dataset: {e}")
            return []
    
    def download_urban100_subset(self, num_images=50):
        """Download Urban100 dataset (high-resolution urban images)"""
        urban100_dir = self.data_dir / "urban100"
        os.makedirs(urban100_dir, exist_ok=True)
        
        # Check if we already have enough Urban100 images
        existing_images = list(urban100_dir.glob("**/*.png"))
        if len(existing_images) >= num_images:
            print(f"Using {len(existing_images)} existing Urban100 images")
            self.dataset_stats['urban100']['count'] = len(existing_images[:num_images])
            return [str(img) for img in existing_images[:num_images]]
        
        # Download Urban100 dataset
        zip_path = self.data_dir / "urban100.zip"
        if not zip_path.exists():
            try:
                print(f"Downloading Urban100 dataset from {URBAN100_URL}...")
                urllib.request.urlretrieve(URBAN100_URL, zip_path)
                print(f"Urban100 dataset downloaded to {zip_path}")
            except Exception as e:
                print(f"Error downloading Urban100 dataset: {e}")
                return []
        
        # Extract the zip file
        try:
            print("Extracting Urban100 dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract only a subset of images to save space
                count = 0
                for file in zip_ref.namelist():
                    if file.lower().endswith('.png'):
                        if count >= num_images:
                            break
                        # Extract the file
                        zip_ref.extract(file, urban100_dir)
                        count += 1
                        if count % 10 == 0:
                            print(f"Extracted {count} images so far...")
            
            # Get the paths of extracted images - search recursively
            urban100_images = list(urban100_dir.glob("**/*.png"))
            urban100_images = [str(img) for img in urban100_images[:num_images]]
            self.dataset_stats['urban100']['count'] = len(urban100_images)
            print(f"Extracted {len(urban100_images)} Urban100 images")
            return urban100_images
            
        except Exception as e:
            print(f"Error extracting Urban100 dataset: {e}")
            return []
    
    def create_synthetic_images(self, num_images=800):
        """Create synthetic images for training with focus on denoising patterns"""
        synthetic_dir = self.data_dir / "synthetic"
        os.makedirs(synthetic_dir, exist_ok=True)
        
        # Check if we already have enough synthetic images
        existing_images = list(synthetic_dir.glob("*.png"))
        if len(existing_images) >= num_images:
            print(f"Using {len(existing_images)} existing synthetic images")
            self.dataset_stats['synthetic']['count'] = len(existing_images[:num_images])
            return [str(img) for img in existing_images[:num_images]]
        
        # Process in batches to manage memory
        batch_size = 50  # Smaller batch size for higher resolution images
        images = []
        
        for batch in range(0, num_images, batch_size):
            batch_end = min(batch + batch_size, num_images)
            print(f"Creating synthetic images {batch+1}-{batch_end}/{num_images}...")
            
            for i in range(batch, batch_end):
                img_path = synthetic_dir / f"synthetic_{i:04d}.png"
                
                if img_path.exists():
                    images.append(str(img_path))
                    continue
                
                # Create a synthetic image with focus on patterns that help denoising
                pattern_types = [
                    'gradient', 'checkerboard', 'noise', 'circles', 'lines', 
                    'texture', 'gradient_noise', 'mixed', 'edge_test', 'detail_test',
                    'dots', 'waves', 'frequency_noise', 'fractal', 'binary_pattern'
                ]
                pattern = pattern_types[i % len(pattern_types)]
                
                # High-res synthetic images for better denoising
                img = np.zeros((self.high_res_img_size, self.high_res_img_size, 3), dtype=np.float32)
                
                if pattern == 'gradient':
                    # Create a gradient image - good for testing smooth area denoising
                    for y in range(img.shape[0]):
                        for x in range(img.shape[1]):
                            r = x / img.shape[1]
                            g = y / img.shape[0]
                            b = (x + y) / (img.shape[0] + img.shape[1])
                            img[y, x] = [r, g, b]
                
                elif pattern == 'checkerboard':
                    # Create a checkerboard pattern - tests edge preservation
                    # Vary tile size based on image index for diversity
                    size_options = [8, 16, 32, 64]
                    tile_size = size_options[i % len(size_options)]
                    for y in range(img.shape[0]):
                        for x in range(img.shape[1]):
                            if ((x // tile_size) + (y // tile_size)) % 2 == 0:
                                img[y, x] = [0.9, 0.9, 0.9]
                            else:
                                img[y, x] = [0.1, 0.1, 0.1]
                
                elif pattern == 'noise':
                    # Create a noise pattern with structure - helps with noise pattern recognition
                    base_size = self.high_res_img_size // 16
                    base = np.random.rand(base_size, base_size, 3)
                    # Use cv2 for efficient resizing
                    if cv2:
                        base = cv2.resize(base, (self.high_res_img_size, self.high_res_img_size), 
                                         interpolation=cv2.INTER_LINEAR)
                    else:
                        # Fallback to simple repeat if cv2 not available
                        base = np.kron(base, np.ones((16, 16, 1)))
                    img = base
                
                elif pattern == 'circles':
                    # Create concentric circles - tests edge preservation and curvature
                    center_y, center_x = img.shape[0] // 2, img.shape[1] // 2
                    for y in range(img.shape[0]):
                        for x in range(img.shape[1]):
                            dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                            img[y, x] = [
                                0.5 + 0.5 * np.sin(dist / 40),
                                0.5 + 0.5 * np.cos(dist / 30),
                                0.5 + 0.5 * np.sin(dist / 20)
                            ]
                
                elif pattern == 'lines':
                    # Create line patterns - critical for edge preservation tests
                    line_thickness = max(1, self.high_res_img_size // 128)  # Scale to resolution
                    for y in range(img.shape[0]):
                        for x in range(img.shape[1]):
                            if (x % (40 * line_thickness)) < (20 * line_thickness):
                                img[y, x, 0] = 0.8
                            if (y % (30 * line_thickness)) < (15 * line_thickness):
                                img[y, x, 1] = 0.7
                            if ((x + y) % (50 * line_thickness)) < (25 * line_thickness):
                                img[y, x, 2] = 0.9
                
                elif pattern == 'texture':
                    # Create texture patterns with varied frequencies
                    freq_scale = max(10, self.high_res_img_size // 25)  # Scale frequency with resolution
                    for y in range(img.shape[0]):
                        for x in range(img.shape[1]):
                            img[y, x] = [
                                0.5 + 0.5 * np.sin(x/freq_scale) * np.cos(y/freq_scale),
                                0.5 + 0.5 * np.sin(x/(freq_scale*1.5)) * np.cos(y/(freq_scale*1.5)),
                                0.5 + 0.5 * np.sin(x/(freq_scale*2)) * np.cos(y/(freq_scale*2))
                            ]
                
                elif pattern == 'gradient_noise':
                    # Create gradient with noise - tests denoising across gradients
                    base = np.zeros((img.shape[0], img.shape[1], 3))
                    for y in range(img.shape[0]):
                        for x in range(img.shape[1]):
                            base[y, x] = [x/img.shape[1], y/img.shape[0], (x+y)/(img.shape[0]+img.shape[1])]
                    
                    noise = np.random.normal(0, 0.1, img.shape)
                    img = base + noise
                
                elif pattern == 'mixed':
                    # Create mixed patterns - good for general testing
                    if i % 3 == 0:
                        # Mix gradient and noise
                        base = np.zeros((img.shape[0], img.shape[1], 3))
                        for y in range(img.shape[0]):
                            for x in range(img.shape[1]):
                                base[y, x] = [x/img.shape[1], y/img.shape[0], (x+y)/(img.shape[0]+img.shape[1])]
                        
                        noise = np.random.normal(0, 0.15, img.shape)
                        img = base + noise
                    elif i % 3 == 1:
                        # Mix circles and lines
                        center_y, center_x = img.shape[0] // 2, img.shape[1] // 2
                        for y in range(img.shape[0]):
                            for x in range(img.shape[1]):
                                dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                                img[y, x, 0] = 0.5 + 0.5 * np.sin(dist / 40)
                                
                                if x % 40 < 20:
                                    img[y, x, 1] = 0.8
                                if y % 30 < 15:
                                    img[y, x, 2] = 0.7
                    else:
                        # Mix textures with checkerboard
                        freq = max(15, self.high_res_img_size // 20)
                        tile_size = self.high_res_img_size // 16
                        for y in range(img.shape[0]):
                            for x in range(img.shape[1]):
                                texture_val = 0.5 + 0.3 * np.sin(x/freq) * np.cos(y/freq)
                                checker_val = 0.8 if ((x // tile_size) + (y // tile_size)) % 2 == 0 else 0.2
                                
                                img[y, x] = [
                                    texture_val, 
                                    checker_val,
                                    (texture_val + checker_val) / 2
                                ]
                
                elif pattern == 'edge_test':
                    # Pattern designed specifically to test edge preservation
                    img.fill(0.5)  # Fill with mid-gray
                    
                    # Add various edges with different contrasts
                    # Horizontal edges
                    edge_widths = [1, 2, 3]
                    contrast_levels = [0.2, 0.5, 0.8]
                    
                    height_div = img.shape[0] // (len(edge_widths) * len(contrast_levels) + 1)
                    
                    idx = 0
                    for width in edge_widths:
                        for contrast in contrast_levels:
                            y_pos = height_div * (idx + 1)
                            idx += 1
                            
                            # Draw horizontal edge
                            img[y_pos-width:y_pos+width, :, 0] = 0.5 + contrast/2
                            img[y_pos+width:y_pos+width*3, :, 0] = 0.5 - contrast/2
                            
                            # Draw vertical edge at different x positions
                            x_pos = img.shape[1] // (idx + 1)
                            img[:, x_pos-width:x_pos+width, 1] = 0.5 + contrast/2
                            img[:, x_pos+width:x_pos+width*3, 1] = 0.5 - contrast/2
                            
                            # Draw diagonal edge
                            for y in range(img.shape[0]):
                                x = y + img.shape[1]//4
                                if 0 <= x < img.shape[1]:
                                    x_range = slice(max(0, x-width), min(img.shape[1], x+width))
                                    img[y, x_range, 2] = 0.5 + contrast/2
                
                elif pattern == 'detail_test':
                    # Pattern to test fine detail preservation
                    img.fill(0.3)  # Dark gray background
                    
                    # Add fine details of varying sizes
                    detail_sizes = [1, 2, 4, 8, 16]
                    spacing = self.high_res_img_size // 10
                    
                    for i, size in enumerate(detail_sizes):
                        y_offset = spacing * (i + 2)
                        
                        # Horizontal row of details
                        for j in range(10):
                            x_offset = spacing * (j + 1)
                            if size == 1:
                                # Single pixel
                                img[y_offset, x_offset, :] = 0.9
                            else:
                                # Small square
                                y_slice = slice(y_offset - size//2, y_offset + size//2 + size%2)
                                x_slice = slice(x_offset - size//2, x_offset + size//2 + size%2)
                                img[y_slice, x_slice, :] = 0.9
                
                elif pattern == 'dots':
                    # Create dot pattern - stress test for detail preservation
                    background_color = np.random.uniform(0.1, 0.4, 3)
                    img.fill(background_color[0])
                    
                    dot_sizes = [1, 2, 3, 4, 5]
                    for y in range(0, img.shape[0], 20):
                        for x in range(0, img.shape[1], 20):
                            # Random dot size
                            dot_size = dot_sizes[np.random.randint(0, len(dot_sizes))]
                            if y+dot_size < img.shape[0] and x+dot_size < img.shape[1]:
                                # Random bright color
                                color = np.random.uniform(0.6, 0.9, 3)
                                y_slice = slice(y, y+dot_size)
                                x_slice = slice(x, x+dot_size)
                                img[y_slice, x_slice, :] = color
                
                elif pattern == 'waves':
                    # Create wave patterns - smooth curves good for filter alignment tests
                    for y in range(img.shape[0]):
                        for x in range(img.shape[1]):
                            # Varying frequency waves
                            freq1 = 0.01 + (i % 5) * 0.01  # Different frequencies based on i
                            freq2 = 0.02 + (i % 3) * 0.01
                            img[y, x] = [
                                0.5 + 0.4 * np.sin(y * freq1),
                                0.5 + 0.4 * np.sin(x * freq1),
                                0.5 + 0.4 * np.sin((x+y) * freq2)
                            ]
                
                elif pattern == 'frequency_noise':
                    # Create frequency-varying noise for testing detail preservation at different scales
                    if cv2:
                        # Perlin-noise approximation using multiple octaves
                        img = np.zeros((self.high_res_img_size, self.high_res_img_size, 3))
                        octaves = 4
                        persistence = 0.5
                        
                        for channel in range(3):
                            amplitude = 1.0
                            for octave in range(octaves):
                                frequency = 2 ** octave
                                size = self.high_res_img_size // frequency
                                
                                # Create random noise at this frequency
                                noise = np.random.rand(size, size)
                                # Resize to full size
                                noise_resized = cv2.resize(noise, (self.high_res_img_size, self.high_res_img_size), 
                                                          interpolation=cv2.INTER_LINEAR)
                                
                                # Add to image with decreasing amplitude
                                img[:, :, channel] += noise_resized * amplitude
                                amplitude *= persistence
                            
                            # Normalize to [0,1]
                            img_min = img[:,:,channel].min()
                            img_max = img[:,:,channel].max()
                            if img_max > img_min:
                                img[:,:,channel] = (img[:,:,channel] - img_min) / (img_max - img_min)
                    else:
                        # Fallback if cv2 not available
                        img = np.random.rand(self.high_res_img_size, self.high_res_img_size, 3)
                
                elif pattern == 'fractal':
                    # Simplified fractal pattern
                    def mandelbrot(h, w, max_iter):
                        y, x = np.ogrid[-1.4:1.4:h*1j, -2:0.8:w*1j]
                        c = x + y*1j
                        z = c
                        divtime = max_iter + np.zeros(z.shape, dtype=int)
                        
                        for i in range(max_iter):
                            z = z**2 + c
                            diverge = z*np.conj(z) > 2**2
                            div_now = diverge & (divtime == max_iter)
                            divtime[div_now] = i
                            z[diverge] = 2
                            
                        return divtime
                    
                    # Adjust max_iter to prevent too much computation
                    size_factor = min(1.0, 1024 / self.high_res_img_size)  # Scale with image size
                    max_iter = max(10, int(30 * size_factor))
                    
                    # Create the fractal pattern
                    fractal = mandelbrot(img.shape[0], img.shape[1], max_iter)
                    fractal_norm = fractal / max_iter
                    for c in range(3):
                        # Vary colors between channels
                        if c == 0:
                            img[:,:,c] = fractal_norm
                        elif c == 1:
                            img[:,:,c] = 1 - fractal_norm
                        else:
                            img[:,:,c] = np.abs(2 * fractal_norm - 1)
                
                elif pattern == 'binary_pattern':
                    # Binary mask patterns - extreme edge case
                    img.fill(0.1)  # Dark background
                    
                    # Create random binary masks
                    mask_count = 5
                    for _ in range(mask_count):
                        mask = np.zeros((img.shape[0], img.shape[1]))
                        
                        # Random mask type
                        mask_type = np.random.randint(0, 3)
                        
                        if mask_type == 0:
                            # Rectangular mask
                            w = np.random.randint(img.shape[1]//8, img.shape[1]//2)
                            h = np.random.randint(img.shape[0]//8, img.shape[0]//2)
                            x = np.random.randint(0, img.shape[1] - w)
                            y = np.random.randint(0, img.shape[0] - h)
                            mask[y:y+h, x:x+w] = 1
                        
                        elif mask_type == 1:
                            # Circular mask
                            radius = np.random.randint(img.shape[0]//16, img.shape[0]//4)
                            center_x = np.random.randint(radius, img.shape[1] - radius)
                            center_y = np.random.randint(radius, img.shape[0] - radius)
                            
                            y, x = np.ogrid[:img.shape[0], :img.shape[1]]
                            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                            mask[dist <= radius] = 1
                        
                        else:
                            # Line pattern
                            thickness = np.random.randint(1, img.shape[0]//32)
                            angle = np.random.uniform(0, np.pi)
                            
                            # Start and end points of the line
                            length = np.sqrt(img.shape[0]**2 + img.shape[1]**2)
                            center_x, center_y = img.shape[1]//2, img.shape[0]//2
                            
                            for i in range(int(length)):
                                x = int(center_x + i * np.cos(angle))
                                y = int(center_y + i * np.sin(angle))
                                
                                # Draw line with thickness
                                for dy in range(-thickness, thickness+1):
                                    for dx in range(-thickness, thickness+1):
                                        if (0 <= y + dy < img.shape[0] and 
                                            0 <= x + dx < img.shape[1]):
                                            mask[y + dy, x + dx] = 1
                        
                        # Apply mask to a random channel with high value
                        channel = np.random.randint(0, 3)
                        img[:,:,channel] = np.maximum(img[:,:,channel], mask * 0.9)
                
                # Save the image with proper clipping
                plt.imsave(img_path, np.clip(img, 0, 1))
                images.append(str(img_path))
            
            # Free memory after each batch
            gc.collect()
        
        self.dataset_stats['synthetic']['count'] = len(images)
        print(f"Created {len(images)} synthetic images")
        return images 

    def load_and_prepare_dataset(self):
        """Load and prepare the dataset with extreme memory optimization for Kaggle"""
        print("Loading and preparing dataset with extreme memory optimization...")
        
        # Keep original dataset sizes but use memory-efficient processing
        div2k_images = self.download_div2k_subset(300)
        bsds_images = self.download_bsds_subset(200)
        kodak_images = self.download_kodak_subset(24)   # Full Kodak dataset (small)
        synthetic_images = self.create_synthetic_images(800)
        
        # Try to download smaller test datasets if available
        set5_images = self.download_set5_subset(5) if hasattr(self, 'download_set5_subset') else []
        set14_images = self.download_set14_subset(14) if hasattr(self, 'download_set14_subset') else []
        urban100_images = self.download_urban100_subset(50) if hasattr(self, 'download_urban100_subset') else []
        
        # Calculate dataset statistics
        print("\nDataset statistics:")
        total_images = 0
        for dataset, stats in self.dataset_stats.items():
            if stats['count'] > 0:
                print(f"  {dataset}: {stats['count']} images (weight: {stats['weight']})")
                total_images += stats['count']
        print(f"  Total: {total_images} images")
        
        # Combine all image sources with balanced but limited sampling
        all_images = []
        
        # Calculate sampling probabilities based on weights and counts
        total_weight = sum(stats['weight'] * stats['count'] 
                          for dataset, stats in self.dataset_stats.items() 
                          if stats['count'] > 0)
        
        # Sample with replacement to get a balanced dataset
        dataset_images = {
            'div2k': div2k_images,
            'bsds': bsds_images,
            'kodak': kodak_images,
            'set5': set5_images,
            'set14': set14_images,
            'urban100': urban100_images,
            'synthetic': synthetic_images
        }
        
        # Calculate sample size for each dataset based on weights - with memory limit
        dataset_samples = {}
        desired_total = min(1000, total_images)  # Limit total images for memory efficiency
        
        for dataset, stats in self.dataset_stats.items():
            if stats['count'] > 0:
                # Calculate proportional sample size
                sample_size = int((stats['weight'] * stats['count'] / total_weight) * desired_total)
                # Ensure we don't sample more than available
                sample_size = min(sample_size, stats['count'], int(desired_total/4))  # Cap each source
                dataset_samples[dataset] = sample_size
        
        print("\nMemory-optimized sampling:")
        for dataset, sample_size in dataset_samples.items():
            if dataset_images[dataset] and sample_size > 0:
                print(f"  {dataset}: {sample_size} images")
                
                # Add sampled images to the combined dataset
                # Sample from the dataset (with replacement if needed)
                if sample_size <= len(dataset_images[dataset]):
                    sampled = random.sample(dataset_images[dataset], sample_size)
                else:
                    # If we need more than available, sample with replacement
                    sampled = random.choices(dataset_images[dataset], k=sample_size)
                
                all_images.extend(sampled)
        
        if not all_images:
            raise ValueError("No images available for fine-tuning")
        
        print(f"Total images after optimized sampling: {len(all_images)}")
        
        # Clean up dataset variables to free memory
        for dataset_name in dataset_images:
            dataset_images[dataset_name] = None
        gc.collect()
        
        # Shuffle the combined dataset
        random.shuffle(all_images)
        
        # Memory-efficient patch extraction with extreme chunking
        clean_patches = []
        patch_count = 0
        
        # Further reduced batch size for image processing
        mem_opt_batch_size = max(5, MEMORY_EFFICIENT_BATCH_SIZE // 2)
        
        # Process images in very small batches to manage memory
        for i in range(0, len(all_images), mem_opt_batch_size):
            if patch_count >= MAX_TOTAL_PATCHES:
                break
                
            batch_images = all_images[i:i+mem_opt_batch_size]
            batch_patches = []
            
            for img_path in tqdm(batch_images, desc=f"Extracting patches from batch {i//mem_opt_batch_size + 1}/{len(all_images)//mem_opt_batch_size + 1}"):
                if patch_count >= MAX_TOTAL_PATCHES:
                    break
                    
                try:
                    # Load image
                    img = Image.open(img_path).convert('RGB')
                    width, height = img.size
                    
                    # Skip if image is too small
                    if width < self.patch_size or height < self.patch_size:
                        continue
                        
                    # Extreme memory-efficient patch extraction
                    # Extract fewer patches with minimal processing
                    max_patches = min(MAX_PATCHES_PER_IMAGE, 2)  # Limit to 2 patches per image maximum
                    
                    # For memory efficiency, just extract corner and center patches
                    patches_extracted = 0
                    
                    # Only extract from corners/center based on image index for variety
                    img_idx = patch_count % 5
                    
                    if img_idx == 0 and width >= self.patch_size and height >= self.patch_size:
                        # Top-left corner
                        patch = img.crop((0, 0, self.patch_size, self.patch_size))
                        patch_array = np.array(patch).astype(np.float32) / 255.0
                        batch_patches.append(patch_array)
                        patch_count += 1
                        patches_extracted += 1
                    
                    elif img_idx == 1 and width >= self.patch_size and height >= self.patch_size:
                        # Bottom-right corner
                        right = max(0, width - self.patch_size)
                        bottom = max(0, height - self.patch_size)
                        patch = img.crop((right, bottom, right + self.patch_size, bottom + self.patch_size))
                        patch_array = np.array(patch).astype(np.float32) / 255.0
                        batch_patches.append(patch_array)
                        patch_count += 1
                        patches_extracted += 1
                    
                    elif img_idx == 2 and width >= self.patch_size and height >= self.patch_size:
                        # Center patch
                        center_x = max(0, (width - self.patch_size) // 2)
                        center_y = max(0, (height - self.patch_size) // 2)
                        patch = img.crop((center_x, center_y, 
                                        center_x + self.patch_size, 
                                        center_y + self.patch_size))
                        patch_array = np.array(patch).astype(np.float32) / 255.0
                        batch_patches.append(patch_array)
                        patch_count += 1
                        patches_extracted += 1
                    
                    elif patches_extracted < max_patches:
                        # Random patch if none extracted yet
                        if width > self.patch_size and height > self.patch_size:
                            x = random.randint(0, width - self.patch_size)
                            y = random.randint(0, height - self.patch_size)
                            patch = img.crop((x, y, x + self.patch_size, y + self.patch_size))
                            patch_array = np.array(patch).astype(np.float32) / 255.0
                            batch_patches.append(patch_array)
                            patch_count += 1
                    
                    # Free memory immediately
                    del img
                    gc.collect()
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
            
            # Add batch patches to main list and clear batch
            clean_patches.extend(batch_patches)
            del batch_patches
            gc.collect()
            
            # Force garbage collection every few batches
            if i % (mem_opt_batch_size * 5) == 0:
                print(f"Memory cleanup at {i}/{len(all_images)} images")
                gc.collect()
        
        print(f"Extracted {len(clean_patches)} patches for fine-tuning")
        
        # Memory-efficient array conversion with aggressive cleanup
        # Convert to numpy array in smaller chunks
        chunk_size = 100  # Even smaller chunks
        final_patches = []
        
        for i in range(0, len(clean_patches), chunk_size):
            # Convert chunk to numpy array
            chunk = np.array(clean_patches[i:i+chunk_size])
            
            # Apply normalization to [-1, 1] range
            chunk = chunk * 2.0 - 1.0
            
            # Add to final patches list
            final_patches.append(chunk)
            
            # Free memory from clean_patches as we go
            for j in range(i, min(i+chunk_size, len(clean_patches))):
                clean_patches[j] = None
            
            # Force garbage collection
            gc.collect()
        
        # Clear original list before combining chunks
        clean_patches = None
        gc.collect()
        
        # Combine chunks with memory management
        if len(final_patches) == 1:
            clean_patches = final_patches[0]
        else:
            # Concatenate progressively to minimize peak memory
            clean_patches = final_patches[0]
            for i in range(1, len(final_patches)):
                clean_patches = np.concatenate([clean_patches, final_patches[i]], axis=0)
                final_patches[i] = None
                gc.collect()
        
        # Clean up
        del final_patches
        del all_images
        gc.collect()
        
        print(f"Final dataset shape: {clean_patches.shape}")
        return clean_patches
    
    def create_tf_dataset(self, clean_images, batch_size):
        """Create TensorFlow dataset with advanced augmentation and noise simulation"""
        # Convert clean_images to float32 if not already
        clean_images = tf.cast(clean_images, tf.float32)
        
        # Create dataset with memory optimization
        dataset = tf.data.Dataset.from_tensor_slices(clean_images)
        
        # Define advanced noise generation with crystal-clear denoising focus
        def add_realistic_noise(clean_img, training=True):
            """Add realistic noise patterns optimized for crystal-clear denoising"""
            # Base noise level - vary based on training vs validation
            if training:
                # More varied noise for training
                noise_level = tf.random.uniform(
                    shape=[], 
                    minval=self.noise_level_min,
                    maxval=self.noise_level_max
                )
            else:
                # More focused noise levels for validation
                noise_options = [0.1, 0.2, 0.3, 0.4, 0.5]
                idx = tf.random.uniform(shape=[], minval=0, maxval=5, dtype=tf.int32)
                noise_level = tf.gather(noise_options, idx)
            
            # Create noisy image starting with clean
            noisy_img = clean_img
            
            # 1. Apply basic Gaussian noise to all images (always present)
            gaussian_stddev = noise_level
            gaussian_noise = tf.random.normal(
                shape=tf.shape(clean_img),
                mean=0.0,
                stddev=gaussian_stddev
            )
            noisy_img = noisy_img + gaussian_noise
            
            # 2. Randomly apply impulse noise (salt & pepper) with carefully calibrated intensity
            # Salt & pepper is critical for real-world denoising performance
            if tf.random.uniform([]) < 0.4:  # 40% chance
                # Salt noise - bright spots
                salt_amount = noise_level * tf.random.uniform([], 0.05, 0.15)
                salt_mask = tf.cast(tf.random.uniform(tf.shape(clean_img)) < salt_amount, tf.float32)
                noisy_img = noisy_img * (1.0 - salt_mask) + salt_mask * 1.0  # Bright value
                
                # Pepper noise - dark spots
                pepper_amount = noise_level * tf.random.uniform([], 0.05, 0.15)
                pepper_mask = tf.cast(tf.random.uniform(tf.shape(clean_img)) < pepper_amount, tf.float32)
                noisy_img = noisy_img * (1.0 - pepper_mask) + pepper_mask * (-1.0)  # Dark value
            
            # 3. Random speckle noise (multiplicative) - particularly challenging for denoisers
            if tf.random.uniform([]) < 0.3:  # 30% chance
                speckle_intensity = noise_level * tf.random.uniform([], 0.8, 1.5)
                speckle = tf.random.normal(
                    shape=tf.shape(clean_img),
                    mean=0.0,
                    stddev=speckle_intensity
                )
                noisy_img = noisy_img + noisy_img * speckle
            
            # 4. Simulate sensor noise with random color channel variation
            if tf.random.uniform([]) < 0.3:  # 30% chance
                # Different noise for each color channel (simulates real camera sensor)
                channel_noise = [
                    tf.random.normal(
                        shape=[tf.shape(clean_img)[0], tf.shape(clean_img)[1], 1], 
                        mean=0.0, 
                        stddev=noise_level * tf.random.uniform([], 0.7, 1.3)
                    ) for _ in range(3)
                ]
                channel_noise = tf.concat(channel_noise, axis=2)
                noisy_img = noisy_img + channel_noise
            
            # 5. Occasionally add structured noise (lines, patterns)
            if tf.random.uniform([]) < 0.2:  # 20% chance
                # Create structured pattern
                pattern_type = tf.random.uniform([], 0, 3, dtype=tf.int32)
                
                # Horizontal lines
                if pattern_type == 0:
                    line_mask = tf.zeros_like(clean_img)
                    height = tf.shape(clean_img)[0]
                    line_spacing = tf.random.uniform([], 8, 32, dtype=tf.int32)
                    line_intensity = noise_level * 2.0
                    
                    for y in tf.range(0, height, line_spacing):
                        line_width = tf.random.uniform([], 1, 3, dtype=tf.int32)
                        start = tf.minimum(y, height - line_width)
                        line_mask = tf.tensor_scatter_nd_update(
                            line_mask,
                            tf.reshape(tf.range(start, start + line_width), [-1, 1]),
                            tf.ones([line_width, tf.shape(clean_img)[1], 3]) * line_intensity
                        )
                    
                    # Apply with random sign to create both bright and dark lines
                    if tf.random.uniform([]) < 0.5:
                        noisy_img = noisy_img + line_mask
                    else:
                        noisy_img = noisy_img - line_mask
                
                # Other structured patterns handled as Gaussian for TF graph compatibility
                else:
                    # Just add more focused Gaussian noise as fallback
                    extra_noise = tf.random.normal(
                        shape=tf.shape(clean_img),
                        mean=0.0,
                        stddev=noise_level * 1.5
                    )
                    noisy_img = noisy_img + extra_noise
            
            # Clip values to ensure they stay in valid range
            noisy_img = tf.clip_by_value(noisy_img, -1.0, 1.0)
            clean_img = tf.clip_by_value(clean_img, -1.0, 1.0)
            
            return noisy_img, clean_img
        
        # Map the noise addition function with controlled parallelism
        dataset = dataset.map(
            add_realistic_noise,
            num_parallel_calls=CPU_THREADS  # Use configured CPU threads instead of AUTOTUNE
        )
        
        # Optimize pipeline with Kaggle-specific settings
        cache_dataset = True  # Change to False if dataset is too large for memory
        if cache_dataset:
            dataset = dataset.cache()  # Cache for speed, but uses memory
        
        # Use smaller shuffle buffer to save memory
        shuffle_buffer = min(1000, len(clean_images))
        dataset = dataset.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=True)
        
        # Use drop_remainder for TPU compatibility and to ensure consistent batch sizes
        dataset = dataset.batch(batch_size, drop_remainder=True)
        
        # Use prefetch with AUTOTUNE to maximize throughput
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        # Validate dataset shape to ensure it matches the model input
        try:
            for x, y in dataset.take(1):
                print(f"Dataset validation - Input shape: {x.shape}, Output shape: {y.shape}")
        except Exception as e:
            print(f"Dataset validation error: {e}")
        
        return dataset 

def train_crystal_clear_denoiser():
    """Main training function with extreme memory optimization for Kaggle"""
    print("Starting memory-optimized fine-tuning for Kaggle environment...")
    start_time = time.time()
    
    # Print TensorFlow and GPU information
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Number of GPUs: {len(tf.config.list_physical_devices('GPU'))}")
    print(f"MirroredStrategy replicas: {strategy.num_replicas_in_sync}")
    
    # Initialize data loader
    data_loader = OptimizedDataLoader(
        base_img_size=BASE_IMG_SIZE,
        high_res_img_size=HIGH_RES_IMG_SIZE,
        noise_level_min=NOISE_LEVEL_MIN,
        noise_level_max=NOISE_LEVEL_MAX,
        patch_size=PATCH_SIZE
    )
    
    # Load and prepare dataset
    clean_patches = data_loader.load_and_prepare_dataset()
    
    # Split dataset into training and validation sets (90/10)
    split_idx = int(0.9 * len(clean_patches))
    train_patches = clean_patches[:split_idx]
    val_patches = clean_patches[split_idx:]
    
    print(f"Training patches: {len(train_patches)}")
    print(f"Validation patches: {len(val_patches)}")
    
    # Create TensorFlow datasets
    train_dataset = data_loader.create_tf_dataset(train_patches, BATCH_SIZE)
    val_dataset = data_loader.create_tf_dataset(val_patches, BATCH_SIZE)
    
    # Free up memory
    del clean_patches
    del data_loader  # Free data loader memory
    gc.collect()
    
    # Create and initialize model
    model = CrystalClearDenoiser(
        base_img_size=BASE_IMG_SIZE,
        high_res_img_size=HIGH_RES_IMG_SIZE
    )
    
    # Load pre-trained model if available
    model.load_pretrained_model(MODEL_PATH)
    
    # Enhance the model architecture
    model.enhance_model()
    
    # Print model summary
    print("Model summary (condensed version to save memory):")
    print(f"Input shape: {model.model.input_shape}")
    print(f"Output shape: {model.model.output_shape}")
    print(f"Total layers: {len(model.model.layers)}")
    print(f"Trainable parameters: {model.model.count_params()}")
    
    # Set up callbacks for training - memory optimized
    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(CHECKPOINT_DIR / "best_model.keras"),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,  # Changed to False to avoid .weights.h5 requirement
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            min_delta=MIN_DELTA,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=PATIENCE // 2,
            min_delta=MIN_DELTA,
            min_lr=MIN_LEARNING_RATE,
            verbose=1
        ),
        # Memory cleanup callback
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: gc.collect(),
            on_batch_end=lambda batch, logs: gc.collect() if batch % 20 == 0 else None
        )
    ]
    
    # Train the model
    print("Starting model training with memory optimization...")
    history = model.train(train_dataset, val_dataset, epochs=EPOCHS, callbacks=callbacks_list)
    
    # Save final model
    final_model_path = str(MODELS_DIR / "crystal_clear_denoiser_final.keras")
    model.model.save(final_model_path)  # Save full model instead of just weights
    print(f"Final model saved to {final_model_path}")
    
    # Generate and save sample results with limited samples
    print("Generating minimal sample results...")
    sample_count = min(2, len(val_patches))  # Reduced sample count
    sample_images = val_patches[:sample_count]
    model.generate_samples(sample_images, noise_levels=[0.1, 0.5])  # Only 2 noise levels
    
    # Clean up before plotting
    del val_patches
    gc.collect()
    
    # Plot and save training history - simple version
    plt.figure(figsize=(10, 4))
    plt.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(MODELS_DIR / "training_history.png"), dpi=100)  # Lower DPI
    plt.close()
    
    # Print training time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Model saved to {final_model_path}")
    
    # Force final garbage collection
    gc.collect()
    
    return model

class CrystalClearDenoiser:
    """Advanced denoising model with crystal-clear output quality"""
    
    def __init__(self, base_img_size=BASE_IMG_SIZE, high_res_img_size=HIGH_RES_IMG_SIZE):
        self.base_img_size = base_img_size
        self.high_res_img_size = high_res_img_size
        self.model = None
    
    def load_pretrained_model(self, model_path):
        """Load pre-trained model with fallbacks for better reliability"""
        print(f"Loading pre-trained model from {model_path}...")
        try:
            if not os.path.exists(model_path):
                print(f"Warning: Model path '{model_path}' does not exist.")
                print("Checking for alternative model paths...")
                
                # Look for alternative models
                potential_paths = [
                    str(MODELS_DIR / "base_model.keras"),
                    str(MODELS_DIR / "denoising_autoencoder_best.keras"),
                    "/kaggle/input/image-encorder-base-model/tensorflow2/default/1/denoising_autoencoder_best.keras",
                    "/kaggle/input/fine-tuned-model-3/advanced_high_noise_denoiser_best.keras"
                ]
                
                for path in potential_paths:
                    if os.path.exists(path):
                        print(f"Found alternative model at: {path}")
                        model_path = path
                        break
                
                if not os.path.exists(model_path):
                    print("No existing model found. Creating a basic model instead...")
                    with strategy.scope():
                        inputs = layers.Input(shape=(self.base_img_size, self.base_img_size, 3))
                        x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
                        x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
                        x = layers.Conv2D(3, 3, padding='same')(x)
                        self.model = Model(inputs, x)
                    return True
            
            with strategy.scope():
                self.model = load_model(model_path)
                print("Pre-trained model loaded successfully")
                print(f"Pre-trained model input shape: {self.model.input_shape}")
                return True
                
        except Exception as e:
            print(f"Error loading pre-trained model: {e}")
            print("Creating a basic model instead...")
            with strategy.scope():
                inputs = layers.Input(shape=(self.base_img_size, self.base_img_size, 3))
                x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
                x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
                x = layers.Conv2D(3, 3, padding='same')(x)
                self.model = Model(inputs, x)
            return True
    
    def enhance_model(self):
        """Enhance model architecture with advanced features - memory-optimized for Kaggle"""
        print("Enhancing model architecture - memory-optimized version...")
        print(f"Using input shape: ({self.high_res_img_size}, {self.high_res_img_size}, 3)")
        
        # Create model within strategy's scope for multi-GPU training
        with strategy.scope():
            # Create a simpler model with reduced memory footprint
            inputs = layers.Input(shape=(self.high_res_img_size, self.high_res_img_size, 3), name="input_layer")
            
            # Simplified attention mechanism - reduced filter count
            spatial_features = layers.Conv2D(8, 5, padding='same')(inputs)  # Reduced from 16,7
            spatial_features = layers.BatchNormalization()(spatial_features)
            spatial_features = layers.Activation('relu')(spatial_features)
            spatial_features = layers.Conv2D(1, 5, padding='same')(spatial_features)  # Reduced from 7
            spatial_attention = layers.Activation('sigmoid')(spatial_features)
            
            # Apply attention - simpler attention mechanism
            enhanced_features = layers.Multiply()([inputs, spatial_attention])
            
            # Very lightweight preprocessing to save memory
            preprocessed = layers.Conv2D(12, 3, padding='same')(enhanced_features)  # Reduced from 16
            preprocessed = layers.BatchNormalization()(preprocessed)
            preprocessed = layers.Activation('relu')(preprocessed)
            preprocessed = layers.Conv2D(3, 3, padding='same')(preprocessed)
            
            # Add skip connection to preserve details
            preprocessed = layers.Add()([preprocessed, inputs])
            
            # Resize to match base model input size
            resized_input = layers.Resizing(
                height=self.base_img_size,
                width=self.base_img_size,
                interpolation='bilinear'
            )(preprocessed)
            
            # Get output from original model (transfer learning)
            base_output = self.model(resized_input)
            
            # Upscale the output back to high resolution
            upscaled = layers.Resizing(
                height=self.high_res_img_size,
                width=self.high_res_img_size,
                interpolation='bilinear'
            )(base_output)
            
            # Lightweight post-processing
            refined = layers.Conv2D(12, 3, padding='same')(upscaled)  # Reduced from 32
            refined = layers.BatchNormalization()(refined)
            refined = layers.Activation('relu')(refined)
            final_output = layers.Conv2D(3, 3, padding='same')(refined)
            
            # Add residual connection for better detail preservation
            outputs = layers.Add()([final_output, inputs])
            
            # Create enhanced model
            enhanced_model = Model(inputs, outputs, name="memory_efficient_denoiser")
            
            # Compile with optimizer - clipvalue removed to avoid extra memory ops
            optimizer = optimizers.Adam(
                learning_rate=INITIAL_LEARNING_RATE,
                clipnorm=1.0  # Keep only clipnorm for stability
            )
            
            enhanced_model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae']  # Removed 'mse' to save memory
            )
        
        # Initialize model with a forward pass
        print("Initializing model with a forward pass...")
        sample_input = tf.random.uniform((1, self.high_res_img_size, self.high_res_img_size, 3))
        _ = enhanced_model(sample_input, training=False)
        del sample_input
        gc.collect()
        
        self.model = enhanced_model
        print("Model enhancement complete - memory-optimized version")
        
        return self.model
    
    def train(self, train_dataset, val_dataset=None, epochs=EPOCHS, callbacks=None):
        """Train the model with advanced techniques for optimal convergence"""
        if self.model is None:
            raise ValueError("Model not loaded or built. Call load_pretrained_model and enhance_model first.")
        
        print(f"Training model for {epochs} epochs...")
        
        # Default callbacks if none provided
        if callbacks is None:
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(CHECKPOINT_DIR / "model_epoch{epoch:03d}.keras"),
                    save_best_only=True,
                    monitor='val_loss' if val_dataset else 'loss',
                    verbose=1
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss' if val_dataset else 'loss',
                    patience=PATIENCE,
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss' if val_dataset else 'loss',
                    factor=0.5,
                    patience=PATIENCE // 2,
                    min_lr=MIN_LEARNING_RATE,
                    verbose=1
                ),
                tf.keras.callbacks.TensorBoard(
                    log_dir=str(LOGS_DIR),
                    update_freq='epoch',
                    histogram_freq=1
                )
            ]
        
        # Custom learning rate warmup
        class WarmupScheduler(tf.keras.callbacks.Callback):
            def __init__(self, warmup_epochs=5, initial_lr=INITIAL_LEARNING_RATE):
                super(WarmupScheduler, self).__init__()
                self.warmup_epochs = warmup_epochs
                self.initial_lr = initial_lr
                self.current_lr = initial_lr
                
            def on_epoch_begin(self, epoch, logs=None):
                if epoch < self.warmup_epochs:
                    # Linear warmup
                    warmup_lr = self.initial_lr * ((epoch + 1) / self.warmup_epochs)
                    # Use compatible approach to set learning rate
                    if hasattr(self.model.optimizer, 'learning_rate'):
                        tf.keras.backend.set_value(self.model.optimizer.learning_rate, warmup_lr)
                    elif hasattr(self.model.optimizer, '_decayed_lr'):
                        tf.keras.backend.set_value(self.model.optimizer._decayed_lr(tf.float32), warmup_lr)
                    elif hasattr(self.model.optimizer, 'lr'):
                        tf.keras.backend.set_value(self.model.optimizer.lr, warmup_lr)
                    else:
                        print("Warning: Could not set learning rate during warmup - optimizer structure not recognized")
                    print(f"\nEpoch {epoch+1}: WarmupScheduler setting learning rate to {warmup_lr:.6f}")
        
        # Add warmup scheduler to callbacks if training from scratch
        # Fixed: Access the optimizer learning rate properly with compatibility for different TF versions
        try:
            # Try to get the current learning rate 
            if hasattr(self.model.optimizer, 'learning_rate'):
                current_lr = self.model.optimizer.learning_rate.numpy()
            elif hasattr(self.model.optimizer, '_decayed_lr'):
                current_lr = self.model.optimizer._decayed_lr(tf.float32).numpy()
            elif hasattr(self.model.optimizer, 'lr'):
                current_lr = self.model.optimizer.lr.numpy()
            else:
                # Default to adding warmup if we can't determine the current learning rate
                current_lr = INITIAL_LEARNING_RATE + 1  
                print("Warning: Could not determine current learning rate, defaulting to using warmup")
            
            if current_lr >= INITIAL_LEARNING_RATE:
                callbacks.append(WarmupScheduler())
                print(f"Adding warmup scheduler (current LR: {current_lr}, initial LR: {INITIAL_LEARNING_RATE})")
        except Exception as e:
            print(f"Error checking learning rate: {e}")
            # Add warmup scheduler by default if there's an error
            callbacks.append(WarmupScheduler())
            print("Adding warmup scheduler due to error checking learning rate")
        
        # Train the model
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Convert history to dictionary for consistency
        history_dict = {
            'loss': history.history['loss'],
            'mae': history.history['mae'],
        }
        
        if val_dataset:
            history_dict['val_loss'] = history.history['val_loss']
            history_dict['val_mae'] = history.history['val_mae']
        
        return history_dict
    
    def generate_samples(self, test_images, noise_levels=[0.1, 0.3, 0.5]):
        """Generate sample results with different noise levels"""
        os.makedirs(SAMPLES_DIR, exist_ok=True)
        
        for i, clean_img in enumerate(test_images[:5]):
            # Create figure for multiple noise levels
            fig, axes = plt.subplots(len(noise_levels) + 1, 3, figsize=(15, 5 * (len(noise_levels) + 1)))
            
            # Original image at the top
            clean_display = (clean_img + 1) / 2.0
            axes[0, 0].imshow(np.clip(clean_display, 0, 1))
            axes[0, 0].set_title("Original")
            axes[0, 0].axis("off")
            
            axes[0, 1].set_visible(False)
            axes[0, 2].set_visible(False)
            
            # For each noise level
            for j, noise_level in enumerate(noise_levels):
                row = j + 1
                
                # Add noise
                noisy_img = clean_img.copy()
                
                # Add Gaussian noise
                noisy_img = noisy_img + np.random.normal(0, noise_level, noisy_img.shape)
                
                # Add salt and pepper noise for higher noise levels
                if noise_level >= 0.3:
                    # Salt noise
                    salt_mask = np.random.random(noisy_img.shape) < noise_level * 0.1
                    noisy_img[salt_mask] = 1.0
                    # Pepper noise
                    pepper_mask = np.random.random(noisy_img.shape) < noise_level * 0.1
                    noisy_img[pepper_mask] = -1.0
                
                noisy_img = np.clip(noisy_img, -1.0, 1.0)
                
                # Predict - use model directly for inference
                denoised_img = self.model.predict(np.expand_dims(noisy_img, 0))[0]
                
                # Convert to [0, 1] range for visualization
                noisy_display = (noisy_img + 1) / 2.0
                denoised_display = (denoised_img + 1) / 2.0
                
                # Display images
                axes[row, 0].imshow(np.clip(clean_display, 0, 1))
                axes[row, 0].set_title("Original")
                axes[row, 0].axis("off")
                
                axes[row, 1].imshow(np.clip(noisy_display, 0, 1))
                axes[row, 1].set_title(f"Noisy (σ={noise_level:.2f})")
                axes[row, 1].axis("off")
                
                axes[row, 2].imshow(np.clip(denoised_display, 0, 1))
                axes[row, 2].set_title("Denoised")
                axes[row, 2].axis("off")
                
                # Calculate and display metrics
                mse = np.mean((clean_display - denoised_display) ** 2)
                psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 100
                axes[row, 2].set_xlabel(f"PSNR: {psnr:.2f} dB")
                
                # Save individual images
                plt.imsave(str(SAMPLES_DIR / f"noisy_{i+1}_level_{noise_level:.2f}.png"), 
                           np.clip(noisy_display, 0, 1))
                plt.imsave(str(SAMPLES_DIR / f"denoised_{i+1}_level_{noise_level:.2f}.png"), 
                           np.clip(denoised_display, 0, 1))
            
            plt.tight_layout()
            plt.savefig(str(SAMPLES_DIR / f"sample_{i+1}_multi_noise.png"), dpi=150)
            plt.close()
            
            # Save original image once
            plt.imsave(str(SAMPLES_DIR / f"original_{i+1}.png"), np.clip(clean_display, 0, 1))

if __name__ == "__main__":
    train_crystal_clear_denoiser() 