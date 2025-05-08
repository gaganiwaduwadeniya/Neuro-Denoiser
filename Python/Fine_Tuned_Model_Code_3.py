import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model, optimizers, callbacks
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

# Set number of CPU threads for data processing
CPU_THREADS = min(8, os.cpu_count() or 4)  # Limit CPU threads to reduce usage

# Configure TensorFlow threading BEFORE any other TensorFlow operations
# This must be done before any other TensorFlow operations to avoid the RuntimeError
try:
    tf.config.threading.set_inter_op_parallelism_threads(CPU_THREADS // 2)
    tf.config.threading.set_intra_op_parallelism_threads(CPU_THREADS)
    print(f"TensorFlow configured to use {CPU_THREADS} CPU threads")
except RuntimeError as e:
    print(f"Warning: Could not set thread parallelism: {e}")
    print("Continuing with default thread settings")

# Check TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Set up GPU memory growth to avoid OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s) with memory growth enabled")
    except RuntimeError as e:
        print(f"GPU setup error: {e}")

# Configure multi-GPU strategy
try:
    strategy = tf.distribute.MirroredStrategy()
    print(f"Using MirroredStrategy with {strategy.num_replicas_in_sync} devices")
except Exception as e:
    print(f"Error setting up MirroredStrategy: {e}")
    strategy = tf.distribute.get_strategy()
    print("Falling back to default strategy")

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Enhanced configuration parameters for high noise and high resolution
BATCH_SIZE = 2 * max(1, strategy.num_replicas_in_sync)  # Reduced batch size for larger images
BASE_IMG_SIZE = 128  # Base image size for training
HIGH_RES_IMG_SIZE = 384  # Reduced from 512 to 384 to lower CPU usage while still being high-res
NOISE_LEVEL_MIN = 0.1   # Increased minimum noise level
NOISE_LEVEL_MAX = 0.5   # Increased maximum noise level
EPOCHS = 200  # Increased epochs for better convergence
LEARNING_RATE = 5e-6  # Lower learning rate for more stable fine-tuning
PATIENCE = 20  # Increased patience for early stopping
MIN_DELTA = 5e-5
BASE_DIR = Path("advanced_fine_tuned_encoder_v2")  # Updated directory

# Enhanced memory optimization parameters
MAX_PATCHES_PER_IMAGE = 8  # Reduced from 10 to decrease CPU load
MAX_TOTAL_PATCHES = 20000  # Reduced from 25000 to decrease CPU load
PATCH_SIZE = 384  # Matches HIGH_RES_IMG_SIZE, reduced from 512
MEMORY_EFFICIENT_BATCH_SIZE = 20  # Reduced from 30 to decrease CPU load

# Set a model path that will be accessible in most environments
# If running in Colab or Kaggle, use their paths; otherwise, use a local path
try:
    import google.colab
    IN_COLAB = True
    MODEL_PATH = "/content/denoising_autoencoder_base.keras"
except ImportError:
    IN_COLAB = False
    if os.path.exists("/kaggle/input"):
        MODEL_PATH = "/kaggle/input/image-encorder-base-model/tensorflow2/default/1/denoising_autoencoder_best.keras" 
    else:
        # Local path - set to use model we'll create if not exists
        MODEL_PATH = str(Path(BASE_DIR) / "models" / "base_model.keras")

# Additional datasets for fine-tuning
DIV2K_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
BSDS_URL = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300-images.tgz"
KODAK_URL = "http://r0k.us/graphics/kodak/kodak/kodim{:02d}.png"
SET5_URL = "https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip"  # Alternative Set5 dataset
SET14_URL = "https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip"  # Alternative Set14 dataset
URBAN100_URL = "https://uofi.box.com/shared/static/65upg43jjd0a4cwsiqgl6o6ixube6klm.zip"  # Alternative Urban100 dataset

# Create necessary directories
os.makedirs(BASE_DIR, exist_ok=True)
MODELS_DIR = BASE_DIR / "models"
SAMPLES_DIR = BASE_DIR / "samples"
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"

for directory in [MODELS_DIR, SAMPLES_DIR, LOGS_DIR, DATA_DIR]:
    os.makedirs(directory, exist_ok=True)

class AdvancedMemoryEfficientDataLoader:
    """Memory-efficient data loader for high-resolution image denoising tasks with enhanced noise handling"""
    
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
    
    def download_div2k_subset(self, num_images=400):  # Reduced from 600 to 400
        """Download a subset of DIV2K dataset"""
        div2k_dir = self.data_dir / "div2k"
        os.makedirs(div2k_dir, exist_ok=True)
        
        # Check if we already have enough DIV2K images
        existing_images = list(div2k_dir.glob("**/*.png")) + list(div2k_dir.glob("**/*.jpg"))
        if len(existing_images) >= num_images:
            print(f"Using {len(existing_images)} existing DIV2K images")
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
            print(f"Successfully extracted {len(div2k_images)} DIV2K images")
            return div2k_images
            
        except Exception as e:
            print(f"Error extracting DIV2K dataset: {e}")
            return []
    
    def download_bsds_subset(self, num_images=300):  # Reduced from 400 to 300
        """Download a subset of Berkeley Segmentation Dataset"""
        bsds_dir = self.data_dir / "bsds"
        os.makedirs(bsds_dir, exist_ok=True)
        
        # Check if we already have enough BSDS images
        existing_images = list(bsds_dir.glob("**/*.jpg"))
        if len(existing_images) >= num_images:
            print(f"Using {len(existing_images)} existing BSDS images")
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
            print(f"Extracted {len(set14_images)} Set14 images")
            return set14_images
            
        except Exception as e:
            print(f"Error extracting Set14 dataset: {e}")
            return []
    
    def download_urban100_subset(self, num_images=50):  # Reduced from 100 to 50
        """Download Urban100 dataset (high-resolution urban images)"""
        urban100_dir = self.data_dir / "urban100"
        os.makedirs(urban100_dir, exist_ok=True)
        
        # Check if we already have enough Urban100 images
        existing_images = list(urban100_dir.glob("**/*.png"))
        if len(existing_images) >= num_images:
            print(f"Using {len(existing_images)} existing Urban100 images")
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
            print(f"Extracted {len(urban100_images)} Urban100 images")
            return urban100_images
            
        except Exception as e:
            print(f"Error extracting Urban100 dataset: {e}")
            return []
    
    def create_synthetic_images(self, num_images=1000):  # Reduced from 2000 to 1000
        """Create synthetic images for training with more complex patterns"""
        synthetic_dir = self.data_dir / "synthetic"
        os.makedirs(synthetic_dir, exist_ok=True)
        
        # Check if we already have enough synthetic images
        existing_images = list(synthetic_dir.glob("*.png"))
        if len(existing_images) >= num_images:
            print(f"Using {len(existing_images)} existing synthetic images")
            return [str(img) for img in existing_images[:num_images]]
        
        # Process in batches to manage memory
        batch_size = 100
        images = []
        
        for batch in range(0, num_images, batch_size):
            batch_end = min(batch + batch_size, num_images)
            print(f"Creating synthetic images {batch+1}-{batch_end}/{num_images}...")
            
            for i in range(batch, batch_end):
                img_path = synthetic_dir / f"synthetic_{i:04d}.png"
                
                if img_path.exists():
                    images.append(str(img_path))
                    continue
                
                # Create a synthetic image
                # Expanded pattern types for more diversity
                pattern_types = [
                    'gradient', 'checkerboard', 'noise', 'circles', 'lines', 
                    'texture', 'gradient_noise', 'mixed', 'fractal', 'stripes',
                    'dots', 'waves', 'perlin_noise', 'voronoi'
                ]
                pattern = pattern_types[i % len(pattern_types)]
                img = np.zeros((self.high_res_img_size, self.high_res_img_size, 3), dtype=np.float32)
                
                if pattern == 'gradient':
                    # Create a gradient image
                    for y in range(img.shape[0]):
                        for x in range(img.shape[1]):
                            r = x / img.shape[1]
                            g = y / img.shape[0]
                            b = (x + y) / (img.shape[0] + img.shape[1])
                            img[y, x] = [r, g, b]
                
                elif pattern == 'checkerboard':
                    # Create a checkerboard pattern
                    tile_size = self.high_res_img_size // 16
                    for y in range(img.shape[0]):
                        for x in range(img.shape[1]):
                            if ((x // tile_size) + (y // tile_size)) % 2 == 0:
                                img[y, x] = [0.9, 0.9, 0.9]
                            else:
                                img[y, x] = [0.1, 0.1, 0.1]
                
                elif pattern == 'noise':
                    # Create a noise pattern with structure
                    base = np.random.rand(self.high_res_img_size // 16, self.high_res_img_size // 16, 3)
                    img = np.kron(base, np.ones((16, 16, 1)))
                
                elif pattern == 'circles':
                    # Create concentric circles
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
                    # Create line patterns
                    for y in range(img.shape[0]):
                        for x in range(img.shape[1]):
                            if x % 40 < 20:
                                img[y, x, 0] = 0.8
                            if y % 30 < 15:
                                img[y, x, 1] = 0.7
                            if (x + y) % 50 < 25:
                                img[y, x, 2] = 0.9
                
                elif pattern == 'texture':
                    # Create texture patterns
                    for y in range(img.shape[0]):
                        for x in range(img.shape[1]):
                            img[y, x] = [
                                0.5 + 0.5 * np.sin(x/20) * np.cos(y/20),
                                0.5 + 0.5 * np.sin(x/30) * np.cos(y/30),
                                0.5 + 0.5 * np.sin(x/40) * np.cos(y/40)
                            ]
                
                elif pattern == 'gradient_noise':
                    # Create gradient with noise
                    base = np.zeros((img.shape[0], img.shape[1], 3))
                    for y in range(img.shape[0]):
                        for x in range(img.shape[1]):
                            base[y, x] = [x/img.shape[1], y/img.shape[0], (x+y)/(img.shape[0]+img.shape[1])]
                    
                    noise = np.random.normal(0, 0.1, img.shape)
                    img = base + noise
                
                elif pattern == 'mixed':
                    # Create mixed patterns
                    if i % 2 == 0:
                        # Mix gradient and noise
                        base = np.zeros((img.shape[0], img.shape[1], 3))
                        for y in range(img.shape[0]):
                            for x in range(img.shape[1]):
                                base[y, x] = [x/img.shape[1], y/img.shape[0], (x+y)/(img.shape[0]+img.shape[1])]
                        
                        noise = np.random.normal(0, 0.15, img.shape)
                        img = base + noise
                    else:
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
                
                # New pattern types for more diversity
                elif pattern == 'fractal':
                    # Simple approximation of fractal-like pattern
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
                    
                    fractal = mandelbrot(img.shape[0], img.shape[1], 20)
                    fractal_norm = fractal / 20.0
                    for c in range(3):
                        img[:,:,c] = fractal_norm * (c+1)/3
                
                elif pattern == 'stripes':
                    # Create diagonal stripes with varying widths
                    for y in range(img.shape[0]):
                        for x in range(img.shape[1]):
                            stripe_value = np.sin((x + y) * 0.1)
                            img[y, x] = [
                                0.5 + 0.4 * stripe_value,
                                0.5 + 0.4 * np.sin((x - y) * 0.05),
                                0.5 + 0.4 * np.cos((x * y) * 0.001)
                            ]
                
                elif pattern == 'dots':
                    # Create dot pattern
                    for y in range(0, img.shape[0], 20):
                        for x in range(0, img.shape[1], 20):
                            if y+10 < img.shape[0] and x+10 < img.shape[1]:
                                color = [
                                    0.2 + 0.7 * ((x+y) % 100) / 100,
                                    0.2 + 0.7 * ((x*y) % 100) / 100,
                                    0.2 + 0.7 * ((x-y) % 100) / 100
                                ]
                                img[y:y+10, x:x+10] = color
                
                elif pattern == 'waves':
                    # Create wave patterns
                    for y in range(img.shape[0]):
                        for x in range(img.shape[1]):
                            img[y, x] = [
                                0.5 + 0.4 * np.sin(y * 0.05),
                                0.5 + 0.4 * np.sin(x * 0.05),
                                0.5 + 0.4 * np.sin((x+y) * 0.05)
                            ]
                
                elif pattern == 'perlin_noise':
                    # Approximation of Perlin noise using multiple frequencies
                    def simple_noise(size, freq):
                        arr = np.zeros((size, size))
                        for y in range(0, size, freq):
                            for x in range(0, size, freq):
                                arr[y:y+freq, x:x+freq] = np.random.random()
                        return arr
                    
                    base_noise1 = simple_noise(img.shape[0], 64)
                    base_noise2 = simple_noise(img.shape[0], 32)
                    base_noise3 = simple_noise(img.shape[0], 16)
                    
                    # Smooth using OpenCV
                    if cv2:
                        base_noise1 = cv2.GaussianBlur(base_noise1, (0, 0), 32)
                        base_noise2 = cv2.GaussianBlur(base_noise2, (0, 0), 16)
                        base_noise3 = cv2.GaussianBlur(base_noise3, (0, 0), 8)
                    
                    # Combine at different weights
                    combined = base_noise1 * 0.5 + base_noise2 * 0.3 + base_noise3 * 0.2
                    img[:,:,0] = combined
                    
                    # Different shifts for other channels
                    combined_g = np.roll(combined, img.shape[0]//4, axis=0)
                    combined_b = np.roll(combined, img.shape[0]//4, axis=1)
                    img[:,:,1] = combined_g
                    img[:,:,2] = combined_b
                
                elif pattern == 'voronoi':
                    # Simplified Voronoi-like pattern
                    num_points = 20
                    points = np.random.randint(0, img.shape[0], size=(num_points, 2))
                    colors = np.random.random((num_points, 3))
                    
                    # For each pixel, find closest point
                    for y in range(img.shape[0]):
                        for x in range(img.shape[1]):
                            distances = np.sqrt(np.sum(((points - [y, x])**2), axis=1))
                            closest_idx = np.argmin(distances)
                            img[y, x] = colors[closest_idx]
                
                # Save the image
                plt.imsave(img_path, np.clip(img, 0, 1))
                images.append(str(img_path))
            
            # Free memory after each batch
            gc.collect()
        
        print(f"Created {len(images)} synthetic images")
        return images 

    def load_and_prepare_dataset(self):
        """Load and prepare the mixed dataset for fine-tuning with memory optimization"""
        print("Loading and preparing dataset for fine-tuning...")
        
        # Get images from different sources with reduced numbers to save CPU
        div2k_images = self.download_div2k_subset(400)  # Reduced from 600 to 400
        bsds_images = self.download_bsds_subset(300)    # Reduced from 400 to 300
        kodak_images = self.download_kodak_subset(24)   # Keep same
        set5_images = self.download_set5_subset(5)      # Keep same
        set14_images = self.download_set14_subset(14)   # Keep same
        urban100_images = self.download_urban100_subset(50)  # Reduced from 100 to 50
        synthetic_images = self.create_synthetic_images(1000)  # Reduced from 2000 to 1000
        
        # Combine all image sources
        all_images = div2k_images + bsds_images + kodak_images + set5_images + set14_images + urban100_images + synthetic_images
        if not all_images:
            raise ValueError("No images available for fine-tuning")
        
        print(f"Total images available: {len(all_images)}")
        random.shuffle(all_images)
        
        # Memory-efficient patch extraction
        clean_patches = []
        patch_count = 0
        
        # Process images in smaller batches with fewer parallel operations
        for i in range(0, len(all_images), MEMORY_EFFICIENT_BATCH_SIZE):
            if patch_count >= MAX_TOTAL_PATCHES:
                break
            
            batch_images = all_images[i:i+MEMORY_EFFICIENT_BATCH_SIZE]
            batch_patches = []
            
            for img_path in tqdm(batch_images, desc=f"Extracting patches from batch {i//MEMORY_EFFICIENT_BATCH_SIZE + 1}/{len(all_images)//MEMORY_EFFICIENT_BATCH_SIZE + 1}"):
                if patch_count >= MAX_TOTAL_PATCHES:
                    break
                
                try:
                    # Load and resize image
                    img = Image.open(img_path).convert('RGB')
                    
                    # Extract patches with simplified logic to reduce CPU usage
                    width, height = img.size
                    max_patches = min(MAX_PATCHES_PER_IMAGE, 
                                    (width // self.patch_size) * (height // self.patch_size))
                    
                    # Prioritize high-detail areas but with simplified computation
                    # Only use edge detection on larger images and with a probability
                    use_edge_detection = width > self.patch_size * 2 and height > self.patch_size * 2 and random.random() < 0.7
                    
                    if use_edge_detection and cv2:
                        # Convert to numpy for edge detection
                        img_np = np.array(img)
                        # Downscale image before edge detection to save CPU
                        scale_factor = max(1, min(width, height) // 1000)
                        if scale_factor > 1:
                            small_img = cv2.resize(img_np, (width // scale_factor, height // scale_factor))
                            gray = cv2.cvtColor(small_img, cv2.COLOR_RGB2GRAY)
                        else:
                            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                        
                        # Use a faster edge detection method
                        edges = cv2.Sobel(gray, cv2.CV_8U, 1, 1, ksize=3)
                        thresh = cv2.threshold(edges, 30, 255, cv2.THRESH_BINARY)[1]
                        
                        # Find contours - more CPU efficient than pixel-wise operations
                        if int(cv2.__version__.split('.')[0]) >= 4:
                            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        else:
                            _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if contours and len(contours) > 0:
                            # Get centers of contours for patch extraction
                            centers = []
                            for contour in contours:
                                M = cv2.moments(contour)
                                if M["m00"] != 0:
                                    cX = int(M["m10"] / M["m00"]) * scale_factor
                                    cY = int(M["m01"] / M["m00"]) * scale_factor
                                    centers.append((cX, cY))
                            
                            # Choose random centers if we have enough
                            if len(centers) > max_patches:
                                selected_centers = random.sample(centers, max_patches)
                                
                                for cX, cY in selected_centers:
                                    # Ensure the patch fits within the image
                                    left = max(0, min(width - self.patch_size, cX - self.patch_size // 2))
                                    top = max(0, min(height - self.patch_size, cY - self.patch_size // 2))
                                    
                                    patch = img.crop((left, top, left + self.patch_size, top + self.patch_size))
                                    patch_array = np.array(patch).astype(np.float32) / 255.0
                                    batch_patches.append(patch_array)
                                    patch_count += 1
                                
                                # Continue to next image
                                continue
                    
                    # Fall back to random patches with stride to reduce computation
                    stride = self.patch_size // 2  # Use stride to reduce computation
                    patches_extracted = 0
                    
                    for y in range(0, height - self.patch_size + 1, stride):
                        if patches_extracted >= max_patches:
                            break
                        for x in range(0, width - self.patch_size + 1, stride):
                            if patches_extracted >= max_patches:
                                break
                            # Only extract patch with some probability to reduce total number
                            if random.random() < 0.3:  # 30% chance to extract a patch at this location
                                patch = img.crop((x, y, x + self.patch_size, y + self.patch_size))
                                patch_array = np.array(patch).astype(np.float32) / 255.0
                                batch_patches.append(patch_array)
                                patch_count += 1
                                patches_extracted += 1
                    
                    # If we didn't extract any patches yet, extract at least one random patch
                    if patches_extracted == 0 and width > self.patch_size and height > self.patch_size:
                        left = random.randint(0, width - self.patch_size)
                        top = random.randint(0, height - self.patch_size)
                        patch = img.crop((left, top, left + self.patch_size, top + self.patch_size))
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
        
        print(f"Extracted {len(clean_patches)} patches for fine-tuning")
        
        # Convert to numpy array in chunks to save memory
        chunk_size = 500  # Reduced from 1000 to lower memory usage
        final_patches = []
        
        for i in range(0, len(clean_patches), chunk_size):
            chunk = np.array(clean_patches[i:i+chunk_size])
            # Normalize to [-1, 1] range
            chunk = chunk * 2.0 - 1.0
            final_patches.append(chunk)
            # Free memory from clean_patches as we go
            for j in range(i, min(i+chunk_size, len(clean_patches))):
                clean_patches[j] = None
            gc.collect()
        
        # Clear clean_patches completely before combining chunks
        clean_patches = None
        gc.collect()
        
        # Combine chunks
        clean_patches = np.concatenate(final_patches, axis=0)
        del final_patches
        gc.collect()
        
        print(f"Clean patches shape: {clean_patches.shape}")
        return clean_patches
    
    def add_noise(self, images):
        """Add more complex noise patterns to images with focus on severe noise patterns"""
        # Create a copy of the images
        noisy_images = images.copy()
        
        # Random noise level for each image with higher levels
        noise_level = np.random.uniform(self.noise_level_min, self.noise_level_max, size=(len(images), 1, 1, 1))
        
        # Add Gaussian noise to all images
        noise = np.random.normal(0, 1, size=images.shape) * noise_level
        noisy_images = images + noise
        
        # Add salt and pepper noise to a larger subset (40%) of images with higher intensity
        salt_pepper_indices = np.random.choice(len(images), size=int(len(images) * 0.4), replace=False)
        for idx in salt_pepper_indices:
            # Higher amount of salt and pepper noise
            amount = 0.15 * noise_level[idx][0][0][0]  # Increased from 0.1 to 0.15
            
            # Salt (white) noise
            mask = np.random.random(images[idx].shape) < amount/2
            noisy_images[idx][mask] = 1.0
            
            # Pepper (black) noise
            mask = np.random.random(images[idx].shape) < amount/2
            noisy_images[idx][mask] = -1.0
        
        # Add speckle noise to another subset (30%)
        speckle_indices = np.random.choice(len(images), size=int(len(images) * 0.3), replace=False)
        for idx in speckle_indices:
            speckle = np.random.normal(0, noise_level[idx][0][0][0] * 1.2, size=images[idx].shape)  # Increased intensity
            noisy_images[idx] += noisy_images[idx] * speckle
        
        # Add structured noise (lines/patterns) to a subset (20%)
        structured_indices = np.random.choice(len(images), size=int(len(images) * 0.2), replace=False)
        for idx in structured_indices:
            # Create horizontal or vertical lines
            if idx % 2 == 0:
                # Horizontal lines
                line_width = np.random.randint(1, 4)
                line_spacing = np.random.randint(10, 30)
                for y in range(0, images[idx].shape[0], line_spacing):
                    if y + line_width < images[idx].shape[0]:
                        noise_value = noise_level[idx][0][0][0] * 2  # More intense noise
                        line_noise = np.random.normal(0, noise_value, size=(line_width, images[idx].shape[1], 3))
                        noisy_images[idx][y:y+line_width, :, :] += line_noise
            else:
                # Vertical lines
                line_width = np.random.randint(1, 4)
                line_spacing = np.random.randint(10, 30)
                for x in range(0, images[idx].shape[1], line_spacing):
                    if x + line_width < images[idx].shape[1]:
                        noise_value = noise_level[idx][0][0][0] * 2  # More intense noise
                        line_noise = np.random.normal(0, noise_value, size=(images[idx].shape[0], line_width, 3))
                        noisy_images[idx][:, x:x+line_width, :] += line_noise
        
        # Add occasional local "blotches" of noise to a subset (10%)
        blotch_indices = np.random.choice(len(images), size=int(len(images) * 0.1), replace=False)
        for idx in blotch_indices:
            num_blotches = np.random.randint(3, 10)
            for _ in range(num_blotches):
                blotch_size = np.random.randint(10, 50)
                if blotch_size < images[idx].shape[0] and blotch_size < images[idx].shape[1]:
                    y_pos = np.random.randint(0, images[idx].shape[0] - blotch_size)
                    x_pos = np.random.randint(0, images[idx].shape[1] - blotch_size)
                    
                    blotch_noise = np.random.normal(0, noise_level[idx][0][0][0] * 3, 
                                                  size=(blotch_size, blotch_size, 3))
                    noisy_images[idx][y_pos:y_pos+blotch_size, x_pos:x_pos+blotch_size, :] += blotch_noise
        
        # Clip values to [-1, 1] range
        noisy_images = np.clip(noisy_images, -1.0, 1.0)
        
        return noisy_images
    
    def create_tf_dataset(self, clean_images, batch_size):
        """Create TensorFlow dataset with memory optimization and advanced noise generation"""
        # Convert clean_images to float32 if not already
        clean_images = tf.cast(clean_images, tf.float32)
        
        # Create dataset with memory optimization
        dataset = tf.data.Dataset.from_tensor_slices(clean_images)
        
        # Add more advanced noise on-the-fly using tf.data.Dataset
        def add_advanced_noise(clean_img):
            # Add random noise level with higher upper bound for extreme cases
            noise_level = tf.random.uniform(
                shape=[], 
                minval=self.noise_level_min,
                maxval=self.noise_level_max
            )
            
            # Add Gaussian noise
            noise = tf.random.normal(
                shape=tf.shape(clean_img),
                mean=0.0,
                stddev=noise_level
            )
            noisy_img = clean_img + noise
            
            # Randomly apply higher intensity salt and pepper noise with reduced probability
            if tf.random.uniform([]) < 0.3:  # Reduced from 0.4 to 0.3
                # Salt noise (higher density for extreme noise cases)
                salt_amount = noise_level * 0.15
                salt_mask = tf.cast(tf.random.uniform(tf.shape(clean_img)) < salt_amount, tf.float32)
                noisy_img = noisy_img * (1.0 - salt_mask) + salt_mask
                
                # Pepper noise
                pepper_amount = noise_level * 0.15
                pepper_mask = tf.cast(tf.random.uniform(tf.shape(clean_img)) < pepper_amount, tf.float32)
                noisy_img = noisy_img * (1.0 - pepper_mask) - pepper_mask
            
            # Randomly apply speckle noise with reduced probability
            if tf.random.uniform([]) < 0.2:  # Reduced from 0.3 to 0.2
                speckle = tf.random.normal(
                    shape=tf.shape(clean_img),
                    mean=0.0,
                    stddev=noise_level * 1.2
                )
                noisy_img = noisy_img + noisy_img * speckle
            
            # Clip values
            noisy_img = tf.clip_by_value(noisy_img, -1.0, 1.0)
            clean_img = tf.clip_by_value(clean_img, -1.0, 1.0)
            
            return noisy_img, clean_img
        
        # Map the noise addition function with controlled parallelism
        dataset = dataset.map(
            add_advanced_noise,
            num_parallel_calls=CPU_THREADS  # Use configured CPU threads instead of AUTOTUNE
        )
        
        # Optimize memory usage
        dataset = dataset.cache()
        dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)  # Reduced from 2000
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Keep AUTOTUNE for prefetch to avoid bottlenecks
        
        # Validate dataset shape to ensure it matches the model input
        try:
            for x, y in dataset.take(1):
                print(f"Dataset validation - Input shape: {x.shape}, Output shape: {y.shape}")
        except Exception as e:
            print(f"Dataset validation error: {e}")
        
        return dataset 

class AdvancedFineTunedModel:
    """Advanced fine-tuned denoising autoencoder model with improved architecture for high-resolution images and high noise levels"""
    
    def __init__(self, base_img_size=BASE_IMG_SIZE, high_res_img_size=HIGH_RES_IMG_SIZE):
        self.base_img_size = base_img_size
        self.high_res_img_size = high_res_img_size
        self.model = None
    
    def load_pretrained_model(self, model_path):
        """Load the pre-trained model"""
        print(f"Loading pre-trained model from {model_path}...")
        try:
            if not os.path.exists(model_path):
                print(f"Warning: Model path '{model_path}' does not exist.")
                print("You may need to adjust the MODEL_PATH variable or create a base model first.")
                # For testing purposes, create a simple base model
                print("Creating a basic model instead...")
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
            # Create a basic model instead
            print("Creating a basic model instead...")
            with strategy.scope():
                inputs = layers.Input(shape=(self.base_img_size, self.base_img_size, 3))
                x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
                x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
                x = layers.Conv2D(3, 3, padding='same')(x)
                self.model = Model(inputs, x)
            return True
    
    def enhance_model(self):
        """Add enhanced architecture for high-resolution images and high noise levels with CPU optimization"""
        print("Enhancing the model architecture for high-resolution images...")
        print(f"Using input shape: ({self.high_res_img_size}, {self.high_res_img_size}, 3)")
        
        # Create model within strategy's scope for multi-GPU training
        with strategy.scope():
            # We'll create a new model with improved architecture for high noise levels
            # but with reduced complexity to improve CPU performance
            
            inputs = layers.Input(shape=(self.high_res_img_size, self.high_res_img_size, 3), name="input_layer")
            
            # Add simpler attention mechanism
            spatial_attn = layers.Conv2D(8, 5, padding='same')(inputs)  # Reduced filters and kernel size
            spatial_attn = layers.BatchNormalization()(spatial_attn)
            spatial_attn = layers.Activation('relu')(spatial_attn)
            spatial_attn = layers.Conv2D(1, 5, padding='same')(spatial_attn)  # Reduced kernel size
            spatial_attn = layers.Activation('sigmoid')(spatial_attn)
            
            # Simplified channel attention
            avg_pool = layers.GlobalAveragePooling2D()(inputs)
            avg_pool = layers.Reshape((1, 1, 3))(avg_pool)
            avg_pool = layers.Conv2D(3, 1, padding='same', activation='sigmoid')(avg_pool)
            
            # Apply attention
            spatial_enhanced = layers.Multiply()([inputs, spatial_attn])
            enhanced_input = layers.Multiply()([spatial_enhanced, avg_pool])
            
            # Apply noise reduction preprocessing
            noise_reduced = layers.Conv2D(16, 3, padding='same')(enhanced_input)
            noise_reduced = layers.BatchNormalization()(noise_reduced)
            noise_reduced = layers.Activation('relu')(noise_reduced)
            noise_reduced = layers.Conv2D(3, 3, padding='same')(noise_reduced)
            noise_reduced = layers.Add()([noise_reduced, enhanced_input])
            
            # Resize the input to match the pre-trained model's expected size
            resized_input = layers.Resizing(
                height=self.base_img_size,
                width=self.base_img_size,
                interpolation='bilinear'
            )(noise_reduced)
            
            # Get output from original model (transfer learning)
            base_output = self.model(resized_input)
            
            # Upscale the output back to high resolution with simplified upsampling
            final_output = layers.Resizing(
                height=self.high_res_img_size,
                width=self.high_res_img_size,
                interpolation='bilinear'
            )(base_output)
            
            # Final cleanup for high-frequency details
            final_output = layers.Conv2D(16, 3, padding='same')(final_output)
            final_output = layers.BatchNormalization()(final_output)
            final_output = layers.Activation('relu')(final_output)
            final_output = layers.Conv2D(3, 3, padding='same')(final_output)
            
            # Skip connection from input to output to maintain image structure
            outputs = layers.Add()([final_output, inputs])
            
            # Create new enhanced model
            enhanced_model = Model(inputs, outputs, name="enhanced_high_noise_denoising_autoencoder")
            
            # Compile model with optimizer - use gradient clipping for stability with high noise
            enhanced_model.compile(
                optimizer=optimizers.Adam(
                    learning_rate=LEARNING_RATE,
                    clipnorm=1.0  # Gradient clipping added for stability
                ),
                loss='mse',
                metrics=['mae']
            )
        
        # Initialize the model by running a forward pass with smaller batch
        sample_input = tf.random.uniform((1, self.high_res_img_size, self.high_res_img_size, 3))
        _ = enhanced_model(sample_input)
        
        self.model = enhanced_model
        
        print("Model enhanced successfully")
        print(f"Enhanced model input shape: {self.model.input_shape}")
        return self.model
    
    def train(self, train_dataset, val_dataset=None, epochs=EPOCHS):
        """Train the model using Keras's built-in fit method with improved callbacks"""
        if self.model is None:
            raise ValueError("Model not loaded or built")
        
        # Early stopping callback with increased patience
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss' if val_dataset else 'loss',
            patience=PATIENCE,
            min_delta=MIN_DELTA,
            restore_best_weights=True,
            verbose=1
        )
        
        # Model checkpoint callback
        model_checkpoint = callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / "advanced_high_noise_denoiser_best.keras"),
            monitor='val_loss' if val_dataset else 'loss',
            save_best_only=True,
            verbose=1
        )
        
        # TensorBoard callback with reduced logging frequency
        tensorboard = callbacks.TensorBoard(
            log_dir=str(LOGS_DIR),
            update_freq='epoch',
            profile_batch=0  # Disable profiling to reduce overhead
        )
        
        # Learning rate scheduler - reduce learning rate when plateau
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss' if val_dataset else 'loss',
            factor=0.5,
            patience=PATIENCE // 2,
            min_delta=MIN_DELTA,
            min_lr=1e-7,
            verbose=1
        )
        
        # Custom callback to clean up memory after each epoch
        class MemoryCleanupCallback(callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                gc.collect()
                if hasattr(tf, 'keras') and hasattr(tf.keras, 'backend'):
                    tf.keras.backend.clear_session()
            
                print(f"Memory cleanup performed after epoch {epoch + 1}")
        
        # Custom callback to save model less frequently to reduce CPU/IO overhead
        class FrequentSaveCallback(callbacks.Callback):
            def __init__(self, save_freq=40):  # Reduced frequency (now every 40 epochs)
                super(FrequentSaveCallback, self).__init__()
                self.save_freq = save_freq
                
            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % self.save_freq == 0:
                    self.model.save(str(MODELS_DIR / f"model_epoch_{epoch+1}.keras"))
                    print(f"Model saved at epoch {epoch+1}")
        
        # Convert existing datasets to non-distributed format if needed
        # Since Keras fit() will handle the distribution strategy automatically
        train_dataset_for_fit = train_dataset
        val_dataset_for_fit = val_dataset
        
        # Check TensorFlow version to handle parameter compatibility
        tf_version = tf.__version__.split('.')
        major_version = int(tf_version[0])
        minor_version = int(tf_version[1]) if len(tf_version) > 1 else 0
        
        # Train the model using Keras fit method which handles distribution internally
        print("Starting training with Keras fit method for multi-GPU support...")
        
        # Different kwargs based on TensorFlow version
        fit_kwargs = {
            'epochs': epochs,
            'verbose': 1,
            'callbacks': [
                early_stopping, 
                model_checkpoint, 
                tensorboard, 
                reduce_lr,
                MemoryCleanupCallback(),
                FrequentSaveCallback(save_freq=40)
            ]
        }
        
        # Only add workers and multiprocessing parameters for older TF versions that support them
        if major_version < 2 or (major_version == 2 and minor_version < 17):
            print("Using multiprocessing parameters (TensorFlow < 2.17)")
            fit_kwargs['workers'] = CPU_THREADS
            fit_kwargs['use_multiprocessing'] = False
        else:
            print("Skipping multiprocessing parameters (TensorFlow >= 2.17)")
        
        # Run the fit method with appropriate parameters
        history = self.model.fit(
            train_dataset_for_fit,
            validation_data=val_dataset_for_fit,
            **fit_kwargs
        )
        
        # Save final model
        self.model.save(str(MODELS_DIR / "advanced_high_noise_denoiser_final.keras"))
        
        # Convert history to dictionary format for compatibility with our existing code
        history_dict = {
            'loss': history.history['loss'],
            'mae': history.history['mae'],
        }
        
        if val_dataset:
            history_dict['val_loss'] = history.history['val_loss']
            history_dict['val_mae'] = history.history['val_mae']
        
        return history_dict
    
    def generate_samples(self, test_images, noise_levels=[0.2, 0.3, 0.5]):
        """Generate sample denoising results with multiple noise levels"""
        os.makedirs(SAMPLES_DIR, exist_ok=True)
        
        for i, clean_img in enumerate(test_images[:5]):  # Use up to 5 test images
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
                noisy_img = clean_img + np.random.normal(0, noise_level, clean_img.shape)
                # Add some salt and pepper noise for extreme cases
                if noise_level >= 0.3:
                    # Salt noise
                    salt_mask = np.random.random(clean_img.shape) < noise_level * 0.1
                    noisy_img[salt_mask] = 1.0
                    # Pepper noise
                    pepper_mask = np.random.random(clean_img.shape) < noise_level * 0.1
                    noisy_img[pepper_mask] = -1.0
                
                noisy_img = np.clip(noisy_img, -1.0, 1.0)
                
                # Predict - use model directly without distribution strategy
                # since we're just doing inference
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


def main():
    """Main fine-tuning function with improved parameters for high-noise high-resolution images"""
    print("Starting advanced fine-tuning of image denoising model for high-noise high-resolution images...")
    start_time = time.time()
    
    # Print TensorFlow version and GPU availability
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Number of GPUs: {len(tf.config.list_physical_devices('GPU'))}")
    print(f"Devices visible to TensorFlow: {tf.config.list_physical_devices()}")
    print(f"MirroredStrategy replicas: {strategy.num_replicas_in_sync}")
    
    # Create data loader and prepare dataset
    data_loader = AdvancedMemoryEfficientDataLoader(
        base_img_size=BASE_IMG_SIZE,
        high_res_img_size=HIGH_RES_IMG_SIZE,
        noise_level_min=NOISE_LEVEL_MIN,
        noise_level_max=NOISE_LEVEL_MAX,
        patch_size=HIGH_RES_IMG_SIZE  # Use HIGH_RES_IMG_SIZE to ensure consistency
    )
    
    # Load and prepare dataset
    clean_patches = data_loader.load_and_prepare_dataset()
    
    # Split into training and validation sets (85/15) - more validation data
    split_idx = int(0.85 * len(clean_patches))
    train_patches = clean_patches[:split_idx]
    val_patches = clean_patches[split_idx:]
    
    print(f"Training patches: {len(train_patches)}")
    print(f"Validation patches: {len(val_patches)}")
    
    # Create TensorFlow datasets with smaller batch size for higher resolution
    train_dataset = data_loader.create_tf_dataset(train_patches, BATCH_SIZE)
    val_dataset = data_loader.create_tf_dataset(val_patches, BATCH_SIZE)
    
    # Free up memory
    del clean_patches
    gc.collect()
    
    # Load pre-trained model and enhance it 
    fine_tuned_model = AdvancedFineTunedModel(
        base_img_size=BASE_IMG_SIZE,
        high_res_img_size=HIGH_RES_IMG_SIZE
    )
    if not fine_tuned_model.load_pretrained_model(MODEL_PATH):
        print("Failed to load pre-trained model. Exiting.")
        return
    
    # Enhance the model with improved architecture
    fine_tuned_model.enhance_model()
    
    # Print model summary
    fine_tuned_model.model.summary()
    
    # Train the model with increased epochs
    history = fine_tuned_model.train(train_dataset, val_dataset, epochs=EPOCHS)
    
    # Generate sample results with multiple noise levels
    sample_count = 5
    sample_images = val_patches[:sample_count]
    fine_tuned_model.generate_samples(sample_images, noise_levels=[0.2, 0.35, 0.5])
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['mae'], label='Training MAE')
    if 'val_mae' in history:
        plt.plot(history['val_mae'], label='Validation MAE')
    plt.title('Mean Absolute Error')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(str(BASE_DIR / "high_noise_fine_tuning_history.png"))
    
    # Print training time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Advanced high-noise fine-tuning completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Model saved to {MODELS_DIR}")
    print(f"Sample results saved to {SAMPLES_DIR}")


if __name__ == "__main__":
    main() 