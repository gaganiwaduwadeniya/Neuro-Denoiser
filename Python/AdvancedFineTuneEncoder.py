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

# Configuration parameters
BATCH_SIZE = 4 * max(1, strategy.num_replicas_in_sync)  # Adjusted batch size for multi-GPU training
BASE_IMG_SIZE = 128  # Base image size for training
HIGH_RES_IMG_SIZE = 256  # Changed from 512 to 256 to match patch size
NOISE_LEVEL_MIN = 0.05
NOISE_LEVEL_MAX = 0.3
EPOCHS = 100  # Increased epochs
LEARNING_RATE = 1e-5  # Lower learning rate for fine-tuning
PATIENCE = 15  # Early stopping patience
MIN_DELTA = 1e-4
BASE_DIR = Path("advanced_fine_tuned_encoder")  # Updated directory

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
SET5_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/Set5.zip"
SET14_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/Set14.zip"
URBAN100_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/Urban100.zip"

# Memory optimization parameters
MAX_PATCHES_PER_IMAGE = 5  # Reduced to save memory
MAX_TOTAL_PATCHES = 10000  # Reduced total patches
PATCH_SIZE = 256  # Matches HIGH_RES_IMG_SIZE
MEMORY_EFFICIENT_BATCH_SIZE = 50  # Batch size for memory-efficient processing

# Create necessary directories
os.makedirs(BASE_DIR, exist_ok=True)
MODELS_DIR = BASE_DIR / "models"
SAMPLES_DIR = BASE_DIR / "samples"
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"

for directory in [MODELS_DIR, SAMPLES_DIR, LOGS_DIR, DATA_DIR]:
    os.makedirs(directory, exist_ok=True)


class AdvancedMemoryEfficientDataLoader:
    """Memory-efficient data loader for high-resolution image denoising tasks"""
    
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
    
    def download_div2k_subset(self, num_images=400):
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
    
    def download_bsds_subset(self, num_images=300):
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
    
    def download_urban100_subset(self, num_images=100):
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
    
    def create_synthetic_images(self, num_images=1000):
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
                pattern_types = ['gradient', 'checkerboard', 'noise', 'circles', 'lines', 
                                 'texture', 'gradient_noise', 'mixed']
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
        
        # Get images from different sources with reduced numbers
        div2k_images = self.download_div2k_subset(200)  # Reduced from 400
        bsds_images = self.download_bsds_subset(150)    # Reduced from 300
        kodak_images = self.download_kodak_subset(24)   # Keep same
        set5_images = self.download_set5_subset(5)      # Keep same
        set14_images = self.download_set14_subset(14)   # Keep same
        urban100_images = self.download_urban100_subset(50)  # Reduced from 100
        synthetic_images = self.create_synthetic_images(500)  # Reduced from 1000
        
        # Combine all image sources
        all_images = div2k_images + bsds_images + kodak_images + set5_images + set14_images + urban100_images + synthetic_images
        if not all_images:
            raise ValueError("No images available for fine-tuning")
        
        print(f"Total images available: {len(all_images)}")
        random.shuffle(all_images)
        
        # Memory-efficient patch extraction
        clean_patches = []
        patch_count = 0
        
        # Process images in smaller batches to manage memory better
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
                    
                    # Extract fewer patches per image
                    width, height = img.size
                    max_patches = min(MAX_PATCHES_PER_IMAGE, 
                                     (width // self.patch_size) * (height // self.patch_size))
                    
                    for _ in range(max_patches):
                        if patch_count >= MAX_TOTAL_PATCHES:
                            break
                        
                        # Get random patch
                        if width > self.patch_size and height > self.patch_size:
                            left = random.randint(0, width - self.patch_size)
                            top = random.randint(0, height - self.patch_size)
                            patch = img.crop((left, top, left + self.patch_size, top + self.patch_size))
                        else:
                            patch = img.resize((self.patch_size, self.patch_size), Image.LANCZOS)
                        
                        # Convert to numpy array and normalize
                        patch_array = np.array(patch).astype(np.float32) / 255.0
                        batch_patches.append(patch_array)
                        patch_count += 1
                    
                    # Free memory
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
        chunk_size = 1000
        final_patches = []
        
        for i in range(0, len(clean_patches), chunk_size):
            chunk = np.array(clean_patches[i:i+chunk_size])
            # Normalize to [-1, 1] range
            chunk = chunk * 2.0 - 1.0
            final_patches.append(chunk)
            gc.collect()
        
        # Combine chunks
        clean_patches = np.concatenate(final_patches, axis=0)
        del final_patches
        gc.collect()
        
        print(f"Clean patches shape: {clean_patches.shape}")
        return clean_patches
    
    def add_noise(self, images):
        """Add random noise to images with focus on salt & pepper noise"""
        # Create a copy of the images
        noisy_images = images.copy()
        
        # Random noise level for each image
        noise_level = np.random.uniform(self.noise_level_min, self.noise_level_max, size=(len(images), 1, 1, 1))
        
        # Add Gaussian noise
        noise = np.random.normal(0, 1, size=images.shape) * noise_level
        noisy_images = images + noise
        
        # Add salt and pepper noise to a larger subset (30%) of images
        salt_pepper_indices = np.random.choice(len(images), size=len(images)//3, replace=False)
        for idx in salt_pepper_indices:
            # Higher amount of salt and pepper noise
            amount = 0.1 * noise_level[idx][0][0][0]  # Increased from 0.05 to 0.1
            
            # Salt (white) noise
            mask = np.random.random(images[idx].shape) < amount/2
            noisy_images[idx][mask] = 1.0
            
            # Pepper (black) noise
            mask = np.random.random(images[idx].shape) < amount/2
            noisy_images[idx][mask] = -1.0
        
        # Add speckle noise to another subset (20%)
        speckle_indices = np.random.choice(len(images), size=len(images)//5, replace=False)
        for idx in speckle_indices:
            speckle = np.random.normal(0, noise_level[idx][0][0][0], size=images[idx].shape)
            noisy_images[idx] += noisy_images[idx] * speckle
        
        # Clip values to [-1, 1] range
        noisy_images = np.clip(noisy_images, -1.0, 1.0)
        
        return noisy_images
    
    def create_tf_dataset(self, clean_images, batch_size):
        """Create TensorFlow dataset with memory optimization"""
        # Convert clean_images to float32 if not already
        clean_images = tf.cast(clean_images, tf.float32)
        
        # Create dataset with memory optimization
        dataset = tf.data.Dataset.from_tensor_slices(clean_images)
        
        # Add noise on-the-fly using tf.data.Dataset
        def add_noise(clean_img):
            # Add random noise level
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
            
            # Add salt and pepper noise randomly
            if tf.random.uniform([]) < 0.3:  # 30% chance of adding salt and pepper noise
                mask = tf.random.uniform(tf.shape(clean_img)) < (noise_level * 0.1)
                salt = tf.cast(mask, tf.float32)
                pepper = tf.cast(tf.random.uniform(tf.shape(clean_img)) < (noise_level * 0.1), tf.float32)
                noisy_img = noisy_img * (1.0 - salt) + salt  # Add salt
                noisy_img = noisy_img * (1.0 - pepper)  # Add pepper
            
            # Clip values
            noisy_img = tf.clip_by_value(noisy_img, -1.0, 1.0)
            clean_img = tf.clip_by_value(clean_img, -1.0, 1.0)
            
            return noisy_img, clean_img
        
        # Map the noise addition function with memory optimization
        dataset = dataset.map(
            add_noise,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Optimize memory usage
        dataset = dataset.cache()
        dataset = dataset.shuffle(buffer_size=1000, reshuffle_each_iteration=True)  # Reduced buffer size
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        # Validate dataset shape to ensure it matches the model input
        try:
            for x, y in dataset.take(1):
                print(f"Dataset validation - Input shape: {x.shape}, Output shape: {y.shape}")
        except Exception as e:
            print(f"Dataset validation error: {e}")
        
        return dataset


class AdvancedFineTunedModel:
    """Advanced fine-tuned denoising autoencoder model with improved architecture for high-resolution images"""
    
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
        """Add enhancements to the model without altering its basic structure"""
        print("Enhancing the model architecture for high-resolution images...")
        print(f"Using input shape: ({self.high_res_img_size}, {self.high_res_img_size}, 3)")
        
        # Create model within strategy's scope for multi-GPU training
        with strategy.scope():
        # We'll create a new model that wraps the existing one and adds attention
        # This approach avoids the layer incompatibility issues
        
        inputs = layers.Input(shape=(self.high_res_img_size, self.high_res_img_size, 3), name="input_layer")
        
        # Add attention mechanism before passing to the original model
        attention = layers.Conv2D(filters=3, kernel_size=1, padding='same')(inputs)
        attention = layers.Activation('sigmoid')(attention)
        enhanced_input = layers.Multiply()([inputs, attention])
        
        # Resize the input to match the pre-trained model's expected size
        resized_input = layers.Resizing(
            height=self.base_img_size,
            width=self.base_img_size,
            interpolation='bilinear'
        )(enhanced_input)
        
        # Get output from original model
        outputs = self.model(resized_input)
        
        # Upscale the output back to high resolution
        upscaled_output = layers.Resizing(
            height=self.high_res_img_size,
            width=self.high_res_img_size,
            interpolation='bilinear'
        )(outputs)
        
        # Create new enhanced model
        enhanced_model = Model(inputs, upscaled_output, name="enhanced_denoising_autoencoder")
        
            # Compile model with optimizer
            enhanced_model.compile(
                optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
                loss='mse',
                metrics=['mae']
            )
        
        # Initialize the model by running a forward pass
        sample_input = tf.random.uniform((1, self.high_res_img_size, self.high_res_img_size, 3))
        _ = enhanced_model(sample_input)
        
        self.model = enhanced_model
        
        print("Model enhanced successfully")
        print(f"Enhanced model input shape: {self.model.input_shape}")
        return self.model
    
    def train(self, train_dataset, val_dataset=None, epochs=EPOCHS):
        """Train the model using Keras's built-in fit method for better distribution support"""
        if self.model is None:
            raise ValueError("Model not loaded or built")
        
        # Early stopping callback
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss' if val_dataset else 'loss',
            patience=PATIENCE,
            min_delta=MIN_DELTA,
            restore_best_weights=True,
            verbose=1
        )
        
        # Model checkpoint callback
        model_checkpoint = callbacks.ModelCheckpoint(
            filepath=str(MODELS_DIR / "advanced_denoising_autoencoder_best.keras"),
            monitor='val_loss' if val_dataset else 'loss',
            save_best_only=True,
            verbose=1
        )
        
        # TensorBoard callback
        tensorboard = callbacks.TensorBoard(
            log_dir=str(LOGS_DIR),
            update_freq='epoch'
        )
        
        # Custom callback to clean up memory after each epoch
        class MemoryCleanupCallback(callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                gc.collect()
                if hasattr(tf, 'keras') and hasattr(tf.keras, 'backend'):
                    tf.keras.backend.clear_session()
            
                print(f"Memory cleanup performed after epoch {epoch + 1}")
        
        # Convert existing datasets to non-distributed format if needed
        # Since Keras fit() will handle the distribution strategy automatically
        train_dataset_for_fit = train_dataset
        val_dataset_for_fit = val_dataset
        
        # Train the model using Keras fit method which handles distribution internally
        print("Starting training with Keras fit method for multi-GPU support...")
        history = self.model.fit(
            train_dataset_for_fit,
            validation_data=val_dataset_for_fit,
            epochs=epochs,
            callbacks=[early_stopping, model_checkpoint, tensorboard, MemoryCleanupCallback()],
            verbose=1
        )
        
        # Save final model
        self.model.save(str(MODELS_DIR / "advanced_denoising_autoencoder_final.keras"))
        
        # Convert history to dictionary format for compatibility with our existing code
        history_dict = {
            'loss': history.history['loss'],
            'mae': history.history['mae'],
        }
        
        if val_dataset:
            history_dict['val_loss'] = history.history['val_loss']
            history_dict['val_mae'] = history.history['val_mae']
        
        return history_dict
    
    def generate_samples(self, test_images, noise_level=0.2):
        """Generate sample denoising results"""
        os.makedirs(SAMPLES_DIR, exist_ok=True)
        
        for i, clean_img in enumerate(test_images[:5]):  # Use up to 5 test images
            # Add noise
            noisy_img = clean_img + np.random.normal(0, noise_level, clean_img.shape)
            noisy_img = np.clip(noisy_img, -1.0, 1.0)
            
            # Predict - use model directly without distribution strategy
            # since we're just doing inference
            denoised_img = self.model.predict(np.expand_dims(noisy_img, 0))[0]
            
            # Convert to [0, 1] range for visualization
            clean_display = (clean_img + 1) / 2.0
            noisy_display = (noisy_img + 1) / 2.0
            denoised_display = (denoised_img + 1) / 2.0
            
            # Create visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(np.clip(clean_display, 0, 1))
            axes[0].set_title("Original")
            axes[0].axis("off")
            
            axes[1].imshow(np.clip(noisy_display, 0, 1))
            axes[1].set_title(f"Noisy (σ={noise_level:.2f})")
            axes[1].axis("off")
            
            axes[2].imshow(np.clip(denoised_display, 0, 1))
            axes[2].set_title("Denoised")
            axes[2].axis("off")
            
            plt.tight_layout()
            plt.savefig(str(SAMPLES_DIR / f"sample_{i+1}.png"), dpi=150)
            plt.close()
            
            # Save individual images
            plt.imsave(str(SAMPLES_DIR / f"original_{i+1}.png"), np.clip(clean_display, 0, 1))
            plt.imsave(str(SAMPLES_DIR / f"noisy_{i+1}.png"), np.clip(noisy_display, 0, 1))
            plt.imsave(str(SAMPLES_DIR / f"denoised_{i+1}.png"), np.clip(denoised_display, 0, 1))


def main():
    """Main fine-tuning function"""
    print("Starting advanced fine-tuning of image denoising model for high-resolution images...")
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
    
    # Split into training and validation sets (90/10)
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
    gc.collect()
    
    # Load pre-trained model and enhance it within strategy scope
    with strategy.scope():
    fine_tuned_model = AdvancedFineTunedModel(
        base_img_size=BASE_IMG_SIZE,
        high_res_img_size=HIGH_RES_IMG_SIZE
    )
    if not fine_tuned_model.load_pretrained_model(MODEL_PATH):
        print("Failed to load pre-trained model. Exiting.")
        return
    
        # Enhance the model - calls will be made inside strategy.scope()
    fine_tuned_model.enhance_model()
    
    # Print model summary
    fine_tuned_model.model.summary()
    
    # Train the model using Keras's fit method which handles distribution automatically
    history = fine_tuned_model.train(train_dataset, val_dataset, epochs=EPOCHS)
    
    # Generate sample results
    sample_count = 5
    sample_images = val_patches[:sample_count]
    fine_tuned_model.generate_samples(sample_images)
    
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
    plt.savefig(str(BASE_DIR / "advanced_fine_tuning_history.png"))
    
    # Print training time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Advanced fine-tuning completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Model saved to {MODELS_DIR}")
    print(f"Sample results saved to {SAMPLES_DIR}")


if __name__ == "__main__":
    main() 