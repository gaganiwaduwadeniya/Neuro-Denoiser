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
except:
    strategy = tf.distribute.get_strategy()
    print("Using default strategy")

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Configuration parameters
BATCH_SIZE = 16 * max(1, strategy.num_replicas_in_sync)  # Increased batch size for better accuracy
IMG_SIZE = 128  # Match the model's expected input size
NOISE_LEVEL_MIN = 0.05
NOISE_LEVEL_MAX = 0.3
EPOCHS = 50  # Fewer epochs for fine-tuning
LEARNING_RATE = 5e-5  # Lower learning rate for fine-tuning
PATIENCE = 10  # Early stopping patience
MIN_DELTA = 1e-4
BASE_DIR = Path("/kaggle/working/fine_tuned_encoder")  # Updated for Kaggle compatibility
MODEL_PATH = "/kaggle/input/denoising-autoencoder/denoising_autoencoder_best.keras"  # Path to the pre-trained model

# Additional datasets for fine-tuning
DIV2K_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
COCO_URL = "https://download.openmmlab.com/mmediting/data/coco_val2017.zip"
COCO_ALT_URL = "https://download.openmmlab.com/mmediting/data/coco_val2017_256.zip"  # Alternative URL
Flickr2K_URL = "https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar"  # High-quality Flickr images
Flickr2K_ALT_URL = "https://data.csail.mit.edu/places/places365/Flickr2K.tar"  # Alternative URL
BSDS_URL = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300-images.tgz"  # Berkeley Segmentation Dataset
MAX_PATCHES_PER_IMAGE = 20  # Reduced to allow more diverse images
MAX_TOTAL_PATCHES = 25000  # Increased total patches

# Create necessary directories
os.makedirs(BASE_DIR, exist_ok=True)
MODELS_DIR = BASE_DIR / "models"
SAMPLES_DIR = BASE_DIR / "samples"
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"

for directory in [MODELS_DIR, SAMPLES_DIR, LOGS_DIR, DATA_DIR]:
    os.makedirs(directory, exist_ok=True)


class MemoryEfficientDataLoader:
    """Memory-efficient data loader for image denoising tasks"""
    
    def __init__(self, img_size=IMG_SIZE, noise_level_min=NOISE_LEVEL_MIN, 
                 noise_level_max=NOISE_LEVEL_MAX, patch_size=IMG_SIZE):
        self.img_size = img_size
        self.noise_level_min = noise_level_min
        self.noise_level_max = noise_level_max
        self.patch_size = patch_size
        self.data_dir = DATA_DIR
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
    
    def download_div2k_subset(self, num_images=200):
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
    
    def download_coco_subset(self, num_images=1000):
        """Download a subset of COCO dataset"""
        coco_dir = self.data_dir / "coco"
        os.makedirs(coco_dir, exist_ok=True)
        
        # Check if we already have enough COCO images
        existing_images = list(coco_dir.glob("**/*.jpg"))
        if len(existing_images) >= num_images:
            print(f"Using {len(existing_images)} existing COCO images")
            return [str(img) for img in existing_images[:num_images]]
        
        # Download the COCO dataset
        zip_path = self.data_dir / "coco.zip"
        if not zip_path.exists():
            # Try primary URL first
            urls_to_try = [COCO_URL, COCO_ALT_URL]
            success = False
            
            for url in urls_to_try:
                if success:
                    break
                    
                try:
                    print(f"Downloading COCO dataset from {url}...")
                    # Use a more robust download method with timeout and retries
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            urllib.request.urlretrieve(url, zip_path)
                            print(f"COCO dataset downloaded to {zip_path}")
                            success = True
                            break
                        except Exception as e:
                            if attempt < max_retries - 1:
                                print(f"Download attempt {attempt+1} failed: {e}. Retrying...")
                                time.sleep(2)  # Wait before retrying
                            else:
                                print(f"Failed to download COCO dataset from {url} after {max_retries} attempts: {e}")
                except Exception as e:
                    print(f"Error downloading COCO dataset from {url}: {e}")
            
            if not success:
                print("Failed to download COCO dataset from all available URLs")
                return []
        
        # Extract the zip file
        try:
            print("Extracting COCO dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract only a subset of images to save space
                count = 0
                for file in zip_ref.namelist():
                    # Check if it's a jpg or jpeg file
                    if file.lower().endswith(('.jpg', '.jpeg')):
                        if count >= num_images:
                            break
                        # Extract the file
                        zip_ref.extract(file, coco_dir)
                        count += 1
                        if count % 100 == 0:
                            print(f"Extracted {count} images so far...")
            
            # Get the paths of extracted images - search recursively
            coco_images = []
            for ext in ['.jpg', '.jpeg']:
                coco_images.extend(list(coco_dir.glob(f"**/*{ext}")))
            
            coco_images = [str(img) for img in coco_images[:num_images]]
            print(f"Extracted {len(coco_images)} COCO images")
            return coco_images
            
        except Exception as e:
            print(f"Error extracting COCO dataset: {e}")
            return []
    
    def download_flickr2k_subset(self, num_images=500):
        """Download a subset of Flickr2K dataset"""
        flickr_dir = self.data_dir / "flickr2k"
        os.makedirs(flickr_dir, exist_ok=True)
        
        # Check if we already have enough Flickr2K images
        existing_images = list(flickr_dir.glob("**/*.png"))
        if len(existing_images) >= num_images:
            print(f"Using {len(existing_images)} existing Flickr2K images")
            return [str(img) for img in existing_images[:num_images]]
        
        # Download the Flickr2K dataset
        tar_path = self.data_dir / "flickr2k.tar"
        if not tar_path.exists():
            # Try primary URL first
            urls_to_try = [Flickr2K_URL, Flickr2K_ALT_URL]
            success = False
            
            for url in urls_to_try:
                if success:
                    break
                    
                try:
                    print(f"Downloading Flickr2K dataset from {url}...")
                    # Use a more robust download method with timeout and retries
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            urllib.request.urlretrieve(url, tar_path)
                            print(f"Flickr2K dataset downloaded to {tar_path}")
                            success = True
                            break
                        except Exception as e:
                            if attempt < max_retries - 1:
                                print(f"Download attempt {attempt+1} failed: {e}. Retrying...")
                                time.sleep(2)  # Wait before retrying
                            else:
                                print(f"Failed to download Flickr2K dataset from {url} after {max_retries} attempts: {e}")
                except Exception as e:
                    print(f"Error downloading Flickr2K dataset from {url}: {e}")
            
            if not success:
                print("Failed to download Flickr2K dataset from all available URLs")
                return []
        
        # Extract the tar file
        try:
            print("Extracting Flickr2K dataset...")
            import tarfile
            with tarfile.open(tar_path, 'r') as tar_ref:
                # Extract only a subset of images to save space
                count = 0
                for member in tar_ref.getmembers():
                    if member.name.lower().endswith('.png'):
                        if count >= num_images:
                            break
                        # Extract the file
                        tar_ref.extract(member, flickr_dir)
                        count += 1
                        if count % 50 == 0:
                            print(f"Extracted {count} images so far...")
            
            # Get the paths of extracted images - search recursively
            flickr_images = list(flickr_dir.glob("**/*.png"))
            flickr_images = [str(img) for img in flickr_images[:num_images]]
            print(f"Extracted {len(flickr_images)} Flickr2K images")
            return flickr_images
            
        except Exception as e:
            print(f"Error extracting Flickr2K dataset: {e}")
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
            import tarfile
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
                img = np.zeros((self.img_size * 2, self.img_size * 2, 3), dtype=np.float32)
                
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
                    tile_size = self.img_size // 8
                    for y in range(img.shape[0]):
                        for x in range(img.shape[1]):
                            if ((x // tile_size) + (y // tile_size)) % 2 == 0:
                                img[y, x] = [0.9, 0.9, 0.9]
                            else:
                                img[y, x] = [0.1, 0.1, 0.1]
                
                elif pattern == 'noise':
                    # Create a noise pattern with structure
                    base = np.random.rand(self.img_size // 8, self.img_size // 8, 3)
                    img = np.kron(base, np.ones((16, 16, 1)))
                
                elif pattern == 'circles':
                    # Create concentric circles
                    center_y, center_x = img.shape[0] // 2, img.shape[1] // 2
                    for y in range(img.shape[0]):
                        for x in range(img.shape[1]):
                            dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
                            img[y, x] = [
                                0.5 + 0.5 * np.sin(dist / 20),
                                0.5 + 0.5 * np.cos(dist / 15),
                                0.5 + 0.5 * np.sin(dist / 10)
                            ]
                
                elif pattern == 'lines':
                    # Create line patterns
                    for y in range(img.shape[0]):
                        for x in range(img.shape[1]):
                            if x % 20 < 10:
                                img[y, x, 0] = 0.8
                            if y % 15 < 7:
                                img[y, x, 1] = 0.7
                            if (x + y) % 25 < 12:
                                img[y, x, 2] = 0.9
                
                elif pattern == 'texture':
                    # Create texture patterns
                    for y in range(img.shape[0]):
                        for x in range(img.shape[1]):
                            img[y, x] = [
                                0.5 + 0.5 * np.sin(x/10) * np.cos(y/10),
                                0.5 + 0.5 * np.sin(x/15) * np.cos(y/15),
                                0.5 + 0.5 * np.sin(x/20) * np.cos(y/20)
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
                                img[y, x, 0] = 0.5 + 0.5 * np.sin(dist / 20)
                                
                                if x % 20 < 10:
                                    img[y, x, 1] = 0.8
                                if y % 15 < 7:
                                    img[y, x, 2] = 0.7
                
                # Save the image
                plt.imsave(img_path, np.clip(img, 0, 1))
                images.append(str(img_path))
            
            # Free memory after each batch
            gc.collect()
        
        print(f"Created {len(images)} synthetic images")
        return images
    
    def load_and_prepare_dataset(self):
        """Load and prepare the mixed dataset for fine-tuning"""
        print("Loading and preparing dataset for fine-tuning...")
        
        # Get images from different sources with increased numbers
        div2k_images = self.download_div2k_subset(400)  # DIV2K dataset
        bsds_images = self.download_bsds_subset(300)    # Berkeley Segmentation Dataset
        kodak_images = self.download_kodak_subset(24)   # Kodak dataset
        set5_images = self.download_set5_subset(5)      # Set5 dataset
        set14_images = self.download_set14_subset(14)   # Set14 dataset
        synthetic_images = self.create_synthetic_images(1000)  # Increased synthetic images
        
        # Combine all image sources
        all_images = div2k_images + bsds_images + kodak_images + set5_images + set14_images + synthetic_images
        if not all_images:
            raise ValueError("No images available for fine-tuning")
        
        print(f"Total images available: {len(all_images)}")
        random.shuffle(all_images)
        
        # Memory-efficient patch extraction: Extract fewer patches per image but from more images
        clean_patches = []
        patch_count = 0
        
        # Process images in smaller batches to manage memory better
        batch_size = 50
        for i in range(0, len(all_images), batch_size):
            if patch_count >= MAX_TOTAL_PATCHES:
                break
            
            batch_images = all_images[i:i+batch_size]
            for img_path in tqdm(batch_images, desc=f"Extracting patches from batch {i//batch_size + 1}/{len(all_images)//batch_size + 1}"):
                if patch_count >= MAX_TOTAL_PATCHES:
                    break
                
                try:
                    # Load and resize image
                    img = Image.open(img_path).convert('RGB')
                    
                    # Extract multiple random patches from the image
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
                            # Resize if image is too small
                            patch = img.resize((self.patch_size, self.patch_size), Image.LANCZOS)
                        
                        # Convert to numpy array and normalize
                        patch_array = np.array(patch).astype(np.float32) / 255.0
                        clean_patches.append(patch_array)
                        patch_count += 1
                    
                    # Free memory
                    del img
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        print(f"Extracted {len(clean_patches)} patches for fine-tuning")
        
        # Convert to numpy array
        clean_patches = np.array(clean_patches)
        print(f"Clean patches shape: {clean_patches.shape}")
        
        # Normalize to [-1, 1] range for better training
        clean_patches = clean_patches * 2.0 - 1.0
        
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
        """Create TensorFlow dataset for training"""
        # Convert clean_images to float32 if not already
        clean_images = tf.cast(clean_images, tf.float32)
        
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
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices(clean_images)
        
        # Map the noise addition function
        dataset = dataset.map(
            add_noise,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Cache the dataset for better performance
        dataset = dataset.cache()
        
        # Shuffle, batch, and prefetch
        dataset = dataset.shuffle(buffer_size=5000, reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        # Validate the dataset
        try:
            # Get one batch to validate
            for x, y in dataset.take(1):
                print(f"Input shape: {x.shape}, Output shape: {y.shape}")
                print(f"Input dtype: {x.dtype}, Output dtype: {y.dtype}")
                print(f"Input range: [{tf.reduce_min(x)}, {tf.reduce_max(x)}]")
                print(f"Output range: [{tf.reduce_min(y)}, {tf.reduce_max(y)}]")
        except Exception as e:
            print(f"Dataset validation error: {e}")
            raise
        
        return dataset


class FineTunedModel:
    """Fine-tuned denoising autoencoder model with improved architecture"""
    
    def __init__(self, img_size=IMG_SIZE):
        self.img_size = img_size
        self.model = None
    
    def load_pretrained_model(self, model_path):
        """Load the pre-trained model"""
        print(f"Loading pre-trained model from {model_path}...")
        try:
            self.model = load_model(model_path)
            print("Pre-trained model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading pre-trained model: {e}")
            return False
    
    def add_skip_connections(self):
        """Add skip connections to the model to better preserve image details"""
        print("Adding skip connections to the model...")
        
        # Get the encoder and decoder parts
        encoder_layers = []
        decoder_layers = []
        
        # Identify encoder and decoder layers
        for layer in self.model.layers:
            if 'conv2d' in layer.name and not 'transpose' in layer.name:
                encoder_layers.append(layer)
            elif 'conv2d_transpose' in layer.name:
                decoder_layers.append(layer)
        
        # Create a new model with skip connections
        inputs = self.model.input
        x = inputs
        
        # Encoder path with skip connections
        skip_connections = []
        for layer in encoder_layers:
            x = layer(x)
            if 'conv2d' in layer.name and not 'transpose' in layer.name:
                skip_connections.append(x)
        
        # Decoder path with skip connections
        for i, layer in enumerate(decoder_layers):
            x = layer(x)
            if i < len(skip_connections) and x.shape[1:3] == skip_connections[-(i+1)].shape[1:3]:
                # Add skip connection
                x = layers.Add()([x, skip_connections[-(i+1)]])
        
        # Output layer
        outputs = self.model.output
        
        # Create new model
        self.model = Model(inputs, outputs, name="fine_tuned_denoising_autoencoder")
        
        # Compile with Adam optimizer and MSE loss
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='mse',
            metrics=['mae']
        )
        
        print("Skip connections added successfully")
        return self.model
    
    def add_attention_mechanism(self):
        """Add attention mechanism to focus on important features"""
        print("Adding attention mechanism to the model...")
        
        # Create a new model with attention
        inputs = self.model.input
        x = inputs
        
        # Add attention layers
        attention = layers.Conv2D(1, kernel_size=1, padding='same')(x)
        attention = layers.Activation('sigmoid')(attention)
        x = layers.Multiply()([x, attention])
        
        # Continue with the rest of the model
        for layer in self.model.layers[1:]:
            x = layer(x)
        
        # Create new model
        self.model = Model(inputs, x, name="fine_tuned_denoising_autoencoder_with_attention")
        
        # Compile with Adam optimizer and MSE loss
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='mse',
            metrics=['mae']
        )
        
        print("Attention mechanism added successfully")
        return self.model
    
    def enhance_model(self):
        """Add enhancements to the model without altering its basic structure"""
        print("Enhancing the model architecture...")
        
        # We'll create a new model that wraps the existing one and adds attention
        # This approach avoids the layer incompatibility issues
        
        inputs = layers.Input(shape=(self.img_size, self.img_size, 3))
        
        # Add attention mechanism before passing to the original model
        attention = layers.Conv2D(filters=3, kernel_size=1, padding='same')(inputs)
        attention = layers.Activation('sigmoid')(attention)
        enhanced_input = layers.Multiply()([inputs, attention])
        
        # Get output from original model
        outputs = self.model(enhanced_input)
        
        # Create new enhanced model
        enhanced_model = Model(inputs, outputs, name="enhanced_denoising_autoencoder")
        
        # Initialize metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_mae = tf.keras.metrics.MeanAbsoluteError(name='train_mae')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_mae = tf.keras.metrics.MeanAbsoluteError(name='val_mae')
        
        # Create optimizer
        self.optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
        
        # Initialize the model by running a forward pass
        sample_input = tf.random.uniform((1, self.img_size, self.img_size, 3))
        _ = enhanced_model(sample_input)
        
        self.model = enhanced_model
        print("Model enhanced successfully")
        return self.model
    
    def train(self, train_dataset, val_dataset=None, epochs=EPOCHS):
        """Train the model with early stopping and learning rate reduction"""
        if self.model is None:
            raise ValueError("Model not loaded or built")
        
        # Initialize metrics if not already initialized
        if not hasattr(self, 'train_loss'):
            self.train_loss = tf.keras.metrics.Mean(name='train_loss')
            self.train_mae = tf.keras.metrics.MeanAbsoluteError(name='train_mae')
            self.val_loss = tf.keras.metrics.Mean(name='val_loss')
            self.val_mae = tf.keras.metrics.MeanAbsoluteError(name='val_mae')
            self.optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
        
        # Training step function
        @tf.function
        def train_step(x, y):
            with tf.GradientTape() as tape:
                predictions = self.model(x, training=True)
                loss = tf.reduce_mean(tf.keras.losses.mse(y, predictions))
            
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            
            self.train_loss(loss)
            self.train_mae(y, predictions)
            return loss
        
        # Validation step function
        @tf.function
        def val_step(x, y):
            predictions = self.model(x, training=False)
            loss = tf.reduce_mean(tf.keras.losses.mse(y, predictions))
            
            self.val_loss(loss)
            self.val_mae(y, predictions)
            return loss
        
        # Early stopping setup
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        
        # Training history
        history = {
            'loss': [],
            'mae': [],
            'val_loss': [],
            'val_mae': []
        }
        
        # Training loop
        for epoch in range(epochs):
            # Reset metrics at the start of every epoch
            self.train_loss.reset_state()
            self.train_mae.reset_state()
            self.val_loss.reset_state()
            self.val_mae.reset_state()
            
            # Training
            print(f"\nEpoch {epoch + 1}/{epochs}")
            progress_bar = tqdm(train_dataset, desc="Training")
            for x, y in progress_bar:
                loss = train_step(x, y)
                progress_bar.set_postfix({
                    'loss': f'{self.train_loss.result():.4f}',
                    'mae': f'{self.train_mae.result():.4f}'
                })
            
            # Validation
            if val_dataset is not None:
                for x, y in val_dataset:
                    val_step(x, y)
            
            # Print epoch results
            template = 'Epoch {}, Loss: {:.4f}, MAE: {:.4f}'
            if val_dataset is not None:
                template += ', Val Loss: {:.4f}, Val MAE: {:.4f}'
                print(template.format(epoch + 1,
                                    self.train_loss.result(),
                                    self.train_mae.result(),
                                    self.val_loss.result(),
                                    self.val_mae.result()))
            else:
                print(template.format(epoch + 1,
                                    self.train_loss.result(),
                                    self.train_mae.result()))
            
            # Update history
            history['loss'].append(float(self.train_loss.result()))
            history['mae'].append(float(self.train_mae.result()))
            if val_dataset is not None:
                history['val_loss'].append(float(self.val_loss.result()))
                history['val_mae'].append(float(self.val_mae.result()))
                
                # Early stopping check
                current_val_loss = float(self.val_loss.result())
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    patience_counter = 0
                    best_weights = self.model.get_weights()
                    # Save best model
                    self.model.save(str(MODELS_DIR / "fine_tuned_denoising_autoencoder_best.keras"))
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                        # Restore best weights
                        self.model.set_weights(best_weights)
                        break
        
        # Save final model
        self.model.save(str(MODELS_DIR / "fine_tuned_denoising_autoencoder_final.keras"))
        
        return history
    
    def generate_samples(self, test_images, noise_level=0.2):
        """Generate sample denoising results"""
        os.makedirs(SAMPLES_DIR, exist_ok=True)
        
        for i, clean_img in enumerate(test_images[:5]):  # Use up to 5 test images
            # Add noise
            noisy_img = clean_img + np.random.normal(0, noise_level, clean_img.shape)
            noisy_img = np.clip(noisy_img, -1.0, 1.0)
            
            # Predict
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
    print("Starting fine-tuning of image denoising model...")
    start_time = time.time()
    
    # Create data loader and prepare dataset
    data_loader = MemoryEfficientDataLoader(
        img_size=IMG_SIZE,
        noise_level_min=NOISE_LEVEL_MIN,
        noise_level_max=NOISE_LEVEL_MAX,
        patch_size=IMG_SIZE
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
    
    # Load pre-trained model
    fine_tuned_model = FineTunedModel(img_size=IMG_SIZE)
    if not fine_tuned_model.load_pretrained_model(MODEL_PATH):
        print("Failed to load pre-trained model. Exiting.")
        return
    
    # Enhance the model - use the new approach to avoid layer incompatibility
    fine_tuned_model.enhance_model()
    
    # Print model summary
    fine_tuned_model.model.summary()
    
    # Train the model
    history = fine_tuned_model.train(train_dataset, val_dataset, epochs=EPOCHS)
    
    # Generate sample results
    fine_tuned_model.generate_samples(val_patches[:5])
    
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
    plt.savefig(str(BASE_DIR / "fine_tuning_history.png"))
    
    # Print training time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Fine-tuning completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Model saved to {MODELS_DIR}")
    print(f"Sample results saved to {SAMPLES_DIR}")


if __name__ == "__main__":
    main() 