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
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

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

# Optimized configuration for higher PSNR (>30dB) and SSIM
BATCH_SIZE = 2  # Small batch size for memory efficiency
IMG_SIZE = 256  # Reduced from 384 to improve memory usage
NOISE_LEVEL_MIN = 0.01  # Lower minimum noise for better clean image reconstruction
NOISE_LEVEL_MAX = 0.4   # Still allow higher noise for robustness
EPOCHS = 100            # Sufficient epochs for convergence
INITIAL_LEARNING_RATE = 1e-5  # Lower learning rate for fine-tuning
MIN_LEARNING_RATE = 1e-7      # Minimum learning rate floor
PATIENCE = 15           # Patience for early stopping
MEMORY_LIMIT = 0.9      # Limit memory usage to 90% of available
BASE_DIR = Path("UltraDenoiser_v6")

# Dataset params - optimized for memory
MAX_TOTAL_PATCHES = 2000  # Limit total patches to manage memory
TRAIN_VALIDATION_SPLIT = 0.9  # 90% training, 10% validation

# Create necessary directories
os.makedirs(BASE_DIR, exist_ok=True)
MODELS_DIR = BASE_DIR / "models"
SAMPLES_DIR = BASE_DIR / "samples"
LOGS_DIR = BASE_DIR / "logs"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Generate synthetic training data if needed
GENERATE_SYNTHETIC_DATA = True  # Set to True to enable synthetic data generation
SYNTHETIC_DATA_SIZE = 500  # Number of synthetic images to generate if no real data is found

# Find existing model to fine-tune
def find_existing_model():
    """Find the most recent model to fine-tune"""
    model_paths = [
        "Fine_Tuned_Model_5/crystal_clear_denoiser_final.keras",
        "Fine_Tuned_Model_5/ultrahd_denoiser_best.keras",
        "Python/Fine_Tuned_Model_5/crystal_clear_denoiser_final.keras",
        "Python/Fine_Tuned_Model_5/ultrahd_denoiser_best.keras",
        # Add .h5 versions in case model was saved in that format
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
    for version in range(5, 0, -1):
        path = f"Fine_Tuned_Model_{version}/crystal_clear_denoiser_final.keras"
        if os.path.exists(path):
            print(f"Found earlier model version at: {path}")
            return path
    
    print("No existing model found. Will create a new model from scratch.")
    return None

class MemoryEfficientDataLoader:
    """Memory-optimized data loader for denoising training"""
    
    def __init__(self, img_size=IMG_SIZE, noise_min=NOISE_LEVEL_MIN, noise_max=NOISE_LEVEL_MAX):
        self.img_size = img_size
        self.noise_min = noise_min
        self.noise_max = noise_max
    
    def load_images(self, limit=MAX_TOTAL_PATCHES):
        """Load and prepare images with memory efficiency in mind"""
        print("Loading images with memory optimization...")
        
        # Collect potential image paths - scan common directories
        image_paths = []
        
        # Look in various directories for existing datasets
        data_dirs = [
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
            print(f"Scanning directories: {existing_dirs}")
            
            for data_dir in existing_dirs:
                for root, _, files in os.walk(data_dir):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                            image_paths.append(os.path.join(root, file))
            
            # Shuffle paths to ensure diverse dataset
            random.shuffle(image_paths)
            
            # Limit to prevent memory issues
            image_paths = image_paths[:min(limit*2, len(image_paths))]
            
            print(f"Found {len(image_paths)} potential images")
        else:
            print("Warning: No data directories found")
        
        # If no images found or synthetic data generation is enabled
        if len(image_paths) == 0 and GENERATE_SYNTHETIC_DATA:
            print("No real images found. Generating synthetic data...")
            return self.generate_synthetic_data(num_images=SYNTHETIC_DATA_SIZE)
        
        # Process images and extract patches
        patches = []
        skipped = 0
        
        for img_path in tqdm(image_paths):
            if len(patches) >= limit:
                break
                
            try:
                # Load and process image
                img = cv2.imread(img_path)
                if img is None:
                    skipped += 1
                    continue
                    
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Skip if image is too small
                if img.shape[0] < self.img_size or img.shape[1] < self.img_size:
                    skipped += 1
                    continue
                
                # Extract a random patch
                h, w = img.shape[:2]
                top = random.randint(0, h - self.img_size)
                left = random.randint(0, w - self.img_size)
                
                patch = img[top:top + self.img_size, left:left + self.img_size]
                
                # Convert to float and normalize to [-1, 1]
                patch = patch.astype(np.float32) / 127.5 - 1.0
                
                patches.append(patch)
                
                # Memory optimization: periodic garbage collection
                if len(patches) % 100 == 0:
                    gc.collect()
                
            except Exception as e:
                skipped += 1
                print(f"Error processing {img_path}: {e}")
                continue
        
        print(f"Extracted {len(patches)} patches, skipped {skipped} images")
        
        # If no patches were extracted from real images, generate synthetic data
        if len(patches) == 0 and GENERATE_SYNTHETIC_DATA:
            print("Could not extract any usable patches. Generating synthetic data...")
            return self.generate_synthetic_data(num_images=SYNTHETIC_DATA_SIZE)
        
        # Convert to numpy array
        patches = np.array(patches)
        
        return patches
    
    def generate_synthetic_data(self, num_images=500):
        """Generate synthetic images for training when no real data is available"""
        print(f"Generating {num_images} synthetic training images...")
        
        synthetic_patches = []
        
        for i in tqdm(range(num_images)):
            # Generate synthetic clean image types
            image_type = random.choice(['gradient', 'pattern', 'shape'])
            
            if image_type == 'gradient':
                # Random gradient image
                direction = random.choice(['horizontal', 'vertical', 'diagonal'])
                color1 = np.array([random.random(), random.random(), random.random()], dtype=np.float32) * 2 - 1
                color2 = np.array([random.random(), random.random(), random.random()], dtype=np.float32) * 2 - 1
                
                x = np.linspace(-1, 1, self.img_size, dtype=np.float32)
                y = np.linspace(-1, 1, self.img_size, dtype=np.float32)
                xx, yy = np.meshgrid(x, y)
                
                if direction == 'horizontal':
                    gradient = xx
                elif direction == 'vertical':
                    gradient = yy
                else:  # diagonal
                    gradient = (xx + yy) / 2
                
                # Normalize gradient to [-1, 1]
                gradient = (gradient - np.min(gradient)) / (np.max(gradient) - np.min(gradient)) * 2 - 1
                
                # Create RGB image
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
                for c in range(3):
                    img[:, :, c] = color1[c] + (color2[c] - color1[c]) * (gradient + 1) / 2
                
            elif image_type == 'pattern':
                # Create pattern image
                pattern_type = random.choice(['checkerboard', 'stripes', 'circles'])
                
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
            
            else:  # shape
                # Random shapes
                img = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
                color = np.array([random.random(), random.random(), random.random()], dtype=np.float32) * 2 - 1
                
                # Fill with base color
                base_color = np.array([random.random(), random.random(), random.random()], dtype=np.float32) * 2 - 1
                img[:, :, :] = base_color
                
                # Add random shapes
                for _ in range(random.randint(3, 8)):
                    shape_type = random.choice(['rectangle', 'circle'])
                    color = np.array([random.random(), random.random(), random.random()], dtype=np.float32) * 2 - 1
                    
                    if shape_type == 'rectangle':
                        x1, y1 = random.randint(0, self.img_size-20), random.randint(0, self.img_size-20)
                        width = random.randint(20, self.img_size//3)
                        height = random.randint(20, self.img_size//3)
                        x2, y2 = min(x1 + width, self.img_size-1), min(y1 + height, self.img_size-1)
                        
                        img[y1:y2, x1:x2, :] = color
                        
                    elif shape_type == 'circle':
                        center_x = random.randint(20, self.img_size-20)
                        center_y = random.randint(20, self.img_size-20)
                        radius = random.randint(10, self.img_size//6)
                        
                        yy, xx = np.ogrid[:self.img_size, :self.img_size]
                        dist = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)
                        mask = dist <= radius
                        
                        for c in range(3):
                            img[:, :, c][mask] = color[c]
            
            # Adjust contrast/brightness randomly
            contrast = np.float32(random.uniform(0.7, 1.3))
            brightness = np.float32(random.uniform(-0.2, 0.2))
            img = img * contrast + brightness
            
            # Ensure values are within [-1, 1]
            img = np.clip(img, -1.0, 1.0)
            
            # Ensure the image is float32
            img = img.astype(np.float32)
            
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
        """Create optimized TensorFlow dataset for training"""
        # Safety check - ensure patches is not empty
        if len(patches) == 0:
            raise ValueError("No patches available for creating dataset. Check the data loading process.")
        
        # Ensure patches are float32 to avoid type mismatches
        if patches.dtype != np.float32:
            print(f"Converting patches from {patches.dtype} to float32")
            patches = patches.astype(np.float32)
        
        # Memory optimization: use tf.data pipeline for efficient data handling
        
        def add_noise(clean_img):
            """Add advanced noise patterns for better denoising performance"""
            # Ensure input is float32 to prevent type mismatches
            clean_img = tf.cast(clean_img, tf.float32)
            
            # Reshape if needed for adding noise correctly - prevent rank issues
            input_shape = tf.shape(clean_img)
            
            # Reshape the input to 4D if it's not already
            clean_img_4d = tf.reshape(clean_img, [1, input_shape[0], input_shape[1], input_shape[2]])
            
            # Randomly select noise level for this image
            noise_level = tf.random.uniform(
                shape=[], 
                minval=self.noise_min,
                maxval=self.noise_max,
                dtype=tf.float32
            )
            
            # Base noise - Gaussian with fine-tuned parameters
            noise = tf.random.normal(
                shape=tf.shape(clean_img_4d),
                mean=0.0,
                stddev=noise_level,
                dtype=tf.float32
            )
            
            # Add spatial correlation to noise for realism - ensure 4D input to avg_pool2d
            correlated_noise = tf.nn.avg_pool2d(
                noise,
                ksize=3,
                strides=1,
                padding='SAME'
            )
            
            # Reshape back to 3D if input was 3D
            noise = tf.reshape(noise, input_shape)
            correlated_noise = tf.reshape(correlated_noise, input_shape)
            
            # Mix original and correlated noise - ensure same data type
            mix_ratio = tf.random.uniform(shape=[], minval=0.6, maxval=1.0, dtype=tf.float32)
            final_noise = tf.cast(noise, tf.float32) * mix_ratio + tf.cast(correlated_noise, tf.float32) * (1 - mix_ratio)
            
            # Add noise to image - ensure consistent data types
            noisy_img = tf.cast(clean_img, tf.float32) + tf.cast(final_noise, tf.float32)
            
            # Add salt & pepper noise occasionally (10% chance)
            if tf.random.uniform(shape=[], dtype=tf.float32) < 0.1:
                # Salt (white pixels)
                salt_mask = tf.cast(
                    tf.random.uniform(tf.shape(clean_img), dtype=tf.float32) < (noise_level * 0.05),
                    tf.float32
                )
                noisy_img = noisy_img * (1 - salt_mask) + salt_mask
                
                # Pepper (black pixels)
                pepper_mask = tf.cast(
                    tf.random.uniform(tf.shape(clean_img), dtype=tf.float32) < (noise_level * 0.05),
                    tf.float32
                )
                noisy_img = noisy_img * (1 - pepper_mask) + (-1.0 * pepper_mask)
            
            # Add compression artifacts occasionally (15% chance)
            if tf.random.uniform(shape=[], dtype=tf.float32) < 0.15:
                # Use reshape to ensure noisy_img is 4D for pooling operation
                noisy_img_4d = tf.reshape(noisy_img, [1, input_shape[0], input_shape[1], input_shape[2]])
                
                # Simulate compression by selective blurring 
                blurred = tf.nn.avg_pool2d(
                    noisy_img_4d,
                    ksize=2,
                    strides=1,
                    padding='SAME'
                )
                
                # Reshape back to 3D
                blurred = tf.reshape(blurred, input_shape)
                
                # Mix original and blurred based on noise level
                compression_ratio = noise_level * 0.5
                noisy_img = noisy_img * (1 - compression_ratio) + blurred * compression_ratio
            
            # Ensure values stay in valid range
            noisy_img = tf.clip_by_value(noisy_img, -1.0, 1.0)
            clean_img = tf.clip_by_value(clean_img, -1.0, 1.0)
            
            return noisy_img, clean_img
        
        # Create dataset with memory-efficient pipeline
        dataset = tf.data.Dataset.from_tensor_slices(patches)
        
        # Apply noise augmentation with controlled parallelism 
        dataset = dataset.map(
            add_noise,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Shuffle only for training
        if is_training:
            buffer_size = min(1000, len(patches))
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
        """Build a high-performance denoising model for PSNR > 30dB"""
        print("Building enhanced denoising model...")
        
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
            x = layers.Conv2D(32, 3, padding='same')(inputs)
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
            
            # Block 3 with attention
            x = layers.MaxPooling2D(2)(x)
            x = self._residual_block(x, 256, kernel_size=3)
            x = self._attention_block(x)
            
            # === Second Stage: Upsample and Refine ===
            # Block 4 with skip connection from Block 2
            x = layers.UpSampling2D(2, interpolation='bilinear')(x)
            x = layers.Concatenate()([x, block2_output])
            x = self._residual_block(x, 128, kernel_size=3)
            
            # Block 5 with skip connection from Block 1
            x = layers.UpSampling2D(2, interpolation='bilinear')(x)
            x = layers.Concatenate()([x, block1_output])
            x = self._residual_block(x, 64, kernel_size=3)
            
            # Final feature integration
            x = layers.Conv2D(32, 3, padding='same')(x)
            x = layers.LeakyReLU(0.2)(x)
            
            # Final output with skip connection from input (critical for PSNR)
            x = layers.Conv2D(3, 3, padding='same')(x)
            outputs = layers.Add()([x, orig_connection])
            
            # Create model
            self.model = Model(inputs, outputs, name="ultrahd_denoiser_v6")
        
        # Configure the model for fine-tuning
        if fine_tune_base:
            # Unfreeze all layers for full fine-tuning
            for layer in self.model.layers:
                layer.trainable = True
        
        # Compile with optimized settings for PSNR/SSIM
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE),
            loss=self._psnr_focused_loss,
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
    
    def _psnr_focused_loss(self, y_true, y_pred):
        """Custom loss function optimized for PSNR>30dB"""
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
        
        # Weighted sum of MSE and gradient loss
        # MSE weight is higher to prioritize PSNR
        return 0.85 * mse + 0.15 * gradient_loss(y_true, y_pred)
    
    def train(self, train_dataset, val_dataset, epochs=EPOCHS):
        """Train the model with memory-optimized approach"""
        print(f"Training model for {epochs} epochs...")
        
        # Create callbacks
        callbacks_list = [
            # Model checkpoint to save best model
            callbacks.ModelCheckpoint(
                filepath=str(MODELS_DIR / "ultrahd_denoiser_v6_best.keras"),
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
        final_model_path = str(MODELS_DIR / "ultrahd_denoiser_v6_final.keras")
        try:
            self.model.save(final_model_path)
            print(f"Final model saved to {final_model_path}")
        except Exception as e:
            print(f"Error saving model: {e}. Trying alternative format...")
            # Try saving in h5 format if keras format fails
            h5_path = str(MODELS_DIR / "ultrahd_denoiser_v6_final.h5")
            self.model.save(h5_path)
            print(f"Final model saved to {h5_path}")
        
        # Save training history
        history_dict = history.history
        np.save(str(LOGS_DIR / "training_history.npy"), history_dict)
        
        # Plot training history
        self._plot_training_history(history)
        
        return history
    
    def evaluate(self, test_dataset, noise_levels=[0.1, 0.2, 0.3]):
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
                    noise = np.random.normal(0, 0.2, clean.shape)
                    noisy = np.clip(clean + noise, -1.0, 1.0)
                    samples.append((noisy, clean))
        
        # Evaluate on samples
        results = []
        
        for idx, (noisy, clean) in enumerate(samples):
            try:
                # Predict denoised image
                denoised = self.model.predict(tf.expand_dims(noisy, 0), verbose=0)[0].numpy()
                
                # Convert from [-1, 1] to [0, 1] for metrics calculation
                noisy_0_1 = (noisy + 1) / 2
                clean_0_1 = (clean + 1) / 2
                denoised_0_1 = (denoised + 1) / 2
                
                # Calculate PSNR
                psnr_noisy = peak_signal_noise_ratio(clean_0_1, noisy_0_1)
                psnr_denoised = peak_signal_noise_ratio(clean_0_1, denoised_0_1)
                
                # Calculate SSIM with multichannel parameter
                ssim_noisy = structural_similarity(
                    clean_0_1, noisy_0_1, 
                    channel_axis=2  # Updated parameter for newer skimage versions
                )
                ssim_denoised = structural_similarity(
                    clean_0_1, denoised_0_1, 
                    channel_axis=2  # Updated parameter for newer skimage versions
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
            if avg_psnr_denoised > 30:
                print("🎉 SUCCESS: Target PSNR > 30dB achieved!")
            else:
                print("⚠️ Target PSNR > 30dB not yet achieved.")
                
            if avg_ssim_denoised > 0.9:
                print("🎉 SUCCESS: Excellent SSIM > 0.9 achieved!")
            elif avg_ssim_denoised > 0.8:
                print("✓ GOOD: Good SSIM > 0.8 achieved.")
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
    print("Starting UltraHD Denoiser v6 fine-tuning pipeline...")
    start_time = time.time()
    
    # Memory optimization before starting
    gc.collect()
    
    # Try setting the default float type to float32
    try:
        # Set TensorFlow to use float32 by default
        tf.keras.backend.set_floatx('float32')
        print("Set TensorFlow default float type to float32")
    except Exception as e:
        print(f"Warning: Could not set TensorFlow default float type: {e}")
    
    # Create data loader
    data_loader = MemoryEfficientDataLoader(
        img_size=IMG_SIZE,
        noise_min=NOISE_LEVEL_MIN,
        noise_max=NOISE_LEVEL_MAX
    )
    
    try:
        # Load and prepare images
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
        
        # Verify data types before creating datasets
        print(f"Training data type: {train_patches.dtype}")
        if val_patches is not None:
            print(f"Validation data type: {val_patches.dtype}")
        
        try:
            # Create TensorFlow datasets with explicit handling
            print("Creating training dataset...")
            train_dataset = data_loader.create_tf_dataset(train_patches, batch_size=BATCH_SIZE, is_training=True)
            
            if val_patches is not None and len(val_patches) >= BATCH_SIZE:
                print("Creating validation dataset...")
                val_dataset = data_loader.create_tf_dataset(val_patches, batch_size=BATCH_SIZE, is_training=False)
            else:
                print("Skipping validation dataset creation (insufficient data)")
                val_dataset = None
            
            # Create and build model
            model = HybridResidualAttentionModel(img_size=IMG_SIZE)
            model.build_model(fine_tune_base=True)
            
            # Train model
            model.train(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                epochs=EPOCHS
            )
            
            # Evaluate model
            test_dataset = val_dataset if val_dataset is not None else train_dataset
            model.evaluate(test_dataset)
            
        except tf.errors.InvalidArgumentError as e:
            print(f"TensorFlow error: {e}")
            if "data type" in str(e).lower() or "dtype" in str(e).lower():
                print("\nERROR: Data type mismatch detected.")
                print("Attempting recovery with manual type conversion...")
                
                # Try fixing with manual type conversion
                train_patches = train_patches.astype(np.float32)
                if val_patches is not None:
                    val_patches = val_patches.astype(np.float32)
                
                # Create datasets after fixing types
                train_dataset = data_loader.create_tf_dataset(train_patches, batch_size=BATCH_SIZE, is_training=True)
                val_dataset = data_loader.create_tf_dataset(val_patches, batch_size=BATCH_SIZE, is_training=False) if val_patches is not None and len(val_patches) >= BATCH_SIZE else None
                
                # Try again with the model
                model = HybridResidualAttentionModel(img_size=IMG_SIZE)
                model.build_model(fine_tune_base=True)
                model.train(train_dataset=train_dataset, val_dataset=val_dataset, epochs=EPOCHS)
            else:
                raise e
        
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
        print(f"UltraHD Denoiser v6 model saved to {MODELS_DIR / 'ultrahd_denoiser_v6_final.keras'}")
        print("Fine-tuning completed successfully!")
    
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
                x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
                x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
                skip1 = x
                x = layers.MaxPooling2D(2)(x)
                
                x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
                x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
                skip2 = x
                x = layers.MaxPooling2D(2)(x)
                
                # Bottleneck
                x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
                x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
                
                # Decoder
                x = layers.UpSampling2D(2)(x)
                x = layers.Concatenate()([x, skip2])
                x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
                x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
                
                x = layers.UpSampling2D(2)(x)
                x = layers.Concatenate()([x, skip1])
                x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
                x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
                
                # Output
                outputs = layers.Conv2D(3, 3, padding='same')(x)
                
                model = Model(inputs, outputs, name="simple_denoiser_v6")
                model.compile(
                    optimizer=optimizers.Adam(learning_rate=1e-4),
                    loss='mse',
                    metrics=['mae']
                )
                return model
            
            # Generate synthetic data with explicit float32 type
            print("Generating synthetic data with explicit float32 type...")
            synthetic_data = data_loader.generate_synthetic_data(num_images=200)
            
            # Verify data type of synthetic data
            print(f"Synthetic data type: {synthetic_data.dtype}")
            if synthetic_data.dtype != np.float32:
                synthetic_data = synthetic_data.astype(np.float32)
                print(f"Converted synthetic data to {synthetic_data.dtype}")
            
            # Split into train/val
            split = int(len(synthetic_data) * 0.8)
            train_data = synthetic_data[:split]
            val_data = synthetic_data[split:]
            
            # Create datasets with explicit float32 type
            print("Creating simplified datasets...")
            
            # Create a simple pipeline without the complex noise function
            def simple_add_noise(image):
                noise = tf.random.normal(
                    shape=tf.shape(image), 
                    mean=0.0, 
                    stddev=0.1, 
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
                    filepath=str(MODELS_DIR / "simple_denoiser_v6_best.keras"),
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
            simple_model.save(str(MODELS_DIR / "simple_denoiser_v6_final.keras"))
            print(f"Simplified model saved to {MODELS_DIR / 'simple_denoiser_v6_final.keras'}")
            
        except Exception as recovery_error:
            print(f"Recovery failed: {recovery_error}")
            print("Please check the error messages above and fix the issues before trying again.")

if __name__ == "__main__":
    main() 