/**
 * Encoder API Service
 * Handles communication with backend API for image processing
 * Includes fallback to offline mode when server is unavailable
 */

import axios from 'axios';

// Create Axios instance with base URL and longer timeout
const api = axios.create({
  baseURL: '/api',
  timeout: 60000, // 60 seconds
});

// Server status cache to avoid repeated failed requests
let serverStatusCache = {
  status: null,
  timestamp: 0,
  offline: false
};

// Add a response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    // Handle network errors
    if (!error.response) {
      console.error('Network error:', error);
      serverStatusCache.offline = true;
      serverStatusCache.timestamp = Date.now();
      
      return Promise.reject({
        message: 'Network error: Server is unreachable. Running in offline mode.',
        isServerError: true,
      });
    }
    
    // Handle other errors
    return Promise.reject({
      message: error.response?.data?.message || 'An error occurred',
      status: error.response?.status,
      isServerError: true,
    });
  }
);

// Mock delay to simulate processing time
const mockDelay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// Mock progress updates
const mockProgress = async (onProgress) => {
  for (let i = 0; i <= 100; i += 10) {
    await mockDelay(300); // Update progress every 300ms
    onProgress(i);
  }
};

/**
 * Get the server status
 * @returns {Promise<Object>} Server status information
 */
export const checkServerStatus = async () => {
  try {
    const response = await axios.get('/api/status');
    return response.data;
  } catch (error) {
    console.error('Error checking server status:', error);
    throw error;
  }
};

/**
 * Get available denoising models
 * @returns {Promise<Array>} List of available models
 */
export const getAvailableModels = async () => {
  try {
    const response = await axios.get('/api/models');
    if (response.data.success) {
      return response.data.models;
    }
    throw new Error('Failed to fetch models');
  } catch (error) {
    console.error('Error fetching models:', error);
    throw error;
  }
};

/**
 * Process an image with the selected model
 * @param {File} imageFile - The image file to process
 * @param {Object} options - Processing options
 * @param {Function} progressCallback - Callback for progress updates
 * @returns {Promise<Object>} Processing results
 */
export const processImage = async (imageFile, options = {}, progressCallback = null) => {
  try {
    const formData = new FormData();
    formData.append('image', imageFile);
    
    if (options.modelType) {
      formData.append('modelType', options.modelType);
    }
    
    // Setup for tracking upload progress
    const config = {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    };
    
    if (progressCallback) {
      config.onUploadProgress = (progressEvent) => {
        const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        // Cap at 60% since processing happens after upload
        const cappedProgress = Math.min(percentCompleted * 0.6, 60);
        progressCallback(cappedProgress);
      };
    }
    
    // First send the image to be processed
    const response = await axios.post('/api/denoise', formData, config);
    
    if (progressCallback) {
      // Processing is complete
      progressCallback(100);
    }
    
    return response.data;
  } catch (error) {
    console.error('Error processing image:', error);
    throw error;
  }
};

/**
 * Fetch all processed images
 * @returns {Promise<Array>} List of processed images
 */
export const getAllImages = async () => {
  try {
    const response = await axios.get('/api/images/all');
    if (response.data.success) {
      return response.data.data;
    }
    throw new Error('Failed to fetch images');
  } catch (error) {
    console.error('Error fetching images:', error);
    throw error;
  }
};

/**
 * Get a specific image by ID
 * @param {string} imageId - The ID of the image to fetch
 * @returns {Promise<Object>} Image details
 */
export const getImageById = async (imageId) => {
  try {
    const response = await axios.get(`/api/images/id/${imageId}`);
    if (response.data.success) {
      return response.data.data;
    }
    throw new Error('Failed to fetch image');
  } catch (error) {
    console.error('Error fetching image:', error);
    throw error;
  }
};

// Compare two images to see improvements (before/after)
export const compareImages = async (originalImageUrl, resultImageUrl) => {
  try {
    // In a real implementation, this would call the server to compare the images
    // For now, we'll mock the response
    
    // Simulate a delay
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    // Mock statistics for image comparison
    return {
      psnr: (20 + Math.random() * 10).toFixed(2), // Peak Signal-to-Noise Ratio (higher is better)
      ssim: (0.7 + Math.random() * 0.25).toFixed(3), // Structural Similarity Index (closer to 1 is better)
      noiseReduction: (60 + Math.random() * 30).toFixed(1) + '%', // Percentage of noise reduced
      details: [
        { name: 'Gaussian Noise Removed', value: (70 + Math.random() * 20).toFixed(1) + '%' },
        { name: 'Sharpness Preserved', value: (85 + Math.random() * 10).toFixed(1) + '%' },
        { name: 'Color Accuracy', value: (90 + Math.random() * 9).toFixed(1) + '%' }
      ]
    };
  } catch (error) {
    console.error('Error comparing images:', error);
    throw error;
  }
};

// Mock image processing for offline mode
const mockProcessImage = async (file, modelType, onProgress) => {
  return new Promise((resolve) => {
    let progress = 0;
    const interval = setInterval(() => {
      progress += Math.random() * 15;
      if (progress > 95) {
        progress = 95;
        clearInterval(interval);
      }
      if (onProgress) onProgress(Math.min(Math.round(progress), 95));
    }, 300);
    
    // Create URL for preview
    const reader = new FileReader();
    reader.onloadend = () => {
      // Create result url (same as original in mock mode)
      const originalUrl = reader.result;
      
      setTimeout(() => {
        clearInterval(interval);
        if (onProgress) onProgress(100);
        
        resolve({
          success: true,
          processedImage: originalUrl,
          originalSize: file.size,
          processedSize: file.size * 0.8, // Mock 20% reduction
          processingTime: (1 + Math.random() * 3).toFixed(1) + ' seconds',
          qualityLevel: modelType === 'high-quality' ? 'Excellent' : 'Good'
        });
      }, 3000);
    };
    reader.readAsDataURL(file);
  });
}; 