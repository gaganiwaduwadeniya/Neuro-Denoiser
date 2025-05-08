const path = require('path');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');
const multer = require('multer');
const sharp = require('sharp');
const modelHelper = require('../utils/modelHelper');

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadPath = path.join(process.env.UPLOAD_PATH || './uploads', 'original');
    
    // Create directory if it doesn't exist
    fs.mkdirSync(uploadPath, { recursive: true });
    cb(null, uploadPath);
  },
  filename: (req, file, cb) => {
    // Generate unique filename
    const uniqueSuffix = `${Date.now()}-${Math.round(Math.random() * 1E9)}`;
    const ext = path.extname(file.originalname);
    cb(null, `${uniqueSuffix}${ext}`);
  }
});

// Configure upload middleware
const upload = multer({
  storage,
  limits: {
    fileSize: 10 * 1024 * 1024 // 10MB limit
  },
  fileFilter: (req, file, cb) => {
    // Accept only images
    if (!file.mimetype.startsWith('image/')) {
      return cb(new Error('Only image files are allowed'), false);
    }
    cb(null, true);
  }
}).single('image');

// @desc    Get model information
// @route   GET /api/encoder/model-info
// @access  Public
exports.getModelInfo = async (req, res) => {
  try {
    const modelInfo = await modelHelper.getModelInfo();
    res.status(200).json({
      success: true,
      data: modelInfo
    });
  } catch (err) {
    console.error('Error getting model info:', err);
    res.status(500).json({
      success: false,
      message: 'Failed to get model information',
      error: err.message
    });
  }
};

// @desc    Process image with the encoder model
// @route   POST /api/encoder/process
// @access  Private
exports.processImage = async (req, res) => {
  try {
    // Handle file upload with multer
    upload(req, req.file, async (err) => {
      if (err) {
        return res.status(400).json({
          success: false,
          message: err.message
        });
      }

      if (!req.file) {
        return res.status(400).json({
          success: false,
          message: 'Please upload an image'
        });
      }

      // Get processing options from request body
      const options = {
        noise_level: parseFloat(req.body.noise_level || 0.5),
        noise_type: req.body.noise_type || 'gaussian',
        save_noisy: req.body.save_noisy === 'true' || req.body.save_noisy === true
      };

      // Paths for processed images
      const inputPath = req.file.path;
      const processedDir = path.join(process.env.UPLOAD_PATH || './uploads', 'processed');
      
      // Create processed directory if it doesn't exist
      fs.mkdirSync(processedDir, { recursive: true });
      
      const outputFilename = `processed-${path.basename(inputPath)}`;
      const outputPath = path.join(processedDir, outputFilename);

      // Process image with model
      const result = await modelHelper.processImage(inputPath, outputPath, options);

      if (!result.success) {
        return res.status(500).json({
          success: false,
          message: 'Image processing failed',
          error: result.error
        });
      }

      // Get file URLs for response
      const baseUrl = `${req.protocol}://${req.get('host')}`;
      const originalUrl = `${baseUrl}/uploads/original/${path.basename(inputPath)}`;
      const processedUrl = `${baseUrl}/uploads/processed/${outputFilename}`;
      
      // Add noisy image URL if it was saved
      let noisyUrl = null;
      if (options.save_noisy && result.noisy_path) {
        const noisyFilename = path.basename(result.noisy_path);
        noisyUrl = `${baseUrl}/uploads/processed/${noisyFilename}`;
      }

      // Return response
      res.status(200).json({
        success: true,
        data: {
          originalUrl,
          processedUrl,
          noisyUrl,
          metrics: result.metrics
        }
      });
    });
  } catch (err) {
    console.error('Error processing image:', err);
    res.status(500).json({
      success: false,
      message: 'Failed to process image',
      error: err.message
    });
  }
};

// @desc    Calculate metrics between two images
// @route   POST /api/encoder/metrics
// @access  Private
exports.calculateMetrics = async (req, res) => {
  try {
    // Validate request
    const { originalPath, processedPath } = req.body;
    if (!originalPath || !processedPath) {
      return res.status(400).json({
        success: false,
        message: 'Please provide both original and processed image paths'
      });
    }

    // Calculate metrics
    const result = await modelHelper.calculateMetrics(originalPath, processedPath);

    if (!result.success) {
      return res.status(500).json({
        success: false,
        message: 'Failed to calculate metrics',
        error: result.error
      });
    }

    // Return response
    res.status(200).json({
      success: true,
      data: {
        metrics: result.metrics
      }
    });
  } catch (err) {
    console.error('Error calculating metrics:', err);
    res.status(500).json({
      success: false,
      message: 'Failed to calculate metrics',
      error: err.message
    });
  }
}; 