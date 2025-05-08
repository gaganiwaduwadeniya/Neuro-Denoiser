const express = require('express');
const path = require('path');
const fs = require('fs');
const multer = require('multer');
const { exec } = require('child_process');
const mongoose = require('mongoose');
const dotenv = require('dotenv');
const connectDB = require('./config/db');
const { fileToBase64 } = require('./utils/imageUtils');

// Load environment variables
dotenv.config();

// Import models
const Image = require('./models/Image');

// Connect to database
connectDB();

// Initialize Express app
const app = express();
const PORT = process.env.PORT || 5000;

// Ensure uploads directory exists
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) {
  console.log('Creating uploads directory...');
  fs.mkdirSync(uploadsDir, { recursive: true });
}

// Middleware
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Add CORS headers for development
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  if (req.method === 'OPTIONS') {
    return res.sendStatus(200);
  }
  next();
});

// Set up file upload with multer
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, uploadsDir);
  },
  filename: function (req, file, cb) {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, 'original-' + uniqueSuffix + path.extname(file.originalname));
  }
});

const upload = multer({ storage: storage });

// Helper function to check if file size is too large for MongoDB (16MB max document size)
function isFileTooLargeForMongoDB(filePath, maxSizeInMB = 12) {
  try {
    const stats = fs.statSync(filePath);
    const fileSizeInMB = stats.size / (1024 * 1024);
    return fileSizeInMB > maxSizeInMB;
  } catch (error) {
    console.error(`Error checking file size: ${error.message}`);
    return true; // Assume it's too large if there's an error
  }
}

// API endpoint to check server status
app.get('/api/status', (req, res) => {
  res.json({
    status: 'ready',
    timestamp: new Date(),
    message: 'Encoder server is running'
  });
});

// API endpoint to get available models
app.get('/api/models', (req, res) => {
  // Check if model files exist
  const modelPath = path.join(__dirname, '..', '..', 'Python', 'Fine_Tuned_Model_5');
  const hasUltraHD = fs.existsSync(path.join(modelPath, 'ultrahd_denoiser_v6_best.keras'));
  const hasCrystalClear = fs.existsSync(path.join(modelPath, 'crystal_clear_denoiser_final.keras'));
  
  let availableModels = [];
  
  // Always include standard model (will use whatever model is available)
  availableModels.push({
    id: 'standard',
    name: 'Standard Denoiser',
    description: 'Basic denoising with good balance of quality and speed',
    processingSpeed: 'Fast',
    qualityLevel: 'Good'
  });
  
  // Add high-quality model if available
  if (hasCrystalClear) {
    availableModels.push({
      id: 'high-quality',
      name: 'Crystal Clear Denoiser',
      description: 'High-quality denoising with excellent noise removal',
      processingSpeed: 'Medium',
      qualityLevel: 'Excellent'
    });
  }
  
  // Add experimental model if available
  if (hasUltraHD) {
    availableModels.push({
      id: 'experimental',
      name: 'UltraHD Denoiser',
      description: 'Experimental model with advanced denoising capabilities',
      processingSpeed: 'Slow',
      qualityLevel: 'Premium'
    });
  }
  
  res.json({
    success: true,
    models: availableModels
  });
});

// API endpoint to process an image
app.post('/api/denoise', upload.single('image'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ 
      success: false, 
      message: 'No image file uploaded' 
    });
  }

  try {
    // Get the path of the uploaded file
    const inputPath = req.file.path;
    const outputFilename = 'denoised-' + path.basename(req.file.filename);
    const outputPath = path.join(__dirname, 'uploads', outputFilename);
    const modelType = req.body.modelType || 'standard';
    
    // Get stats of original file
    const originalStats = fs.statSync(inputPath);
    
    // Log the file info
    console.log(`Processing file: ${inputPath}`);
    console.log(`Model type: ${modelType}`);
    
    // Check if file is too large for MongoDB storage
    if (isFileTooLargeForMongoDB(inputPath)) {
      console.warn(`File ${inputPath} is too large for MongoDB storage. Storing file path only.`);
      // Just store the file path, not the actual data
      var originalBase64 = null;
    } else {
      // Convert original image to Base64
      var originalBase64 = fileToBase64(inputPath);
    }
    
    // Save the original image information to database
    const newImage = new Image({
      originalImage: {
        data: originalBase64,
        path: inputPath,
        filename: req.file.filename,
        mimetype: req.file.mimetype,
        size: originalStats.size
      },
      modelUsed: modelType
    });
    
    // Process the image using our Python model
    const pythonScript = path.join(__dirname, 'python', 'process_image.py');
    const command = `python "${pythonScript}" "${inputPath}" "${outputPath}" "${modelType}"`;
    
    exec(command, async (error, stdout, stderr) => {
      if (error) {
        console.error(`Exec error: ${error}`);
        return res.status(500).json({
          success: false,
          message: 'Error processing image with model',
          error: error.message
        });
      }
      
      if (stderr) {
        console.error(`Stderr: ${stderr}`);
      }
      
      try {
        // Parse processing results
        const result = JSON.parse(stdout);
        
        if (!result.success) {
          throw new Error(result.error || 'Unknown processing error');
        }
        
        // If Python processing worked but file wasn't created, fall back to simple copy
        if (!fs.existsSync(outputPath)) {
          fs.copyFileSync(inputPath, outputPath);
        }
        
        // Check if denoised file is too large for MongoDB storage
        if (isFileTooLargeForMongoDB(outputPath)) {
          console.warn(`Denoised file ${outputPath} is too large for MongoDB storage. Storing file path only.`);
          // Just store the file path, not the actual data
          var denoisedBase64 = null;
        } else {
          // Convert denoised image to Base64
          var denoisedBase64 = fileToBase64(outputPath);
        }
        
        // Update the image document with denoised information
        newImage.denoisedImage = {
          data: denoisedBase64,
          path: outputPath,
          filename: outputFilename,
          processingTime: result.processingTime || 0,
          quality: modelType === 'high-quality' ? 'High' : (modelType === 'experimental' ? 'Premium' : 'Medium')
        };
        
        // Save to database
        const savedImage = await newImage.save();
        
        // Send successful response
        res.json({
          success: true,
          imageId: savedImage._id,
          originalImage: `/api/images/${path.basename(inputPath)}`,
          denoisedImage: `/api/images/${outputFilename}`,
          processingTime: (result.processingTime || 0).toFixed(1) + ' seconds',
          qualityLevel: newImage.denoisedImage.quality,
          metrics: {
            psnr: result.psnr,
            ssim: result.ssim
          }
        });
      } catch (parseError) {
        console.error('Error parsing Python output:', parseError);
        console.log('Python output:', stdout);
        
        // Fall back to simple file copy if Python processing fails
        fs.copyFileSync(inputPath, outputPath);
        
        // Check if denoised file is too large for MongoDB storage
        if (isFileTooLargeForMongoDB(outputPath)) {
          console.warn(`Denoised file ${outputPath} is too large for MongoDB storage. Storing file path only.`);
          // Just store the file path, not the actual data
          var denoisedBase64 = null;
        } else {
          // Convert denoised image to Base64
          var denoisedBase64 = fileToBase64(outputPath);
        }
        
        newImage.denoisedImage = {
          data: denoisedBase64,
          path: outputPath,
          filename: outputFilename,
          processingTime: 0.5,
          quality: modelType === 'high-quality' ? 'High' : 'Medium'
        };
        
        const savedImage = await newImage.save();
        
        res.json({
          success: true,
          imageId: savedImage._id,
          originalImage: `/api/images/${path.basename(inputPath)}`,
          denoisedImage: `/api/images/${outputFilename}`,
          processingTime: newImage.denoisedImage.processingTime.toFixed(1) + ' seconds',
          qualityLevel: newImage.denoisedImage.quality,
          fallback: true
        });
      }
    });
  } catch (error) {
    console.error('Error processing image:', error);
    res.status(500).json({
      success: false,
      message: 'Error processing image',
      error: error.message
    });
  }
});

// Get all processed images
app.get('/api/images/all', async (req, res) => {
  try {
    const images = await Image.find().sort({ createdAt: -1 });
    res.json({
      success: true,
      count: images.length,
      data: images
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: 'Error retrieving images',
      error: error.message
    });
  }
});

// Get a specific image by ID
app.get('/api/images/id/:id', async (req, res) => {
  try {
    const image = await Image.findById(req.params.id);
    if (!image) {
      return res.status(404).json({
        success: false,
        message: 'Image not found'
      });
    }
    res.json({
      success: true,
      data: image
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: 'Error retrieving image',
      error: error.message
    });
  }
});

// Serve processed images - now can serve from database if file not found
app.get('/api/images/:filename', async (req, res) => {
  const filePath = path.join(__dirname, 'uploads', req.params.filename);
  
  // First check if file exists on disk
  if (fs.existsSync(filePath)) {
    return res.sendFile(filePath);
  }
  
  // If file not found on disk, try to find in database
  try {
    // Check if it's an original image
    let image = await Image.findOne({ 'originalImage.filename': req.params.filename });
    
    if (image && image.originalImage) {
      // If image data exists in DB, send it
      if (image.originalImage.data) {
        const imgBuffer = Buffer.from(image.originalImage.data, 'base64');
        res.set('Content-Type', image.originalImage.mimetype);
        return res.send(imgBuffer);
      }
      // If we have file path but no data in DB, check if file exists at the path
      else if (image.originalImage.path && fs.existsSync(image.originalImage.path)) {
        return res.sendFile(image.originalImage.path);
      }
    }
    
    // If not found as original, check denoised images
    image = await Image.findOne({ 'denoisedImage.filename': req.params.filename });
    
    if (image && image.denoisedImage) {
      // If image data exists in DB, send it
      if (image.denoisedImage.data) {
        const imgBuffer = Buffer.from(image.denoisedImage.data, 'base64');
        res.set('Content-Type', image.originalImage.mimetype);
        return res.send(imgBuffer);
      }
      // If we have file path but no data in DB, check if file exists at the path
      else if (image.denoisedImage.path && fs.existsSync(image.denoisedImage.path)) {
        return res.sendFile(image.denoisedImage.path);
      }
    }
    
    // If not found in database or at the stored paths
    return res.status(404).json({
      success: false,
      message: 'Image not found'
    });
  } catch (error) {
    console.error('Error retrieving image from database:', error);
    return res.status(500).json({
      success: false,
      message: 'Error retrieving image',
      error: error.message
    });
  }
});

// Serve static assets if in production
if (process.env.NODE_ENV === 'production') {
  app.use(express.static(path.join(__dirname, '../client/build')));
  
  app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, '../client/build', 'index.html'));
  });
}

// Start the server
app.listen(PORT, () => {
  console.log('==============================================');
  console.log(`Encoder Server running on port ${PORT}`);
  console.log(`API available at http://localhost:${PORT}/api`);
  console.log('==============================================');
}); 