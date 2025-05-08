const express = require('express');
const { 
  getModelInfo, 
  processImage, 
  calculateMetrics 
} = require('../controllers/encoderController');
const { protect } = require('../middleware/auth');

const router = express.Router();

// Public routes
router.get('/model-info', getModelInfo);

// Protected routes
router.post('/process', protect, processImage);
router.post('/metrics', protect, calculateMetrics);

module.exports = router; 