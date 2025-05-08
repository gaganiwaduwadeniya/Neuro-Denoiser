const express = require('express');
const {
  createProject,
  getProjects,
  getProject,
  updateProject,
  deleteProject,
  addProcessedImage
} = require('../controllers/projectController');
const { protect } = require('../middleware/auth');

const router = express.Router();

// All project routes are protected
router.use(protect);

// Base routes
router.route('/')
  .get(getProjects)
  .post(createProject);

// Individual project routes
router.route('/:id')
  .get(getProject)
  .put(updateProject)
  .delete(deleteProject);

// Add processed image to project
router.post('/:id/images', addProcessedImage);

module.exports = router; 