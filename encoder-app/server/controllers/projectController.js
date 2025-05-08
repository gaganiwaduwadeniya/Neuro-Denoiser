const path = require('path');
const fs = require('fs');
const Project = require('../models/Project');
const modelHelper = require('../utils/modelHelper');

// @desc    Create new project
// @route   POST /api/projects
// @access  Private
exports.createProject = async (req, res) => {
  try {
    // Check if original image exists in request
    if (!req.body.originalImage) {
      return res.status(400).json({
        success: false,
        message: 'Please upload an original image'
      });
    }

    // Create new project
    const project = await Project.create({
      name: req.body.name,
      description: req.body.description || '',
      user: req.user.id,
      originalImage: req.body.originalImage,
      tags: req.body.tags || []
    });

    res.status(201).json({
      success: true,
      data: project
    });
  } catch (err) {
    console.error('Error creating project:', err);
    res.status(500).json({
      success: false,
      message: 'Failed to create project',
      error: err.message
    });
  }
};

// @desc    Get all projects for user
// @route   GET /api/projects
// @access  Private
exports.getProjects = async (req, res) => {
  try {
    // Get query parameters
    const page = parseInt(req.query.page, 10) || 1;
    const limit = parseInt(req.query.limit, 10) || 10;
    const startIndex = (page - 1) * limit;
    const endIndex = page * limit;
    const status = req.query.status || 'active';

    // Build query
    const query = {
      user: req.user.id,
      status
    };

    // Add tag filtering if provided
    if (req.query.tag) {
      query.tags = req.query.tag;
    }

    // Count total documents
    const total = await Project.countDocuments(query);

    // Execute query with pagination
    const projects = await Project.find(query)
      .sort({ createdAt: -1 })
      .skip(startIndex)
      .limit(limit);

    // Pagination result
    const pagination = {};

    if (endIndex < total) {
      pagination.next = {
        page: page + 1,
        limit
      };
    }

    if (startIndex > 0) {
      pagination.prev = {
        page: page - 1,
        limit
      };
    }

    res.status(200).json({
      success: true,
      count: projects.length,
      pagination,
      total,
      data: projects
    });
  } catch (err) {
    console.error('Error getting projects:', err);
    res.status(500).json({
      success: false,
      message: 'Failed to get projects',
      error: err.message
    });
  }
};

// @desc    Get single project
// @route   GET /api/projects/:id
// @access  Private
exports.getProject = async (req, res) => {
  try {
    const project = await Project.findById(req.params.id);

    if (!project) {
      return res.status(404).json({
        success: false,
        message: 'Project not found'
      });
    }

    // Check if user owns the project
    if (project.user.toString() !== req.user.id && req.user.role !== 'admin') {
      return res.status(401).json({
        success: false,
        message: 'Not authorized to access this project'
      });
    }

    res.status(200).json({
      success: true,
      data: project
    });
  } catch (err) {
    console.error('Error getting project:', err);
    res.status(500).json({
      success: false,
      message: 'Failed to get project',
      error: err.message
    });
  }
};

// @desc    Update project
// @route   PUT /api/projects/:id
// @access  Private
exports.updateProject = async (req, res) => {
  try {
    let project = await Project.findById(req.params.id);

    if (!project) {
      return res.status(404).json({
        success: false,
        message: 'Project not found'
      });
    }

    // Check if user owns the project
    if (project.user.toString() !== req.user.id && req.user.role !== 'admin') {
      return res.status(401).json({
        success: false,
        message: 'Not authorized to update this project'
      });
    }

    // Filter fields that can be updated
    const updateData = {};
    if (req.body.name) updateData.name = req.body.name;
    if (req.body.description) updateData.description = req.body.description;
    if (req.body.tags) updateData.tags = req.body.tags;
    if (req.body.status) updateData.status = req.body.status;

    // Update project
    project = await Project.findByIdAndUpdate(
      req.params.id,
      updateData,
      { new: true, runValidators: true }
    );

    res.status(200).json({
      success: true,
      data: project
    });
  } catch (err) {
    console.error('Error updating project:', err);
    res.status(500).json({
      success: false,
      message: 'Failed to update project',
      error: err.message
    });
  }
};

// @desc    Delete project
// @route   DELETE /api/projects/:id
// @access  Private
exports.deleteProject = async (req, res) => {
  try {
    const project = await Project.findById(req.params.id);

    if (!project) {
      return res.status(404).json({
        success: false,
        message: 'Project not found'
      });
    }

    // Check if user owns the project
    if (project.user.toString() !== req.user.id && req.user.role !== 'admin') {
      return res.status(401).json({
        success: false,
        message: 'Not authorized to delete this project'
      });
    }

    // Set project status to deleted instead of actually deleting
    await Project.findByIdAndUpdate(req.params.id, { status: 'deleted' });

    res.status(200).json({
      success: true,
      data: {}
    });
  } catch (err) {
    console.error('Error deleting project:', err);
    res.status(500).json({
      success: false,
      message: 'Failed to delete project',
      error: err.message
    });
  }
};

// @desc    Add processed image to project
// @route   POST /api/projects/:id/images
// @access  Private
exports.addProcessedImage = async (req, res) => {
  try {
    const project = await Project.findById(req.params.id);

    if (!project) {
      return res.status(404).json({
        success: false,
        message: 'Project not found'
      });
    }

    // Check if user owns the project
    if (project.user.toString() !== req.user.id && req.user.role !== 'admin') {
      return res.status(401).json({
        success: false,
        message: 'Not authorized to update this project'
      });
    }

    // Validate request body
    const { path: imagePath, type, settings, metrics } = req.body;
    if (!imagePath || !type || !settings) {
      return res.status(400).json({
        success: false,
        message: 'Please provide image path, type, and settings'
      });
    }

    // Add processed image to project
    const newImage = {
      type,
      path: imagePath,
      settings,
      metrics: metrics || {},
      createdAt: Date.now()
    };

    project.processedImages.push(newImage);
    await project.save();

    res.status(201).json({
      success: true,
      data: newImage
    });
  } catch (err) {
    console.error('Error adding processed image:', err);
    res.status(500).json({
      success: false,
      message: 'Failed to add processed image',
      error: err.message
    });
  }
}; 