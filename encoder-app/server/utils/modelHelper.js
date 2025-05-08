const path = require('path');
const fs = require('fs');
const { PythonShell } = require('python-shell');

// Path to the Python script that will handle the model operations
const pythonScriptPath = path.join(__dirname, '..', 'python', 'model_handler.py');

// Initialize the Python environment and model
const initializeModel = async () => {
  return new Promise((resolve, reject) => {
    const options = {
      mode: 'text',
      pythonPath: 'python', // or path to python executable
      pythonOptions: ['-u'], // get print results in real-time
      scriptPath: path.dirname(pythonScriptPath),
      args: ['initialize', process.env.MODEL_PATH]
    };

    PythonShell.run(path.basename(pythonScriptPath), options, (err, results) => {
      if (err) {
        console.error('Error initializing model:', err);
        return reject(err);
      }
      
      const result = results[results.length - 1];
      try {
        const parsedResult = JSON.parse(result);
        return resolve(parsedResult);
      } catch (parseError) {
        console.error('Error parsing Python output:', parseError);
        return reject(parseError);
      }
    });
  });
};

// Process an image with the model
const processImage = async (inputPath, outputPath, options = {}) => {
  return new Promise((resolve, reject) => {
    // Ensure the input file exists
    if (!fs.existsSync(inputPath)) {
      return reject(new Error(`Input file not found: ${inputPath}`));
    }

    // Convert options object to JSON string
    const optionsStr = JSON.stringify(options);

    const pythonOptions = {
      mode: 'text',
      pythonPath: 'python', // or path to python executable
      pythonOptions: ['-u'], // get print results in real-time
      scriptPath: path.dirname(pythonScriptPath),
      args: ['process', inputPath, outputPath, optionsStr]
    };

    PythonShell.run(path.basename(pythonScriptPath), pythonOptions, (err, results) => {
      if (err) {
        console.error('Error processing image:', err);
        return reject(err);
      }
      
      const result = results[results.length - 1];
      try {
        const parsedResult = JSON.parse(result);
        return resolve(parsedResult);
      } catch (parseError) {
        console.error('Error parsing Python output:', parseError);
        return reject(parseError);
      }
    });
  });
};

// Get model information
const getModelInfo = async () => {
  return new Promise((resolve, reject) => {
    const options = {
      mode: 'text',
      pythonPath: 'python', // or path to python executable
      pythonOptions: ['-u'], // get print results in real-time
      scriptPath: path.dirname(pythonScriptPath),
      args: ['info', process.env.MODEL_PATH]
    };

    PythonShell.run(path.basename(pythonScriptPath), options, (err, results) => {
      if (err) {
        console.error('Error getting model info:', err);
        return reject(err);
      }
      
      const result = results[results.length - 1];
      try {
        const parsedResult = JSON.parse(result);
        return resolve(parsedResult);
      } catch (parseError) {
        console.error('Error parsing Python output:', parseError);
        return reject(parseError);
      }
    });
  });
};

// Calculate metrics between two images
const calculateMetrics = async (originalPath, processedPath) => {
  return new Promise((resolve, reject) => {
    // Ensure both files exist
    if (!fs.existsSync(originalPath)) {
      return reject(new Error(`Original file not found: ${originalPath}`));
    }
    if (!fs.existsSync(processedPath)) {
      return reject(new Error(`Processed file not found: ${processedPath}`));
    }

    const options = {
      mode: 'text',
      pythonPath: 'python', // or path to python executable
      pythonOptions: ['-u'], // get print results in real-time
      scriptPath: path.dirname(pythonScriptPath),
      args: ['metrics', originalPath, processedPath]
    };

    PythonShell.run(path.basename(pythonScriptPath), options, (err, results) => {
      if (err) {
        console.error('Error calculating metrics:', err);
        return reject(err);
      }
      
      const result = results[results.length - 1];
      try {
        const parsedResult = JSON.parse(result);
        return resolve(parsedResult);
      } catch (parseError) {
        console.error('Error parsing Python output:', parseError);
        return reject(parseError);
      }
    });
  });
};

module.exports = {
  initializeModel,
  processImage,
  getModelInfo,
  calculateMetrics
}; 