const fs = require('fs');

/**
 * Convert a file to Base64 string
 * @param {string} filePath - Path to the file
 * @returns {string} Base64 encoded string
 */
function fileToBase64(filePath) {
  try {
    const fileData = fs.readFileSync(filePath);
    return Buffer.from(fileData).toString('base64');
  } catch (error) {
    console.error(`Error converting file to Base64: ${error.message}`);
    throw error;
  }
}

/**
 * Convert Base64 string to a file
 * @param {string} base64Data - Base64 encoded string
 * @param {string} outputPath - Path to save the file
 * @returns {boolean} Success status
 */
function base64ToFile(base64Data, outputPath) {
  try {
    const buffer = Buffer.from(base64Data, 'base64');
    fs.writeFileSync(outputPath, buffer);
    return true;
  } catch (error) {
    console.error(`Error converting Base64 to file: ${error.message}`);
    return false;
  }
}

/**
 * Get MIME type from Base64 data
 * @param {string} base64Data - Base64 encoded string with MIME type
 * @returns {string} MIME type
 */
function getMimeTypeFromBase64(base64Data) {
  try {
    const match = base64Data.match(/^data:([A-Za-z-+\/]+);base64,/);
    if (match && match[1]) {
      return match[1];
    }
    return 'application/octet-stream'; // Default MIME type
  } catch (error) {
    console.error(`Error extracting MIME type: ${error.message}`);
    return 'application/octet-stream';
  }
}

module.exports = {
  fileToBase64,
  base64ToFile,
  getMimeTypeFromBase64
}; 