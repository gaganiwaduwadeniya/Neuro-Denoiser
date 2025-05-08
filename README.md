# NeuroDenoiser: AI-Powered Image Denoising Application

A modern web application for denoising images using advanced neural networks.

## Features

- Modern, futuristic UI with a dark theme and neon accents
- Real-time image denoising using neural network models
- Multiple denoising models with different quality/speed tradeoffs
- Drag-and-drop image upload capability
- Image gallery to view and compare processed images
- MongoDB integration for storing processing history

## Prerequisites

- Node.js (v14 or later)
- Python 3.8+ with TensorFlow 2.x
- MongoDB (local or Atlas cloud instance)

## Getting Started

1. Clone the repository:
   ```
   git clone <repository-url>
   cd NeuroDenoiser
   ```

2. Install dependencies:
   ```
   npm run install-deps
   ```

3. Create a `.env` file in the `encoder-app/server` directory with the following content:
   ```
   PORT=5000
   MONGO_URI=mongodb://localhost:27017/neurodenoiser
   ```

4. Install Python dependencies:
   ```
   pip install tensorflow pillow scikit-image numpy
   ```

5. Start the application:
   ```
   npm start
   ```

This will start both the Node.js server (on port 5000) and the React client (on port 3000).

## Project Structure

- `/encoder-app/client`: React frontend application
- `/encoder-app/server`: Node.js backend server
  - `/config`: Database connection configuration
  - `/models`: MongoDB schema definitions
  - `/python`: Python scripts for image processing
  - `/uploads`: Directory for uploaded and processed images
- `/Python`: Neural network model files
  - `/Fine_Tuned_Model_5`: Contains the trained models for denoising

## Using the Application

1. Open your browser and navigate to `http://localhost:3000`
2. The dashboard displays application statistics and features
3. Navigate to the Denoise Tool to upload and process images
4. View your processed images in the Gallery view
5. Download your processed images directly from the application

## Troubleshooting

- If you see "Proxy error" messages in the client console, ensure the server is running on port 5000.
- If models aren't loading, check that the path to the Python models is correct and that TensorFlow is properly installed.
- For MongoDB connection issues, verify your connection string in the `.env` file.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 