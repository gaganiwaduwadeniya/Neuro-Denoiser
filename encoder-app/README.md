# UltraHD Image Encoder Application

A powerful image encoder application that allows users to process images with advanced AI models.

## Features

- Upload and encode images with different quality settings
- Advanced image processing models with excellent quality
- Offline mode support when server is unavailable
- Image comparison and statistics
- Project management for saved images

## Getting Started

### Prerequisites

- Node.js (v14 or later)
- npm (v6 or later)

### Installation

1. Clone this repository
2. Install dependencies:
   ```
   npm install
   cd client
   npm install
   cd ../server
   npm install
   ```

### Running the Application

#### Option 1: Using the Fixed Startup Script (Recommended)

1. Double-click on `start-fixed.bat` in the main folder
2. This will start both the server and client in separate windows
3. Wait for the application to initialize (approximately 10-15 seconds)
4. Open your browser and navigate to `http://localhost:3000`

#### Option 2: Manual Start

1. Start the server:
   ```
   cd server
   npm start
   ```

2. In a separate terminal, start the client:
   ```
   cd client
   npm start
   ```

3. Open your browser and navigate to `http://localhost:3000`

## Troubleshooting

If you encounter connection issues between the client and server:

1. Make sure both the server and client are running
   - Server should be on port 5000
   - Client should be on port 3000

2. Check if the ports are available:
   - Try stopping any other applications that might be using these ports
   - You can check with `netstat -ano | findstr "5000"` and `netstat -ano | findstr "3000"`

3. If the client shows "Running in offline mode":
   - This is a fallback feature and the application will still work with limited functionality
   - Check that the server is running properly on port 5000

4. Try restarting both the server and client:
   - Close all terminal windows
   - Run the `start-fixed.bat` script again

## Application Structure

- `client/` - React frontend application
- `server/` - Express backend server
- `start-fixed.bat` - Windows startup script

## License

This project is licensed under the MIT License. 