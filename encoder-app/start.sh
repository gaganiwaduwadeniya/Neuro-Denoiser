#!/bin/bash

echo "==============================================="
echo "UltraHD Encoder - All-in-One Startup Script"
echo "==============================================="
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH."
    echo "Please install Python 3 and try again."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed or not in PATH."
    echo "Please install Node.js and try again."
    exit 1
fi

# Install required Python packages
echo "Installing Python dependencies..."
pip3 install tensorflow numpy pillow scikit-image python-shell

# Install NPM dependencies
echo
echo "Installing application dependencies..."
npm install

# Install server dependencies
cd server
npm install
cd ..

# Install client dependencies 
cd client
npm install
cd ..

echo
echo "==============================================="
echo "All dependencies installed!"
echo "Starting the application..."
echo "==============================================="
echo

# Make the script executable
chmod +x start.sh

# Start the application
npm start

echo
echo "Application stopped." 