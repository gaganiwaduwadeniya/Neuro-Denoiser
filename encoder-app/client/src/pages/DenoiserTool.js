import React, { useState, useEffect, useRef } from 'react';
import { toast } from 'react-toastify';
import axios from 'axios';
import { FaUpload, FaRandom, FaEye, FaDownload, FaMagic, FaSpinner } from 'react-icons/fa';
import './DenoiserTool.css';

const DenoiserTool = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [resultUrl, setResultUrl] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('standard');
  const [processingDetails, setProcessingDetails] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  
  const fileInputRef = useRef(null);
  const dropZoneRef = useRef(null);
  
  // Fetch available models on component mount
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await axios.get('/api/models');
        if (response.data.success) {
          setModels(response.data.models);
        }
      } catch (error) {
        toast.error('Failed to load denoising models');
        console.error('Error fetching models:', error);
      }
    };
    
    fetchModels();
    
    // Start the progress animation when processing
    if (isProcessing) {
      const interval = setInterval(() => {
        setProgress(prev => {
          const newProgress = prev + (100 - prev) * 0.05;
          return newProgress >= 99 ? 99 : newProgress;
        });
      }, 200);
      
      return () => clearInterval(interval);
    } else {
      setProgress(0);
    }
    
  }, [isProcessing]);
  
  // Handle file selection
  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    if (!file.type.match('image.*')) {
      toast.error('Please select an image file');
      return;
    }
    
    // Reset previous state
    setResultUrl('');
    setProcessingDetails(null);
    setSelectedFile(file);
    
    // Create preview URL
    const reader = new FileReader();
    reader.onload = () => setPreviewUrl(reader.result);
    reader.readAsDataURL(file);
  };
  
  // Trigger file input click
  const handleUploadClick = () => {
    fileInputRef.current.click();
  };
  
  // Handle drag events
  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };
  
  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.currentTarget === dropZoneRef.current) {
      setIsDragging(false);
    }
  };
  
  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };
  
  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.type.match('image.*')) {
        setSelectedFile(file);
        
        const reader = new FileReader();
        reader.onload = () => setPreviewUrl(reader.result);
        reader.readAsDataURL(file);
        
        // Reset previous results
        setResultUrl('');
        setProcessingDetails(null);
      } else {
        toast.error('Please select an image file');
      }
    }
  };
  
  // Handle form submission and image processing
  const handleProcess = async () => {
    if (!selectedFile) {
      toast.error('Please select an image file first');
      return;
    }
    
    setIsProcessing(true);
    setProgress(0);
    
    try {
      const formData = new FormData();
      formData.append('image', selectedFile);
      formData.append('modelType', selectedModel);
      
      // Call the API
      const response = await axios.post('/api/denoise', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      
      if (response.data.success) {
        // Set the result URL
        setResultUrl(response.data.denoisedImage);
        
        // Set processing details
        setProcessingDetails({
          processingTime: response.data.processingTime,
          qualityLevel: response.data.qualityLevel,
          imageId: response.data.imageId
        });
        
        toast.success('Image denoised successfully!');
      } else {
        toast.error(response.data.message || 'Failed to process image');
      }
    } catch (error) {
      toast.error('Error during image processing');
      console.error('Processing error:', error);
    } finally {
      setIsProcessing(false);
      setProgress(100);
    }
  };
  
  // Handle download of processed image
  const handleDownload = () => {
    if (!resultUrl) return;
    
    // Create a temporary link and click it to download the image
    const link = document.createElement('a');
    link.href = resultUrl;
    link.download = 'denoised-image.jpg';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
  
  return (
    <div className="denoiser-page">
      <div className="page-header slide-up">
        <h1>Neural Image Denoiser</h1>
        <p className="subtitle">Enhance your noisy images with AI-power</p>
      </div>
      
      <div className="denoiser-container">
        <div className="tools-panel card fade-in">
          <div className="panel-header">
            <h3>Denoising Models</h3>
          </div>
          
          <div className="models-list">
            {models.map(model => (
              <div 
                key={model.id}
                className={`model-option ${selectedModel === model.id ? 'selected' : ''}`}
                onClick={() => setSelectedModel(model.id)}
              >
                <div className="model-info">
                  <h4>{model.name}</h4>
                  <p>{model.description}</p>
                  
                  <div className="model-badges">
                    <span className="badge speed-badge">
                      Speed: {model.processingSpeed}
                    </span>
                    <span className="badge quality-badge">
                      Quality: {model.qualityLevel}
                    </span>
                  </div>
                </div>
                
                <div className="model-selector">
                  <div className="selector-dot"></div>
                </div>
              </div>
            ))}
          </div>
          
          <div className="panel-divider"></div>
          
          <div className="panel-actions">
            <button 
              className={`btn btn-lg btn-block btn-primary ${isProcessing ? 'processing' : ''}`}
              onClick={handleProcess}
              disabled={!selectedFile || isProcessing}
            >
              {isProcessing ? (
                <>
                  <FaSpinner className="spinner" />
                  Denoising Image...
                </>
              ) : (
                <>
                  <FaMagic />
                  Denoise Image
                </>
              )}
            </button>
            
            {isProcessing && (
              <div className="progress-container">
                <div className="progress-bar" style={{ width: `${progress}%` }}></div>
                <span className="progress-text">{Math.round(progress)}%</span>
              </div>
            )}
          </div>
        </div>
        
        <div className="image-workspace">
          <div 
            className={`upload-container card ${isDragging ? 'dragging' : ''} ${previewUrl ? 'has-image' : ''}`}
            onClick={handleUploadClick}
            onDragEnter={handleDragEnter}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            ref={dropZoneRef}
          >
            {previewUrl ? (
              <div className="image-preview">
                <img src={previewUrl} alt="Original" className="preview-img" />
                <div className="image-overlay">
                  <div className="overlay-content">
                    <FaEye />
                    <span>View Original</span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="upload-placeholder">
                <div className="upload-icon">
                  <FaUpload />
                </div>
                <h3>Upload Noisy Image</h3>
                <p>Drag & drop an image here or click to browse</p>
                <p className="file-types">Supports JPG, PNG, WEBP</p>
              </div>
            )}
            
            <input 
              type="file"
              ref={fileInputRef}
              onChange={handleFileSelect}
              accept="image/*"
              className="file-input"
            />
          </div>
          
          <div className={`result-container card ${resultUrl ? 'has-result' : ''}`}>
            {resultUrl ? (
              <div className="result-content">
                <div className="result-image-container">
                  <img src={resultUrl} alt="Denoised" className="result-img" />
                  
                  <button className="btn btn-sm download-btn" onClick={handleDownload}>
                    <FaDownload />
                    Download
                  </button>
                </div>
                
                {processingDetails && (
                  <div className="result-details">
                    <h3>Denoising Results</h3>
                    <div className="details-grid">
                      <div className="detail-item">
                        <span className="detail-label">Processing Time</span>
                        <span className="detail-value">{processingDetails.processingTime}</span>
                      </div>
                      <div className="detail-item">
                        <span className="detail-label">Quality Level</span>
                        <span className="detail-value">{processingDetails.qualityLevel}</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="empty-result">
                <div className="placeholder-icon">
                  <FaRandom />
                </div>
                <h3>Result will appear here</h3>
                <p>Select a model and denoise your image to see the results</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default DenoiserTool; 