import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { toast } from 'react-toastify';
import { FaCalendarAlt, FaDownload, FaSearch, FaEye, FaTimes, FaSpinner } from 'react-icons/fa';
import './GalleryView.css';

const GalleryView = () => {
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedImage, setSelectedImage] = useState(null);
  const [compareView, setCompareView] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [filteredImages, setFilteredImages] = useState([]);
  
  useEffect(() => {
    const fetchImages = async () => {
      try {
        const response = await axios.get('/api/images/all');
        if (response.data.success) {
          setImages(response.data.data);
          setFilteredImages(response.data.data);
        }
      } catch (error) {
        toast.error('Failed to fetch images');
        console.error('Error fetching images:', error);
      } finally {
        setLoading(false);
      }
    };
    
    fetchImages();
  }, []);
  
  // Filter images based on search term
  useEffect(() => {
    if (searchTerm.trim() === '') {
      setFilteredImages(images);
    } else {
      const filtered = images.filter(image => 
        image.originalImage.filename.toLowerCase().includes(searchTerm.toLowerCase())
      );
      setFilteredImages(filtered);
    }
  }, [searchTerm, images]);
  
  // Handle image selection for viewing
  const handleImageSelect = (image) => {
    setSelectedImage(image);
  };
  
  // Close the image viewer
  const closeViewer = () => {
    setSelectedImage(null);
    setCompareView(false);
  };
  
  // Toggle comparison view
  const toggleCompareView = () => {
    setCompareView(!compareView);
  };
  
  // Format date
  const formatDate = (dateString) => {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
  };
  
  // Handle image download
  const handleDownload = (url, filename) => {
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
  
  return (
    <div className="gallery-page">
      <div className="page-header slide-up">
        <h1>Image Gallery</h1>
        <p className="subtitle">Your denoised images collection</p>
      </div>
      
      <div className="gallery-controls">
        <div className="search-container">
          <FaSearch className="search-icon" />
          <input 
            type="text" 
            className="search-input" 
            placeholder="Search images..." 
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
        </div>
        
        <div className="filter-options">
          <span className="image-count">{filteredImages.length} images</span>
        </div>
      </div>
      
      {loading ? (
        <div className="loading-container">
          <FaSpinner className="spinner" />
          <p>Loading your images...</p>
        </div>
      ) : filteredImages.length === 0 ? (
        <div className="empty-gallery">
          <div className="empty-icon"></div>
          <h3>No Images Found</h3>
          <p>You haven't processed any images yet or no results match your search.</p>
        </div>
      ) : (
        <div className="gallery-grid">
          {filteredImages.map(image => (
            <div 
              key={image._id} 
              className="gallery-item card"
              onClick={() => handleImageSelect(image)}
            >
              <div className="image-preview">
                <img src={`/api/images/${image.denoisedImage.filename}`} alt="Denoised" />
                <div className="image-overlay">
                  <div className="overlay-content">
                    <FaEye />
                    <span>View</span>
                  </div>
                </div>
              </div>
              
              <div className="image-info">
                <h3>{image.originalImage.filename.split('-')[1]}</h3>
                <div className="image-meta">
                  <div className="meta-item">
                    <FaCalendarAlt className="meta-icon" />
                    <span>{formatDate(image.createdAt)}</span>
                  </div>
                </div>
                <div className="image-tags">
                  <span className="tag">{image.modelUsed}</span>
                  <span className="tag">{image.denoisedImage.quality}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
      
      {/* Image Viewer */}
      {selectedImage && (
        <div className="image-viewer">
          <div className="viewer-backdrop" onClick={closeViewer}></div>
          
          <div className="viewer-content">
            <button className="close-viewer" onClick={closeViewer}>
              <FaTimes />
            </button>
            
            <div className="viewer-header">
              <h2>{selectedImage.originalImage.filename.split('-')[1]}</h2>
              <div className="viewer-controls">
                <button 
                  className={`btn btn-sm ${compareView ? 'btn-secondary' : ''}`} 
                  onClick={toggleCompareView}
                >
                  {compareView ? 'Single View' : 'Compare View'}
                </button>
              </div>
            </div>
            
            <div className={`viewer-images ${compareView ? 'compare-mode' : ''}`}>
              {compareView && (
                <div className="original-image">
                  <h3>Original</h3>
                  <div className="image-container">
                    <img src={`/api/images/${selectedImage.originalImage.filename}`} alt="Original" />
                  </div>
                  <button 
                    className="btn btn-sm download-btn" 
                    onClick={() => handleDownload(`/api/images/${selectedImage.originalImage.filename}`, 'original-' + selectedImage.originalImage.filename)}
                  >
                    <FaDownload /> Download Original
                  </button>
                </div>
              )}
              
              <div className="denoised-image">
                <h3>Denoised</h3>
                <div className="image-container">
                  <img src={`/api/images/${selectedImage.denoisedImage.filename}`} alt="Denoised" />
                </div>
                <button 
                  className="btn btn-sm download-btn" 
                  onClick={() => handleDownload(`/api/images/${selectedImage.denoisedImage.filename}`, 'denoised-' + selectedImage.originalImage.filename)}
                >
                  <FaDownload /> Download Denoised
                </button>
              </div>
            </div>
            
            <div className="viewer-details">
              <div className="detail-column">
                <h3>Image Details</h3>
                <div className="detail-row">
                  <span className="detail-label">Processed On</span>
                  <span className="detail-value">{formatDate(selectedImage.createdAt)}</span>
                </div>
                <div className="detail-row">
                  <span className="detail-label">Model Used</span>
                  <span className="detail-value">{selectedImage.modelUsed === 'standard' ? 'Standard Denoiser' : 'Advanced Denoiser'}</span>
                </div>
              </div>
              
              <div className="detail-column">
                <h3>Processing Details</h3>
                <div className="detail-row">
                  <span className="detail-label">Quality Level</span>
                  <span className="detail-value">{selectedImage.denoisedImage.quality}</span>
                </div>
                <div className="detail-row">
                  <span className="detail-label">Processing Time</span>
                  <span className="detail-value">{selectedImage.denoisedImage.processingTime.toFixed(1)} seconds</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default GalleryView; 