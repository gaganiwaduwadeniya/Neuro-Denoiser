import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { FaTimes, FaHome, FaImage, FaProjectDiagram, FaTools } from 'react-icons/fa';
import { Badge } from 'react-bootstrap';
import './Sidebar.css';

const Sidebar = ({ isOpen, toggleSidebar }) => {
  const location = useLocation();
  
  // Simple check if a route is active
  const isActive = (path) => location.pathname === path;
  
  // Model status - in a real app this would come from context/API
  const modelStatus = {
    status: 'ready', // 'ready', 'loading', 'error'
    inputShape: [384, 384, 3],
    hasGPU: true
  };

  // Handle close button click
  const handleClose = () => {
    if (toggleSidebar) {
      toggleSidebar();
    }
  };
  
  return (
    <div className={`sidebar ${isOpen ? 'active' : ''}`}>
      <div className="sidebar-header">
        <h5>UltraHD Encoder</h5>
        <button 
          className="close-button" 
          onClick={handleClose}
          aria-label="Close sidebar"
        >
          <FaTimes />
        </button>
      </div>
      
      <div className="sidebar-content">
        <div className="model-status">
          <h6>Model Status</h6>
          <div className="status-card">
            <div className="status-indicator">
              <span 
                className={`status-dot ${
                  modelStatus.status === 'ready' ? 'ready' : 
                  modelStatus.status === 'loading' ? 'loading' : 'error'
                }`}
              ></span>
              <span className="status-text">{modelStatus.status}</span>
              {modelStatus.status === 'ready' && 
                <Badge bg="success" className="status-badge">Ready</Badge>
              }
            </div>
            <div className="status-details">
              <div>Input shape: {modelStatus.inputShape.join(' × ')}</div>
              <div>GPU: {modelStatus.hasGPU ? 'Available ✓' : 'Not available ✗'}</div>
            </div>
          </div>
        </div>
        
        <nav className="sidebar-nav">
          <div className="nav-heading">Main</div>
          <ul className="nav-list">
            <li className={`nav-item ${isActive('/') ? 'active' : ''}`}>
              <Link to="/" className="nav-link">
                <FaHome className="nav-icon" /> Dashboard
              </Link>
            </li>
            <li className={`nav-item ${isActive('/encoder') ? 'active' : ''}`}>
              <Link to="/encoder" className="nav-link">
                <FaImage className="nav-icon" /> Encoder
              </Link>
            </li>
            <li className={`nav-item ${isActive('/encoder-tool') ? 'active' : ''}`}>
              <Link to="/encoder-tool" className="nav-link">
                <FaTools className="nav-icon" /> Advanced Tools
              </Link>
            </li>
            <li className={`nav-item ${location.pathname.includes('/projects') ? 'active' : ''}`}>
              <Link to="/projects" className="nav-link">
                <FaProjectDiagram className="nav-icon" /> My Projects
              </Link>
            </li>
          </ul>
        </nav>
      </div>
      
      <div className="sidebar-footer">
        <div className="version-info">
          <p>UltraHD Encoder v1.0.0</p>
          <p>Fine_Tuned_Model_5</p>
        </div>
      </div>
    </div>
  );
};

export default Sidebar; 