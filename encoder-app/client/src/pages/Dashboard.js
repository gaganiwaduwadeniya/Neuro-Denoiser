import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';
import { toast } from 'react-toastify';
import { FaImage, FaBrain, FaArrowRight, FaServer, FaClock } from 'react-icons/fa';
import './Dashboard.css';

const Dashboard = () => {
  const [stats, setStats] = useState({
    totalImages: 0,
    lastProcessed: null,
    serverStatus: 'checking'
  });
  
  useEffect(() => {
    const fetchStats = async () => {
      try {
        // Check server status
        const statusRes = await axios.get('/api/status');
        
        // Get processed images
        const imagesRes = await axios.get('/api/images/all');
        
        if (imagesRes.data.success) {
          setStats({
            totalImages: imagesRes.data.count || 0,
            lastProcessed: imagesRes.data.data[0] || null,
            serverStatus: statusRes.data.status || 'unknown'
          });
        }
      } catch (error) {
        console.error('Error fetching dashboard data:', error);
        setStats(prev => ({
          ...prev,
          serverStatus: 'error'
        }));
      }
    };
    
    fetchStats();
    
    // Set up a timer to animate the counter
    const counters = document.querySelectorAll('.stat-value');
    counters.forEach(counter => {
      const target = parseInt(counter.getAttribute('data-target'));
      const count = () => {
        const speed = 50;
        const inc = target / speed;
        const current = parseInt(counter.innerText);
        if (current < target) {
          counter.innerText = Math.ceil(current + inc);
          setTimeout(count, 20);
        } else {
          counter.innerText = target;
        }
      };
      count();
    });
  }, []);
  
  return (
    <div className="dashboard-page">
      <div className="dashboard-hero">
        <div className="hero-content slide-up">
          <h1>NeuroDenoiser</h1>
          <p className="hero-subtitle">
            Advanced AI-powered Image Denoising
          </p>
          
          <Link to="/denoise" className="btn btn-lg btn-primary hero-cta">
            <FaImage />
            Start Denoising
            <FaArrowRight className="arrow-icon" />
          </Link>
        </div>
        
        <div className="hero-graphic">
          <div className="neural-network">
            <div className="node input-node n1"></div>
            <div className="node input-node n2"></div>
            <div className="node input-node n3"></div>
            
            <div className="node hidden-node h1"></div>
            <div className="node hidden-node h2"></div>
            <div className="node hidden-node h3"></div>
            <div className="node hidden-node h4"></div>
            
            <div className="node output-node o1"></div>
            <div className="node output-node o2"></div>
            
            <svg className="connections" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
              {/* Input to Hidden connections */}
              <path className="connection" d="M 80,100 L 200,70" />
              <path className="connection" d="M 80,100 L 200,130" />
              <path className="connection" d="M 80,100 L 200,190" />
              <path className="connection" d="M 80,100 L 200,250" />
              
              <path className="connection" d="M 80,160 L 200,70" />
              <path className="connection" d="M 80,160 L 200,130" />
              <path className="connection" d="M 80,160 L 200,190" />
              <path className="connection" d="M 80,160 L 200,250" />
              
              <path className="connection" d="M 80,220 L 200,70" />
              <path className="connection" d="M 80,220 L 200,130" />
              <path className="connection" d="M 80,220 L 200,190" />
              <path className="connection" d="M 80,220 L 200,250" />
              
              {/* Hidden to Output connections */}
              <path className="connection" d="M 200,70 L 320,130" />
              <path className="connection" d="M 200,70 L 320,190" />
              
              <path className="connection" d="M 200,130 L 320,130" />
              <path className="connection" d="M 200,130 L 320,190" />
              
              <path className="connection" d="M 200,190 L 320,130" />
              <path className="connection" d="M 200,190 L 320,190" />
              
              <path className="connection" d="M 200,250 L 320,130" />
              <path className="connection" d="M 200,250 L 320,190" />
            </svg>
          </div>
        </div>
      </div>
      
      <div className="dashboard-stats">
        <div className="stat-card card fade-in">
          <div className="stat-icon">
            <FaImage />
          </div>
          <div className="stat-info">
            <h3>Processed Images</h3>
            <div className="stat-value" data-target={stats.totalImages}>0</div>
            <p className="stat-description">Total denoised images</p>
          </div>
        </div>
        
        <div className="stat-card card fade-in">
          <div className="stat-icon">
            <FaBrain />
          </div>
          <div className="stat-info">
            <h3>Neural Models</h3>
            <div className="stat-value" data-target="2">0</div>
            <p className="stat-description">Available denoising models</p>
          </div>
        </div>
        
        <div className="stat-card card fade-in">
          <div className={`stat-icon ${stats.serverStatus === 'ready' ? 'active' : 'inactive'}`}>
            <FaServer />
          </div>
          <div className="stat-info">
            <h3>Server Status</h3>
            <div className={`server-status ${stats.serverStatus}`}>
              {stats.serverStatus === 'ready' ? 'Online' : 
               stats.serverStatus === 'error' ? 'Offline' : 'Checking...'}
            </div>
            <p className="stat-description">Neural processing server</p>
          </div>
        </div>
        
        <div className="stat-card card fade-in">
          <div className="stat-icon">
            <FaClock />
          </div>
          <div className="stat-info">
            <h3>Latest Denoising</h3>
            <div className="last-processed">
              {stats.lastProcessed ? 
                new Date(stats.lastProcessed.createdAt).toLocaleTimeString() : 
                'No images processed yet'}
            </div>
            <p className="stat-description">Most recent processed image</p>
          </div>
        </div>
      </div>
      
      <div className="dashboard-features">
        <h2 className="section-title">Enhance Your Images</h2>
        
        <div className="features-grid">
          <div className="feature-card card">
            <div className="feature-icon noise-reduction"></div>
            <h3>Advanced Noise Reduction</h3>
            <p>Our AI model effectively removes noise while preserving important image details and textures.</p>
          </div>
          
          <div className="feature-card card">
            <div className="feature-icon detail-preservation"></div>
            <h3>Detail Preservation</h3>
            <p>Edge-aware denoising keeps sharp details intact while smoothing out noisy areas.</p>
          </div>
          
          <div className="feature-card card">
            <div className="feature-icon speed"></div>
            <h3>Fast Processing</h3>
            <p>Optimized neural networks provide quick processing times even for high-resolution images.</p>
          </div>
          
          <div className="feature-card card">
            <div className="feature-icon gallery"></div>
            <h3>Image Gallery</h3>
            <p>Access all your processed images in one place with our convenient gallery view.</p>
          </div>
        </div>
      </div>
      
      <div className="dashboard-cta">
        <div className="cta-content">
          <h2>Ready to Denoise Your Images?</h2>
          <p>Upload your images and let our AI handle the rest.</p>
          <Link to="/denoise" className="btn btn-lg btn-primary">
            Start Now <FaArrowRight />
          </Link>
        </div>
      </div>
    </div>
  );
};

export default Dashboard; 