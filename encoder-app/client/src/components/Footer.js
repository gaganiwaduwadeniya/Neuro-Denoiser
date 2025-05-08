import React from 'react';
import './Footer.css';

const Footer = () => {
  return (
    <footer className="app-footer">
      <div className="footer-container">
        <div className="footer-content">
          <div className="footer-brand">
            <span className="footer-brand-name">NeuroDenoiser</span>
            <span className="footer-slogan">Advanced Neural Image Denoising</span>
          </div>
          
          <div className="footer-info">
            <p>© {new Date().getFullYear()} NeuroDenoiser. All rights reserved.</p>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer; 