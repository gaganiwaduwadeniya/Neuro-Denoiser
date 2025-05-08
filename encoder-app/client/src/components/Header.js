import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { FaBrain, FaImage, FaImages } from 'react-icons/fa';
import './Header.css';

const Header = () => {
  const location = useLocation();
  const [scrolled, setScrolled] = useState(false);
  
  // Add scroll event listener to change header style on scroll
  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 50) {
        setScrolled(true);
      } else {
        setScrolled(false);
      }
    };
    
    window.addEventListener('scroll', handleScroll);
    
    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);
  
  return (
    <header className={`app-header ${scrolled ? 'scrolled' : ''}`}>
      <div className="header-container">
        <Link to="/" className="brand">
          <div className="logo-wrapper">
            <FaBrain className="logo-icon pulse" />
          </div>
          <span className="logo-text">NeuroDenoiser</span>
        </Link>
        
        <nav className="nav-links">
          <Link to="/" className={location.pathname === '/' ? 'active' : ''}>
            Dashboard
          </Link>
          <Link to="/denoise" className={location.pathname === '/denoise' ? 'active' : ''}>
            <FaImage className="nav-icon" />
            Denoise Image
          </Link>
          <Link to="/gallery" className={location.pathname === '/gallery' ? 'active' : ''}>
            <FaImages className="nav-icon" />
            Gallery
          </Link>
        </nav>
      </div>
    </header>
  );
};

export default Header; 