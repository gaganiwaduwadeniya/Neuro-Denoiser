import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import './App.css';

// Components
import Header from './components/Header';
import Footer from './components/Footer';
import Dashboard from './pages/Dashboard';
import DenoiserTool from './pages/DenoiserTool';
import GalleryView from './pages/GalleryView';
import NotFound from './pages/NotFound';

function App() {
  return (
    <Router>
      <div className="app-wrapper">
        <Header />
        
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/denoise" element={<DenoiserTool />} />
            <Route path="/gallery" element={<GalleryView />} />
            <Route path="/404" element={<NotFound />} />
            <Route path="*" element={<Navigate to="/404" replace />} />
          </Routes>
        </main>
        
        <Footer />
        
        {/* Toast notifications container */}
        <ToastContainer
          position="bottom-right"
          autoClose={5000}
          hideProgressBar={false}
          newestOnTop
          closeOnClick
          rtl={false}
          pauseOnFocusLoss
          draggable
          pauseOnHover
          theme="dark"
        />
      </div>
    </Router>
  );
}

export default App;
