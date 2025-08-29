import React from 'react';
import { ArrowRightCircleIcon } from '@heroicons/react/24/outline';

const WelcomeScreen = ({ onStart }) => {
  return (
    <div className="main-content">
      <h1 className="main-title">Hello Document Miners!</h1>
      <p className="main-subtitle">
        Unlock insights from your documents with our AI-powered assistant. Upload your files and start querying!
      </p>
      <button onClick={onStart} className="start-button">
        <span>Start Mining</span>
        <ArrowRightCircleIcon className="start-button-icon" />
      </button>
    </div>
  );
};

export default WelcomeScreen;