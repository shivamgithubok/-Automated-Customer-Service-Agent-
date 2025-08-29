import React from 'react';
import { ArrowRightCircleIcon } from '@heroicons/react/24/outline';
import UploadFile from './Upload_file';

const UploadPanelWrapper = ({ isPanelVisible, onToggle, onClose }) => {
  return (
    <>
      <div className={`upload-panel ${isPanelVisible ? 'show' : ''}`}>
        <div className="upload-panel-content">
          <UploadFile show={isPanelVisible} onClose={onClose} />
        </div>
        <div className="panel-close-button-container">
          <button onClick={onClose} className="panel-close-button">
            Close Panel
          </button>
        </div>
      </div>

      <button
        onClick={onToggle}
        className="upload-toggle-button"
        aria-label="Toggle upload panel"
      >
        <ArrowRightCircleIcon className={`upload-toggle-icon ${isPanelVisible ? 'rotate' : ''}`} />
      </button>
    </>
  );
};

export default UploadPanelWrapper;
