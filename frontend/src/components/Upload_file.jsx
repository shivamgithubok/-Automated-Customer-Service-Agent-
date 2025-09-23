import React, { useState, useCallback, useRef } from 'react';
import { CloudArrowUpIcon, LinkIcon, DocumentTextIcon, DocumentIcon, XMarkIcon, CheckCircleIcon } from '@heroicons/react/24/outline';
import axios from 'axios';

const UploadFile = ({ show, onClose }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [uploadedItems, setUploadedItems] = useState([]);
  const [selectedUploadType, setSelectedUploadType] = useState('file');
  const [linkInput, setLinkInput] = useState('url');
  const fileInputRef = useRef(null);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = Array.from(e.dataTransfer.files);
    processFiles(files);
  }, []);

  const handleFileInputChange = useCallback((e) => {
    const files = Array.from(e.target.files);
    processFiles(files);
  }, []);


const uploadFilesToBackend = async (files) => {
  try {
    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file); // Changed to 'files' to match backend
    });

    const response = await fetch('http://localhost:5000/api/upload', {
      method: 'POST',
      body: formData
    });

    if (response.ok) {
      const responseData = await response.json();
      if (responseData.message === "Files processed") {
        return { success: true, details: responseData.details };
      }
    }
    return { success: false, error: "Upload failed" };
  } catch (error) {
    console.error('Error uploading files:', error);
    return { 
      success: false, 
      error: error.response?.data?.detail || error.message || "Error uploading files" 
    };
  }
};


  const processFiles = async (files) => {
    const newFiles = files.map(file => {
      const isValidType = file.type === 'application/pdf' || file.type === 'text/plain';
      return {
        id: file.name + Date.now(),
        name: file.name,
        type: file.type,
        size: file.size,
        file: file,
        isValid: isValidType,
        status: isValidType ? 'processing' : 'failed',
      };
    });

    setUploadedItems((prevItems) => [...prevItems, ...newFiles]);

    const validFiles = newFiles.filter(f => f.isValid).map(f => f.file);
    if (validFiles.length > 0) {
      const uploadResult = await uploadFilesToBackend(validFiles);
      
      setUploadedItems(prevItems => 
        prevItems.map(item => {
          const matchingNewFile = newFiles.find(f => f.id === item.id);
          if (matchingNewFile) {
            return {
              ...item,
              status: uploadResult.success ? 'uploaded' : 'failed',
              error: uploadResult.error
            };
          }
          return item;
        })
      );

      if (!uploadResult.success) {
        console.error('Upload failed:', uploadResult.error);
      }
    }

    const invalidFiles = newFiles.filter(f => !f.isValid);
    if (invalidFiles.length > 0) {
      console.warn('Some files were skipped. Only PDF and TXT files are supported.');
    }
  };

  const handleAddLink = async () => {   
  try {
    const response = await fetch('http://localhost:5000/scrap', {
      method: 'POST',
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: linkInput })
    });

    if (response.ok) {
      const responseData = await response.json();
      console.log("Scraped:", responseData);
      return { success: true, details: responseData.data };
    }
    return { success: false, error: "Scraping failed" };
  }
  catch (err) {
    console.error("Error scraping link:", err);
    return { success: false, error: err.message };
  }
};




  const handleRemoveItem = (idToRemove) => {
    setUploadedItems((prevItems) => prevItems.filter(item => item.id !== idToRemove));
  };

  const triggerFileInput = () => {
    fileInputRef.current.click();
  };

  const getItemIcon = (type, status) => {
    if (status === 'uploaded') {
      if (type === 'application/pdf') return <DocumentIcon className="file-icon pdf-icon" />;
      if (type === 'text/plain') return <DocumentTextIcon className="file-icon txt-icon" />;
      if (type === 'link') return <LinkIcon className="file-icon link-icon" />;
    }
    return <DocumentIcon className="file-icon default-icon" />;
  };

  return (
    <div className="upload-file-wrapper">
      <div className="upload-header">
        <h2 className="upload-title">Upload Knowledge</h2>
        <button
          onClick={onClose}
          className="upload-close-button"
          aria-label="Close upload panel"
        >
          <XMarkIcon className="upload-close-icon" />
        </button>
      </div>

      {/* Upload Type Selection Tabs */}
      <div className="upload-type-tabs">
        <button
          onClick={() => setSelectedUploadType('file')}
          className={`tab-button ${selectedUploadType === 'file' ? 'active' : ''}`}
        >
          <CloudArrowUpIcon className="tab-icon" /> Files (PDF, TXT)
        </button>
        <button
          onClick={() => setSelectedUploadType('link')}
          className={`tab-button ${selectedUploadType === 'link' ? 'active' : ''}`}
        >
          <LinkIcon className="tab-icon" /> Links (URLs)
        </button>
      </div>

      {/* Conditional Upload Area based on selected type */}
      {selectedUploadType === 'file' && (
        <div
          className={`upload-area ${isDragging ? 'dragging' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <CloudArrowUpIcon className="upload-area-icon" />
          <p className="upload-area-text">Drag & Drop your files here</p>
          <p className="upload-area-subtext">(Supports PDF, TXT • Max 10MB per file)</p>
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileInputChange}
            multiple
            accept=".pdf,.txt"
            className="hidden-file-input"
          />
          <button
            onClick={triggerFileInput}
            className="browse-files-button"
          >
            Or Browse Files
          </button>
        </div>
      )}

      {selectedUploadType === 'link' && (
          <div className="link-input-area">
            <p className="link-input-text">Enter URL to add as knowledge source</p>
            <input
              type="url"
              value={linkInput}
              onChange={(e) => setLinkInput(e.target.value)}
              placeholder="e.g., https://example.com/documentation"
              className="link-input-field"
            />
            <button
              onClick={handleAddLink}
              className="add-link-button"
            >
              Add Link
            </button>
          </div>
        )}


      {/* Uploaded Items Display */}
      <div className="uploaded-items-display">
        <h3 className="uploaded-items-title">Uploaded Items ({uploadedItems.length}):</h3>
        {uploadedItems.length === 0 ? (
          <p className="no-items-message">No items added yet. Start by uploading files or links!</p>
        ) : (
          <ul className="uploaded-items-list">
            {uploadedItems.map((item) => (
              <li key={item.id} className="uploaded-item">
                <div className="item-info">
                  {getItemIcon(item.type, item.status)}
                  <span className="item-name">{item.name}</span>
                  {item.status === 'processing' && (
                    <span className="processing-status">Processing...</span>
                  )}
                  {item.status === 'uploaded' && (
                    <CheckCircleIcon className="item-status-icon success" title="Successfully uploaded" />
                  )}
                  {item.status === 'failed' && (
                    <div className="error-status" title={item.error || "Upload failed"}>
                      <XMarkIcon className="error-icon" />
                    </div>
                  )}
                </div>
                <button
                  onClick={() => handleRemoveItem(item.id)}
                  className="remove-item-button"
                  aria-label={`Remove ${item.name}`}
                >
                  <XMarkIcon className="remove-item-icon" />
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
};

export default UploadFile;
