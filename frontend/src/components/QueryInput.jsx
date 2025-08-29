import React, { useState } from 'react';
import "./query.css";

const PlusIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" style={{width: '1.5rem', height: '1.5rem'}}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 4.5v15m7.5-7.5h-15" />
  </svg>
);

const ToolsIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 6h9.75M10.5 6a1.5 1.5 0 1 1-3 0m3 0a1.5 1.5 0 1 0-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 0 1-3 0m3 0a1.5 1.5 0 0 0-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 0 1-3 0m3 0a1.5 1.5 0 0 0-3 0m-9.75 0h9.75" />
    </svg>
);

const MicrophoneIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" style={{width: '1.5rem', height: '1.5rem'}}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 18.75a6 6 0 0 0 6-6v-1.5a6 6 0 0 0-12 0v1.5a6 6 0 0 0 6 6Z" />
    <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v.75a7.5 7.5 0 0 1-7.5 7.5h-.75a7.5 7.5 0 0 1-7.5-7.5v-.75" />
    <path strokeLinecap="round" strokeLinejoin="round" d="M12 12.75a.75.75 0 0 0 .75-.75v-4.5a.75.75 0 0 0-1.5 0v4.5a.75.75 0 0 0 .75.75Z" />
  </svg>
);


const QueryInput = ({ value, onChange, onSubmit, loading }) => {
  const [showTools, setShowTools] = useState(false);

  return (
    <div className="query-input-container">
      <div className="chat-bar">
        {/* Plus Icon Button */}
        {/* <button className="icon-button plus-button">
          <PlusIcon />
        </button> */}

        {/* Tools Button and Dropdown */}
        <div className="tools-container">
          <button
            onClick={() => setShowTools(!showTools)}
            className="tools-button"
          >
            <ToolsIcon />
            <span>Tools</span>
          </button>
          {showTools && (
            <div className="tools-dropdown">
              <a href="#">MCP Mode</a>
              <a href="#">Download Report</a>
            </div>
          )}
        </div>

        {/* Main input form */}
        <form onSubmit={onSubmit} className="query-form">
          <input
            type="text"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            placeholder="Ask Gemini"
            className="query-input"
            disabled={loading}
          />
        </form>

        {/* Microphone Icon Button */}
        {/* <button className="icon-button">
          {loading ? (
             <div className="loading-spinner"></div>
          ) : (
            <MicrophoneIcon />
          )}
        </button> */}
      </div>
    </div>
  );
};

export default QueryInput;

