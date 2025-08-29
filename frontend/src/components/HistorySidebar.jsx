import React from 'react';
import { PlusIcon, ChatBubbleLeftIcon, Bars3Icon } from '@heroicons/react/24/outline';

const HistorySidebar = ({ 
  isOpen, 
  onToggle, 
  chatHistory, 
  currentChatId, 
  onNewChat, 
  onSelectChat 
}) => {
  return (
    <>
      <div className={`history-sidebar ${isOpen ? 'open' : ''}`}>
        <div className="history-header">
          <button className="new-chat-button" onClick={onNewChat}>
            <PlusIcon className="new-chat-icon" /> New Chat
          </button>
        </div>
        <div className="chat-history">
          {chatHistory.map(chat => (
            <div
              key={chat.id}
              className={`chat-history-item ${currentChatId === chat.id ? 'active' : ''}`}
              onClick={() => onSelectChat(chat.id)}
            >
              <ChatBubbleLeftIcon className="chat-history-icon" />
              <span className="chat-history-title">{chat.title}</span>
              <span className="chat-history-time">
                {new Date(chat.timestamp).toLocaleDateString()}
              </span>
            </div>
          ))}
        </div>
      </div>
      <button 
        className={`sidebar-toggle ${isOpen ? 'open' : ''}`}
        onClick={onToggle}
        aria-label="Toggle history sidebar"
      >
        <Bars3Icon className="sidebar-toggle-icon" />
      </button>
    </>
  );
};

export default HistorySidebar;
