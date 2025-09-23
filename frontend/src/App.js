import React, { useState } from 'react';
import './App.css';
import WelcomeScreen from './components/WelcomeScreen';
import ChatContainer from './components/ChatContainer';
import HistorySidebar from './components/HistorySidebar';
import QueryInput from './components/QueryInput';
import UploadPanelWrapper from './components/UploadPanelWrapper';

function App() {
  const [isPanelVisible, setIsPanelVisible] = useState(false);
  const [queryInput, setQueryInput] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [isStarted, setIsStarted] = useState(false);
  const [messages, setMessages] = useState([]);
  const [chatHistory, setChatHistory] = useState([]);
  const [currentChatId, setCurrentChatId] = useState(null);
  const [isHistorySidebarOpen, setIsHistorySidebarOpen] = useState(true);
  const [selectedUploadType, setSelectedUploadType] = useState("document"); 
  const [linkInput, setLinkInput] = useState('url');

  const handleMcpQuery = async () => {
    // Check if there's a question to ask
    if (!queryInput.trim()) {
      const errorMessage = {
        type: 'error',
        content: 'Please enter a question first.',
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, errorMessage]);
      return;
    }

    setLoading(true);
    
    // Add user message immediately
    const userMessage = {
      type: 'user',
      content: queryInput,
      timestamp: new Date().toISOString()
    };
    setMessages(prev => [...prev, userMessage]);

    try {
      const response = await fetch('http://localhost:5000/api/ask-mcp', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: queryInput }),
      });

      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.detail || 'Failed to get response from MCP server');
      }

      const aiMessage = {
        type: 'assistant',
        content: data.answer,
        timestamp: new Date().toISOString(),
        isMcp: true
      };

      setMessages(prev => [...prev, aiMessage]);
      setResponse(data.answer);
      setQueryInput(''); // Clear input after successful response

    } catch (error) {
      console.error('MCP Error:', error);
      const errorMessage = {
        type: 'error',
        content: error.message || 'Error connecting to MCP server. Please try again.',
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleStartClick = () => {
    setIsStarted(true);
    createNewChat();
  };

  const togglePanel = () => {
    setIsPanelVisible(!isPanelVisible);
  };

  const toggleHistorySidebar = () => {
    setIsHistorySidebarOpen(!isHistorySidebarOpen);
  };

  const createNewChat = () => {
    const newChatId = Date.now().toString();
    const newChat = {
      id: newChatId,
      title: 'New Chat',
      timestamp: new Date().toISOString(),
      messages: []
    };
    setChatHistory(prev => [newChat, ...prev]);
    setCurrentChatId(newChatId);
    setMessages([]);
  };

  const selectChat = (chatId) => {
    setCurrentChatId(chatId);
    const chat = chatHistory.find(c => c.id === chatId);
    if (chat) {
      setMessages(chat.messages);
    }
  };

  const handleClosePanel = () => {
    setIsPanelVisible(false);
  };
const handleQuerySubmit = async (e) => {
  e.preventDefault();
  if (!queryInput.trim()) return;

  // Add user message immediately
  const userMessage = {
    type: 'user',
    content: queryInput,
    timestamp: new Date().toISOString()
  };
  setMessages(prev => [...prev, userMessage]);

  setLoading(true);

  try {
    // Send the query to the backend
    let endpoint = "http://localhost:5000/api/ask-rag";
    console.log("Selected Upload Type:", selectedUploadType);
    if (selectedUploadType === "link") {
      endpoint = "http://localhost:5000/api/ask-scraped";
    }
    
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        question: queryInput, 
        url: linkInput  
      }),
    });

    // const response = await fetch('http://localhost:5000/ask', {
    //   method: 'POST',
    //   headers: {
    //     'Content-Type': 'application/json',
    //   },
    //   body: JSON.stringify({ question: queryInput }),
    // });

    // Check if the response is ok
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Server error occurred');
    }

    // Parse the response data
    const data = await response.json();

    // Validate response structure
    if (!data || !data.answer) {
      throw new Error('Invalid response from server: Missing answer');
    }

    // Add AI response with both question and answer
    const aiMessage = {
      type: 'assistant',
      content: data.answer,
      timestamp: new Date().toISOString(),
      sources: data.source_documents || [],  // Ensure this is an array
      docsFound: data.num_docs_found || 0     // Ensure this is a number
    };

    // Create updated messages array including both user message and AI response
    const updatedMessages = [...messages, userMessage, aiMessage];
    setMessages(updatedMessages);
    setResponse(data.answer);

    // Update chat history with the new messages and current query
    setChatHistory(prev => prev.map(chat => {
      if (chat.id === currentChatId) {
        return {
          ...chat,
          messages: updatedMessages,
          title: userMessage.content.slice(0, 30) + '...'  // Trim title to avoid overflow
        };
      }
      return chat;
    }));

  } catch (error) {
    console.error('Error:', error);

    // Handle errors gracefully
    const errorMessage = {
      type: 'error',
      content: error.message || 'Error processing your query. Please try again.',
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage, errorMessage]);
    setResponse(error.message || 'Error processing your query. Please try again.');
  } finally {
    setLoading(false);
    setQueryInput(''); // Clear input field after submission
  }
};


  return (
    <div className={`app-container ${isPanelVisible ? 'panel-open' : ''} ${isHistorySidebarOpen ? 'sidebar-open' : ''}`}>
      {isStarted && (
        <HistorySidebar
          isOpen={isHistorySidebarOpen}
          onToggle={toggleHistorySidebar}
          chatHistory={chatHistory}
          currentChatId={currentChatId}
          onNewChat={createNewChat}
          onSelectChat={selectChat}
        />
      )}

      {!isStarted ? (
        <WelcomeScreen onStart={handleStartClick} />
      ) : (
        <ChatContainer messages={messages} loading={loading} />
      )}
      
      {isStarted && (
        <>
          <UploadPanelWrapper
            isPanelVisible={isPanelVisible}
            onToggle={togglePanel}
            onClose={handleClosePanel}
            selectedUploadType={selectedUploadType}
            setSelectedUploadType={setSelectedUploadType}
          />

          <QueryInput
            value={queryInput}
            onChange={setQueryInput}
            onSubmit={handleQuerySubmit}
            onMcpClick={handleMcpQuery}
            loading={loading}
          />
        </>
      )}
    </div>
  );
}

export default App;