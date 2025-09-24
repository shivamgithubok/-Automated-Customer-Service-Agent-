import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const ChatContainer = ({ messages, loading }) => {
  return (
    <div className="chat-container">
      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="welcome-message">
            <h2>Welcome to Document Mining Assistant</h2>
            <p>Ask questions about your documents and I'll help you find answers!</p>
            <h1>UPLOAD YOUR DOCUMENTS TO GET STARTED</h1>
          </div>
        ) : (
          messages.map((message, index) => (
            <div key={index} className={`chat-message ${message.type}`}>
              <div className="message-icon">
                {message.type === 'user' ? '👤' : message.type === 'assistant' ? '🤖' : '⚠️'}
              </div>
              <div className="message-content">
                {/* --- THIS IS THE CORRECTED CODE --- */}
                {/* Instead of putting the className ON ReactMarkdown, we wrap it in a div */}
                {message.type === 'assistant' ? (
                  <div className="message-text">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {message.content}
                    </ReactMarkdown>
                  </div>
                ) : (
                  <div className="message-text">{message.content}</div>
                )}
                {/* --- END OF CORRECTION --- */}

                {message.sources && message.sources.length > 0 && (
                  <div className="message-sources">
                    <details>
                      <summary>Sources ({message.docsFound} documents found)</summary>
                      <ul>
                        {message.sources.map((source, idx) => (
                          <li key={idx}>{source}</li>
                        ))}
                      </ul>
                    </details>
                  </div>
                )}
                <div className="message-timestamp">
                  {new Date(message.timestamp).toLocaleTimeString()}
                </div>
              </div>
            </div>
          ))
        )}
        {loading && (
          <div className="chat-message assistant loading">
            <div className="message-icon">🤖</div>
            <div className="message-content">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatContainer;






// const ChatContainer = ({ messages, loading }) => {
//   return (
//     <div className="chat-container">
//       <div className="chat-messages">
//         {messages.length === 0 ? (
//           <div className="welcome-message">
//             <h2>Welcome to Document Mining Assistant</h2>
//             <p>Ask questions about your documents and I'll help you find answers!</p>
//             <h1>UPLOAD YOUR DOCUMENTS TO GET STARTED</h1>
//           </div>
//         ) : (
//           messages.map((message, index) => (
//             <div key={index} className={`chat-message ${message.type}`}>
//               <div className="message-icon">
//                 {message.type === 'user' ? '👤' : message.type === 'assistant' ? '🤖' : '⚠️'}
//               </div>
//               <div className="message-content">
//                 <div className="message-text">{message.content}</div>
//                 {message.sources && message.sources.length > 0 && (
//                   <div className="message-sources">
//                     <details>
//                       <summary>Sources ({message.docsFound} documents found)</summary>
//                       <ul>
//                         {message.sources.map((source, idx) => (
//                           <li key={idx}>{source}</li>
//                         ))}
//                       </ul>
//                     </details>
//                   </div>
//                 )}
//                 <div className="message-timestamp">
//                   {new Date(message.timestamp).toLocaleTimeString()}
//                 </div>
//               </div>
//             </div>
//           ))
//         )}
//         {loading && (
//           <div className="chat-message assistant loading">
//             <div className="message-icon">🤖</div>
//             <div className="message-content">
//               <div className="typing-indicator">
//                 <span></span>
//                 <span></span>
//                 <span></span>
//               </div>
//             </div>
//           </div>
//         )}
//       </div>
//     </div>
//   );
// };

// export default ChatContainer;