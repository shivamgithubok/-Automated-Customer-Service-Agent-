# Automated Customer Service Agent 🤖

An AI-powered document analysis and customer service system that combines RAG (Retrieval Augmented Generation) with web scraping capabilities.

## Features ✨

- **Document Processing**: Upload and analyze PDF and TXT files
- **Web Scraping**: Extract and analyze content from websites
- **Intelligent Querying**: Ask questions about uploaded documents
- **Multiple Tools**:
  - MCP Server Mode: Enhanced context handling
  - Scraping Mode: Website content analysis
  - Regular RAG: Document question-answering

## Tech Stack 🛠

### Frontend
- React.js
- Tailwind CSS
- Heroicons
- Axios

### Backend
- FastAPI
- LangChain
- Google Gemini AI
- ChromaDB & FAISS
- BeautifulSoup4
- Selenium (for dynamic scraping)

## Getting Started 🚀

### Prerequisites
- Python 3.9+
- Node.js 14+
- Google Gemini API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Automated-Customer-Service-Agent.git
cd Automated-Customer-Service-Agent
```

2. Set up the backend:
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

3. Configure environment variables:
Create a `.env` file in the backend directory:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

4. Set up the frontend:
```bash
cd ../frontend
npm install
```

### Running the Application

1. Start the backend server:
```bash
cd backend
.venv\Scripts\activate
python main.py
```

2. Start the frontend development server:
```bash
cd frontend
npm start
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend: http://localhost:5000

## Usage 💡

1. **Document Upload**:
   - Support for PDF and TXT files
   - Drag & drop interface
   - Size limit: 10MB per file

2. **Web Scraping**:
   - Enter URL in the scraping tool
   - Supports both static and dynamic websites
   - Automatic content extraction

3. **Querying**:
   - Ask questions about uploaded documents
   - Query scraped web content
   - Get structured responses

## Project Structure 📁

```
Automated-Customer-Service-Agent/
├── backend/
│   ├── agents/
│   │   ├── Rag_agent.py
│   │   ├── Mcp_agents.py
│   │   └── scraping_agents.py
│   ├── routes/
│   │   └── api.py
│   └── main.py
└── frontend/
    ├── src/
    │   ├── components/
    │   │   ├── QueryInput.jsx
    │   │   ├── Upload_file.jsx
    │   │   └── WelcomeScreen.jsx
    │   └── App.js
    └── public/
```

## API Endpoints 🌐

- `POST /api/upload`: Upload documents
- `POST /api/ask`: Query documents
- `POST /scrap`: Scrape websites
- `POST /ask_scrap`: Query scraped content
- `POST /ask-mcp`: Use MCP mode

## Contributing 🤝

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


# Automated Customer Service Agent 🤖

<div align="center">
  <img src="frontend/src/assets/images/welcome-screen.png" alt="Welcome Screen" width="600"/>
</div>

// ...existing content...

## UI Screenshots 📸

### Welcome Screen
<img src="frontend/src/assets/images/welcome-screen.png" alt="Welcome Screen" width="800"/>

### Document Upload
<img src="frontend/src/assets/images/upload-interface.png" alt="Upload Interface" width="800"/>

### Chat Interface
<img src="frontend/src/assets/images/chat-interface.png" alt="Chat Interface" width="800"/>

### Scraping Tool
<img src="frontend/src/assets/images/scraping-tool.png" alt="Scraping Tool" width="800"/>