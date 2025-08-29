from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.agents import initialize_agent, Tool
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pathlib import Path
import os

# -----------------------
# 1. LLM Setup
# -----------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# -----------------------
# 2. External Tools
# -----------------------
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())

tools = [
    Tool(name="Wikipedia", func=wiki.run, description="General knowledge queries"),
    Tool(name="Arxiv", func=arxiv.run, description="Research paper queries"),
]

# -----------------------
# 3. Vector DB Setup
# -----------------------
BASE_PATH = Path(__file__).parent
VECTOR_PATH = BASE_PATH / "vectorstore"

if VECTOR_PATH.exists():
    vector_store = FAISS.load_local(
        str(VECTOR_PATH), embeddings, allow_dangerous_deserialization=True
    )
else:
    vector_store = FAISS.from_texts(["initial"], embeddings)
    vector_store.save_local(str(VECTOR_PATH))

def save_to_memory(question: str, answer: str):
    """Save Q&A into FAISS"""
    doc = Document(page_content=f"Q: {question}\nA: {answer}")
    vector_store.add_documents([doc])
    vector_store.save_local(str(VECTOR_PATH))
    print("💾 Saved to memory:", question[:50])

# -----------------------
# 4. MCP Agent
# -----------------------
agent = initialize_agent(tools=tools, llm=llm, agent="zero-shot-react-description", verbose=True)

def mcp_ask(question: str):
    """Ask MCP Agent and also store Q&A into FAISS"""
    answer = agent.run(question)
    save_to_memory(question, answer)
    return answer

# -----------------------
# 5. Example
# -----------------------
# if __name__ == "__main__":
#     q = "What is quantum computing?"
#     print("🤖 MCP:", mcp_ask(q))
