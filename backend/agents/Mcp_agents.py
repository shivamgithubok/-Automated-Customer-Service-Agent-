from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.agents import initialize_agent, Tool
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.1,
)

print("Initializing local embedding model (all-MiniLM-L6-v2)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'} # Use 'cuda' if you have a GPU
)
print("Embedding model loaded.")


BASE_PATH = Path(__file__).parent
VECTOR_PATH = BASE_PATH / "production_vectorstore" 
def load_documents_into_vectorstore(documents: list, force_recreate: bool = False):
    """Loads a list of text documents into the FAISS vector store."""
    if VECTOR_PATH.exists() and not force_recreate:
        print(f"📁 Loading existing vector store from {VECTOR_PATH}...")
        return FAISS.load_local(
            str(VECTOR_PATH), embeddings, allow_dangerous_deserialization=True
        )
    else:
        print("🧠 Creating new vector store from documents...")
        doc_objects = [Document(page_content=doc) for doc in documents]
        db = FAISS.from_documents(doc_objects, embeddings)
        db.save_local(str(VECTOR_PATH))
        print(f"✅ Vector store created and saved at {VECTOR_PATH}")
        return db

# Initialize the vector store (replace with your actual documents)
example_docs = [
    "The Model Context Protocol (MCP) is an open protocol for standardizing application context for LLMs.",
    "SQL's ORDER BY clause is used to sort the result-set in ascending or descending order.",
    "The Pokémon technical assessment requires building a data resource and a battle simulation tool."
]
# Use force_recreate=True the first time you run this after changing the model
vector_store = load_documents_into_vectorstore(example_docs, force_recreate=False)

def search_local_documents(query: str) -> str:
    """Searches the internal knowledge base for information."""
    results = vector_store.similarity_search(query, k=3)
    if not results:
        return "No relevant information found in the local documents."
    return "\n---\n".join([doc.page_content for doc in results])

web_search = TavilySearchResults(max_results=3)

tools = [
    Tool(
        name="InternalDocumentSearch",
        func=search_local_documents,
        description="Use this tool FIRST to find specific information within the user-provided documents and knowledge base."
    ),
    Tool(
        name="WebSearch",
        func=web_search.run,
        description="Use this tool for real-time information, current events, or topics not found in the internal documents."
    ),
    Tool(
        name="Wikipedia",
        func=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()).run,
        description="Use this for general knowledge and historical queries about entities, people, and places."
    ),
    Tool(
        name="Arxiv",
        func=ArxivQueryRun(api_wrapper=ArxivAPIWrapper()).run,
        description="Use this for questions about scientific research papers and technical topics."
    ),
]

system_prompt = """
You are MCP, a powerful AI assistant and reasoning engine. Your goal is to answer any question, whether simple or critical, by intelligently using the tools at your disposal.

**YOUR THINKING PROCESS:**
1.  **Always start by using the `InternalDocumentSearch` tool.** The user's primary context is in their documents.
2.  If the internal documents do not contain the answer, broaden your search using the `WebSearch` or `Wikipedia` tools.
3.  For highly technical or scientific questions, use the `Arxiv` tool.
4.  Synthesize the information from all sources into a single, coherent, and logical answer. Do not just recite the tool output.
5.  If you use information from a source, CITE IT. For web search, provide the URL. For documents, state "According to the provided documents...".

**YOUR OUTPUT FORMAT (like Gemini):**
- Use Markdown for clear formatting (headings, bold, lists).
- For code examples, use Markdown code blocks with the language specified (e.g., ```sql ... ```).
- Start with a direct, summary answer. Then, provide the detailed explanation below.
- If you used multiple sources, use bullet points to list the key findings from each.
"""

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors="Check your output and make sure it conforms!",
    agent_kwargs={
        "prefix": system_prompt,
    }
)

def run_mcp_agent(question: str):
    """Runs the full MCP agent and returns a formatted answer."""
    if not question or not question.strip():
        raise ValueError("Please provide a valid question")
        
    try:
        response = agent.invoke({"input": question})
        return response.get("output", "I apologize, but I encountered an error and couldn't generate a response.")
    except Exception as e:
        print(f"Error during agent execution: {e}")
        raise ValueError(f"Error from MCP Agent: {str(e)}")


# ---------------------------------
# # 7. Example Usage
# # ---------------------------------
# if __name__ == "__main__":
#     # Question that should use the internal document search
#     q1 = "What is the purpose of the Pokémon technical assessment?"
#     print("\n--- Asking about a local document ---")
#     print("🤖 MCP Response:\n", run_mcp_agent(q1))

#     # Question that requires web search
#     q2 = "What were the key announcements from the latest Google I/O event?"
#     print("\n--- Asking a real-time web search question ---")
#     print("🤖 MCP Response:\n", run_mcp_agent(q2))```