import requests
from bs4 import BeautifulSoup
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import JsonOutputParser
import hashlib
import os
from pathlib import Path
from typing import List, Dict

# --- Selenium Imports for Dynamic Scraping ---
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class ScrapingAgent:
    def __init__(self):
        self.base_path = Path(__file__).parent.parent
        self.chroma_path = self.base_path / "chromadb_scraping"
        self.chroma_path.mkdir(parents=True, exist_ok=True)

        # --- Use local embeddings for consistency ---
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.vector_store = Chroma(
            persist_directory=str(self.chroma_path),
            embedding_function=self.embeddings
        )

        # --- Cache for storing the full, raw HTML of scraped pages ---
        self.html_cache: Dict[str, str] = {}
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, api_key=os.getenv("GOOGLE_API_KEY"))

    def generate_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _scrape_static_website(self, url: str) -> str:
        """Performs a simple scrape using requests for static HTML."""
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        return response.text

    def _scrape_dynamic_website(self, url: str) -> str:
        """Performs a dynamic scrape using Selenium to render JavaScript."""
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        try:
            driver.get(url)
            # Wait for a key element to be present or just a general wait time
            # For a more robust solution, wait for a specific element that signals content load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            page_html = driver.page_source
            return page_html
        finally:
            driver.quit()

    def scrape_website(self, url: str, dynamic: bool = False) -> dict:
        """
        Scrapes a website, caches its full HTML, and vectorizes its text content.
        Set `dynamic=True` for JavaScript-heavy pages.
        """
        try:
            if dynamic:
                html_content = self._scrape_dynamic_website(url)
            else:
                html_content = self._scrape_static_website(url)

            # --- Store the full, raw HTML in our cache ---
            self.html_cache[url] = html_content
            
            soup = BeautifulSoup(html_content, "html.parser")
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()
            text = soup.get_text(separator='\n', strip=True)

            if not text:
                return {"status": "success", "message": "Scraped, but no text content found for vectorization."}

            # Vectorize the text content for semantic search
            splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
            chunks = splitter.split_text(text)
            docs = [Document(page_content=chunk, metadata={"source": url}) for chunk in chunks]
            self.vector_store.add_documents(docs)
            
            return {"status": "success", "message": f"Successfully scraped and vectorized content from {url}"}

        except Exception as e:
            raise ValueError(f"Failed to scrape webpage: {str(e)}")

    def _create_planner_prompt(self):
        """Creates a prompt to make the LLM act as a query planner."""
        prompt_text = """
        You are a query planning expert. Your job is to analyze the user's question and determine the best way to answer it based on the available tools.

        You have two choices:
        1.  `general_query`: For questions that require understanding or summarizing the content of the page. Examples: "What is this page about?", "Summarize the main points.", "Who is the author?".
        2.  `structured_scrape`: For questions that explicitly ask to extract data from specific HTML elements using classes or IDs. The user must provide the selectors. The output should be a table. Examples: "Get the text from the div with id 'news'", "Make a table of product names with class 'prod-title' and prices with class 'price'".

        Analyze the user's question and respond with a JSON object that follows this structure:
        {{
            "query_type": "general_query" or "structured_scrape",
            "selectors": ["#id_name", ".class_name", "tag_name"],
            "column_names": ["Column 1", "Column 2"]
        }}

        - If it's a `general_query`, the `selectors` and `column_names` arrays should be empty.
        - If it's a `structured_scrape`, extract all CSS selectors (like `#main-content`, `.product-title`) and the requested column names from the question.

        Question: {question}
        JSON Response:
        """
        return PromptTemplate.from_template(prompt_text)
    
    def _perform_structured_scrape(self, html_content: str, selectors: List[str], column_names: List[str]) -> str:
        """Uses BeautifulSoup to perform the actual scrape and formats as a Markdown table."""
        soup = BeautifulSoup(html_content, "html.parser")
        
        # Scrape the data for each selector
        scraped_data = []
        for selector in selectors:
            elements = soup.select(selector)
            scraped_data.append([el.get_text(strip=True) for el in elements])
        
        # Format the output as a Markdown table
        if not scraped_data or not any(scraped_data):
            return "Could not find any elements matching the specified selectors on the page."

        # Build header
        header = "| " + " | ".join(column_names) + " |"
        separator = "| " + " | ".join(["---"] * len(column_names)) + " |"
        
        # Build rows, transposing the data
        max_len = max(len(col) for col in scraped_data)
        rows = []
        for i in range(max_len):
            row_data = []
            for col in scraped_data:
                row_data.append(col[i] if i < len(col) else "")
            rows.append("| " + " | ".join(row_data) + " |")

        return "\n".join([header, separator] + rows)

    def _answer_general_question(self, question: str, url: str) -> str:
        """Answers a general question using the vectorized text content (RAG)."""
        docs = self.vector_store.similarity_search(query=question, k=4, filter={"source": url})
        if not docs:
            return "I couldn't find any relevant information on the scraped page to answer that question."

        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        prompt = PromptTemplate.from_template(
            "You are an expert assistant. Answer the user's question based ONLY on the following context from the scraped webpage.\n\nCONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:"
        )
        chain = prompt | self.llm
        response = chain.invoke({"context": context, "question": question})
        return response.content

    def ask(self, question: str, url: str) -> str:
        """
        Main entry point for asking a question about a scraped URL.
        It plans the query, then executes the appropriate action.
        """
        if url not in self.html_cache:
            raise ValueError(f"URL not scraped yet. Please scrape '{url}' first.")

        # 1. Plan the query
        planner_prompt = self._create_planner_prompt()
        planner_chain = planner_prompt | self.llm | JsonOutputParser()
        plan = planner_chain.invoke({"question": question})

        # 2. Execute the plan
        if plan.get("query_type") == "structured_scrape":
            if not plan.get("selectors") or not plan.get("column_names"):
                return "For a structured scrape, please specify the class or ID selectors and the desired column names in your question."
            
            return self._perform_structured_scrape(
                html_content=self.html_cache[url],
                selectors=plan["selectors"],
                column_names=plan["column_names"]
            )
        else: # Default to general query
            return self._answer_general_question(question, url)

# # --- Example Usage ---
# if __name__ == '__main__':
#     # Make sure to set your GOOGLE_API_KEY as an environment variable
#     # os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"

#     # URL of the React page you want to scrape
#     # Note: Public LinkedIn profiles are often protected from scraping.
#     # This example uses a different dynamic site.
#     url_to_scrape = "https://quotes.toscrape.com/js/" # A site that uses JS to load content

#     # 1. Initialize the agent
#     agent = DynamicScrapingAgent()

#     try:
#         # 2. Scrape the website (use dynamic=True for JS-heavy sites)
#         print(f"Scraping {url_to_scrape}...")
#         scrape_status = agent.scrape_website(url_to_scrape, dynamic=True)
#         print(scrape_status['message'])

#         # 3. Ask a general question about the content
#         print("\n--- Asking a General Question ---")
#         question1 = "Who is the author of the quote that starts with 'The world as we have created it'?"
#         answer1 = agent.ask(question=question1, url=url_to_scrape)
#         print(f"Q: {question1}\nA: {answer1}")

#         # 4. Ask a structured question to extract specific data
#         print("\n--- Asking a Structured Question ---")
#         question2 = "Create a table with the quote texts using the selector '.text' and the authors using the selector '.author'. Name the columns 'Quote' and 'Author'."
#         answer2 = agent.ask(question=question2, url=url_to_scrape)
#         print(f"Q: {question2}\nA:\n{answer2}")

#     except ValueError as e:
#         print(f"An error occurred: {e}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")