import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException

# app = FastAPI()

# Storage for scraped content
scraped_data = {}


# Function to scrape website content
def scrape_website(url: str) -> dict:
    """Scrape website content and store in dictionary"""
    try:
        # Validate URL
        if not url:
            raise ValueError("URL cannot be empty")
            
        # Add headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Fetch the webpage
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        # Ensure we're getting HTML content
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' not in content_type:
            raise ValueError(f"Invalid content type: {content_type}")

        # Parse HTML
        soup = BeautifulSoup(response.content, "html.parser")

        # Extract text by class or id
        content_by_class = {}
        for tag in soup.find_all(True):  # Search for all tags
            if tag.get('class'):
                for class_name in tag.get('class'):
                    if class_name not in content_by_class:
                        content_by_class[class_name] = []
                    content_by_class[class_name].append(tag.get_text(strip=True))
            if tag.get('id'):
                content_by_class[tag.get('id')] = tag.get_text(strip=True)

        # Store the scraped content globally
        scraped_data['content_by_class'] = content_by_class

        return {"status": "success", "message": "Website scraped successfully."}

    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch webpage: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error scraping webpage: {str(e)}")


# Function to answer query based on class or id
def answer_query(query: str) -> dict:
    """Answer the query based on class or ID search"""
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    response = []

    # Search for the query in scraped data by class or id
    for class_name, content_list in scraped_data.get('content_by_class', {}).items():
        if query.lower() in class_name.lower():
            response.append({"class": class_name, "content": content_list})

    # If query doesn't match any class or id, return a not found message
    if not response:
        raise HTTPException(status_code=404, detail="No content found for the provided query.")

    return {"status": "success", "data": response}