import os
import json
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain_tavily import TavilySearch

load_dotenv()

tavily_api_key = os.getenv('TAVILY_API_KEY')

# ==========================================
# 1. WEB SEARCH TOOL
# ==========================================

class WebSearchInput(BaseModel):
    query: str = Field(description="The web search query string.")

@tool("web_search", args_schema=WebSearchInput)
def web_search(query: str) -> str:
    """This is a web search tool. It takes the user query and fetches live internet search results."""
    
    if not tavily_api_key:
        return json.dumps({"error": "TAVILY_API_KEY is missing from environment variables."})

    try:
        tool_ref = TavilySearch(max_results=3, api_key=tavily_api_key)
        search_result = tool_ref.invoke(query.strip())

        # Safely extract structured results
        raw_results = search_result.get("results", []) if isinstance(search_result, dict) else []
        
        results = [
            {
                "title": news.get("title", "No Title"),
                "url": news.get("url", ""),
                "description": news.get("content", ""),
            }
            for news in raw_results
        ]

        return json.dumps(results, ensure_ascii=False)
    
    except Exception as e:
        return json.dumps({"error": f"Search failed: {str(e)}"})


# ==========================================
# 2. WEB SCRAPING TOOL
# ==========================================

class WebScrapingInput(BaseModel):
    url: str = Field(description="The exact HTTP/HTTPS URL of the web page to scrape.")

@tool("web_scrapping", args_schema=WebScrapingInput)
def web_scrapping(url: str) -> str:
    """This tool fetches the raw text content from a given website URL, stripping HTML tags."""
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, timeout=15, headers=headers)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove non-content structural elements
        for element in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'noscript']):
            element.decompose()

        text_content = soup.get_text(separator='\n', strip=True)

        # Truncate content to avoid token blowup (Max 4,000 characters)
        cleaned_text = text_content[:4000]
        
        return cleaned_text if cleaned_text else "No extractable text found on page."

    except requests.exceptions.RequestException as e:
        return f"Error occurred while fetching the URL: {str(e)}"