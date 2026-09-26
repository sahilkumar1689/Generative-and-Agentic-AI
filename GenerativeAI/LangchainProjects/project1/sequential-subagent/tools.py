import os
import json
import requests
from rich import print
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_tavily import TavilySearch

load_dotenv()


tavily_api_key = os.getenv('TAVILY_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')


# Creating tools:

@tool
def web_search(query:str) -> str:
    """This is a web search tool, it takes the user query and fetch the browser result for you."""

    tool_ref = TavilySearch(max_results=1,api_key=tavily_api_key)

    search_result = tool_ref.invoke(query.strip())

    # print(search_result)
    results = [
        {
            "title": news["title"],
            "url": news["url"],
            "description": news["content"],
        }
        for news in search_result["results"]
    ]

    return json.dumps(results)


@tool
def web_scrapping(url: str) -> str:
    """This is a web scrapping tool, it takes the user query and fetch the browser result for you."""

    try:
        response = requests.get(url,timeout=60, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()  # Raise an exception for HTTP errors
        # print("response: ",response.text)
        soup = BeautifulSoup(response.text, 'html.parser')
        for i in soup(['script','style','header','footer','nav']):
            i.decompose()
        text_content = soup.get_text(separator='\n', strip=True)
        # print("text_content: ",text_content[3000:500])
        return text_content[1000:100]
    except requests.exceptions.RequestException as e:
        return f"Error occurred while fetching the URL: {e}"

