from langchain.tools import tool
import requests 
from bs4 import BeautifulSoup 
from tavily import TavilyClient 
from dotenv import load_dotenv 
import os
from rich import print

load_dotenv()


tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY")) 

@tool
def web_search(query:str)->str:
    """search the web recent and reliable information on topic .Returns titles,URLs and snippets."""
    results=tavily.search(query=query,max_results=5)
    out = []
    for r in results['results']:
        out.append(
        f'Title:{r['title']}\n URLs:{r['url']}\n Snippet:{r['content'][:300]}\n')
    return "\n------\n".join(out)

@tool 
def scrap_url(url: str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading."""
    try:
        # 1. Set headers to mimic a real web browser (prevents 403 Forbidden errors)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        # 2. Fetch the webpage content safely with a timeout limit
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()  # Throws an exception for 404 or 500 errors
        
        # 3. Parse the raw HTML structure using BeautifulSoup
        soup = BeautifulSoup(response.text, "html.parser")
        
        # 4. Decompose/Remove boilerplate noise and script structures
        noise_elements = ["script", "style", "nav", "footer", "header", "aside", "form"]
        for element in soup(noise_elements):
            element.decompose()
            
        # 5. Extract and strip raw readable text string sequences
        # The separator space keeps text formatting isolated cleanly
        clean_text = soup.get_text(separator=" ", strip=True)
        
        return clean_text
        
    except requests.exceptions.RequestException as e:
        return f"Error fetching the URL: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"


