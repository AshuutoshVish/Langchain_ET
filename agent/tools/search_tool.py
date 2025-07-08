import os
from langchain.tools import Tool
from serpapi import GoogleSearch
from dotenv import load_dotenv

load_dotenv()

def search_google(query: str) -> str:
    params = {
        "q": query,
        "api_key": os.getenv("SERPAPI_API_KEY"),
        "engine": "google"
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    try:
        return results["organic_results"][0]["snippet"]
    except:
        return "No search results found."

search_tool = Tool(
    name="Search",
    description="Use this tool to search the internet using SerpAPI.",
    func=search_google
)
