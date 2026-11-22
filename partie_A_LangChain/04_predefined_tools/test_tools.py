
import os
from dotenv import load_dotenv, find_dotenv

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_tavily import TavilySearch

load_dotenv(find_dotenv())

# Tool 1 : DuckDuckGo
duckduckgo = DuckDuckGoSearchRun()

print("=== Test DuckDuckGo ===")
print(duckduckgo.invoke(
    "What is LangChain?"
))

# Tool 2 : Tavily
tavily = TavilySearch(
    max_results=3,
    search_depth="basic"
)

print("\n=== Test Tavily ===")
print(tavily.invoke({
    "query": "What is LangGraph?"
}))
