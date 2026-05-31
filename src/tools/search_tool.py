from langchain_community.tools import DuckDuckGoSearchRun

def get_search_tool():
    return DuckDuckGoSearchRun()