import os
from typing import Dict, List, Any, Optional
import json
from langchain_tavily import TavilySearch
from .config_util import load_config


class TavilyClient:
    """
    Client for Tavily search API integration
    """
    def __init__(self, debug: bool = False, config_path: Optional[str] = None):
        """
        Initialize the Tavily search client
        
        Args:
            debug: Enable debug output
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path)
        self.debug = debug
        
        # Get Tavily API key, prioritize env var, then config
        self.api_key = os.environ.get("TAVILY_API_KEY", None)
        if not self.api_key and self.config.get("tavily", {}).get("api_key"):
            self.api_key = self.config["tavily"]["api_key"]
        
        if not self.api_key:
            raise ValueError("Tavily API key not found. Please set TAVILY_API_KEY environment variable or add to config.yaml")
        
        # Set up default search parameters from config
        self.search_kwargs = self.config.get("tavily", {}).get("search_kwargs", {})
        
        # Initialize the Tavily search tool
        os.environ["TAVILY_API_KEY"] = self.api_key
        self.search_tool = TavilySearch(
            max_results=self.search_kwargs.get("max_results", 5),
            include_answer=self.search_kwargs.get("include_answer", True),
            include_images=self.search_kwargs.get("include_images", False),
            include_raw_content=self.search_kwargs.get("include_raw_content", False),
        )
        
        if self.debug:
            print(f"[DEBUG] TavilyClient initialized with max_results={self.search_kwargs.get('max_results', 5)}")
    
    def search(self, query: str, **kwargs) -> str:
        """
        Perform a Tavily search
        
        Args:
            query: The search query
            **kwargs: Additional search parameters
                - max_results: Number of results to return
                - include_domains: List of domains to include
                - exclude_domains: List of domains to exclude
        
        Returns:
            Formatted search results as a string
        """
        try:
            if self.debug:
                print(f"[DEBUG] TavilyClient searching for: {query}")
                print(f"[DEBUG] TavilyClient search params: {kwargs}")
            
            # Override search parameters if provided
            search_params = {}
            if "max_results" in kwargs:
                search_params["max_results"] = kwargs.pop("max_results")
            if "include_domains" in kwargs:
                search_params["include_domains"] = kwargs.pop("include_domains")
            if "exclude_domains" in kwargs:
                search_params["exclude_domains"] = kwargs.pop("exclude_domains")
            
            # Use the LangChain Tavily Search tool
            search_result = self.search_tool.invoke({
                "query": query, 
                **search_params
            })
            
            if self.debug:
                print(f"[DEBUG] TavilyClient received result: {search_result[:100]}...")
                
            return self.format_results(search_result)
        except Exception as e:
            error_msg = f"Error performing Tavily search: {str(e)}"
            if self.debug:
                print(f"[DEBUG] {error_msg}")
            return error_msg
    
    def format_results(self, results: Dict[str, Any]) -> str:
        """
        Format the Tavily search results into a readable string
        
        Args:
            results: The raw search results
        
        Returns:
            Formatted search results as a string
        """
        if isinstance(results, str):
            # Sometimes the result might already be a string (JSON)
            try:
                results = json.loads(results)
            except json.JSONDecodeError:
                return results
        
        formatted = f"Search results for: {results.get('query', 'Unknown query')}\n\n"
        
        # Include Tavily's answer if available
        if results.get("answer"):
            formatted += f"Tavily Summary: {results['answer']}\n\n"
        
        # Format individual search results
        if results.get("results"):
            for i, result in enumerate(results["results"], 1):
                formatted += f"{i}. {result.get('title', 'No title')}\n"
                formatted += f"   URL: {result.get('url', 'No URL')}\n"
                if result.get("content"):
                    content = result["content"]
                    # Truncate content if too long
                    if len(content) > 300:
                        content = content[:297] + "..."
                    formatted += f"   Content: {content}\n"
                formatted += "\n"
        else:
            formatted += "No search results found.\n"
        
        return formatted 