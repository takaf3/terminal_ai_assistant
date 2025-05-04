import os
import requests
import json
from typing import Optional, Dict, Any, List
from .config_util import load_config

class BraveClient:
    """Client for interacting with the Brave Search API."""
    
    def __init__(self, api_key: Optional[str] = None, debug: bool = False, config_path: Optional[str] = None):
        """Initialize the Brave Search client.
        
        Args:
            api_key: The Brave Search API key. If not provided, will look for it in config or BRAVE_SEARCH_API_KEY environment variable.
            debug: Whether to print debug information.
            config_path: Optional path to a custom config file.
        """
        # Load config to check for API key
        config = load_config(config_path)
        
        # Try to get API key from various sources in order of preference:
        # 1. Explicit parameter passed
        # 2. Configuration file
        # 3. Environment variable
        self.api_key = api_key or config["brave"]["api_key"] or os.environ.get("BRAVE_SEARCH_API_KEY")
        
        if not self.api_key:
            raise ValueError("Brave Search API key not found. Please set it in config.yaml or the BRAVE_SEARCH_API_KEY environment variable.")
        
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        self.debug = debug
        self.search_kwargs = config["brave"].get("search_kwargs", {"count": 5})
    
    def search(self, query: str, count: Optional[int] = None) -> Dict[str, Any]:
        """Perform a web search using Brave Search API.
        
        Args:
            query: The search query
            count: Number of results to return (overrides default from config)
        
        Returns:
            Dictionary containing search results
        """
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key
        }
        
        # Use count parameter if provided, otherwise use from config
        search_kwargs = self.search_kwargs.copy()
        if count is not None:
            search_kwargs["count"] = count
        
        params = {
            "q": query,
            **search_kwargs
        }
        
        if self.debug:
            print(f"[DEBUG] Making Brave search request: URL={self.base_url}, Query={query}")
        
        try:
            response = requests.get(self.base_url, headers=headers, params=params)
            
            if self.debug:
                print(f"[DEBUG] Brave API Response Status: {response.status_code}")
                if response.status_code != 200:
                    print(f"[DEBUG] Error Response Content: {response.text}")
            
            response.raise_for_status()
            result = response.json()
            
            if self.debug:
                print(f"[DEBUG] Brave API Response: {json.dumps(result, indent=2)[:200]}...")
            
            return result
        except requests.exceptions.RequestException as e:
            if self.debug:
                print(f"[DEBUG] Brave API Request Exception: {str(e)}")
            raise Exception(f"Error performing Brave search: {str(e)}")
    
    def format_results(self, results: Dict[str, Any]) -> str:
        """Format search results into a readable string.
        
        Args:
            results: The search results from the Brave Search API
            
        Returns:
            Formatted string with search results
        """
        if self.debug:
            print(f"[DEBUG] Formatting Brave search results: {type(results)}")
        
        # Check if results is valid and has the 'web' field with 'results'
        if not results or "web" not in results or "results" not in results["web"]:
            if self.debug:
                print(f"[DEBUG] No valid Brave search results found")
            return "No results found."
        
        formatted_results = []
        for i, result in enumerate(results["web"]["results"], 1):
            title = result.get("title", "No title")
            url = result.get("url", "No URL")
            description = result.get("description", "")
            
            formatted_results.append(f"{i}. {title}\n   URL: {url}\n   Summary: {description}\n")
        
        return "\n".join(formatted_results) 