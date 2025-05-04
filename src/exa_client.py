import os
import requests
import json
from typing import Optional, Dict, Any, List
from .config_util import load_config

class ExaClient:
    """Client for interacting with the Exa.ai API."""
    
    def __init__(self, api_key: Optional[str] = None, debug: bool = False, config_path: Optional[str] = None):
        """Initialize the Exa client.
        
        Args:
            api_key: The Exa.ai API key. If not provided, will look for it in config or EXA_API_KEY environment variable.
            debug: Whether to print debug information.
            config_path: Optional path to a custom config file.
        """
        # Load config to check for API key
        config = load_config(config_path)
        
        # Try to get API key from various sources in order of preference:
        # 1. Explicit parameter passed
        # 2. Configuration file
        # 3. Environment variable
        self.api_key = api_key or config["exa"]["api_key"] or os.environ.get("EXA_API_KEY")
        
        if not self.api_key:
            raise ValueError("Exa API key not found. Please set it in config.yaml or the EXA_API_KEY environment variable.")
        
        self.base_url = "https://api.exa.ai"
        self.debug = debug
    
    def search(self, query: str, num_results: int = 5, livecrawl: str = "fallback") -> Dict[str, Any]:
        """Perform a web search using Exa.ai.
        
        Args:
            query: The search query
            num_results: Number of results to return (default: 5)
            livecrawl: Livecrawl strategy: 'always' to always crawl live, 
                       'fallback' to only crawl when index has no result
        
        Returns:
            Dictionary containing search results
        """
        url = f"{self.base_url}/search"
        
        headers = {
            "content-type": "application/json",
            "x-api-key": self.api_key
        }
        
        data = {
            "query": query,
            "numResults": num_results,
            "livecrawl": livecrawl,
            "contents": {
                "text": True
            }
        }
        
        if self.debug:
            print(f"[DEBUG] Making Exa search request: URL={url}, Query={query}")
        
        try:
            response = requests.post(url, headers=headers, json=data)
            
            if self.debug:
                print(f"[DEBUG] Exa API Response Status: {response.status_code}")
                if response.status_code != 200:
                    print(f"[DEBUG] Error Response Content: {response.text}")
            
            response.raise_for_status()
            result = response.json()
            
            if self.debug:
                print(f"[DEBUG] Exa API Response: {json.dumps(result, indent=2)[:200]}...")
            
            return result
        except requests.exceptions.RequestException as e:
            if self.debug:
                print(f"[DEBUG] Exa API Request Exception: {str(e)}")
            raise Exception(f"Error performing Exa.ai search: {str(e)}")
    
    def format_results(self, results: Dict[str, Any]) -> str:
        """Format search results into a readable string.
        
        Args:
            results: The search results from the Exa.ai API
            
        Returns:
            Formatted string with search results
        """
        if self.debug:
            print(f"[DEBUG] Formatting results: {type(results)}")
        
        # Check if results is valid and has the 'results' field
        if not results or "results" not in results:
            if self.debug:
                print(f"[DEBUG] No valid results found: {json.dumps(results, indent=2)[:300]}...")
            return "No results found."
        
        formatted_results = []
        for i, result in enumerate(results["results"], 1):
            title = result.get("title", "No title")
            url = result.get("url", "No URL")
            snippet = result.get("text", "")
            
            # Truncate text if it's too long
            if len(snippet) > 300:
                snippet = snippet[:300] + "..."
            
            formatted_results.append(f"{i}. {title}\n   URL: {url}\n   Summary: {snippet}\n")
        
        return "\n".join(formatted_results) 