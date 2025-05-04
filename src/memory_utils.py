import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from .vector_memory import VectorMemory

# Global vector memory instance
vector_memory = None

def get_vector_memory(debug=False, config_path=None) -> VectorMemory:
    """
    Get or create the vector memory instance.
    
    Args:
        debug: Enable debug output
        config_path: Path to configuration file
    
    Returns:
        VectorMemory instance
    """
    global vector_memory
    
    if vector_memory is None:
        vector_memory = VectorMemory(debug=debug, config_path=config_path)
    
    return vector_memory

def memory_tool(operation: str, fact: Optional[str] = None, category: Optional[str] = None, 
                key: Optional[str] = None, value: Optional[str] = None, 
                description: Optional[str] = None, date: Optional[str] = None, 
                world_fact: Optional[str] = None, query: Optional[str] = None,
                debug: bool = False, config_path: Optional[str] = None) -> str:
    """
    Memory tool that provides persistent memory capabilities for the agent.
    
    Args:
        operation: Operation to perform: get, add_fact, add_preference, add_date, add_world_fact, search
        fact: Fact about the user to remember
        category: Category for preference
        key: Key for preference
        value: Value for preference
        description: Description of important date
        date: Date string
        world_fact: Fact about the world
        query: Search query
        debug: Enable debug output
        config_path: Path to configuration file
    
    Returns:
        Result of the operation
    """
    try:
        # Initialize the vector memory
        vm = get_vector_memory(debug=debug, config_path=config_path)
        
        if operation == "get":
            # Get all memories
            result = []
            
            # Get user facts
            facts = vm.get_all_memories("facts")
            if facts:
                result.append("User Facts:")
                for doc in facts:
                    result.append(f"- {doc.page_content}")
            
            # Get preferences
            preferences = vm.get_all_memories("preferences")
            if preferences:
                result.append("\nUser Preferences:")
                for doc in preferences:
                    result.append(f"- {doc.page_content}")
            
            # Get important dates
            dates = vm.get_all_memories("dates")
            if dates:
                result.append("\nImportant Dates:")
                for doc in dates:
                    result.append(f"- {doc.page_content}")
            
            # Get world facts
            world_facts = vm.get_all_memories("world_facts")
            if world_facts:
                result.append("\nWorld Facts:")
                for doc in world_facts:
                    result.append(f"- {doc.page_content}")
            
            if not result:
                return "No memories found."
            
            return "\n".join(result)
        
        elif operation == "add_fact":
            if not fact:
                return "Error: No fact provided."
            
            doc_id = vm.add_fact(fact)
            return f"Fact added to memory: {fact}"
        
        elif operation == "add_preference":
            if not category or not key or not value:
                return "Error: Missing category, key, or value for preference."
            
            doc_id = vm.add_preference(category, key, value)
            return f"Preference added to memory: {category}/{key}={value}"
        
        elif operation == "add_date":
            if not description or not date:
                return "Error: Missing description or date."
            
            doc_id = vm.add_date(description, date)
            return f"Important date added to memory: {description} - {date}"
        
        elif operation == "add_world_fact":
            if not world_fact:
                return "Error: No world fact provided."
            
            doc_id = vm.add_world_fact(world_fact)
            return f"World fact added to memory: {world_fact}"
        
        elif operation == "search":
            if not query:
                return "Error: No search query provided."
            
            results = vm.search(query)
            
            if not results:
                return f"No memories found matching '{query}'."
            
            result_texts = [f"Memory search results for '{query}':"]
            
            for doc, score in results:
                # Round score to 4 decimal places for readability
                score_rounded = round(score, 4)
                result_texts.append(f"- [{score_rounded}] {doc.page_content}")
            
            return "\n".join(result_texts)
        
        else:
            return f"Error: Unknown operation '{operation}'. Valid operations: get, add_fact, add_preference, add_date, add_world_fact, search."
    
    except Exception as e:
        return f"Error using memory tool: {str(e)}" 