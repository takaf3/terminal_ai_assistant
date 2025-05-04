import os
import json
from datetime import datetime
from pathlib import Path

# Memory file path
MEMORY_DIR = Path(os.path.expanduser("~/.terminal_assistant"))
MEMORY_FILE = MEMORY_DIR / "user_memory.json"

def ensure_memory_file_exists():
    """Ensure memory directory and file exist"""
    MEMORY_DIR.mkdir(exist_ok=True, parents=True)
    
    if not MEMORY_FILE.exists():
        with open(MEMORY_FILE, 'w') as f:
            json.dump({
                "user_facts": [],
                "world_facts": [],
                "preferences": {},
                "history": {
                    "important_dates": {},
                    "conversations": []
                },
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            }, f, indent=2)
    
    return MEMORY_FILE

def get_memory():
    """Read and return the current memory"""
    ensure_memory_file_exists()
    
    try:
        with open(MEMORY_FILE, 'r') as f:
            memory_data = json.load(f)
            # Ensure world_facts exists even in older memory files
            if "world_facts" not in memory_data:
                memory_data["world_facts"] = []
            return memory_data
    except Exception as e:
        print(f"Error reading memory file: {str(e)}")
        return {
            "user_facts": [],
            "world_facts": [],
            "preferences": {},
            "history": {
                "important_dates": {},
                "conversations": []
            },
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat()
        }

def save_memory(memory_data):
    """Save the memory to disk"""
    ensure_memory_file_exists()
    
    # Update the last_updated timestamp
    memory_data["last_updated"] = datetime.now().isoformat()
    
    try:
        with open(MEMORY_FILE, 'w') as f:
            json.dump(memory_data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving memory file: {str(e)}")
        return False

def add_user_fact(fact):
    """Add a new fact about the user"""
    memory = get_memory()
    
    # Add the new fact with timestamp
    memory["user_facts"].append({
        "fact": fact,
        "added_at": datetime.now().isoformat()
    })
    
    return save_memory(memory)

def add_world_fact(fact):
    """Add a new fact about the world"""
    memory = get_memory()
    
    # Add the new fact with timestamp
    memory["world_facts"].append({
        "fact": fact,
        "added_at": datetime.now().isoformat()
    })
    
    return save_memory(memory)

def add_user_preference(category, preference):
    """Add or update a user preference"""
    memory = get_memory()
    
    if category not in memory["preferences"]:
        memory["preferences"][category] = {}
    
    memory["preferences"][category].update(preference)
    
    return save_memory(memory)

def add_important_date(description, date_str):
    """Add an important date to remember"""
    memory = get_memory()
    
    memory["history"]["important_dates"][description] = {
        "date": date_str,
        "added_at": datetime.now().isoformat()
    }
    
    return save_memory(memory)

def search_memory(query):
    """Search the memory for specific information"""
    memory = get_memory()
    results = []
    
    # Convert query to lowercase for case-insensitive search
    query_lower = query.lower()
    
    # Special handling for common queries
    is_name_query = any(term in query_lower for term in ["name", "call me", "who am i"])
    is_birthday_query = any(term in query_lower for term in ["birthday", "born", "birth date"])
    is_preference_query = any(term in query_lower for term in ["like", "prefer", "favorite", "preference"])
    
    # Extract keywords from query
    keywords = [word.strip() for word in query_lower.split() if len(word.strip()) > 2]
    
    # Search user facts
    for fact in memory["user_facts"]:
        fact_lower = fact["fact"].lower()
        
        # Direct match
        if query_lower in fact_lower:
            results.append(("User fact", fact["fact"]))
            continue
            
        # Special case for name queries
        if is_name_query and any(name_word in fact_lower for name_word in ["name", "call", "called"]):
            results.append(("User fact", fact["fact"]))
            continue
            
        # Keyword matching
        if any(keyword in fact_lower for keyword in keywords):
            results.append(("User fact", fact["fact"]))
            continue
    
    # Search world facts
    for fact in memory["world_facts"]:
        fact_lower = fact["fact"].lower()
        
        # Direct match
        if query_lower in fact_lower:
            results.append(("World fact", fact["fact"]))
            continue
            
        # Keyword matching
        if any(keyword in fact_lower for keyword in keywords):
            results.append(("World fact", fact["fact"]))
    
    # Search user preferences
    for category, prefs in memory["preferences"].items():
        for key, value in prefs.items():
            pref_text = f"{category}: {key} = {value}".lower()
            
            # Direct match
            if query_lower in pref_text:
                results.append(("Preference", f"{category}: {key} = {value}"))
                continue
                
            # Special case for preference queries
            if is_preference_query and (
                any(keyword in key.lower() for keyword in keywords) or 
                any(keyword in str(value).lower() for keyword in keywords)
            ):
                results.append(("Preference", f"{category}: {key} = {value}"))
                continue
            
            # Keyword matching
            if any(keyword in pref_text for keyword in keywords):
                results.append(("Preference", f"{category}: {key} = {value}"))
    
    # Search important dates
    for desc, date_info in memory["history"]["important_dates"].items():
        date_text = f"{desc}: {date_info['date']}".lower()
        
        # Direct match
        if query_lower in date_text:
            results.append(("Important date", f"{desc}: {date_info['date']}"))
            continue
            
        # Special case for birthday queries
        if is_birthday_query and "birthday" in desc.lower():
            results.append(("Important date", f"{desc}: {date_info['date']}"))
            continue
            
        # Keyword matching
        if any(keyword in date_text for keyword in keywords):
            results.append(("Important date", f"{desc}: {date_info['date']}"))
    
    return results

def memory_tool(operation, **kwargs):
    """Main function to interact with the memory system"""
    if operation == "get":
        return get_memory()
    
    elif operation == "add_fact":
        fact = kwargs.get("fact", "")
        if not fact:
            return {"error": "No fact provided"}
        
        success = add_user_fact(fact)
        return {"success": success, "message": "User fact added to memory"}
    
    elif operation == "add_world_fact":
        fact = kwargs.get("world_fact", "")
        if not fact:
            return {"error": "No world fact provided"}
        
        success = add_world_fact(fact)
        return {"success": success, "message": "World fact added to memory"}
    
    elif operation == "add_preference":
        category = kwargs.get("category", "")
        preference_key = kwargs.get("key", "")
        preference_value = kwargs.get("value", "")
        
        if not category or not preference_key:
            return {"error": "Category and preference key required"}
        
        success = add_user_preference(category, {preference_key: preference_value})
        return {"success": success, "message": f"User preference added to memory: {category}/{preference_key}"}
    
    elif operation == "add_date":
        description = kwargs.get("description", "")
        date = kwargs.get("date", "")
        
        if not description or not date:
            return {"error": "Description and date required"}
        
        success = add_important_date(description, date)
        return {"success": success, "message": f"Important date added to memory: {description}"}
    
    elif operation == "search":
        query = kwargs.get("query", "")
        
        if not query:
            return {"error": "Search query required"}
        
        results = search_memory(query)
        
        if not results:
            return {"success": True, "message": "No results found in memory for the query.", "results": []}
        
        # Format results more clearly
        formatted_results = "\n".join([f"- {category}: {details}" for category, details in results])
        return {
            "success": True, 
            "message": f"Memory search results:\n\n{formatted_results}", 
            "results": results
        }
    
    else:
        return {"error": f"Unknown operation: {operation}"} 