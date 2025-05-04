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
            return json.load(f)
    except Exception as e:
        print(f"Error reading memory file: {str(e)}")
        return {
            "user_facts": [],
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
    
    # Search user facts
    for fact in memory["user_facts"]:
        if query_lower in fact["fact"].lower():
            results.append(("User fact", fact["fact"]))
    
    # Search user preferences
    for category, prefs in memory["preferences"].items():
        for key, value in prefs.items():
            if query_lower in str(value).lower() or query_lower in key.lower():
                results.append(("Preference", f"{category}: {key} = {value}"))
    
    # Search important dates
    for desc, date_info in memory["history"]["important_dates"].items():
        if query_lower in desc.lower() or query_lower in date_info["date"].lower():
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
            return {"success": True, "message": "No results found for the query", "results": []}
        
        return {"success": True, "message": f"Found {len(results)} matches", "results": results}
    
    else:
        return {"error": f"Unknown operation: {operation}"} 