import os
from langchain.chains import ConversationalRetrievalChain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from openai import OpenAI
from datetime import datetime
from tzlocal import get_localzone
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from .mcp_client import MCPClient
from .config_util import load_config
from .memory_utils import memory_tool
import sys
import random
import time

# Import ExaClient conditionally
try:
    from .exa_client import ExaClient
    HAS_EXA_CLIENT = True
except (ImportError, ValueError):
    HAS_EXA_CLIENT = False

# Import BraveClient conditionally
try:
    from .brave_client import BraveClient
    HAS_BRAVE_CLIENT = True
except (ImportError, ValueError):
    HAS_BRAVE_CLIENT = False

# Import TavilyClient conditionally
try:
    from .tavily_client import TavilyClient
    HAS_TAVILY_CLIENT = True
except (ImportError, ValueError):
    HAS_TAVILY_CLIENT = False

# LangGraph and tools imports
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal, TypedDict, List, Dict, Any, Optional
import operator
import json
from langgraph.graph import END, StateGraph
from langchain_core.messages import AIMessage, HumanMessage
from langchain_experimental.utilities import PythonREPL

# Helper function to determine if a model is a reasoning model
def is_reasoning_model(model_name: str) -> bool:
    # OpenAI o-series, azure o3-mini, o4-mini, o1, o1-mini, etc
    model_lower = model_name.lower()
    return (
        model_lower.startswith("o3") or model_lower.startswith("o4") or model_lower.startswith("o1") or "o3-mini" in model_lower or "o4-mini" in model_lower or "o1-mini" in model_lower
    )

class AgenticAssistant:
    def __init__(self, python_autopilot=False, config_path=None, model_override=None, base_url_override=None):
        # Load configuration from config.yaml if exists
        self.config = load_config(config_path)
        
        self.mcp_client = MCPClient()
        # Get debug mode from configuration
        self.debug_mode = self.config["debug"]
        
        # Initialize ExaClient if available
        self.has_exa_search = False
        if HAS_EXA_CLIENT:
            try:
                self.exa_client = ExaClient(debug=self.debug_mode, config_path=config_path)
                self.has_exa_search = True
            except ValueError as e:
                if self.debug_mode:
                    print(f"[DEBUG] ExaClient initialization failed: {str(e)}")
                self.has_exa_search = False
        
        # Initialize BraveClient if available
        self.has_brave_search = False
        if HAS_BRAVE_CLIENT:
            try:
                self.brave_client = BraveClient(debug=self.debug_mode, config_path=config_path)
                self.has_brave_search = True
            except ValueError as e:
                if self.debug_mode:
                    print(f"[DEBUG] BraveClient initialization failed: {str(e)}")
                self.has_brave_search = False
        
        # Initialize TavilyClient if available
        self.has_tavily_search = False
        if HAS_TAVILY_CLIENT:
            try:
                self.tavily_client = TavilyClient(debug=self.debug_mode, config_path=config_path)
                self.has_tavily_search = True
            except ValueError as e:
                if self.debug_mode:
                    print(f"[DEBUG] TavilyClient initialization failed: {str(e)}")
                self.has_tavily_search = False
        
        # Check if any search tool is available
        self.has_web_search = self.has_exa_search or self.has_brave_search or self.has_tavily_search
        
        # Modern LangChain memory usage: use BaseChatMessageHistory instead of ConversationBufferMemory
        self.memory = ChatMessageHistory()
        # LLM configuration from config
        self.llm_base_url = base_url_override if base_url_override else self.config["llm"]["base_url"]
        self.llm_model = model_override if model_override else self.config["llm"]["model"]
        self.llm_api_key = self.config["llm"]["api_key"] or os.environ.get("OPENAI_API_KEY")
        self.llm_temperature = self.config["llm"]["temperature"]
        self.llm_max_tokens = self.config["llm"]["max_tokens"]
        
        # Set max_completion_tokens based on model type
        if is_reasoning_model(self.llm_model):
            self.llm_max_completion_tokens = self.config["llm"]["max_completion_tokens"]
        else:
            self.llm_max_completion_tokens = self.config["llm"]["max_tokens"]
            
        self.client = OpenAI(api_key=self.llm_api_key, base_url=self.llm_base_url)
        
        # Define tools
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "dice_roll",
                    "description": "Roll a dice with specified number of sides. Returns the result.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "sides": {
                                "type": "integer",
                                "description": "Number of sides on the dice",
                                "default": 6
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "python_repl",
                    "description": "A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with print(...). You can use this to run OS commands via subprocess or os modules, manipulate files, install packages, and perform various system operations.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to execute. Use print(...) to see output. Can include imports, OS commands via subprocess.run() or os.system(), file operations, etc."
                            }
                        },
                        "required": ["code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "memory_tool",
                    "description": "Store and retrieve information about the user in a persistent memory file. Use this to remember user facts, preferences, important dates, and world facts.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "description": "The operation to perform on memory: get, add_fact, add_preference, add_date, add_world_fact, search",
                                "enum": ["get", "add_fact", "add_preference", "add_date", "add_world_fact", "search"]
                            },
                            "fact": {
                                "type": "string",
                                "description": "A fact about the user to remember (for add_fact operation)"
                            },
                            "world_fact": {
                                "type": "string",
                                "description": "A fact about the world to remember (for add_world_fact operation)"
                            },
                            "category": {
                                "type": "string",
                                "description": "The category for the preference (for add_preference operation)"
                            },
                            "key": {
                                "type": "string",
                                "description": "The preference key (for add_preference operation)"
                            },
                            "value": {
                                "type": "string",
                                "description": "The preference value (for add_preference operation)"
                            },
                            "description": {
                                "type": "string",
                                "description": "Description of the important date (for add_date operation)"
                            },
                            "date": {
                                "type": "string",
                                "description": "The date string in any reasonable format (for add_date operation)"
                            },
                            "query": {
                                "type": "string",
                                "description": "The query to search for in the agent's memory (for search operation)"
                            }
                        },
                        "required": ["operation"]
                    }
                }
            }
        ]
        
        # Add tavily_search tool first if TavilyClient is available
        if self.has_tavily_search:
            self.tools.append({
                "type": "function",
                "function": {
                    "name": "tavily_search",
                    "description": "Search the web for information on a topic using Tavily Search. Preferred web search tool that returns relevant search results with AI-generated summaries.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Number of results to return (default: 5)",
                                "default": 5
                            },
                            "include_domains": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "description": "List of domains to include in the search results (e.g. ['wikipedia.org', 'cnn.com'])"
                            },
                            "exclude_domains": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "description": "List of domains to exclude from the search results"
                            }
                        },
                        "required": ["query"]
                    }
                }
            })
        
        # Add web_search tool if Exa is available
        if self.has_exa_search:
            self.tools.append({
                "type": "function",
                "function": {
                    "name": "exa_search",
                    "description": "Search the web for information on a topic using Exa. Returns relevant search results.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query"
                            },
                            "num_results": {
                                "type": "integer",
                                "description": "Number of results to return (default: 5)",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                }
            })
        
        # Add brave_search tool if BraveClient is available
        if self.has_brave_search:
            self.tools.append({
                "type": "function",
                "function": {
                    "name": "brave_search",
                    "description": "Search the web for information on a topic using Brave Search. Returns relevant search results.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query"
                            },
                            "count": {
                                "type": "integer",
                                "description": "Number of results to return (default: 5)",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                }
            })
        
        # Initialize LLM for the agent
        self.llm = ChatOpenAI(
            model=self.llm_model,
            temperature=self.llm_temperature,
            api_key=self.llm_api_key,
            base_url=self.llm_base_url,
            streaming=True
        )

        # Python REPL instance
        self.python_repl = PythonREPL()

        # Autopilot mode for Python REPL (no approval needed)
        self.python_autopilot = python_autopilot or self.config["python_autopilot"]

    def set_python_autopilot(self, enabled):
        """Enable or disable autopilot mode for Python REPL."""
        self.python_autopilot = enabled

    def add_memory_entity(self, name: str, entity_type: str, observations: list) -> str:
        entity = {"name": name, "entityType": entity_type, "observations": observations}
        return self.mcp_client.memory_create_entities([entity])

    def add_memory_observations(self, entity_name: str, new_obs: list) -> str:
        observation = {"entityName": entity_name, "contents": new_obs}
        return self.mcp_client.memory_add_observations([observation])

    def search_memory(self, query: str) -> str:
        return self.mcp_client.memory_search_nodes(query)
    
    def run(self, user_input: str):
        """Runs the assistant, yielding response chunks."""
        try:
            # Get current time and timezone
            try:
                now = datetime.now(get_localzone())
                current_time_str = now.strftime('%Y-%m-%d %H:%M:%S %Z%z')
            except Exception as tz_error:
                # Fallback if tzlocal fails
                now = datetime.now()
                current_time_str = now.strftime('%Y-%m-%d %H:%M:%S (local timezone unknown)')
                print(f"[Warning] Could not determine local timezone: {tz_error}", file=sys.stderr)
            
            # Format message history
            formatted_history = []
            for msg in self.memory.messages:
                if isinstance(msg, HumanMessage):
                    formatted_history.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    formatted_history.append({"role": "assistant", "content": msg.content})
            
            # Create system message with current time and tool capabilities
            tools_description = "- dice_roll: Roll a dice with a specified number of sides\n"
            tools_description += "- python_repl: Execute Python code including OS commands via subprocess/os modules, file operations, package installation, and various system tasks\n"
            tools_description += "- memory_tool: Store and retrieve persistent information about the user such as facts, preferences, important dates, and world facts\n"
            tools_description += "- tavily_search: Search the web for information using Tavily Search with AI-generated summaries (preferred web search tool)\n"
            if self.has_exa_search:
                tools_description += "- exa_search: Search the web for information using Exa\n"
            if self.has_brave_search:
                tools_description += "- brave_search: Search the web for information using Brave Search\n"
            
            system_message = {
                "role": "system", 
                "content": (
                    "You are a helpful and friendly personal AI assistant. "
                    "Be concise and informative in your responses. "
                    f"The current time is {current_time_str}.\n\n"
                    "You have access to the following tools:\n"
                    f"{tools_description}\n"
                    "INSTRUCTIONS:\n"
                    "- Always respond directly to the user's request\n"
                    "- Use tools only when needed\n"
                    "- Be concise in your responses\n"
                    "- When a tool is used, explain the results clearly\n"
                    "- When searching for information online, prefer using tavily_search as your default web search tool\n"
                    "- Only use other search tools like exa_search or brave_search if the user specifically requests them or if tavily_search is unavailable\n"
                    "- You can use the python_repl tool to execute Python code for various tasks\n"
                    "- Through Python, you can run OS commands using libraries like subprocess or os\n"
                    "- You can create, read, and manipulate files using Python's file handling capabilities\n"
                    "- You can install and use Python packages when needed\n"
                    "- When executing system commands, always use Python's subprocess or os modules via the python_repl tool\n"
                    "- Use the memory_tool to remember important facts about the user, their preferences, important dates, and general facts about the world\n"
                    "- Before asking for information the user has already shared, check the memory using memory_tool with 'search' operation"
                )
            }
            
            # Prepare the messages for the OpenAI client
            messages_to_send = [system_message] + formatted_history + [{"role": "user", "content": user_input}]
            
            # Create parameters for the API call
            params = {
                "model": self.llm_model,
                "messages": messages_to_send,
                "tools": self.tools,
                "stream": True  # Enable streaming
            }
            
            # Set appropriate parameters based on model type
            if is_reasoning_model(self.llm_model):
                params["max_completion_tokens"] = self.llm_max_completion_tokens
            else:
                params["temperature"] = self.llm_temperature
                params["max_tokens"] = self.llm_max_tokens
            
            # Start the streaming response
            full_response = ""
            current_tool_calls = []
            
            # Make the API call
            stream = self.client.chat.completions.create(**params)
            
            # Process the streaming response
            for chunk in stream:
                # Handle content chunks
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content
                
                # Handle tool calls in chunks
                if chunk.choices[0].delta.tool_calls:
                    for tc in chunk.choices[0].delta.tool_calls:
                        # Initialize new tool calls
                        if tc.index is not None:
                            idx = tc.index
                            while len(current_tool_calls) <= idx:
                                current_tool_calls.append({
                                    "id": "",
                                    "function": {"name": "", "arguments": ""}
                                })
                            
                            # Update the tool call ID
                            if tc.id:
                                current_tool_calls[idx]["id"] = tc.id
                            
                            # Update function information
                            if tc.function:
                                if tc.function.name:
                                    current_tool_calls[idx]["function"]["name"] += tc.function.name
                                if tc.function.arguments:
                                    current_tool_calls[idx]["function"]["arguments"] += tc.function.arguments
            
            # After receiving all chunks, process any tool calls
            for tool_call in current_tool_calls:
                function_name = tool_call["function"]["name"]
                
                if function_name == "dice_roll":
                    # Parse arguments
                    try:
                        args = json.loads(tool_call["function"]["arguments"])
                        sides = args.get("sides", 6)
                    except (json.JSONDecodeError, AttributeError):
                        sides = 6
                    
                    # Emit tool start marker
                    yield f"[TOOL_START]: dice_roll"
                    
                    # Roll the dice
                    result = random.randint(1, sides)
                    tool_result = f"Rolled a {result} on a {sides}-sided die."
                    
                    # Add a small delay to make the tool animation visible
                    time.sleep(1.5)
                    
                    # Emit tool end marker
                    yield f"[TOOL_END]"
                    
                    # Yield the tool result
                    yield f"\nTool result: {tool_result}\n"
                    full_response += f"\nTool result: {tool_result}\n"
                
                elif function_name == "python_repl":
                    # Parse arguments
                    try:
                        args = json.loads(tool_call["function"]["arguments"])
                        code = args.get("code", "")
                    except (json.JSONDecodeError, AttributeError):
                        code = ""
                    
                    # If autopilot mode is enabled, skip approval
                    if self.python_autopilot:
                        # Emit tool start marker
                        yield f"[TOOL_START]: python_repl"
                        # Run the code
                        try:
                            repl_result = self.python_repl.run(code)
                            tool_result = f"Python REPL output:\n{repl_result.strip()}"
                        except Exception as e:
                            tool_result = f"Python REPL error: {str(e)}"
                        time.sleep(1.5)
                        yield f"[TOOL_END]"
                        yield f"\nTool result: {tool_result}\n"
                        full_response += f"\nTool result: {tool_result}\n"
                    else:
                        # Normal approval flow
                        # Stop thinking animation before approval prompt (handled in CLI)
                        yield "__STOP_ANIMATION__"
                        # Ask user for approval BEFORE tool animation
                        print("\n" + "="*50)
                        print("[python_repl] The agent wants to run the following code:\n")
                        print(code)
                        print("\n" + "="*50)
                        approval = input("\033[93mApprove running this code? [Y/n]: \033[0m").strip().lower()
                        if approval not in ("", "y", "yes"):  # User did not approve
                            tool_result = "\033[91m[python_repl] Execution cancelled by user.\033[0m"
                            yield f"\n{tool_result}\n"
                            full_response += f"\n{tool_result}\n"
                        else:
                            # Emit tool start marker only if approved
                            yield f"[TOOL_START]: python_repl"
                            # Run the code
                            try:
                                repl_result = self.python_repl.run(code)
                                tool_result = f"Python REPL output:\n{repl_result.strip()}"
                            except Exception as e:
                                tool_result = f"Python REPL error: {str(e)}"
                            time.sleep(1.5)
                            yield f"[TOOL_END]"
                            yield f"\nTool result: {tool_result}\n"
                            full_response += f"\nTool result: {tool_result}\n"
                
                elif function_name == "exa_search" and self.has_exa_search:
                    # Parse arguments
                    try:
                        args = json.loads(tool_call["function"]["arguments"])
                        query = args.get("query", "")
                        num_results = args.get("num_results", 5)
                    except (json.JSONDecodeError, AttributeError):
                        query = "error parsing query"
                        num_results = 5
                    
                    # Emit tool start marker
                    yield f"[TOOL_START]: exa_search"
                    
                    # Perform web search
                    try:
                        search_results = self.exa_client.search(query, num_results=num_results)
                        formatted_results = self.exa_client.format_results(search_results)
                        tool_result = f"Exa web search results for '{query}':\n\n{formatted_results}"
                    except Exception as e:
                        tool_result = f"Error performing Exa web search: {str(e)}"
                    
                    # Add a small delay to make the tool animation visible
                    time.sleep(1.5)
                    
                    # Emit tool end marker
                    yield f"[TOOL_END]"
                    
                    # Yield the tool result
                    yield f"\nTool result: {tool_result}\n"
                    full_response += f"\nTool result: {tool_result}\n"
                
                elif function_name == "brave_search" and self.has_brave_search:
                    # Parse arguments
                    try:
                        args = json.loads(tool_call["function"]["arguments"])
                        query = args.get("query", "")
                        count = args.get("count", 5)
                    except (json.JSONDecodeError, AttributeError):
                        query = "error parsing query"
                        count = 5
                    
                    # Emit tool start marker
                    yield f"[TOOL_START]: brave_search"
                    
                    # Perform brave search
                    try:
                        search_results = self.brave_client.search(query, count=count)
                        formatted_results = self.brave_client.format_results(search_results)
                        tool_result = f"Brave Search results for '{query}':\n\n{formatted_results}"
                    except Exception as e:
                        tool_result = f"Error performing Brave Search: {str(e)}"
                    
                    # Add a small delay to make the tool animation visible
                    time.sleep(1.5)
                    
                    # Emit tool end marker
                    yield f"[TOOL_END]"
                    
                    # Yield the tool result
                    yield f"\nTool result: {tool_result}\n"
                    full_response += f"\nTool result: {tool_result}\n"
                
                elif function_name == "tavily_search" and self.has_tavily_search:
                    # Parse arguments
                    try:
                        args = json.loads(tool_call["function"]["arguments"])
                        query = args.get("query", "")
                        max_results = args.get("max_results", 5)
                        include_domains = args.get("include_domains", [])
                        exclude_domains = args.get("exclude_domains", [])
                    except (json.JSONDecodeError, AttributeError):
                        query = "error parsing query"
                        max_results = 5
                        include_domains = []
                        exclude_domains = []
                    
                    # Emit tool start marker
                    yield f"[TOOL_START]: tavily_search"
                    
                    # Perform tavily search
                    try:
                        search_results = self.tavily_client.search(query, max_results=max_results, include_domains=include_domains, exclude_domains=exclude_domains)
                        formatted_results = self.tavily_client.format_results(search_results)
                        tool_result = f"Tavily Search results for '{query}':\n\n{formatted_results}"
                    except Exception as e:
                        tool_result = f"Error performing Tavily Search: {str(e)}"
                    
                    # Add a small delay to make the tool animation visible
                    time.sleep(1.5)
                    
                    # Emit tool end marker
                    yield f"[TOOL_END]"
                    
                    # Yield the tool result
                    yield f"\nTool result: {tool_result}\n"
                    full_response += f"\nTool result: {tool_result}\n"
                
                elif function_name == "memory_tool":
                    # Parse arguments
                    try:
                        args = json.loads(tool_call["function"]["arguments"])
                        operation = args.get("operation", "")
                        
                        # Build kwargs based on the operation
                        kwargs = {}
                        if operation == "add_fact":
                            kwargs["fact"] = args.get("fact", "")
                        elif operation == "add_preference":
                            kwargs["category"] = args.get("category", "")
                            kwargs["key"] = args.get("key", "")
                            kwargs["value"] = args.get("value", "")
                        elif operation == "add_date":
                            kwargs["description"] = args.get("description", "")
                            kwargs["date"] = args.get("date", "")
                        elif operation == "add_world_fact":
                            kwargs["world_fact"] = args.get("world_fact", "")
                        elif operation == "search":
                            kwargs["query"] = args.get("query", "")
                    except (json.JSONDecodeError, AttributeError) as e:
                        operation = "error"
                        kwargs = {}
                        error_msg = str(e)
                    
                    # Emit tool start marker
                    yield f"[TOOL_START]: memory_tool ({operation})"
                    
                    # Call the memory tool
                    try:
                        result = memory_tool(operation, **kwargs)
                        
                        if operation == "get":
                            # Format the memory data for better readability
                            formatted_memory = json.dumps(result, indent=2)
                            tool_result = f"Current memory contents:\n\n{formatted_memory}"
                        elif operation == "search":
                            if "error" in result:
                                tool_result = f"Memory search error: {result['error']}"
                            else:
                                # Format search results
                                if not result.get("results"):
                                    tool_result = "No results found in memory for the query."
                                else:
                                    formatted_results = "\n".join([f"- {r_type}: {r_content}" for r_type, r_content in result.get("results", [])])
                                    tool_result = f"Memory search results:\n\n{formatted_results}"
                        else:
                            # For all other operations
                            if "error" in result:
                                tool_result = f"Memory operation error: {result['error']}"
                            else:
                                tool_result = result.get("message", "Operation completed.")
                    except Exception as e:
                        tool_result = f"Error executing memory tool: {str(e)}"
                    
                    # Add a small delay to make the tool animation visible
                    time.sleep(1.5)
                    
                    # Emit tool end marker
                    yield f"[TOOL_END]"
                    
                    # Yield the tool result
                    yield f"\nTool result: {tool_result}\n"
                    full_response += f"\nTool result: {tool_result}\n"
                
                # Create a follow-up with the tool result
                follow_up_messages = messages_to_send + [
                    {"role": "assistant", "content": full_response, "tool_calls": [
                        {"id": tool_call["id"], "type": "function", "function": {
                            "name": function_name, 
                            "arguments": tool_call["function"]["arguments"]
                        }}
                    ]},
                    {"role": "tool", "tool_call_id": tool_call["id"], "name": function_name, "content": tool_result}
                ]
                
                # Make a follow-up API call to let the model respond to the tool result
                follow_up_params = {
                    "model": self.llm_model,
                    "messages": follow_up_messages,
                    "stream": True
                }
                
                if is_reasoning_model(self.llm_model):
                    follow_up_params["max_completion_tokens"] = self.llm_max_completion_tokens
                else:
                    follow_up_params["temperature"] = self.llm_temperature
                    follow_up_params["max_tokens"] = self.llm_max_tokens
                
                # Stream the follow-up response
                follow_up_stream = self.client.chat.completions.create(**follow_up_params)
                for chunk in follow_up_stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        yield content
            
            # Add to memory for next conversation turn
            self.memory.add_user_message(user_input)
            self.memory.add_ai_message(full_response)
            
            # Add to MCP persistent memory (if used)
            self.add_memory_observations("user", [user_input])
            self.add_memory_observations("assistant", [full_response])
                
        except Exception as e:
            error_message = f"[Agent Error] {str(e)}"
            yield error_message
            self.memory.add_user_message(user_input)
            self.memory.add_ai_message(error_message)
            self.add_memory_observations("user", [user_input])
            self.add_memory_observations("assistant", [error_message])
