# Terminal AI Assistant

A command-line AI assistant with agentic capabilities, using LangChain/LangGraph for advanced orchestration and an MCP client interface for tool integration.

## Features
- Command-line interface for conversations and commands
- Agentic workflow orchestration using LangChain/LangGraph
- MCP client for easy tool/plugin integration
- Persistent memory using the MCP memory server (optional), enabling the assistant to remember user facts and context over time
- "Thinking..." animation while waiting for responses from reasoning models (e.g., o1, o4)
- Web search capability using Tavily Search API (default), Exa.ai API, and Brave Search API
- Dice rolling tool for simple probability simulation

## Getting Started
1. Install dependencies:
    - Python >= 3.9
    - pip

2. Install requirements:
    ```bash
    pip install -r requirements.txt
    ```

3. Configure the assistant:
    You have two options for configuration:
    
    **Option 1: Environment Variables**
    ```bash
    # Required for LLM access
    export OPENAI_API_KEY="your-openai-api-key"
    
    # Optional for web search functionality (at least one required for search)
    export EXA_API_KEY="your-exa-api-key"
    export BRAVE_SEARCH_API_KEY="your-brave-search-api-key"
    export TAVILY_API_KEY="your-tavily-api-key"
    ```
    
    **Option 2: Configuration File**
    Create or edit `config.yaml` in the project root directory:
    ```yaml
    # LLM Settings
    llm:
      base_url: "https://api.openai.com/v1"
      model: "gpt-4.1-mini"
      temperature: 0.7
      max_tokens: 1024
      max_completion_tokens: 2048  # For reasoning models
      api_key: "your-openai-api-key"
    
    # Exa Web Search Settings
    exa:
      api_key: "your-exa-api-key"
    
    # Brave Search Settings
    brave:
      api_key: "your-brave-search-api-key"
      search_kwargs:
        count: 5  # Number of results to return
        
    # Tavily Search Settings
    tavily:
      api_key: "your-tavily-api-key"
      search_kwargs:
        max_results: 5  # Number of results to return
        include_answer: true
        include_images: false
        include_raw_content: false
    
    # Debug Settings
    debug: false
    
    # Python REPL Settings
    python_autopilot: false
    ```

4. (Optional) Run the MCP Memory Server for persistent memory:
    **With Docker:**
    ```bash
    docker run -i -v claude-memory:/app/dist --rm mcp/memory
    ```
    **With npx:**
    ```bash
    npx -y @modelcontextprotocol/server-memory
    ```
    To use a custom memory storage file, set the MEMORY_FILE_PATH env var:
    ```bash
    MEMORY_FILE_PATH=/path/to/your/memory.json npx -y @modelcontextprotocol/server-memory
    ```
    For more info, see: https://github.com/modelcontextprotocol/servers/tree/main/src/memory

5. Run the assistant:
    ```bash
    python main.py
    ```

6. Run in debug mode to see API and processing information:
    ```bash
    python main.py --debug
    ```

7. Use a custom configuration file:
    ```bash
    python main.py --config /path/to/your/custom_config.yaml
    ```

## Folder Structure
- `main.py`: Entry point for the CLI app
- `src/`: Source code for CLI and agent
- `src/agentic_assistant.py`: Core assistant logic and LLM interaction
- `src/mcp_client.py`: Client for MCP integrations (tools, memory)
- `src/exa_client.py`: Client for Exa.ai web search API
- `src/brave_client.py`: Client for Brave Search API
- `src/tavily_client.py`: Client for Tavily Search API
- `.gitignore`: Standard Python git ignore file
- `requirements.txt`: Python dependencies
- `README.md`: This file

## Command Line Options
The assistant supports the following command-line options:

- `--debug`: Enable debug mode to see detailed processing information
- `--autopilot`: Enable autopilot mode to run Python code without approval
- `--config CONFIG_FILE`: Specify a path to a custom configuration file

## Configuration
The assistant can be configured using either the `config.yaml` file or environment variables:

### Config File (recommended)
Create a `config.yaml` file in the root directory with the following structure:
```yaml
# LLM Settings
llm:
  base_url: "https://api.openai.com/v1"
  model: "gpt-4.1-mini"
  temperature: 0.7
  max_tokens: 1024
  max_completion_tokens: 2048  # For reasoning models
  api_key: "your-openai-api-key"

# Exa Web Search Settings
exa:
  api_key: "your-exa-api-key"

# Brave Search Settings
brave:
  api_key: "your-brave-search-api-key"
  search_kwargs:
    count: 5  # Number of results to return

# Tavily Search Settings
tavily:
  api_key: "your-tavily-api-key"
  search_kwargs:
    max_results: 5  # Number of results to return
    include_answer: true
    include_images: false
    include_raw_content: false

# Debug Settings
debug: false

# Python REPL Settings
python_autopilot: false
```

### Environment Variables (alternative)
- `OPENAI_API_KEY`: Your OpenAI API key for LLM access (required)
- `LLM_MODEL`: The model to use (default: gpt-4.1-mini)
- `LLM_TEMPERATURE`: Temperature setting for LLM responses (default: 0.7)
- `LLM_MAX_TOKENS`: Maximum tokens for responses (default: 1024)
- `LLM_MAX_COMPLETION_TOKENS`: Maximum tokens for reasoning models (default: 2048)
- `EXA_API_KEY`: Your Exa.ai API key for web search (optional)
- `BRAVE_SEARCH_API_KEY`: Your Brave Search API key (optional)
- `TAVILY_API_KEY`: Your Tavily Search API key (optional)
- `DEBUG`: Set to "true" to enable debug output (or use --debug flag)

## Tools
The assistant comes with the following built-in tools:
1. **dice_roll**: Roll a dice with a specified number of sides
   - Example: "Roll a 20-sided die for me"

2. **tavily_search**: Search the web for information on a topic using Tavily Search with AI-generated summaries (default web search tool)
   - Example: "Search for current stock market trends"
   - Example: "Find the latest advancements in quantum computing"
   - Example: "Search only Wikipedia for information about the last Olympics"

3. **web_search**: Search the web for information on a topic using Exa.ai (alternative search tool)
   - Example: "Use Exa to search for the latest news about AI"
   - Example: "What's the weather in London using Exa?"

4. **brave_search**: Search the web for information on a topic using Brave Search (alternative search tool)
   - Example: "Use Brave to search for information about climate change"
   - Example: "Brave search for restaurants in Tokyo"

5. **memory_tool**: Store and retrieve persistent information about the user
   - Example: "Remember that I prefer dark chocolate over milk chocolate"
   - Example: "Remember my birthday is on June 15th"
   - Example: "What do you know about my food preferences?"
   - Operations:
     - **get**: Retrieve all stored memory
     - **add_fact**: Add a fact about the user
     - **add_preference**: Add a user preference
     - **add_date**: Add an important date
     - **search**: Search memory for specific information

## Roadmap
- Extensible tool/plugin system
- Advanced multi-turn conversation capability
- Support for additional external APIs
- Image generation capabilities
- Voice interaction mode