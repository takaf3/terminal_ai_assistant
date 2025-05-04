import os
import sys
import threading
import time
import argparse
from termcolor import cprint, colored
from prompt_toolkit import prompt
from prompt_toolkit.styles import Style
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_message_histories import ChatMessageHistory
from src.config_util import load_config


def is_reasoning_model(model_name: str) -> bool:
    # OpenAI o-series, azure o3-mini, o4-mini, o1, o1-mini, etc
    model_lower = model_name.lower()
    return (
        model_lower.startswith("o3") or model_lower.startswith("o4") or model_lower.startswith("o1") or "o3-mini" in model_lower or "o4-mini" in model_lower or "o1-mini" in model_lower
    )


stop_animation_event = threading.Event()


def _thinking_animation(stop_event):
    """Runs the thinking animation in a separate thread."""
    chars = [" .  ", " .. ", " ..."]
    idx = 0
    while not stop_event.wait(0.3): # Check event every 0.3 seconds
        try:
            text = f"Thinking{chars[idx % len(chars)]}"
            sys.stdout.write('\r' + text)
            sys.stdout.flush()
            idx += 1
        except OSError: # Handle potential race condition on exit
            break
    # Clear the line when done
    sys.stdout.write('\r' + ' ' * (len("Thinking   ") + 5) + '\r')
    sys.stdout.flush()


def _tool_animation(stop_event, tool_name):
    """Runs the tool usage animation in a separate thread."""
    chars = ["⚙️ ", "🔧 ", "🛠️ "]
    idx = 0
    while not stop_event.wait(0.4): # Check event every 0.4 seconds
        try:
            text = f"{chars[idx % len(chars)]}Using tool: {tool_name}{' ' * 10}"
            sys.stdout.write('\r' + text)
            sys.stdout.flush()
            idx += 1
        except OSError: # Handle potential race condition on exit
            break
    # Clear the line when done
    sys.stdout.write('\r' + ' ' * (len(f"Using tool: {tool_name}") + 20) + '\r')
    sys.stdout.flush()


def print_welcome():
    cprint("\n=== Terminal AI Assistant ===", "cyan", attrs=["bold"])
    cprint("Type your query or 'exit' to quit.", "green")


def check_required_env_vars():
    """Check for required environment variables and display warnings."""
    missing_vars = []
    
    if not os.environ.get("OPENAI_API_KEY"):
        missing_vars.append("OPENAI_API_KEY")
    
    # Check for web search API keys and warn if not available
    has_search_apis = False
    
    if not os.environ.get("EXA_API_KEY"):
        cprint("\nINFO: EXA_API_KEY environment variable is not set.", "yellow")
        cprint("Exa web search functionality will not be available.", "yellow")
    else:
        has_search_apis = True
    
    if not os.environ.get("BRAVE_SEARCH_API_KEY"):
        cprint("\nINFO: BRAVE_SEARCH_API_KEY environment variable is not set.", "yellow")
        cprint("Brave search functionality will not be available.", "yellow")
    else:
        has_search_apis = True
        
    if not has_search_apis:
        cprint("\nWARNING: No search API keys are set. Web search functionality will be unavailable.", "yellow")
        cprint("Set either EXA_API_KEY or BRAVE_SEARCH_API_KEY environment variable to enable web search.", "yellow")
    
    if missing_vars:
        cprint(f"\nERROR: The following required environment variables are missing: {', '.join(missing_vars)}", "red")
        cprint("Please set these environment variables before running the assistant.", "red")
        sys.exit(1)


def main():
    # Parse command-line arguments first
    parser = argparse.ArgumentParser(description="Terminal AI Assistant")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--autopilot", action="store_true", help="Enable autopilot mode (runs Python code without approval)")
    parser.add_argument("--config", metavar="CONFIG_FILE", help="Path to custom config file (default: ./config.yaml)")
    parser.add_argument("-m", "--model", help="Model ID to use (overrides config file and environment variables)")
    parser.add_argument("-b", "--base-url", help="API base URL to use (overrides config file and environment variables)")
    args = parser.parse_args()
    
    # Load configuration from specified or default config file
    config = load_config(args.config)
    
    # Set debug environment variable if debug flag is enabled
    if args.debug or config["debug"]:
        os.environ["DEBUG"] = "true"
        cprint("Debug mode enabled.", "yellow")

    # Display model override notification if specified
    if args.model:
        cprint(f"Model override: Using model '{args.model}'", "yellow")

    # Display base URL override notification if specified
    if args.base_url:
        cprint(f"Base URL override: Using API base URL '{args.base_url}'", "yellow")

    # Initialize autopilot mode from command line argument or config
    autopilot_mode = args.autopilot or config["python_autopilot"]
    if autopilot_mode:
        cprint("Autopilot mode enabled. Python code will run without approval.", "yellow")
        cprint("Type 'autopilot off' to disable.", "yellow")
    
    # Check for required environment variables
    check_required_env_vars()
    
    print_welcome()
    
    try:
        from src.agentic_assistant import AgenticAssistant
        assistant = AgenticAssistant(python_autopilot=autopilot_mode, config_path=args.config, 
                                    model_override=args.model, base_url_override=args.base_url)
    except ValueError as e:
        cprint(f"\nERROR: {str(e)}", "red")
        sys.exit(1)
    
    animation_thread = None

    while True:
        try:
            pretty_prompt = [("class:prompt", "👤 > ")]
            prompt_style = Style.from_dict({"prompt": ""})
            user_input = prompt(pretty_prompt, style=prompt_style)
        except (EOFError, KeyboardInterrupt):
            if animation_thread and animation_thread.is_alive():
                stop_animation_event.set()
                animation_thread.join()
            cprint("\nGoodbye!", "yellow")
            break
        if user_input.strip().lower() in ["exit", "quit"]:
            if animation_thread and animation_thread.is_alive():
                stop_animation_event.set()
                animation_thread.join()
            cprint("Exiting. See you next time!", "yellow")
            break
        
        # Handle autopilot toggle commands
        autopilot_cmd = user_input.strip().lower()
        if autopilot_cmd in ["autopilot on", "autopilot true", "autopilot enable"]:
            autopilot_mode = True
            assistant.set_python_autopilot(True)
            cprint("Autopilot mode enabled. Python code will run without approval.", "yellow")
            continue
        elif autopilot_cmd in ["autopilot off", "autopilot false", "autopilot disable"]:
            autopilot_mode = False
            assistant.set_python_autopilot(False)
            cprint("Autopilot mode disabled. Python code requires approval.", "yellow")
            continue
        elif autopilot_cmd in ["autopilot", "autopilot status"]:
            status = "enabled" if autopilot_mode else "disabled"
            cprint(f"Autopilot mode is currently {status}.", "yellow")
            continue

        # Always show thinking animation for any model
        stop_animation_event.clear()
        animation_thread = threading.Thread(target=_thinking_animation, args=(stop_animation_event,))
        animation_thread.start()

        # Run the assistant and process the stream
        first_chunk_received = False
        in_tool_execution = False
        tool_animation_thread = None
        current_tool_name = None
        
        try:
            for chunk in assistant.run(user_input):
                # Special marker to stop animation without showing to user
                if chunk == "__STOP_ANIMATION__":
                    # Stop thinking animation if it's running
                    if animation_thread and animation_thread.is_alive():
                        stop_animation_event.set()
                        animation_thread.join()
                        animation_thread = None
                    continue  # Skip printing this marker
                
                # Check if this is a tool execution marker
                if chunk.startswith("[TOOL_START]"):
                    # Extract the tool name
                    tool_parts = chunk.split(":")
                    if len(tool_parts) > 1:
                        current_tool_name = tool_parts[1].strip()
                    else:
                        current_tool_name = "Unknown Tool"
                    
                    # Stop thinking animation if it's running
                    if animation_thread and animation_thread.is_alive():
                        stop_animation_event.set()
                        animation_thread.join()
                        animation_thread = None
                    
                    # Start tool animation
                    stop_animation_event.clear()
                    tool_animation_thread = threading.Thread(target=_tool_animation, args=(stop_animation_event, current_tool_name))
                    tool_animation_thread.start()
                    in_tool_execution = True
                    continue  # Skip printing this marker
                
                # Check if this is the end of tool execution
                if chunk.startswith("[TOOL_END]"):
                    # Stop tool animation
                    if tool_animation_thread and tool_animation_thread.is_alive():
                        stop_animation_event.set()
                        tool_animation_thread.join()
                        tool_animation_thread = None
                    in_tool_execution = False
                    continue  # Skip printing this marker
                
                # Stop animation on first content chunk (if not in tool execution)
                if (not first_chunk_received and not in_tool_execution and 
                    animation_thread and animation_thread.is_alive()):
                    stop_animation_event.set()
                    animation_thread.join()
                    animation_thread = None
                    first_chunk_received = True
                    # Add robot emoji prefix for first chunk of agent response
                    print("🤖 > ", end='', flush=True)
                
                # Print the chunk content
                if chunk.startswith("\nTool result:"):
                    # Stop tool animation if it's still running
                    if tool_animation_thread and tool_animation_thread.is_alive():
                        stop_animation_event.set()
                        tool_animation_thread.join()
                        tool_animation_thread = None
                    # Print tool result with color
                    cprint(chunk, "cyan")
                else:
                    # Print normal content
                    print(chunk, end='', flush=True)
            
            print()  # Add a newline after the stream finishes
        
        except Exception as e:
            # Ensure animations stop even if run() fails
            if animation_thread and animation_thread.is_alive():
                stop_animation_event.set()
                animation_thread.join()
                animation_thread = None
            
            if tool_animation_thread and tool_animation_thread.is_alive():
                stop_animation_event.set()
                tool_animation_thread.join()
                tool_animation_thread = None
            
            cprint(f"\n[Error during processing stream] {e}", "red")

        # Ensure all animation threads are definitely stopped
        if animation_thread and animation_thread.is_alive():
            stop_animation_event.set()
            animation_thread.join()
            animation_thread = None
        
        if tool_animation_thread and tool_animation_thread.is_alive():
            stop_animation_event.set()
            tool_animation_thread.join()
            tool_animation_thread = None


if __name__ == "__main__":
    main()