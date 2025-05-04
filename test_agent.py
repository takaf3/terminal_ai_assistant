import os
import sys
import threading
import time
from termcolor import cprint
from src.agentic_assistant import AgenticAssistant

# Animation stop event
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

def main():
    # Initialize the agent
    assistant = AgenticAssistant()
    
    # Test the agent with a query that should use the dice_roll tool
    test_query = "Roll a 20-sided die for me"
    
    print(f"Testing agent with query: {test_query}")
    print("Response:")
    
    # Start thinking animation
    stop_animation_event.clear()
    animation_thread = threading.Thread(target=_thinking_animation, args=(stop_animation_event,))
    animation_thread.start()
    
    # Variables to track tool usage
    first_chunk_received = False
    in_tool_execution = False
    tool_animation_thread = None
    current_tool_name = None
    
    try:
        # Process the response chunks
        for chunk in assistant.run(test_query):
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
        
        print("\n\nTest completed.")
    
    except Exception as e:
        # Ensure animations stop if there's an error
        if animation_thread and animation_thread.is_alive():
            stop_animation_event.set()
            animation_thread.join()
        
        if tool_animation_thread and tool_animation_thread.is_alive():
            stop_animation_event.set()
            tool_animation_thread.join()
        
        print(f"\n\nError: {e}")
        print("Test failed.")

if __name__ == "__main__":
    main() 