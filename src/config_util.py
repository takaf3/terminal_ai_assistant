import os
import yaml
from typing import Dict, Any, Optional


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from config.yaml file if it exists, otherwise use environment variables.
    
    Args:
        config_path: Optional path to a custom config file. If not provided, uses './config.yaml'.
    
    Returns:
        Dict containing all configuration parameters with their values
    """
    config = {
        "llm": {
            "base_url": os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
            "model": os.environ.get("LLM_MODEL", "gpt-4.1-mini"),
            "api_key": os.environ.get("OPENAI_API_KEY", ""),
            "temperature": float(os.environ.get("LLM_TEMPERATURE", 0.7)),
            "max_tokens": int(os.environ.get("LLM_MAX_TOKENS", 1024)),
            "max_completion_tokens": int(os.environ.get("LLM_MAX_COMPLETION_TOKENS", 2048)),
        },
        "exa": {
            "api_key": os.environ.get("EXA_API_KEY", ""),
        },
        "debug": os.environ.get("DEBUG", "false").lower() in ("true", "1", "t", "yes"),
        "python_autopilot": False,
    }
    
    # Use specified config file path or default to config.yaml
    config_file = config_path or "config.yaml"
    
    # Check if config file exists
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                yaml_config = yaml.safe_load(f)
                
            # Update config with values from YAML
            if yaml_config:
                merge_configs(config, yaml_config)
                
            # Set environment variables for backward compatibility
            _set_env_vars_from_config(config)
            
            if config_path:  # Only print if custom path is used
                print(f"Configuration loaded from: {config_file}")
                
        except Exception as e:
            print(f"[Warning] Error loading config file '{config_file}': {str(e)}")
    elif config_path:  # Only warn if custom path was specified but doesn't exist
        print(f"[Warning] Config file not found: {config_file}. Using default settings.")
    
    return config


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> None:
    """
    Recursively merge override_config into base_config.
    
    Args:
        base_config: Base configuration dictionary to be updated
        override_config: Override configuration dictionary with new values
    """
    for key, value in override_config.items():
        if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
            merge_configs(base_config[key], value)
        else:
            # Only update if value is not None and not an empty string
            if value is not None and (not isinstance(value, str) or value.strip()):
                base_config[key] = value


def _set_env_vars_from_config(config: Dict[str, Any]) -> None:
    """
    Set environment variables from config for backwards compatibility.
    
    Args:
        config: Loaded configuration dictionary
    """
    # LLM settings
    if config.get("llm", {}).get("base_url"):
        os.environ["LLM_BASE_URL"] = str(config["llm"]["base_url"])
    
    if config.get("llm", {}).get("model"):
        os.environ["LLM_MODEL"] = str(config["llm"]["model"])
    
    if config.get("llm", {}).get("api_key"):
        os.environ["OPENAI_API_KEY"] = str(config["llm"]["api_key"])
    
    if config.get("llm", {}).get("temperature") is not None:
        os.environ["LLM_TEMPERATURE"] = str(config["llm"]["temperature"])
    
    if config.get("llm", {}).get("max_tokens") is not None:
        os.environ["LLM_MAX_TOKENS"] = str(config["llm"]["max_tokens"])
    
    if config.get("llm", {}).get("max_completion_tokens") is not None:
        os.environ["LLM_MAX_COMPLETION_TOKENS"] = str(config["llm"]["max_completion_tokens"])
    
    # Exa settings
    if config.get("exa", {}).get("api_key"):
        os.environ["EXA_API_KEY"] = str(config["exa"]["api_key"])
    
    # Debug setting
    if config.get("debug") is not None:
        os.environ["DEBUG"] = "true" if config["debug"] else "false" 