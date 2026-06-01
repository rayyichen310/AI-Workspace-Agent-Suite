import os
import sys
import logging
from dotenv import load_dotenv

def setup_environment(log_filename: str = "mcp_server.log") -> None:
    """
    Initialize shared environment settings, including loading dotenv and setting up Logging.
    """
    # 1. Load .env file and override system variables with values from .env
    load_dotenv(override=True)

    # 2. Set the logging level for langchain_google_genai to avoid excessive noise
    logging.getLogger("langchain_google_genai").setLevel(logging.ERROR)

    # 3. Create logs directory 
    os.makedirs("logs", exist_ok=True)

    # 4. Redirect standard error (stderr) to the log file
    log_path = os.path.join("logs", log_filename)
    log_file = open(log_path, "w", encoding="utf-8")
    os.dup2(log_file.fileno(), sys.stderr.fileno())

def get_mcp_config(agent_type: str) -> dict:
    base_env = {
        "GOOGLE_OAUTH_CLIENT_ID": os.environ.get("GOOGLE_OAUTH_CLIENT_ID", ""),
        "GOOGLE_OAUTH_CLIENT_SECRET": os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET", ""),
        "USER_GOOGLE_EMAIL": os.environ.get("USER_GOOGLE_EMAIL", ""),
        "LOG_LEVEL": "ERROR",
        "PYTHONWARNINGS": "ignore"
    }

    if agent_type == "refund":
        return {
            "workspace": {
                "command": "uvx",
                "args": [
                    "workspace-mcp", 
                    "--single-user", 
                    # "--tool-tier", "core",
                    "--permissions", "gmail:full"
                ],
                "transport": "stdio",
                "env": base_env
            }
        }
        
    elif agent_type == "calendar":
        calendar_env = base_env.copy()
        calendar_env.update({
            "OAUTHLIB_INSECURE_TRANSPORT": os.environ.get("OAUTHLIB_INSECURE_TRANSPORT", "1"),
            "WORKSPACE_MCP_HTTP_PORT": os.environ.get("WORKSPACE_MCP_HTTP_PORT", "8765")
        })
        
        return {
            "workspace": {
                "command": "workspace-mcp",
                "args": [
                    "--single-user", 
                    "--permissions", "calendar:full"
                ],
                "transport": "stdio",
                "env": calendar_env
            }
        }
        
    else:
        raise ValueError(f"Unsupported agent_type: {agent_type}. Please use 'refund' or 'calendar'.")
