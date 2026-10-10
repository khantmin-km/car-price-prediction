import os

def _env_flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).lower() == "true"

HOST = os.environ.get("HOST", "0.0.0.0") 
PORT = int(os.environ.get("PORT", 8050))  
DEBUG = _env_flag("DEBUG", "false")  
DEMO_MODE = _env_flag("DEMO_MODE", "false")  
