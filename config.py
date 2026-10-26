import os
from pathlib import Path

# Load .env from the project root
env_path = Path(__file__).parent / ".env"
if not env_path.exists():
    raise FileNotFoundError(f"No .env file at {env_path}. Create it in Section 1.6.")
with open(env_path) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()

# Expose as attributes
KLIQUE_BASE_URL = os.environ["KLIQUE_BASE_URL"].rstrip("/")
KLIQUE_MODEL = os.environ["KLIQUE_MODEL"]
KLIQUE_GATEWAY_TOKEN = os.environ["KLIQUE_GATEWAY_TOKEN"]
