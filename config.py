import os
from pathlib import Path
from urllib.parse import urlsplit

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

# Section 2: the LLM Gateway is always at <hostname>/llm, on the model endpoint's host
_base = urlsplit(KLIQUE_BASE_URL)
KLIQUE_LLM_GATEWAY_URL = f"{_base.scheme}://{_base.netloc}/llm"

# Section 2.10 (optional): the Klique API server and an access key, used to list a profile's models.
# Notebooks that need them check for None and fall back to a hand-typed list.
KLIQUE_API_SERVER = (os.environ.get("KLIQUE_API_SERVER") or "").rstrip("/") or None
KLIQUE_ACCESS_KEY = os.environ.get("KLIQUE_ACCESS_KEY") or None
KLIQUE_SECRET_KEY = os.environ.get("KLIQUE_SECRET_KEY") or None
