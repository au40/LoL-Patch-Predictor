"""Central configuration + paths. Loads secrets from .env if present."""
from pathlib import Path
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv is optional; env vars can also be set in the shell.
    pass

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

RIOT_API_KEY = os.getenv("RIOT_API_KEY", "")
RIOT_REGION = os.getenv("RIOT_REGION", "americas")
RIOT_PLATFORM = os.getenv("RIOT_PLATFORM", "na1")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
