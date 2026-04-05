# server/settings.py
from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import logging, os, sys

_log = logging.getLogger("chatdft")

# detect paths
HERE = Path(__file__).resolve().parent                      # .../server
ROOT = HERE.parent                                          # repo root
CWD  = Path(os.getcwd()).resolve()

# candidate .env locations (first existing wins)
CANDIDATES = [
    ROOT / ".env",
    HERE / ".env",
    CWD / ".env",
]

ENV_FILE = next((p for p in CANDIDATES if p.is_file()), None)

if ENV_FILE is None:
    _log.info(".env not found — using exported environment variables")
else:
    _log.info("Loading .env: %s", ENV_FILE)

class Settings(BaseSettings):
    # REQUIRED
    DATABASE_URL: str
    HPC_HOST: str
    HPC_USER: str
    REMOTE_BASE: str

    # OPTIONAL/DEFAULTS
    LOCAL_RUNS: str = "./runs"
    SYNC_INTERVAL_SEC: int = 20
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # pydantic-settings v2
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE) if ENV_FILE else None,
        env_file_encoding="utf-8",
        extra="ignore",
    )

settings = Settings()

# debug print (DATABASE_URL omitted for brevity)
_printable = {k: v for k, v in settings.model_dump().items() if k != "DATABASE_URL"}
_log.info("Settings loaded: %s", _printable)

import yaml, pathlib
SERVERS = {}
p = pathlib.Path(__file__).parent / "servers.yaml"
if p.exists():
    SERVERS = yaml.safe_load(p.read_text())["servers"]
def get_server(name:str): return SERVERS[name]