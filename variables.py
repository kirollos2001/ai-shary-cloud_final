# variables.py (hardened)
import os
import sys

def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)

def _get_str_env(name: str, default: str | None = None, allow_empty: bool = False) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return default
    if not allow_empty and raw.strip() == "":
        return default
    return raw

def _get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        _warn(f"Invalid int for {name}={raw!r}, falling back to {default}")
        return default

def _get_bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "t", "yes", "y", "on")

def _get_hour_env(name: str, default: int) -> int:
    v = _get_int_env(name, default)
    if v < 0 or v > 23:
        _warn(f"{name} out of range 0-23: {v}, clamping")
        v = max(0, min(23, v))
    return v

# Gemini
GEMINI_API_KEY = _get_str_env("GEMINI_API_KEY")
GEMINI_MODEL_NAME = "gemini-2.5-flash"

# Database
DB_HOST = _get_str_env("DB_HOST", "77.37.35.26")
DB_PORT = _get_int_env("DB_PORT", 3306)
DB_NAME = _get_str_env("DB_NAME")
DB_USER = _get_str_env("DB_USER")
DB_PASSWORD = _get_str_env("DB_PASSWORD")

# Email
EMAIL_HOST = _get_str_env("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT = _get_int_env("EMAIL_PORT", 587)
EMAIL_USER = _get_str_env("EMAIL_USER")
EMAIL_PASSWORD = _get_str_env("EMAIL_PASSWORD")
TEAM_EMAIL = _get_str_env("TEAM_EMAIL")

# API
USER_INFO_API_URL = _get_str_env("USER_INFO_API_URL", "https://shary.eg/api/UserInfo")

# Cache
CACHE_DIR = "cache/"
LEADS_CACHE_FILE = "leads_cache.json"
CONVERSATIONS_CACHE_FILE = "conversations_cache.json"
UNITS_CACHE_FILE = "units.json"
NEW_LAUNCHES_CACHE_FILE = "new_launches.json"
DEVELOPERS_CACHE_FILE = "developers.json"

# App
APP_PORT = _get_int_env("APP_PORT", 8080)
APP_HOST = _get_str_env("APP_HOST", "0.0.0.0")
DEBUG_MODE = _get_bool_env("DEBUG_MODE", False)

# Scheduler
CACHE_SYNC_HOUR = _get_hour_env("CACHE_SYNC_HOUR", 4)
DB_SYNC_HOUR = _get_hour_env("DB_SYNC_HOUR", 3)

# Session
SESSION_TIMEOUT = _get_int_env("SESSION_TIMEOUT", 3600)

# Logging
LOG_LEVEL = _get_str_env("LOG_LEVEL", "INFO")


for name, val in [("DB_NAME", DB_NAME), ("DB_USER", DB_USER), ("DB_PASSWORD", DB_PASSWORD)]:
    if not val:
        _warn(f"{name} is not set; DB connections may fail later.")
