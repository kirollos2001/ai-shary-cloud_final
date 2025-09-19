# variables.py (hardened + Secret Manager support)
import os
import sys
import json
import base64
import logging
import tempfile
from typing import Optional, Dict, Any, List

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

# -----------------------------------------------------------------------------
# Secret Manager helpers
# -----------------------------------------------------------------------------
_SM_CLIENT = None
_SM_CACHE: Dict[str, str] = {}

def _resolve_project_id() -> Optional[str]:
    # Priority: explicit env -> common GCP envs -> SA JSON (if present)
    for key in ("GOOGLE_PROJECT_ID", "GOOGLE_CLOUD_PROJECT", "GCP_PROJECT", "PROJECT_ID"):
        v = _get_str_env(key)
        if v:
            return v
    # If we already have SA JSON loaded below, we will read it there.
    return None

def _load_sa_json_from_env() -> Optional[Dict[str, Any]]:
    """Load service account JSON from:
       - GOOGLE_CLOUD_CREDENTIALS_JSON (raw JSON)
       - GOOGLE_CLOUD_CREDENTIALS_B64 (base64 JSON)
       - GOOGLE_CLOUD_CREDENTIALS_FILE (filepath)
       - GOOGLE_CLOUD_CREDENTIALS_SECRET (secret name in Secret Manager)
    """
    # 1) Inline JSON
    raw_json = _get_str_env("GOOGLE_CLOUD_CREDENTIALS_JSON")
    if raw_json:
        try:
            return json.loads(raw_json)
        except Exception:
            _warn("Invalid GOOGLE_CLOUD_CREDENTIALS_JSON; JSON parse failed")

    # 2) Base64 JSON
    b64 = _get_str_env("GOOGLE_CLOUD_CREDENTIALS_B64")
    if b64:
        try:
            decoded = base64.b64decode(b64)
            return json.loads(decoded.decode("utf-8"))
        except Exception:
            _warn("Invalid GOOGLE_CLOUD_CREDENTIALS_B64; base64/JSON parse failed")

    # 3) File path
    fpath = _get_str_env("GOOGLE_CLOUD_CREDENTIALS_FILE")
    if fpath and os.path.exists(fpath):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            _warn(f"Failed reading GOOGLE_CLOUD_CREDENTIALS_FILE at {fpath!r}")

    # 4) Secret Manager (secret contains full SA JSON)
    secret_name = _get_str_env("GOOGLE_CLOUD_CREDENTIALS_SECRET")
    if secret_name:
        txt = _fetch_secret(secret_name)  # may return None if not accessible
        if txt:
            try:
                return json.loads(txt)
            except Exception:
                _warn("GOOGLE_CLOUD_CREDENTIALS_SECRET payload is not valid JSON")

    return None

def _secretmanager_client():
    """Create a Secret Manager client using ADC if available, otherwise try SA JSON from env."""
    global _SM_CLIENT
    if _SM_CLIENT is not None:
        return _SM_CLIENT

    try:
        from google.cloud import secretmanager  # type: ignore
    except Exception as e:
        _warn(f"google-cloud-secret-manager not installed: {e}")
        return None

    creds = None
    # Prefer ADC on Cloud Run (attached service account)
    try:
        import google.auth  # type: ignore
        creds, _ = google.auth.default()
    except Exception:
        creds = None

    if creds is None:
        # Try explicit SA JSON from env/file/secret
        sa_info = _load_sa_json_from_env()
        if sa_info:
            try:
                from google.oauth2 import service_account  # type: ignore
                creds = service_account.Credentials.from_service_account_info(sa_info)
            except Exception as e:
                _warn(f"Failed building SA credentials from JSON: {e}")

    try:
        _SM_CLIENT = secretmanager.SecretManagerServiceClient(credentials=creds) if creds \
                     else secretmanager.SecretManagerServiceClient()
        return _SM_CLIENT
    except Exception as e:
        _warn(f"Failed to initialize Secret Manager client: {e}")
        return None

def _fetch_secret(secret_name: str, project_id: Optional[str] = None, version: str = "latest") -> Optional[str]:
    """Fetch secret value (text) from Secret Manager. Supports:
       - Full resource path (projects/.../secrets/.../versions/...)
       - Bare secret ID (requires project_id)
    """
    if not secret_name:
        return None

    # Cache by full key (name@version@project)
    cache_key = f"{project_id or ''}@{secret_name}@{version}"
    if cache_key in _SM_CACHE:
        return _SM_CACHE[cache_key]

    client = _secretmanager_client()
    if client is None:
        return None

    # Build name
    if secret_name.startswith("projects/"):
        resource = secret_name
        # If user passed projects/.../secrets/... only, add /versions/latest
        if "/versions/" not in resource:
            resource = f"{resource}/versions/{version}"
    else:
        pid = project_id or _resolve_project_id()
        if pid is None:
            # Try to derive project from SA JSON as last resort
            sa = _load_sa_json_from_env()
            if isinstance(sa, dict):
                pid = sa.get("project_id")
        if not pid:
            _warn("No project_id available to access Secret Manager (set GOOGLE_PROJECT_ID).")
            return None
        resource = f"projects/{pid}/secrets/{secret_name}/versions/{version}"

    try:
        resp = client.access_secret_version(name=resource)  # type: ignore
        payload = resp.payload.data.decode("utf-8")  # type: ignore
        _SM_CACHE[cache_key] = payload
        return payload
    except Exception as e:
        _warn(f"Failed to access secret {secret_name!r}: {e}")
        return None

def _get_secret_or_env(value_env: str, secret_env: str | None = None, default: str | None = None) -> Optional[str]:
    """Read plaintext value from ENV first, else from Secret Manager using secret name in secret_env.
       If secret_env is None, it will try f"{value_env}_SECRET".
    """
    v = _get_str_env(value_env)
    if v:
        return v
    sname = _get_str_env(secret_env or f"{value_env}_SECRET")
    if sname:
        val = _fetch_secret(sname)
        if val is not None:
            logging.warning(f"{value_env} loaded from Secret Manager secret {sname}")
            return val
        logging.warning(f"Failed to load {value_env} from Secret Manager secret {sname}")
    return default

# -----------------------------------------------------------------------------
# Google Service Account (exposed for code that needs Credentials)
# -----------------------------------------------------------------------------
GOOGLE_CLOUD_CREDENTIALS: Optional[Dict[str, Any]] = _load_sa_json_from_env()

# Derive project id as best-effort
GOOGLE_PROJECT_ID = (
    _get_str_env("GOOGLE_PROJECT_ID")
    or _get_str_env("GOOGLE_CLOUD_PROJECT")
    or _get_str_env("GCP_PROJECT")
    or (GOOGLE_CLOUD_CREDENTIALS.get("project_id") if isinstance(GOOGLE_CLOUD_CREDENTIALS, dict) else None)
)

# Backwards compatibility: some modules expect GOOGLE_CLOUD_PROJECT
# Expose it as an alias to the resolved project id above
GOOGLE_CLOUD_PROJECT = GOOGLE_PROJECT_ID

# Speech
SPEECH_LANGUAGE = _get_str_env("SPEECH_LANGUAGE") or _get_str_env("SPEECH_TO_TEXT_LANGUAGE", "ar-EG")

def get_google_sa_credentials(scopes: Optional[List[str]] = None):
    """Return google.oauth2.service_account.Credentials if SA JSON is available.
       Otherwise try ADC (useful on Cloud Run with attached service account).
       Returns None if nothing configured.
    """
    scopes = scopes or []
    try:
        if GOOGLE_CLOUD_CREDENTIALS:
            from google.oauth2 import service_account  # type: ignore
            return service_account.Credentials.from_service_account_info(
                GOOGLE_CLOUD_CREDENTIALS, scopes=scopes
            )
        # Fallback to ADC (Cloud Run recommended path)
        import google.auth  # type: ignore
        creds, _ = google.auth.default(scopes=scopes if scopes else None)
        return creds
    except Exception as e:
        _warn(f"Google credentials unavailable: {e}")
        return None

# -----------------------------------------------------------------------------
# App configuration (read ENV first, then Secret Manager for sensitive values)
# Set *_SECRET envs to the Secret Manager names (or full resource paths).
# -----------------------------------------------------------------------------

# Gemini
# - Either set GEMINI_API_KEY directly, or set GEMINI_API_KEY_SECRET with your secret name.
GEMINI_API_KEY = _get_secret_or_env("GEMINI_API_KEY")
GEMINI_MODEL_NAME = "gemini-2.5-flash"

# Database (recommend putting these in secrets)
DB_HOST = _get_str_env("DB_HOST", "77.37.35.26")
DB_PORT = _get_int_env("DB_PORT", 3306)
_DB_ENV_RAW = {n: _get_str_env(n) for n in ("DB_NAME", "DB_USER", "DB_PASSWORD")}
DB_NAME = _get_secret_or_env("DB_NAME")  # prefers ENV DB_NAME else DB_NAME_SECRET
DB_USER = _get_secret_or_env("DB_USER")
DB_PASSWORD = _get_secret_or_env("DB_PASSWORD")

for n, env_val, val in [
    ("DB_NAME", _DB_ENV_RAW["DB_NAME"], DB_NAME),
    ("DB_USER", _DB_ENV_RAW["DB_USER"], DB_USER),
    ("DB_PASSWORD", _DB_ENV_RAW["DB_PASSWORD"], DB_PASSWORD),
]:
    if val and env_val is None and _get_str_env(f"{n}_SECRET"):
        logging.warning(f"{n} retrieved from Secret Manager")
    if not val:
        _warn(f"{n} is not set; DB connections may fail later.")
# Email (also good candidates for secrets)
EMAIL_HOST = _get_str_env("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT = _get_int_env("EMAIL_PORT", 587)
EMAIL_USER = _get_secret_or_env("EMAIL_USER")
EMAIL_PASSWORD = _get_secret_or_env("EMAIL_PASSWORD")
TEAM_EMAIL = _get_str_env("TEAM_EMAIL")

# API
USER_INFO_API_URL = _get_str_env("USER_INFO_API_URL", "https://shary.eg/api/UserInfo")

# Cache
CACHE_DIR = os.path.join(tempfile.gettempdir(), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ChromaDB persistence directory
# Allow overriding via env so deployments can point at a mounted GCS bucket.
_CHROMA_DIR_OVERRIDE = _get_str_env("CHROMA_PERSIST_DIR")
if _CHROMA_DIR_OVERRIDE:
    CHROMA_PERSIST_DIR = _CHROMA_DIR_OVERRIDE
else:
    CHROMA_PERSIST_DIR = os.path.join(tempfile.gettempdir(), "chroma_db")
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
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






