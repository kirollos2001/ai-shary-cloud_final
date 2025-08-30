import os

# Gemini 2.0 Flash Configuration
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL_NAME = "gemini-2.5-flash"

# Database Configuration
DB_HOST = os.environ.get("DB_HOST", "77.37.35.26")
DB_PORT = int(os.environ.get("DB_PORT", 3306))
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")

# Email Configuration
EMAIL_HOST = os.environ.get("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT = int(os.environ.get("EMAIL_PORT", 587))
EMAIL_USER = os.environ.get("EMAIL_USER")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
TEAM_EMAIL = os.environ.get("TEAM_EMAIL")

# API Configuration
USER_INFO_API_URL = os.environ.get("USER_INFO_API_URL", "https://shary.eg/api/UserInfo")

# Cache Configuration
CACHE_DIR = "cache/"
LEADS_CACHE_FILE = "leads_cache.json"
CONVERSATIONS_CACHE_FILE = "conversations_cache.json"
UNITS_CACHE_FILE = "units.json"
NEW_LAUNCHES_CACHE_FILE = "new_launches.json"
DEVELOPERS_CACHE_FILE = "developers.json"

# Application Configuration
APP_PORT = int(os.environ.get("APP_PORT", 8080))
APP_HOST = os.environ.get("APP_HOST", "0.0.0.0")
DEBUG_MODE = os.environ.get("DEBUG_MODE", "False").lower() == "true"

# Scheduler Configuration
CACHE_SYNC_HOUR = int(os.environ.get("CACHE_SYNC_HOUR", 4))
DB_SYNC_HOUR = int(os.environ.get("DB_SYNC_HOUR", 3))

# Session Configuration
SESSION_TIMEOUT = int(os.environ.get("SESSION_TIMEOUT", 3600))  # 1 hour in seconds

# Logging Configuration
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
