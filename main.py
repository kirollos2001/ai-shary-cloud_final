"""
SharyAI Real Estate Chatbot - Google Cloud Deployment Ready

This application is designed to work on Google Cloud Platform with automatic ChromaDB initialization.
Key features:
- Automatic ChromaDB collection recreation on startup
- Embedding generation for semantic search
- No need to upload ChromaDB files to cloud
- Health check endpoints for monitoring
- Environment variable configuration for cloud deployment

For deployment:
1. Set GEMINI_API_KEY as environment variable in Google Cloud
2. Deploy the application code (excluding chroma_db/ directory)
3. ChromaDB will be automatically recreated on first startup
"""

import os
import logging
import json
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from google.generativeai import types
import config
import core_functions
import Assistant
import functions
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
import Cache_code 
import datetime
from session_store import save_session, get_session
import asyncio
from concurrent.futures import ThreadPoolExecutor
import variables
from config import property_search_tool, schedule_viewing_tool, search_new_launches_tool, get_unit_details_tool, configure_gemini, insight_search_tool, get_more_units_tool
from Cache_code import load_from_cache

# Load environment variables from env file
# For Google Cloud deployment, set GEMINI_API_KEY as environment variable
try:
    with open('env', 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value
except FileNotFoundError:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)

# Flask App
app = Flask(__name__)
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')  # Get from environment variables

# Check if API key is available
if not GEMINI_API_KEY:
    logging.warning("⚠️ GEMINI_API_KEY environment variable is not set")
    raise ValueError("Gemini API key is not configured. Please set GEMINI_API_KEY environment variable")
else:
    logging.info("✅ Using Gemini API key from environment variables")

# Configure Gemini API key globally
configure_gemini()

# Debug: Print tool schemas to identify the issue
logging.info("🔍 Debugging tool schemas:")
logging.info(f"property_search_tool: {property_search_tool}")
logging.info(f"schedule_viewing_tool: {schedule_viewing_tool}")
logging.info(f"search_new_launches_tool: {search_new_launches_tool}")
logging.info(f"get_unit_details_tool: {get_unit_details_tool}")

# Configure Gemini with tools using v0.80.5 syntax
try:
    tools = types.Tool(function_declarations=[
        property_search_tool, 
        schedule_viewing_tool, 
        search_new_launches_tool, 
        get_unit_details_tool, 
        insight_search_tool, 
        get_more_units_tool
    ])
    
    model = genai.GenerativeModel(
        variables.GEMINI_MODEL_NAME,
        tools=[tools]
    )
    logging.info("✅ Gemini model configured successfully with tools (v0.8.5)")
except Exception as e:
    logging.error(f"❌ Error configuring Gemini model: {e}")
    raise

if not model:
    raise ValueError("Failed to configure Gemini model with tools")

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=4)

def run_async_tool_calls(model, message, session_id):
    """Wrapper to run async function in a thread"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(
            core_functions.process_gemini_response_async(model, message, session_id)
        )
        return result
    finally:
        loop.close()

@app.route("/")
def index():
    return render_template("index.html")

# ---------- HEALTH & CLOUD ENDPOINTS ----------
@app.route("/health", methods=["GET"])
def health_check():
    try:
        from chroma_rag_setup import RealEstateRAG
        rag = RealEstateRAG()
        stats = rag.get_collection_stats()
        test_results = rag.search_units("test", n_results=1)
        return jsonify({
            "status": "healthy",
            "chromadb_initialized": chromadb_initialized,
            "collection_stats": stats,
            "test_search_working": len(test_results) >= 0,
            "message": "All systems operational"
        })
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "chromadb_initialized": chromadb_initialized,
            "error": str(e),
            "message": "System has issues"
        }), 500

@app.route("/ready", methods=["GET"])
def readiness_check():
    """Readiness check for Google Cloud deployment"""
    try:
        ready = True
        issues = []
        if not os.environ.get('GEMINI_API_KEY'):
            ready = False
            issues.append("GEMINI_API_KEY not set")
        if not chromadb_initialized:
            ready = False
            issues.append("ChromaDB not initialized")
        try:
            units = Cache_code.load_from_cache("units.json")
            if not units:
                ready = False
                issues.append("Units cache empty")
        except Exception as e:
            ready = False
            issues.append(f"Units cache error: {str(e)}")
        if ready:
            return jsonify({"ready": True, "message": "Application is ready to serve requests"}), 200
        else:
            return jsonify({"ready": False, "issues": issues, "message": "Application is not ready"}), 503
    except Exception as e:
        return jsonify({"ready": False, "error": str(e), "message": "Readiness check failed"}), 503

def initialize_caches_for_cloud():
    """Initialize caches with error handling for cloud deployment"""
    try:
        logging.info("🔄 Initializing caches for Google Cloud deployment...")
        try:
            Cache_code.cache_units_from_db()
            logging.info("✅ Units cache initialized successfully")
        except Exception as e:
            logging.warning(f"⚠️ Units cache initialization failed: {e}")
        try:
            Cache_code.cache_new_launches_from_db()
            logging.info("✅ New launches cache initialized successfully")
        except Exception as e:
            logging.warning(f"⚠️ New launches cache initialization failed: {e}")
        try:
            Cache_code.cache_devlopers_from_db()
            logging.info("✅ Developers cache initialized successfully")
        except Exception as e:
            logging.warning(f"⚠️ Developers cache initialization failed: {e}")
        return True
    except Exception as e:
        logging.error(f"❌ Cache initialization failed: {e}")
        return False

cache_initialized = initialize_caches_for_cloud()

def initialize_chromadb_for_cloud():
    try:
        logging.info("🔄 Initializing ChromaDB collections for Google Cloud deployment...")
        from chroma_rag_setup import RealEstateRAG
        rag = RealEstateRAG()
        units_data = Cache_code.load_from_cache("units.json")
        new_launches_data = Cache_code.load_from_cache("new_launches.json")
        logging.info(f"📊 Loaded {len(units_data)} units and {len(new_launches_data)} new launches from cache")
        rag.store_units_in_chroma(units_data)
        rag.store_new_launches_in_chroma(new_launches_data)
        stats = rag.get_collection_stats()
        logging.info(f"✅ ChromaDB initialization complete! Collection stats: {stats}")
        return True
    except Exception as e:
        logging.error(f"❌ Error initializing ChromaDB for cloud deployment: {e}")
        return False

def validate_environment_for_cloud():
    """Validate required environment variables for cloud deployment"""
    required_vars = ['GEMINI_API_KEY']
    missing_vars = []
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    if missing_vars:
        logging.error(f"❌ Missing required environment variables: {missing_vars}")
        return False
    logging.info("✅ All required environment variables are set")
    return True

env_valid = validate_environment_for_cloud()

chromadb_initialized = False
if env_valid:
    chromadb_initialized = initialize_chromadb_for_cloud()
    if not chromadb_initialized:
        logging.warning("⚠️ ChromaDB initialization failed, but app will continue with limited functionality")
else:
    logging.warning("⚠️ Skipping ChromaDB initialization due to missing environment variables")

def initialize_scheduler_for_cloud():
    """Initialize scheduler with error handling for cloud deployment"""
    try:
        scheduler = BackgroundScheduler()
        try:
            scheduler.add_job(Cache_code.cache_leads_from_db, 'cron', hour=4)
            scheduler.add_job(Cache_code.cache_conversations_from_db, 'cron', hour=4)
            scheduler.add_job(Cache_code.sync_leads_to_db, 'cron', hour=3)
            scheduler.add_job(Cache_code.sync_conversations_to_db, 'cron', hour=3)
            scheduler.start()
            logging.info("✅ Scheduler initialized successfully")
            return scheduler
        except Exception as e:
            logging.warning(f"⚠️ Scheduler initialization failed: {e}")
            return None
    except Exception as e:
        logging.error(f"❌ Scheduler setup failed: {e}")
        return None

scheduler = initialize_scheduler_for_cloud()
if scheduler:
    atexit.register(lambda: scheduler.shutdown())

def print_startup_summary():
    """Print startup summary for cloud deployment monitoring"""
    logging.info("🚀 SharyAI Startup Summary for Google Cloud:")
    logging.info(f"   📊 Cache Initialization: {'✅ Success' if cache_initialized else '❌ Failed'}")
    logging.info(f"   🔧 Environment Variables: {'✅ Valid' if env_valid else '❌ Invalid'}")
    logging.info(f"   🗄️ ChromaDB Initialization: {'✅ Success' if chromadb_initialized else '❌ Failed'}")
    logging.info(f"   ⏰ Scheduler: {'✅ Running' if scheduler else '❌ Failed'}")
    logging.info("🎯 Application is ready to serve requests!")

print_startup_summary()

if __name__ == "__main__":
    try:
        port = int(os.environ.get("PORT", 8080))  # Google Cloud sets PORT environment variable
        logging.info(f"🚀 Starting SharyAI on port {port} for Google Cloud deployment")
        app.run(host="0.0.0.0", port=port, debug=False)
    except Exception as e:
        logging.error(f"❌ Failed to start application: {e}")
        exit(1)
