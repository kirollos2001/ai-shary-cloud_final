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
1. Set GEMINI_API_KEY as environment variable in Google Cloud (Secret Manager recommended)
2. Deploy the application code (excluding chroma_db/ directory)
3. ChromaDB will be automatically recreated on first startup
4. Use Cloud Scheduler to trigger weekly jobs (/tasks/nightly)
"""

import os
import logging
import json
import datetime
import asyncio
import atexit
import time
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from google.generativeai import types

# --- Local modules ---
import config
import core_functions
import Assistant
import functions
# Import Cache_code module
import Cache_code
from session_store import save_session, get_session
import variables
from config import (
    property_search_tool,
    schedule_viewing_tool,
    search_new_launches_tool,
    get_unit_details_tool,
    insight_search_tool,
    get_more_units_tool,
    configure_gemini,
)

# -------------------------------------------------------
# Optional local .env loader (ignored on Cloud Run)
# -------------------------------------------------------
try:
    with open('env', 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value
except FileNotFoundError:
    pass

# -------------------------------------------------------
# Logging
# -------------------------------------------------------
logging.basicConfig(level=logging.INFO)

# -------------------------------------------------------
# Flask App
# -------------------------------------------------------
app = Flask(__name__)

# Predefine global flag to avoid NameError in endpoints before init
chromadb_initialized = False

# -------------------------------------------------------
# Gemini API Key
# -------------------------------------------------------
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    logging.warning("⚠️ GEMINI_API_KEY environment variable is not set")
    raise ValueError("Gemini API key is not configured. Please set GEMINI_API_KEY environment variable")
else:
    logging.info("✅ Using Gemini API key from environment variables")

# Configure Gemini globally
configure_gemini()

# Debug tools schemas
logging.info("🔍 Debugging tool schemas:")
logging.info(f"property_search_tool: {property_search_tool}")
logging.info(f"schedule_viewing_tool: {schedule_viewing_tool}")
logging.info(f"search_new_launches_tool: {search_new_launches_tool}")
logging.info(f"get_unit_details_tool: {get_unit_details_tool}")

# Configure Gemini model with tools
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
    logging.info("✅ Gemini model configured successfully with tools")
except Exception as e:
    logging.error(f"❌ Error configuring Gemini model: {e}")
    logging.error(f"❌ Tool schemas that failed:")
    logging.error(f"   property_search_tool: {property_search_tool}")
    logging.error(f"   schedule_viewing_tool: {schedule_viewing_tool}")
    raise

if not model:
    raise ValueError("Failed to configure Gemini model with tools")

# -------------------------------------------------------
# Thread pool for async ops
# -------------------------------------------------------
executor = ThreadPoolExecutor(max_workers=4)



# -------------------------------------------------------
# Routes
# -------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start", methods=["GET"])
def start_conversation():
    logging.info("Starting a new conversation...")

    try:
        # Try to fetch user info from API using thread pool
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            client_info = loop.run_until_complete(config.fetch_user_info_from_api())
        finally:
            loop.close()

    except Exception as e:
        logging.warning(f"Failed to fetch user info from API, using fallback: {e}")
        client_info = {
            "user_id": f"guest_{int(time.time())}",
            "name": "Guest User",
            "phone": "Not Provided",
            "email": "guest@example.com",
        }
    
    config.client_sessions["active_client"] = client_info
    session_id = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logging.info(f"New session created with ID: {session_id}")
    config.client_sessions[session_id] = client_info
    return jsonify({"thread_id": session_id})

@app.route("/chat", methods=["GET", "POST"])
def chat():
    if request.method == "GET":
        # Handle GET requests (e.g., for browser testing)
        return jsonify({
            "message": "This endpoint requires a POST request with JSON body. Example: {'message': 'Your query', 'thread_id': 'optional_session_id'}",
            "status": "Use POST for chatting"
        })

    data = request.json or {}
    thread_id = data.get("thread_id")
    user_message = f"Current Date: {datetime.datetime.now().strftime('%B %d, %Y')}\n" + data.get('message', '')

    logging.info(f"Received /chat request: thread_id={thread_id}, user_message={user_message}")

    if not user_message.strip():
        logging.warning("No user message provided in /chat request.")
        return jsonify({"error": "Message is required"}), 400

    if not thread_id:
        client_info = data.get("client_info")
        if not client_info or not client_info.get("user_id"):
            logging.warning("Client info missing or incomplete in /chat request.")
            return jsonify({"error": "Client info is missing or incomplete"}), 400

        thread_id = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logging.info(f"🧵 New session created: {thread_id}")
        save_session(thread_id, client_info)
    else:
        # Quick session check - don't block on missing sessions
        client_info = get_session(thread_id)
        if not client_info:
            # Create minimal client info immediately, don't block
            client_info = {
                "user_id": f"new_user_{thread_id}",
                "name": "New User",
                "phone": "Not Provided",
                "email": "Not Provided"
            }
            # Save session in background
            try:
                executor.submit(save_session, thread_id, client_info)
            except Exception as e:
                logging.warning(f"Background session save failed: {e}")
            logging.info(f"Created default session for {thread_id}: {client_info}")

    user_id = client_info["user_id"]

    # Conversation history - load quickly with fallback
    conversation_history = []
    try:
        conversations = functions.load_from_cache("conversations_cache.json")
        conversation = next(
            (c for c in conversations if str(c.get("conversation_id")) == str(thread_id)
             and str(c.get("user_id")) == str(user_id)),
            None
        )
        if conversation:
            # Limit history to last 5 messages for performance
            recent_messages = conversation.get("description", [])[-5:]
            conversation_history = [
                msg["message"]
                for msg in recent_messages
                if msg.get("sender") == "Client"
            ]
    except Exception as e:
        logging.warning(f"Could not load conversation history: {e}")
        # Continue without history - don't block the response

    # Current preferences - load in background
    current_preferences = {}
    try:
        # Load preferences in background to avoid blocking
        executor.submit(lambda: functions.get_conversation_preferences(thread_id, user_id))
    except Exception as e:
        logging.warning(f"Background preferences loading failed: {e}")

    # Conversation path
    conversation_path = None
    if "إطلاقات جديدة" in user_message or "new launch" in user_message.lower() or "🚀" in user_message:
        conversation_path = "new_launches"
    elif "وحدات متاحة" in user_message or "available units" in user_message.lower() or "🏠" in user_message:
        conversation_path = "available_units"
    elif conversation_history:
        recent_messages = conversation_history[-3:]
        for msg in recent_messages:
            if "إطلاقات جديدة" in msg or "new launch" in msg.lower() or "🚀" in msg:
                conversation_path = "new_launches"
                break
            elif "وحدات متاحة" in msg or "available units" in msg.lower() or "🏠" in msg:
                conversation_path = "available_units"
                break

    # Extract preferences with LLM
    extracted_info = functions.extract_client_preferences_llm(
        user_message,
        conversation_history,
        current_preferences,
        conversation_path
    )

    # 🔥 ENHANCED: Store client information in real-time
    try:
        # Update client info if new information is provided
        updated_client_info = client_info.copy()
        if extracted_info.get('name') and extracted_info['name'] != 'Unknown':
            updated_client_info['name'] = extracted_info['name']
        if extracted_info.get('phone') and extracted_info['phone'] != 'Unknown':
            updated_client_info['phone'] = extracted_info['phone']
        if extracted_info.get('email') and extracted_info['email'] != 'Unknown':
            updated_client_info['email'] = extracted_info['email']
        
        # Store updated client info in conversation cache
        functions.store_client_info_in_conversation(thread_id, user_id, {
            'name': updated_client_info.get('name', 'Unknown'),
            'phone': updated_client_info.get('phone', 'Unknown'),
            'email': updated_client_info.get('email', 'Unknown'),
            'last_updated': time.time()
        })
        
        # Update session with new info
        if updated_client_info != client_info:
            save_session(thread_id, updated_client_info)
            client_info = updated_client_info
            
        logging.info(f"✅ Updated client info: {updated_client_info.get('name')}, {updated_client_info.get('phone')}, {updated_client_info.get('email')}")
    except Exception as e:
        logging.error(f"❌ Failed to update client info: {e}")

    # Store lead data for later processing (after response is sent)
    lead_data = {
        "user_id": user_id,
        "name": client_info.get("name", ""),
        "phone": client_info.get("phone", ""),
        "email": client_info.get("email", ""),
        **extracted_info
    }

    try:
        logging.info(
            f"Calling run_async_tool_calls with model, user_message, session_id={thread_id}"
        )
        # Use the thread pool executor to run the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                core_functions.process_gemini_response_async(model, user_message, thread_id)
            )
        finally:
            loop.close()
        logging.info(f"Result from run_async_tool_calls: {result}")

        if result and "error" in result:
            bot_response = f"❌ خطأ: {result['error']}"
        elif result and "function_output" in result:
            function_output = result['function_output']
            function_name = result.get('function_name')

            if function_name == 'property_search':
                # Check if this is a validation message asking for more information
                if function_output.get('source') == 'validation':
                    # Display the validation message directly
                    bot_response = function_output.get('message', 'محتاج منك معلومات إضافية للبحث.')
                elif function_output.get('results'):
                    results = function_output.get('results', [])
                    real_results_str = ""
                    for line in results[:10]:
                        if isinstance(line, str):
                            real_results_str += line + "\n"
                        elif isinstance(line, dict):
                            real_results_str += (
                                f"ID:{line.get('id','غير متوفر')} | "
                                f"{line.get('name_ar', line.get('name_en','غير متوفر'))} | "
                                f"السعر: {line.get('price', 'غير متوفر')} | "
                                f"غرف: {line.get('Bedrooms', line.get('bedrooms','غير متوفر'))} | "
                                f"حمام: {line.get('Bathrooms', line.get('bathrooms','غير متوفر'))}\n"
                            )
                    message = function_output.get('message', '')
                    follow_up = function_output.get('follow_up', '')
                    complete_response = f"{message}\n\n{real_results_str}\n{follow_up}".strip()
                    bot_response = complete_response
                else:
                    # Handle other cases like no results or errors
                    bot_response = function_output.get('message', f"✅ تم تنفيذ {function_name} بنجاح: {function_output}")

            elif function_name == 'search_new_launches':
                message = function_output.get('message', '')
                results = function_output.get('results', [])
                follow_up = function_output.get('follow_up', '')
                results_str = "\n".join(results) if isinstance(results, list) else str(results)
                bot_response = f"{message}\n\n{results_str}\n{follow_up}".strip()

            elif function_name == 'get_unit_details':
                msg = function_output.get('message') or function_output.get('error')
                bot_response = msg if msg else f"✅ تم تنفيذ {function_name} بنجاح."

            elif function_name == 'schedule_viewing':
                bot_response = function_output.get('message', '✅ تم حجز الموعد بنجاح!')

            elif function_name == 'insight_search':
                message = function_output.get('message', '')
                results = function_output.get('results', [])
                results_str = "\n".join(results) if isinstance(results, list) else str(results)
                bot_response = f"{message}\n\n{results_str}".strip()

            elif function_name == 'get_more_units':
                if function_output.get('results'):
                    results = function_output.get('results', [])
                    real_results_str = ""
                    for line in results[:10]:
                        if isinstance(line, str):
                            real_results_str += line + "\n"
                        elif isinstance(line, dict):
                            real_results_str += (
                                f"ID:{line.get('id','غير متوفر')} | "
                                f"{line.get('name_ar', line.get('name_en','غير متوفر'))} | "
                                f"السعر: {line.get('price', 'غير متوفر')} | "
                                f"غرف: {line.get('Bedrooms', line.get('bedrooms','غير متوفر'))} | "
                                f"حمام: {line.get('Bathrooms', line.get('bathrooms','غير متوفر'))}\n"
                            )
                    message = function_output.get('message', '')
                    follow_up = function_output.get('follow_up', '')
                    bot_response = f"{message}\n\n{real_results_str}\n{follow_up}".strip()
                else:
                    bot_response = function_output.get('message', 'لم يتم العثور على وحدات إضافية.')

            else:
                bot_response = function_output.get('message') or f"✅ تم تنفيذ {function_name} بنجاح: {function_output}"
        elif result and "text_response" in result:
            bot_response = result["text_response"]
        else:
            bot_response = "❌ لم يتم استلام رد من المساعد."
            logging.warning("No valid response from Gemini or tool calls.")

        # Send response to UI immediately (don't block on cache operations)
        response_data = {"response": bot_response, "thread_id": thread_id}
        
        # Process cache operations in background AFTER response is sent
        try:
            # Create lead in background (non-blocking)
            executor.submit(functions.create_lead, lead_data)
        except Exception as e:
            logging.warning(f"Background lead creation failed: {e}")
        
        # Log user conversation in background (non-blocking)
        try:
            executor.submit(functions.log_conversation_to_db, thread_id, user_id, user_message)
        except Exception as e:
            logging.warning(f"Background user conversation logging failed: {e}")
        
        # Log bot response to database in background (non-blocking)
        try:
            executor.submit(functions.log_conversation_to_db, thread_id, "bot", bot_response)
        except Exception as e:
            logging.warning(f"Background bot response logging failed: {e}")
        
        return jsonify(response_data)
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return jsonify({"error": "Failed to generate response"}), 500

# -------------------------------------------------------
# Test endpoints
# -------------------------------------------------------
@app.route("/test_gemini", methods=["GET"])
def test_gemini():
    try:
        logging.info("Testing Gemini model connectivity...")
        response = model.generate_content("Hello, Gemini! Please reply with a short confirmation message.")
        logging.info(f"Gemini test response: {response.text}")
        return jsonify({"response": response.text})
    except Exception as e:
        logging.error(f"Gemini test error: {e}")
        return jsonify({"error": str(e)})

@app.route("/test_smart_search", methods=["POST"])
def test_smart_search():
    try:
        data = request.json or {}
        query = data.get("query", "")
        search_args = data.get("search_args", {})
        if not query:
            return jsonify({"error": "Query is required"}), 400
        logging.info(f"Testing smart search with query: {query}")
        query_type = functions.classify_query_type_with_llm(query)
        smart_results = functions.smart_property_search(query, search_args)
        return jsonify({
            "query": query,
            "classification": query_type,
            "smart_search_results": smart_results
        })
    except Exception as e:
        logging.error(f"Smart search test error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/test_classification", methods=["POST"])
def test_classification():
    try:
        data = request.json or {}
        query = data.get("query", "")
        if not query:
            return jsonify({"error": "Query is required"}), 400
        logging.info(f"Testing classification with query: {query}")
        query_type = functions.classify_query_type_with_llm(query)
        return jsonify({
            "query": query,
            "classification": query_type,
            "explanation": f"Query '{query}' was classified as: {query_type}"
        })
    except Exception as e:
        logging.error(f"Classification test error: {e}")
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------------
# Health / Readiness
# -------------------------------------------------------
@app.route("/health", methods=["GET"])
def health_check():
    try:
        from chroma_rag_setup import get_rag_instance
        rag = get_rag_instance()
        stats = rag.get_collection_stats()
        test_results = rag.search_units("test", n_results=1) or []
        ok_search = len(test_results) > 0
        status = "healthy" if (chromadb_initialized and ok_search) else "degraded"
        return jsonify({
            "status": status,
            "chromadb_initialized": chromadb_initialized,
            "collection_stats": stats,
            "test_search_working": ok_search,
            "message": "All systems operational" if ok_search else "Chroma search returned no results"
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

# -------------------------------------------------------
# Cloud initialization helpers
# -------------------------------------------------------
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

def initialize_chromadb_for_cloud():
    """Recreate ChromaDB collections and embed data from caches"""
    try:
        logging.info("🔄 Initializing ChromaDB collections for Google Cloud deployment...")
        from chroma_rag_setup import get_rag_instance
        rag = get_rag_instance()
        units_data = Cache_code.load_from_cache("units.json")
        new_launches_data = Cache_code.load_from_cache("new_launches.json")
        logging.info(f"📊 Loaded {len(units_data)} units and {len(new_launches_data)} new launches from cache")
        logging.info("🔄 Storing units in ChromaDB with embeddings...")
        rag.store_units_in_chroma(units_data)
        logging.info("🔄 Storing new launches in ChromaDB with embeddings...")
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
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    if missing_vars:
        logging.error(f"❌ Missing required environment variables: {missing_vars}")
        logging.error("Please set these variables in Google Cloud deployment")
        return False
    logging.info("✅ All required environment variables are set")
    return True

# Environment + Chroma init
env_valid = validate_environment_for_cloud()
cache_initialized = initialize_caches_for_cloud()  # Run even if Chroma fails; app can work with limited features
if env_valid:
    chromadb_initialized = initialize_chromadb_for_cloud()
    if not chromadb_initialized:
        logging.warning("⚠️ ChromaDB initialization failed, but app will continue with limited functionality")
else:
    logging.warning("⚠️ Skipping ChromaDB initialization due to missing environment variables")

# -------------------------------------------------------
# Cron endpoints (secured by CRON_TOKEN)
# -------------------------------------------------------
CRON_TOKEN = os.environ.get("CRON_TOKEN")

def _check_cron_auth(req):
    token = req.headers.get("X-CRON-TOKEN")
    if not CRON_TOKEN or token != CRON_TOKEN:
        return False
    return True

@app.route("/tasks/cache-refresh", methods=["POST"])
def task_cache_refresh():
    if not _check_cron_auth(request):
        return jsonify({"error": "unauthorized"}), 401
    try:
        Cache_code.cache_units_from_db()
        Cache_code.cache_new_launches_from_db()
        Cache_code.cache_devlopers_from_db()
        return jsonify({"status": "ok", "task": "cache-refresh"}), 200
    except Exception as e:
        logging.exception("cache-refresh failed")
        return jsonify({"error": str(e)}), 500

@app.route("/tasks/chroma-rebuild", methods=["POST"])
def task_chroma_rebuild():
    if not _check_cron_auth(request):
        return jsonify({"error": "unauthorized"}), 401
    try:
        ok = initialize_chromadb_for_cloud()
        # ✅ update readiness flag
        global chromadb_initialized
        chromadb_initialized = bool(ok)
        return jsonify({"status": "ok" if ok else "failed", "task": "chroma-rebuild"}), 200
    except Exception as e:
        logging.exception("chroma-rebuild failed")
        return jsonify({"error": str(e)}), 500

@app.route("/tasks/nightly", methods=["POST"])
def task_nightly():
    if not _check_cron_auth(request):
        return jsonify({"error": "unauthorized"}), 401
    try:
        # 1) refresh caches
        Cache_code.cache_units_from_db()
        Cache_code.cache_new_launches_from_db()
        Cache_code.cache_devlopers_from_db()
        # 2) rebuild chroma
        ok = initialize_chromadb_for_cloud()
        # ✅ update readiness flag
        global chromadb_initialized
        chromadb_initialized = bool(ok)
        return jsonify({"status": "ok" if ok else "partial", "task": "nightly"}), 200
    except Exception as e:
        logging.exception("nightly failed")
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------------
# Scheduler init (disabled by env on Cloud Run)
# -------------------------------------------------------
def initialize_scheduler_for_cloud():
    """Initialize APScheduler (disable on Cloud Run; use Cloud Scheduler instead)"""
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        scheduler = BackgroundScheduler()
        try:
            scheduler.add_job(Cache_code.cache_leads_from_db, 'cron', hour=4)
            scheduler.add_job(Cache_code.cache_conversations_from_db, 'cron', hour=4)
            scheduler.add_job(Cache_code.sync_leads_to_db, 'cron', hour=3)
            scheduler.add_job(Cache_code.sync_conversations_to_db, 'cron', hour=3)
            scheduler.start()
            logging.info("✅ Internal APScheduler initialized successfully (local/dev)")
            return scheduler
        except Exception as e:
            logging.warning(f"⚠️ Scheduler initialization failed: {e}")
            return None
    except Exception as e:
        logging.error(f"❌ Scheduler setup failed: {e}")
        return None

# Only enable internal scheduler when not on Cloud Run
scheduler = None
if os.environ.get("DISABLE_INTERNAL_SCHEDULER") != "1":
    scheduler = initialize_scheduler_for_cloud()
    if scheduler:
        atexit.register(lambda: scheduler.shutdown())

# -------------------------------------------------------
# Startup summary
# -------------------------------------------------------
def print_startup_summary():
    logging.info("🚀 SharyAI Startup Summary for Google Cloud:")
    logging.info(f"   📊 Cache Initialization: {'✅ Success' if cache_initialized else '❌ Failed'}")
    logging.info(f"   🔧 Environment Variables: {'✅ Valid' if env_valid else '❌ Invalid'}")
    logging.info(f"   🗄️ ChromaDB Initialization: {'✅ Success' if chromadb_initialized else '❌ Failed'}")
    logging.info(f"   ⏰ Internal Scheduler: {'✅ Running' if scheduler else '⏸️ Disabled/Failed'}")
    logging.info("🎯 Application is ready to serve requests!")

print_startup_summary()

# -------------------------------------------------------
# Entrypoint
# -------------------------------------------------------
if __name__ == "__main__":
    try:
        port = int(os.environ.get("PORT", 8080))  # Cloud Run sets PORT
        logging.info(f"🚀 Starting SharyAI on port {port} for Google Cloud deployment")
        app.run(host="0.0.0.0", port=port, debug=False)
    except Exception as e:
        logging.error(f"❌ Failed to start application: {e}")
        exit(1)
