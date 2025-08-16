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
from config import property_search_tool, schedule_viewing_tool, search_new_launches_tool, get_unit_details_tool, configure_gemini
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
    # Create Tool wrapper for function declarations
    tools = types.Tool(function_declarations=[property_search_tool, schedule_viewing_tool, search_new_launches_tool, get_unit_details_tool])
    
    model = genai.GenerativeModel(
        variables.GEMINI_MODEL_NAME,
        tools=[tools]
    )
    logging.info("✅ Gemini model configured successfully with tools (v0.8.5)")
except Exception as e:
    logging.error(f"❌ Error configuring Gemini model: {e}")
    logging.error(f"❌ Tool schemas that failed:")
    logging.error(f"   property_search_tool: {property_search_tool}")
    logging.error(f"   schedule_viewing_tool: {schedule_viewing_tool}")
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

@app.route("/start", methods=["GET"])
def start_conversation():
    logging.info("Starting a new conversation...")
    client_info = config.fetch_user_info_from_api()
    config.client_sessions["active_client"] = client_info
    # Generate a simple session ID for Gemini (no threads needed)
    session_id = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logging.info(f"New session created with ID: {session_id}")
    config.client_sessions[session_id] = client_info
    return jsonify({"thread_id": session_id})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    thread_id = data.get("thread_id")
    user_message = f"Current Date: {datetime.datetime.now().strftime('%B %d, %Y')}\n" + data.get('message', '')

    logging.info(f"Received /chat request: thread_id={thread_id}, user_message={user_message}")

    if not user_message:
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
        client_info = get_session(thread_id)
        if not client_info:
            # Handle first-time users or missing sessions gracefully
            logging.info(f"Session not found for {thread_id}, creating default client info")
            client_info = {
                "user_id": f"new_user_{thread_id}",
                "name": "New User",
                "phone": "Not Provided", 
                "email": "Not Provided"
            }
            # Save the default session for future use
            import config
            config.client_sessions[thread_id] = client_info
            logging.info(f"Created default session for {thread_id}: {client_info}")

    user_id = client_info["user_id"]
    
    # Get conversation history for context-aware preference extraction
    conversation_history = []
    try:
        conversations = functions.load_from_cache("conversations_cache.json")
        conversation = next(
            (c for c in conversations if str(c.get("conversation_id")) == str(thread_id) and str(c.get("user_id")) == str(user_id)),
            None
        )
        if conversation:
            conversation_history = [msg["message"] for msg in conversation.get("description", []) if msg.get("sender") == "Client"]
    except Exception as e:
        logging.warning(f"Could not load conversation history: {e}")
    
    # Get current accumulated preferences
    current_preferences = functions.get_conversation_preferences(thread_id, user_id)
    
    # Determine conversation path based on user message and history
    conversation_path = None
    if "إطلاقات جديدة" in user_message or "new launch" in user_message.lower() or "🚀" in user_message:
        conversation_path = "new_launches"
    elif "وحدات متاحة" in user_message or "available units" in user_message.lower() or "🏠" in user_message:
        conversation_path = "available_units"
    elif conversation_history:
        # Check recent conversation history for path indicators
        recent_messages = conversation_history[-3:]  # Last 3 messages
        for msg in recent_messages:
            if "إطلاقات جديدة" in msg or "new launch" in msg.lower() or "🚀" in msg:
                conversation_path = "new_launches"
                break
            elif "وحدات متاحة" in msg or "available units" in msg.lower() or "🏠" in msg:
                conversation_path = "available_units"
                break
    
    # Extract preferences using LLM with conversation context
    extracted_info = functions.extract_client_preferences_llm(
        user_message, 
        conversation_history, 
        current_preferences,
        conversation_path
    )
    
    lead_data = {
        "user_id": user_id,
        "name": client_info.get("name", ""),
        "phone": client_info.get("phone", ""),
        "email": client_info.get("email", ""),
        **extracted_info
    }
    functions.create_lead(lead_data)
    functions.log_conversation_to_db(thread_id, user_id, user_message)

    try:
        # Process response with function calling
        logging.info(f"Calling run_async_tool_calls with model, user_message, session_id={thread_id}")
        result = run_async_tool_calls(model, user_message, thread_id)
        logging.info(f"Result from run_async_tool_calls: {result}")
        

        
        if result and "error" in result:
            bot_response = f"❌ خطأ: {result['error']}"
        elif result and "function_output" in result:
            function_output = result['function_output']
            function_name = result.get('function_name')

            if function_name == 'property_search':
                if function_output.get('results'):
                    results = function_output.get('results', [])
                    # After property_search, format the real results as a string
                    real_results_str = ""
                    # Support both string-formatted lines and dict items
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

                    # Build complete response with message, results, and follow-up
                    message = function_output.get('message', '')
                    follow_up = function_output.get('follow_up', '')
                    complete_response = f"{message}\n\n{real_results_str}\n{follow_up}".strip()
                    bot_response = complete_response
                else:
                    bot_response = f"✅ تم تنفيذ {function_name} بنجاح: {function_output}"

            elif function_name == 'search_new_launches':
                # Show new launches message + list + follow-up (already LLM filtered)
                message = function_output.get('message', '')
                results = function_output.get('results', [])
                follow_up = function_output.get('follow_up', '')
                results_str = "\n".join(results) if isinstance(results, list) else str(results)
                bot_response = f"{message}\n\n{results_str}\n{follow_up}".strip()

            elif function_name == 'get_unit_details':
                # Handle formatted unit details cleanly
                msg = function_output.get('message') or function_output.get('error')
                if msg:
                    bot_response = msg
                else:
                    bot_response = f"✅ تم تنفيذ {function_name} بنجاح."

            elif function_name == 'schedule_viewing':
                bot_response = function_output.get('message', '✅ تم حجز الموعد بنجاح!')

            else:
                # Generic fallback
                bot_response = function_output.get('message') or f"✅ تم تنفيذ {function_name} بنجاح: {function_output}"
        elif result and "text_response" in result:
            bot_response = result["text_response"]
        else:
            bot_response = "❌ لم يتم استلام رد من المساعد."
            logging.warning("No valid response from Gemini or tool calls.")

        # Log the bot response
        functions.log_conversation_to_db(thread_id, "bot", bot_response)
        
        return jsonify({"response": bot_response, "thread_id": thread_id})
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return jsonify({"error": "Failed to generate response"}), 500

# Add a /test_gemini endpoint to test Gemini connectivity
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

# Add a /test_smart_search endpoint to test the improved classification system
@app.route("/test_smart_search", methods=["POST"])
def test_smart_search():
    try:
        data = request.json
        query = data.get("query", "")
        search_args = data.get("search_args", {})
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        logging.info(f"Testing smart search with query: {query}")
        
        # Test the classification
        query_type = functions.classify_query_type_with_llm(query)
        
        # Test the smart search
        smart_results = functions.smart_property_search(query, search_args)
        
        return jsonify({
            "query": query,
            "classification": query_type,
            "smart_search_results": smart_results
        })
        
    except Exception as e:
        logging.error(f"Smart search test error: {e}")
        return jsonify({"error": str(e)}), 500

# Add a /test_classification endpoint to test just the classification
@app.route("/test_classification", methods=["POST"])
def test_classification():
    try:
        data = request.json
        query = data.get("query", "")
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        logging.info(f"Testing classification with query: {query}")
        
        # Test the classification
        query_type = functions.classify_query_type_with_llm(query)
        
        return jsonify({
            "query": query,
            "classification": query_type,
            "explanation": f"Query '{query}' was classified as: {query_type}"
        })
        
    except Exception as e:
        logging.error(f"Classification test error: {e}")
        return jsonify({"error": str(e)}), 500

# Add a /health endpoint to check ChromaDB status 'GOOGLE CLOUD'
@app.route("/health", methods=["GET"])
def health_check():
    try:
        # Check if ChromaDB is working
        from chroma_rag_setup import RealEstateRAG
        rag = RealEstateRAG()
        stats = rag.get_collection_stats()
        
        # Test a simple search
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

# Add a /ready endpoint for Google Cloud readiness probe 'GOOGLE CLOUD'
@app.route("/ready", methods=["GET"])
def readiness_check():
    """Readiness check for Google Cloud deployment"""
    try:
        # Basic readiness check - app is ready if it can respond
        ready = True
        issues = []
        
        # Check if environment variables are set
        if not os.environ.get('GEMINI_API_KEY'):
            ready = False
            issues.append("GEMINI_API_KEY not set")
        
        # Check if ChromaDB is initialized
        if not chromadb_initialized:
            ready = False
            issues.append("ChromaDB not initialized")
        
        # Check if caches are loaded
        try:
            units = Cache_code.load_from_cache("units.json")
            if not units:
                ready = False
                issues.append("Units cache empty")
        except Exception as e:
            ready = False
            issues.append(f"Units cache error: {str(e)}")
        
        if ready:
            return jsonify({
                "ready": True,
                "message": "Application is ready to serve requests"
            }), 200
        else:
            return jsonify({
                "ready": False,
                "issues": issues,
                "message": "Application is not ready"
            }), 503
            
    except Exception as e:
        return jsonify({
            "ready": False,
            "error": str(e),
            "message": "Readiness check failed"
        }), 503
# 'GOOGLE CLOUD'    
# Initialize caches with error handling for GCP deployment GOOGLE CLOUD
def initialize_caches_for_cloud():
    """Initialize caches with error handling for cloud deployment"""
    try:
        logging.info("🔄 Initializing caches for Google Cloud deployment...")
        
        # Initialize caches with error handling
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

# Initialize caches for cloud deployment
cache_initialized = initialize_caches_for_cloud()
#'GOOGLE CLOUD' 

# Initialize and recreate ChromaDB collections for Google Cloud deployment
# This ensures the app works without needing to upload ChromaDB files 'GOOGLE CLOUD'
def initialize_chromadb_for_cloud():
    try:
        logging.info("🔄 Initializing ChromaDB collections for Google Cloud deployment...")
        
        # Import ChromaDB setup
        from chroma_rag_setup import RealEstateRAG
        
        # Initialize RAG system
        rag = RealEstateRAG()
        
        # Load data from cache
        units_data = Cache_code.load_from_cache("units.json")
        new_launches_data = Cache_code.load_from_cache("new_launches.json")
        
        logging.info(f"📊 Loaded {len(units_data)} units and {len(new_launches_data)} new launches from cache")
        
        # Store units in ChromaDB with embeddings
        logging.info("🔄 Storing units in ChromaDB with embeddings...")
        rag.store_units_in_chroma(units_data)
        
        # Store new launches in ChromaDB with embeddings
        logging.info("🔄 Storing new launches in ChromaDB with embeddings...")
        rag.store_new_launches_in_chroma(new_launches_data)
        
        # Get collection stats to verify
        stats = rag.get_collection_stats()
        logging.info(f"✅ ChromaDB initialization complete! Collection stats: {stats}")
        
        return True
        
    except Exception as e:
        logging.error(f"❌ Error initializing ChromaDB for cloud deployment: {e}")
        return False

# Validate environment variables for GCP deployment
def validate_environment_for_cloud():
    """Validate required environment variables for cloud deployment"""
    required_vars = ['GEMINI_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        logging.error(f"❌ Missing required environment variables: {missing_vars}")
        logging.error("Please set these variables in Google Cloud deployment")
        return False
    
    logging.info("✅ All required environment variables are set")
    return True

# Validate environment variables
env_valid = validate_environment_for_cloud()

# Initialize ChromaDB for cloud deployment
chromadb_initialized = False
if env_valid:
    chromadb_initialized = initialize_chromadb_for_cloud()
    if not chromadb_initialized:
        logging.warning("⚠️ ChromaDB initialization failed, but app will continue with limited functionality")
else:
    logging.warning("⚠️ Skipping ChromaDB initialization due to missing environment variables")
# 'GOOGLE CLOUD'
# Configure scheduler with error handling for GCP deployment GOOGLE CLOUD
def initialize_scheduler_for_cloud():
    """Initialize scheduler with error handling for cloud deployment"""
    try:
        scheduler = BackgroundScheduler()
        
        # Add scheduled jobs with error handling
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

# Initialize scheduler for cloud deployment
scheduler = initialize_scheduler_for_cloud()
# GOOGLE CLOUD

# Proper shutdown handling
if scheduler:
    atexit.register(lambda: scheduler.shutdown())

# Print startup summary for GCP deployment
def print_startup_summary():
    """Print startup summary for cloud deployment monitoring"""
    logging.info("🚀 SharyAI Startup Summary for Google Cloud:")
    logging.info(f"   📊 Cache Initialization: {'✅ Success' if cache_initialized else '❌ Failed'}")
    logging.info(f"   🔧 Environment Variables: {'✅ Valid' if env_valid else '❌ Invalid'}")
    logging.info(f"   🗄️ ChromaDB Initialization: {'✅ Success' if chromadb_initialized else '❌ Failed'}")
    logging.info(f"   ⏰ Scheduler: {'✅ Running' if scheduler else '❌ Failed'}")
    logging.info("🎯 Application is ready to serve requests!")

# Print startup summary
print_startup_summary()

if __name__ == "__main__":
    # Google Cloud deployment configuration
    # The app will automatically initialize ChromaDB on startup
    # No need to upload chroma_db/ directory to cloud
    try:
        port = int(os.environ.get("PORT", 8080))  # Google Cloud sets PORT environment variable
        logging.info(f"🚀 Starting SharyAI on port {port} for Google Cloud deployment")
        app.run(host="0.0.0.0", port=port, debug=False)
    except Exception as e:
        logging.error(f"❌ Failed to start application: {e}")
        # Exit with error code for Google Cloud to detect startup failure
        exit(1)
