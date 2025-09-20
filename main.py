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

from flask import Flask, render_template, request, jsonify, make_response
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
import speech_utils
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
# Set Google Cloud credentials from variables.py
# -------------------------------------------------------
if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
    credentials_dict = variables.GOOGLE_CLOUD_CREDENTIALS
    if isinstance(credentials_dict, dict):
        try:
            import json
            import tempfile

            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_credentials_file:
                json.dump(credentials_dict, temp_credentials_file, indent=2)
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_credentials_file.name
            logging.info(f"✅ Set Google Cloud credentials from variables.py: {temp_credentials_file.name}")
        except Exception as e:
            logging.warning(f"⚠️ Failed to set Google Cloud credentials from variables.py: {e}")
            logging.warning("⚠️ Audio transcription may not work")
    else:
        logging.warning("⚠️ GOOGLE_CLOUD_CREDENTIALS not configured; relying on default credentials")
# -------------------------------------------------------
# Logging
# -------------------------------------------------------
logging.basicConfig(level=logging.INFO)
# Verify database connectivity early
try:
    _conn = config.get_db_connection()
    _conn.close()
    logging.info("✅ Database connection established")
except Exception as e:
    logging.warning(f"⚠️ Database connection failed: {e}")
# -------------------------------------------------------
# Flask App
# -------------------------------------------------------
app = Flask(__name__)

# Predefine global flag to avoid NameError in endpoints before init
chromadb_initialized = False

# -------------------------------------------------------
# Gemini API Key
# -------------------------------------------------------
GEMINI_API_KEY = getattr(variables, 'GEMINI_API_KEY', None)
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
    # Use the raw user message without prefixing date/time to avoid
    # breaking downstream regex extractors (e.g., bedrooms).
    user_message = data.get('message', '')

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
        # Safety: If the model tried to open unit details without an explicit details intent, steer back
        try:
            if isinstance(result, dict) and result.get('function_name') == 'get_unit_details':
                import re as _re
                _hint_re = r'(تفاصيل|details|show\s+unit|رقم\s*الوحدة|\bunit\b|\bid\b)'
                _cont = bool(_re.search(r'(نكمل|كمل|من غير|بدون|continue|skip)', user_message or '', flags=_re.IGNORECASE))
                if not _re.search(_hint_re, user_message or '', flags=_re.IGNORECASE) and not _cont:
                    bot_response = (
                        "حابب أساعدك بالبحث أولاً. قبل ما أبدأ أبحث لك، ممكن تقولّي المساحة اللي تناسبك بالمتر؟ "
                        "وكمان عدد سنوات التقسيط اللي تحبها؟ (مثلاً: المساحة 150 م²، التقسيط 8 سنين)."
                    )
                    response_data = {"response": bot_response, "thread_id": thread_id}
                    try:
                        executor.submit(functions.create_lead, lead_data)
                    except Exception as e:
                        logging.warning(f"Background lead creation failed: {e}")
                    try:
                        executor.submit(functions.log_conversation_to_db, thread_id, user_id, user_message)
                    except Exception as e:
                        logging.warning(f"Background user conversation logging failed: {e}")
                    try:
                        executor.submit(functions.log_conversation_to_db, thread_id, "bot", bot_response)
                    except Exception as e:
                        logging.warning(f"Background bot response logging failed: {e}")
                    return jsonify(response_data)
        except Exception:
            pass

        # Early fallback: if the model errored, try to run a search directly
        try:
            if result and "error" in result:
                auto_triggered = False
                prefs = {}
                try:
                    prefs = functions.get_conversation_preferences(thread_id, user_id)
                except Exception:
                    prefs = {}

                merged_prefs = functions.extract_client_preferences_llm(user_message, current_preferences=prefs) or {}

                # Merge with last known lead (cached) if present
                try:
                    from Cache_code import load_from_cache as _load_cache
                    cached_leads = _load_cache("leads_cache.json") or []
                    last_lead = next((l for l in cached_leads if str(l.get("user_id")) == str(user_id)), None)
                    if last_lead:
                        for k in ["budget", "location", "property_type", "bedrooms", "bathrooms"]:
                            if not merged_prefs.get(k) and last_lead.get(k):
                                merged_prefs[k] = last_lead.get(k)
                except Exception:
                    pass

                # If user is asking for unit details by ID, handle first
                try:
                    import re as _re
                    msg_clean = (user_message or "").strip()
                    id_match = _re.search(r"\bID[:：]?\s*(\d+)\b", msg_clean, flags=_re.IGNORECASE)
                    if not id_match and msg_clean.isdigit() and 3 <= len(msg_clean) <= 8:
                        id_match = [_re.match(r"(\d+)", msg_clean)] if msg_clean else None
                        id_val = msg_clean
                    else:
                        id_val = id_match.group(1) if id_match else None
                    if id_val:
                        details_out = functions.get_unit_details({"unit_id": id_val})
                        if isinstance(details_out, dict):
                            msg = details_out.get('message') or details_out.get('error') or str(details_out)
                            bot_response = msg
                            auto_triggered = True
                except Exception:
                    pass

                if auto_triggered:
                    response_data = {"response": bot_response, "thread_id": thread_id}
                    try:
                        executor.submit(functions.create_lead, lead_data)
                    except Exception as e:
                        logging.warning(f"Background lead creation failed: {e}")
                    try:
                        executor.submit(functions.log_conversation_to_db, thread_id, user_id, user_message)
                    except Exception as e:
                        logging.warning(f"Background user conversation logging failed: {e}")
                    try:
                        executor.submit(functions.log_conversation_to_db, thread_id, "bot", bot_response)
                    except Exception as e:
                        logging.warning(f"Background bot response logging failed: {e}")
                    return jsonify(response_data)

                # Heuristics for common Arabic inputs
                try:
                    txt = user_message.lower()
                    if not merged_prefs.get("location"):
                        if "القاهرة الجديدة" in user_message or "new cairo" in txt:
                            merged_prefs["location"] = "القاهرة الجديدة"
                    if not merged_prefs.get("property_type"):
                        if "شقة" in user_message or "apartment" in txt:
                            merged_prefs["property_type"] = "شقة"
                    if not merged_prefs.get("bedrooms"):
                        import re
                        if "غرفتين" in user_message:
                            merged_prefs["bedrooms"] = 2
                        else:
                            m = re.search(r"(\d+)\s*غرف", user_message)
                            if m:
                                merged_prefs["bedrooms"] = int(m.group(1))
                except Exception:
                    pass

                # If still no location, try scan conversation history for keywords
                try:
                    if not merged_prefs.get("location"):
                        conversations = functions.load_from_cache("conversations_cache.json")
                        convo = next((c for c in conversations if str(c.get("conversation_id")) == str(thread_id)), None)
                        if convo:
                            desc = convo.get("description", [])
                            for msg in desc:
                                text = (msg.get("message") or "")
                                if "القاهرة الجديدة" in text:
                                    merged_prefs["location"] = "القاهرة الجديدة"
                                    break
                except Exception:
                    pass

                # If user explicitly asked for launches, call new launch search
                try:
                    if any(k in user_message for k in ["إطلاق", "اطلاق", "اطلاقات جديدة"]) or any(k in txt for k in ["new launch", "launches", "launch"]):
                        nl_args = {
                            "property_type": merged_prefs.get("property_type", ""),
                            "location": merged_prefs.get("location", ""),
                            "compound": merged_prefs.get("compound_name", ""),
                            "session_id": thread_id,
                        }
                        nl_out = functions.search_new_launches(nl_args)
                        message = nl_out.get('message', '') if isinstance(nl_out, dict) else ''
                        results = nl_out.get('results', []) if isinstance(nl_out, dict) else []
                        follow_up = nl_out.get('follow_up', '') if isinstance(nl_out, dict) else ''
                        results_str = "\n".join(results) if isinstance(results, list) else str(results)
                        bot_response = f"{message}\n\n{results_str}\n{follow_up}".strip()
                        auto_triggered = True
                except Exception:
                    pass

                if auto_triggered:
                    response_data = {"response": bot_response, "thread_id": thread_id}
                    try:
                        executor.submit(functions.create_lead, lead_data)
                    except Exception as e:
                        logging.warning(f"Background lead creation failed: {e}")
                    try:
                        executor.submit(functions.log_conversation_to_db, thread_id, user_id, user_message)
                    except Exception as e:
                        logging.warning(f"Background user conversation logging failed: {e}")
                    try:
                        executor.submit(functions.log_conversation_to_db, thread_id, "bot", bot_response)
                    except Exception as e:
                        logging.warning(f"Background bot response logging failed: {e}")
                    return jsonify(response_data)

                has_location = bool(merged_prefs.get("location"))
                has_budget = merged_prefs.get("budget", 0) > 0
                has_property_type = bool(merged_prefs.get("property_type"))
                logging.info(f"Auto-trigger (error fallback) — loc:{has_location} budget:{has_budget} type:{has_property_type}")

                # Relax: trigger if at least two core signals are present
                if (has_budget and has_property_type) or (has_budget and has_location) or (has_property_type and has_location):
                    search_args = {
                        "location": merged_prefs.get("location", ""),
                        "budget": merged_prefs.get("budget", 0),
                        "property_type": merged_prefs.get("property_type", ""),
                        "bedrooms": merged_prefs.get("bedrooms", 0),
                        "bathrooms": merged_prefs.get("bathrooms", 0),
                        "compound": merged_prefs.get("compound_name", "")
                    }
                    # Ask about area and installment years before searching (mandatory questions to ask; user may skip)
                    _cont = False
                    try:
                        import re as _re
                        _cont = bool(_re.search(r'(نكمل|كمل|من غير|بدون|continue|skip)', user_message or '', flags=_re.IGNORECASE))
                    except Exception:
                        _cont = False
                    if (not search_args.get("apartment_area") or not search_args.get("installment_years")) and not _cont:
                        bot_response = (
                            "قبل ما أبدأ أبحث لك، ممكن تقولّي المساحة اللي تناسبك بالمتر؟ "
                            "وكمان عدد سنوات التقسيط اللي تحبها؟ (مثلاً: المساحة 150 م²، التقسيط 8 سنين). "
                            "لو تحب نكمل من غيرهم، قولّي نكمل وخلاص."
                        )
                        response_data = {"response": bot_response, "thread_id": thread_id}
                        try:
                            executor.submit(functions.create_lead, lead_data)
                        except Exception as e:
                            logging.warning(f"Background lead creation failed: {e}")
                        try:
                            executor.submit(functions.log_conversation_to_db, thread_id, user_id, user_message)
                        except Exception as e:
                            logging.warning(f"Background user conversation logging failed: {e}")
                        try:
                            executor.submit(functions.log_conversation_to_db, thread_id, "bot", bot_response)
                        except Exception as e:
                            logging.warning(f"Background bot response logging failed: {e}")
                        return jsonify(response_data)
                    function_output = functions.property_search(search_args)
                    if function_output:
                        if function_output.get('results'):
                            results = function_output.get('results', [])
                            real_results_str = ""
                            for line in results[:10]:
                                if isinstance(line, str):
                                    real_results_str += line + "\n"
                                elif isinstance(line, dict):
                                    real_results_str += (
                                        f"ID:{line.get('id','N/A')} | "
                                        f"{line.get('name_ar', line.get('name_en','N/A'))} | "
                                        f"Price: {line.get('price', 'N/A')} | "
                                        f"Rooms: {line.get('Bedrooms', line.get('bedrooms','N/A'))} | "
                                        f"Baths: {line.get('Bathrooms', line.get('bathrooms','N/A'))}\n"
                                    )
                            message = function_output.get('message', '')
                            follow_up = function_output.get('follow_up', '')
                            bot_response = f"{message}\n\n{real_results_str}\n{follow_up}".strip()
                        else:
                            bot_response = function_output.get('message', str(function_output))
                        auto_triggered = True

                if auto_triggered:
                    response_data = {"response": bot_response, "thread_id": thread_id}
                    try:
                        executor.submit(functions.create_lead, lead_data)
                    except Exception as e:
                        logging.warning(f"Background lead creation failed: {e}")
                    try:
                        executor.submit(functions.log_conversation_to_db, thread_id, user_id, user_message)
                    except Exception as e:
                        logging.warning(f"Background user conversation logging failed: {e}")
                    try:
                        executor.submit(functions.log_conversation_to_db, thread_id, "bot", bot_response)
                    except Exception as e:
                        logging.warning(f"Background bot response logging failed: {e}")
                    return jsonify(response_data)
        except Exception as _fe:
            logging.warning(f"Early error-fallback search block failed: {_fe}")

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
                msg = function_output.get('message', '')
                bot_response = msg if msg else f"✅ تم حجز موعد المعاينة بنجاح!"
            
            elif function_name == 'get_current_datetime':
                # Let LLM infer exact desired date/time using the current datetime
                try:
                    inferred = functions.infer_meeting_details_via_llm(user_message, function_output)
                    desired_date = inferred.get('desired_date')
                    desired_time = inferred.get('desired_time')
                    meeting_type = inferred.get('meeting_type') or 'zoom'
                    if desired_date and desired_time:
                        sched_args = {
                            'conversation_id': thread_id,
                            'desired_date': desired_date,
                            'desired_time': desired_time,
                            'meeting_type': meeting_type,
                        }
                        try:
                            from session_store import get_session as _get_session
                            _ci = (_get_session(thread_id) or {})
                            if _ci:
                                sched_args.update({
                                    'client_id': _ci.get('user_id', 1),
                                    'name': _ci.get('name', 'Unknown'),
                                    'phone': _ci.get('phone', 'Not Provided'),
                                    'email': _ci.get('email', 'Not Provided'),
                                })
                        except Exception:
                            pass
                        out = functions.schedule_viewing(sched_args)
                        bot_response = out.get('message', 'تم حجز الموعد بنجاح!')
                    else:
                        # Fallback to informative message from datetime tool
                        bot_response = function_output.get('message') or 'تم حساب التاريخ.'
                except Exception as _e:
                    logging.warning(f'Auto-chain scheduling after get_current_datetime failed: {_e}')
                    bot_response = function_output.get('message') or 'تم حساب التاريخ.'
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
            # Try to auto-trigger a search if enough info is present
            auto_triggered = False
            try:
                prefs = {}
                try:
                    prefs = functions.get_conversation_preferences(thread_id, user_id)
                except Exception:
                    prefs = {}
                merged_prefs = functions.extract_client_preferences_llm(user_message, current_preferences=prefs)
                # Merge with last known lead (cached) if present
                try:
                    from Cache_code import load_from_cache as _load_cache
                    cached_leads = _load_cache("leads_cache.json") or []
                    last_lead = next((l for l in cached_leads if str(l.get("user_id")) == str(user_id)), None)
                    if last_lead:
                        for k in ["budget", "location", "property_type", "bedrooms", "bathrooms"]:
                            if not merged_prefs.get(k) and last_lead.get(k):
                                merged_prefs[k] = last_lead.get(k)
                except Exception:
                    pass

                # 1) Unit details intent (by ID or 'تفاصيل')
                try:
                    import re as _re
                    msg_clean = (user_message or "").strip()
                    id_match = _re.search(r"\bID[:：]?\s*(\d+)\b", msg_clean, flags=_re.IGNORECASE)
                    id_val = None
                    if id_match:
                        id_val = id_match.group(1)
                    elif msg_clean.isdigit() and 3 <= len(msg_clean) <= 8:
                        id_val = msg_clean
                    if ("تفاصيل" in user_message or "details" in user_message.lower() or id_val):
                        if id_val:
                            details_out = functions.get_unit_details({"unit_id": id_val})
                            if isinstance(details_out, dict):
                                msg = details_out.get('message') or details_out.get('error') or str(details_out)
                                bot_response = msg
                                auto_triggered = True
                except Exception:
                    pass

                # 2) New launches intent
                if not auto_triggered:
                    try:
                        txt = user_message.lower()
                        if any(k in user_message for k in ["إطلاق", "اطلاق", "اطلاقات جديدة"]) or any(k in txt for k in ["new launch", "launches", "launch"]):
                            nl_args = {
                                "property_type": merged_prefs.get("property_type", ""),
                                "location": merged_prefs.get("location", ""),
                                "compound": merged_prefs.get("compound_name", ""),
                                "session_id": thread_id,
                            }
                            nl_out = functions.search_new_launches(nl_args)
                            message = nl_out.get('message', '') if isinstance(nl_out, dict) else ''
                            results = nl_out.get('results', []) if isinstance(nl_out, dict) else []
                            follow_up = nl_out.get('follow_up', '') if isinstance(nl_out, dict) else ''
                            results_str = "\n".join(results) if isinstance(results, list) else str(results)
                            bot_response = f"{message}\n\n{results_str}\n{follow_up}".strip()
                            auto_triggered = True
                    except Exception:
                        pass

                # 3) Conversation summary intent
                if not auto_triggered:
                    try:
                        if any(k in user_message for k in ["ملخص"]) or ("summary" in user_message.lower()):
                            name = client_info.get("name", "Unknown") if isinstance(client_info, dict) else "Unknown"
                            phone = client_info.get("phone", "Unknown") if isinstance(client_info, dict) else "Unknown"
                            email = client_info.get("email", "Unknown") if isinstance(client_info, dict) else "Unknown"
                            summary_text = functions.enhanced_conversation_summary_with_client_info(
                                user_id, thread_id, name, phone, email, None, None, None, None
                            )
                            bot_response = summary_text if isinstance(summary_text, str) else str(summary_text)
                            auto_triggered = True
                    except Exception:
                        pass

                has_location = bool(merged_prefs.get("location"))
                has_budget = merged_prefs.get("budget", 0) > 0
                has_property_type = bool(merged_prefs.get("property_type"))
                logging.info(f"Auto-trigger (text fallback) — loc:{has_location} budget:{has_budget} type:{has_property_type}")
                if has_location and has_budget and has_property_type:
                    search_args = {
                        "location": merged_prefs.get("location"),
                        "budget": merged_prefs.get("budget", 0),
                        "property_type": merged_prefs.get("property_type"),
                        "bedrooms": merged_prefs.get("bedrooms", 0),
                        "bathrooms": merged_prefs.get("bathrooms", 0),
                        "compound": merged_prefs.get("compound_name", "")
                    }
                    # Ask about area and installment years before searching (mandatory questions)
                    _cont = False
                    try:
                        import re as _re
                        _cont = bool(_re.search(r'(نكمل|كمل|من غير|بدون|continue|skip)', user_message or '', flags=_re.IGNORECASE))
                    except Exception:
                        _cont = False
                    if (not search_args.get("apartment_area") or not search_args.get("installment_years")) and not _cont:
                        bot_response = (
                            "قبل ما أبدأ أبحث لك، ممكن تقولّي المساحة اللي تناسبك بالمتر؟ "
                            "وكمان عدد سنوات التقسيط اللي تحبها؟ (مثلاً: المساحة 150 م²، التقسيط 8 سنين). "
                            "لو تحب نكمل من غيرهم، قولّي نكمل وخلاص."
                        )
                        response_data = {"response": bot_response, "thread_id": thread_id}
                        try:
                            executor.submit(functions.create_lead, lead_data)
                        except Exception as e:
                            logging.warning(f"Background lead creation failed: {e}")
                        try:
                            executor.submit(functions.log_conversation_to_db, thread_id, user_id, user_message)
                        except Exception as e:
                            logging.warning(f"Background user conversation logging failed: {e}")
                        try:
                            executor.submit(functions.log_conversation_to_db, thread_id, "bot", bot_response)
                        except Exception as e:
                            logging.warning(f"Background bot response logging failed: {e}")
                        return jsonify(response_data)
                    function_output = functions.property_search(search_args)
                    # Format like property_search branch if results exist
                    if function_output and function_output.get('results'):
                        results = function_output.get('results', [])
                        real_results_str = ""
                        for line in results[:10]:
                            if isinstance(line, str):
                                real_results_str += line + "\n"
                            elif isinstance(line, dict):
                                real_results_str += (
                                    f"ID:{line.get('id','N/A')} | "
                                    f"{line.get('name_ar', line.get('name_en','N/A'))} | "
                                    f"Price: {line.get('price', 'N/A')} | "
                                    f"Rooms: {line.get('Bedrooms', line.get('bedrooms','N/A'))} | "
                                    f"Baths: {line.get('Bathrooms', line.get('bathrooms','N/A'))}\n"
                                )
                        message = function_output.get('message', '')
                        follow_up = function_output.get('follow_up', '')
                        bot_response = f"{message}\n\n{real_results_str}\n{follow_up}".strip()
                        auto_triggered = True
                    else:
                        # If no results, fall back to model text
                        bot_response = result["text_response"]
                        auto_triggered = True
            except Exception as _e:
                logging.warning(f"Auto-trigger via text fallback failed: {_e}")
            if not auto_triggered:
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

@app.route("/chat/audio", methods=["POST"])
def chat_audio():
    """Accepts an audio file and returns its transcript.

    - Expects multipart/form-data with field 'audio' or 'file' or 'voice'
    - Optional form field 'thread_id' to pass through (not required)
    - If audio is empty or transcription yields empty text, returns 204 No Content
    """
    try:
        # Quick empty body check
        try:
            content_length = int(request.headers.get('Content-Length') or 0)
            if content_length == 0:
                return ('', 204)
        except Exception:
            pass

        file = (
            request.files.get('audio')
            or request.files.get('file')
            or request.files.get('voice')
        )
        if not file:
            # Some clients send raw bytes directly
            raw = request.get_data(cache=False, as_text=False)
            if not raw:
                return ('', 204)
            mime_type = request.headers.get('Content-Type')
            transcript = speech_utils.transcribe_audio(raw, mime_type)
        else:
            data = file.read() if hasattr(file, 'read') else None
            if not data:
                return ('', 204)
            mime_type = getattr(file, 'mimetype', None) or request.headers.get('Content-Type')
            transcript = speech_utils.transcribe_audio(data, mime_type)

        if not transcript or not str(transcript).strip():
            # Ignore empty audio or failed transcription: no noisy message
            return ('', 204)

        # Pass through optional thread_id for client convenience
        thread_id = request.form.get('thread_id') or request.args.get('thread_id')
        resp = {"transcript": transcript.strip()}
        if thread_id:
            resp["thread_id"] = thread_id
        return jsonify(resp)
    except Exception as e:
        logging.error(f"/chat/audio error: {e}")
        # Do not surface noisy errors to user; treat like empty audio
        return ('', 204)

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
        if getattr(rag, "is_read_only", False):
            try:
                stats = rag.get_collection_stats()
                logging.info(
                    "📊 ChromaDB already available (read-only). "
                    f"Units: {stats.get('units_count', 0)}, "
                    f"New launches: {stats.get('new_launches_count', 0)}, "
                    f"Total: {stats.get('total_count', 0)}"
                )
            except Exception as stats_error:
                logging.warning(f"⚠️ Unable to fetch Chroma stats in read-only mode: {stats_error}")
            logging.info("✅ Skipping Chroma rebuild because pre-generated embeddings are being used from GCS.")
            return True
        units_data = Cache_code.load_from_cache("units.json")
        new_launches_data = Cache_code.load_from_cache("new_launches.json")
        logging.info(f"📊 Loaded {len(units_data)} units and {len(new_launches_data)} new launches from cache")
        logging.info("🔄 Storing units in ChromaDB with embeddings...")
        rag.store_units_in_chroma(units_data)
        logging.info("🔄 Storing new launches in ChromaDB with embeddings...")
        rag.store_new_launches_in_chroma(new_launches_data)
        stats = rag.get_collection_stats()
        logging.info(
            "✅ ChromaDB initialization complete! "
            f"Units stored: {stats.get('units_count', 0)}, "
            f"New launches stored: {stats.get('new_launches_count', 0)}, "
            f"Total: {stats.get('total_count', 0)}"
        )
        return True
    except Exception as e:
        if isinstance(e, RuntimeError) and "READ-ONLY MODE" in str(e).upper():
            logging.info("✅ Detected Chroma read-only runtime error; treating as successful initialization.")
            return Tru
        logging.error(f"❌ Error initializing ChromaDB for cloud deployment: {e}")
        if 'rag' in locals():
            try:
                stats = rag.get_collection_stats()
                logging.info(f"ℹ️ ChromaDB collection stats at failure: {stats}")
            except Exception as stats_error:
                logging.warning(f"⚠️ Unable to read ChromaDB stats after failure: {stats_error}")
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
if cache_initialized:
    try:
        units_count = len(Cache_code.load_from_cache("units.json"))
        launches_count = len(Cache_code.load_from_cache("new_launches.json"))
        logging.info(
            f"📦 Cache contains {units_count} units and {launches_count} new launches"
        )
    except Exception as e:
        logging.warning(f"⚠️ Could not read cache counts: {e}")
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
