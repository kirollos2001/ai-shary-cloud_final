import os
import uuid  # Added for default session_id
import json
from typing import Dict, Optional, List, Tuple
# from langchain.memory import ConversationSummaryMemory, ConversationEntityMemory, CombinedMemory
# from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
import variables
from redis_utils import get_redis

# 1) امنع استخدام كوتا مشروع GCP (ADC/Quota Project)
for k in (
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_CLOUD_PROJECT",
    "GCLOUD_PROJECT",
    "GOOGLE_CLOUD_QUOTA_PROJECT",
    "GOOGLE_PROJECT_ID",
):
    os.environ.pop(k, None)

# 2) هات الـ API Key من variables.py فقط
GOOGLE_API_KEY = getattr(variables, "GEMINI_API_KEY", None)

if not GOOGLE_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in variables.py")

# 3) عرّف الموديل باستخدام الـ API Key فقط
genai.configure(api_key=GOOGLE_API_KEY)

# (اختياري) Debug بسيط علشان تتأكد إنه شغّال بمفتاح API مش بمشروع GCP
print("Using API key only? ->", bool(GOOGLE_API_KEY) and not any(
    os.getenv(v) for v in [
        "GOOGLE_APPLICATION_CREDENTIALS","GOOGLE_CLOUD_PROJECT",
        "GCLOUD_PROJECT","GOOGLE_CLOUD_QUOTA_PROJECT","GOOGLE_PROJECT_ID"
    ]
))

# Session-based chat storage using Gemini's built-in chat state
_session_chats: Dict[str, genai.ChatSession] = {}

# --- Redis-backed persistence (optional) ---
_HISTORY_KEY = "mem:{sid}:history"
_SUMMARY_KEY = "mem:{sid}:summary"
_ENTITIES_KEY = "mem:{sid}:entities"

# def _build_memory() -> CombinedMemory:
#     """Create a combined memory using LangChain built-ins."""
#     if llm is None:
#         raise ValueError("LLM not initialized; GEMINI_API_KEY is required.")
#     
#     # Use unique keys to avoid CombinedMemory collisions; we'll map back to
#     # the legacy keys (history/entities) in the facade.
#     summary = ConversationSummaryMemory(
#         llm=llm,
#         memory_key="summary_history",
#         input_key="input",
#         output_key="output",
#         return_messages=False
#     )
#     entity = ConversationEntityMemory(
#         llm=llm,
#         memory_key="entities",
#         input_key="input",
#         output_key="output",
#         # Separate chat history key to avoid internal collisions
#         chat_history_key="entity_history"
#     )
#     return CombinedMemory(memories=[summary, entity])

def _redis_keys(session_id: str) -> Tuple[str, str, str]:
    return (
        _HISTORY_KEY.format(sid=session_id),
        _SUMMARY_KEY.format(sid=session_id),
        _ENTITIES_KEY.format(sid=session_id),
    )

def _load_snapshot_from_redis(session_id: str) -> Tuple[Optional[str], Optional[dict], List[Tuple[str, str]]]:
    r = get_redis()
    if not r:
        return None, None, []
    h_key, s_key, e_key = _redis_keys(session_id)
    try:
        pipe = r.pipeline()
        pipe.get(s_key)
        pipe.get(e_key)
        pipe.lrange(h_key, 0, -1)
        s_val, e_val, hist = pipe.execute()
        summary = s_val.decode("utf-8") if s_val else None
        entities = json.loads(e_val) if e_val else None
        history: List[Tuple[str, str]] = []
        for item in hist or []:
            try:
                pair = json.loads(item)
                history.append((pair.get("input", ""), pair.get("output", "")))
            except Exception:
                continue
        return summary, entities, history
    except Exception:
        return None, None, []

def _persist_to_redis(session_id: str, input_text: str, output_text: str, summary: Optional[str], entities: Optional[dict]) -> None:
    r = get_redis()
    if not r:
        return
    h_key, s_key, e_key = _redis_keys(session_id)
    try:
        pipe = r.pipeline()
        # append history
        pipe.rpush(h_key, json.dumps({"input": input_text, "output": output_text}))
        # update snapshot
        if summary is not None:
            pipe.set(s_key, summary)
        if entities is not None:
            pipe.set(e_key, json.dumps(entities, ensure_ascii=False))
        pipe.execute()
    except Exception:
        pass

def _clear_redis(session_id: Optional[str] = None) -> None:
    r = get_redis()
    if not r:
        return
    try:
        if session_id is None:
            # no mass delete to avoid impacting other instances
            return
        h_key, s_key, e_key = _redis_keys(session_id)
        r.delete(h_key, s_key, e_key)
    except Exception:
        pass

def _build_gemini_tools_list():
    """Build tools list for Gemini function calling from config tool declarations.

    Returns a list in the format expected by the SDK:
    [{"function_declarations": [ ... ]}]
    """
    try:
        import config  # Local import to avoid heavy globals and circular deps
        tool_names = [
            "schedule_viewing_tool",
            "create_lead_tool",
            "property_search_tool",
            "search_new_launches_tool",
            "get_unit_details_tool",
            "insight_search_tool",
            "get_more_units_tool",
        ]
        function_declarations = []
        for name in tool_names:
            tool_decl = getattr(config, name, None)
            if isinstance(tool_decl, dict) and tool_decl.get("name"):
                function_declarations.append(tool_decl)

        if function_declarations:
            return [{"function_declarations": function_declarations}]
    except Exception as e:
        # Non-fatal: fall back to no-tools if anything goes wrong
        print(f"Warning: could not build Gemini tools list: {e}")
    return None


def get_chat_session(session_id: str) -> genai.ChatSession:
    """Return the chat session for the given session id."""
    if session_id is None:
        session_id = str(uuid.uuid4())  # Generate a default if None
    
    if session_id not in _session_chats:
        # Create new Gemini model and chat session with tool declarations
        tools = _build_gemini_tools_list()
        if tools:
            model = genai.GenerativeModel(variables.GEMINI_MODEL_NAME, tools=tools)
        else:
            model = genai.GenerativeModel(variables.GEMINI_MODEL_NAME)
        chat = model.start_chat(history=[])
        _session_chats[session_id] = chat
        
        # Load history from Redis if available (optional)
        # _summary, _entities, history = _load_snapshot_from_redis(session_id)
        # if history:
        #     # Note: Gemini chat sessions don't support loading history directly
        #     # We'll rely on the built-in memory instead
        #     pass
    
    return _session_chats[session_id]

# def get_memory(session_id: str) -> CombinedMemory:
#     """Return the memory instance for the given session id."""
#     if session_id is None:
#         session_id = str(uuid.uuid4())  # Generate a default if None
#     if session_id not in _session_memories:
#         mem = _build_memory()
#         # Seed from Redis snapshot by replaying history if present
#         _summary, _entities, history = _load_snapshot_from_redis(session_id)
#         if history:
#             for inp, out in history[-20:]:
#                 try:
#                     mem.save_context({"input": inp}, {"output": out})
#                 except Exception:
#                     break
#         _session_memories[session_id] = mem
#     return _session_memories[session_id]

class _MemoryFacade:
    """Facade that supports both object-style and callable-style usage.

    - Object-style: memory_manager.load_memory_variables(...), .save_context(...)
      operates on a global default session id "global".
    - Callable-style: memory_manager(session_id) -> ChatSession
    """

    def __init__(self) -> None:
        self._default_session_id = "global"

    def __call__(self, session_id: str | None) -> genai.ChatSession:
        return get_chat_session(session_id or self._default_session_id)

    # Proxy common memory API to the default session
    def load_memory_variables(self, inputs: dict) -> dict:
        # Route to specific session if provided
        session_id = None
        if isinstance(inputs, dict):
            session_id = inputs.pop("session_id", None)
        
        # Get chat session
        chat = self(session_id or self._default_session_id)
        
        # Return empty memory variables for compatibility
        # (Gemini chat sessions handle memory internally)
        return {
            "history": "",  # Gemini handles this internally
            "entities": {}   # Gemini handles this internally
        }

    def save_context(self, inputs: dict, outputs: dict) -> None:
        # Route to specific session if provided
        session_id = None
        if isinstance(inputs, dict):
            session_id = inputs.pop("session_id", None)
        
        # Get chat session
        chat = self(session_id or self._default_session_id)
        
        # Send message to chat session (this automatically saves to memory)
        try:
            user_message = inputs.get("input", "")
            if user_message:
                response = chat.send_message(user_message)
                # The response is automatically stored in chat history
                
                # Optional: Persist to Redis for backup
                # _persist_to_redis(session_id or self._default_session_id,
                #                   user_message,
                #                   response.text if response.text else "",
                #                   None,  # No summary needed
                #                   None)  # No entities needed
        except Exception as e:
            print(f"Error saving context to chat session: {e}")

    def clear(self) -> None:
        reset_memory(self._default_session_id)

    # Legacy no-op used by core_functions.update_conversation_context
    def update_search_results(self, session_id: str, results) -> None:  # noqa: ANN001
        # Intentionally a no-op for backward compatibility
        pass

# Export facade instance under the expected name
memory_manager = _MemoryFacade()

def reset_memory(session_id: str | None = None) -> None:
    """Reset memory for a specific session or all sessions.

    - If session_id is provided, rebuild that session's memory.
    - If None, clear all sessions.
    """
    if session_id is None:
        _session_chats.clear()
        # Do not global-delete Redis content from here
        return
    _session_chats.pop(session_id, None)
    _clear_redis(session_id)
    # Create new chat session (preserve tools)
    tools = _build_gemini_tools_list()
    if tools:
        model = genai.GenerativeModel(variables.GEMINI_MODEL_NAME, tools=tools)
    else:
        model = genai.GenerativeModel(variables.GEMINI_MODEL_NAME)
    chat = model.start_chat(history=[])
    _session_chats[session_id] = chat

def clear_memory(session_id: str) -> None:
    """Alias to reset a specific session's memory."""
    reset_memory(session_id)

__all__ = ["get_chat_session", "memory_manager", "reset_memory", "clear_memory"]
