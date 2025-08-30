import os
import uuid  # Added for default session_id
from typing import Dict
from langchain.memory import ConversationSummaryMemory, ConversationEntityMemory, CombinedMemory
from langchain_google_genai import ChatGoogleGenerativeAI
import variables

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
llm = ChatGoogleGenerativeAI(
    model=variables.GEMINI_MODEL_NAME,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7,
    timeout=60,
    max_retries=1,  # قلّل الانتظار لو حصل Rate Limit
)

# (اختياري) Debug بسيط علشان تتأكد إنه شغّال بمفتاح API مش بمشروع GCP
print("Using API key only? ->", bool(GOOGLE_API_KEY) and not any(
    os.getenv(v) for v in [
        "GOOGLE_APPLICATION_CREDENTIALS","GOOGLE_CLOUD_PROJECT",
        "GCLOUD_PROJECT","GOOGLE_CLOUD_QUOTA_PROJECT","GOOGLE_PROJECT_ID"
    ]
))


def _build_memory() -> CombinedMemory:
    """Create a combined memory using LangChain built-ins."""
    if llm is None:
        raise ValueError("LLM not initialized; GEMINI_API_KEY is required.")
    
    # Use unique keys to avoid CombinedMemory collisions; we'll map back to
    # the legacy keys (history/entities) in the facade.
    summary = ConversationSummaryMemory(
        llm=llm,
        memory_key="summary_history",
        input_key="input",
        output_key="output",
        return_messages=False
    )
    entity = ConversationEntityMemory(
        llm=llm,
        memory_key="entities",
        input_key="input",
        output_key="output",
        # Separate chat history key to avoid internal collisions
        chat_history_key="entity_history"
    )
    return CombinedMemory(memories=[summary, entity])


_session_memories: Dict[str, CombinedMemory] = {}


def get_memory(session_id: str) -> CombinedMemory:
    """Return the memory instance for the given session id."""
    if session_id is None:
        session_id = str(uuid.uuid4())  # Generate a default if None
    if session_id not in _session_memories:
        _session_memories[session_id] = _build_memory()
    return _session_memories[session_id]

class _MemoryFacade:
    """Facade that supports both object-style and callable-style usage.

    - Object-style: memory_manager.load_memory_variables(...), .save_context(...)
      operates on a global default session id "global".
    - Callable-style: memory_manager(session_id) -> CombinedMemory
    """

    def __init__(self) -> None:
        self._default_session_id = "global"

    def __call__(self, session_id: str | None) -> CombinedMemory:
        return get_memory(session_id or self._default_session_id)

    # Proxy common memory API to the default session
    def load_memory_variables(self, inputs: dict) -> dict:
        # Route to specific session if provided
        session_id = None
        if isinstance(inputs, dict):
            session_id = inputs.pop("session_id", None)
        raw = self(session_id or self._default_session_id).load_memory_variables(inputs)
        # Map internal keys back to expected legacy keys
        mapped = {}
        if "summary_history" in raw:
            mapped["history"] = raw.get("summary_history", "")
        if "entities" in raw:
            mapped["entities"] = raw.get("entities", {})
        # Preserve any other keys as-is
        for k, v in raw.items():
            if k not in ("summary_history", "entities"):
                mapped[k] = v
        return mapped

    def save_context(self, inputs: dict, outputs: dict) -> None:
        # Route to specific session if provided
        session_id = None
        if isinstance(inputs, dict):
            session_id = inputs.pop("session_id", None)
        self(session_id or self._default_session_id).save_context(inputs, outputs)

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
        _session_memories.clear()
        return
    _session_memories[session_id] = _build_memory()


def clear_memory(session_id: str) -> None:
    """Alias to reset a specific session's memory."""
    reset_memory(session_id)


__all__ = ["get_memory", "memory_manager", "reset_memory", "clear_memory"]
