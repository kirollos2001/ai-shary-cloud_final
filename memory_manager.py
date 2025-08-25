import os
from langchain.memory import ConversationSummaryMemory, ConversationEntityMemory, CombinedMemory
from langchain_google_genai import ChatGoogleGenerativeAI
import variables

# Set Gemini API key if available
api_key = getattr(variables, "GEMINI_API_KEY", None)
if api_key:
    os.environ["GEMINI_API_KEY"] = api_key

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model=variables.GEMINI_MODEL_NAME,
    google_api_key=api_key,
    temperature=0.7,
) if api_key else None


def _build_memory() -> CombinedMemory:
    """Create a combined memory using LangChain built-ins."""
    summary = ConversationSummaryMemory(llm=llm, memory_key="history", return_messages=False)
    entity = ConversationEntityMemory(llm=llm)
    return CombinedMemory(memories=[summary, entity])


_session_memories: dict[str, CombinedMemory] = {}


def get_memory(session_id: str) -> CombinedMemory:
    """Return the memory instance for the given session id."""
    if session_id not in _session_memories:
        _session_memories[session_id] = _build_memory()
    return _session_memories[session_id]


def memory_manager(session_id: str) -> CombinedMemory:
    """Backward compatible alias for get_memory."""
    return get_memory(session_id)


__all__ = ["get_memory", "memory_manager"]
