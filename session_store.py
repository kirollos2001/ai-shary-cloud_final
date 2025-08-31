import os
import json
from typing import Dict, Any

from redis_utils import get_redis

SESSION_FILE = "client_sessions.json"
_REDIS_HASH = "sessions"


def _load_all_sessions_file() -> Dict[str, Any]:
    if not os.path.exists(SESSION_FILE):
        return {}
    try:
        with open(SESSION_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_all_sessions_file(sessions: Dict[str, Any]) -> None:
    with open(SESSION_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2, ensure_ascii=False)


def save_session(thread_id: str, client_info: Dict[str, Any]) -> None:
    r = get_redis()
    if r:
        try:
            r.hset(_REDIS_HASH, thread_id, json.dumps(client_info, ensure_ascii=False))
            return
        except Exception:
            pass
    # Fallback to file
    sessions = _load_all_sessions_file()
    sessions[thread_id] = client_info
    _save_all_sessions_file(sessions)


def get_session(session_id: str):
    r = get_redis()
    if r:
        try:
            raw = r.hget(_REDIS_HASH, session_id)
            if raw:
                return json.loads(raw)
        except Exception:
            pass
    sessions = _load_all_sessions_file()
    session = sessions.get(session_id)
    import logging
    logging.info(f"get_session({session_id}) -> {session}")
    return session
