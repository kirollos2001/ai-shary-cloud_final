import os
from typing import Optional

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover - redis might not be installed locally
    redis = None  # type: ignore


_client: Optional["redis.Redis"] = None


def get_redis() -> Optional["redis.Redis"]:
    """Return a singleton Redis client if configuration is available.

    Reads config from variables.py when present; otherwise from environment.
    Returns None if redis package is unavailable or configuration is missing/invalid.
    """
    global _client
    if _client is not None:
        return _client

    # Late import to avoid circulars
    try:
        import variables  # type: ignore
    except Exception:
        variables = None  # type: ignore

    if redis is None:
        return None

    url = None
    host = None
    port = None
    db = None
    password = None

    # Prefer variables.py
    if variables is not None:
        url = getattr(variables, "REDIS_URL", None)
        host = getattr(variables, "REDIS_HOST", None)
        port = getattr(variables, "REDIS_PORT", None)
        db = getattr(variables, "REDIS_DB", None)
        password = getattr(variables, "REDIS_PASSWORD", None)

    # Fallback to env
    url = url or os.environ.get("REDIS_URL")
    host = host or os.environ.get("REDIS_HOST")
    if port is None:
        port_env = os.environ.get("REDIS_PORT")
        port = int(port_env) if port_env else None
    if db is None:
        db_env = os.environ.get("REDIS_DB")
        db = int(db_env) if db_env else None
    password = password or os.environ.get("REDIS_PASSWORD")

    try:
        if url:
            client = redis.Redis.from_url(url, socket_timeout=1.0, socket_connect_timeout=1.0)
        elif host:
            client = redis.Redis(host=host, port=port or 6379, db=db or 0, password=password,
                                 socket_timeout=1.0, socket_connect_timeout=1.0)
        else:
            return None

        # Lightweight connectivity test
        try:
            client.ping()
        except Exception:
            return None

        _client = client
        return _client
    except Exception:
        return None

