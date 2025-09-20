import os
import json
import logging
import time
import tempfile
from filelock import FileLock
from variables import (
    CACHE_DIR,
    UNITS_CACHE_FILE,
    NEW_LAUNCHES_CACHE_FILE,
    DEVELOPERS_CACHE_FILE,
    LEADS_CACHE_FILE,
    CONVERSATIONS_CACHE_FILE,
)

_GCS_IMPORT_ERROR = None

try:
    from google.cloud import storage
except ImportError as exc:  # pragma: no cover - depends on optional dependency
    storage = None  # type: ignore[assignment]
    _GCS_IMPORT_ERROR = exc
# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# If the optional google-cloud-storage dependency is missing, surface a clear warning
if _GCS_IMPORT_ERROR:
    logger.warning(
        "⚠️ google-cloud-storage is not installed: %s. "
        "Cache uploads to Google Cloud Storage will be skipped.",
        _GCS_IMPORT_ERROR,
    )

# =========================
# Google Cloud Storage Configuration
# =========================
GCS_BUCKET_NAME = "sharyai2025-cache"
GCS_CACHE_PREFIX = "cache/"  # Path in GCS where cache files are stored

# Initialize GCS client
if storage is None:
    gcs_bucket = None
else:
    try:
        gcs_client = storage.Client()
        gcs_bucket = gcs_client.bucket(GCS_BUCKET_NAME)
        logger.info(f"✅ Connected to GCS bucket: {GCS_BUCKET_NAME}")
    except Exception as e:
        logger.warning(
            f"⚠️ Failed to initialize GCS client: {e}. Cache updates will remain local."
        )
        gcs_bucket = None

# Ensure the cache directory exists at runtime
os.makedirs(CACHE_DIR, exist_ok=True)

# Flag to skip any DB operations
SKIP_DB_INIT = True

def _log_cache_length(filename):
    """Log the number of items stored in a cache file."""
    fpath = os.path.join(CACHE_DIR, filename)
    if os.path.exists(fpath):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info("Cache %s length: %s", filename, len(data))
        except Exception as e:
            logger.error("Failed to read %s: %s", fpath, e)
    else:
        logger.warning("%s not found at %s", filename, fpath)

_log_cache_length("units.json")
_log_cache_length("new_launches.json")

def _upload_cache_to_gcs():
    """Upload all files in CACHE_DIR to GCS."""
    if not gcs_bucket:
        logger.warning("⚠️ GCS client not initialized. Skipping cache upload to GCS.")
        return
    try:
        for file in os.listdir(CACHE_DIR):
            if file.endswith(".json"):  # Only upload JSON cache files
                local_path = os.path.join(CACHE_DIR, file)
                gcs_path = f"{GCS_CACHE_PREFIX}{file}"
                _upload_to_gcs(local_path, gcs_path)
        logger.info(
            f"✅ Uploaded cache files from {CACHE_DIR} to gs://{GCS_BUCKET_NAME}/{GCS_CACHE_PREFIX}"
        )
    except Exception as e:
        logger.error(f"❌ Failed to upload cache files to GCS: {e}")


def _download_cache_from_gcs(filename: str) -> bool:
    """Ensure a cache file is present locally by downloading it from GCS."""
    if not gcs_bucket:
        logger.warning(
            f"⚠️ GCS client not initialized. Cannot download {filename} from bucket."
        )
        return False

    gcs_path = f"{GCS_CACHE_PREFIX}{filename}"
    local_path = os.path.join(CACHE_DIR, filename)
    lock = FileLock(local_path + ".lock")

    try:
        with lock:
            blob = gcs_bucket.blob(gcs_path)
            if not blob.exists():
                logger.warning(
                    f"⚠️ Cache file {gcs_path} does not exist in GCS. Keeping local copy."
                )
                return False

            fd, tmp_path = tempfile.mkstemp(dir=CACHE_DIR)
            os.close(fd)
            try:
                blob.download_to_filename(tmp_path)
                os.replace(tmp_path, local_path)
                logger.info(
                    f"✅ Downloaded cache file gs://{GCS_BUCKET_NAME}/{gcs_path} -> {local_path}"
                )
                _log_cache_length(filename)
                return True
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
    except Exception as e:
        logger.error(f"❌ Failed to download cache file {filename} from GCS: {e}")
        return False


def _upload_cache_to_gcs():
    """Upload all files in CACHE_DIR to GCS."""
    if not gcs_bucket:
        logger.warning("⚠️ GCS client not initialized. Skipping cache upload to GCS.")
        return
    try:
        for file in os.listdir(CACHE_DIR):
            if file.endswith(".json"):  # Only upload JSON cache files
                local_path = os.path.join(CACHE_DIR, file)
                gcs_path = f"{GCS_CACHE_PREFIX}{file}"
                _upload_to_gcs(local_path, gcs_path)
        logger.info(f"✅ Uploaded cache files from {CACHE_DIR} to gs://{GCS_BUCKET_NAME}/{GCS_CACHE_PREFIX}")
    except Exception as e:
        logger.error(f"❌ Failed to upload cache files to GCS: {e}")

def save_to_cache(filename, data):
    """Save data to a cache file and upload to GCS."""
    path = os.path.join(CACHE_DIR, filename)
    lock = FileLock(path + ".lock")
    try:
        with lock:
            fd, tmp_path = tempfile.mkstemp(dir=CACHE_DIR)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
                    json.dump(data, tmp_file, ensure_ascii=False, indent=2, default=str)
                os.replace(tmp_path, path)
                logger.info(f"✅ Saved to cache file: {path}")
                # Upload the updated file to GCS
                _upload_to_gcs(path, f"{GCS_CACHE_PREFIX}{filename}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
    except Exception as e:
        logger.error(f"❌ Failed to save {filename} to cache: {e}")

def load_from_cache(filename):
    """Load data from a cache file."""
    path = os.path.join(CACHE_DIR, filename)
    lock = FileLock(path + ".lock")
    if os.path.exists(path):
        for attempt in range(2):
            try:
                with lock:
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                if attempt == 0:
                    logger.warning(f"Decode error in {filename}, retrying once: {e}")
                    time.sleep(0.1)
                    continue
                logger.error(f"JSON decode error in {filename}: {e}")
                backup_path = path + f".corrupted_{int(time.time())}"
                try:
                    os.replace(path, backup_path)
                    logger.warning(f"Backed up corrupted JSON file to {backup_path}")
                except Exception as e2:
                    logger.error(f"Failed to backup corrupted JSON file: {e2}")
                return []
            except Exception as e:
                logger.error(f"Unexpected error loading {filename}: {e}")
                return []
    return []

def append_to_cache(filename, entry):
    """Append a new entry to the cache file and upload to GCS."""
    path = os.path.join(CACHE_DIR, filename)
    data = load_from_cache(filename)
    data.append(entry)
    save_to_cache(filename, data)


def upsert_to_cache(filename, entry, key_field):
    """
    Insert or update an entry in cache based on a key field and upload to GCS.
    If an entry with the same key_field value exists, it will be updated.
    Otherwise, a new entry will be added.
    """
    path = os.path.join(CACHE_DIR, filename)
    data = load_from_cache(filename)

    # Find existing entry
    updated = False
    for i, existing in enumerate(data):
        if existing.get(key_field) == entry.get(key_field):
            data[i] = entry
            updated = True
            break

    # If not found, append new entry
    if not updated:
        data.append(entry)

    save_to_cache(filename, data)


def _ensure_cache_from_gcs(filename: str) -> bool:
    """Helper to hydrate a specific cache file from GCS when DB sync is disabled."""
    if not SKIP_DB_INIT:
        logger.info(
            f"SKIP_DB_INIT is disabled. Skipping GCS download for {filename} (expecting DB sync)."
        )
        return False

    success = _download_cache_from_gcs(filename)
    if not success:
        if os.path.exists(os.path.join(CACHE_DIR, filename)):
            logger.info(
                f"Using existing local cache for {filename} after failed GCS download."
            )
            return True
        logger.warning(
            f"⚠️ Cache file {filename} not found locally and failed to download from GCS."
        )
        return False
    return True


def cache_units_from_db():
    """Populate units cache. In cloud mode we rely on GCS instead of hitting MySQL."""
    if _ensure_cache_from_gcs(UNITS_CACHE_FILE):
        logger.info("✅ Units cache hydrated from GCS.")
    else:
        logger.warning("⚠️ Units cache not refreshed; using existing data if available.")


def cache_new_launches_from_db():
    """Populate new launches cache from the pre-generated GCS snapshot."""
    if _ensure_cache_from_gcs(NEW_LAUNCHES_CACHE_FILE):
        logger.info("✅ New launches cache hydrated from GCS.")
    else:
        logger.warning("⚠️ New launches cache not refreshed; using existing data if available.")


def cache_devlopers_from_db():
    """Populate developers cache from GCS snapshot (no DB access in cloud mode)."""
    if _ensure_cache_from_gcs(DEVELOPERS_CACHE_FILE):
        logger.info("✅ Developers cache hydrated from GCS.")
    else:
        logger.warning("⚠️ Developers cache not refreshed; using existing data if available.")


def cache_leads_from_db():
    """Ensure leads cache exists locally (download snapshot when available)."""
    if _ensure_cache_from_gcs(LEADS_CACHE_FILE):
        logger.info("✅ Leads cache hydrated from GCS.")
    else:
        logger.info(
            "ℹ️ Leads cache not downloaded from GCS. It will be created locally when leads arrive."
        )


def cache_conversations_from_db():
    """Ensure conversations cache exists locally (download snapshot when available)."""
    if _ensure_cache_from_gcs(CONVERSATIONS_CACHE_FILE):
        logger.info("✅ Conversations cache hydrated from GCS.")
    else:
        logger.info(
            "ℹ️ Conversations cache not downloaded from GCS. It will be created locally when needed."
        )


def sync_leads_to_db():
    """Placeholder for DB sync (disabled in GCS-first deployments)."""
    if SKIP_DB_INIT:
        logger.info("⏭️ Skipping leads DB sync in GCS cache mode.")
        return False
    logger.warning(
        "⚠️ Leads DB sync requested but SKIP_DB_INIT is False and no implementation is available."
    )
    return False


def sync_conversations_to_db():
    """Placeholder for DB sync (disabled in GCS-first deployments)."""
    if SKIP_DB_INIT:
        logger.info("⏭️ Skipping conversations DB sync in GCS cache mode.")
        return False
    logger.warning(
        "⚠️ Conversations DB sync requested but SKIP_DB_INIT is False and no implementation is available."
    )
    return False

def main():
    """Main function to demonstrate cache operations (for testing)."""
    logger.info("🚀 Starting cache manager...")
    # Example: Load and log cache contents
    units = load_from_cache("units.json")
    new_launches = load_from_cache("new_launches.json")
    logger.info(f"📊 Cache stats: units={len(units)}, new_launches={len(new_launches)}")
    # Example: Append a dummy entry and sync to GCS
    dummy_unit = {"id": "test123", "name_en": "Test Unit", "name_ar": "وحدة اختبار"}
    append_to_cache("units.json", dummy_unit)
    logger.info("✅ Appended dummy unit and synced to GCS")
    # Example: Upsert a dummy entry and sync to GCS
    dummy_launch = {"id": "launch123", "name_en": "Test Launch", "name_ar": "إطلاق اختبار"}
    upsert_to_cache("new_launches.json", dummy_launch, "id")
    logger.info("✅ Upserted dummy launch and synced to GCS")

if __name__ == "__main__":
    main()


