import os
import json
import logging
import time
import tempfile
from filelock import FileLock
from variables import CACHE_DIR
from google.cloud import storage
from google.cloud.exceptions import NotFound

# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# Google Cloud Storage Configuration
# =========================
GCS_BUCKET_NAME = "sharyai2025-cache"
GCS_CACHE_PREFIX = "cache/"  # Path in GCS where cache files are stored

# Initialize GCS client
try:
    gcs_client = storage.Client()
    gcs_bucket = gcs_client.bucket(GCS_BUCKET_NAME)
    logger.info(f"✅ Connected to GCS bucket: {GCS_BUCKET_NAME}")
except Exception as e:
    logger.warning(f"⚠️ Failed to initialize GCS client: {e}. Cache updates will remain local.")
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

def _upload_to_gcs(local_path: str, gcs_path: str) -> bool:
    """Upload a file from local filesystem to GCS."""
    if not gcs_bucket:
        logger.warning(f"⚠️ GCS client not initialized. Cannot upload {local_path} to {gcs_path}.")
        return False
    try:
        blob = gcs_bucket.blob(gcs_path)
        blob.upload_from_filename(local_path)
        logger.info(f"✅ Uploaded {local_path} to {gcs_path} (size: {os.path.getsize(local_path)} bytes)")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to upload {local_path} to {gcs_path}: {e}")
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
