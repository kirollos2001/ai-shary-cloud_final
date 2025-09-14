import os
import json
import logging
import time
import tempfile
from filelock import FileLock

# ملاحظة: بلاش import من config/db_operations على مستوى الملف لتجنب circular imports
# هنستورد جوّه الدوال بس عند الحاجة.

CACHE_DIR = "cache"
# إنشاء الفولدر بأمان حتى مع تعدد الـworkers (race-safe)
os.makedirs(CACHE_DIR, exist_ok=True)

# فلاج لتعطيل أي عمليات DB مؤقتًا (أثناء النشر الأول/التجربة)
SKIP_DB_INIT = os.getenv("SKIP_DB_INIT") == "1"

def _log_cache_length(filename):
    """Log the number of items stored in a cache file."""
    fpath = os.path.join(CACHE_DIR, filename)
    if os.path.exists(fpath):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            logging.info("Cache %s length: %s", filename, len(data))
        except Exception as e:
            logging.error("Failed to read %s: %s", fpath, e)
    else:
        logging.warning("%s not found at %s", filename, fpath)

_log_cache_length("units.json")
_log_cache_length("new_launches.json")

def _db_config_ok():
    """تأكد من توفر مفاتيح الـDB الأساسية قبل أي اتصال"""
    required = ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        logging.warning(f"DB config missing keys: {missing} — skipping DB ops.")
        return False
    return True

def save_to_cache(filename, data):
    path = os.path.join(CACHE_DIR, filename)
    lock = FileLock(path + ".lock")
    try:
        with lock:
            fd, tmp_path = tempfile.mkstemp(dir=CACHE_DIR)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
                    json.dump(data, tmp_file, ensure_ascii=False, indent=2, default=str)
                os.replace(tmp_path, path)
                logging.info(f"✅ Saved to cache file: {path}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
    except Exception as e:
        logging.error(f"❌ Failed to save {filename} to cache: {e}")

def load_from_cache(filename):
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
                    logging.warning(f"Decode error in {filename}, retrying once: {e}")
                    time.sleep(0.1)
                    continue
                logging.error(f"JSON decode error in {filename}: {e}")
                backup_path = path + f".corrupted_{int(time.time())}"
                try:
                    os.replace(path, backup_path)
                    logging.warning(f"Backed up corrupted JSON file to {backup_path}")
                except Exception as e2:
                    logging.error(f"Failed to backup corrupted JSON file: {e2}")
                return []
            except Exception as e:
                logging.error(f"Unexpected error loading {filename}: {e}")
                return []
    return []

def append_to_cache(filename, entry):
    path = os.path.join(CACHE_DIR, filename)
    data = load_from_cache(filename)
    data.append(entry)
    save_to_cache(filename, data)

def upsert_to_cache(filename, entry, key_field):
    """
    Insert or update an entry in cache based on a key field.
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

def enrich_units_with_names():
    query = """
    SELECT 
        u.id, u.name_ar, u.reference_no, u.apartment_area, u.price, u.max_price,
        u.installment_years, u.delivery_in,
        u.Bedrooms, u.Bathrooms, u.garages, u.address, u.sale_type,
        d.name_ar AS developer_name,
        c.name_ar AS compound_name,
        p.name_ar AS property_type_name,
        f.name_ar AS finishing_type_name,
        a.name_ar AS area_name,
        ci.name_ar AS city_name,
        co.name_ar AS country_name
    FROM units u
    LEFT JOIN developers d ON u.developer_id = d.id
    LEFT JOIN compounds c ON u.compound_id = c.id
    LEFT JOIN property_settings p ON u.property_id = p.id AND p.type = 'property'
    LEFT JOIN property_settings f ON u.finishing_type_id = f.id AND f.type = 'types_finishing'
    LEFT JOIN cities a ON u.area_id = a.id
    LEFT JOIN countries co ON u.country_id = co.id
    LEFT JOIN cities ci ON a.country_id = ci.country_id
    WHERE u.status = 1
    """
    if SKIP_DB_INIT or not _db_config_ok():
        logging.info("⏭️ Skipping enrich_units_with_names DB fetch (SKIP_DB_INIT or DB config missing).")
        return []
    from db_operations import fetch_data
    try:
        logging.info("🔍 Fetching enriched units from database...")
        units = fetch_data(query)
        logging.info(f"✅ Fetched {len(units)} enriched units from DB")
        if units:
            logging.info(f"🔢 Example unit: {units[0]}")
        return units
    except Exception as e:
        logging.error(f"❌ Failed to fetch units: {e}")
        return []

def cache_units_from_db():
    if SKIP_DB_INIT or not _db_config_ok():
        logging.info("⏭️ Skipping units DB fetch (SKIP_DB_INIT or DB config missing).")
        save_to_cache("units.json", [])
        return
    from config import get_db_connection
    cursor = None
    connection = None
    try:
        logging.info("🔍 Fetching enriched units from database with compound names and media...")
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        cursor.execute("""
            SELECT u.*, c.name_ar AS compound_name_ar, c.name_en AS compound_name_en,
                   c.image AS compound_image, c.video AS compound_video
            FROM units u
            LEFT JOIN compounds c ON u.compound_id = c.id
        """)
        units = cursor.fetchall()

        for unit in units:
            if unit.get('image'):
                unit['new_image'] = f"https://shary.eg/images/compounds/units/{unit['image']}"
            elif unit.get('compound_image'):
                unit['new_image'] = f"https://shary.eg/images/compounds/{unit['compound_image']}"
            else:
                unit['new_image'] = ""

        save_to_cache("units.json", units)
        logging.info(f"✅ Saved {len(units)} units to cache (with compound names, media, and new_image)")
    except Exception as e:
        logging.error(f"❌ Failed to fetch units: {e}")
        save_to_cache("units.json", [])
    finally:
        if cursor is not None:
            try:
                cursor.close()
            except Exception:
                pass
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass

def cache_devlopers_from_db():
    if SKIP_DB_INIT or not _db_config_ok():
        logging.info("⏭️ Skipping developers DB fetch (SKIP_DB_INIT or DB config missing).")
        save_to_cache("developers.json", [])
        return
    from db_operations import fetch_data
    try:
        developers = fetch_data("SELECT * FROM developers")
        save_to_cache("developers.json", developers)
        logging.info(f"✅ Saved {len(developers)} developers to cache")
    except Exception as e:
        logging.error(f"❌ Error caching developers: {e}")

def cache_new_launches_from_db():
    if SKIP_DB_INIT or not _db_config_ok():
        logging.info("⏭️ Skipping new launches DB fetch (SKIP_DB_INIT or DB config missing).")
        save_to_cache("new_launches.json", [])
        return
    from db_operations import fetch_data
    try:
        new_launches_query = """
        SELECT 
            nl.*,
            d.name_ar AS developer_name,
            c.name_ar AS compound_name_ar,
            c.name_en AS compound_name_en,
            c.image AS compound_image,
            c.video AS compound_video,
            pt.name_ar AS property_type_name,
            ci.name_ar AS city_name
        FROM new_launches nl
        LEFT JOIN compounds c ON nl.compound_id = c.id
        LEFT JOIN developers d ON c.developer_id = d.id
        LEFT JOIN property_settings pt ON c.property_id = pt.id AND pt.type = 'property'
        LEFT JOIN cities ci ON c.area_id = ci.id
        """
        new_launches = fetch_data(new_launches_query)

        for launch in new_launches:
            if launch.get('image'):
                launch['new_image'] = f"https://shary.eg/images/new_launch/{launch['image']}"
            else:
                launch['new_image'] = ""

        save_to_cache("new_launches.json", new_launches)
        logging.info(f"✅ Saved {len(new_launches)} new launches to cache (with compound names, media, and new_image)")
    except Exception as e:
        logging.error(f"❌ Error caching new launches: {e}")

def cache_leads_from_db():
    if SKIP_DB_INIT or not _db_config_ok():
        logging.info("⏭️ Skipping leads DB fetch (SKIP_DB_INIT or DB config missing).")
        save_to_cache("leads.json", [])
        return
    from db_operations import fetch_data
    try:
        leads = fetch_data("SELECT * FROM leads")
        save_to_cache("leads.json", leads)
        logging.info(f"✅ Cached {len(leads)} leads to leads.json")
    except Exception as e:
        logging.error(f"❌ Failed to cache leads: {e}")

def sync_leads_to_db():
    if SKIP_DB_INIT or not _db_config_ok():
        logging.info("⏭️ Skipping sync_leads_to_db (SKIP_DB_INIT or DB config missing).")
        return
    from db_operations import execute_query
    leads = load_from_cache("leads_updates.json")
    if not leads:
        logging.info("⚠️ No new leads to sync.")
        return

    for lead in leads:
        try:
            user_id = lead.get("user_id")
            existing = load_from_cache("leads.json")
            match = next((l for l in existing if l.get("user_id") == user_id), None)

            if match:
                # UPDATE
                fields = []
                values = []
                for field in ["name", "phone", "email", "property_preferences", "budget", "location", "property_type", "bedrooms", "bathrooms"]:
                    if lead.get(field) not in [None, "", 0]:
                        fields.append(f"{field} = %s")
                        values.append(lead[field])
                values.append(user_id)
                update_query = f"UPDATE leads SET {', '.join(fields)} WHERE user_id = %s"
                execute_query(update_query, tuple(values))
            else:
                # INSERT
                insert_query = """
                    INSERT INTO leads (user_id, name, phone, email, property_preferences, budget, location, property_type, bedrooms, bathrooms)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                values = (
                    user_id,
                    lead.get("name", ""),
                    lead.get("phone", ""),
                    lead.get("email", ""),
                    lead.get("property_preferences", ""),
                    lead.get("budget", 0),
                    lead.get("location", ""),
                    lead.get("property_type", ""),
                    lead.get("bedrooms", 0),
                    lead.get("bathrooms", 0),
                )
                execute_query(insert_query, values)

        except Exception as e:
            logging.error(f"❌ Failed to sync lead for user {lead.get('user_id')}: {e}")

    save_to_cache("leads_updates.json", [])
    logging.info(f"✅ Synced {len(leads)} leads to DB and cleared leads_updates.json")

def cache_conversations_from_db():
    if SKIP_DB_INIT or not _db_config_ok():
        logging.info("⏭️ Skipping conversations DB fetch (SKIP_DB_INIT or DB config missing).")
        save_to_cache("conversations.json", [])
        return
    from db_operations import fetch_data
    try:
        conversations = fetch_data("SELECT * FROM conversations")
        save_to_cache("conversations.json", conversations)
        logging.info(f"✅ Cached {len(conversations)} conversations")
    except Exception as e:
        logging.error(f"❌ Failed to cache conversations: {e}")

def sync_conversations_to_db():
    if SKIP_DB_INIT or not _db_config_ok():
        logging.info("⏭️ Skipping sync_conversations_to_db (SKIP_DB_INIT or DB config missing).")
        return
    from db_operations import execute_query
    convos = load_from_cache("conversations_updates.json")
    if not convos:
        logging.info("⚠️ No new conversations to sync.")
        return

    for convo in convos:
        try:
            insert_query = """
                INSERT INTO conversations (
                    conversation_id, user_id, description, created_at, updated_at
                )
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    description = VALUES(description),
                    updated_at = VALUES(updated_at)
            """
            values = (
                convo.get("conversation_id"),
                convo.get("user_id"),
                json.dumps(convo.get("description", []), ensure_ascii=False),
                convo.get("created_at"),
                convo.get("updated_at")
            )
            print("CREATED AT:", convo.get("created_at"))
            print("UPDATED AT:", convo.get("updated_at"))

            execute_query(insert_query, values)
        except Exception as e:
            logging.error(f"❌ Failed to sync conversation: {e}")

    save_to_cache("conversations_updates.json", [])
    logging.info(f"✅ Synced {len(convos)} conversations to DB and cleared conversations_updates.json")

