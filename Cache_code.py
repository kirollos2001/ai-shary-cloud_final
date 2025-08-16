import os
import json
import logging

from config import get_db_connection

CACHE_DIR = os.environ.get("CACHE_DIR", "/tmp/cache")
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR, exist_ok=True)

def save_to_cache(filename, data):
    try:
        path = os.path.join(CACHE_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        logging.info(f"✅ Saved to cache file: {path}")
    except Exception as e:
        logging.error(f"❌ Failed to save {filename} to cache: {e}")

def load_from_cache(filename):
    path = os.path.join(CACHE_DIR, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def append_to_cache(filename, entry):
    path = os.path.join(CACHE_DIR, filename)
    data = load_from_cache(filename)
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




import logging

def cache_units_from_db():
    cursor = None
    connection = None
    try:
        logging.info("🔍 Fetching enriched units from database with compound names and media...")
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        # Join units with compounds to get compound name_ar, name_en, image, and video
        cursor.execute("""
            SELECT u.*, c.name_ar AS compound_name_ar, c.name_en AS compound_name_en,
                   c.image AS compound_image, c.video AS compound_video
            FROM units u
            LEFT JOIN compounds c ON u.compound_id = c.id
        """)
        units = cursor.fetchall()

        # Add new_image field to each unit
        for unit in units:
            # Logic for units: 
            # If image is not null: https://shary.eg/images/compounds/units/ + image
            # If image is null: https://shary.eg/images/compounds/ + compound_image
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
        save_to_cache("cache/units.json", [])
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

def cache_devlopers_from_db():
    from db_operations import fetch_data
    try:
        developers = fetch_data("SELECT * FROM developers")
        save_to_cache("developers.json", developers)
        logging.info(f"✅ Saved {len(developers)} developers to cache")
    except Exception as e:
        logging.error(f"❌ Error caching developers: {e}")
def cache_new_launches_from_db():
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
        
        # Add new_image field to each new launch
        for launch in new_launches:
            # Logic for new launches: https://shary.eg/images/new_launch/ + image
            if launch.get('image'):
                launch['new_image'] = f"https://shary.eg/images/new_launch/{launch['image']}"
            else:
                launch['new_image'] = ""
        
        save_to_cache("new_launches.json", new_launches)
        logging.info(f"✅ Saved {len(new_launches)} new launches to cache (with compound names, media, and new_image)")
    except Exception as e:
        logging.error(f"❌ Error caching new launches: {e}")
def cache_leads_from_db():
    from db_operations import fetch_data
    try:
        leads = fetch_data("SELECT * FROM leads")
        save_to_cache("leads.json", leads)
        logging.info(f"✅ Cached {len(leads)} leads to leads.json")
    except Exception as e:
        logging.error(f"❌ Failed to cache leads: {e}")

def sync_leads_to_db():
    from db_operations import execute_query
    leads = load_from_cache("leads_updates.json")
    if not leads:
        logging.info("⚠️ No new leads to sync.")
        return

    for lead in leads:
        try:
            user_id = lead.get("user_id")
            existing = load_from_cache("leads.json")
            match = next((l for l in existing if l["user_id"] == user_id), None)

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
    from db_operations import fetch_data
    try:
        conversations = fetch_data("SELECT * FROM conversations")
        save_to_cache("conversations.json", conversations)
        logging.info(f"✅ Cached {len(conversations)} conversations")
    except Exception as e:
        logging.error(f"❌ Failed to cache conversations: {e}")



def sync_conversations_to_db():
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



