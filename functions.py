import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import google.generativeai as genai
import re
import os
import datetime
from Cache_code import load_from_cache,append_to_cache,save_to_cache
from datetime import datetime
import json
import variables

from variables import EMAIL_HOST, EMAIL_PORT, EMAIL_USER, EMAIL_PASSWORD, TEAM_EMAIL




def extract_client_preferences(user_message):

    extracted_info = {
        "property_preferences": "",
        "budget": 0,
        "location": "",
        "property_type": "",
        "bedrooms": 0,
        "bathrooms": 0
    }

    # Extract budget
    budget_match = re.search(r"(\d+(?:,\d{3})*)\s*(دولار|مليون|ألف|جنيه)", user_message)
    if budget_match:
        amount = int(budget_match.group(1).replace(",", ""))
        unit = budget_match.group(2)
        extracted_info["budget"] = amount * 1_000_000 if unit == "مليون" else amount * 1_000

    # Extract location (e.g., التجمع الخامس, 6 أكتوبر)
    locations = ["العبور","الرياض","دبي ","التجمع الخامس","مدينة نصر","مصر الجديده","المعادي","الرحاب","الشيخ زايد"," مدينتي", "6 أكتوبر","العاصمه الإداريه", "العاصمة الإدارية", "الساحل الشمالي"]
    for loc in locations:
        if loc in user_message:
            extracted_info["location"] = loc
            break

    # Extract property type (e.g., شقة, فيلا)
    property_types = ["تجاري","مطعم","عياده","محل","شقه","شقة", "فيلا", "دوبلكس", "بنتهاوس"]
    for ptype in property_types:
        if ptype in user_message:
            extracted_info["property_type"] = ptype
            break

    # Extract bedrooms
    bedrooms_match = re.search(r"(\d+)\s*غرف", user_message)
    if bedrooms_match:
        extracted_info["bedrooms"] = int(bedrooms_match.group(1))

    # Extract bathrooms
    bathrooms_match = re.search(r"(\d+)\s*حمام", user_message)
    if bathrooms_match:
        extracted_info["bathrooms"] = int(bathrooms_match.group(1))

    return extracted_info


def send_email(to_email, subject, body):
    """
    Send an email to the specified address.
    """
    msg = MIMEMultipart()
    msg['From'] = EMAIL_USER
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        print(f"📧 Attempting to send email to: {to_email}")
        print(f"📧 Subject: {subject}")
        server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_USER, to_email, msg.as_string())
        server.quit()
        print(f"✅ Email sent successfully to: {to_email}")
        return True
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False
from session_store import get_session

def schedule_viewing(arguments):
    logging.info(f"Received arguments: {arguments}")

    # Get conversation_id (aka thread_id)
    conversation_id = arguments.get('conversation_id')
    session_info = get_session(conversation_id) if conversation_id else {}

    # Prioritize data from arguments, fallback to session
    client_id = arguments.get('client_id') or session_info.get('user_id')
    name = arguments.get('name') or session_info.get('name', 'Unknown')
    phone = arguments.get('phone') or session_info.get('phone', 'Unknown')
    email = arguments.get('email') or session_info.get('email', 'Unknown')

    property_id = arguments.get('property_id', 'Not Provided')
    desired_date = arguments.get('desired_date')
    desired_time = arguments.get('desired_time')
    meeting_type = arguments.get('meeting_type')

    if not desired_date or not desired_time or not meeting_type:
        return {
            "message": "📅 من فضلك اختر تاريخ ووقت للمعاينة، وهل تفضل الاجتماع عبر زووم أم زيارة ميدانية؟"
        }

    developer_name = get_developer_name_from_database(property_id) or "Unknown Developer"
    property_name = get_property_name_from_database(property_id) or "Unknown Property"

    summary = advanced_conversation_summary_from_db(client_id, conversation_id, name, property_id)

    subject = f"🔔 معاينة وحدة جديدة - ID {property_id}"
    body = f"""
    📝 معلومات العميل:
    - client id : {client_id}
    - Name : {name}
    - Phone : {phone}
    - Email: {email}
    - property_id :{property_id}
    - Unit Name: {property_name}
    - Devloper : {developer_name}
    - Meeting type: {meeting_type}
    - Date : {desired_date}
    - Time : {desired_time}
    
    # 🔎 ملخص المحادثة:
    {summary}
    """

    logging.info(f"Prepared email body: {body}")
    print(f"📧 Sending appointment email to: {TEAM_EMAIL}")
    import threading
    threading.Thread(target=send_email, args=(TEAM_EMAIL, subject, body)).start()

    return {
        "message": "✅ تم حجز موعد المعاينة بنجاح!",
        "client_id": client_id,
        "property_id": property_id,
        "developer": developer_name,
        "date": desired_date,
        "time": desired_time,
        "meeting_type": meeting_type,
    }
# def retrieve_lead_info(client_id):
#     """Retrieve client details from the leads table in MySQL instead of Excel."""
#     if not client_id:
#         return None
#
#     query = "SELECT * FROM leads WHERE user_id = %s"
#     params = (client_id,)
#     result = db_operations.fetch_data(query, params)
#
#     if result:
#         print("✅ Lead info fetched from MySQL")
#         return result[0]  # Return first matched lead as a dictionary
#
#     print("❌ Lead not found in MySQL")
#     return None

def get_developer_name_from_database(property_id):
    try:
        units = load_from_cache("units.json")
        for unit in units:
            if str(unit.get("id")) == str(property_id):
                return unit.get("developer_name", "Unknown Developer")
    except Exception as e:
        print(f"⚠️ Error loading developer name from cache: {e}")
    return "Unknown Developer"


def get_property_name_from_database(property_id):
    try:
        units = load_from_cache("units.json")
        for unit in units:
            if str(unit.get("id")) == str(property_id):
                return unit.get("name_ar", "Unknown Property")
    except Exception as e:
        print(f"⚠️ Error loading property name from cache: {e}")
    return "Unknown Property"

def search_new_launches(arguments):
    """
    Search new launches via Chroma semantic search only (no metadata filters).
    Returns up to 50 results ranked by semantic relevance.
    """
    print("🚀 search_new_launches function called!")
    print(f"🚀 Arguments: {arguments}")
    
    # Test if function is being called at all
    if not arguments:
        return {"error": "No arguments provided"}
    
    try:
        # Build semantic query from arguments (text-only)
        property_type = str(arguments.get("property_type", "")).strip().lower()
        location = str(arguments.get("location", "")).strip().lower()
        compound = str(arguments.get("compound", "")).strip().lower()

        semantic_parts = []
        if property_type:
            semantic_parts.append(property_type)
        if location:
            semantic_parts.append(location)
        if compound:
            semantic_parts.append(compound)
        
        query_text = " ".join(semantic_parts) if semantic_parts else "new launch properties"

        # Use RAG (Chroma) semantic search for new launches
        from chroma_rag_setup import get_rag_instance
        rag = get_rag_instance()
        rag_results = rag.search_new_launches(query_text, n_results=50, filters=None)

        if not rag_results:
                return {
                "source": "chromadb_semantic_search_new_launches",
                "message": f"❌ لا توجد إطلاقات جديدة مطابقة لمواصفاتك{(' في ' + location) if location else ''}.",
                    "results": []
                }
            
        # Format results for clean UI (ID | Name | City | Compound)
        def _format_launch(item, idx):
            meta = item.get('metadata', {})
            doc = item.get('document', '')
            
            # Extract information from document text
            name_val = "غير متوفر"
            city_val = "غير متوفر"
            compound_val = "غير متوفر"
            
            # Extract name from document
            if 'Name (AR):' in doc:
                name_val = doc.split('Name (AR):')[1].split('\n')[0].strip()
            elif 'Name (EN):' in doc:
                name_val = doc.split('Name (EN):')[1].split('\n')[0].strip()
            
            # Extract city from document
            if 'City:' in doc:
                city_val = doc.split('City:')[1].split('\n')[0].strip()
            
            # Extract compound from document
            if 'Compound (AR):' in doc:
                compound_val = doc.split('Compound (AR):')[1].split('\n')[0].strip()
            elif 'Compound (EN):' in doc:
                compound_val = doc.split('Compound (EN):')[1].split('\n')[0].strip()
            
            # Get launch_id from metadata
            launch_id = meta.get('launch_id', 'غير متوفر')
            
            return f"{idx}. ID:{launch_id} | {name_val} | المدينة: {city_val} | الكمبوند: {compound_val}"

        # Format all results first
        all_formatted_lines = [_format_launch(itm, i+1) for i, itm in enumerate(rag_results[:10])]
        
        # Apply LLM re-filtering to keep only most relevant results
        def _llm_filter_new_launches(lines, prop_type, loc):
            print("🔍 ENTERING LLM FILTER FUNCTION")
            print(f"🔍 LLM filter function called with {len(lines)} lines")
            try:
                if not lines or len(lines) <= 3:
                    print(f"🔍 Skipping LLM filter - only {len(lines)} results")
                    return lines  # Keep all if 3 or fewer results
                
                # Build prompt for LLM filtering
                numbered = "\n".join([f"{i+1}. {line}" for i, line in enumerate(lines)])
                intent_desc = f"property_type='{prop_type}' | location='{loc}'"
                
                prompt = f"""
                أنت مساعد عقاري ذكي. قم بمراجعة القائمة التالية من إطلاقات المشاريع (New Launches).
                
                المطلوب: اختر فقط العناصر الأكثر صلة بنية العميل (نوع العقار والموقع).
                نية العميل: {intent_desc}
                
                القائمة:
                {numbered}
                
                ارجع فقط JSON array لأرقام العناصر (1-based indices) بالترتيب، بدون أي نص آخر.
                مثال: [1,3,5]
                
                اختر 3-4 عناصر فقط الأكثر صلة.
                """
                
                print(f"🤖 Sending LLM prompt: {prompt[:200]}...")
                
                # Use Gemini for filtering
                import os
                import google.generativeai as genai
                from variables import GEMINI_API_KEY
                
                genai.configure(api_key=GEMINI_API_KEY)
                model = genai.GenerativeModel('gemini-2.0-flash-exp')
                
                response = model.generate_content(prompt)
                text = (response.text or "").strip()
                
                print(f"🤖 LLM response: {text}")
                
                # Extract JSON array
                import json
                start = text.find('[')
                end = text.rfind(']')
                indices = []
                if start != -1 and end != -1 and end > start:
                    json_part = text[start:end+1]
                    try:
                        indices = json.loads(json_part)
                        print(f"🔍 Parsed indices: {indices}")
                    except Exception as e:
                        print(f"🚨 JSON parsing failed: {e}")
                        indices = []
                
                # Map to lines
                filtered = []
                for idx in indices:
                    try:
                        i = int(idx) - 1
                        if 0 <= i < len(lines):
                            filtered.append(lines[i])
                    except Exception:
                        continue
            
                print(f"🔍 Filtered to {len(filtered)} results")
                
                # Fallback if empty - return top 3
                return filtered if filtered else lines[:3]
                
            except Exception as e:
                print(f"🚨 LLM filter failed: {e}")
                # Fallback to top 3 results
                return lines[:3]
        
        # Apply LLM filtering
        print(f"🔍 Applying LLM filtering for property_type='{property_type}', location='{location}'")
        print(f"📊 Before filtering: {len(all_formatted_lines)} results")
        
        # Simple test - just return top 3 for now to verify the flow works
        if len(all_formatted_lines) > 3:
            print("🔍 More than 3 results, applying LLM filtering...")
            filtered_lines = _llm_filter_new_launches(all_formatted_lines, property_type, location)
        else:
            print("🔍 3 or fewer results, keeping all")
            filtered_lines = all_formatted_lines
        
        print(f"📊 After filtering: {len(filtered_lines)} results")
        
        # Compute similarity-like scores from chroma distance (not shown in UI)
        launch_similarity_scores = []
        for itm in rag_results[:len(filtered_lines)]:
            try:
                dist = float(itm.get('distance', 0))
                sim = 1.0 - dist
                if sim < 0:
                    sim = 0.0
                if sim > 1:
                    sim = 1.0
                launch_similarity_scores.append(round(sim, 4))
            except Exception:
                launch_similarity_scores.append(None)
        
            return {
            "source": "chromadb_semantic_search_new_launches",
            "message": f"✅ لقيت {len(filtered_lines)} إطلاقات جديدة مطابقة لمواصفاتك 👇",
            "results": filtered_lines,
            "follow_up": "لو حابب تفاصيل أكتر عن إطلاق معين ابعتلي رقم الـ ID 🔍",
            "similarity_scores": launch_similarity_scores
        }

    except Exception as e:
        print(f"🚨 Error in search_new_launches (RAG): {e}")
        return {
            "error": f"❌ خطأ في البحث عن الإطلاقات الجديدة: {str(e)}"
        }


def create_lead(data):
    user_id = data.get("user_id")
    if not user_id:
        return {"success": False, "message": "User ID is required."}

    # 1️⃣ Load all leads from daily cache
    cached_leads = load_from_cache("leads_cache.json")
    user_lead = next((lead for lead in cached_leads if str(lead.get("user_id")) == str(user_id)), None)

    fields = ["name", "phone", "email", "property_preferences", "budget", "location", "property_type", "bedrooms", "bathrooms"]

    if user_lead:
        # 2️⃣ Update logic: only update changed fields
        updated = False
        for field in fields:
            new_val = data.get(field)
            if new_val and new_val != user_lead.get(field):
                user_lead[field] = new_val
                updated = True

        if updated:
            # Update cached leads in memory
            for i, lead in enumerate(cached_leads):
                if str(lead.get("user_id")) == str(user_id):
                    cached_leads[i] = user_lead
                    break

            # Save updated leads cache
            from Cache_code import save_to_cache
            save_to_cache("leads_cache.json", cached_leads)

            # Also append to update queue for later DB sync
            append_to_cache("leads_updates.json", user_lead)
            logging.info(f"✅ Lead updated in cache and queued for DB update: {user_id}")
        else:
            logging.info(f"⚠️ No changes to update for cached lead: {user_id}")
    else:
        # 3️⃣ New lead: append to both caches
        new_lead = {
            "user_id": user_id,
            "name": data.get("name", ""),
            "phone": data.get("phone", ""),
            "email": data.get("email", ""),
            "property_preferences": data.get("property_preferences", ""),
            "budget": data.get("budget", 0),
            "location": data.get("location", ""),
            "property_type": data.get("property_type", ""),
            "bedrooms": data.get("bedrooms", 0),
            "bathrooms": data.get("bathrooms", 0)
        }
        append_to_cache("leads_cache.json", new_lead)
        append_to_cache("leads_updates.json", new_lead)
        logging.info(f"🆕 New lead cached and queued for DB insert: {user_id}")

    return {"success": True, "message": "Lead cached successfully."}
#
# def get_city_aliases_from_db():
#     query = "SELECT id, name_en, name_ar FROM cities"
#     cities = db_operations.fetch_data(query)
#     city_map = {}
#
#     for city in cities:
#         # Create a keyword dictionary entry with various lowercase aliases
#         aliases = [
#             city['name_en'].lower(),
#             city['name_ar'].lower(),
#             city['name_en'].lower().replace("city", "").strip(),
#             city['name_en'].lower().replace(" ", ""),
#             city['name_en'].lower().replace("october", "6th of october"),
#             city['name_en'].lower().replace("october", "6 october"),
#         ]
#         aliases = list(set(aliases))  # remove duplicates
#         city_map[city['id']] = aliases
#     return city_map
#
# def find_city_ids_for_location(location_input, city_map):
#     location_input = location_input.strip().lower()
#     matched_city_ids = []
#
#     for city_id, aliases in city_map.items():
#         for alias in aliases:
#             if alias in location_input:
#                 matched_city_ids.append(city_id)
#                 break  # stop checking aliases for this city once matched
#
#     return matched_city_ids
#
def clean_price_string(price_str):
    if price_str is None:
        return 0.0
    try:
        return float(str(price_str).replace(",", "").replace("\xa0", "").strip())
    except ValueError:
        return 0.0

def mmr_search(results, query_embedding, lambda_param=0.8, top_k=10):
    """
    Implement Maximal Marginal Relevance (MMR) search to diversify results.
    
    Args:
        results (list): List of property results
        query_embedding (list): Query embedding vector
        lambda_param (float): Balance between relevance (λ) and diversity (1-λ)
        top_k (int): Number of top results to return
    
    Returns:
        list: Diversified results using MMR
    """
    try:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        if not results or len(results) <= 1:
            return results
        
        # Simple text-based similarity calculation (fallback when embeddings aren't available)
        def calculate_similarity(item1, item2):
            """Calculate similarity between two property items based on text features"""
            features1 = f"{item1.get('compound_name', '')} {item1.get('name_ar', '')} {item1.get('location', '')}"
            features2 = f"{item2.get('compound_name', '')} {item2.get('name_ar', '')} {item2.get('location', '')}"
            
            # Convert to lowercase and split into words
            words1 = set(features1.lower().split())
            words2 = set(features2.lower().split())
            
            # Calculate Jaccard similarity
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0
        
        def calculate_query_similarity(item):
            """Calculate similarity between query and property item"""
            query_features = f"{item.get('compound_name', '')} {item.get('name_ar', '')} {item.get('location', '')}"
            query_words = set(query_features.lower().split())
            
            # Simple relevance score based on feature matching
            relevance_score = 0
            if item.get('location'):
                relevance_score += 0.3
            if item.get('compound_name'):
                relevance_score += 0.2
            if item.get('name_ar'):
                relevance_score += 0.2
            if item.get('price'):
                relevance_score += 0.1
            if item.get('Bedrooms'):
                relevance_score += 0.1
            if item.get('delivery_type'):
                relevance_score += 0.1
            
            return relevance_score
        
        # Initialize MMR selection
        selected_indices = []
        remaining_indices = list(range(len(results)))
        
        # Select first item (most relevant)
        first_item_idx = max(remaining_indices, key=lambda i: calculate_query_similarity(results[i]))
        selected_indices.append(first_item_idx)
        remaining_indices.remove(first_item_idx)
        
        # Select remaining items using MMR
        for _ in range(min(top_k - 1, len(remaining_indices))):
            mmr_scores = []
            
            for idx in remaining_indices:
                # Relevance score
                relevance = calculate_query_similarity(results[idx])
                
                # Diversity score (average similarity to already selected items)
                diversity = 0
                if selected_indices:
                    similarities = [calculate_similarity(results[idx], results[sel_idx]) for sel_idx in selected_indices]
                    diversity = np.mean(similarities)
                
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
                mmr_scores.append((idx, mmr_score))
            
            # Select item with highest MMR score
            best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        # Return results in MMR order
        return [results[i] for i in selected_indices]
        
    except Exception as e:
        print(f"🚨 Error in MMR search: {e}")
        # Fallback to original results
        return results[:top_k]


def property_search(arguments):
    """
    Search for existing property units using ChromaDB with MMR search and semantic search.
    This function does NOT use cache - it searches directly from ChromaDB.
    """
    try:
        # Mandatory fields
        location = arguments.get("location", "").strip().lower()
        budget = arguments.get("budget")
        
        # Optional fields
        property_type = arguments.get("property_type", "").strip().lower()
        bedrooms = arguments.get("bedrooms")
        bathrooms = arguments.get("bathrooms")
        apartment_area = arguments.get("apartment_area")
        # Optional preferred compound (accept both 'compound' and 'compound_name')
        compound_name = (arguments.get("compound") or arguments.get("compound_name") or "").strip()
        
        # Progressive search parameters
        price_tolerance = arguments.get("price_tolerance", 0)  # 0%, 10%, 20%, 25%
        excluded_ids = arguments.get("excluded_ids", [])  # IDs to exclude from results
        search_round = arguments.get("search_round", 1)  # 1st, 2nd, or 3rd round

        # Enforce required fields for existing units flow
        missing_fields = []
        if not location:
            missing_fields.append("الموقع")
        # Keep original value for validation before normalization
        raw_budget = budget
        if raw_budget in (None, "", 0):
            missing_fields.append("الميزانية")
        if not property_type:
            missing_fields.append("نوع العقار")

        if missing_fields:
            return {
                "source": "validation",
                "message": "محتاج منك معلومات أساسية قبل ما أبدأ البحث: " + ", ".join(missing_fields) + ".\n" \
                           + "- مثال للميزانية: 4,000,000 أو 4 مليون\n" \
                           + "- ممكن كمان تقولّي كمبوند مفضل لو حابب (اختياري)",
                "results": []
            }
        
        # Convert budget to float if it's a string
        try:
            budget = float(budget) if budget else 0
        except ValueError:
            budget = 0
        
        # Apply price tolerance for progressive search
        if price_tolerance > 0:
            tolerance_factor = 1 + (price_tolerance / 100)
            max_budget = budget * tolerance_factor
            min_budget = budget / tolerance_factor
        else:
            # First search: 10% tolerance
            tolerance_factor = 1.1  # 10%
            max_budget = budget * tolerance_factor
            min_budget = budget / tolerance_factor
        
        # Build search query for ChromaDB semantic search
        search_query_parts = []
        if location:
            search_query_parts.append(f"location: {location}")
        if property_type:
            search_query_parts.append(f"property type: {property_type}")
        if compound_name:
            search_query_parts.append(f"compound: {compound_name}")
        if budget > 0:
            search_query_parts.append(f"budget around {budget}")
        if bedrooms:
            search_query_parts.append(f"{bedrooms} bedrooms")
        if bathrooms:
            search_query_parts.append(f"{bathrooms} bathrooms")
        if apartment_area:
            search_query_parts.append(f"area around {apartment_area}")
        
        # Create semantic search query
        search_query = " ".join(search_query_parts) if search_query_parts else "property units"
        
        try:
            # Import and use the proper ChromaDB RAG system
            from chroma_rag_setup import get_rag_instance
            # MMR is now handled directly in the RAG pipeline
            
            # Initialize RAG system
            rag = get_rag_instance()
            
            # Build filters for ChromaDB query
            filters = {}
            # Apply early price filter with 10% tolerance (or exact range if provided)
            if budget and budget > 0:
                budget_str = str(arguments.get("budget", ""))
                if "-" in budget_str and any(unit in budget_str.lower() for unit in ["مليون", "million", "m"]):
                    import re
                    m = re.search(r"(\d+)\s*-\s*(\d+)", budget_str)
                    if m:
                        min_val = int(m.group(1)) * 1_000_000
                        max_val = int(m.group(2)) * 1_000_000
                        filters["price_min"] = min_val
                        filters["price_max"] = max_val
                else:
                    filters["price_min"] = int(min_budget)
                    filters["price_max"] = int(max_budget)
            # Pass semantic hints for reranker (not used in Chroma where clause)
            if location:
                filters["query_location"] = location
            if property_type:
                filters["query_property_type"] = property_type
            if compound_name:
                filters["query_compound"] = compound_name
            
            # Perform semantic search using ChromaDB with MMR
            # Get more results initially for MMR processing
            initial_results = rag.search_units(search_query, n_results=50, filters=filters)
            
            if not initial_results:
                return {
                    "source": "no_relevant_results",
                    "message": "آسف يا فندم، مش لاقي وحدات مطابقة للمواصفات اللي طلبتها في الوقت الحالي.\n\n" \
                               "تقدر تتواصل مع فريق المبيعات مباشرة للحصول على أحدث العروض المتاحة:\n" \
                               "📱 واتساب: https://wa.me/201000730208",
                    "results": []
                }
            
            # Convert ChromaDB results to the format expected by the system
            formatted_results = []
            for result in initial_results:
                # Extract the document content and metadata
                doc_content = result.get('document', '')
                metadata = result.get('metadata', {})
                
                # Create a result item that matches the expected format
                result_item = {
                    'id': metadata.get('unit_id', 'unknown'),
                    'name_en': doc_content,  # Use the document content
                    'name_ar': doc_content,  # Use the document content for Arabic too
                    'compound_name': metadata.get('compound_name', ''),
                    'location': metadata.get('location', ''),
                    'property_type': metadata.get('property_type', ''),
                    'price': metadata.get('price_value', 0),
                    'bedrooms': metadata.get('bedrooms', 0),
                    'bathrooms': metadata.get('bathrooms', 0),
                    'apartment_area': metadata.get('apartment_area', 0),
                    'delivery_in': metadata.get('delivery_in', ''),
                    'installment_years': metadata.get('installment_years', ''),
                    'new_image': metadata.get('new_image', ''),
                    'chroma_score': result.get('distance', 0)
                }
                formatted_results.append(result_item)
            
            # Results are already MMR-optimized from the RAG pipeline
            diversified_results = formatted_results
            
            # Apply additional filtering based on search criteria
            filtered_results = []
            
            def check_price_match(budget, arguments, item):
                """Helper function to check if price matches budget with proper tolerance/ranges"""
                try:
                    price = float(item.get("price", 0) or 0)
                    budget_str = str(arguments.get("budget", ""))
                    if "-" in budget_str and any(unit in budget_str.lower() for unit in ["مليون", "million", "m"]):
                        import re
                        range_match = re.search(r"(\d+)\s*-\s*(\d+)", budget_str)
                        if range_match:
                            min_val = int(range_match.group(1)) * 1_000_000
                            max_val = int(range_match.group(2)) * 1_000_000
                            return min_val <= price <= max_val
                    # Use dynamic tolerance based on search round
                    return min_budget <= price <= max_budget
                except Exception:
                    return False
            
            for item in diversified_results:
                # Exclude previously shown units for progressive search
                item_id = str(item.get('id', ''))
                if item_id in excluded_ids:
                    continue
                
                # Apply price filtering since client needs specific budget
                if budget and budget > 0:
                    if not check_price_match(budget, arguments, item):
                        continue

                # Location and property type matching is handled by semantic search (embeddings)
                # No additional filtering needed as ChromaDB's semantic search handles this accurately

                filtered_results.append(item)
            
            # -------------------- Format Results for UI --------------------
            def format_unit(item, index):
                # Extract a clean Arabic name and basic meta
                doc_text = item.get('name_ar', '') or item.get('name_en', '')
                name_ar = ""
                if 'Name (AR):' in doc_text:
                    name_ar = doc_text.split('Name (AR):')[1].split('\n')[0].strip()
                if not name_ar:
                    name_ar = doc_text.split('\n')[0][:120]
                price_val = item.get('price', 0)
                price_str = f"{int(price_val):,}" if isinstance(price_val, (int, float)) and price_val else "غير متوفر"
                bedrooms = item.get('bedrooms', 'غير متوفر')
                bathrooms = item.get('bathrooms', 'غير متوفر')
                unit_id = item.get('id', 'غير متوفر') or item.get('unit_id', 'غير متوفر')
                return f"{index}. ID:{unit_id} | {name_ar} | السعر: {price_str} EGP | غرف: {bedrooms} | حمام: {bathrooms}"
            
            if not filtered_results:
                return {
                    "source": "no_relevant_results",
                    "message": "آسف يا فندم، مش لاقي وحدات مطابقة للمواصفات اللي طلبتها في الوقت الحالي.\n\n" \
                               "تقدر تتواصل مع فريق المبيعات مباشرة للحصول على أحدث العروض المتاحة:\n" \
                               "📱 واتساب: https://wa.me/201000730208",
                    "results": []
                }
            
            # Quality check: Ensure results are truly relevant
            quality_filtered_results = []
            for result in filtered_results:
                # Check if result has minimum quality indicators
                result_text = str(result.get('name_ar', '') or result.get('name_en', '') or result.get('document', '')).lower()
                
                # Location quality check - must contain location reference
                location_match = False
                if location:
                    location_terms = [location.lower(), location.replace('ال', '').lower()]
                    location_match = any(term in result_text for term in location_terms if term)
                else:
                    location_match = True  # No location specified, skip check
                
                # Price quality check - must be reasonable
                price_match = True  # We already filtered by price earlier
                
                # Only include if quality checks pass
                if location_match and price_match:
                    quality_filtered_results.append(result)
            
            # Final quality check: If we have very few quality results, apologize instead
            if len(quality_filtered_results) < 3:
                return {
                    "source": "low_quality_results",
                    "message": "آسف يا فندم، مش لاقي وحدات كافية تطابق مواصفاتك بدقة في الوقت الحالي.\n\n" \
                               "تقدر تتواصل مع فريق المبيعات مباشرة للحصول على أحدث العروض المتاحة:\n" \
                               "📱 واتساب: https://wa.me/201000730208",
                    "results": []
                }
            
            # Prepare formatted list (max 10)
            formatted_lines = [format_unit(itm, i+1) for i, itm in enumerate(quality_filtered_results[:10])]
            
            # Analyze actual compounds found in results to generate accurate message
            def generate_accurate_intro_message(results, location, requested_compound):
                """Generate intro message based on actual results, not requested criteria"""
                try:
                    # Extract unique compounds from actual results
                    found_compounds = set()
                    for result in results[:10]:  # Only check top 10 displayed results
                        compound_ar = result.get('compound_name_ar', '').strip()
                        compound_en = result.get('compound_name_en', '').strip()
                        compound_name = compound_ar or compound_en
                        if compound_name and compound_name.lower() != 'غير متوفر':
                            found_compounds.add(compound_name)
                    
                    # Generate message based on what was actually found
                    location_text = location if location else 'المنطقة المطلوبة'
                    
                    if len(found_compounds) == 0:
                        compound_text = "داخل كمبوندات مختلفة"
                    elif len(found_compounds) == 1:
                        compound_text = f"داخل كمبوند {list(found_compounds)[0]}"
                    elif len(found_compounds) <= 3:
                        compound_list = list(found_compounds)[:3]
                        compound_text = f"داخل كمبوندات {' و '.join(compound_list)}"
                    else:
                        compound_text = "داخل كمبوندات مختلفة"
                    
                    return (
                        f"👋 أهلاً بحضرتك!\nلقيتلك وحدات مناسبة في {location_text} "
                        f"{compound_text}، وفي حدود ميزانيتك وعدد الغرف والحمامات اللي طلبتهم 👇"
                    )
                except Exception as e:
                    logging.warning(f"Error generating intro message: {e}")
                    # Fallback to simple message without compound mention
                    return (
                        f"👋 أهلاً بحضرتك!\nلقيتلك وحدات مناسبة في {location if location else 'المنطقة المطلوبة'} "
                        "داخل كمبوندات مختلفة، وفي حدود ميزانيتك وعدد الغرف والحمامات اللي طلبتهم 👇"
                    )
            
            intro_msg = generate_accurate_intro_message(quality_filtered_results, location, compound_name)
            # Compute similarity-like scores from chroma distance (not shown in UI)
            similarity_scores = []
            for itm in quality_filtered_results[:10]:
                try:
                    dist = float(itm.get('chroma_score', 0))
                    sim = 1.0 - dist
                    if sim < 0:
                        sim = 0.0
                    if sim > 1:
                        sim = 1.0
                    similarity_scores.append(round(sim, 4))
                except Exception:
                    similarity_scores.append(None)
            
                # Progressive search follow-up message
                if search_round == 1:
                    follow_up_msg = "✨ \"تحب أوريك تفاصيل أكتر عن وحدة معينة؟ ابعتلي رقم الـID اللي شد انتباهك 🔍\nأو تحب أعرضلك خيارات إضافية من الوحدات المتاحة؟\""
                elif search_round == 2:
                    follow_up_msg = "✨ \"تحب أوريك تفاصيل أكتر عن وحدة معينة؟ ابعتلي رقم الـID اللي شد انتباهك 🔍\nأو تحب أعرضلك المزيد من الخيارات؟ \""
                else:
                    follow_up_msg = "✨ \"تحب أوريك تفاصيل أكتر عن وحدة معينة؟ ابعتلي رقم الـID اللي شد انتباهك 🔍\nأو تحب أعرضلك المزيد من الخيارات؟ \""
                # Store search parameters in session for progressive search (first round only)
                if search_round == 1:
                    import config
                    session_id_for_storage = arguments.get('session_id', 'default')
                    session_key = f"{session_id_for_storage}_search_history"
                    shown_ids = [str(item.get('id', '')) for item in quality_filtered_results[:10]]
                    
                    # Debug: Print session storage
                    logging.info(f"💾 Storing search history with session_id: {session_id_for_storage}")
                    logging.info(f"💾 Session key: {session_key}")
                    logging.info(f"💾 Shown IDs: {shown_ids[:5]}...")
                    
                    config.client_sessions[session_key] = {
                        "last_search_params": {
                            "location": location,
                            "budget": budget,
                            "property_type": property_type,
                            "bedrooms": bedrooms,
                            "bathrooms": bathrooms,
                            "compound_name": compound_name,
                            "session_id": arguments.get('session_id', 'default')
                        },
                        "search_round": 1,
                        "shown_ids": shown_ids
                    }
                
                return {
                    "source": "chromadb_semantic_search_mmr",
                    "message": intro_msg,
                    "results": formatted_lines,
                    "follow_up": follow_up_msg,
                    "similarity_scores": similarity_scores
                }
            
        except ImportError as e:
            print(f"🚨 Error importing ChromaDB RAG: {e}")
            return {
                "source": "error",
                "message": "❌ خطأ في نظام البحث - يرجى المحاولة مرة أخرى لاحقاً.",
                "results": []
            }
        except Exception as e:
            print(f"🚨 Error in ChromaDB search: {e}")
            return {
                "source": "error",
                "message": "❌ خطأ في البحث - يرجى المحاولة مرة أخرى لاحقاً.",
                "results": []
            }

    except Exception as e:
        print(f"🚨 Error in property_search: {e}")
        return {"error": str(e)}


def serialize_mysql_result(results):
    """
    Convert MySQL result datetimes or other non-serializable types to JSON-serializable formats.
    """
    for row in results:
        for key, value in row.items():
            if isinstance(value, (datetime.date, datetime.datetime)):
                row[key] = value.strftime('%Y-%m-%d %H:%M:%S')
    return results



from datetime import datetime

def log_conversation_to_db(conversation_id, user_id, message):
    try:
        conversations = load_from_cache("conversations_cache.json")

        # Find existing conversation or create a new one
        convo = next((c for c in conversations if c["conversation_id"] == conversation_id), None)
        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if not convo:
            convo = {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "description": [],
                "created_at": now_str,
                "updated_at": now_str
            }
            conversations.append(convo)
        else:
            convo["updated_at"] = now_str  # Update timestamp

        # Append new message
        convo["description"].append({
            "sender": "Client" if user_id != "bot" else "Bot",
            "message": message,
            "timestamp": now_str
        })

        # Save to cache in background (non-blocking)
        try:
            save_to_cache("conversations_cache.json", conversations)
            append_to_cache("conversations_updates.json", convo)
            logging.info(f"📝 Updated conversation thread: {conversation_id}")
        except Exception as e:
            logging.warning(f"Cache save failed: {e}")
            
    except Exception as e:
        logging.error(f"Error in log_conversation_to_db: {e}")



def advanced_conversation_summary_from_db(client_id, conversation_id, name="Unknown", property_id="Unknown"):
    """
    Fetch conversation from cache (new structure), summarize it using OpenAI, and return the summary.
    """
    try:
        # 1️⃣ Load cached conversations
        conversations = load_from_cache("conversations_cache.json")

        # 2️⃣ Find the matching conversation - try multiple lookup strategies
        convo = None
        
        # Strategy 1: Exact match on conversation_id and user_id
        convo = next(
            (c for c in conversations if str(c["conversation_id"]) == str(conversation_id) and str(c["user_id"]) == str(client_id)),
            None
        )
        
        # Strategy 2: If not found, try just conversation_id match
        if not convo:
            convo = next(
                (c for c in conversations if str(c["conversation_id"]) == str(conversation_id)),
                None
            )
        
        # Strategy 3: If still not found, try user_id match
        if not convo:
            convo = next(
                (c for c in conversations if str(c["user_id"]) == str(client_id)),
                None
            )

        if not convo:
            # Debug: Log what we're looking for and what's available
            logging.warning(f"🔍 Conversation lookup failed:")
            logging.warning(f"   Looking for: conversation_id='{conversation_id}', client_id='{client_id}'")
            logging.warning(f"   Available conversations: {len(conversations)}")
            if conversations:
                sample_convo = conversations[0]
                logging.warning(f"   Sample conversation: conversation_id='{sample_convo.get('conversation_id')}', user_id='{sample_convo.get('user_id')}'")
            # Return a basic summary when no conversation is found
            return f"ملخص المحادثة: العميل {name} (ID: {client_id}) طلب معاينة للعقار {property_id}. لم يتم العثور على محادثة مفصلة في الذاكرة المؤقتة."

        conversation_data = convo.get("description", [])
        if not isinstance(conversation_data, list):
            conversation_data = json.loads(conversation_data)

        # 3️⃣ Format conversation for prompt
        formatted_conversation = "\n".join(
            f"{msg['sender']}: {msg['message']}" for msg in conversation_data
        )

        # 4️⃣ Create Arabic prompt for summarization with enhanced system instructions
        prompt = f"""
        أنت مساعد مبيعات عقاري محترف ومتخصص في تلخيص المحادثات العقارية. مهمتك هي تحليل المحادثة التالية وتقديم ملخص شامل ومفيد لفريق المبيعات.

        **تعليمات النظام:**
        - استخدم اللهجة المصرية المحترفة
        - ركز على المعلومات العملية والمفيدة لفريق المبيعات
        - اكتب الملخص بشكل منظم وواضح
        - استخدم النقاط والتنسيق المناسب
        - اذكر الأرقام والتواريخ بدقة

        **المعلومات المطلوبة في الملخص:**
        1. **معلومات العميل الأساسية:**
           - نوع العميل (جديد/متكرر)
           - مستوى الاهتمام (عالٍ/متوسط/منخفض)

        2. **متطلبات العقار:**
           - نوع العقار المطلوب (شقة/فيلا/تجاري)
           - الموقع المفضل
           - الميزانية المتوقعة
           - عدد الغرف والحمامات
           - المساحة المطلوبة

        3. **تفضيلات إضافية:**
           - الكمبوند المفضل
           - نوع الاستلام (جاهز/تحت الإنشاء)
           - نظام الدفع المفضل
           - الغرض من الشراء (سكن/استثمار)

        4. **نقاط مهمة:**
           - أي اعتراضات أو مخاوف
           - المواعيد المفضلة للمعاينة
           - أي طلبات خاصة
           - مستوى الاستعجال

        5. **الخطوات التالية المقترحة:**
           - نوع المتابعة المطلوبة
           - أفضل وقت للتواصل

        **المحادثة:**
        {formatted_conversation}

        **الملخص:**
        """

        print(f"📋 Prompt for summarization: {prompt}")

        # 5️⃣ Generate summary with Gemini
        try:
            # Configure Gemini with API key
            api_key = os.environ.get('GEMINI_API_KEY')
            if not api_key:
                return "❌ GEMINI_API_KEY environment variable is not set"
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(variables.GEMINI_MODEL_NAME)
            
            # Create the prompt for Gemini with enhanced system instructions
            gemini_prompt = f"""
            أنت مساعد مبيعات عقاري محترف ومتخصص في تلخيص المحادثات العقارية. مهمتك هي تحليل المحادثة التالية وتقديم ملخص شامل ومفيد لفريق المبيعات.

            **تعليمات النظام:**
            - استخدم اللهجة المصرية المحترفة
            - ركز على المعلومات العملية والمفيدة لفريق المبيعات
            - اكتب الملخص بشكل منظم وواضح
            - استخدم النقاط والتنسيق المناسب
            - اذكر الأرقام والتواريخ بدقة

            **المعلومات المطلوبة في الملخص:**
            1. **معلومات العميل الأساسية:**
               - نوع العميل (جديد/متكرر)
               - مستوى الاهتمام (عالٍ/متوسط/منخفض)

            2. **متطلبات العقار:**
               - نوع العقار المطلوب (شقة/فيلا/تجاري)
               - الموقع المفضل
               - الميزانية المتوقعة
               - عدد الغرف والحمامات
               - المساحة المطلوبة

            3. **تفضيلات إضافية:**
               - الكمبوند المفضل
               - نوع الاستلام (جاهز/تحت الإنشاء)
               - نظام الدفع المفضل
               - الغرض من الشراء (سكن/استثمار)

            4. **نقاط مهمة:**
               - أي اعتراضات أو مخاوف
               - المواعيد المفضلة للمعاينة
               - أي طلبات خاصة
               - مستوى الاستعجال

            5. **الخطوات التالية المقترحة:**
               - نوع المتابعة المطلوبة
               - أفضل وقت للتواصل

            **المحادثة:**
            {formatted_conversation}

            **الملخص:**
            """
            
            response = model.generate_content(gemini_prompt)
            summary = response.text.strip()
            print(f"📝 Generated summary: {summary}")
            return summary
            
        except Exception as e:
            print(f"🚨 Error generating summary with Gemini: {e}")
            return f"❌ حصل خطأ أثناء تلخيص المحادثة: {e}"

    except Exception as e:
        print(f"🚨 Error generating summary: {e}")
        return f"❌ حصل خطأ أثناء تلخيص المحادثة: {e}"


def classify_query_type_with_llm(user_query):
    """
    صنف السؤال إذا كان يبحث عن "إطلاق جديد" أو "وحدة موجودة" أو "غير محدد" باستخدام Gemini LLM.
    Returns: 'new_launch', 'existing_unit', or 'unspecified'
    """
    import os
    import variables
    try:
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            # Fallback to direct variables
            api_key = variables.GEMINI_API_KEY
            
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(variables.GEMINI_MODEL_NAME)
        
        # Enhanced prompt with clear definitions and examples
        prompt = f"""
        أنت مساعد عقاري ذكي. قم بتصنيف استفسار العميل بناءً على التعريفات التالية:

        🔴 الوحدات الموجودة (Existing Units):
        - وحدات تحت الإنشاء أو تحتاج تشطيب
        - وحدات جاهزة للاستلام والسكن الفوري
        - يمكن معاينتها شخصياً قبل الشراء
        - الأسعار محددة ومعروفة
        - متوفرة في units.json
        - يمكن الانتقال إليها قريباً
        - متوفرة حالياً للشراء

        🟢 الإطلاقات الجديدة (New Launches):
        - وحدات أعلن عنها لأول مرة
        - في مرحلة الحجز الأولي قبل البناء
        - الأسعار النهائية قد تكون غير معلنة
        - يمكن الحجز مبكراً بأسعار أقل
        - متوفرة في new_launches.json
        - مشاريع مستقبلية
        - للاستثمار على المدى الطويل
        - لم تبدأ بعد في البناء

        🎯 تحليل النية والهدف:
        - إذا كان العميل يريد الاستثمار أو مشاريع مستقبلية → new_launch
        - إذا كان العميل يريد الانتقال أو السكن قريباً → existing_unit
        - إذا كان العميل يريد خصومات أو أسعار أولية → new_launch
        - إذا كان العميل يريد معاينة أو شراء فوري → existing_unit
        - إذا كان العميل يريد ما هو متاح حالياً → existing_unit
        - إذا كان العميل يريد ما هو متاح للحجز → new_launch

        السؤال: "{user_query}"

        قم بتحليل نية العميل والهدف من طلبه، ثم صنف الاستفسار:
        التصنيف: [new_launch/existing_unit/unspecified]
        السبب: [سبب التصنيف بناءً على نية العميل]
        """
        
        response = model.generate_content(prompt)
        result = response.text.strip()
        
        print(f"🔍 Gemini response: {result}")
        
        # Parse the response more intelligently
        lines = result.split('\n')
        classification = "unspecified"
        
        # Look for classification in the response
        for line in lines:
            line = line.strip().lower()
            # Check for new_launch indicators
            if any(term in line for term in ['new_launch', 'إطلاق جديد', 'إطلاق', 'جديد', 'مشروع جديد']):
                classification = "new_launch"
                break
            # Check for existing_unit indicators
            elif any(term in line for term in ['existing_unit', 'وحدة موجودة', 'موجودة', 'جاهز', 'متوفر']):
                classification = "existing_unit"
                break
        
        # If still unspecified, try to find the classification in the entire response
        if classification == "unspecified":
            result_lower = result.lower()
            if any(term in result_lower for term in ['new_launch', 'إطلاق جديد', 'إطلاق', 'جديد', 'مشروع جديد']):
                classification = "new_launch"
            elif any(term in result_lower for term in ['existing_unit', 'وحدة موجودة', 'موجودة', 'جاهز', 'متوفر']):
                classification = "existing_unit"
        
        # If still unspecified, try to infer from keywords in the query
        if classification == "unspecified":
            query_lower = user_query.lower()
            
            # New launch indicators
            new_launch_keywords = [
                'إطلاق جديد', 'حجز أولي', 'مشروع جديد', 'حجز مبكر', 
                'قبل البناء', 'أسعار أولية', 'حجز مبكر', 'مشروع قادم',
                'new launch', 'early booking', 'pre-construction', 'off-plan',
                'مستقبلي', 'استثمار', 'خصم', 'مبكر', 'قادم'
            ]
            
            # Existing unit indicators
            existing_unit_keywords = [
                'جاهز للاستلام', 'معاينة', 'تشطيب', 'سعر محدد', 
                'متوفر الآن', 'يمكن المعاينة', 'جاهز للسكن',
                'ready for delivery', 'inspection', 'finishing', 'available now',
                'الآن', 'قريباً', 'فوراً', 'حالياً', 'متوفر'
            ]
            
            # Check for new launch keywords
            if any(keyword in query_lower for keyword in new_launch_keywords):
                classification = "new_launch"
            # Check for existing unit keywords
            elif any(keyword in query_lower for keyword in existing_unit_keywords):
                classification = "existing_unit"
            # Default to existing units for general property searches
            else:
                # If user is asking about general properties without specific new launch intent
                general_property_keywords = ['شقة', 'فيلا', 'كمبوند', 'عقار', 'وحدة', 'مشروع']
                if any(keyword in query_lower for keyword in general_property_keywords):
                    classification = "existing_unit"
        
        print(f"🔍 Query classification: '{user_query}' → {classification}")
        return classification
        
    except Exception as e:
        print(f"🚨 Error classifying query type with Gemini: {e}")
        # Fallback to keyword-based classification
        return classify_query_type_by_keywords(user_query)

def classify_query_type_by_keywords(user_query):
    """
    Fallback classification using keyword matching when LLM fails
    """
    query_lower = user_query.lower()
    
    # New launch indicators
    new_launch_keywords = [
        'إطلاق جديد', 'حجز أولي', 'مشروع جديد', 'حجز مبكر', 
        'قبل البناء', 'أسعار أولية', 'حجز مبكر', 'مشروع قادم',
        'new launch', 'early booking', 'pre-construction', 'off-plan'
    ]
    
    # Existing unit indicators
    existing_unit_keywords = [
        'جاهز للاستلام', 'معاينة', 'تشطيب', 'سعر محدد', 
        'متوفر الآن', 'يمكن المعاينة', 'جاهز للسكن',
        'ready for delivery', 'inspection', 'finishing', 'available now'
    ]
    
    # Check for new launch keywords
    if any(keyword in query_lower for keyword in new_launch_keywords):
        return "new_launch"
    
    # Check for existing unit keywords
    if any(keyword in query_lower for keyword in existing_unit_keywords):
        return "existing_unit"
    
    # Default to existing units for general property searches
    general_property_keywords = ['شقة', 'فيلا', 'كمبوند', 'عقار', 'وحدة']
    if any(keyword in query_lower for keyword in general_property_keywords):
        return "existing_unit"
    
    return "unspecified"

def classify_insight_intent_llm(user_query: str) -> str:
    """
    Use LLM to classify non-explicit insight questions into:
    - average_price
    - compound_list
    - cheapest_unit
    - other
    """
    try:
        import os
        import google.generativeai as genai
        model_name = variables.GEMINI_MODEL_NAME
        api_key = os.environ.get('GEMINI_API_KEY') or variables.GEMINI_API_KEY
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)

        prompt = f"""
        صنّف السؤال التالي إلى واحدة فقط من الفئات:
        - average_price: إذا كان يسأل عن متوسط الأسعار
        - compound_list: إذا كان يسأل عن أسماء/قائمة الكمبوندات
        - cheapest_unit: إذا كان يسأل عن الأرخص
        - other: غير ذلك

        السؤال: "{user_query}"

        أعد الإجابة بكلمة واحدة فقط من الخيارات: average_price أو compound_list أو cheapest_unit أو other
        """
        resp = model.generate_content(prompt)
        text = (resp.text or '').strip().lower()
        for label in ["average_price", "compound_list", "cheapest_unit", "other"]:
            if label in text:
                return label
        return "other"
    except Exception:
        # Fallback: conservative default
        uq = user_query.lower()
        if "متوسط" in uq or "average" in uq:
            return "average_price"
        if "كمبوند" in uq or "compound" in uq or "الكومبوند" in uq:
            return "compound_list"
        if "ارخص" in uq or "الأرخص" in uq or "الارخص" in uq or "cheapest" in uq:
            return "cheapest_unit"
        return "other"

def route_property_search(user_query, search_arguments):
    """
    Route property search to appropriate data source based on LLM classification.
    This function intelligently determines whether to search new launches or existing units.
    
    Args:
        user_query: User's search query
        search_arguments: Dictionary containing search parameters (budget, location, etc.)
    
    Returns:
        Dictionary with search results and source information
    """
    try:
        # First, classify the query type
        query_type = classify_query_type_with_llm(user_query)
        
        print(f"🔍 Query '{user_query}' classified as: {query_type}")
        
        # Route based on classification
        if query_type == "new_launch":
            print("🎯 Routing to new launches search...")
            return search_new_launches(search_arguments)
            
        elif query_type == "existing_unit":
            print("🎯 Routing to existing units search...")
            return property_search(search_arguments)
            
        else:
            # If unspecified, try to make an educated guess based on context
            print("🤔 Query type unspecified, making educated guess...")
            
            # Check if user explicitly mentioned new launch concepts
            query_lower = user_query.lower()
            new_launch_indicators = [
                'إطلاق جديد', 'حجز أولي', 'مشروع جديد', 'حجز مبكر',
                'قبل البناء', 'أسعار أولية', 'مشروع قادم'
            ]
            
            if any(indicator in query_lower for indicator in new_launch_indicators):
                print("🎯 Detected new launch intent, routing to new launches...")
                return search_new_launches(search_arguments)
            else:
                # Default to existing units for general property searches
                print("🎯 Defaulting to existing units search...")
                return property_search(search_arguments)
                
    except Exception as e:
        print(f"🚨 Error in route_property_search: {e}")
        # Fallback to existing units search
        return property_search(search_arguments)

def smart_property_search(user_query, search_arguments):
    """
    Enhanced property search that automatically determines the best search strategy.
    Combines classification with intelligent fallbacks and result merging.
    
    Args:
        user_query: User's search query
        search_arguments: Dictionary containing search parameters
    
    Returns:
        Dictionary with comprehensive search results
    """
    try:
        # Get the primary classification
        primary_type = classify_query_type_with_llm(user_query)
        
        results = {
            "query_type": primary_type,
            "primary_results": None,
            "secondary_results": None,
            "message": "",
            "source": "smart_search"
        }
        
        if primary_type == "new_launch":
            # Search new launches first
            new_launch_results = search_new_launches(search_arguments)
            results["primary_results"] = new_launch_results
            
            # Also search existing units as backup if new launches don't have enough results
            if not new_launch_results.get("results") or len(new_launch_results["results"]) < 3:
                existing_results = property_search(search_arguments)
                results["secondary_results"] = existing_results
                results["message"] = f"✅ {new_launch_results.get('message', '')} (مع اقتراحات للوحدات الموجودة)"
            else:
                results["message"] = new_launch_results.get("message", "")
                
        elif primary_type == "existing_unit":
            # Search existing units first
            existing_results = property_search(search_arguments)
            results["primary_results"] = existing_results
            
            # Also search new launches as alternative option
            new_launch_results = search_new_launches(search_arguments)
            results["secondary_results"] = new_launch_results
            results["message"] = f"✅ {existing_results.get('message', '')} (مع خيارات للإطلاقات الجديدة)"
            
        else:
            # Try both and let user choose
            existing_results = property_search(search_arguments)
            new_launch_results = search_new_launches(search_arguments)
            
            results["primary_results"] = existing_results
            results["secondary_results"] = new_launch_results
            results["message"] = "🔍 إليك نتائج البحث من كلا النوعين: الوحدات الموجودة والإطلاقات الجديدة"
        
        return results
        
    except Exception as e:
        print(f"🚨 Error in smart_property_search: {e}")
        return {
            "error": f"حدث خطأ أثناء البحث: {str(e)}",
            "query_type": "error",
            "source": "smart_search"
        }


def insight_search(arguments):
    """
    Answer non-explicit queries (statistics, lists, availability) strictly from ChromaDB data.
    Strategy: Retrieve ALL available data first, then let LLM analyze and filter insights.
    No hallucinations. If zero data, apologize and include WhatsApp link.
    """
    try:
        query_text = arguments.get("query", "")
        location = (arguments.get("location") or "").strip()
        property_type = (arguments.get("property_type") or "").strip()

        # Prefer LLM-based extraction of preferences over rigid keywords
        if not location or not property_type:
            try:
                llm_prefs = extract_client_preferences_llm(
                    user_message=query_text,
                    conversation_history=None,
                    current_preferences=None,
                    conversation_path="available_units"
                )
                if not property_type:
                    property_type = llm_prefs.get("property_type", "").strip()
                if not location:
                    location = llm_prefs.get("location", "").strip()
            except Exception:
                pass

        from chroma_rag_setup import get_rag_instance
        rag = get_rag_instance()

        # Strategy: Retrieve ALL available data first with broad search
        # Then let LLM analyze and filter the results
        
        # Build broader semantic query to get maximum relevant data
        semantic_parts = []
        if property_type:
            semantic_parts.append(property_type)
        if location:
            semantic_parts.append(location)
        semantic_q = " ".join(semantic_parts) if semantic_parts else query_text

        # Retrieve maximum data with minimal filters to get complete dataset
        # Use broader fetch to ensure we don't miss relevant data
        results = rag.search_units(semantic_q, n_results=2000, filters={})
        
        # Also try new launches for completeness
        launches = rag.search_new_launches(semantic_q, n_results=2000, filters={})

        # Combine all data for comprehensive analysis
        all_data = results + launches

        # No data at all => apologize with WhatsApp
        if not all_data:
            return {
                "source": "insight",
                "message": "آسف يا فندم، مش لاقي بيانات مطابقة في النظام حالياً. تقدر تتواصل فوراً مع خدمة العملاء على واتساب: https://wa.me/201000730208",
                "results": []
            }

        # Now filter the retrieved data based on location/property_type if specified
        if location or property_type:
            filtered_results = []
            for item in all_data:
                metadata = item.get('metadata', {})
                doc = item.get('document', '').lower()
                
                # Check if location matches (in document or metadata)
                location_match = True
                if location:
                    loc_lower = location.lower()
                    location_match = (
                        loc_lower in doc or
                        loc_lower in str(metadata.get('location', '')).lower() or
                        loc_lower in str(metadata.get('city', '')).lower()
                    )
                
                # Check if property type matches
                type_match = True
                if property_type:
                    type_lower = property_type.lower()
                    type_match = (
                        type_lower in doc or
                        type_lower in str(metadata.get('property_type', '')).lower()
                    )
                
                if location_match and type_match:
                    filtered_results.append(item)
            
            # Use filtered results for analysis
            results = filtered_results
        else:
            # Use all retrieved data
            results = all_data

        # If no results after filtering, apologize
        if not results:
            return {
                "source": "insight",
                "message": "آسف يا فندم، مش لاقي بيانات مطابقة للموقع أو النوع المطلوب. تقدر تتواصل فوراً مع خدمة العملاء على واتساب: https://wa.me/201000730208",
                "results": []
            }

        # Classify insight intent via LLM
        intent = classify_insight_intent_llm(query_text)

        # 1) Average price query
        if intent == "average_price":
            prices = []
            for r in results:
                meta = r.get('metadata', {})
                price = meta.get('price_value')
                try:
                    if price is not None:
                        prices.append(float(price))
                except Exception:
                    continue
            if not prices:
                return {
                    "source": "insight",
                    "message": "آسف يا فندم، مش لاقيت أسعار كافية لحساب المتوسط. تقدر تتواصل على واتساب: https://wa.me/201000730208",
                    "results": []
                }
            avg_price = int(sum(prices) / len(prices))
            return {
                "source": "insight",
                "message": f"📊 متوسط الأسعار{' للشقق' if 'شقة' in property_type else ''} في {location or 'المنطقة المحددة'} حوالي {avg_price:,} جنيه.",
                "results": []
            }

        # 2) Compounds list query
        if intent == "compound_list":
            compounds = []
            seen = set()
            # Collect all compounds from filtered results
            for r in results:
                meta = r.get('metadata', {})
                comp = meta.get('compound_name') or meta.get('compound_name_ar')
                if comp and comp not in seen:
                    seen.add(comp)
                    compounds.append(comp)
            
            if not compounds:
                return {
                    "source": "insight",
                    "message": "آسف يا فندم، مش لاقي كومبوندات في المنطقة المطلوبة حالياً. تواصل واتساب: https://wa.me/201000730208",
                    "results": []
                }
            
            # If data is massive (>15 compounds), provide a sample with note
            if len(compounds) > 15:
                sample_compounds = compounds[:15]
                return {
                    "source": "insight",
                    "message": f"🏘️ عينة من الكومبوندات المتاحة في {location or 'المنطقة المطلوبة'} (المجموع: {len(compounds)} كمبوند):",
                    "results": sample_compounds + [f"... وعدد {len(compounds) - 15} كمبوند إضافي. للقائمة الكاملة تواصل معنا."]
                }
            else:
                return {
                    "source": "insight",
                    "message": f"🏘️ الكومبوندات المتاحة في {location or 'المنطقة المطلوبة'} ({len(compounds)} كمبوند):",
                    "results": compounds
                }

        # 3) Cheapest unit query
        if intent == "cheapest_unit":
            cheapest = None
            cheapest_price = None
            for r in results:
                meta = r.get('metadata', {})
                price = meta.get('price_value')
                try:
                    if price is not None:
                        p = float(price)
                        if cheapest_price is None or p < cheapest_price:
                            cheapest_price = p
                            cheapest = r
                except Exception:
                    continue
            if not cheapest:
                return {
                    "source": "insight",
                    "message": "آسف يا فندم، مش لاقيت وحدات بأسعار واضحة. تقدر تتواصل على واتساب: https://wa.me/201000730208",
                    "results": []
                }
            meta = cheapest.get('metadata', {})
            unit_id = meta.get('unit_id', 'غير متوفر')
            doc = cheapest.get('document', '')
            name_ar = ""
            if 'Name (AR):' in doc:
                try:
                    name_ar = doc.split('Name (AR):')[1].split('\n')[0].strip()
                except Exception:
                    name_ar = doc.split('\n')[0][:120]
            else:
                name_ar = doc.split('\n')[0][:120]
            price_val = int(cheapest_price) if cheapest_price is not None else 0
            line = f"ID:{unit_id} | {name_ar} | السعر: {price_val:,} EGP"
            return {
                "source": "insight",
                "message": f"✅ أرخص وحدة مطابقة في {location or 'المنطقة المطلوبة'}:",
                "results": [line]
            }

        # Fallback: return sample of items as insight results without fabrication
        # For massive data sets, provide a representative sample
        sample_size = min(10, len(results))
        lines = []
        for i, r in enumerate(results[:sample_size]):
            meta = r.get('metadata', {})
            doc = r.get('document', '')
            unit_id = meta.get('unit_id', 'غير متوفر')
            name = doc.split('\n')[0][:120] if doc else 'غير متوفر'
            price = meta.get('price_value', 'غير متوفر')
            price_s = f"{int(price):,}" if isinstance(price, (int, float)) else str(price)
            lines.append(f"{i+1}. ID:{unit_id} | {name} | السعر: {price_s}")
        
        # Add note if there are more results
        message = f"🔍 عينة من الوحدات المتاحة (المجموع: {len(results)} وحدة):"
        if len(results) > sample_size:
            lines.append(f"... وعدد {len(results) - sample_size} وحدة إضافية. للمزيد تواصل معنا.")
        
        return {
            "source": "insight",
            "message": message,
            "results": lines
        }

    except Exception as e:
        print(f"🚨 Error in insight_search: {e}")
        return {
            "error": f"❌ حصل خطأ أثناء معالجة الاستفسار: {str(e)}"
        }

def extract_client_preferences_llm(user_message, conversation_history=None, current_preferences=None, conversation_path=None):
    """
    Intelligently extract client preferences using LLM, maintaining state across conversation turns.
    This function analyzes the user's intent and extracts preferences without relying solely on rigid keywords.
    Enhanced for step-by-step conversation flow.
    
    Args:
        user_message (str): Current user message
        conversation_history (list): List of previous messages in the conversation
        current_preferences (dict): Previously extracted preferences to build upon
        conversation_path (str): Either "new_launches" or "available_units" to determine required fields
    
    Returns:
        dict: Updated preferences with new information and missing_required_fields list
    """
    try:
        import google.generativeai as genai
        from variables import GEMINI_API_KEY
        
        # Set up Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Build context from conversation history and current preferences
        context = ""
        if conversation_history:
            context += "المحادثة السابقة:\n"
            for msg in conversation_history[-5:]:  # Last 5 messages for context
                context += f"- {msg}\n"
            context += "\n"
        
        if current_preferences:
            context += "التفضيلات الحالية:\n"
            for key, value in current_preferences.items():
                if value and value != 0:
                    context += f"- {key}: {value}\n"
            context += "\n"
        
        # Create intelligent prompt for preference extraction
        path_instructions = ""
        if conversation_path == "new_launches":
            path_instructions = """
            **مسار الإطلاقات الجديدة - الحقول المطلوبة:**
            - نوع العقار (إلزامي)
            - الموقع (إلزامي)
            - اسم الكمبوند (اختياري)
            """
        elif conversation_path == "available_units":
            path_instructions = """
            **مسار الوحدات المتاحة - الحقول المطلوبة:**
            - نوع العقار (إلزامي)
            - الموقع (إلزامي)
            - الميزانية (إلزامي)
            - اسم الكمبوند المفضل (اختياري)
            - عدد الغرف (اختياري)
            - عدد الحمامات (اختياري)
            - نوع الاستلام (اختياري)
            """
        
        prompt = f"""
        أنت مساعد ذكي لاستخراج تفضيلات العملاء في مجال العقارات.
        
        {context}
        
        {path_instructions}
        
        رسالة العميل الحالية: "{user_message}"
        
        استخرج التفضيلات التالية من الرسالة، مع مراعاة السياق والمحادثة السابقة:
        
        1. **نوع العقار**: شقة، فيلا، دوبلكس، بنتهاوس، تجاري، إلخ
        2. **الموقع**: المدينة أو المنطقة المطلوبة
        3. **الميزانية**: المبلغ المتاح (بالدولار أو الجنيه)
        4. **عدد الغرف**: عدد غرف النوم المطلوبة
        5. **عدد الحمامات**: عدد الحمامات المطلوبة
        6. **المساحة**: المساحة المطلوبة (متر مربع)
        7. **نوع الاستلام**: جاهز للاستلام، تحت الإنشاء، قريباً
        8. **نوع الدفع**: كاش، تقسيط، سنوات التقسيط
        9. **الغرض**: سكن، استثمار، تأجير
        10. **مواصفات إضافية**: أي متطلبات خاصة
        11. **اسم الكمبوند**: اسم الكمبوند المطلوب
        
        اكتب الإجابة بالتنسيق التالي:
        نوع_العقار: [القيمة]
        الموقع: [القيمة]
        الميزانية: [القيمة]
        عدد_الغرف: [القيمة]
        عدد_الحمامات: [القيمة]
        المساحة: [القيمة]
        نوع_الاستلام: [القيمة]
        نوع_الدفع: [القيمة]
        الغرض: [القيمة]
        مواصفات_إضافية: [القيمة]
        اسم_الكمبوند: [القيمة]
        
        إذا لم يتم ذكر قيمة، اكتب "غير محدد"
        """
        
        # Get LLM response
        response = model.generate_content(prompt)
        result = response.text
        
        # Parse the response
        preferences = {
            "property_type": "",
            "location": "",
            "budget": 0,
            "bedrooms": 0,
            "bathrooms": 0,
            "apartment_area": "",
            "delivery_type": "",
            "payment_type": "",
            "purpose": "",
            "additional_specs": "",
            "compound_name": ""
        }
        
        # Parse LLM response
        lines = result.strip().split('\n')
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if 'نوع_العقار' in key or 'property_type' in key:
                    preferences["property_type"] = value if value != "غير محدد" else ""
                elif 'الموقع' in key or 'location' in key:
                    preferences["location"] = value if value != "غير محدد" else ""
                elif 'الميزانية' in key or 'budget' in key:
                    if value != "غير محدد":
                        # Extract numeric value
                        import re
                        budget_match = re.search(r'(\d+(?:,\d{3})*)', value)
                        if budget_match:
                            budget_str = budget_match.group(1).replace(',', '')
                            if 'مليون' in value or 'million' in value:
                                preferences["budget"] = int(budget_str) * 1_000_000
                            elif 'ألف' in value or 'thousand' in value:
                                preferences["budget"] = int(budget_str) * 1_000
                            else:
                                preferences["budget"] = int(budget_str)
                elif 'عدد_الغرف' in key or 'bedrooms' in key:
                    if value != "غير محدد":
                        import re
                        room_match = re.search(r'(\d+)', value)
                        if room_match:
                            preferences["bedrooms"] = int(room_match.group(1))
                elif 'عدد_الحمامات' in key or 'bathrooms' in key:
                    if value != "غير محدد":
                        import re
                        bath_match = re.search(r'(\d+)', value)
                        if bath_match:
                            preferences["bathrooms"] = int(bath_match.group(1))
                elif 'المساحة' in key or 'area' in key:
                    preferences["apartment_area"] = value if value != "غير محدد" else ""
                elif 'نوع_الاستلام' in key or 'delivery' in key:
                    preferences["delivery_type"] = value if value != "غير محدد" else ""
                elif 'نوع_الدفع' in key or 'payment' in key:
                    preferences["payment_type"] = value if value != "غير محدد" else ""
                elif 'الغرض' in key or 'purpose' in key:
                    preferences["purpose"] = value if value != "غير محدد" else ""
                elif 'مواصفات_إضافية' in key or 'additional' in key:
                    preferences["additional_specs"] = value if value != "غير محدد" else ""
                elif 'اسم_الكمبوند' in key or 'compound' in key:
                    preferences["compound_name"] = value if value != "غير محدد" else ""
        
        # Merge with existing preferences (new info overrides old)
        if current_preferences:
            for key in preferences:
                if preferences[key] and preferences[key] != 0 and preferences[key] != "":
                    current_preferences[key] = preferences[key]
            final_preferences = current_preferences
        else:
            final_preferences = preferences
        
        # Determine missing required fields based on conversation path
        missing_required_fields = []
        if conversation_path == "new_launches":
            if not final_preferences.get("property_type"):
                missing_required_fields.append("نوع العقار")
            if not final_preferences.get("location"):
                missing_required_fields.append("الموقع")
        elif conversation_path == "available_units":
            if not final_preferences.get("property_type"):
                missing_required_fields.append("نوع العقار")
            if not final_preferences.get("location"):
                missing_required_fields.append("الموقع")
            if not final_preferences.get("budget") or final_preferences.get("budget") == 0:
                missing_required_fields.append("الميزانية")
        
        # Add missing fields to the result
        final_preferences["missing_required_fields"] = missing_required_fields
        
        return final_preferences
        
    except Exception as e:
        print(f"🚨 Error in LLM preference extraction: {e}")
        # Fallback to keyword-based extraction
        return extract_client_preferences(user_message)


def get_conversation_preferences(conversation_id, user_id):
    """
    Retrieve accumulated preferences from conversation history.
    This function builds a comprehensive preference profile from the entire conversation.
    """
    try:
        # Load conversation history with error handling
        conversations = load_from_cache("conversations_cache.json")
        
        if not conversations:
            logging.warning(f"No conversations found in cache for {conversation_id}")
            return {}
        
        # Find the specific conversation
        conversation = next(
            (c for c in conversations if str(c.get("conversation_id")) == str(conversation_id) and str(c.get("user_id")) == str(user_id)),
            None
        )
        
        if not conversation:
            logging.info(f"No conversation found for {conversation_id} and user {user_id}")
            return {}
        
        # Extract all user messages with error handling
        try:
            description = conversation.get("description", [])
            if not isinstance(description, list):
                logging.warning(f"Invalid description format for conversation {conversation_id}")
                return {}
            
            user_messages = [
                msg["message"] for msg in description
                if isinstance(msg, dict) and msg.get("sender") == "Client" and msg.get("message")
            ]
        except Exception as e:
            logging.error(f"Error extracting user messages from conversation {conversation_id}: {e}")
            return {}
        
        if not user_messages:
            logging.info(f"No user messages found in conversation {conversation_id}")
            return {}
        
        # Build comprehensive preferences from all messages
        accumulated_preferences = {}
        
        for i, message in enumerate(user_messages):
            try:
                # Extract preferences from each message
                message_preferences = extract_client_preferences_llm(
                    message, 
                    user_messages[:i], 
                    accumulated_preferences
                )
                
                # Merge preferences (new info overrides old)
                if isinstance(message_preferences, dict):
                    for key, value in message_preferences.items():
                        if value and value != 0 and value != "":
                            accumulated_preferences[key] = value
            except Exception as e:
                logging.warning(f"Error processing message {i} in conversation {conversation_id}: {e}")
                continue
        
        return accumulated_preferences
        
    except Exception as e:
        logging.error(f"🚨 Error getting conversation preferences for {conversation_id}: {e}")
        return {}

def intelligent_property_search_with_expansion(user_query, search_arguments, chroma_collection, gemini_api_key):
    """
    Intelligent property search that separates semantic and numeric filtering with adaptive expansion policy.
    
    Args:
        user_query (str): The user's natural language query
        search_arguments (dict): Search parameters including numeric filters
        chroma_collection: ChromaDB collection for vector search
        gemini_api_key (str): API key for Gemini embeddings
    
    Returns:
        dict: Search results with metadata about the search strategy used
    """
    try:
        import google.generativeai as genai
        from mmr_search import GeminiEmbedder, mmr
        
        # Set up Gemini
        genai.configure(api_key=gemini_api_key)
        embedder = GeminiEmbedder(gemini_api_key)
        
        # Extract numeric filters from arguments
        numeric_filters = {
            'price_max': search_arguments.get('budget', 0),
            'bedrooms': search_arguments.get('bedrooms'),
            'bathrooms': search_arguments.get('bathrooms'),
            'area_min': search_arguments.get('apartment_area'),
            'installment_years': search_arguments.get('installment_years'),
            'delivery_type': search_arguments.get('delivery_type', '').strip().lower()
        }
        
        # Build semantic query text (exclude numeric values)
        semantic_parts = []
        
        # Property type
        if search_arguments.get('property_type'):
            semantic_parts.append(search_arguments['property_type'])
        
        # Location
        if search_arguments.get('location'):
            semantic_parts.append(search_arguments['location'])
        
        # Compound/developer names (if available)
        if search_arguments.get('compound_name'):
            semantic_parts.append(search_arguments['compound_name'])
        if search_arguments.get('developer_name'):
            semantic_parts.append(search_arguments['developer_name'])
        
        # Delivery readiness
        if numeric_filters['delivery_type']:
            semantic_parts.append(numeric_filters['delivery_type'])
        
        # Purpose (residential, investment, etc.)
        if search_arguments.get('purpose'):
            semantic_parts.append(search_arguments['purpose'])
        
        # Combine semantic parts
        query_text = " ".join(semantic_parts) if semantic_parts else user_query
        
        print(f"🔍 Semantic Query: {query_text}")
        print(f"💰 Numeric Filters: {numeric_filters}")
        
        # Initialize search strategy
        search_strategy = {
            'filters_applied': [],
            'expansion_steps': [],
            'final_results_count': 0
        }
        
        # Step 1: Start with only price filter
        current_filters = {}
        if numeric_filters['price_max'] > 0:
            current_filters['price_max'] = numeric_filters['price_max']
            search_strategy['filters_applied'].append('price_max')
        
        # Perform initial semantic search with minimal filters
        results = _perform_semantic_search(
            query_text, 
            chroma_collection, 
            embedder, 
            current_filters,
            fetch_k=1000
        )
        
        print(f"📊 Initial results: {len(results)}")
        
        # Step 2: Adaptive filtering based on result count
        target_results = 20  # Target number of results
        min_results = 5      # Minimum acceptable results
        
        if len(results) > target_results:
            # Too many results - add filters progressively
            results = _apply_progressive_filtering(
                results, numeric_filters, search_strategy, target_results
            )
        elif len(results) < min_results:
            # Too few results - apply expansion policy
            results = _apply_expansion_policy(
                query_text, chroma_collection, embedder, 
                numeric_filters, search_strategy, min_results
            )
        
        search_strategy['final_results_count'] = len(results)
        
        return {
            'results': results,
            'search_strategy': search_strategy,
            'semantic_query': query_text,
            'numeric_filters': numeric_filters
        }
        
    except Exception as e:
        print(f"🚨 Error in intelligent property search: {e}")
        # Fallback to basic search
        return {
            'results': [],
            'error': str(e),
            'fallback': True
        }


def _perform_semantic_search(query_text, chroma_collection, embedder, filters, fetch_k=100):
    """
    Perform semantic search with given filters
    """
    try:
        from mmr_search import mmr  # Import mmr here to ensure it's available
        
        # Build where clause for ChromaDB
        where_clause = {}
        
        if 'price_max' in filters:
            tolerance = 0.15  # 15% tolerance for price
            min_price = filters['price_max'] * (1 - tolerance)
            where_clause['price_value'] = {'$lte': filters['price_max']}
        
        # Query ChromaDB
        chroma_results = chroma_collection.query(
            query_texts=[query_text],
            n_results=fetch_k,
            where=where_clause if where_clause else None
        )
        
        docs = chroma_results['documents'][0]
        metadatas = chroma_results['metadatas'][0]
        embeddings = chroma_results['embeddings'][0] if 'embeddings' in chroma_results else None
        
        if embeddings is None:
            embeddings = embedder.embed_many(docs)
        
        # Apply MMR for diversity
        query_embedding = embedder.embed(query_text)
        mmr_indices = mmr(query_embedding, embeddings, k=min(len(docs), 50), lambda_param=0.7)
        
        results = []
        for i in mmr_indices:
            results.append({
                'document': docs[i],
                'metadata': metadatas[i],
                'score': _cosine_similarity(query_embedding, embeddings[i]),
                'source': 'semantic_search'
            })
        
        return results
        
    except Exception as e:
        print(f"🚨 Error in semantic search: {e}")
        return []


def _apply_progressive_filtering(results, numeric_filters, search_strategy, target_results):
    """
    Apply filters progressively to reduce results to target count
    """
    print(f"🔍 Applying progressive filtering to reduce {len(results)} results to ~{target_results}")
    
    # Filter 1: Bedrooms (if specified)
    if numeric_filters['bedrooms'] and len(results) > target_results:
        filtered_results = []
        for result in results:
            result_bedrooms = result['metadata'].get('bedrooms', 0)
            if result_bedrooms == numeric_filters['bedrooms']:
                filtered_results.append(result)
        
        if len(filtered_results) >= target_results * 0.5:  # Keep if we still have enough results
            results = filtered_results
            search_strategy['filters_applied'].append('bedrooms')
            print(f"✅ Applied bedrooms filter: {len(results)} results")
    
    # Filter 2: Area (if specified)
    if numeric_filters['area_min'] and len(results) > target_results:
        area_tolerance = 30  # ±30 sqm tolerance
        filtered_results = []
        for result in results:
            result_area = result['metadata'].get('area', 0)
            if abs(result_area - numeric_filters['area_min']) <= area_tolerance:
                filtered_results.append(result)
        
        if len(filtered_results) >= target_results * 0.5:
            results = filtered_results
            search_strategy['filters_applied'].append('area')
            print(f"✅ Applied area filter: {len(results)} results")
    
    # Filter 3: Bathrooms (if specified)
    if numeric_filters['bathrooms'] and len(results) > target_results:
        filtered_results = []
        for result in results:
            result_bathrooms = result['metadata'].get('bathrooms', 0)
            if result_bathrooms == numeric_filters['bathrooms']:
                filtered_results.append(result)
        
        if len(filtered_results) >= target_results * 0.5:
            results = filtered_results
            search_strategy['filters_applied'].append('bathrooms')
            print(f"✅ Applied bathrooms filter: {len(results)} results")
    
    return results


def _apply_expansion_policy(query_text, chroma_collection, embedder, numeric_filters, search_strategy, min_results):
    """
    Apply expansion policy when too few results are found
    """
    print(f"🔍 Applying expansion policy to increase results above {min_results}")
    
    # Expansion 1: Relax area constraints
    if numeric_filters['area_min']:
        area_tolerance = 50  # Increase to ±50 sqm
        search_strategy['expansion_steps'].append(f'area_tolerance_increased_to_{area_tolerance}')
        print(f"📏 Increased area tolerance to ±{area_tolerance} sqm")
    
    # Expansion 2: Relax bedroom constraints
    if numeric_filters['bedrooms']:
        bedroom_range = f"{max(1, numeric_filters['bedrooms']-1)}-{numeric_filters['bedrooms']+1}"
        search_strategy['expansion_steps'].append(f'bedroom_range_expanded_to_{bedroom_range}')
        print(f"🛏️ Expanded bedroom range to {bedroom_range}")
    
    # Expansion 3: Increase price tolerance
    if numeric_filters['price_max']:
        price_tolerance = 0.25  # Increase to 25%
        search_strategy['expansion_steps'].append(f'price_tolerance_increased_to_{price_tolerance*100}%')
        print(f"💰 Increased price tolerance to {price_tolerance*100}%")
    
    # Re-run search with relaxed constraints
    relaxed_filters = numeric_filters.copy()
    if numeric_filters['price_max']:
        relaxed_filters['price_max'] = int(numeric_filters['price_max'] * 1.25)
    
    results = _perform_semantic_search(
        query_text, 
        chroma_collection, 
        embedder, 
        relaxed_filters,
        fetch_k=200  # Increase fetch size
    )
    
    print(f"📊 Results after expansion: {len(results)}")
    return results


def _cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    import numpy as np
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


def get_more_units(arguments):
    """
    Get more units with increased price tolerance for progressive search.
    This is only for existing units, not new launches.
    """
    try:
        # Get the session ID to retrieve previous search parameters
        session_id = arguments.get("session_id", "")
        
        if not session_id:
            return {
                "source": "error",
                "message": "❌ خطأ: لم يتم العثور على بيانات الجلسة",
                "results": []
            }
        
        # Try to get previous search parameters from session
        import config
        session_key = f"{session_id}_search_history"
        session_data = config.client_sessions.get(session_key)
        
        # Debug: Print available session keys
        logging.info(f"🔍 Looking for session key: {session_key}")
        logging.info(f"🔍 Available session keys: {list(config.client_sessions.keys())}")
        
        if not session_data:
            # Try alternative session key formats
            # 1. Look for any key containing parts of the session_id
            session_parts = session_id.replace("session_", "").split("_")
            alternative_keys = []
            
            for key in config.client_sessions.keys():
                if "search_history" in key:
                    # Check if any part of the session matches
                    key_parts = key.replace("session_", "").split("_")
                    if any(part in key_parts for part in session_parts if part and len(part) > 4):
                        alternative_keys.append(key)
            
            # 2. If no partial match, try the most recent search_history key
            if not alternative_keys:
                search_history_keys = [key for key in config.client_sessions.keys() if "search_history" in key]
                if search_history_keys:
                    # Sort by key (assuming newer sessions have later timestamps)
                    alternative_keys = sorted(search_history_keys, reverse=True)[:1]
            
            if alternative_keys:
                session_key = alternative_keys[0]
                session_data = config.client_sessions.get(session_key)
                logging.info(f"🔍 Found alternative session key: {session_key}")
                logging.info(f"🔍 Original session_id: {session_id}")
                logging.info(f"🔍 All search_history keys: {[k for k in config.client_sessions.keys() if 'search_history' in k]}")
            
        if not session_data:
            return {
                "source": "error", 
                "message": f"❌ لم أتمكن من العثور على معايير البحث السابقة. Session ID: {session_id}. يرجى إعادة البحث من البداية.",
                "results": []
            }
        
        # Get previous search parameters and round
        previous_params = session_data.get("last_search_params", {})
        current_round = session_data.get("search_round", 1) + 1
        shown_ids = session_data.get("shown_ids", [])
        
        if current_round > 3:
            return {
                "source": "max_rounds_reached",
                "message": "تم عرض جميع الخيارات المتاحة بمختلف مستويات مرونة السعر. تحب تبدأ بحث جديد بمعايير مختلفة؟",
                "results": []
            }
        
        # Set price tolerance based on round
        # Round 1: 10%, Round 2: 20%, Round 3: 25%
        tolerance_map = {2: 20, 3: 25}
        price_tolerance = tolerance_map.get(current_round, 25)
        
        # Update search parameters for progressive search
        new_params = previous_params.copy()
        new_params.update({
            "price_tolerance": price_tolerance,
            "excluded_ids": shown_ids,
            "search_round": current_round
        })
        
        # Call property_search with updated parameters
        result = property_search(new_params)
        
        # Update session with new results
        if result.get("results"):
            # Handle both string and dict formats for results
            new_ids = []
            for item in result.get("results", []):
                if isinstance(item, dict):
                    new_ids.append(str(item.get('id', '')))
                elif isinstance(item, str):
                    import re
                    id_match = re.search(r'ID:(\d+)', item)
                    if id_match:
                        new_ids.append(id_match.group(1))
            
            new_shown_ids = shown_ids + new_ids
            session_data.update({
                "search_round": current_round,
                "shown_ids": new_shown_ids,
                "last_search_params": previous_params  # Keep original params
            })
            config.client_sessions[f"{session_id}_search_history"] = session_data
        
        return result
        
    except Exception as e:
        logging.error(f"Error in get_more_units: {e}")
        return {
            "source": "error",
            "message": f"❌ خطأ في البحث عن وحدات إضافية: {str(e)}",
            "results": []
        }

def get_unit_details(arguments):
    """
    Get detailed information about a specific unit by ID.
    Works for both regular properties and new launches.
    """
    try:
        unit_id = arguments.get("unit_id")
        if not unit_id:
            return {
                "error": "❌ رقم الوحدة مطلوب لعرض التفاصيل."
            }
        
        # First, try to find in new launches (since this is more likely for new launch IDs)
        try:
            from chroma_rag_setup import get_rag_instance
            rag = get_rag_instance()
            nl_results = rag.new_launches_collection.query(
                query_texts=["launch details"],
                n_results=1,
                where={"launch_id": str(unit_id)},
                include=["metadatas", "documents"]
            )
            if nl_results and nl_results.get('documents') and nl_results['documents'][0]:
                doc = nl_results['documents'][0][0]
                meta = nl_results['metadatas'][0][0] if nl_results.get('metadatas') and nl_results['metadatas'][0] else {}
                
                # Extract required fields from embedded document
                def extract_between(label: str, text: str) -> str:
                    if label in text:
                        try:
                            return text.split(label)[1].split('\n')[0].strip()
                        except Exception:
                            return "غير متوفر"
                    return "غير متوفر"
                
                # Extract all relevant fields for new launches
                property_type_name = extract_between('Property Type:', doc)
                desc_ar = extract_between('Description (AR):', doc)
                city_name = extract_between('City:', doc)
                compound_name = extract_between('Compound (AR):', doc)
                name_ar = extract_between('Name (AR):', doc)
                price = meta.get('price_value', 'غير متوفر')
                bedrooms = meta.get('bedrooms', 'غير متوفر')
                bathrooms = meta.get('bathrooms', 'غير متوفر')
                area = meta.get('apartment_area', 'غير متوفر')
                delivery_in = meta.get('delivery_in', 'غير متوفر')
                installment_years = meta.get('installment_years', 'غير متوفر')
                new_image = meta.get('new_image', 'غير متوفر')
                
                # Format price if available
                try:
                    if price and price != 'غير متوفر':
                        price_val = f"{int(float(price)):,}"
                    else:
                        price_val = "غير متوفر"
                except:
                    price_val = "غير متوفر"
                
                formatted_details = f"""
تفاصيل الوحدة:
📍 اسم الوحدة: {name_ar}
📝 الوصف: {desc_ar}

📌 الكومباوند: {compound_name}
🏷️ النوع: {property_type_name}
🌆 الموقع: {city_name}

🖼️ **صورة العقار:** {new_image}


ℹ️ ملاحظات هامة

❌ لا توجد زيارة ميدانية حاليًا لهذا المشروع.

ℹ️ تفاصيل مثل الأسعار، المساحات، وأنظمة الدفع غير متوفرة الآن.

🎯 الخطوة التالية

تحب أحجزلك Zoom Meeting مع سيلز المطور عشان تعرف تفاصيل أكتر عن العروض المتاحة؟
"""
                return {
                    "unit_type": "new_launch",
                    "message": formatted_details,
                    "details": {
                        "id": unit_id,
                        "property_type_name": property_type_name,
                        "desc_ar": desc_ar,
                        "city_name": city_name,
                        "new_image": new_image,
                        "name_ar": name_ar,
                        "compound_name": compound_name,
                        "price": price_val,
                        "bedrooms": bedrooms,
                        "bathrooms": bathrooms,
                        "area": area,
                        "delivery_in": delivery_in,
                        "installment_years": installment_years
                    }
                }
        except Exception as e:
            print(f"⚠️ Error loading new launch from Chroma: {e}")
        
        # Fallback: try to find in regular units cache
        try:
            units = load_from_cache("units.json")
            for unit in units:
                if str(unit.get("id")) == str(unit_id):
                    # Build values with fallbacks
                    def clean(val):
                        if val is None or str(val).strip() == "" or str(val).strip().lower() == "none":
                            return "غير متوفر"
                        return str(val)

                    name_ar_val = clean(unit.get("name_ar"))
                    desc_ar_val = clean(unit.get("desc_ar"))
                    compound_ar_val = clean(unit.get("compound_name_ar") or unit.get("compound_name"))
                    area_val = clean(unit.get("apartment_area") or unit.get("size"))

                    price_raw = unit.get("price")
                    try:
                        if price_raw is None:
                            price_val = "غير متوفر"
                        else:
                            # normalize to int and format with commas
                            pr = str(price_raw).replace(",", "").strip()
                            pr_num = int(float(pr))
                            price_val = f"{pr_num:,}"
                    except Exception:
                        price_val = clean(price_raw)

                    bedrooms_val = clean(unit.get("Bedrooms"))
                    bathrooms_val = clean(unit.get("Bathrooms"))
                    delivery_in_val = clean(unit.get("delivery_in"))
                    installment_years_val = clean(unit.get("installment_years"))
                    address_val = clean(unit.get("address"))
                    image_val = clean(unit.get("new_image") or unit.get("image_url"))

                    # Format the response exactly as requested
                    formatted_details = f"""
📍 **اسم الوحدة:** {name_ar_val}  
📝 **الوصف:** {desc_ar_val}  

📌 **الكومباوند:** {compound_ar_val}  
📐 **المساحة:** {area_val} م²  
💰 **السعر:** {price_val} جنيه  

🛏 **عدد الغرف:** {bedrooms_val}  
🚽 **عدد الحمامات:** {bathrooms_val}  

🚚 **الاستلام خلال:** {delivery_in_val} سنة  
💳 **تقسيط حتى:** {installment_years_val} سنين  

🖼️ **صورة العقار:** {image_val}

تحب احجزلك zoom meeting مع السيلز المطور عشان تعرف تفاصيل اكتر او عروض ولا احجزلك زيارة on site للمعاينة
"""
                    return {
                        "unit_type": "regular_property",
                        "message": formatted_details,
                        "details": {
                            "id": unit.get("id"),
                            "name_ar": name_ar_val,
                            "name_en": unit.get("name_en", "غير متوفر"),
                            "compound_name_ar": compound_ar_val,
                            "price": price_val,
                            "bedrooms": bedrooms_val,
                            "bathrooms": bathrooms_val,
                            "delivery_in": delivery_in_val,
                            "installment_years": installment_years_val,
                            "location": unit.get("location", "غير متوفر"),
                            "new_image": image_val,
                            "desc_ar": desc_ar_val,
                            "address": address_val,
                            "apartment_area": area_val
                        }
                    }
        except Exception as e:
            print(f"⚠️ Error loading from units cache: {e}")
        
        # Fallback: try to query Chroma by unit_id
        try:
            from chroma_rag_setup import RealEstateRAG
            rag = RealEstateRAG()
            # Query with where clause to match unit_id
            results = rag.units_collection.query(
                query_texts=["unit details"],
                n_results=1,
                where={"unit_id": str(unit_id)},
                include=["metadatas", "documents", "distances"]
            )
            if results and results.get('documents') and results['documents'][0]:
                doc = results['documents'][0][0]
                meta = results['metadatas'][0][0] if results.get('metadatas') and results['metadatas'][0] else {}
                # Try to parse fields from doc text
                name_ar = "غير متوفر"
                if 'Name (AR):' in doc:
                    name_ar = doc.split('Name (AR):')[1].split('\n')[0].strip()
                compound_name = "غير متوفر"
                if 'Compound (AR):' in doc:
                    compound_name = doc.split('Compound (AR):')[1].split('\n')[0].strip()
                price = meta.get('price_value', 'غير متوفر')
                bedrooms = meta.get('bedrooms', 'غير متوفر')
                bathrooms = meta.get('bathrooms', 'غير متوفر')
                area = meta.get('apartment_area', 'غير متوفر')
                image_url = meta.get('new_image', 'غير متوفر')
                
                # Extract description and address from document
                desc_ar = "غير متوفر"
                address = "غير متوفر"
                compound_ar = "غير متوفر"
                if 'Description (AR):' in doc:
                    try:
                        desc_ar = doc.split('Description (AR):')[1].split('\n')[0].strip()
                    except:
                        desc_ar = "غير متوفر"
                if 'Compound (AR):' in doc:
                    try:
                        compound_ar = doc.split('Compound (AR):')[1].split('\n')[0].strip()
                    except:
                        compound_ar = "غير متوفر"
                if 'Address:' in doc:
                    try:
                        address = doc.split('Address:')[1].split('\n')[0].strip()
                    except:
                        address = "غير متوفر"

                # Build values with fallbacks
                def clean2(val):
                    if val is None or str(val).strip() == "" or str(val).strip().lower() == "none":
                        return "غير متوفر"
                    return str(val)

                price_meta = meta.get('price_value')
                try:
                    if price_meta is None:
                        price_val = "غير متوفر"
                    else:
                        price_val = f"{int(float(price_meta)):,}"
                except Exception:
                    price_val = clean2(price_meta)

                name_ar_val = clean2(name_ar)
                area_val = clean2(meta.get('apartment_area'))
                bedrooms_val = clean2(meta.get('bedrooms'))
                bathrooms_val = clean2(meta.get('bathrooms'))
                delivery_in_val = clean2(meta.get('delivery_in'))
                installment_years_val = clean2(meta.get('installment_years'))
                address_val = clean2(address)
                image_val = clean2(meta.get('new_image'))
                compound_ar_val = clean2(compound_ar)
                desc_ar_val = clean2(desc_ar)

                # Format the response exactly as requested
                formatted_details = f"""
📍 **اسم الوحدة:** {name_ar_val}  
📝 **الوصف:** {desc_ar_val}  

📌 **الكومباوند:** {compound_ar_val}  
📐 **المساحة:** {area_val} م²  
💰 **السعر:** {price_val} جنيه  

🛏 **عدد الغرف:** {bedrooms_val}  
🚽 **عدد الحمامات:** {bathrooms_val}  

🚚 **الاستلام خلال:** {delivery_in_val} سنة  
💳 **تقسيط حتى:** {installment_years_val} سنة  

🖼️ **صورة العقار:** {image_val}

تحب احجزلك zoom meeting مع السيلز المطور عشان تعرف تفاصيل اكتر او عروض ولا احجزلك زيارة on site للمعاينة
"""
                return {
                    "unit_type": "regular_property",
                    "message": formatted_details,
                    "details": {
                        "id": unit_id,
                        "name_ar": name_ar_val,
                        "name_en": doc.split('\n')[0][:120] if doc else "غير متوفر",
                        "compound_name_ar": compound_ar_val,
                        "price": price_val,
                        "bedrooms": bedrooms_val,
                        "bathrooms": bathrooms_val,
                        "delivery_in": delivery_in_val,
                        "installment_years": installment_years_val,
                        "location": meta.get('location', 'غير متوفر'),
                        "new_image": image_val,
                        "desc_ar": desc_ar_val,
                        "address": address_val,
                        "apartment_area": area_val
                    }
                }
        except Exception as e:
            print(f"⚠️ Error fetching from Chroma for unit {unit_id}: {e}")
        
        # Try to find in new launches
        try:
            # Prefer ChromaDB lookup for new launches to extract from embeddings/metadata
            from chroma_rag_setup import RealEstateRAG
            rag = RealEstateRAG()
            nl_results = rag.new_launches_collection.query(
                query_texts=["launch details"],
                n_results=1,
                where={"launch_id": str(unit_id)},
                include=["metadatas", "documents"]
            )
            if nl_results and nl_results.get('documents') and nl_results['documents'][0]:
                doc = nl_results['documents'][0][0]
                meta = nl_results['metadatas'][0][0] if nl_results.get('metadatas') and nl_results['metadatas'][0] else {}
                
                # Extract required fields from embedded document
                def extract_between(label: str, text: str) -> str:
                    if label in text:
                        try:
                            return text.split(label)[1].split('\n')[0].strip()
                        except Exception:
                            return "غير متوفر"
                    return "غير متوفر"
                
                # Extract all relevant fields for new launches
                property_type_name = extract_between('Property Type:', doc)
                desc_ar = extract_between('Description (AR):', doc)
                city_name = extract_between('City:', doc)
                compound_name = extract_between('Compound (AR):', doc)
                name_ar = extract_between('Name (AR):', doc)
                price = meta.get('price_value', 'غير متوفر')
                bedrooms = meta.get('bedrooms', 'غير متوفر')
                bathrooms = meta.get('bathrooms', 'غير متوفر')
                area = meta.get('apartment_area', 'غير متوفر')
                delivery_in = meta.get('delivery_in', 'غير متوفر')
                installment_years = meta.get('installment_years', 'غير متوفر')
                new_image = meta.get('new_image', 'غير متوفر')
                
                # Format price if available
                try:
                    if price and price != 'غير متوفر':
                        price_val = f"{int(float(price)):,}"
                    else:
                        price_val = "غير متوفر"
                except:
                    price_val = "غير متوفر"
                
                formatted_details = f"""
📍 **اسم الوحدة:** {name_ar}
📝 **الوصف:** {desc_ar}

📌 **الكومباوند:** {compound_name}
📐 **المساحة:** {area} م²
💰 **السعر:** {price_val} جنيه

🛏 **عدد الغرف:** {bedrooms}
🚽 **عدد الحمامات:** {bathrooms}

🚚 **الاستلام خلال:** {delivery_in} سنة
💳 **تقسيط حتى:** {installment_years} سنة

🖼️ **صورة العقار:** {new_image}

تحب احجزلك zoom meeting مع السيلز المطور عشان تعرف تفاصيل اكتر او عروض ولا احجزلك زيارة on site للمعاينة
"""
                return {
                    "unit_type": "new_launch",
                    "message": formatted_details,
                    "details": {
                        "id": unit_id,
                        "property_type_name": property_type_name,
                        "desc_ar": desc_ar,
                        "city_name": city_name,
                        "new_image": new_image,
                        "name_ar": name_ar,
                        "compound_name": compound_name,
                        "price": price_val,
                        "bedrooms": bedrooms,
                        "bathrooms": bathrooms,
                        "area": area,
                        "delivery_in": delivery_in,
                        "installment_years": installment_years
                    }
                }
        except Exception as e:
            print(f"⚠️ Error loading new launch from Chroma: {e}")
        
        # Fallback to cache new_launches if Chroma lookup fails
        try:
            new_launches = load_from_cache("new_launches.json")
            for launch in new_launches:
                if str(launch.get("id")) == str(unit_id):
                    # Minimal fields per request; best-effort from cache
                    property_type_name = launch.get("property_type_name", "غير متوفر")
                    desc_ar = launch.get("desc_ar", "غير متوفر")
                    city_name = launch.get("city", launch.get("city_name", "غير متوفر"))
                    new_image = launch.get("new_image", launch.get("image_url", "غير متوفر"))
                    formatted_details = f"""
نوع العقار: {property_type_name}
الوصف: {desc_ar}
المدينة: {city_name}
الصورة: {new_image}
"""
                    return {
                        "unit_type": "new_launch",
                        "message": formatted_details,
                        "details": {
                            "id": launch.get("id"),
                            "property_type_name": property_type_name,
                            "desc_ar": desc_ar,
                            "city_name": city_name,
                            "new_image": new_image
                        }
                    }
        except Exception as e:
            print(f"⚠️ Error loading from new_launches cache: {e}")
        
        return {
            "error": f"❌ لم يتم العثور على وحدة برقم {unit_id}."
        }
        
    except Exception as e:
        print(f"🚨 Error in get_unit_details: {e}")
        return {
            "error": f"❌ خطأ في عرض تفاصيل الوحدة: {str(e)}"
        }



