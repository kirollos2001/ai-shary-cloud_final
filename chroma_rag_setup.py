import chromadb
import json
import os
import logging
import shutil
import time
import sqlite3
import google.generativeai as genai
import numpy as np
import variables
from typing import List, Dict, Any, Optional
from chromadb.config import Settings
# Configure Gemini API
os.environ["GEMINI_API_KEY"] = variables.GEMINI_API_KEY
genai.configure(api_key=variables.GEMINI_API_KEY)
class GeminiEmbeddingFunction:
    """Custom embedding function using Gemini embeddings"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
    
    def name(self) -> str:
        return "gemini-embedding-001"
    def __call__(self, input: list) -> list:
        try:
            result = genai.embed_content(
                model="models/gemini-embedding-001",
                content=input,
                task_type="SEMANTIC_SIMILARITY"
            )
            return [e for e in result['embedding']]
        except Exception as e:
            logger.error(f"❌ Error generating embedding: {e}")
            return [[0.0] * 3072 for _ in input]  # Updated to 3072 dimensions
    
    def embed(self, text: str) -> List[float]:
        """Single text embedding - required by ChromaDB"""
        try:
            result = genai.embed_content(
                model="models/gemini-embedding-001",
                content=text,
                task_type="SEMANTIC_SIMILARITY"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"❌ Error generating single embedding: {e}")
            return [0.0] * 3072  # Return zero vector as fallback
    
    def embed_many(self, texts: List[str]) -> List[List[float]]:
        """Batch embed multiple texts - process in chunks to avoid API limits"""
        if not texts:
            return []
        
        batch_size = 100  # Process in batches to avoid API limits
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = []
            for text in batch:
                try:
                    result = genai.embed_content(
                        model="models/gemini-embedding-001",
                        content=text,
                        task_type="SEMANTIC_SIMILARITY"
                    )
                    batch_embeddings.append(result['embedding'])
                except Exception as e:
                    logger.error(f"Error embedding text: {e}")
                    # Use zero vector as fallback
                    batch_embeddings.append([0.0] * 3072)  # Updated to 3072 dimensions
            all_embeddings.extend(batch_embeddings)
            
        return all_embeddings
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Singleton cache for RAG instance
_rag_instance = None
class RealEstateRAG:
    def __init__(self, persist_directory: str = variables.CHROMA_PERSIST_DIR):
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)
        chroma_settings = Settings(anonymized_telemetry=False, allow_reset=True)
        self._chroma_settings = chroma_settings
        self.client = chromadb.PersistentClient(path=self.persist_directory, settings=chroma_settings)
        self.embedder = GeminiEmbeddingFunction(variables.GEMINI_API_KEY)
        # Cache for query embeddings to avoid regeneration
        self._query_embedding_cache = {}
        
        # Initialize collections with schema-mismatch recovery
        try:
            self._create_or_get_collections()
        except Exception as e:
            err_msg = str(e)
            logger.error(f"Error creating collections: {err_msg}")
            err_msg_lower = err_msg.lower()
            if (
                isinstance(e, sqlite3.OperationalError)
                or "no such column: collections.topic" in err_msg_lower
                or "embeddings_queue" in err_msg_lower
            ):
                if "embeddings_queue" in err_msg_lower:
                    logger.warning(
                        "Detected ChromaDB embeddings queue conflict. Attempting automatic reset..."
                    )
                else:
                    logger.warning("Detected ChromaDB schema mismatch. Attempting automatic reset...")
                self._reset_chroma_storage(chroma_settings)
                self._create_or_get_collections()
            else:
                raise
        logger.info("ChromaDB initialized with collections")
        
    def _parse_float(self, value, default=None):
        try:
            if value is None:
                return default
            if isinstance(value, (int, float)):
                return float(value)
            s = str(value).strip()
            # Remove commas and non-digit characters except dot
            import re
            m = re.search(r"\d+(?:\.\d+)?", s.replace(',', ''))
            if m:
                return float(m.group(0))
            return default
        except Exception:
            return default

    def _parse_int(self, value, default=None):
        try:
            if value is None:
                return default
            if isinstance(value, (int, float)):
                return int(value)
            s = str(value).strip()
            import re
            m = re.search(r"\d+", s)
            if m:
                return int(m.group(0))
            return default
        except Exception:
            return default

    def _compute_post_filter_indices(self, results: Dict, filters: Dict) -> List[int]:
        """Return indices of results that pass post filters for area and installment years.

        - area tolerance: ±10%
        - installment years tolerance: ±2
        Only applies if corresponding filter values are provided by caller.
        """
        indices = []
        try:
            docs = results.get('documents') or []
            if not docs or not docs[0]:
                return indices

            metas_list = (results.get('metadatas') or [[]])[0] if results.get('metadatas') else []

            target_area = None
            target_installment = None
            if filters:
                # Accept either 'area' or 'apartment_area' from filters
                target_area = filters.get('apartment_area') if 'apartment_area' in filters else filters.get('area')
                target_installment = filters.get('installment_years')

            # If neither filter provided, return empty to indicate no filtering
            if target_area in (None, "", 0) and target_installment in (None, "", 0):
                return []

            area_low, area_high = None, None
            if target_area not in (None, "", 0):
                area_val = self._parse_float(target_area)
                if area_val is not None and area_val > 0:
                    tol = 0.10
                    area_low, area_high = area_val * (1 - tol), area_val * (1 + tol)

            inst_low, inst_high = None, None
            if target_installment not in (None, "", 0):
                inst_val = self._parse_int(target_installment)
                if inst_val is not None:
                    inst_low, inst_high = inst_val - 2, inst_val + 2

            # Evaluate each item in first result set
            for i in range(len(docs[0])):
                meta = metas_list[i] if metas_list and i < len(metas_list) else {}
                ok = True

                if area_low is not None and area_high is not None:
                    area_meta = self._parse_float(meta.get('apartment_area'))
                    # If metadata missing or unparsable, fail this condition
                    if area_meta is None or not (area_low <= area_meta <= area_high):
                        ok = False

                if ok and inst_low is not None and inst_high is not None:
                    inst_meta = self._parse_int(meta.get('installment_years'))
                    if inst_meta is None or not (inst_low <= inst_meta <= inst_high):
                        ok = False

                if ok:
                    indices.append(i)

            return indices
        except Exception as e:
            logger.warning(f"Post-filter index computation failed: {e}")
            return []

    def _get_cached_query_embedding(self, query: str) -> List[float]:
        """Get cached query embedding or generate and cache it"""
        if query in self._query_embedding_cache:
            return self._query_embedding_cache[query]
        
        # Generate new embedding and cache it
        embedding = self.embedder.embed(query)
        self._query_embedding_cache[query] = embedding
        return embedding
    def reset_collections(self):
        """Reset collections to start fresh with correct dimensions"""
        try:
            # Delete existing collections
            try:
                self.client.delete_collection("real_estate_units")
                logger.info("Deleted existing real_estate_units collection")
            except Exception:
                logger.info("real_estate_units collection doesn't exist, creating new one")
    
            try:
                self.client.delete_collection("new_launches")
                logger.info("Deleted existing new_launches collection")
            except Exception:
                logger.info("new_launches collection doesn't exist, creating new one")
    
            # Create fresh collections with correct embedding dimensions
            self.units_collection = self.client.create_collection(
                name="real_estate_units",
                metadata={"description": "Real estate units data for RAG with 3072-dim embeddings"},
                embedding_function=self.embedder,
                            # Add HNSW configuration to prevent the error
            hnsw_config={
                "M": 64,
                "ef_construction": 200
            }
            )
    
            self.new_launches_collection = self.client.create_collection(
                name="new_launches",
                metadata={"description": "New property launches data for RAG with 3072-dim embeddings"},
                embedding_function=self.embedder,
                            # Add HNSW configuration to prevent the error
            hnsw_config={
                "M": 64,
                "ef_construction": 200
            }
            )
    
            logger.info("Fresh collections created with 3072-dimensional embeddings")
    
        except Exception as e:
            logger.error(f"Error resetting collections: {e}")
    
    def _create_or_get_collections(self):
        """Create or get required collections."""
        self.units_collection = self.client.get_or_create_collection(
            name="real_estate_units",
            metadata={"description": "Real estate units data for RAG"},
            embedding_function=self.embedder
        )
    
        self.new_launches_collection = self.client.get_or_create_collection(
            name="new_launches",
            metadata={"description": "New property launches data for RAG"},
            embedding_function=self.embedder
        )
    def _get_collection_count(self, collection, collection_name: str) -> Optional[int]:
        """Safely obtain a collection count while logging any issues."""
        try:
            return collection.count()
        except Exception as e:
            logger.warning(
                f"⚠️ Could not determine {collection_name} collection count: {e}"
            )
            return None

    def _is_embeddings_queue_conflict(self, error: Exception) -> bool:
        """Check if the exception is due to an embeddings queue schema conflict."""
        message = str(error).lower()
        return "embeddings_queue" in message and "already exists" in message

    def _should_reset_for_client_error(self, error: Exception) -> bool:
        """Determine if a Chroma client error should trigger a persistence reset."""
        message = str(error).lower()
        if "embeddings_queue" in message:
            return True
        if isinstance(error, sqlite3.OperationalError):
            conflict_markers = ["already exists", "duplicate column", "no such column"]
            return any(marker in message for marker in conflict_markers)
        return False

    def _create_client_with_recovery(self, chroma_settings: Settings):
        """Create a persistent Chroma client with automatic recovery from schema conflicts."""
        try:
            return chromadb.PersistentClient(path=self.persist_directory, settings=chroma_settings)
        except Exception as error:
            if self._should_reset_for_client_error(error):
                logger.warning(
                    "Detected ChromaDB persistence conflict during client creation. Resetting storage directory and retrying..."
                )
                self._reset_chroma_storage(chroma_settings, skip_client_reset=True)
                return self.client
            raise

    def _reset_chroma_storage(self, chroma_settings: Settings, skip_client_reset: bool = False):
        """Reset persistent storage to recover from schema mismatches.
    
        1) Try client.reset() if allowed.
        2) Backup existing directory and recreate a fresh one.
        """
        # Try logical reset first
        try:
            if not skip_client_reset and getattr(self, "client", None) is not None and hasattr(self.client, "reset"):
                self.client.reset()
        except Exception:
            pass
    
        # Backup existing directory and recreate
        try:
            if os.path.isdir(self.persist_directory):
                backup_dir = f"{self.persist_directory}_backup_{int(time.time())}"
                shutil.move(self.persist_directory, backup_dir)
        except Exception:
            pass
    
        os.makedirs(self.persist_directory, exist_ok=True)
        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory, settings=chroma_settings)
        except Exception as error:
            logger.error(f"❌ Failed to recreate ChromaDB client after reset: {error}")
            raise
    
    def load_cache_data(self) -> tuple:
        try:
            units_path = os.path.join(variables.CACHE_DIR, "units.json")
            new_launches_path = os.path.join(variables.CACHE_DIR, "new_launches.json")
            units_data = []
            new_launches_data = []
            if os.path.exists(units_path):
                with open(units_path, 'r', encoding='utf-8') as f:
                    units_data = json.load(f)
                logger.info(f"✅ Loaded {len(units_data)} units from cache")
            else:
                logger.warning("⚠️ units.json not found in cache directory")
            if os.path.exists(new_launches_path):
                with open(new_launches_path, 'r', encoding='utf-8') as f:
                    new_launches_data = json.load(f)
                logger.info(f"✅ Loaded {len(new_launches_data)} new launches from cache")
            else:
                logger.warning("⚠️ new_launches.json not found in cache directory")
            return units_data, new_launches_data
        except Exception as e:
            logger.error(f"❌ Error loading cache data: {e}")
            return [], []
    def prepare_units_documents(self, units_data: List[Dict]) -> tuple:
        documents = []
        metadatas = []
        ids = []
        for unit in units_data:
            name_en = unit.get('name_en', '')
            name_ar = unit.get('name_ar', '')
            desc_en = unit.get('desc_en', '')
            desc_ar = unit.get('desc_ar', '')
            compound_name_en = unit.get('compound_name_en', '')
            compound_name_ar = unit.get('compound_name_ar', '')
            apartment_area = unit.get('apartment_area', '')
            price = unit.get('price', '')
            bedrooms = unit.get('Bedrooms', '')
            bathrooms = unit.get('Bathrooms', '')
            delivery_in = unit.get('delivery_in', '')
            installment_years = unit.get('installment_years', '')
            sale_type = unit.get('sale_type', '')
            address = unit.get('address', '')
            # Remove the fields from embedding text and move them to metadata
            text_for_embedding = f"""
Unit_description_ar: الوحدات دي ممكن تكون تحت الإنشاء او محتاجة تشطيب، او جاهزة دلوقتي للاستلام والسكن، وممكن تقدر تروح تعاينها بنفسك قبل ما تشتري
Unit_description_en: These units may are  under construction and may require  finishing — they are ready now for handover and immediate move-in. You can also visit and inspect them yourself before buying.
Name (EN): {name_en}
Name (AR): {name_ar}
Description (EN): {desc_en}
Description (AR): {desc_ar}
Located in:
Compound (EN): {compound_name_en}
Compound (AR): {compound_name_ar}
Sale type: {sale_type}
Address: {address}
""".strip()
            # Convert price to float for filtering
            price_value = 0.0
            try:
                price_str = str(unit.get('price', '0')).replace(',', '').strip()
                if price_str and price_str.isdigit():
                    price_value = float(price_str)
            except (ValueError, TypeError):
                price_value = 0.0
            metadata = {
                "new_image": str(unit.get('new_image', '')),
                "image": str(unit.get('new_image', '')),  # Backward compatibility
                "unit_id": str(unit.get('id', '')),
                "price_value": price_value,
                "bedrooms": str(unit.get('Bedrooms', '')),
                "bathrooms": str(unit.get('Bathrooms', '')),
                "apartment_area": str(unit.get('apartment_area', '')),
                "installment_years": str(unit.get('installment_years', '')),
                "delivery_in": str(unit.get('delivery_in', '')),
            }
            documents.append(text_for_embedding)
            metadatas.append(metadata)
            ids.append(f"unit_{unit.get('id', 'unknown')}")
        return documents, metadatas, ids
    def prepare_new_launches_documents(self, new_launches_data: List[Dict]) -> tuple:
        documents = []
        metadatas = []
        ids = []
        for launch in new_launches_data:
            desc_en = launch.get('desc_en', '')
            desc_ar = launch.get('desc_ar', '')
            compound_name_en = launch.get('compound_name_en', '')
            compound_name_ar = launch.get('compound_name_ar', '')
            developer_name = launch.get('developer_name', '')
            property_type_name = launch.get('property_type_name', '')
            city_name = launch.get('city_name', '')
            text_for_embedding = f"""
New_Launch_Description_ar: الوحدات الـ New Launch هي وحدات لسه المُطور معلِن عنها لأول مرة، ولسه في مرحلة الحجز الأولي قبل ما يبدأ التنفيذ والبناء. الميزة الأساسية إنك بتحجز بدري بسعر أقل، بتختار أحسن موقع داخل الكمبوند، وبتستفيد من تسهيلات في الدفع، حتى لو السعر النهائي لسه مش معلن.
New_Launch_Description_en: New Launch units are newly announced properties offered at early reservation stages, allowing buyers to reserve at lower prices, choose prime locations within the compound, and benefit from flexible payment plans.
Name (EN): {desc_en}
Name (AR): {desc_ar}
Description (EN): {desc_en}
Description (AR): {desc_ar}
Located in:
Compound (EN): {compound_name_en}
Compound (AR): {compound_name_ar}
Developer: {developer_name}
Property Type: {property_type_name}
City: {city_name}
Area: Coming Soon
Price: Coming Soon
Bedrooms: Coming Soon
Bathrooms: Coming Soon
Delivery in: Coming Soon
Installments over: Coming Soon
Sale type: New Launch
Address: {city_name}
""".strip()
            metadata = {
                "new_image": str(launch.get('new_image', '')),
                "launch_id": str(launch.get('id', '')),
            }
            documents.append(text_for_embedding)
            metadatas.append(metadata)
            ids.append(f"launch_{launch.get('id', 'unknown')}")
        return documents, metadatas, ids
    def store_units_in_chroma(self, units_data: List[Dict], retry_on_conflict: bool = True):
        if not units_data:
            logger.warning("⚠️ No units data to store")
            return
        documents, metadatas, ids = self.prepare_units_documents(units_data)
        # Clean metadata to keep only necessary fields
        metadatas = [
            {k: v for k, v in m.items() if k in ['new_image', 'unit_id', 'price_value', 'bedrooms', 'bathrooms', 'apartment_area', 'installment_years', 'delivery_in']}
            for m in metadatas
        ]
        before_count = self._get_collection_count(self.units_collection, "real_estate_units")
        try:
            # Add documents without manually deleting (let ChromaDB handle duplicates)
            self.units_collection.add(documents=documents, metadatas=metadatas, ids=ids)
            after_count = self._get_collection_count(self.units_collection, "real_estate_units")
            if after_count is not None:
                if before_count is not None:
                    added = after_count - before_count
                    if added != len(units_data):
                        logger.warning(
                            f"⚠️ Expected to add {len(units_data)} units but collection count changed by {added}. Total now {after_count}."
                        )
                    else:
                        logger.info(
                            f"✅ Successfully stored {len(units_data)} units in ChromaDB (total now {after_count})"
                        )
                else:
                    logger.info(
                        f"✅ Stored {len(units_data)} units in ChromaDB (current total {after_count}; previous count unavailable)"
                    )
            else:
                logger.info(
                    f"✅ Stored {len(units_data)} units in ChromaDB (total count unavailable)"
                )
        except Exception as e:
            if retry_on_conflict and self._is_embeddings_queue_conflict(e):
                logger.warning(
                    "Detected embeddings queue conflict while storing units. Resetting ChromaDB storage and retrying once..."
                )
                self._reset_chroma_storage(self._chroma_settings)
                self._create_or_get_collections()
                self.store_units_in_chroma(units_data, retry_on_conflict=False)
                return
            logger.error(f"❌ Error storing units in ChromaDB: {e}")
    def store_new_launches_in_chroma(self, new_launches_data: List[Dict], retry_on_conflict: bool = True):
        if not new_launches_data:
            logger.warning("⚠️ No new launches data to store")
            return
        documents, metadatas, ids = self.prepare_new_launches_documents(new_launches_data)
        # Clean metadata to keep only necessary fields
        metadatas = [
            {k: v for k, v in m.items() if k in ['new_image', 'launch_id', 'id', 'name', 'property_type_name', 'city_name']}
            for m in metadatas
        ]
        before_count = self._get_collection_count(self.new_launches_collection, "new_launches")
        try:
            # Add documents without manually deleting (let ChromaDB handle duplicates)
            self.new_launches_collection.add(documents=documents, metadatas=metadatas, ids=ids)
            after_count = self._get_collection_count(self.new_launches_collection, "new_launches")
            if after_count is not None:
                if before_count is not None:
                    added = after_count - before_count
                    if added != len(new_launches_data):
                        logger.warning(
                            f"⚠️ Expected to add {len(new_launches_data)} new launches but collection count changed by {added}. Total now {after_count}."
                        )
                    else:
                        logger.info(
                            f"✅ Successfully stored {len(new_launches_data)} new launches in ChromaDB (total now {after_count})"
                        )
                else:
                    logger.info(
                        f"✅ Stored {len(new_launches_data)} new launches in ChromaDB (current total {after_count}; previous count unavailable)"
                    )
            else:
                logger.info(
                    f"✅ Stored {len(new_launches_data)} new launches in ChromaDB (total count unavailable)"
                )
        except Exception as e:
            if retry_on_conflict and self._is_embeddings_queue_conflict(e):
                logger.warning(
                    "Detected embeddings queue conflict while storing new launches. Resetting ChromaDB storage and retrying once..."
                )
                self._reset_chroma_storage(self._chroma_settings)
                self._create_or_get_collections()
                self.store_new_launches_in_chroma(new_launches_data, retry_on_conflict=False)
                return
            logger.error(f"❌ Error storing new launches in ChromaDB: {e}")
    def get_collection_stats(self) -> Dict[str, int]:
        """Get statistics about the collections"""
        try:
            units_count = self.units_collection.count()
            launches_count = self.new_launches_collection.count()
            return {
                'units_count': units_count,
                'new_launches_count': launches_count,
                'total_count': units_count + launches_count
            }
        except Exception as e:
            logger.error(f"❌ Error getting collection stats: {e}")
            return {'units_count': 0, 'new_launches_count': 0, 'total_count': 0}
    def reset_collections(self):
        """Reset collections to ensure correct dimensions"""
        try:
            # Delete existing collections
            self.client.delete_collection("real_estate_units")
            self.client.delete_collection("new_launches")
            
            # Recreate collections
            self.units_collection = self.client.create_collection(
                name="real_estate_units",
                metadata={"description": "Real estate units data for RAG"},
                embedding_function=self.embedder,
                            # Add HNSW configuration to prevent the error
            hnsw_config={
                "M": 64,
                "ef_construction": 200
            }
            )
            
            self.new_launches_collection = self.client.create_collection(
                name="new_launches",
                metadata={"description": "New property launches data for RAG"},
                embedding_function=self.embedder,
                            # Add HNSW configuration to prevent the error
            hnsw_config={
                "M": 64,
                "ef_construction": 200
            }
            )
            
            logger.info("✅ Collections reset successfully")
        except Exception as e:
            logger.error(f"❌ Error resetting collections: {e}")
    def process_query_with_mmr(self, query: str, n_results: int = 10, filters: Dict = None) -> List[Dict]:
        """Process query using MMR for optimal diversity and relevance"""
        try:
            # Get results from both collections
            units_results = self.search_units(query, n_results, filters)
            launches_results = self.search_new_launches(query, n_results, filters)
            
            # Combine results
            all_results = []
            
            # Add units with type identifier
            for result in units_results:
                if result:
                    result_copy = result.copy()
                    result_copy['source'] = 'units'
                    all_results.append(result_copy)
            
            # Add new launches with type identifier
            for result in launches_results:
                if result:
                    result_copy = result.copy()
                    result_copy['source'] = 'new_launches'
                    all_results.append(result_copy)
            
            # Apply MMR if we have enough results
            if len(all_results) >= 3:
                try:
                    from mmr_search import mmr
                    
                    # Generate query embedding (cached to avoid regeneration)
                    query_embedding = self._get_cached_query_embedding(query)
                    
                    # Extract embeddings from results (if available)
                    embeddings = []
                    valid_results = []
                    
                    for result in all_results:
                        if result and 'embedding' in result:
                            embeddings.append(result['embedding'])
                            valid_results.append(result)
                        elif result:
                            # Create a mock embedding if none exists
                            mock_embedding = [0.1] * 3072
                            embeddings.append(mock_embedding)
                            valid_results.append(result)
                    
                    if embeddings and len(embeddings) > 0:
                        # Apply MMR for diversity
                        mmr_indices = mmr(query_embedding, embeddings, k=min(n_results, len(embeddings)), lambda_param=0.9)
                        
                        # Return MMR-optimized results
                        mmr_results = []
                        for idx in mmr_indices:
                            if idx < len(valid_results):
                                mmr_results.append(valid_results[idx])
                        
                        return mmr_results[:n_results]
                    else:
                        # Fallback: return top results
                        return all_results[:n_results]
                        
                except Exception as e:
                    logger.warning(f"⚠️ MMR processing failed: {e}")
                    return all_results[:n_results]
            else:
                # Return all results if we have fewer than 3
                return all_results[:n_results]
                
        except Exception as e:
            logger.error(f"❌ Error in process_query_with_mmr: {e}")
            return []
    def search_units(self, query: str, n_results: int = 20, filters: Dict = None) -> List[Dict]:
        """Search units collection using MMR-optimized RAG pipeline"""
        import time
        start_time = time.time()
        max_execution_time = 10.0  # 10 seconds timeout (matching functions.py)
        
        try:
            # Build where clause for filters - ChromaDB requires separate clauses for range queries
            where_clauses = []
            if filters:
                # Apply early price filtering (metadata) if provided
                if 'price_min' in filters and filters['price_min']:
                    where_clauses.append({'price_value': {'$gte': filters['price_min']}})
                if 'price_max' in filters and filters['price_max']:
                    where_clauses.append({'price_value': {'$lte': filters['price_max']}})
                # Skip bedrooms/bathrooms for now as requested
                pass  # No filters applied at ChromaDB level for now
            # 1. Get candidates for MMR processing (RAG pipeline) - reduced for performance
            fetch_k = 100  # Reduced further for faster end-to-end latency
            
            # Use separate where clauses for ChromaDB compatibility
            if where_clauses:
                if len(where_clauses) == 1:
                    # Single condition - pass directly
                    results = self.units_collection.query(
                        query_texts=[query],
                        n_results=fetch_k,
                        where=where_clauses[0],
                        include=['embeddings', 'metadatas', 'distances', 'documents'],
                    )
                else:
                    # Multiple conditions - use $and
                    results = self.units_collection.query(
                        query_texts=[query],
                        n_results=fetch_k,
                        where={"$and": where_clauses},
                        include=['embeddings', 'metadatas', 'distances', 'documents'],
                    )
            else:
                # No where clause filters
                results = self.units_collection.query(
                    query_texts=[query],
                    n_results=fetch_k,
                    include=['embeddings', 'metadatas', 'distances', 'documents'],
                )
            
            # Check timeout before MMR processing
            if time.time() - start_time > max_execution_time:
                logger.warning("⚠️ Search timeout - returning early results")
                return self._format_direct_results(results, n_results)
            
            # Check timeout before MMR processing
            if time.time() - start_time > max_execution_time:
                logger.warning("⚠️ New launches search timeout - returning early results")
                return self._format_direct_results(results, n_results)
            
            # 2. Apply MMR directly for diversity and relevance
            if results and 'documents' in results and results['documents'] and results['documents'][0]:
                logger.info(f"📊 Found {len(results['documents'][0])} documents from ChromaDB")
                
                try:
                    # Generate query embedding for MMR (cached to avoid redundant calls)
                    query_embedding = self._get_cached_query_embedding(query)
                    logger.info("✅ Generated query embedding")
                    
                    # Get embeddings from results - prefer ChromaDB embeddings to avoid regeneration
                    embeddings = None
                    if 'embeddings' in results and results['embeddings'] is not None and len(results['embeddings']) > 0 and len(results['embeddings'][0]) > 0:
                        embeddings = results['embeddings'][0]
                        logger.info(f"✅ Found {len(embeddings)} embeddings from ChromaDB")
                    else:
                        logger.warning("⚠️ No embeddings returned from ChromaDB - this will cause performance issues")
                    
                    if embeddings is not None and len(embeddings) > 0:
                        # Post-filter by area (±10%) and installment years (±2) before MMR
                        try:
                            post_indices = self._compute_post_filter_indices(results, filters or {})
                            use_post_filter = len(post_indices) > 0
                            if use_post_filter:
                                logger.info(f"Post-filter candidates: {len(post_indices)}")
                                if len(post_indices) < 20:
                                    logger.info("Post-filter < 20; using full set for MMR")
                                    use_post_filter = False
                            if use_post_filter:
                                # Rebuild results arrays to only include the filtered pool
                                pool_indices = post_indices
                                try:
                                    results['documents'][0] = [results['documents'][0][i] for i in pool_indices]
                                except Exception:
                                    pass
                                try:
                                    if 'metadatas' in results and results['metadatas'] and results['metadatas'][0]:
                                        results['metadatas'][0] = [results['metadatas'][0][i] for i in pool_indices]
                                except Exception:
                                    pass
                                try:
                                    if 'distances' in results and results['distances'] and results['distances'][0]:
                                        results['distances'][0] = [results['distances'][0][i] for i in pool_indices]
                                except Exception:
                                    pass
                                embeddings = [embeddings[i] for i in pool_indices]
                        except Exception as _pf_err:
                            logger.warning(f"Post-filter step failed or skipped: {_pf_err}")
                        # If we're close to timeout, shrink pool size aggressively
                        try:
                            elapsed = time.time() - start_time
                            if elapsed > max_execution_time * 0.6:
                                max_pool = 80
                                if 'documents' in results and results['documents'] and results['documents'][0]:
                                    results['documents'][0] = results['documents'][0][:max_pool]
                                if 'metadatas' in results and results['metadatas'] and results['metadatas'][0]:
                                    results['metadatas'][0] = results['metadatas'][0][:max_pool]
                                if 'distances' in results and results['distances'] and results['distances'][0]:
                                    results['distances'][0] = results['distances'][0][:max_pool]
                                embeddings = embeddings[:max_pool]
                                logger.info(f"⏱️ Near timeout; reduced pool to {len(embeddings)} for MMR")
                        except Exception:
                            pass
                        # Apply faster vectorized MMR for optimal diversity
                        try:
                            from mmr_search import mmr_fast as mmr
                        except Exception:
                            from mmr_search import mmr
                        mmr_indices = mmr(query_embedding, embeddings, k=min(n_results, 20), lambda_param=0.9)
                        logger.info(f"✅ MMR selected {len(mmr_indices)} indices: {mmr_indices}")
                        
                        # Get MMR-optimized results
                        mmr_results = []
                        for i in mmr_indices:
                            if i < len(results['documents'][0]):
                                mmr_results.append({
                                    'document': results['documents'][0][i],
                                    'metadata': results['metadatas'][0][i] if 'metadatas' in results and results['metadatas'] and results['metadatas'][0] else {},
                                    'distance': results['distances'][0][i] if 'distances' in results and results['distances'] and results['distances'][0] else None
                                })
                        
                        logger.info(f"✅ Created {len(mmr_results)} MMR results")
                        
                        # Apply numeric re-ranking with bonus for near targets
                        reranked_results = self._apply_numeric_reranking(mmr_results, filters)
                        logger.info(f"✅ Re-ranked to {len(reranked_results)} results")
                        
                        # Apply deduplication based on ID
                        deduplicated_results = self._deduplicate_results(reranked_results)
                        logger.info(f"✅ Deduplicated to {len(deduplicated_results)} results")
                        
                        # Check timeout before returning results
                        if time.time() - start_time > max_execution_time:
                            logger.warning("⚠️ Search timeout - returning early results")
                            return deduplicated_results[:n_results]
                        
                        return deduplicated_results[:n_results]
                    else:
                        # Fallback: return top results if embeddings not available
                        logger.warning("⚠️ Embeddings not available, returning top results")
                        return self._format_direct_results(results, n_results)
                        
                except ImportError:
                    logger.warning("⚠️ MMR module not available, using direct results")
                    return self._format_direct_results(results, n_results)
                except Exception as e:
                    logger.error(f"❌ Error in MMR processing: {e}")
                    import traceback
                    traceback.print_exc()
                    return self._format_direct_results(results, n_results)
            else:
                logger.warning("⚠️ No documents returned from ChromaDB")
                return []
            
            return []
            
        except Exception as e:
            logger.error(f"❌ Error searching units: {e}")
            return []
        finally:
            execution_time = time.time() - start_time
            if execution_time > max_execution_time:
                logger.warning(f"⚠️ Search took {execution_time:.2f}s (exceeded {max_execution_time}s limit)")
            else:
                logger.info(f"✅ Search completed in {execution_time:.2f}s")
    def _format_direct_results(self, results: Dict, n_results: int) -> List[Dict]:
        """Helper method to format results when MMR is not available"""
        formatted_results = []
        if results and 'documents' in results and results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0][:n_results]):
                metadata = {}
                if 'metadatas' in results and results['metadatas'] and results['metadatas'][0] and i < len(results['metadatas'][0]):
                    metadata = results['metadatas'][0][i]
                
                distance = None
                if 'distances' in results and results['distances'] and results['distances'][0] and i < len(results['distances'][0]):
                    distance = results['distances'][0][i]
                
                formatted_results.append({
                    'document': doc,
                    'metadata': metadata,
                    'distance': distance
                })
        return formatted_results
    def _apply_numeric_reranking(self, results: List[Dict], filters: Dict = None) -> List[Dict]:
        """Apply numeric re-ranking with strict price filtering and bonus for near targets"""
        if not filters:
            return results
        
        def calculate_target_score(item, metadata):
            score = 0
            price = metadata.get('price_value', 0)
            doc_text = str(item.get('document', '')).lower()
            
            # Price filtering - very lenient approach (no penalties, only bonuses)
            if 'price_min' in filters and 'price_max' in filters:
                if price >= filters['price_min'] and price <= filters['price_max']:
                    # Bonus for being within range, with higher bonus for closer to target
                    target_price = (filters['price_min'] + filters['price_max']) / 2
                    if price > 0:
                        price_diff = abs(price - target_price) / target_price
                        # Higher bonus for being close to target (up to 2.0)
                        score += (1 - price_diff) * 2.0
                # No penalty for out-of-range prices - let them through
            
            # Textual relevance bonuses based on query hints (location, property type)
            # Location bonus
            query_location = filters.get('query_location')
            if query_location:
                loc = str(query_location).strip().lower()
                # Simple variants for common Arabic/English mentions
                location_variants = {loc}
                if 'القاهرة الجديدة' in loc or 'new cairo' in loc:
                    location_variants.update({'القاهرة الجديدة', 'new cairo', 'التجمع', 'التجمع الخامس'})
                if 'راس الحكمة' in loc or 'ras al hekma' in loc or 'ras el hekma' in loc:
                    location_variants.update({'راس الحكمة', 'ras el hekma', 'ras al hekma', 'راس الحكمه'})
                # Apply bonus if any variant appears in document text
                if any(variant in doc_text for variant in location_variants):
                    score += 0.8
                else:
                    # Partial token match bonus (looser)
                    for token in loc.split():
                        if token and token in doc_text:
                            score += 0.2
                            break
            # Property type bonus
            query_property_type = filters.get('query_property_type')
            if query_property_type:
                ptype = str(query_property_type).strip().lower()
                type_variants_map = {
                    'شقة': {'شقة', 'شقق', 'apartment', 'apt'},
                    'شاليه': {'شاليه', 'شاليهات', 'chalet'},
                    'فيلا': {'فيلا', 'فيلات', 'villa'},
                }
                variants = set()
                for key, vals in type_variants_map.items():
                    if key in ptype:
                        variants = vals
                        break
                if not variants:
                    variants = {ptype}
                if any(v in doc_text for v in variants):
                    score += 0.5
            # Bedroom matching bonus
            if 'bedrooms' in filters and filters['bedrooms'] > 0:
                bedrooms = metadata.get('bedrooms', 0)
                if bedrooms == filters['bedrooms']:
                    score += 0.3  # Exact match bonus
                elif bedrooms > filters['bedrooms']:
                    score += 0.1  # Higher bedrooms bonus
                else:
                    score -= 0.2  # Penalty for fewer bedrooms
            
            # Bathroom matching bonus
            if 'bathrooms' in filters and filters['bathrooms'] > 0:
                bathrooms = metadata.get('bathrooms', 0)
                if bathrooms == filters['bathrooms']:
                    score += 0.2  # Exact match bonus
                elif bathrooms > filters['bathrooms']:
                    score += 0.1  # Higher bathrooms bonus
            
            return score
        
        # Calculate scores and sort
        scored_results = []
        for item in results:
            metadata = item.get('metadata', {})
            score = calculate_target_score(item, metadata)
            scored_results.append((score, item))
        
        # Sort by score (descending) and return items
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [item for score, item in scored_results]
    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicates based on ID"""
        seen_ids = set()
        unique_results = []
        
        for item in results:
            metadata = item.get('metadata', {})
            item_id = metadata.get('id') or metadata.get('launch_id') or metadata.get('unit_id')
            
            if item_id and item_id not in seen_ids:
                seen_ids.add(item_id)
                unique_results.append(item)
            elif not item_id:
                # If no ID, use document content hash as fallback
                doc_content = str(item.get('document', ''))
                if doc_content not in seen_ids:
                    seen_ids.add(doc_content)
                    unique_results.append(item)
        
        return unique_results
    def search_new_launches(self, query: str, n_results: int = 10, filters: Dict = None) -> List[Dict]:
        """Search new launches collection using MMR-optimized RAG pipeline"""
        import time
        start_time = time.time()
        max_execution_time = 8.0  # 8 seconds timeout
        
        try:
            # Build where clause for filters - ChromaDB requires separate clauses for range queries
            where_clauses = []
            if filters:
                if 'price_min' in filters and filters['price_min']:
                    where_clauses.append({'price_value': {'$gte': filters['price_min']}})
                if 'price_max' in filters and filters['price_max']:
                    where_clauses.append({'price_value': {'$lte': filters['price_max']}})
                if 'bedrooms' in filters and filters['bedrooms']:
                    where_clauses.append({'bedrooms': {'$gte': filters['bedrooms']}})
                if 'bathrooms' in filters and filters['bathrooms']:
                    where_clauses.append({'bathrooms': {'$gte': filters['bathrooms']}})
                if 'location' in filters and filters['location']:
                    # Note: Location filtering would need to be done post-search since it's in the document text
                    pass
            # 1. Get candidates for MMR processing (RAG pipeline) - reduced for performance
            fetch_k = 100  # Reduced from 1000 to 100 for better performance
            
            # Use separate where clauses for ChromaDB compatibility
            if where_clauses:
                if len(where_clauses) == 1:
                    # Single condition - pass directly
                    results = self.new_launches_collection.query(
                        query_texts=[query],
                        n_results=fetch_k,
                        where=where_clauses[0],
                        include=['embeddings', 'metadatas', 'distances', 'documents']
                    )
                else:
                    # Multiple conditions - use $and
                    results = self.new_launches_collection.query(
                        query_texts=[query],
                        n_results=fetch_k,
                        where={"$and": where_clauses},
                        include=['embeddings', 'metadatas', 'distances', 'documents']
                    )
            else:
                results = self.new_launches_collection.query(
                    query_texts=[query],
                    n_results=fetch_k,
                    include=['embeddings', 'metadatas', 'distances', 'documents']
                )
            
            # 2. Apply MMR directly for diversity and relevance
            if results and 'documents' in results and results['documents'] and results['documents'][0]:
                try:
                    # Generate query embedding for MMR
                    query_embedding = self._get_cached_query_embedding(query)
                    
                    # Get embeddings from results or generate them
                    embeddings = None
                    if 'embeddings' in results and results['embeddings'] is not None and len(results['embeddings']) > 0 and len(results['embeddings'][0]) > 0:
                        embeddings = results['embeddings'][0]
                    
                    if embeddings is not None and len(embeddings) > 0:
                        # Apply MMR algorithm for optimal diversity - reduced k for performance
                        from mmr_search import mmr
                        mmr_indices = mmr(query_embedding, embeddings, k=10, lambda_param=0.8)  # Further reduced for better performance
                        
                        # Get MMR-optimized results
                        mmr_results = []
                        for i in mmr_indices:
                            if i < len(results['documents'][0]):
                                metadata = {}
                                if 'metadatas' in results and results['metadatas'] and results['metadatas'][0] and i < len(results['metadatas'][0]):
                                    metadata = results['metadatas'][0][i]
                                
                                distance = None
                                if 'distances' in results and results['distances'] and results['distances'][0] and i < len(results['distances'][0]):
                                    distance = results['distances'][0][i]
                                
                                mmr_results.append({
                                    'document': results['documents'][0][i],
                                    'metadata': metadata,
                                    'distance': distance
                                })
                        
                        # Apply numeric re-ranking with bonus for near targets
                        reranked_results = self._apply_numeric_reranking(mmr_results, filters)
                        
                        # Apply deduplication based on ID
                        deduplicated_results = self._deduplicate_results(reranked_results)
                        
                        return deduplicated_results[:n_results]
                    else:
                        # Fallback: return top results if embeddings not available
                        logger.warning("⚠️ Embeddings not available, returning top results")
                        return self._format_direct_results(results, n_results)
                        
                except ImportError:
                    logger.warning("⚠️ MMR module not available, using direct results")
                    return self._format_direct_results(results, n_results)
                except Exception as e:
                    logger.error(f"❌ Error in MMR processing: {e}")
                    return self._format_direct_results(results, n_results)
            
            return []
            
        except Exception as e:
            logger.error(f"❌ Error searching new launches: {e}")
            return []
        finally:
            execution_time = time.time() - start_time
            if execution_time > max_execution_time:
                logger.warning(f"⚠️ New launches search took {execution_time:.2f}s (exceeded {max_execution_time}s limit)")
            else:
                logger.info(f"✅ New launches search completed in {execution_time:.2f}s")
    def search_all(self, query: str, n_results: int = 20) -> Dict[str, List[Dict]]:
        """Search both collections using MMR-optimized RAG pipeline"""
        try:
            # Use the improved search methods that now include MMR
            units_results = self.search_units(query, n_results // 2)
            launches_results = self.search_new_launches(query, n_results // 2)
            
            return {
                'units': units_results,
                'new_launches': launches_results,
                'total_results': len(units_results) + len(launches_results),
                'search_method': 'mmr_optimized_rag'
            }
        except Exception as e:
            logger.error(f"❌ Error in search_all: {e}")
            return {
                'units': [],
                'new_launches': [],
                'total_results': 0,
                'search_method': 'error'
            }
def get_rag_instance() -> 'RealEstateRAG':
    """Return a shared RealEstateRAG instance."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RealEstateRAG()
    return _rag_instance           
def main():
    logger.info("🚀 Starting ChromaDB RAG setup for real estate data...")
    rag = RealEstateRAG()
    
    # Reset collections to ensure correct dimensions
    rag.reset_collections()
    
    units_data, new_launches_data = rag.load_cache_data()
    rag.store_units_in_chroma(units_data)
    rag.store_new_launches_in_chroma(new_launches_data)
    stats = rag.get_collection_stats()
    logger.info(f"📊 ChromaDB Statistics: {stats}")
if __name__ == "__main__":
    main()
