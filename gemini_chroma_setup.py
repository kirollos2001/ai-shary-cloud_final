import chromadb
import json
import os
import logging
import google.generativeai as genai
from typing import List, Dict, Any
from chromadb.config import Settings
import asyncio
import time
import os
import variables

os.environ["GEMINI_API_KEY"] = variables.GEMINI_API_KEY
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiEmbeddingFunction:
    """Custom embedding function using Gemini embeddings"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)

    def name(self) -> str:
        """Return the name of this embedding function"""
        return "gemini-embedding-001"

    def __call__(self, input: list) -> list:
        try:
            result = genai.embed_content(
                model="models/gemini-embedding-001",
                content=input,
                task_type="SEMANTIC_SIMILARITY",
                output_dimensionality=3072
            )
            # result['embedding'] is a list of lists (one per input)
            return [e for e in result['embedding']]
        except Exception as e:
            logger.error(f"❌ Error generating embedding for text: {e}")
            return [[0.0] * 3072 for _ in input]
    
    def embed(self, text: str) -> List[float]:
        """Single text embedding - required by ChromaDB"""
        try:
            result = genai.embed_content(
                model="models/gemini-embedding-001",
                content=text,
                task_type="SEMANTIC_SIMILARITY",
                output_dimensionality=3072
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"❌ Error generating single embedding: {e}")
            return [0.0] * 3072  # Return zero vector as fallback

def clean_metadata(metadata):
    return {k: (v if v is not None else "") for k, v in metadata.items()}

class RealEstateRAGWithGemini:
    def __init__(self, gemini_api_key: str, persist_directory: str = "./chroma_db_gemini"):
        """
        Initialize ChromaDB with Gemini embeddings
        
        Args:
            gemini_api_key: Google Gemini API key
            persist_directory: Directory to store ChromaDB data
        """
        self.persist_directory = persist_directory
        self.gemini_api_key = gemini_api_key
        
        # Initialize Gemini embedding function
        self.embedding_function = GeminiEmbeddingFunction(gemini_api_key)
        
        # Initialize ChromaDB client with Gemini embeddings
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Create collections with Gemini embeddings
        self.units_collection = self.client.get_or_create_collection(
            name="real_estate_units_gemini",
            embedding_function=self.embedding_function,
            metadata={"description": "Real estate units data with Gemini embeddings"}
        )
        
        self.new_launches_collection = self.client.get_or_create_collection(
            name="new_launches_gemini",
            embedding_function=self.embedding_function,
            metadata={"description": "New property launches data with Gemini embeddings"}
        )
        
        logger.info("✅ ChromaDB initialized with Gemini embeddings")

    def reset_collections(self):
        """Reset collections to start fresh with correct dimensions"""
        try:
            # Delete existing collections
            try:
                self.client.delete_collection("real_estate_units_gemini")
                logger.info("🗑️ Deleted existing real_estate_units_gemini collection")
            except Exception:
                logger.info("📋 real_estate_units_gemini collection doesn't exist, creating new one")
            
            try:
                self.client.delete_collection("new_launches_gemini")
                logger.info("🗑️ Deleted existing new_launches_gemini collection")
            except Exception:
                logger.info("📋 new_launches_gemini collection doesn't exist, creating new one")
            
            # Create fresh collections with correct embedding dimensions
            self.units_collection = self.client.create_collection(
                name="real_estate_units_gemini",
                metadata={"description": "Real estate units data with Gemini 3072-dim embeddings"},
                embedding_function=self.embedding_function
            )

            self.new_launches_collection = self.client.create_collection(
                name="new_launches_gemini",
                metadata={"description": "New property launches data with Gemini 3072-dim embeddings"},
                embedding_function=self.embedding_function
            )
            
            logger.info("✅ Collections reset with 3072-dim Gemini embeddings")
        except Exception as e:
            logger.error(f"❌ Failed to reset collections: {e}")

    def load_cache_data(self) -> tuple:
        """Load data from cache files"""
        try:
            # Load units data
            units_path = os.path.join("cache", "units.json")
            if os.path.exists(units_path):
                with open(units_path, 'r', encoding='utf-8') as f:
                    units_data = json.load(f)
                logger.info(f"✅ Loaded {len(units_data)} units from cache")
            else:
                units_data = []
                logger.warning("⚠️ units.json not found in cache directory")

            # Load new launches data
            new_launches_path = os.path.join("cache", "new_launches.json")
            if os.path.exists(new_launches_path):
                with open(new_launches_path, 'r', encoding='utf-8') as f:
                    new_launches_data = json.load(f)
                logger.info(f"✅ Loaded {len(new_launches_data)} new launches from cache")
            else:
                new_launches_data = []
                logger.warning("⚠️ new_launches.json not found in cache directory")

            return units_data, new_launches_data
            
        except Exception as e:
            logger.error(f"❌ Error loading cache data: {e}")
            return [], []

    def prepare_units_documents(self, units_data: List[Dict]) -> tuple:
        """Prepare units data for ChromaDB storage with Gemini embeddings"""
        documents = []
        metadatas = []
        ids = []
        
        for unit in units_data:
            # Create a comprehensive document text (fields moved to metadata)
            doc_text = f"""
            Property: {unit.get('name_en', '')} / {unit.get('name_ar', '')}
            Unit ID: {unit.get('id', '')}
            Address: {unit.get('address', '')}
            Sale Type: {unit.get('sale_type', '')}
            Compound: {unit.get('compound_name_en', '')} / {unit.get('compound_name_ar', '')}
            Description: {unit.get('desc_en', '')}
            Arabic Description: {unit.get('desc_ar', '')}
            Unit Status Arabic: الوحدات دي مش تحت الإنشاء ولا محتاجة تشطيب، بل جاهزة دلوقتي للاستلام والسكن، وكمان تقدر تروح تعاينها بنفسك قبل ما تشتري.
            Unit Status English: These units are not under construction and don't require any finishing — they are ready now for handover and immediate move-in. You can also visit and inspect them yourself before buying.
            """.strip()
            
            # Prepare metadata
            metadata = {
                "new_image": unit.get('new_image', ''),
                "name_en": unit.get('name_en', ''),
                "name_ar": unit.get('name_ar', ''),
                "apartment_area": unit.get('apartment_area', ''),
                "price": unit.get('price', ''),
                "image": unit.get('image', ''),
                "video": unit.get('video', ''),
                "address": unit.get('address', ''),
                "delivery_in": unit.get('delivery_in', ''),
                "bedrooms": unit.get('Bedrooms', ''),
                "bathrooms": unit.get('Bathrooms', ''),
                "garages": unit.get('garages', ''),
                "installment_years": unit.get('installment_years', ''),
                "desc_en": unit.get('desc_en', ''),
                "desc_ar": unit.get('desc_ar', ''),
                "compound_name_ar": unit.get('compound_name_ar', ''),
                "compound_name_en": unit.get('compound_name_en', ''),
                "compounds_image": unit.get('compounds_image', ''),
                "compounds_video": unit.get('compounds_video', ''),
                "sale_type": unit.get('sale_type', ''),
                "unit_id": unit.get('id', ''),
                "embedding_model": "gemini-embedding-001"
            }
            metadata = clean_metadata(metadata)
            
            documents.append(doc_text)
            metadatas.append(metadata)
            ids.append(f"unit_{unit.get('id', 'unknown')}")
        
        return documents, metadatas, ids

    def prepare_new_launches_documents(self, new_launches_data: List[Dict]) -> tuple:
        """Prepare new launches data for ChromaDB storage with Gemini embeddings"""
        documents = []
        metadatas = []
        ids = []
        
        for launch in new_launches_data:
            # Create a comprehensive document text
            doc_text = f"""
            New Launch: {launch.get('desc_en', '')} / {launch.get('desc_ar', '')}
            Unit ID: {launch.get('id', '')}
            Developer: {launch.get('developer_name', '')}
            Property Type: {launch.get('property_type_name', '')}
            City: {launch.get('city_name', '')}
            Compound: {launch.get('compound_name_en', '')} / {launch.get('compound_name_ar', '')}
            Unit Status Arabic: الوحدة دي لسه في مرحلة البناء، ومش جاهزة لا للاستلام ولا إنك تعاينها بنفسك، وهتحتاج تنتظر فترة معينة حسب الجدول الزمني للمشروع.
            Unit Status English: This unit is still under construction, not ready for handover or inspection, and you will need to wait for a certain period according to the project timeline.
            """.strip()
            
            # Prepare metadata
            metadata = {
                "new_image": launch.get('new_image', ''),
                "desc_en": launch.get('desc_en', ''),
                "desc_ar": launch.get('desc_ar', ''),
                "image": launch.get('image', ''),
                "developer_name": launch.get('developer_name', ''),
                "property_type_name": launch.get('property_type_name', ''),
                "city_name": launch.get('city_name', ''),
                "compound_name_ar": launch.get('compound_name_ar', ''),
                "compound_name_en": launch.get('compound_name_en', ''),
                "launch_id": launch.get('id', ''),
                "embedding_model": "gemini-embedding-001"
            }
            metadata = clean_metadata(metadata)
            
            documents.append(doc_text)
            metadatas.append(metadata)
            ids.append(f"launch_{launch.get('id', 'unknown')}")
        
        return documents, metadatas, ids

    def store_units_in_chroma(self, units_data: List[Dict]):
        """Store units data in ChromaDB with Gemini embeddings"""
        if not units_data:
            logger.warning("⚠️ No units data to store")
            return
        
        documents, metadatas, ids = self.prepare_units_documents(units_data)
        
        try:
            # Clear existing data
            self.units_collection.delete(where={"type": "unit"})
            
            # Add new data with Gemini embeddings
            logger.info("🔄 Generating Gemini embeddings for units...")
            start_time = time.time()
            
            self.units_collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            end_time = time.time()
            logger.info(f"✅ Successfully stored {len(units_data)} units in ChromaDB with Gemini embeddings")
            logger.info(f"⏱️ Embedding generation took {end_time - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"❌ Error storing units in ChromaDB: {e}")

    def store_new_launches_in_chroma(self, new_launches_data: List[Dict]):
        """Store new launches data in ChromaDB with Gemini embeddings"""
        if not new_launches_data:
            logger.warning("⚠️ No new launches data to store")
            return
        
        documents, metadatas, ids = self.prepare_new_launches_documents(new_launches_data)
        
        try:
            # Clear existing data
            self.new_launches_collection.delete(where={"type": "new_launch"})
            
            # Add new data with Gemini embeddings
            logger.info("🔄 Generating Gemini embeddings for new launches...")
            start_time = time.time()
            
            self.new_launches_collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            end_time = time.time()
            logger.info(f"✅ Successfully stored {len(new_launches_data)} new launches in ChromaDB with Gemini embeddings")
            logger.info(f"⏱️ Embedding generation took {end_time - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"❌ Error storing new launches in ChromaDB: {e}")

    def search_units(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search units collection using Gemini embeddings"""
        try:
            results = self.units_collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        'document': doc,
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if 'distances' in results else None
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Error searching units: {e}")
            return []

    def search_new_launches(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search new launches collection using Gemini embeddings"""
        try:
            results = self.new_launches_collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        'document': doc,
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if 'distances' in results else None
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Error searching new launches: {e}")
            return []

    def search_all(self, query: str, n_results: int = 10) -> Dict[str, List[Dict]]:
        """Search both collections using Gemini embeddings"""
        units_results = self.search_units(query, n_results // 2)
        launches_results = self.search_new_launches(query, n_results // 2)
        
        return {
            'units': units_results,
            'new_launches': launches_results
        }

    def get_collection_stats(self) -> Dict[str, int]:
        """Get statistics about stored data"""
        try:
            units_count = self.units_collection.count()
            launches_count = self.new_launches_collection.count()
            
            return {
                'units_count': units_count,
                'new_launches_count': launches_count,
                'total_documents': units_count + launches_count,
                'embedding_model': 'gemini-embedding-001'
            }
        except Exception as e:
            logger.error(f"❌ Error getting collection stats: {e}")
            return {'units_count': 0, 'new_launches_count': 0, 'total_documents': 0, 'embedding_model': 'gemini-embedding-001'}

def main():
    """Main function to set up ChromaDB with Gemini embeddings"""
    logger.info("🚀 Starting ChromaDB RAG setup with Gemini embeddings...")
    
    # Get Gemini API key from environment or config
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if not gemini_api_key:
        logger.error("❌ GEMINI_API_KEY environment variable not set")
        return
    
    # Initialize RAG system with Gemini embeddings
    rag = RealEstateRAGWithGemini(gemini_api_key)
    
    # Reset collections to ensure correct dimensions
    rag.reset_collections()
    
    # Load data from cache
    units_data, new_launches_data = rag.load_cache_data()
    
    # Store data in ChromaDB with Gemini embeddings
    rag.store_units_in_chroma(units_data)
    rag.store_new_launches_in_chroma(new_launches_data)
    
    # Get and display statistics
    stats = rag.get_collection_stats()
    logger.info(f"📊 ChromaDB Statistics: {stats}")
    
    # Test search functionality
    logger.info("🔍 Testing search functionality with Gemini embeddings...")
    
    # Test units search
    units_results = rag.search_units("apartment with 3 bedrooms", n_results=3)
    logger.info(f"✅ Units search test returned {len(units_results)} results")
    
    # Test new launches search
    launches_results = rag.search_new_launches("new compound", n_results=3)
    logger.info(f"✅ New launches search test returned {len(launches_results)} results")
    
    logger.info("🎉 ChromaDB RAG setup with Gemini embeddings completed successfully!")

if __name__ == "__main__":
    main() 