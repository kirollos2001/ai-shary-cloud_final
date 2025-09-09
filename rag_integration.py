"""
ChromaDB RAG Integration for Real Estate Chatbot
"""

import logging
from typing import List, Dict, Any
from chroma_rag_setup import get_rag_instance
from functions import classify_query_type_with_llm
import numpy as np

logger = logging.getLogger(__name__)

class ChatbotRAGIntegration:
    def __init__(self):
        """Initialize RAG integration for chatbot"""
        self.rag = get_rag_instance()
        logger.info("✅ RAG integration initialized")

    def search_properties(self, query: str, search_type: str = "all", n_results: int = 5, use_mmr: bool = True, fetch_k: int = 100) -> Dict[str, Any]:
        """
        Search properties using RAG with optional MMR
        Args:
            query: User's search query
            search_type: "units", "new_launches", "all", or "auto"
            n_results: Number of results to return
            use_mmr: Whether to use MMR for retrieval
            fetch_k: Number of candidates to fetch for MMR
        Returns:
            Dictionary with search results and formatted response
        """
        try:
            # Use LLM classifier if search_type is 'auto' or not provided
            if search_type == "auto" or search_type not in ["units", "new_launches", "all"]:
                classification = classify_query_type_with_llm(query)
                if classification == "new_launch":
                    search_type = "new_launches"
                elif classification == "existing_unit":
                    search_type = "units"
                else:
                    search_type = "all"

            # Helper: flatten ChromaDB results
            def flatten_results(results):
                docs = [doc for sublist in results['documents'] for doc in sublist]
                metadatas = [meta for sublist in results['metadatas'] for meta in sublist]
                embeddings = [np.array(e) for sublist in results.get('embeddings', []) for e in sublist]
                return docs, metadatas, embeddings

            # Helper: MMR
            def cosine_similarity_score(embedding1, embedding2):
                dot_product = np.dot(embedding1, embedding2)
                norm_embedding1 = np.linalg.norm(embedding1)
                norm_embedding2 = np.linalg.norm(embedding2)
                if norm_embedding1 == 0 or norm_embedding2 == 0:
                    return 0
                return dot_product / (norm_embedding1 * norm_embedding2)

            def maximal_marginal_relevance(query_embedding, docs, embeddings, k=20, lambda_param=0.5):
                if k > len(docs):
                    k = len(docs)
                selected_indices = []
                unselected_indices = list(range(len(docs)))
                relevance_scores = [cosine_similarity_score(query_embedding, emb) for emb in embeddings]
                for _ in range(k):
                    if not unselected_indices:
                        break
                    best_idx = -1
                    max_mmr_score = -float('inf')
                    for current_idx in unselected_indices:
                        diversity_score = 0
                        if selected_indices:
                            diversity_score = max([cosine_similarity_score(embeddings[current_idx], embeddings[prev_idx]) for prev_idx in selected_indices])
                        mmr_score = 0.5 * relevance_scores[current_idx] - 0.5 * diversity_score
                        if mmr_score > max_mmr_score:
                            max_mmr_score = mmr_score
                            best_idx = current_idx
                    if best_idx != -1:
                        selected_indices.append(best_idx)
                        unselected_indices.remove(best_idx)
                    else:
                        break
                return selected_indices

            # Embed user query with Gemini
            from gemini_chroma_setup import GeminiEmbeddingFunction
            import variables
            embedder = GeminiEmbeddingFunction(variables.GEMINI_API_KEY)
            query_emb = np.array(embedder([query])[0])

            query_args = dict(query_texts=[query], n_results=fetch_k, include=['documents', 'embeddings', 'metadatas'])

            if search_type == "units":
                results = self.rag.units_collection.query(**query_args)
                docs, metadatas, embeddings = flatten_results(results)
                if use_mmr:
                    mmr_indices = maximal_marginal_relevance(query_emb, docs, embeddings, k=n_results)
                    final_results = [
                        {"document": docs[i], "metadata": metadatas[i], "distance": None}
                        for i in mmr_indices
                    ]
                else:
                    final_results = [
                        {"document": doc, "metadata": meta, "distance": None}
                        for doc, meta in zip(docs[:n_results], metadatas[:n_results])
                    ]
                return self._format_units_response(final_results, query)
            elif search_type == "new_launches":
                results = self.rag.new_launches_collection.query(**query_args)
                docs, metadatas, embeddings = flatten_results(results)
                if use_mmr:
                    mmr_indices = maximal_marginal_relevance(query_emb, docs, embeddings, k=n_results)
                    final_results = [
                        {"document": docs[i], "metadata": metadatas[i], "distance": None}
                        for i in mmr_indices
                    ]
                else:
                    final_results = [
                        {"document": doc, "metadata": meta, "distance": None}
                        for doc, meta in zip(docs[:n_results], metadatas[:n_results])
                    ]
                return self._format_launches_response(final_results, query)
            else:  # "all"
                units_results = self.rag.units_collection.query(**query_args)
                launches_results = self.rag.new_launches_collection.query(**query_args)
                units_docs, units_metadatas, units_embeddings = flatten_results(units_results)
                launches_docs, launches_metadatas, launches_embeddings = flatten_results(launches_results)
                if use_mmr:
                    units_mmr_indices = maximal_marginal_relevance(query_emb, units_docs, units_embeddings, k=n_results//2)
                    launches_mmr_indices = maximal_marginal_relevance(query_emb, launches_docs, launches_embeddings, k=n_results//2)
                    all_results = {
                        'units': [
                            {"document": units_docs[i], "metadata": units_metadatas[i], "distance": None}
                            for i in units_mmr_indices
                        ],
                        'new_launches': [
                            {"document": launches_docs[i], "metadata": launches_metadatas[i], "distance": None}
                            for i in launches_mmr_indices
                        ]
                    }
                else:
                    all_results = {
                        'units': [
                            {"document": doc, "metadata": meta, "distance": None}
                            for doc, meta in zip(units_docs[:n_results//2], units_metadatas[:n_results//2])
                        ],
                        'new_launches': [
                            {"document": doc, "metadata": meta, "distance": None}
                            for doc, meta in zip(launches_docs[:n_results//2], launches_metadatas[:n_results//2])
                        ]
                    }
                return self._format_combined_response(all_results, query)
        except Exception as e:
            logger.error(f"❌ Error in property search: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "formatted_response": "Sorry, I encountered an error while searching for properties."
            }

    def _format_units_response(self, results: List[Dict], query: str) -> Dict[str, Any]:
        """Format units search results"""
        if not results:
            return {
                "success": True,
                "results": [],
                "formatted_response": f"I couldn't find any properties matching '{query}'. Please try different keywords or criteria."
            }
        
        formatted_response = f"🏠 Found {len(results)} properties matching '{query}':\n\n"
        
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            formatted_response += f"{i}. **{metadata.get('name_en', 'N/A')}**\n"
            formatted_response += f"   📍 Location: {metadata.get('address', 'N/A')}\n"
            formatted_response += f"   🏢 Compound: {metadata.get('compound_name_en', 'N/A')}\n"
            formatted_response += f"   💰 Price: {metadata.get('price', 'N/A')} EGP\n"
            formatted_response += f"   📐 Area: {metadata.get('apartment_area', 'N/A')} sqm\n"
            formatted_response += f"   🛏️ Bedrooms: {metadata.get('bedrooms', 'N/A')}\n"
            formatted_response += f"   🚿 Bathrooms: {metadata.get('bathrooms', 'N/A')}\n"
            formatted_response += f"   🚗 Garages: {metadata.get('garages', 'N/A')}\n"
            formatted_response += f"   📅 Delivery: {metadata.get('delivery_in', 'N/A')}\n"
            formatted_response += f"   💳 Installment: {metadata.get('installment_years', 'N/A')} years\n"
            formatted_response += f"   🏷️ Sale Type: {metadata.get('sale_type', 'N/A')}\n\n"
        
        return {
            "success": True,
            "results": results,
            "formatted_response": formatted_response
        }

    def _format_launches_response(self, results: List[Dict], query: str) -> Dict[str, Any]:
        """Format new launches search results"""
        if not results:
            return {
                "success": True,
                "results": [],
                "formatted_response": f"I couldn't find any new launches matching '{query}'. Please try different keywords."
            }
        
        formatted_response = f"🏗️ Found {len(results)} new launches matching '{query}':\n\n"
        
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            formatted_response += f"{i}. **{metadata.get('desc_en', 'N/A')}**\n"
            formatted_response += f"   🏢 Developer: {metadata.get('developer_name', 'N/A')}\n"
            formatted_response += f"   🏠 Property Type: {metadata.get('property_type_name', 'N/A')}\n"
            formatted_response += f"   🌆 City: {metadata.get('city_name', 'N/A')}\n"
            formatted_response += f"   🏘️ Compound: {metadata.get('compound_name_en', 'N/A')}\n\n"
        
        return {
            "success": True,
            "results": results,
            "formatted_response": formatted_response
        }

    def _format_combined_response(self, all_results: Dict[str, List[Dict]], query: str) -> Dict[str, Any]:
        """Format combined search results"""
        units_results = all_results.get('units', [])
        launches_results = all_results.get('new_launches', [])
        
        total_results = len(units_results) + len(launches_results)
        
        if total_results == 0:
            return {
                "success": True,
                "results": all_results,
                "formatted_response": f"I couldn't find any properties or new launches matching '{query}'. Please try different keywords."
            }
        
        formatted_response = f"🔍 Found {total_results} results matching '{query}':\n\n"
        
        # Add units section
        if units_results:
            formatted_response += f"🏠 **Available Properties ({len(units_results)}):**\n"
            for i, result in enumerate(units_results[:3], 1):  # Show top 3
                metadata = result['metadata']
                formatted_response += f"{i}. {metadata.get('name_en', 'N/A')} - {metadata.get('price', 'N/A')} EGP\n"
            formatted_response += "\n"
        
        # Add new launches section
        if launches_results:
            formatted_response += f"🏗️ **New Launches ({len(launches_results)}):**\n"
            for i, result in enumerate(launches_results[:3], 1):  # Show top 3
                metadata = result['metadata']
                formatted_response += f"{i}. {metadata.get('desc_en', 'N/A')[:50]}...\n"
            formatted_response += "\n"
        
        formatted_response += "💡 *Tip: You can ask for more specific details about any property or new launch.*"
        
        return {
            "success": True,
            "results": all_results,
            "formatted_response": formatted_response
        }

    def get_property_details(self, property_id: str, property_type: str = "unit") -> Dict[str, Any]:
        """Get detailed information about a specific property"""
        try:
            if property_type == "unit":
                results = self.rag.search_units(f"unit_id:{property_id}", n_results=1)
            else:
                results = self.rag.search_new_launches(f"launch_id:{property_id}", n_results=1)
            
            if not results:
                return {
                    "success": False,
                    "error": "Property not found",
                    "formatted_response": "Sorry, I couldn't find the requested property details."
                }
            
            result = results[0]
            metadata = result['metadata']
            
            if property_type == "unit":
                return self._format_detailed_unit(metadata)
            else:
                return self._format_detailed_launch(metadata)
                
        except Exception as e:
            logger.error(f"❌ Error getting property details: {e}")
            return {
                "success": False,
                "error": str(e),
                "formatted_response": "Sorry, I encountered an error while retrieving property details."
            }

    def _format_detailed_unit(self, metadata: Dict) -> Dict[str, Any]:
        """Format detailed unit information"""
        formatted_response = f"🏠 **{metadata.get('name_en', 'N/A')}**\n\n"
        formatted_response += f"📍 **Address:** {metadata.get('address', 'N/A')}\n"
        formatted_response += f"🏢 **Compound:** {metadata.get('compound_name_en', 'N/A')} / {metadata.get('compound_name_ar', 'N/A')}\n"
        formatted_response += f"💰 **Price:** {metadata.get('price', 'N/A')} EGP\n"
        formatted_response += f"📐 **Area:** {metadata.get('apartment_area', 'N/A')} sqm\n"
        formatted_response += f"🛏️ **Bedrooms:** {metadata.get('bedrooms', 'N/A')}\n"
        formatted_response += f"🚿 **Bathrooms:** {metadata.get('bathrooms', 'N/A')}\n"
        formatted_response += f"🚗 **Garages:** {metadata.get('garages', 'N/A')}\n"
        formatted_response += f"📅 **Delivery:** {metadata.get('delivery_in', 'N/A')}\n"
        formatted_response += f"💳 **Installment:** {metadata.get('installment_years', 'N/A')} years\n"
        formatted_response += f"🏷️ **Sale Type:** {metadata.get('sale_type', 'N/A')}\n\n"
        
        if metadata.get('desc_en'):
            formatted_response += f"📝 **Description:** {metadata.get('desc_en', 'N/A')}\n\n"
        
        if metadata.get('desc_ar'):
            formatted_response += f"📝 **الوصف:** {metadata.get('desc_ar', 'N/A')}\n\n"
        
        return {
            "success": True,
            "metadata": metadata,
            "formatted_response": formatted_response
        }

    def _format_detailed_launch(self, metadata: Dict) -> Dict[str, Any]:
        """Format detailed new launch information"""
        formatted_response = f"🏗️ **{metadata.get('desc_en', 'N/A')}**\n\n"
        formatted_response += f"🏢 **Developer:** {metadata.get('developer_name', 'N/A')}\n"
        formatted_response += f"🏠 **Property Type:** {metadata.get('property_type_name', 'N/A')}\n"
        formatted_response += f"🌆 **City:** {metadata.get('city_name', 'N/A')}\n"
        formatted_response += f"🏘️ **Compound:** {metadata.get('compound_name_en', 'N/A')} / {metadata.get('compound_name_ar', 'N/A')}\n\n"
        
        if metadata.get('desc_ar'):
            formatted_response += f"📝 **الوصف:** {metadata.get('desc_ar', 'N/A')}\n\n"
        
        return {
            "success": True,
            "metadata": metadata,
            "formatted_response": formatted_response
        }

# Example usage function
def example_usage():
    """Example of how to use the RAG integration"""
    rag_integration = ChatbotRAGIntegration()
    
    # Example searches
    queries = [
        "apartment with 3 bedrooms in New Cairo",
        "villa compound",
        "new launch by Hyde Park",
        "property under 5 million EGP"
    ]
    
    for query in queries:
        print(f"\n🔍 Searching for: '{query}'")
        result = rag_integration.search_properties(query, search_type="all", n_results=3)
        
        if result["success"]:
            print(result["formatted_response"])
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")

def test_gemini_query_embedding():
    from gemini_chroma_setup import GeminiEmbeddingFunction
    import variables
    embedder = GeminiEmbeddingFunction(variables.GEMINI_API_KEY)
    emb = embedder(["شقة في القاهرة الجديدة"])
    print(f"Gemini embedding dimension: {len(emb[0])}")

def test_mmr_retrieval():
    print("\n--- MMR Retrieval Test ---")
    rag = ChatbotRAGIntegration()
    query = "شقة في القاهرة الجديدة"
    result = rag.search_properties(query, search_type="units", n_results=5, use_mmr=True, fetch_k=100)
    print(f"Query: {query}")
    print("Top 5 MMR-selected results:")
    for i, r in enumerate(result["results"], 1):
        print(f"Result {i}:")
        print("Document:", r["document"])
        print("Metadata:", r["metadata"])
        print("-" * 40)
    print("\nFormatted response:")
    print(result["formatted_response"])

if __name__ == "__main__":
    test_gemini_query_embedding()
    test_mmr_retrieval()
    example_usage() 
