"""
ChromaDB RAG Integration with Gemini Embeddings for Real Estate Chatbot
"""

import logging
import os
from typing import List, Dict, Any
from gemini_chroma_setup import RealEstateRAGWithGemini

logger = logging.getLogger(__name__)

class ChatbotRAGIntegrationWithGemini:
    def __init__(self, gemini_api_key: str = None):
        """
        Initialize RAG integration with Gemini embeddings
        
        Args:
            gemini_api_key: Google Gemini API key (optional, will use env var if not provided)
        """
        if not gemini_api_key:
            gemini_api_key = os.getenv('GEMINI_API_KEY')
        
        if not gemini_api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable or pass it as parameter.")
        
        self.rag = RealEstateRAGWithGemini(gemini_api_key)
        logger.info("✅ RAG integration with Gemini embeddings initialized")

    def search_properties(self, query: str, search_type: str = "all", n_results: int = 5) -> Dict[str, Any]:
        """
        Search properties using RAG with Gemini embeddings
        
        Args:
            query: User's search query
            search_type: "units", "new_launches", or "all"
            n_results: Number of results to return
        
        Returns:
            Dictionary with search results and formatted response
        """
        try:
            if search_type == "units":
                results = self.rag.search_units(query, n_results)
                return self._format_units_response(results, query)
            
            elif search_type == "new_launches":
                results = self.rag.search_new_launches(query, n_results)
                return self._format_launches_response(results, query)
            
            else:  # "all"
                all_results = self.rag.search_all(query, n_results)
                return self._format_combined_response(all_results, query)
                
        except Exception as e:
            logger.error(f"❌ Error in property search: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "formatted_response": "Sorry, I encountered an error while searching for properties.",
                "embedding_model": "gemini-embedding-001"
            }

    def _format_units_response(self, results: List[Dict], query: str) -> Dict[str, Any]:
        """Format units search results with intelligent filtering"""
        if not results:
            return {
                "success": True,
                "results": [],
                "formatted_response": f"I couldn't find any properties matching '{query}'. Please try different keywords or criteria.",
                "embedding_model": "gemini-embedding-001"
            }
        
        # Filter and rank results intelligently
        filtered_results = self._filter_and_rank_results(results, query)
        
        # Show only top 5 best matches
        top_results = filtered_results[:5]
        
        formatted_response = f"✅ وجدت {len(top_results)} وحدة:\n"
        
        for i, result in enumerate(top_results, 1):
            metadata = result['metadata']
            
            price = metadata.get('price', 'غير متوفر')
            name = metadata.get('name_ar', metadata.get('name_en', 'غير متوفر'))
            bedrooms = metadata.get('bedrooms', 'غير متوفر')
            compound = metadata.get('compound_name_ar', metadata.get('compound_name_en', ''))
            
            # Format price nicely
            if price and price != 'غير متوفر':
                try:
                    price_num = float(str(price).replace(',', ''))
                    price_formatted = f"{price_num:,.0f} جنيه"
                except:
                    price_formatted = f"{price} جنيه"
            else:
                price_formatted = "غير متوفر"
            
            formatted_response += f"� {price_formatted} - {compound} - {name[:50]}{'...' if len(name) > 50 else ''}\n"
        
        return {
            "success": True,
            "results": top_results,
            "formatted_response": formatted_response,
            "embedding_model": "gemini-embedding-001"
        }
    
    def _filter_and_rank_results(self, results: List[Dict], query: str) -> List[Dict]:
        """Filter and rank results based on relevance and criteria"""
        scored_results = []
        
        # Extract query criteria
        query_lower = query.lower()
        target_bedrooms = None
        target_budget = None
        target_location = None
        
        # Extract bedrooms from query
        import re
        bedroom_match = re.search(r'(\d+)\s*(?:bedrooms?|غرف)', query_lower)
        if bedroom_match:
            target_bedrooms = int(bedroom_match.group(1))
        
        # Extract budget from query
        budget_match = re.search(r'budget\s*(\d+)', query_lower)
        if budget_match:
            target_budget = float(budget_match.group(1))
        
        # Extract location from query
        if 'new cairo' in query_lower or 'القاهرة الجديدة' in query_lower:
            target_location = 'new cairo'
        elif 'sheikh zayed' in query_lower or 'الشيخ زايد' in query_lower:
            target_location = 'sheikh zayed'
        
        print(f"🔍 DEBUG: Filtering criteria - bedrooms: {target_bedrooms}, budget: {target_budget}, location: {target_location}")
        
        for i, result in enumerate(results):
            metadata = result['metadata']
            score = 0
            score_details = []
            
            # Base similarity score
            similarity = 1 - result.get('distance', 0.5)
            score += similarity * 100
            score_details.append(f"similarity: {similarity:.2f} ({similarity*100:.0f} pts)")
            
            # Bedroom matching bonus
            if target_bedrooms:
                unit_bedrooms = metadata.get('bedrooms')
                if unit_bedrooms == target_bedrooms:
                    score += 50  # Perfect match
                    score_details.append(f"bedroom perfect match: +50")
                elif abs(unit_bedrooms - target_bedrooms) == 1:
                    score += 20  # Close match
                    score_details.append(f"bedroom close match: +20")
                else:
                    score_details.append(f"bedroom mismatch: {unit_bedrooms} vs {target_bedrooms}")
            
            # Budget matching bonus
            if target_budget:
                unit_price = metadata.get('price')
                if unit_price:
                    try:
                        price_num = float(str(unit_price).replace(',', ''))
                        budget_diff = abs(price_num - target_budget) / target_budget
                        if budget_diff <= 0.2:  # Within 20%
                            score += 40
                            score_details.append(f"budget close match: +40")
                        elif budget_diff <= 0.4:  # Within 40%
                            score += 20
                            score_details.append(f"budget fair match: +20")
                        else:
                            score_details.append(f"budget mismatch: {budget_diff:.1%}")
                    except:
                        pass
            
            # Location matching bonus (INCREASED WEIGHT)
            if target_location:
                unit_name = str(metadata.get('name_en', '') + ' ' + metadata.get('name_ar', '')).lower()
                unit_address = str(metadata.get('address', '')).lower()
                location_text = unit_name + ' ' + unit_address
                
                if target_location == 'new cairo':
                    if any(term in location_text for term in ['new cairo', 'القاهرة الجديدة', 'cairo new']):
                        score += 100  # INCREASED from 30 to 100
                        score_details.append(f"location match: +100")
                    else:
                        score_details.append(f"location mismatch: no New Cairo found")
                elif target_location == 'sheikh zayed':
                    if any(term in location_text for term in ['sheikh zayed', 'الشيخ زايد']):
                        score += 100  # INCREASED from 30 to 100
                        score_details.append(f"location match: +100")
            
            # Debug for top 10 results
            if i < 10:
                name_short = metadata.get('name_ar', metadata.get('name_en', 'N/A'))[:50]
                print(f"   {i+1}. {name_short}... | Score: {score:.0f} | {', '.join(score_details)}")
            
            scored_results.append((score, result))
        
        # Sort by score (highest first)
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        print(f"🔍 DEBUG: Top 5 after ranking:")
        for i, (score, result) in enumerate(scored_results[:5]):
            metadata = result['metadata']
            name_short = metadata.get('name_ar', metadata.get('name_en', 'N/A'))[:50]
            print(f"   {i+1}. Score: {score:.0f} | {name_short}...")
        
        return [result for score, result in scored_results]
        
        return {
            "success": True,
            "results": results,
            "formatted_response": formatted_response,
            "embedding_model": "gemini-embedding-001"
        }

    def _format_launches_response(self, results: List[Dict], query: str) -> Dict[str, Any]:
        """Format new launches search results"""
        if not results:
            return {
                "success": True,
                "results": [],
                "formatted_response": f"I couldn't find any new launches matching '{query}'. Please try different keywords.",
                "embedding_model": "gemini-embedding-001"
            }
        
        formatted_response = f"🏗️ Found {len(results)} new launches matching '{query}' (using Gemini embeddings):\n\n"
        
        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            similarity_score = 1 - result.get('distance', 0) if result.get('distance') else None
            
            formatted_response += f"{i}. **{metadata.get('desc_en', 'N/A')}**"
            if similarity_score:
                formatted_response += f" (Similarity: {similarity_score:.2f})"
            formatted_response += "\n"
            
            formatted_response += f"   🏢 Developer: {metadata.get('developer_name', 'N/A')}\n"
            formatted_response += f"   🏠 Property Type: {metadata.get('property_type_name', 'N/A')}\n"
            formatted_response += f"   🌆 City: {metadata.get('city_name', 'N/A')}\n"
            formatted_response += f"   🏘️ Compound: {metadata.get('compound_name_en', 'N/A')}\n\n"
        
        return {
            "success": True,
            "results": results,
            "formatted_response": formatted_response,
            "embedding_model": "gemini-embedding-001"
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
                "formatted_response": f"I couldn't find any properties or new launches matching '{query}'. Please try different keywords.",
                "embedding_model": "gemini-embedding-001"
            }
        
        formatted_response = f"🔍 Found {total_results} results matching '{query}' (using Gemini embeddings):\n\n"
        
        # Add units section
        if units_results:
            formatted_response += f"🏠 **Available Properties ({len(units_results)}):**\n"
            for i, result in enumerate(units_results[:3], 1):  # Show top 3
                metadata = result['metadata']
                similarity_score = 1 - result.get('distance', 0) if result.get('distance') else None
                formatted_response += f"{i}. {metadata.get('name_en', 'N/A')} - {metadata.get('price', 'N/A')} EGP"
                if similarity_score:
                    formatted_response += f" (Similarity: {similarity_score:.2f})"
                formatted_response += "\n"
            formatted_response += "\n"
        
        # Add new launches section
        if launches_results:
            formatted_response += f"🏗️ **New Launches ({len(launches_results)}):**\n"
            for i, result in enumerate(launches_results[:3], 1):  # Show top 3
                metadata = result['metadata']
                similarity_score = 1 - result.get('distance', 0) if result.get('distance') else None
                formatted_response += f"{i}. {metadata.get('desc_en', 'N/A')[:50]}..."
                if similarity_score:
                    formatted_response += f" (Similarity: {similarity_score:.2f})"
                formatted_response += "\n"
            formatted_response += "\n"
        
        formatted_response += "💡 *Tip: You can ask for more specific details about any property or new launch.*"
        formatted_response += "\n🤖 *Powered by Gemini embeddings for better semantic understanding*"
        
        return {
            "success": True,
            "results": all_results,
            "formatted_response": formatted_response,
            "embedding_model": "gemini-embedding-001"
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
                    "formatted_response": "Sorry, I couldn't find the requested property details.",
                    "embedding_model": "gemini-embedding-001"
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
                "formatted_response": "Sorry, I encountered an error while retrieving property details.",
                "embedding_model": "gemini-embedding-001"
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
        
        formatted_response += "🤖 *Retrieved using Gemini embeddings for optimal semantic matching*"
        
        return {
            "success": True,
            "metadata": metadata,
            "formatted_response": formatted_response,
            "embedding_model": "gemini-embedding-001"
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
        
        formatted_response += "🤖 *Retrieved using Gemini embeddings for optimal semantic matching*"
        
        return {
            "success": True,
            "metadata": metadata,
            "formatted_response": formatted_response,
            "embedding_model": "gemini-embedding-001"
        }

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics and embedding information"""
        try:
            stats = self.rag.get_collection_stats()
            stats.update({
                "embedding_model": "gemini-embedding-001",
                "embedding_dimension": 768,  # Gemini embedding dimension
                "features": [
                    "Semantic search with Gemini embeddings",
                    "Multi-language support (English/Arabic)",
                    "Similarity scoring",
                    "Fast retrieval",
                    "Contextual understanding"
                ]
            })
            return stats
        except Exception as e:
            logger.error(f"❌ Error getting system stats: {e}")
            return {"error": str(e)}

# Example usage function
def example_usage():
    """Example of how to use the RAG integration with Gemini embeddings"""
    try:
        # Initialize with Gemini API key
        rag_integration = ChatbotRAGIntegrationWithGemini()
        
        # Example searches
        queries = [
            "apartment with 3 bedrooms in New Cairo",
            "villa compound with swimming pool",
            "new launch by Hyde Park developer",
            "property under 5 million EGP",
            "compound in Sheikh Zayed area"
        ]
        
        print("🚀 Testing RAG with Gemini Embeddings")
        print("=" * 50)
        
        for query in queries:
            print(f"\n🔍 Searching for: '{query}'")
            result = rag_integration.search_properties(query, search_type="all", n_results=3)
            
            if result["success"]:
                print(result["formatted_response"])
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        # Get system stats
        stats = rag_integration.get_system_stats()
        print(f"\n📊 System Statistics: {stats}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure GEMINI_API_KEY environment variable is set")

if __name__ == "__main__":
    example_usage() 
