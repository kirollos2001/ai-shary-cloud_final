import numpy as np
from typing import List, Dict, Any, Callable
import google.generativeai as genai
import logging

logger = logging.getLogger(__name__)

# --- MMR Implementation ---
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

def mmr(query_embedding, doc_embeddings, k=10, lambda_param=0.9):
    """
    Maximal Marginal Relevance (MMR) for result diversification.
    Args:
        query_embedding: Embedding of the query
        doc_embeddings: List of embeddings for candidate documents
        k: Number of results to return
        lambda_param: Trade-off between relevance and diversity
    Returns:
        List of selected indices
    """
    selected = []
    candidates = list(range(len(doc_embeddings)))
    
    # Compute similarity to query for all docs
    sim_to_query = [cosine_similarity(query_embedding, emb) for emb in doc_embeddings]
    
    for _ in range(k):
        if not candidates:
            break
        if not selected:
            # Select the most relevant doc first
            idx = int(np.argmax(sim_to_query))
            selected.append(idx)
            candidates.remove(idx)
        else:
            mmr_scores = []
            for idx in candidates:
                relevance = sim_to_query[idx]
                diversity = max([cosine_similarity(doc_embeddings[idx], doc_embeddings[s]) for s in selected])
                mmr_score = lambda_param * relevance - (1 - lambda_param) * diversity
                mmr_scores.append(mmr_score)
            best_idx = candidates[int(np.argmax(mmr_scores))]
            selected.append(best_idx)
            candidates.remove(best_idx)
    return selected

# --- Gemini Embedding Utility ---
class GeminiEmbedder:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.embedding_model = genai.get_model("models/embedding-001")
    def embed(self, text: str) -> List[float]:
        result = self.embedding_model.embed_content(content=text, task_type="retrieval_query")
        return result.embedding
    def embed_many(self, texts: List[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]

# --- Combined Search ---
def combined_mmr_keyword_search(
    query: str,
    chroma_collection,
    keyword_search_fn: Callable[[str], List[Dict]],
    gemini_api_key: str,
    k: int = 10,
    fetch_k: int = 1000,
    lambda_param: float = 0.8,
    price_filter: float = None
) -> List[Dict]:
    """
    Combined price-filtered MMR (embedding) and keyword search.
    Args:
        query: User query
        chroma_collection: ChromaDB collection object
        keyword_search_fn: Function for keyword search (returns list of dicts)
        gemini_api_key: Gemini API key
        k: Number of final results
        fetch_k: Number of candidates to fetch from ChromaDB
        lambda_param: MMR trade-off
        price_filter: Price to filter by (if None, no price filtering)
    Returns:
        List of merged, deduplicated results
    """
    embedder = GeminiEmbedder(gemini_api_key)
    query_emb = embedder.embed(query)
    
    # 1. Price-based filtering first (if price_filter is provided)
    if price_filter is not None:
        # Calculate price range with 10% tolerance
        tolerance = 0.10
        min_price = price_filter * (1 - tolerance)
        max_price = price_filter * (1 + tolerance)
        
        # Filter by price range using metadata
        price_filtered_results = chroma_collection.query(
            query_texts=[query],
            n_results=fetch_k,
            where={
                "price_value": {
                    "$gte": min_price,
                    "$lte": max_price
                }
            }
        )
        
        # If price filtering returns results, use those for embedding search
        if price_filtered_results['documents'][0]:
            docs = price_filtered_results['documents'][0]
            metadatas = price_filtered_results['metadatas'][0]
            embeddings = price_filtered_results['embeddings'][0] if 'embeddings' in price_filtered_results else None
            
            if embeddings is None:
                # If Chroma doesn't return embeddings, re-embed docs
                embeddings = embedder.embed_many(docs)
            
            # 2. MMR selection on price-filtered results
            mmr_indices = mmr(query_emb, embeddings, k=k, lambda_param=lambda_param)
            mmr_results = [
                {
                    'document': docs[i],
                    'metadata': metadatas[i],
                    'score': cosine_similarity(query_emb, embeddings[i]),
                    'source': 'price_filtered_embedding_mmr'
                }
                for i in mmr_indices
            ]
        else:
            # No results with price filter, return empty
            mmr_results = []
    else:
        # No price filtering - use original approach
        # 1. Embedding search (fetch_k)
        chroma_results = chroma_collection.query(query_texts=[query], n_results=fetch_k)
        docs = chroma_results['documents'][0]
        metadatas = chroma_results['metadatas'][0]
        embeddings = chroma_results['embeddings'][0] if 'embeddings' in chroma_results else None
        
        if embeddings is None:
            # If Chroma doesn't return embeddings, re-embed docs
            embeddings = embedder.embed_many(docs)
        
        # 2. MMR selection
        mmr_indices = mmr(query_emb, embeddings, k=k, lambda_param=lambda_param)
        mmr_results = [
            {
                'document': docs[i],
                'metadata': metadatas[i],
                'score': cosine_similarity(query_emb, embeddings[i]),
                'source': 'embedding_mmr'
            }
            for i in mmr_indices
        ]
    
    # 3. Traditional keyword search (also apply price filter if available)
    keyword_results = keyword_search_fn(query)
    
    # Apply price filter to keyword results if price_filter is provided
    if price_filter is not None:
        tolerance = 0.10
        min_price = price_filter * (1 - tolerance)
        max_price = price_filter * (1 + tolerance)
        
        filtered_keyword_results = []
        for result in keyword_results:
            # Extract price from keyword result metadata
            result_price = result.get('metadata', {}).get('price_value', 0)
            if min_price <= result_price <= max_price:
                result['source'] = 'price_filtered_keyword'
                filtered_keyword_results.append(result)
        keyword_results = filtered_keyword_results
    else:
        for r in keyword_results:
            r['source'] = 'keyword'
    
    # 4. Merge and deduplicate (by unique id if available, else by doc text)
    seen = set()
    merged = []
    for r in mmr_results + keyword_results:
        uid = r['metadata'].get('unit_id') or r['metadata'].get('launch_id') or r.get('document')
        if uid not in seen:
            merged.append(r)
            seen.add(uid)
    
    return merged[:k]

def price_filtered_mmr_search(
    query: str,
    chroma_collection,
    gemini_api_key: str,
    target_price: float,
    k: int = 10,
    fetch_k: int = 1000,
    lambda_param: float = 0.9
) -> List[Dict]:
    """
    Simplified price-filtered MMR search without keyword search.
    Args:
        query: User query
        chroma_collection: ChromaDB collection object
        gemini_api_key: Gemini API key
        target_price: Target price for filtering
        k: Number of results to return
        fetch_k: Number of candidates to fetch from ChromaDB
        lambda_param: MMR trade-off
    Returns:
        List of price-filtered MMR results
    """
    return combined_mmr_keyword_search(
        query=query,
        chroma_collection=chroma_collection,
        keyword_search_fn=lambda q: [],  # No keyword search
        gemini_api_key=gemini_api_key,
        k=k,
        fetch_k=fetch_k,
        lambda_param=lambda_param,
        price_filter=target_price
    ) 
