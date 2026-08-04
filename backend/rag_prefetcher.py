import re
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util

# Global variable to reuse the model instance across runs (saves memory and initialization time)
_embedding_model = None

def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        print("[RAG] Loading SentenceTransformer model ('all-MiniLM-L6-v2')...")
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """
    Splits long paper text or abstract into sliding window word chunks.
    """
    if not text or len(text.strip()) == 0:
        return []
        
    words = text.split()   
    chunks = []
    
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start += (chunk_size - overlap)
        
    return chunks

def prefetch_top_chunks(query: str, paper_texts: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Given a search query and a list of paper text blocks,
    splits them into chunks, embeds them, and returns the top-k most relevant chunks.
    
    Each returned chunk is represented as:
    {
        "text": str,
        "score": float
    }
    """
    if not paper_texts:
        return []
        
    # 1. Collect all chunks across all provided papers
    all_chunks = []
    for text in paper_texts:
        all_chunks.extend(chunk_text(text))
        
    if not all_chunks:
        return []
        
    try:
        model = get_embedding_model()
        
        # 2. Generate embeddings for query and all text chunks
        query_embedding = model.encode(query, convert_to_tensor=True)
        chunk_embeddings = model.encode(all_chunks, convert_to_tensor=True)
        
        # 3. Calculate cosine similarities
        cosine_scores = util.cos_sim(query_embedding, chunk_embeddings)[0]
        
        # 4. Sort and return top-k chunks
        scored_chunks = []
        for idx, score in enumerate(cosine_scores):
            scored_chunks.append({
                "text": all_chunks[idx],
                "score": float(score)
            })
            
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]
        
    except Exception as e:
        print(f"[RAG] Error during prefetch: {e}")
        # Fallback: if model load or embedding fails, return raw snippets as a fallback list
        return [{"text": c[:400] + "...", "score": 0.5} for c in all_chunks[:top_k]]
