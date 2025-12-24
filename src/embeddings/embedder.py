from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class Embedder:
    """Generate embeddings using free Sentence Transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print("📥 Loading embedding model (first time downloads ~80MB)...")
        self.model = SentenceTransformer(model_name)
        print("✓ Model loaded successfully")
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        
        print(f"🔄 Embedding {len(texts)} chunks...")
        embeddings = self.model.encode(
            texts, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"✓ Generated {len(embeddings)} embeddings")
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a single query"""
        return self.model.encode([query], convert_to_numpy=True)[0]
