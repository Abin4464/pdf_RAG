import faiss
import numpy as np
import pickle
from typing import List, Tuple

class FAISSStore:
    """FAISS vector database for storing and retrieving embeddings"""
    
    def __init__(self, dimension: int = 384):  # Changed from 1536 to 384
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []
    
    def add(self, embeddings: np.ndarray, texts: List[str]):
        """Add embeddings and corresponding texts to the index"""
        
        # Ensure embeddings are float32
        embeddings = embeddings.astype('float32')
        
        # Add to FAISS index
        self.index.add(embeddings)
        
        # Store texts
        self.texts.extend(texts)
        
        print(f"✓ Added {len(texts)} documents to FAISS index")
        print(f"  Total documents in index: {self.index.ntotal}")
    
    def search(self, query_embedding: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        """Search for top-k most similar documents"""
        
        # Ensure query is float32 and 2D
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Get results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.texts):
                results.append((self.texts[idx], float(distances[0][i])))
        
        return results
    
    def save(self, path: str):
        """Save index and texts to disk"""
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.texts", 'wb') as f:
            pickle.dump(self.texts, f)
        print(f"✓ Saved index to {path}")
    
    def load(self, path: str):
        """Load index and texts from disk"""
        self.index = faiss.read_index(f"{path}.index")
        with open(f"{path}.texts", 'rb') as f:
            self.texts = pickle.load(f)
        print(f"✓ Loaded index from {path}")