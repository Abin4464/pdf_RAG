from typing import List, Tuple

class Retriever:
    """Retrieve relevant context from vector store"""
    
    def __init__(self, vectorstore, embedder, top_k: int = 3):
        self.vectorstore = vectorstore
        self.embedder = embedder
        self.top_k = top_k
    
    def retrieve(self, query: str) -> List[Tuple[str, float]]:
        """Retrieve top-k relevant chunks for a query"""
        
        print(f"\n🔍 Searching for: '{query}'")
        
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)
        
        # Search in vector store
        results = self.vectorstore.search(query_embedding, k=self.top_k)
        
        print(f"✓ Retrieved {len(results)} relevant chunks")
        return results
    
    def build_context(self, results: List[Tuple[str, float]]) -> str:
        """Build context string from retrieved chunks"""
        
        context = ""
        for i, (text, distance) in enumerate(results, 1):
            context += f"[Chunk {i}]\n{text}\n\n"
        
        return context.strip()
