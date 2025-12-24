import tiktoken
from typing import List

class TextChunker:
    """Split text into chunks with overlap"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def chunk(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        
        # Tokenize the text
        tokens = self.encoding.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            # Get chunk
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            
            # Decode back to text
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            # Move start position with overlap
            start += self.chunk_size - self.chunk_overlap
        
        print(f"✓ Created {len(chunks)} chunks")
        return chunks
