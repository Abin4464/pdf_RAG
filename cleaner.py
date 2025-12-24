import re

class TextCleaner:
    """Clean and normalize extracted text"""
    
    @staticmethod
    def clean(text: str) -> str:
        """Remove extra whitespace, special characters, and normalize text"""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-\']', '', text)
        
        # Remove multiple periods
        text = re.sub(r'\.{2,}', '.', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        print(f"✓ Cleaned text: {len(text)} characters")
        return text