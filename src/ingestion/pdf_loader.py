import PyPDF2
from typing import List

class PDFLoader:
    """Extracts raw text from PDF files"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def load(self) -> str:
        """Load and extract text from PDF"""
        try:
            with open(self.file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()
                
                print(f"✓ Loaded {len(pdf_reader.pages)} pages from PDF")
                return text
        
        except Exception as e:
            print(f"✗ Error loading PDF: {str(e)}")
            return ""
