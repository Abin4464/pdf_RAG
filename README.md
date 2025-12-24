# pdf_RAG
#  RAG System for PDF-Based Question Answering

A **Retrieval-Augmented Generation (RAG)** system that allows users to ask natural language questions over PDF documents and receive answers grounded strictly in the document content.

This project is built **from scratch in Python**, focusing on **ETL pipelines, semantic search, and contextual retrieval**, without relying on heavy frameworks like LangChain or LlamaIndex.

---

## What This Project Does

- Extracts text from PDF documents
- Cleans and preprocesses raw text
- Splits documents into overlapping semantic chunks
- Converts text into vector embeddings
- Stores embeddings in a FAISS vector database
- Retrieves the most relevant chunks using semantic similarity
- Generates answers using retrieved context (RAG)

The system ensures **low hallucination** by grounding responses in document data.

---

##  Architecture Overview
rag_project/
│
├── data/
│ └── documents/
│ └── sample.pdf
│
├── src/
│ ├── ingestion/ # PDF loading
│ ├── preprocessing/ # Cleaning & chunking
│ ├── embeddings/ # Embedding generation
│ ├── vectorstore/ # FAISS index
│ ├── retrieval/ # Semantic search
│ ├── llm/ # LLM interaction
│ └── utils/ # Configuration
│
├── main.py # Entry point
├── requirements.txt
└── README.md


---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Sentence Transformers** – text embeddings
- **FAISS** – vector similarity search
- **PyPDF** – PDF text extraction
- **LLM API (optional)** – response generation

All components are modular and replaceable.

---

##  How to Run

###  Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

🔹 Loading document...
🔹 Cleaning text...
🔹 Chunking text...
🔹 Generating embeddings...
🔹 Building vector store...
✅ System ready. Ask your question:



