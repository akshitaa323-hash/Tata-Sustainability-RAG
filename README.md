# 🌍 Tata Sustainability AI Assistant (RAG Pipeline)

An end-to-end Retrieval-Augmented Generation (RAG) pipeline built to analyze and query the "Tata Solutions to Plastic Pollution" ESG report. This AI assistant retrieves exact context from the document and generates highly accurate, human-like responses without hallucinations.

## 🚀 Key Features
* **Custom Data Ingestion:** Automates the loading and chunking of complex PDF documents.
* **Local Vector Storage:** Uses ChromaDB for fast, secure, and local semantic search.
* **Open-Source Embeddings:** Powered by HuggingFace's `all-MiniLM-L6-v2` model for high-quality text vectorization.
* **Advanced LLM Integration:** Connected to Meta's `Llama-3.1-8b` via the Groq API for lightning-fast inference.

## 🛠️ Tech Stack
* **Language:** Python
* **Framework:** LangChain
* **Vector Database:** ChromaDB
* **Embeddings:** HuggingFace
* **LLM:** Groq (Llama-3.1)

## 🏗️ Architecture Workflow
1. **Document Loading:** Extracts text from the target PDF.
2. **Text Splitting:** Divides the text into overlapping 1000-character chunks to preserve context.
3. **Embedding:** Converts chunks into mathematical vectors.
4. **Storage:** Saves vectors into a local Chroma database.
5. **Retrieval & Generation:** User queries fetch the Top-3 most relevant chunks, passing them as context to the LLM to generate the final answer.

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/Tata-Sustainability-RAG.git](https://github.com/your-username/Tata-Sustainability-RAG.git)
   cd Tata-Sustainability-RAG