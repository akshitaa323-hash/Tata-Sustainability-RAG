from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

# --- PREVIOUS CODE (That you have already written) ---
print("Loading PDF...")
loader = PyPDFLoader("Tata_Solutions_to_Plastic_Pollution.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
print(f"Success! Your PDF has been divided into {len(chunks)} chunks.")

# --- NEW CODE (For Embeddings and Database) ---
print("\nNow converting these chunks into Embeddings (numbers)...")

# Loading the free open-source embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Directory name to save the database
persist_directory = "./chroma_db"

print("Creating the Vector Database (ChromaDB)... This might take 1-2 minutes.")
# Saving chunks and embeddings into ChromaDB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory
)

print("Boom! 🚀 Your data has been successfully saved in ChromaDB!")