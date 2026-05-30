from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import warnings

# To suppress unnecessary LangChain warnings
warnings.filterwarnings("ignore")

print("Loading database...")

# 1. Load Embeddings and Database
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
persist_directory = "./chroma_db"
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 2. Groq LLM Setup (Make sure to enter your key)
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant", 
    temperature=0.3 
)

# 3. Simple Custom Prompt Template
prompt_template = ChatPromptTemplate.from_template("""
You are a helpful AI assistant. Use the following context from the Tata Sustainability Report to answer the question.
If you don't know the answer, just say you don't know. Do not make up information.

Context:
{context}

Question: {question}

Answer:
""")

# --- CHAT INTERFACE ---
print("\n" + "="*50)
print("🤖 Tata Sustainability AI Assistant is Ready!")
print("Type 'exit' to stop.")
print("="*50)

while True:
    user_input = input("\nYour question: ")
    
    if user_input.lower() in ['exit', 'quit']:
        print("Bye! Chatbot is shutting down.")
        break
        
    print("🤖 Thinking...")
    
    try:
        # MANUAL RAG PIPELINE (Without any 'chains' module)
        
        # Step A: Retrieve relevant chunks from the database based on the user's question
        docs = retriever.invoke(user_input)
        
        # Step B: Combine those chunks into a single text string
        context_text = "\n\n".join([doc.page_content for doc in docs])
        
        # Step C: Format the prompt and send it to Groq AI
        final_prompt = prompt_template.format_messages(context=context_text, question=user_input)
        response = llm.invoke(final_prompt)
        
        print("\n✅ Answer:\n", response.content)
    except Exception as e:
        print("\n❌ An error occurred:", e)