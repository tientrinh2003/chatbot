from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import logging
import warnings
import shutil
import uuid
import time
from datetime import datetime
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import ollama
from langdetect import detect, LangDetectException
from dotenv import load_dotenv
from fastapi.responses import JSONResponse

# Load environment variables
load_dotenv()

app = FastAPI(title="Blood Pressure Chatbot API", version="1.0.0")

# CORS configuration from .env
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")

DATA_PATH = os.getenv("DATA_DIR", "./data")
MODEL_NAME = os.getenv("LLM_MODEL", "llama3.1:8b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
DB_PATH = os.getenv("PERSIST_DIR", "./db")
VECTOR_STORE_NAME = f"huyetap-rag-{str(uuid.uuid4())[:8]}"  # Unique collection name

# Enhanced prompt template
SYSTEM_TEMPLATE = """You are a virtual health assistant specializing in blood pressure.
Use the following pieces of context and the chat history to answer the user's question.

IMPORTANT: You MUST respond in the SAME LANGUAGE as the user's question:
- If user asks in Vietnamese, respond in Vietnamese
- If user asks in English, respond in English

If you don't know the answer, just say that you don't know, don't try to make up an answer.

If the user describes severe symptoms (chest pain, shortness of breath, fainting), you MUST reply urgently in the user's language: 
- Vietnamese: "Bạn nên đi khám bác sĩ hoặc đến cơ sở y tế ngay lập tức."
- English: "You should see a doctor or go to a medical facility immediately."

Keep the answer concise and helpful.

Chat History:
{chat_history}

Context:
{context}

Question: {question}
Answer in the same language as the question:"""

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = []
    language: str = "vi"
    session_id: str = "default"

# Global variables
vector_db = None
llm = None
sessions = {}

@app.middleware("http")
async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        logging.error(f"Unhandled exception: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )

def clean_database():
    """Clean old database with better error handling"""
    if os.path.exists(DB_PATH):
        try:
            # Try to remove the directory
            shutil.rmtree(DB_PATH)
            logging.info("Cleaned old database")
        except PermissionError as e:
            # Database is in use, create unique collection name
            global VECTOR_STORE_NAME
            VECTOR_STORE_NAME = f"huyetap-rag-{str(uuid.uuid4())[:8]}"
            logging.warning(f"Database in use, using new collection: {VECTOR_STORE_NAME}")
        except Exception as e:
            logging.warning(f"Could not clean old database: {e}")
            # Generate new collection name
            VECTOR_STORE_NAME = f"huyetap-rag-{str(uuid.uuid4())[:8]}"

    # Ensure directory exists
    os.makedirs(DB_PATH, exist_ok=True)

def load_existing_vector_db():
    """Try to load existing vector database"""
    try:
        if os.path.exists(DB_PATH):
            embeddings = OllamaEmbeddings(
                model=EMBEDDING_MODEL,
                base_url="http://127.0.0.1:11434"
            )

            # Try to find existing collections
            try:
                vector_db = Chroma(
                    collection_name=VECTOR_STORE_NAME,
                    embedding_function=embeddings,
                    persist_directory=DB_PATH
                )

                # Test if database has data
                test_results = vector_db.similarity_search("test", k=1)
                if test_results:
                    logging.info(f"✅ Loaded existing vector database with data")
                    return vector_db
            except:
                pass  # Collection doesn't exist, will create new one

        return None
    except Exception as e:
        logging.warning(f"Could not load existing database: {e}")
        return None

def ingest_documents(data_path):
    """Load documents with better error handling"""
    documents = []
    if not os.path.exists(data_path):
        logging.error(f"Data path {data_path} does not exist!")
        return documents

    for file in os.listdir(data_path):
        if file.startswith('.'):
            continue

        path = os.path.join(data_path, file)

        try:
            if file.endswith((".txt", ".md")):
                loader = TextLoader(path, encoding="utf-8")
                docs = loader.load()
            elif file.endswith(".pdf"):
                loader = PyPDFLoader(path)
                docs = loader.load()
            else:
                continue

            for d in docs:
                d.metadata["source"] = file
            documents.extend(docs)
            logging.info(f"Loaded {file}: {len(docs)} documents")

        except Exception as e:
            logging.error(f"Error loading {file}: {e}")
            continue

    logging.info(f"Total loaded {len(documents)} documents.")
    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    logging.info(f"Split into {len(chunks)} chunks.")
    return chunks

def create_vector_db(chunks):
    try:
        # Pull models if needed
        try:
            ollama.pull(EMBEDDING_MODEL)
            logging.info(f"Embedding model {EMBEDDING_MODEL} ready")
        except Exception as e:
            logging.warning(f"Could not pull embedding model: {e}")

        embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url="http://127.0.0.1:11434"
        )

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=VECTOR_STORE_NAME,
            persist_directory=DB_PATH
        )

        vector_db.persist()
        logging.info(f"Vector database created: {VECTOR_STORE_NAME}")
        return vector_db

    except Exception as e:
        logging.error(f"Error creating vector database: {e}")
        raise

def create_conversational_chain(retriever, llm_model):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

    prompt = PromptTemplate(
        template=SYSTEM_TEMPLATE,
        input_variables=["chat_history", "context", "question"]
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_model,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key='answer',
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    return chain

def get_language(text):
    try:
        lang = detect(text)
        return lang if lang in ["vi", "en"] else "vi"
    except LangDetectException:
        return "vi"

@app.on_event("startup")
async def startup_event():
    global vector_db, llm

    try:
        logging.info("Initializing chatbot...")

        # Try to load existing database first
        vector_db = load_existing_vector_db()

        if not vector_db:
            # Clean and prepare database
            clean_database()

            # Load documents
            docs = ingest_documents(DATA_PATH)
            if not docs:
                logging.warning("No documents loaded. Using minimal configuration.")
                return

            # Create vector database
            chunks = split_documents(docs)
            vector_db = create_vector_db(chunks)

        # Initialize LLM
        try:
            ollama.pull(MODEL_NAME)
            logging.info(f"LLM model {MODEL_NAME} ready")
        except Exception as e:
            logging.warning(f"Could not pull LLM model: {e}")

        llm = ChatOllama(
            model=MODEL_NAME,
            base_url="http://127.0.0.1:11434"
        )

        logging.info("✅ Chatbot initialized successfully!")

    except Exception as e:
        logging.error(f"❌ Failed to initialize chatbot: {e}")

@app.get("/")
async def root():
    return {
        "message": "Blood Pressure Chatbot API",
        "status": "healthy" if llm and vector_db else "initializing",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if llm and vector_db else "degraded",
        "model": MODEL_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "collection": VECTOR_STORE_NAME,
        "data_path": DATA_PATH
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        if not llm or not vector_db:
            raise HTTPException(
                status_code=503,
                detail="Chatbot is still initializing. Please try again later."
            )

        # Create or get session
        if request.session_id not in sessions:
            retriever = vector_db.as_retriever(search_kwargs={"k": 5})
            sessions[request.session_id] = create_conversational_chain(retriever, llm)

        chain = sessions[request.session_id]

        # Process message
        result = chain.invoke({"question": request.message})
        answer = result["answer"]

        # Extract sources
        source_docs = result.get("source_documents", [])
        sources = list(dict.fromkeys(
            doc.metadata.get("source") for doc in source_docs
            if doc.metadata.get("source")
        ))[:3]

        # Detect language
        language = get_language(request.message)

        return ChatResponse(
            answer=answer,
            sources=sources,
            language=language,
            session_id=request.session_id
        )

    except Exception as e:
        logging.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing message: {str(e)}"
        )

@app.delete("/chat/{session_id}")
async def reset_conversation(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
        return {"message": f"Session {session_id} reset successfully"}
    return {"message": "Session not found"}

@app.post("/admin/reset-db")
async def reset_database():
    global vector_db, sessions
    try:
        sessions.clear()
        clean_database()
        await startup_event()
        return {"message": "Database reset successfully"}
    except Exception as e:
        logging.error(f"Database reset error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/reload-data")
async def reload_data():
    """Reload data without restarting server"""
    global vector_db, sessions
    try:
        sessions.clear()

        # Load documents
        docs = ingest_documents(DATA_PATH)
        if not docs:
            return {"message": "No documents found to reload"}

        # Create new vector database
        chunks = split_documents(docs)
        vector_db = create_vector_db(chunks)

        return {
            "message": f"Data reloaded successfully",
            "documents": len(docs),
            "chunks": len(chunks)
        }
    except Exception as e:
        logging.error(f"Reload data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")