#!/usr/bin/env python3
"""
Enhanced SmartBP Chatbot API with SBM Integration
Supports role-based conversations with patient data context
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
import logging
import warnings
import shutil
import uuid
import time
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Literal
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

# Suppress noisy pypdf warnings
logging.getLogger("pypdf._reader").setLevel(logging.ERROR)

# Note: Memory and chains moved in LangChain 1.x
# We'll use simpler LLM + retriever flow instead of ConversationalRetrievalChain

# Import Ollama components (required)
from langchain_ollama import OllamaEmbeddings, ChatOllama
import ollama

# Import HuggingFace embeddings for RAG
from langchain_huggingface import HuggingFaceEmbeddings
import sentence_transformers
from langdetect import detect, LangDetectException
from dotenv import load_dotenv
from fastapi.responses import JSONResponse

# Load environment variables
load_dotenv()

app = FastAPI(title="SmartBP Chatbot API", version="2.0.0")

# CORS configuration
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for RAG components
vectorstore = None
conversation_chains = {}
embeddings_model = None  # Cache embeddings to reuse

# Lifespan context manager (FastAPI 0.93+)
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler for startup and shutdown"""
    # Startup
    logging.info("🚀 Starting SmartBP Chatbot API...")
    logging.info("📚 Initializing knowledge base...")
    
    # Delete old incompatible vectorstore if exists
    old_db_path = "./db"
    if os.path.exists(old_db_path) and os.path.isdir(old_db_path):
        try:
            # Check if it's the old 768-dim vectorstore
            chroma_meta = os.path.join(old_db_path, "chroma.sqlite3")
            if os.path.exists(chroma_meta):
                logging.info(f"🗑️  Removing old incompatible vectorstore: {old_db_path}")
                shutil.rmtree(old_db_path)
        except Exception as e:
            logging.warning(f"Could not remove old db: {e}")
    
    # Initialize once
    initialize_vectorstore()
    logging.info("✅ Startup completed")
    
    yield  # App runs here
    
    # Shutdown
    logging.info("🛑 Shutting down...")
    conversation_chains.clear()
    
app = FastAPI(
    title="SmartBP Chatbot API", 
    version="2.0.0",
    lifespan=lifespan
)

# Remove old on_event decorators
# @app.on_event("startup")  # DELETED
# @app.on_event("shutdown")  # DELETED

DATA_PATH = os.getenv("DATA_DIR", "./data")
MODEL_NAME = os.getenv("LLM_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Use consistent 384-dim embeddings to match vectorstore
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
DB_PATH = os.getenv("PERSIST_DIR", "./db/chroma_db_384")
EMBEDDING_DIM = 384  # MiniLM produces 384-dim embeddings

# Fallback configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_MOCK_RESPONSES = os.getenv("USE_MOCK_RESPONSES", "false").lower() == "true"

# Pydantic Models
class UserContext(BaseModel):
    id: str
    name: Optional[str] = None
    email: Optional[str] = None
    role: Literal["PATIENT", "DOCTOR", "ADMIN"]
    dateOfBirth: Optional[str] = None
    gender: Optional[Literal["MALE", "FEMALE", "OTHER"]] = None
    phone: Optional[str] = None

class MeasurementData(BaseModel):
    id: str
    sys: int
    dia: int
    pulse: int
    method: Literal["BLUETOOTH", "MANUAL"]
    takenAt: str
    trend: Optional[Dict[str, Any]] = None

class PatientSummary(BaseModel):
    latest_measurements: List[MeasurementData] = []
    measurement_count: int = 0
    avg_sys: float = 0
    avg_dia: float = 0
    risk_assessment: str = "Unknown"
    recent_notes: Optional[List[str]] = []

class DoctorContext(BaseModel):
    assigned_patients_count: int = 0
    recent_alerts: Optional[List[str]] = []
    pending_reviews: Optional[int] = 0

class SessionMetadata(BaseModel):
    device_info: Optional[str] = None
    location: Optional[str] = None

class ChatContext(BaseModel):
    user: UserContext
    role_specific_data: Optional[Dict[str, Any]] = None
    timestamp: str
    session_metadata: Optional[SessionMetadata] = None

class EnhancedChatRequest(BaseModel):
    message: str
    user_id: str
    conversation_id: Optional[str] = None
    context: ChatContext
    language: Optional[Literal["en", "vi", "auto"]] = "auto"  # New language parameter

class DataInsights(BaseModel):
    mentioned_measurements: bool = False
    health_recommendations: List[str] = []
    follow_up_actions: List[str] = []

class EnhancedChatResponse(BaseModel):
    success: bool
    response: str
    conversation_id: str
    suggestions: Optional[List[str]] = []
    requires_medical_attention: bool = False
    data_insights: Optional[DataInsights] = None
    detected_language: Optional[str] = None  # New field for detected language

# Language detection function
def detect_language(text: str) -> str:
    """Detect language from input text"""
    try:
        detected = detect(text)
        return "en" if detected == "en" else "vi"
    except LangDetectException:
        # Default to Vietnamese if detection fails
        return "vi"

# Unified intelligent prompt template that adapts to user's language
UNIFIED_TEMPLATE = """You are SmartBP, an intelligent health assistant specializing in blood pressure management and healthcare.

CRITICAL LANGUAGE RULE: 
- If the user asks in Vietnamese (contains Vietnamese characters like á, à, ạ, ể, ở, ủ, etc.), respond COMPLETELY in Vietnamese
- If the user asks in English, respond COMPLETELY in English  
- NEVER mix languages in a single response
- When in doubt, analyze the user's question for Vietnamese words like "huyết áp", "sức khỏe", "bác sĩ" and respond in Vietnamese

USER CONTEXT:
- Name: {patient_name}
- Age: {patient_age} 
- Role: {role}
- Recent BP average: {avg_bp} mmHg
- Total measurements: {measurement_count}
- Risk level: {risk_level}
- Latest readings: {recent_measurements}

CORE GUIDELINES:
- ALWAYS match the user's language naturally
- Use their actual health data when relevant
- For serious symptoms, urgently recommend seeing a doctor
- Provide personalized advice based on their BP trends
- Be supportive and encouraging
- Draw from medical knowledge base when needed

Medical Knowledge Context: {context}
Chat History: {chat_history}

User's Question: {question}

Response (in user's language):"""

# Global variables for RAG components
vectorstore = None
conversation_chains = {}

def load_all_documents():
    """Load all document files from data directory"""
    documents = []
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        logging.warning(f"Data directory '{data_dir}' not found")
        return documents
    
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        
        try:
            if filename.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
                docs = loader.load()
                # Add metadata
                for doc in docs:
                    doc.metadata['source'] = filename
                    doc.metadata['type'] = 'text'
                documents.extend(docs)
                logging.info(f"📄 Loaded text file: {filename}")
                
            elif filename.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                # Add metadata  
                for doc in docs:
                    doc.metadata['source'] = filename
                    doc.metadata['type'] = 'pdf'
                documents.extend(docs)
                logging.info(f"📄 Loaded PDF file: {filename}")
                
        except Exception as e:
            logging.error(f"❌ Error loading {filename}: {e}")
            continue
    
    logging.info(f"📚 Total documents loaded: {len(documents)}")
    return documents

def initialize_vectorstore():
    """Initialize vectorstore with all documents - only called once at startup"""
    global vectorstore, embeddings_model
    
    # Early return if already initialized
    if vectorstore is not None:
        logging.info("✅ Vectorstore already initialized, skipping...")
        return vectorstore
    
    try:
        # Load all documents
        documents = load_all_documents()
        
        if not documents:
            logging.warning("⚠️ No documents loaded, using fallback responses only")
            return None
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        splits = text_splitter.split_documents(documents)
        logging.info(f"📄 Document chunks created: {len(splits)}")
        
        # Initialize embeddings - use 384-dim MiniLM for consistency
        embeddings_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'}
        )
        # Verify dimension
        test_embed = embeddings_model.embed_query("test")
        actual_dim = len(test_embed)
        logging.info(f"✅ Using HuggingFace embeddings - Dimension: {actual_dim}")
        
        if actual_dim != EMBEDDING_DIM:
            logging.warning(f"⚠️ Embedding dimension mismatch: expected {EMBEDDING_DIM}, got {actual_dim}")
        
        # Create or load vectorstore
        try:
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings_model,
                persist_directory=DB_PATH
            )
            logging.info(f"✅ Vectorstore initialized successfully at {DB_PATH}")
        except Exception as e:
            logging.error(f"❌ Vectorstore creation failed: {e}")
            # Try with fallback
            logging.info("💡 Attempting fallback...")
            vectorstore = None
        
        return vectorstore
        
    except Exception as e:
        logging.error(f"❌ Failed to initialize vectorstore: {e}")
        import traceback
        traceback.print_exc()
        return None

class MockEmbeddings:
    """Mock embeddings for when no embedding service is available"""
    
    def embed_documents(self, texts):
        """Return mock embeddings for documents - 384-dim to match MiniLM"""
        return [[0.1] * EMBEDDING_DIM for _ in texts]
    
    def embed_query(self, text):
        """Return mock embedding for query - 384-dim to match MiniLM"""
        return [0.1] * EMBEDDING_DIM

def initialize_rag_system():
    """Initialize the RAG system with medical documents"""
    global vectorstore
    
    embeddings = None
    
    # Try different embedding services in priority order
    try:
        # 1. Try HuggingFace/SentenceTransformers (best for Vietnamese)
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    model_kwargs={'device': 'cpu'}  # Ensure CPU usage for stability
                )
                logging.info("✅ Using HuggingFace embeddings (best for Vietnamese)")
            except Exception as e:
                logging.warning(f"HuggingFace embeddings failed: {e}")
        
        # 2. Try Ollama embeddings
        if not embeddings and OLLAMA_AVAILABLE:
            try:
                ollama_client = ollama.Client()
                ollama_client.list()  # Test connection
                embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
                logging.info("✅ Using Ollama embeddings")
            except Exception as e:
                logging.warning(f"Ollama embeddings failed: {e}")
        
        # 3. Try OpenAI embeddings
        if not embeddings and OPENAI_AVAILABLE and OPENAI_API_KEY:
            try:
                embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
                logging.info("✅ Using OpenAI embeddings")
            except Exception as e:
                logging.warning(f"OpenAI embeddings failed: {e}")
        
        # 4. Final fallback to mock embeddings
        if not embeddings:
            embeddings = MockEmbeddings()
            logging.info("✅ Using mock embeddings (fallback mode - limited RAG)")
            
    except Exception as e:
        embeddings = MockEmbeddings()
        logging.error(f"❌ All embedding services failed, using mock: {e}")
    
    try:
        # Load and process documents
        documents = []
        if os.path.exists(DATA_PATH):
            for filename in os.listdir(DATA_PATH):
                file_path = os.path.join(DATA_PATH, filename)
                try:
                    if filename.endswith('.txt'):
                        loader = TextLoader(file_path, encoding='utf-8')
                        documents.extend(loader.load())
                        logging.info(f"📄 Loaded {filename}")
                    elif filename.endswith('.pdf'):
                        loader = PyPDFLoader(file_path)
                        documents.extend(loader.load())
                        logging.info(f"📄 Loaded {filename}")
                except Exception as e:
                    logging.warning(f"Failed to load {filename}: {e}")
        else:
            logging.warning(f"Data directory {DATA_PATH} not found")
        
        if not documents:
            logging.warning("No documents loaded - RAG will use fallback context")
            
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents) if documents else []
        
        # Create vector store (only if we have real embeddings and documents)
        if texts and not isinstance(embeddings, MockEmbeddings):
            try:
                vectorstore = Chroma.from_documents(
                    documents=texts,
                    embedding=embeddings,
                    persist_directory=DB_PATH
                )
                logging.info(f"✅ RAG system initialized with {len(texts)} document chunks")
            except Exception as e:
                logging.warning(f"Vector store creation failed: {e} - using fallback")
                vectorstore = None
        else:
            logging.info("📚 RAG system initialized in fallback mode")
            vectorstore = None
        
        return True
        
    except Exception as e:
        logging.error(f"❌ Failed to initialize RAG system: {e}")
        return False

def get_fallback_medical_context(question: str) -> str:
    """Provide medical context when RAG system is not available"""
    question_lower = question.lower()
    
    if any(word in question_lower for word in ["huyết áp", "blood pressure", "bp", "cao", "thấp"]):
        return """
Thông tin cơ bản về huyết áp:
- Huyết áp bình thường: < 120/80 mmHg
- Tiền tăng huyết áp: 120-129/<80 mmHg  
- Tăng huyết áp độ 1: 130-139/80-89 mmHg
- Tăng huyết áp độ 2: ≥140/90 mmHg
- Khủng hoảng tăng huyết áp: ≥180/120 mmHg

Yếu tố nguy cơ: tuổi tác, di truyền, béo phì, thiếu vận động, stress, ăn mặn.
Biến chứng: đột quỵ, nhồi máu cơ tim, suy tim, suy thận.
Theo dõi: đo huyết áp thường xuyên, uống thuốc đều đặn.
"""
    
    elif any(word in question_lower for word in ["ăn", "diet", "dinh dưỡng", "thức ăn", "muối"]):
        return """
Chế độ ăn cho người tăng huyết áp (chế độ DASH):
- Giảm muối: <2.3g natri/ngày (1 thìa cà phê)
- Tăng rau xanh, trái cây: 4-5 phần/ngày
- Chọn ngũ cốc nguyên hạt
- Protein từ cá, gà, đậu, hạt
- Sữa ít béo: 2-3 ly/ngày
- Hạn chế: đồ chiên, đồ ngọt, đồ đóng hộp
- Uống đủ nước: 1.5-2L/ngày
"""
    
    elif any(word in question_lower for word in ["tập", "exercise", "vận động", "thể dục"]):
        return """
Vận động cho người tăng huyết áp:
- Aerobic: 150 phút/tuần cường độ vừa (đi bộ nhanh, bơi lội, đạp xe)
- Tập tạ: 2-3 lần/tuần, 8-12 động tác, mỗi động tác 8-12 lần
- Khởi động: 5-10 phút
- Thư giãn: 5-10 phút
- Tránh: vận động quá sức, nhịn thở khi tập tạ
- Theo dõi: đo huyết áp trước và sau tập
"""
    
    elif any(word in question_lower for word in ["thuốc", "medication", "điều trị"]):
        return """
Điều trị tăng huyết áp:
- Thuốc chẹn ACE: lisinopril, enalapril
- Thuốc chẹn thụ thể angiotensin: losartan, valsartan
- Thuốc lợi tiểu: hydrochlorothiazide
- Thuốc chẹn kênh canxi: amlodipine, nifedipine
- Thuốc chẹn beta: metoprolol, atenolol

Lưu ý: Uống thuốc đúng giờ, không tự ý ngừng thuốc, theo dõi tác dụng phụ.
"""
    
    return "Kiến thức y tế cơ bản về quản lý sức khỏe và tăng huyết áp."

def generate_enhanced_mock_response(message: str, role: str, language: str, medical_context: str, context: 'ChatContext') -> str:
    """Generate enhanced mock responses with user context and medical knowledge"""
    message_lower = message.lower()
    user_name = context.user.name or "bạn"
    
    # Enhanced language detection
    if language == "auto":
        language = detect_language(message)
    
    # Improved Vietnamese detection with more comprehensive rules
    vietnamese_chars = any(char in message for char in 'àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ')
    
    # Expanded Vietnamese keywords
    vietnamese_keywords = ['huyết', 'áp', 'của', 'tôi', 'như', 'thế', 'nào', 'gì', 'làm', 'sao', 'cách', 'xử', 'dụng', 'ứng', 'dng', 'bác', 'sĩ', 'bệnh', 'nhân', 'là', 'và', 'có', 'được', 'không', 'chào', 'xin', 'về', 'sức', 'khỏe', 'đo', 'kết', 'quả']
    vietnamese_words = sum(1 for word in vietnamese_keywords if word in message_lower)
    
    # English keywords  
    english_keywords = ['blood', 'pressure', 'how', 'what', 'doctor', 'patient', 'health', 'measurement', 'is', 'are', 'the', 'and', 'or', 'hello', 'hi', 'about', 'system', 'help', 'can', 'you']
    english_words = sum(1 for word in english_keywords if word in message_lower)
    
    # Smart detection: Vietnamese chars = definite Vietnamese
    if vietnamese_chars:
        response_language = "vi"
    # No Vietnamese chars: compare word counts
    elif vietnamese_words > english_words:
        response_language = "vi"
    elif english_words > vietnamese_words:
        response_language = "en" 
    # Equal or no keywords: check common patterns
    else:
        # Common English greetings and short phrases
        english_patterns = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'how are you', 'thank you', 'thanks']
        if any(pattern in message_lower for pattern in english_patterns):
            response_language = "en"
        else:
            response_language = "vi"  # Default to Vietnamese
        
    # Override: If user asks about Vietnamese terms, always respond in Vietnamese
    vietnamese_terms = ['huyết áp', 'sức khỏe', 'bác sĩ', 'bệnh nhân', 'cách sử dụng', 'hướng dẫn']
    if any(term in message_lower for term in vietnamese_terms):
        response_language = "vi"
    
    # Blood pressure specific responses
    if any(word in message_lower for word in ["huyết áp", "blood pressure", "bp", "đo"]):
        if response_language == "vi":
            if role == "PATIENT":
                return f"Chào {user_name}! Dựa trên thông tin y tế:\n\n{medical_context}\n\nTôi khuyên bạn nên:\n- Đo huyết áp đều đặn cùng giờ mỗi ngày\n- Ghi lại kết quả để theo dõi xu hướng\n- Tuân thủ chế độ ăn DASH\n- Tập thể dục đều đặn\n\nNếu huyết áp >180/120 mmHg, hãy đến bệnh viện ngay lập tức."
            else:  # DOCTOR/ADMIN
                return f"Thông tin lâm sàng cho bác sĩ {user_name}:\n\n{medical_context}\n\nKhuyến nghị:\n- Đánh giá nguy cơ tim mạch toàn diện\n- Xem xét điều chỉnh thuốc nếu cần\n- Giáo dục bệnh nhân về lối sống\n- Theo dõi tuân thủ điều trị"
        else:  # English
            if role == "PATIENT":
                return f"Hello {user_name}! Based on medical information:\n\n{medical_context}\n\nI recommend:\n- Monitor BP regularly at same time daily\n- Record results to track trends\n- Follow DASH diet\n- Exercise regularly\n\nIf BP >180/120 mmHg, seek emergency care immediately."
            else:
                return f"Clinical information for Dr. {user_name}:\n\n{medical_context}\n\nRecommendations:\n- Assess comprehensive cardiovascular risk\n- Consider medication adjustment if needed\n- Patient lifestyle education\n- Monitor treatment adherence"
    
    # Diet and nutrition responses
    elif any(word in message_lower for word in ["ăn", "diet", "nutrition", "dinh dưỡng"]):
        if response_language == "vi":
            return f"Chế độ dinh dưỡng cho {user_name}:\n\n{medical_context}\n\nLời khuyên thực tế:\n- Nấu ăn tại nhà để kiểm soát muối\n- Đọc nhãn thực phẩm\n- Thay thế muối bằng thảo mộc, gia vị\n- Ăn nhiều bữa nhỏ trong ngày"
        else:
            return f"Nutrition guidance for {user_name}:\n\n{medical_context}\n\nPractical tips:\n- Cook at home to control sodium\n- Read food labels\n- Use herbs and spices instead of salt\n- Eat smaller, frequent meals"
    
    # Exercise responses  
    elif any(word in message_lower for word in ["tập", "exercise", "vận động"]):
        if response_language == "vi":
            return f"Hướng dẫn tập luyện cho {user_name}:\n\n{medical_context}\n\nBắt đầu từ từ:\n- Tuần 1-2: đi bộ 15-20 phút/ngày\n- Tuần 3-4: tăng lên 30 phút\n- Luôn đo huyết áp trước và sau tập"
        else:
            return f"Exercise guidance for {user_name}:\n\n{medical_context}\n\nStart gradually:\n- Week 1-2: walk 15-20 minutes daily\n- Week 3-4: increase to 30 minutes\n- Always check BP before and after exercise"
    
    # General health responses
    else:
        if response_language == "vi":
            return f"Xin chào {user_name}! Tôi là trợ lý sức khỏe SmartBP. {medical_context}\n\nTôi có thể giúp bạn về:\n- Quản lý huyết áp\n- Chế độ ăn uống\n- Tập thể dục\n- Thuốc và điều trị\n\nBạn có câu hỏi gì về sức khỏe không?"
        else:
            return f"Hello {user_name}! I'm your SmartBP health assistant. {medical_context}\n\nI can help you with:\n- Blood pressure management\n- Diet and nutrition\n- Exercise guidance\n- Medications and treatment\n\nWhat health questions do you have?"

def generate_mock_response(message: str, role: str, language: str = "vi") -> str:
    """Generate mock responses when RAG system is not available"""
    message_lower = message.lower()
    
    # Blood pressure related responses
    bp_keywords = ["huyết áp", "blood pressure", "bp", "đo", "measurement", "mmhg"]
    diet_keywords = ["ăn", "thức ăn", "diet", "food", "nutrition", "dinh dưỡng"]
    exercise_keywords = ["tập thể dục", "exercise", "workout", "thể dục", "vận động"]
    
    if language == "vi":
        if any(keyword in message_lower for keyword in bp_keywords):
            if role == "PATIENT":
                return "Tôi hiểu bạn quan tâm về huyết áp. Để đo huyết áp chính xác, bạn nên ngồi thẳng, thư giãn 5 phút trước khi đo. Huyết áp bình thường dưới 120/80 mmHg. Bạn có thể sử dụng tính năng đo huyết áp trong ứng dụng SmartBP để theo dõi."
            elif role == "DOCTOR":
                return "Về quản lý huyết áp bệnh nhân, tôi khuyến nghị theo dõi thường xuyên và phân tích xu hướng. Bạn có thể xem dữ liệu đo từ hệ thống SmartBP để đánh giá tình trạng bệnh nhân."
        elif any(keyword in message_lower for keyword in diet_keywords):
            return "Chế độ ăn DASH (Dietary Approaches to Stop Hypertension) được khuyến nghị cho người có huyết áp cao: Tăng rau củ, hoa quả, giảm muối, hạn chế thức ăn chế biến sẵn."
        elif any(keyword in message_lower for keyword in exercise_keywords):
            return "Tập thể dục đều đặn giúp giảm huyết áp hiệu quả. Nên tập ít nhất 30 phút/ngày, 5 ngày/tuần. Các bài tập tim mạch nhẹ như đi bộ nhanh, bơi lội rất tốt."
        else:
            return "Xin chào! Tôi là trợ lý AI của SmartBP. Hiện tại hệ thống đang trong chế độ cơ bản. Bạn có thể hỏi về huyết áp, chế độ ăn uống, hoặc tập thể dục. Tôi sẽ cố gắng hỗ trợ bạn tốt nhất có thể."
    else:  # English
        if any(keyword in message_lower for keyword in bp_keywords):
            if role == "PATIENT":
                return "I understand you're asking about blood pressure. For accurate measurement, sit upright and relax for 5 minutes before measuring. Normal blood pressure is below 120/80 mmHg. You can use the SmartBP app's measurement feature to track your readings."
            elif role == "DOCTOR":
                return "For patient blood pressure management, I recommend regular monitoring and trend analysis. You can review measurement data from the SmartBP system to assess patient status."
        elif any(keyword in message_lower for keyword in diet_keywords):
            return "The DASH diet (Dietary Approaches to Stop Hypertension) is recommended for high blood pressure: Increase vegetables, fruits, reduce salt, limit processed foods."
        elif any(keyword in message_lower for keyword in exercise_keywords):
            return "Regular exercise effectively helps reduce blood pressure. Aim for at least 30 minutes/day, 5 days/week. Light cardiovascular exercises like brisk walking and swimming are excellent."
        else:
            return "Hello! I'm the SmartBP AI assistant. The system is currently in basic mode. You can ask about blood pressure, diet, or exercise. I'll do my best to help you."

def initialize_llm():
    """Initialize Ollama LLM"""
    try:
        logging.info("🔵 Testing Ollama connection...")
        ollama_client = ollama.Client()
        ollama_client.list()  # Test connection
        logging.info("🔵 Creating ChatOllama instance...")
        llm = ChatOllama(model=MODEL_NAME, temperature=0.7, base_url=OLLAMA_BASE_URL)
        logging.info(f"✅ Using Ollama LLM: {MODEL_NAME}")
        return llm
    except Exception as e:
        logging.error(f"❌ Ollama LLM failed: {e}")
        logging.warning("⚠️ Make sure Ollama is running: ollama serve")
        return None

def analyze_bp_risk(measurements: List[MeasurementData]) -> str:
    """Analyze blood pressure risk based on recent measurements"""
    if not measurements:
        return "Unknown"
    
    recent = measurements[:5]  # Last 5 measurements
    avg_sys = sum(m.sys for m in recent) / len(recent)
    avg_dia = sum(m.dia for m in recent) / len(recent)
    
    if avg_sys >= 180 or avg_dia >= 110:
        return "Critical - Hypertensive Crisis"
    elif avg_sys >= 140 or avg_dia >= 90:
        return "High - Stage 2 Hypertension"
    elif avg_sys >= 130 or avg_dia >= 80:
        return "Elevated - Stage 1 Hypertension"
    elif avg_sys >= 120:
        return "Elevated - Prehypertension"
    else:
        return "Normal"

def format_patient_context(context: ChatContext) -> Dict[str, str]:
    """Format patient context for prompt template"""
    user = context.user
    patient_data = context.role_specific_data or {}
    
    # Calculate age if date of birth provided
    age = "Unknown"
    if user.dateOfBirth:
        try:
            birth_date = datetime.fromisoformat(user.dateOfBirth.replace('Z', '+00:00'))
            age = str(datetime.now().year - birth_date.year)
        except:
            pass
    
    # Format measurements
    measurements = patient_data.get('latest_measurements', [])
    recent_measurements = ""
    if measurements:
        recent_measurements = f"Recent: {measurements[0]['sys']}/{measurements[0]['dia']} mmHg"
    
    return {
        "patient_name": user.name or "Patient",
        "patient_age": age,
        "avg_bp": f"{patient_data.get('avg_sys', 0):.0f}/{patient_data.get('avg_dia', 0):.0f}",
        "measurement_count": str(patient_data.get('measurement_count', 0)),
        "risk_level": patient_data.get('risk_assessment', 'Unknown'),
        "recent_measurements": recent_measurements
    }

def create_conversation_chain(role: str, context: ChatContext, language: str = "vi"):
    """Create a conversation chain for specific role with context and language"""
    global vectorstore
    
    # If vectorstore is None, try to initialize RAG system
    if not vectorstore:
        logging.warning("⚠️ Vectorstore not initialized, attempting to initialize RAG system...")
        try:
            initialize_rag_system()
        except Exception as e:
            logging.error(f"❌ Failed to initialize RAG system: {e}")
            # Return None to trigger fallback responses
            return None
    
    # If still no vectorstore after initialization attempt, return None for fallback
    if not vectorstore:
        logging.warning("⚠️ No vectorstore available, will use fallback responses")
        return None
    
    # Use unified template that adapts to user's language
    # Format context based on role for unified template
    if role == "PATIENT":
        template_vars = format_patient_context(context)
    elif role == "DOCTOR":
        doctor_data = context.role_specific_data or {}
        template_vars = {
            "patient_name": context.user.name or "Doctor",
            "patient_age": "N/A",
            "avg_bp": "Various patients",
            "measurement_count": doctor_data.get('assigned_patients_count', 0),
            "risk_level": f"{doctor_data.get('pending_reviews', 0)} pending reviews",
            "recent_measurements": ", ".join(doctor_data.get('recent_alerts', []) or ["No recent alerts"])
        }
    else:  # ADMIN
        template_vars = {
            "patient_name": context.user.name or "Admin",
            "patient_age": "N/A",
            "avg_bp": "System wide",
            "measurement_count": "All users",
            "risk_level": "System monitoring",
            "recent_measurements": "System logs available"
        }
    
    # Create unified prompt template
    final_template = UNIFIED_TEMPLATE.format(
        patient_name=template_vars.get('patient_name', 'User'),
        patient_age=template_vars.get('patient_age', 'Unknown'),
        role=role,
        avg_bp=template_vars.get('avg_bp', 'Unknown'),
        measurement_count=template_vars.get('measurement_count', '0'),
        risk_level=template_vars.get('risk_level', 'Unknown'),
        recent_measurements=template_vars.get('recent_measurements', 'None'),
        chat_history="{chat_history}",
        context="{context}",
        question="{question}"
    )
    
    prompt = PromptTemplate(
        template=final_template,
        input_variables=["chat_history", "context", "question"]
    )
    
    # Initialize LLM with fallbacks
    llm = initialize_llm()
    
    # Return None if no LLM is available (will trigger mock responses)
    if llm is None or vectorstore is None:
        logging.warning("⚠️ No LLM or vectorstore available - conversation chain disabled")
        return None
    
    # Create a simple retriever + LLM chain (replacing deprecated ConversationalRetrievalChain)
    # Return a dict-like object that mimics the chain interface
    class SimpleRAGChain:
        def __init__(self, llm, retriever, prompt):
            self.llm = llm
            self.retriever = retriever
            self.prompt = prompt
            self.chat_history = []
        
        def invoke(self, inputs):
            """Simplified chain invocation"""
            try:
                question = inputs.get("question", "")
                chat_hist = inputs.get("chat_history", "")
                
                # Retrieve context
                docs = self.retriever.invoke(question)
                doc_text = "\n".join([doc.page_content for doc in docs])
                
                # Format prompt with retrieved docs and chat history
                formatted_prompt = self.prompt.format(
                    chat_history=chat_hist,
                    context=doc_text,
                    question=question
                )
                
                # Get LLM response
                response = self.llm.invoke(formatted_prompt)
                if hasattr(response, 'content'):
                    return {"text": response.content}
                return {"text": str(response)}
            except Exception as e:
                logging.error(f"Chain invocation error: {e}")
                return {"text": ""}
    
    chain = SimpleRAGChain(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        prompt=prompt
    )
    
    return chain

# No longer using @app.on_event - using lifespan context manager instead
# Vectorstore is now initialized in the lifespan handler
@app.post("/chat")
async def enhanced_chat_endpoint(request: EnhancedChatRequest):
    """Enhanced chat endpoint with full SBM integration and multilingual support"""
    try:
        logging.info(f"🔵 Received chat request from user: {request.user_id}")
        logging.info(f"🔵 Message: {request.message}")
        logging.info(f"🔵 Language: {request.language}")
        logging.info(f"🔵 Context user role: {request.context.user.role}")
        
        # Validate request
        if not request.message.strip():
            logging.error("❌ Empty message")
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        logging.info("🔵 Starting language detection...")
        # Determine language to use
        if request.language == "auto":
            # Auto-detect language from message
            detected_lang = detect_language(request.message)
        else:
            # Use specified language
            detected_lang = request.language or "vi"
        
        logging.info(f"🔵 Detected language: {detected_lang}")
        
        # Generate conversation ID if not provided
        conv_id = request.conversation_id or f"{request.user_id}_{int(time.time())}"
        logging.info(f"🔵 Conversation ID: {conv_id}")
        
        # Create conversation chain key with language
        chain_key = f"{conv_id}_{detected_lang}"
        logging.info(f"🔵 Chain key: {chain_key}")
        
        # Create or get conversation chain
        if chain_key not in conversation_chains:
            logging.info("🔵 Creating new conversation chain...")
            conversation_chains[chain_key] = create_conversation_chain(
                request.context.user.role, 
                request.context,
                detected_lang
            )
        else:
            logging.info("🔵 Using existing conversation chain...")
        
        chain = conversation_chains[chain_key]
        logging.info(f"🔵 Chain status: {chain is not None}")
        
        # Process the question
        if chain is None or vectorstore is None:
            logging.info("🔵 Using fallback response (no chain/vectorstore)")
            # Use enhanced fallback response when RAG system is not available
            medical_context = get_fallback_medical_context(request.message)
            response_text = generate_enhanced_mock_response(
                request.message, 
                request.context.user.role, 
                detected_lang,
                medical_context,
                request.context
            )
        else:
            try:
                logging.info("🔵 Executing chain query...")
                result = chain.invoke({"question": request.message, "chat_history": ""})
                response_text = result["text"]
                logging.info("🔵 Chain execution successful")
            except Exception as e:
                logging.error(f"❌ Chain execution failed: {e}")
                import traceback
                logging.error(f"Full traceback: {traceback.format_exc()}")
                # Enhanced fallback on error
                medical_context = get_fallback_medical_context(request.message)
                response_text = generate_enhanced_mock_response(
                    request.message, 
                    request.context.user.role, 
                    detected_lang,
                    medical_context,
                    request.context
                )
        
        logging.info("🔵 Analyzing response urgency...")
        # Analyze response for medical urgency (multilingual keywords)
        urgent_keywords = ["khẩn cấp", "ngay lập tức", "emergency", "immediately", "crisis", "urgent"]
        requires_attention = any(keyword in response_text.lower() for keyword in urgent_keywords)
        
        logging.info("🔵 Generating suggestions...")
        # Generate suggestions based on role and language
        suggestions = []
        if request.context.user.role == "PATIENT":
            if detected_lang == "vi":
                suggestions = [
                    "Xem lịch sử đo huyết áp",
                    "Hướng dẫn đo huyết áp đúng cách", 
                    "Tư vấn chế độ ăn uống"
                ]
            else:
                suggestions = [
                    "View blood pressure history",
                    "Blood pressure measurement guide",
                    "Diet and nutrition advice"
                ]
        elif request.context.user.role == "DOCTOR":
            if detected_lang == "vi":
                suggestions = [
                    "Xem bệnh nhân cần theo dõi",
                    "Phân tích xu hướng huyết áp",
                    "Tạo ghi chú khám bệnh"
                ]
            else:
                suggestions = [
                    "View patients requiring monitoring",
                    "Analyze blood pressure trends",
                    "Create clinical notes"
                ]
        
        logging.info("🔵 Creating response object...")
        # Create response
        response = EnhancedChatResponse(
            success=True,
            response=response_text,
            conversation_id=conv_id,
            suggestions=suggestions,
            requires_medical_attention=requires_attention,
            detected_language=detected_lang,
            data_insights=DataInsights(
                mentioned_measurements="huyết áp" in request.message.lower() or "blood pressure" in request.message.lower(),
                health_recommendations=[],
                follow_up_actions=[]
            )
        )
        
        logging.info("✅ Chat request completed successfully")
        return response
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logging.error(f"Chat endpoint error: {e}")
        logging.error(f"Full traceback: {error_details}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 5001))  # Use PORT from .env or default to 5001 to avoid conflicts
    uvicorn.run(app, host="0.0.0.0", port=port)

