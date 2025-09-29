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

# Configuration
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")

DATA_PATH = os.getenv("DATA_DIR", "./data")
MODEL_NAME = os.getenv("LLM_MODEL", "llama3.1:8b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
DB_PATH = os.getenv("PERSIST_DIR", "./db")

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

# Enhanced role-specific prompt templates with bilingual support
SYSTEM_TEMPLATES = {
    "PATIENT": {
        "vi": """Bạn là trợ lý sức khỏe cá nhân của {patient_name}, chuyên về quản lý huyết áp.

THÔNG TIN BỆNH NHÂN:
- Tên: {patient_name}
- Tuổi: {patient_age}
- Huyết áp trung bình gần đây: {avg_bp} mmHg
- Tổng số lần đo: {measurement_count}
- Mức độ nguy cơ: {risk_level}
- Kết quả đo gần nhất: {recent_measurements}

HƯỚNG DẪN QUAN TRỌNG:
- LUÔN trả lời bằng tiếng Việt
- Gọi tên bệnh nhân khi phù hợp để tạo sự thân thiện
- Tham khảo dữ liệu huyết áp thực tế của họ khi liên quan
- Nếu có triệu chứng nghiêm trọng, khẩn cấp khuyên đi khám bác sĩ ngay
- Đưa ra lời khuyên sức khỏe cụ thể dựa trên xu hướng đo của họ
- Động viên và hỗ trợ tích cực trong hành trình quản lý sức khỏe

Lịch sử trò chuyện: {chat_history}
Cơ sở kiến thức y tế: {context}
Câu hỏi của bệnh nhân: {question}
Trả lời của trợ lý sức khỏe (BẰNG TIẾNG VIỆT):""",
        
        "en": """You are a personal health assistant for {patient_name}, specializing in blood pressure management.

PATIENT INFORMATION:
- Name: {patient_name}
- Age: {patient_age}
- Recent average blood pressure: {avg_bp} mmHg
- Total measurements: {measurement_count}
- Risk level: {risk_level}
- Latest readings: {recent_measurements}

IMPORTANT GUIDELINES:
- ALWAYS respond in English
- Address the patient by name when appropriate to create familiarity
- Reference their actual blood pressure data when relevant
- If there are serious symptoms, urgently advise seeing a doctor immediately
- Provide specific health advice based on their measurement trends
- Encourage and provide positive support in their health management journey

Chat history: {chat_history}
Medical knowledge base: {context}
Patient's question: {question}
Health assistant response (IN ENGLISH):"""
    },

    "DOCTOR": {
        "vi": """Bạn là trợ lý hỗ trợ quyết định lâm sàng cho Bác sĩ {doctor_name}, chuyên về quản lý tăng huyết áp.

THÔNG TIN BÁC SĨ:
- Số bệnh nhân được phân công: {patient_count}
- Số ca cần xem xét: {pending_reviews}
- Cảnh báo gần đây: {recent_alerts}

Khi thảo luận về bệnh nhân cụ thể, hãy cung cấp:
- Phân tích lâm sàng dựa trên xu hướng huyết áp
- Khuyến nghị phân tầng nguy cơ
- Đề xuất điều chỉnh điều trị
- Quan sát về tuân thủ điều trị của bệnh nhân
- Khuyến nghị lịch tái khám

HƯỚNG DẪN QUAN TRỌNG:
- LUÔN trả lời bằng tiếng Việt
- Đưa ra khuyến nghị lâm sàng dựa trên bằng chứng
- Tham khảo các hướng dẫn và giao thức y tế
- Đề xuất chiến lược theo dõi bệnh nhân cụ thể
- Hỗ trợ ra quyết định lâm sàng bằng phân tích dữ liệu

Lịch sử trò chuyện: {chat_history}
Cơ sở kiến thức y tế: {context}
Câu hỏi lâm sàng: {question}
Trả lời hỗ trợ quyết định lâm sàng (BẰNG TIẾNG VIỆT):""",
        
        "en": """You are a clinical decision support assistant for Dr. {doctor_name}, specializing in hypertension management.

DOCTOR INFORMATION:
- Assigned patients count: {patient_count}
- Pending reviews: {pending_reviews}
- Recent alerts: {recent_alerts}

When discussing specific patients, please provide:
- Clinical analysis based on blood pressure trends
- Risk stratification recommendations
- Treatment adjustment suggestions
- Observations on patient treatment compliance
- Follow-up schedule recommendations

IMPORTANT GUIDELINES:
- ALWAYS respond in English
- Provide evidence-based clinical recommendations
- Reference medical guidelines and protocols
- Suggest patient-specific monitoring strategies
- Support clinical decision-making with data analysis

Chat history: {chat_history}
Medical knowledge base: {context}
Clinical question: {question}
Clinical decision support response (IN ENGLISH):"""
    },

    "ADMIN": {
        "vi": """Bạn là trợ lý quản trị hệ thống cho nền tảng SmartBP.

NGỮ CẢNH QUẢN TRỊ:
- Giám sát và khắc phục sự cố hệ thống
- Quản lý người dùng và kiểm soát truy cập
- Phân tích dữ liệu và báo cáo
- Cấu hình nền tảng

HƯỚNG DẪN QUAN TRỌNG:
- LUÔN trả lời bằng tiếng Việt
- Cung cấp thông tin kỹ thuật về hiệu suất hệ thống
- Hỗ trợ các tác vụ quản lý người dùng
- Hỗ trợ phân tích dữ liệu và báo cáo
- Hỗ trợ cấu hình và bảo trì hệ thống

Lịch sử trò chuyện: {chat_history}
Cơ sở kiến thức: {context}
Câu hỏi hệ thống: {question}
Trả lời quản trị hệ thống (BẰNG TIẾNG VIỆT):""",
        
        "en": """You are a system administration assistant for the SmartBP platform.

ADMINISTRATIVE CONTEXT:
- System monitoring and troubleshooting
- User management and access control
- Data analysis and reporting
- Platform configuration

IMPORTANT GUIDELINES:
- ALWAYS respond in English
- Provide technical information about system performance
- Support user management tasks
- Assist with data analysis and reporting
- Support system configuration and maintenance

Chat history: {chat_history}
Knowledge base: {context}
System question: {question}
System administration response (IN ENGLISH):"""
    }
}

# Global variables for RAG components
vectorstore = None
conversation_chains = {}

def initialize_rag_system():
    """Initialize the RAG system with medical documents"""
    global vectorstore
    
    try:
        # Initialize embeddings
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        
        # Load and process documents
        documents = []
        for filename in os.listdir(DATA_PATH):
            file_path = os.path.join(DATA_PATH, filename)
            if filename.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
            elif filename.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=DB_PATH
        )
        
        logging.info(f"✅ Initialized RAG system with {len(texts)} document chunks")
        return True
        
    except Exception as e:
        logging.error(f"❌ Failed to initialize RAG system: {e}")
        return False

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
    if not vectorstore:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    # Get role-specific template based on language
    template = SYSTEM_TEMPLATES.get(role, {}).get(language, SYSTEM_TEMPLATES["PATIENT"]["vi"])
    
    # Format context based on role
    if role == "PATIENT":
        template_vars = format_patient_context(context)
    elif role == "DOCTOR":
        doctor_data = context.role_specific_data or {}
        template_vars = {
            "doctor_name": context.user.name or "Doctor",
            "patient_count": str(doctor_data.get('assigned_patients_count', 0)),
            "pending_reviews": str(doctor_data.get('pending_reviews', 0)),
            "recent_alerts": ", ".join(doctor_data.get('recent_alerts', []) or [])
        }
    else:  # ADMIN
        template_vars = {}
    
    # Create prompt template with partial formatting
    if role == "PATIENT":
        final_template = template.format(
            patient_name=template_vars.get('patient_name', 'Patient'),
            patient_age=template_vars.get('patient_age', 'Unknown'),
            avg_bp=template_vars.get('avg_bp', 'Unknown'),
            measurement_count=template_vars.get('measurement_count', '0'),
            risk_level=template_vars.get('risk_level', 'Unknown'),
            recent_measurements=template_vars.get('recent_measurements', 'None'),
            chat_history="{chat_history}",
            context="{context}",
            question="{question}"
        )
    elif role == "DOCTOR":
        final_template = template.format(
            doctor_name=template_vars.get('doctor_name', 'Doctor'),
            patient_count=template_vars.get('patient_count', '0'),
            pending_reviews=template_vars.get('pending_reviews', '0'),
            recent_alerts=template_vars.get('recent_alerts', 'None'),
            chat_history="{chat_history}",
            context="{context}",
            question="{question}"
        )
    else:  # ADMIN
        final_template = template.format(
            chat_history="{chat_history}",
            context="{context}",
            question="{question}"
        )
    
    prompt = PromptTemplate(
        template=final_template,
        input_variables=["chat_history", "context", "question"]
    )
    
    # Initialize LLM
    llm = ChatOllama(model=MODEL_NAME, temperature=0.7)
    
    # Create conversation chain
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    
    return chain

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logging.info("🚀 Starting SmartBP Chatbot API...")
    
    if not os.path.exists(DATA_PATH):
        logging.error(f"❌ Data directory not found: {DATA_PATH}")
        return
    
    success = initialize_rag_system()
    if success:
        logging.info("✅ SmartBP Chatbot API is ready!")
    else:
        logging.error("❌ Failed to initialize RAG system")

@app.post("/chat", response_model=EnhancedChatResponse)
async def enhanced_chat_endpoint(request: EnhancedChatRequest):
    """Enhanced chat endpoint with full SBM integration and multilingual support"""
    try:
        # Validate request
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Determine language to use
        if request.language == "auto":
            # Auto-detect language from message
            detected_lang = detect_language(request.message)
        else:
            # Use specified language
            detected_lang = request.language or "vi"
        
        # Generate conversation ID if not provided
        conv_id = request.conversation_id or f"{request.user_id}_{int(time.time())}"
        
        # Create conversation chain key with language
        chain_key = f"{conv_id}_{detected_lang}"
        
        # Create or get conversation chain
        if chain_key not in conversation_chains:
            conversation_chains[chain_key] = create_conversation_chain(
                request.context.user.role, 
                request.context,
                detected_lang
            )
        
        chain = conversation_chains[chain_key]
        
        # Process the question
        result = chain({"question": request.message})
        response_text = result["answer"]
        
        # Analyze response for medical urgency (multilingual keywords)
        urgent_keywords = ["khẩn cấp", "ngay lập tức", "emergency", "immediately", "crisis", "urgent"]
        requires_attention = any(keyword in response_text.lower() for keyword in urgent_keywords)
        
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
        
        return response
        
    except Exception as e:
        logging.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 5000))  # Use PORT from .env or default to 5000
    uvicorn.run(app, host="0.0.0.0", port=port)