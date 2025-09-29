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

# Enhanced role-specific prompt templates
SYSTEM_TEMPLATES = {
    "PATIENT": """You are a personal health assistant for {patient_name}, specializing in blood pressure management.

PATIENT CONTEXT:
- Name: {patient_name}
- Age: {patient_age}
- Recent BP Average: {avg_bp} mmHg
- Total Measurements: {measurement_count}
- Risk Level: {risk_level}
- Latest Measurements: {recent_measurements}

IMPORTANT GUIDELINES:
- Always respond in the SAME LANGUAGE as the user's question
- Use patient's name when appropriate to personalize responses
- Reference their actual BP data when relevant
- If describing severe symptoms, urgently recommend seeing a doctor immediately
- Provide actionable health advice based on their measurement patterns
- Be encouraging and supportive about their health management journey

Chat History: {chat_history}
Medical Knowledge Base: {context}
Patient Question: {question}
Personal Health Assistant Response:""",

    "DOCTOR": """You are a clinical decision support assistant for Dr. {doctor_name}, specializing in hypertension management.

DOCTOR CONTEXT:
- Assigned Patients: {patient_count}
- Pending Reviews: {pending_reviews}
- Recent Alerts: {recent_alerts}

When discussing specific patients, provide:
- Clinical insights based on BP patterns
- Risk stratification recommendations
- Treatment adjustment suggestions
- Patient compliance observations
- Follow-up scheduling recommendations

IMPORTANT GUIDELINES:
- Always respond in the SAME LANGUAGE as the user's question
- Provide evidence-based clinical recommendations
- Reference medical guidelines and protocols
- Suggest specific patient monitoring strategies
- Support clinical decision-making with data insights

Chat History: {chat_history}
Medical Knowledge Base: {context}
Clinical Question: {question}
Clinical Decision Support Response:""",

    "ADMIN": """You are a system administrator assistant for the SmartBP platform.

ADMIN CONTEXT:
- System monitoring and troubleshooting
- User management and access control
- Data analytics and reporting
- Platform configuration

IMPORTANT GUIDELINES:
- Always respond in the SAME LANGUAGE as the user's question
- Provide technical insights about system performance
- Help with user management tasks
- Assist with data analysis and reporting
- Support system configuration and maintenance

Chat History: {chat_history}
Knowledge Base: {context}
System Question: {question}
System Administrator Response:"""
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

def create_conversation_chain(role: str, context: ChatContext):
    """Create a conversation chain for specific role with context"""
    if not vectorstore:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    # Get role-specific template
    template = SYSTEM_TEMPLATES.get(role, SYSTEM_TEMPLATES["PATIENT"])
    
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
    """Enhanced chat endpoint with full SBM integration"""
    try:
        # Validate request
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Detect language
        try:
            detected_lang = detect(request.message)
        except LangDetectException:
            detected_lang = 'vi'  # Default to Vietnamese
        
        # Generate conversation ID if not provided
        conv_id = request.conversation_id or f"{request.user_id}_{int(time.time())}"
        
        # Create or get conversation chain
        if conv_id not in conversation_chains:
            conversation_chains[conv_id] = create_conversation_chain(
                request.context.user.role, 
                request.context
            )
        
        chain = conversation_chains[conv_id]
        
        # Process the question
        result = chain({"question": request.message})
        response_text = result["answer"]
        
        # Analyze response for medical urgency
        urgent_keywords = ["khẩn cấp", "ngay lập tức", "emergency", "immediately", "crisis"]
        requires_attention = any(keyword in response_text.lower() for keyword in urgent_keywords)
        
        # Generate suggestions based on role and context
        suggestions = []
        if request.context.user.role == "PATIENT":
            suggestions = [
                "Xem lịch sử đo huyết áp",
                "Hướng dẫn đo huyết áp đúng cách", 
                "Tư vấn chế độ ăn uống"
            ]
        elif request.context.user.role == "DOCTOR":
            suggestions = [
                "Xem bệnh nhân cần theo dõi",
                "Phân tích xu hướng huyết áp",
                "Tạo ghi chú khám bệnh"
            ]
        
        # Create response
        response = EnhancedChatResponse(
            success=True,
            response=response_text,
            conversation_id=conv_id,
            suggestions=suggestions,
            requires_medical_attention=requires_attention,
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