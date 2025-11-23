#!/usr/bin/env python3
"""
Debug script to test chatbot components independently
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

# Test imports
print("🔍 Testing imports...")
try:
    from enhanced_chatbot_api import (
        OLLAMA_AVAILABLE, MODEL_NAME, EMBEDDING_MODEL,
        initialize_llm, initialize_rag_system
    )
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)

# Test Ollama availability
print(f"\n🔍 Testing Ollama...")
print(f"OLLAMA_AVAILABLE: {OLLAMA_AVAILABLE}")
print(f"MODEL_NAME: {MODEL_NAME}")
print(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")

if OLLAMA_AVAILABLE:
    try:
        import ollama
        client = ollama.Client()
        models = client.list()
        print(f"✅ Ollama connected - {len(models.get('models', []))} models")
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")

# Test LLM initialization
print(f"\n🔍 Testing LLM initialization...")
try:
    llm = initialize_llm()
    if llm:
        print(f"✅ LLM initialized: {type(llm)}")
    else:
        print("⚠️ LLM is None - will use mock responses")
except Exception as e:
    print(f"❌ LLM initialization failed: {e}")

# Test embeddings and vectorstore through RAG system
print(f"\n🔍 Testing RAG system...")
try:
    # This will test both embeddings and vectorstore
    rag_result = initialize_rag_system()
    print(f"✅ RAG system initialized successfully")
except Exception as e:
    print(f"❌ RAG system initialization failed: {e}")

print(f"\n🎯 Debug test completed!")