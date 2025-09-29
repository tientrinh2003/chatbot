#!/usr/bin/env python3
"""
Quick test script to debug chatbot API
"""

import requests
import json

def test_health():
    """Test health endpoint"""
    try:
        response = requests.get("http://localhost:5000/health", timeout=5)
        print(f"Health check: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_chat_simple():
    """Test simple chat"""
    payload = {
        "message": "Hello",
        "user_id": "test@example.com",
        "conversation_id": "test_123",
        "context": {
            "user": {
                "id": "test_user",
                "name": "Test User",
                "email": "test@example.com", 
                "role": "PATIENT"
            },
            "timestamp": "2025-09-29T10:00:00Z"
        }
    }
    
    try:
        response = requests.post(
            "http://localhost:5000/chat",
            json=payload,
            timeout=30
        )
        print(f"Chat test: {response.status_code}")
        if response.status_code != 200:
            print(f"Error response: {response.text}")
        else:
            data = response.json()
            print(f"Success: {data.get('success', False)}")
            print(f"Response preview: {data.get('response', '')[:100]}...")
        return response.status_code == 200
    except Exception as e:
        print(f"Chat test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Enhanced Chatbot API...")
    print("=" * 50)
    
    print("\n1. Health Check:")
    health_ok = test_health()
    
    if health_ok:
        print("\n2. Simple Chat Test:")
        chat_ok = test_chat_simple()
        
        if chat_ok:
            print("\n✅ All tests passed!")
        else:
            print("\n❌ Chat test failed")
    else:
        print("\n❌ Health check failed - is API running?")