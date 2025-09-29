#!/usr/bin/env python3
"""
Quick setup and test script for Enhanced SmartBP Chatbot Integration
Kiểm tra tính năng mới và test integration
"""

import subprocess
import sys
import os
import time
import requests
from pathlib import Path

def print_step(step, description):
    print(f"\n🔷 Step {step}: {description}")
    print("=" * 50)

def check_python_requirements():
    """Kiểm tra Python packages cần thiết"""
    required_packages = [
        'fastapi', 'uvicorn', 'pydantic', 'ollama',
        'langchain', 'langchain-community', 'langchain-ollama', 
        'chromadb', 'langdetect', 'python-dotenv'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("✅ All Python packages installed")
        return True

def check_ollama_models():
    """Kiểm tra Ollama models"""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            models = result.stdout
            required_models = ['llama3.1:8b', 'nomic-embed-text']
            missing_models = []
            
            for model in required_models:
                if model not in models:
                    missing_models.append(model)
            
            if missing_models:
                print(f"❌ Missing Ollama models: {', '.join(missing_models)}")
                print("Run: ollama pull llama3.1:8b && ollama pull nomic-embed-text")
                return False
            else:
                print("✅ All Ollama models available")
                return True
        else:
            print("❌ Ollama not running or accessible")
            return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False

def start_enhanced_chatbot():
    """Khởi động enhanced chatbot"""
    chatbot_path = Path("enhanced_chatbot_api.py")
    if not chatbot_path.exists():
        print("❌ enhanced_chatbot_api.py not found")
        return None
    
    print("🚀 Starting Enhanced Chatbot API...")
    try:
        process = subprocess.Popen([
            sys.executable, "enhanced_chatbot_api.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for startup
        time.sleep(5)
        
        # Test health check
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Enhanced Chatbot API running on port 5000")
            return process
        else:
            print("❌ Chatbot API health check failed")
            return None
    except Exception as e:
        print(f"❌ Error starting chatbot: {e}")
        return None

def test_enhanced_api():
    """Test enhanced chatbot API với sample data"""
    print("🧪 Testing Enhanced Chatbot API...")
    
    # Sample patient context
    test_request = {
        "message": "Huyết áp 140/90 của tôi có nghiêm trọng không?",
        "user_id": "test@example.com",
        "conversation_id": "test_conversation",
        "context": {
            "user": {
                "id": "test_user",
                "name": "Nguyễn Văn A",
                "email": "test@example.com",
                "role": "PATIENT",
                "dateOfBirth": "1980-01-01",
                "gender": "MALE"
            },
            "role_specific_data": {
                "latest_measurements": [
                    {
                        "id": "m1",
                        "sys": 140,
                        "dia": 90,
                        "pulse": 75,
                        "method": "MANUAL",
                        "takenAt": "2025-09-29T10:00:00Z"
                    }
                ],
                "measurement_count": 5,
                "avg_sys": 135.0,
                "avg_dia": 85.0,
                "risk_assessment": "Elevated - Stage 1 Hypertension",
                "recent_notes": []
            },
            "timestamp": "2025-09-29T10:00:00Z"
        }
    }
    
    try:
        response = requests.post(
            "http://localhost:5000/chat", 
            json=test_request,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API Response successful")
            print(f"📝 Response: {data.get('response', 'No response')[:100]}...")
            print(f"🔍 Suggestions: {data.get('suggestions', [])}")
            print(f"⚠️ Medical Attention: {data.get('requires_medical_attention', False)}")
            return True
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test request failed: {e}")
        return False

def check_nextjs_setup():
    """Kiểm tra Next.js setup"""
    sbm_path = Path("../sbm")  # Adjust path as needed
    
    if not sbm_path.exists():
        print("❌ SBM Next.js project not found")
        return False
    
    # Check if package.json exists
    package_json = sbm_path / "package.json"
    if not package_json.exists():
        print("❌ package.json not found in SBM project")
        return False
    
    # Check key files
    key_files = [
        "types/chatbot.ts",
        "components/EnhancedChatInterface.tsx", 
        "app/api/chatbot/route.ts"
    ]
    
    missing_files = []
    for file in key_files:
        if not (sbm_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing SBM files: {', '.join(missing_files)}")
        return False
    else:
        print("✅ SBM Next.js files ready")
        return True

def print_next_steps():
    """In hướng dẫn next steps"""
    print("\n🎯 NEXT STEPS:")
    print("=" * 50)
    print("1. Start Next.js development server:")
    print("   cd ../sbm && pnpm dev")
    print()
    print("2. Test the integration:")
    print("   - Login as PATIENT/DOCTOR/ADMIN")
    print("   - Navigate to chat interface")
    print("   - Test role-based responses")
    print()
    print("3. Monitor logs:")
    print("   - Python chatbot: Check terminal output")
    print("   - Next.js: Check browser console")
    print("   - API: Check network requests")
    print()
    print("4. Production deployment:")
    print("   - Configure environment variables")
    print("   - Setup Docker containers")
    print("   - Configure HTTPS & CORS")

def main():
    """Main setup and test workflow"""
    print("🚀 Enhanced SmartBP Chatbot Integration Setup")
    print("=" * 60)
    
    # Step 1: Check Python requirements
    print_step(1, "Checking Python requirements")
    if not check_python_requirements():
        print("❌ Setup failed - install missing packages first")
        return
    
    # Step 2: Check Ollama models
    print_step(2, "Checking Ollama models")
    if not check_ollama_models():
        print("⚠️ Warning: Ollama models missing - chatbot may not work")
    
    # Step 3: Check data directory
    print_step(3, "Checking data directory")
    data_path = Path("./data")
    if data_path.exists():
        files = list(data_path.glob("*"))
        print(f"✅ Data directory found with {len(files)} files")
    else:
        print("❌ Data directory not found - create ./data with medical documents")
    
    # Step 4: Start enhanced chatbot
    print_step(4, "Starting Enhanced Chatbot API")
    chatbot_process = start_enhanced_chatbot()
    if not chatbot_process:
        print("❌ Failed to start chatbot - check logs")
        return
    
    # Step 5: Test API
    print_step(5, "Testing Enhanced API")
    api_working = test_enhanced_api()
    
    # Step 6: Check Next.js setup  
    print_step(6, "Checking Next.js setup")
    nextjs_ready = check_nextjs_setup()
    
    # Summary
    print("\n📊 SETUP SUMMARY:")
    print("=" * 50)
    print(f"✅ Python packages: Ready")
    print(f"{'✅' if check_ollama_models() else '⚠️'} Ollama models: {'Ready' if check_ollama_models() else 'Missing'}")
    print(f"✅ Enhanced Chatbot: Running")
    print(f"{'✅' if api_working else '❌'} API Testing: {'Passed' if api_working else 'Failed'}")
    print(f"{'✅' if nextjs_ready else '❌'} Next.js Setup: {'Ready' if nextjs_ready else 'Missing files'}")
    
    if api_working and nextjs_ready:
        print("\n🎉 Integration setup completed successfully!")
        print_next_steps()
    else:
        print("\n⚠️ Some issues found - check logs and fix before proceeding")
    
    # Keep chatbot running
    if chatbot_process:
        try:
            print("\n💡 Chatbot running in background. Press Ctrl+C to stop.")
            chatbot_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping chatbot...")
            chatbot_process.terminate()

if __name__ == "__main__":
    main()