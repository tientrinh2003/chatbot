#!/usr/bin/env python3
"""
Test script cho Blood Pressure Chatbot API
Kiểm tra các chức năng chính của chatbot
"""

import requests
import time

# Cấu hình
BASE_URL = "http://localhost:8000"
TEST_SESSION = "test-session-001"

def test_api_health():
    """Kiểm tra API có hoạt động không"""
    print("🔍 Testing API health...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Status: {data['status']}")
            print(f"📊 Model: {data['model']}")
            print(f"🔗 Collection: {data['collection']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return False

def test_chat(message, expected_keywords=None):
    """Test chat functionality"""
    print(f"\n💬 Testing: '{message}'")

    try:
        payload = {
            "message": message,
            "session_id": TEST_SESSION
        }

        response = requests.post(
            f"{BASE_URL}/chat",
            json=payload,
            timeout=60  # Tăng timeout lên 60s
        )

        if response.status_code == 200:
            data = response.json()
            answer = data['answer']
            sources = data.get('sources', [])
            language = data.get('language', 'unknown')

            print(f"🤖 Answer: {answer[:100]}...")  # Giảm độ dài hiển thị
            print(f"📚 Sources: {sources}")
            print(f"🌐 Language: {language}")

            # Kiểm tra keywords nếu có
            if expected_keywords:
                found_keywords = []
                for keyword in expected_keywords:
                    if keyword.lower() in answer.lower():
                        found_keywords.append(keyword)

                if found_keywords:
                    print(f"✅ Found keywords: {found_keywords}")
                    return True
                else:
                    print(f"⚠️ No expected keywords found: {expected_keywords}")
                    return False  # Chỉ fail nếu không tìm thấy keywords
            else:
                return True  # Pass nếu không có keywords để check

        else:
            print(f"❌ Chat failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Chat error: {e}")
        return False

def test_vietnamese_questions():
    """Test với câu hỏi tiếng Việt"""
    print("\n🇻🇳 Testing Vietnamese questions...")

    questions = [
        {
            "question": "Huyết áp bao nhiêu là bình thường?",
            "keywords": ["120", "80", "bình thường", "mmHg"]
        },
        {
            "question": "Làm thế nào để đo huyết áp chính xác?",
            "keywords": ["đo", "cách", "chính xác", "máy"]
        },
        {
            "question": "Tôi bị đau đầu và chóng mặt, có sao không?",
            "keywords": ["bác sĩ", "khám", "y tế"]
        }
    ]

    success_count = 0
    for q in questions:
        if test_chat(q["question"], q["keywords"]):
            success_count += 1
        time.sleep(2)  # Nghỉ giữa các câu hỏi

    print(f"\n📊 Vietnamese test results: {success_count}/{len(questions)} passed")
    return success_count == len(questions)

def test_english_questions():
    """Test với câu hỏi tiếng Anh"""
    print("\n🇺🇸 Testing English questions...")

    questions = [
        {
            "question": "What is normal blood pressure?",
            "keywords": ["120", "80", "normal", "mmHg"]
        },
        {
            "question": "How to measure blood pressure accurately?",
            "keywords": ["measure", "accurate", "blood pressure"]
        },
        {
            "question": "I have chest pain and shortness of breath",
            "keywords": ["doctor", "medical", "immediately"]
        }
    ]

    success_count = 0
    for q in questions:
        if test_chat(q["question"], q["keywords"]):
            success_count += 1
        time.sleep(2)

    print(f"\n📊 English test results: {success_count}/{len(questions)} passed")
    return success_count == len(questions)

def test_session_management():
    """Test quản lý session"""
    print("\n🔄 Testing session management...")

    # Test reset session
    try:
        response = requests.delete(f"{BASE_URL}/chat/{TEST_SESSION}")
        if response.status_code == 200:
            print("✅ Session reset successful")
            return True
        else:
            print(f"❌ Session reset failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Session reset error: {e}")
        return False

def main():
    """Chạy tất cả tests"""
    print("🚀 Starting Blood Pressure Chatbot Tests")
    print("=" * 50)

    # Test kết nối API
    if not test_api_health():
        print("❌ API not ready. Please start the server first.")
        return

    time.sleep(2)

    # Test các chức năng
    tests_passed = 0
    total_tests = 4

    if test_vietnamese_questions():
        tests_passed += 1

    if test_english_questions():
        tests_passed += 1

    if test_session_management():
        tests_passed += 1

    # Test final chat
    if test_chat("Cảm ơn bạn!", ["cảm ơn", "chúc"]):
        tests_passed += 1

    # Kết quả tổng
    print("\n" + "=" * 50)
    print(f"🎯 TEST SUMMARY: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print("🎉 All tests PASSED! Chatbot is working perfectly!")
    elif tests_passed >= total_tests * 0.8:
        print("✅ Most tests passed. Chatbot is working well!")
    else:
        print("⚠️ Some tests failed. Please check the chatbot configuration.")

    print("\n💡 To test manually, visit: http://localhost:8000/docs")

if __name__ == "__main__":
    main()
