# Blood Pressure Chatbot API

🩺 **Chatbot hỗ trợ tư vấn huyết áp thông minh** sử dụng RAG (Retrieval-Augmented Generation) với LLaMA và ChromaDB.

## ✨ Tính năng

- 🤖 **AI Chatbot** hỗ trợ song ngữ (Việt/Anh)
- 📚 **RAG System** tìm kiếm thông tin từ tài liệu y khoa
- 🔄 **Session Management** lưu trữ lịch sử hội thoại
- 🌐 **REST API** sẵn sàng tích hợp với web apps
- 🏥 **Medical Safety** phát hiện triệu chứng nghiêm trọng

## 🚀 Cài đặt

### Yêu cầu hệ thống
- Python 3.8+
- Ollama (với models: llama3.1:8b, nomic-embed-text)

### Các bước cài đặt

1. **Clone repository**
```bash
git clone https://github.com/yourusername/blood-pressure-chatbot.git
cd blood-pressure-chatbot
```

2. **Tạo virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

3. **Cài đặt dependencies**
```bash
pip install -r requirements.txt
```

4. **Khởi động Ollama và tải models**
```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

5. **Chạy chatbot**
```bash
python run_chatbot.py
```

## 📁 Cấu trúc thư mục

```
chatbot/
├── chatbot_api.py          # Core API logic
├── run_chatbot.py         # Server startup script
├── test_chatbot.py        # Test suite
├── requirements.txt       # Python dependencies
├── .env                  # Environment configuration
├── restart_chatbot.bat   # Windows restart script
├── data/                 # Medical documents
│   ├── about_bot.txt
│   ├── about_bot_en.txt
│   └── *.pdf
└── db/                   # ChromaDB storage (auto-generated)
```

## 🔧 API Endpoints

### Health Check
```bash
GET /health
```

### Chat
```bash
POST /chat
{
  "message": "Huyết áp bao nhiêu là bình thường?",
  "session_id": "user-123"
}
```

### Reset Session
```bash
DELETE /chat/{session_id}
```

## 🌐 Tích hợp với Next.js

```javascript
const response = await fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    message: 'What is normal blood pressure?',
    session_id: 'user-session-123'
  })
});

const data = await response.json();
console.log(data.answer);
```

## 🧪 Testing

Chạy test suite để kiểm tra các chức năng:

```bash
python test_chatbot.py
```

## ⚙️ Configuration

Chỉnh sửa file `.env` để cấu hình:

```env
LLM_MODEL=llama3.1:8b
EMBEDDING_MODEL=nomic-embed-text
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
API_HOST=0.0.0.0
API_PORT=8000
```

## 📊 Test Results

- ✅ **Vietnamese Support**: 100% test passed
- ✅ **English Support**: 100% test passed  
- ✅ **Session Management**: Working
- ✅ **API Health**: Healthy
- ✅ **CORS**: Configured for Next.js

## 🔒 Medical Disclaimer

⚠️ **Quan trọng**: Chatbot này chỉ cung cấp thông tin tham khảo. Không thay thế cho việc khám và tư vấn y khoa chuyên nghiệp.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

MIT License - xem file LICENSE để biết chi tiết.

## 👨‍💻 Author

Được phát triển với ❤️ bởi [Your Name]

## 🔗 Related Projects

- [Next.js Frontend](https://github.com/tientrinh2003/sbm) - Web interface
- [Ollama](https://ollama.ai/) - Local LLM runtime
