#!/usr/bin/env python3
import subprocess
import sys
import os
import shutil


def cleanup_db():
    """Clean old database"""
    db_path = "./db"
    if os.path.exists(db_path):
        try:
            shutil.rmtree(db_path)
            print("🧹 Cleaned old database")
        except Exception as e:
            print(f"⚠️ Could not clean database: {e}")


def check_ollama():
    """Check if Ollama is running"""
    try:
        result = subprocess.run(['ollama', 'list'],
                                capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def pull_models():
    """Download required models"""
    models = ["llama3.1:8b", "nomic-embed-text"]
    for model in models:
        print(f"📥 Checking model {model}...")
        try:
            result = subprocess.run(['ollama', 'list'],
                                    capture_output=True, text=True)
            if model not in result.stdout:
                print(f"📥 Downloading {model}...")
                subprocess.run(['ollama', 'pull', model], timeout=600)
                print(f"✅ Downloaded {model}")
            else:
                print(f"✅ Model {model} ready")
        except Exception as e:
            print(f"⚠️ Error with model {model}: {e}")


def check_data_folder():
    """Check data folder"""
    data_path = "./data"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print(f"📁 Created {data_path}")

    files = [f for f in os.listdir(data_path)
             if f.endswith(('.pdf', '.txt', '.md')) and not f.startswith('.')]

    if files:
        print(f"✅ Found {len(files)} files: {', '.join(files[:3])}")
    else:
        print("⚠️ No data files found. Add PDF/TXT files to data/ folder")

    return len(files) > 0


def main():
    print("🚀 Starting Blood Pressure Chatbot API...")

    # Check Ollama
    if not check_ollama():
        print("❌ Ollama not running! Please start Ollama first.")
        print("Visit: https://ollama.ai/")
        return

    # Clean database
    cleanup_db()

    # Check data
    has_data = check_data_folder()
    if not has_data:
        print("Continue without data? (y/n): ", end="")
        if input().lower() != 'y':
            return

    # Download models
    pull_models()

    # Create directories
    os.makedirs("db", exist_ok=True)

    # Start server
    print("🌟 Starting API server at http://localhost:8000")
    print("📋 API Docs: http://localhost:8000/docs")

    try:
        subprocess.run([sys.executable, "-m", "uvicorn",
                        "chatbot_api:app", "--reload",
                        "--host", "0.0.0.0", "--port", "8000"])
    except KeyboardInterrupt:
        print("\n👋 Server stopped!")


if __name__ == "__main__":
    main()