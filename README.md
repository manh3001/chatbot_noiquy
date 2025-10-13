# 🤖 Chatbot Nội Quy Công Ty (RAG + Ollama + LangChain + FastAPI)

## 🧠 Mục tiêu
Dự án xây dựng chatbot thông minh giúp trả lời câu hỏi về **nội quy công ty** dựa trên mô hình ngôn ngữ lớn (LLM).

## ⚙️ Công nghệ
- FastAPI (triển khai API)
- LangChain (RAG pipeline)
- ChromaDB (vector database)
- HuggingFace Embeddings
- Ollama (LLM LLaMA3 hoặc Mistral)
- Python dotenv, pydantic

## 🚀 Cách chạy
```bash
# 1. Clone dự án
git clone https://github.com/<tên_bạn>/chatbot_noiquy.git
cd chatbot_noiquy

# 2. Cài môi trường
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 3. Chạy Ollama server
ollama serve

# 4. Khởi động API
uvicorn app.main:app --reload
