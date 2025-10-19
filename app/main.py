from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from app.rag_chain import create_qa_chain
import logging
import time
import uuid
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_chain import load_cache, save_cache
from datetime import datetime
from fastapi.responses import FileResponse
import os

# Cấu hình môi trường & logging
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatbot_api")


# Khởi tạo FastAPI
app = FastAPI(title="Company Chatbot API", version="1.0.0")

CACHE = load_cache()

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Schema cho request
class QuestionRequest(BaseModel):
    question: str = Field(..., example="Công ty làm việc từ mấy giờ?")
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

# Khởi tạo QA Chain
qa_chain = create_qa_chain()
logger.info(" QA chain loaded successfully")

# Endpoint hỏi đáp
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    question = request.question.strip()
    logger.info(f"Nhận câu hỏi: {question}")
    start_time = time.time()

    try:
        # Kiểm tra cache
        if question in CACHE:
            logger.info(f"Lấy câu trả lời từ cache: {question}")
            answer = CACHE[question]["answer"]
            timestamp = CACHE[question]["timestamp"]
            elapsed = round(time.time() - start_time, 2)
            return JSONResponse({
                "session_id": request.session_id,
                "question": question,
                "answer": answer,
                "response_time": elapsed,
                "cached": True,
                "timestamp": timestamp
            })

        # Nếu chưa có trong cache gọi mô hình
        logger.info("Đang xử lý bằng mô hình RAG...")

        # Sử dụng API mới của LangChain
        response = qa_chain.invoke({"query": question})
        if isinstance(response, dict):
            answer = response.get("result") or response.get("output_text") or str(response)
        else:
            answer = str(response)

        elapsed = round(time.time() - start_time, 2)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Lưu cache
        CACHE[question] = {"answer": answer, "timestamp": timestamp}
        save_cache(CACHE)

        logger.info(f"Trả lời xong trong {elapsed}s")

        return JSONResponse({
            "session_id": request.session_id,
            "question": question,
            "answer": answer,
            "response_time": elapsed,
            "cached": False,
            "timestamp": timestamp
        })

    except Exception as e:
        logger.exception("Lỗi khi xử lý câu hỏi:")
        raise HTTPException(status_code=500, detail=str(e))

    
# Kiểm tra API
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Chatbot API is running!"}

# Mount thư mục static ở đường dẫn /static
app.mount("/static", StaticFiles(directory="static"), name="static")

# Trả file index.html khi truy cập /
@app.get("/")
async def root():
    index_path = os.path.join("static", "index.html")
    return FileResponse(index_path)
