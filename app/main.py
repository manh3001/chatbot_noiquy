from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from app.rag_chain import create_qa_chain
import logging
import time
import uuid
from dotenv import load_dotenv

# ---------------------------
# 1️⃣ Cấu hình môi trường & logging
# ---------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatbot_api")

# ---------------------------
# 2️⃣ Khởi tạo FastAPI
# ---------------------------
app = FastAPI(title="Company Chatbot API", version="1.0.0")

# ---------------------------
# 3️⃣ Cấu hình CORS
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # chỉnh lại domain nếu cần
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# 4️⃣ Schema cho request
# ---------------------------
class QuestionRequest(BaseModel):
    question: str = Field(..., example="Công ty làm việc từ mấy giờ?")
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

# ---------------------------
# 5️⃣ Khởi tạo QA Chain
# ---------------------------
qa_chain = create_qa_chain()
logger.info("✅ QA chain loaded successfully")

# ---------------------------
# 6️⃣ Endpoint hỏi–đáp
# ---------------------------
@app.post("/ask")
async def ask_question(request: QuestionRequest):
    logger.info(f"❓ Nhận câu hỏi: {request.question}")
    start_time = time.time()

    try:
        answer = qa_chain.run(request.question)
        elapsed = round(time.time() - start_time, 2)
        logger.info(f"✅ Trả lời trong {elapsed}s")

        return JSONResponse({
            "session_id": request.session_id,
            "question": request.question,
            "answer": answer,
            "response_time": elapsed
        })
    except Exception as e:
        logger.exception("❌ Lỗi khi xử lý câu hỏi:")
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------
# 7️⃣ Kiểm tra API
# ---------------------------
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Chatbot API is running!"}
