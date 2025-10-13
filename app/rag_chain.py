from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
import logging
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

def create_qa_chain():
    """
    Tạo chuỗi hỏi–đáp (RetrievalQA) gồm:
      - Load tài liệu nội quy công ty
      - Chia nhỏ, tạo embedding
      - Lưu vào Chroma VectorDB
      - Khởi tạo mô hình Ollama LLM
    """

    logger.info("🚀 Bắt đầu khởi tạo RAG chain...")

    # 1️⃣ Load tài liệu nội quy
    loader = TextLoader("data/company_rules.txt", encoding="utf-8")
    documents = loader.load()

    # 2️⃣ Chia nhỏ văn bản
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(documents)
    logger.info(f"✅ Số đoạn sau khi chia: {len(split_docs)}")

    # 3️⃣ Tạo embedding
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4️⃣ Tạo / load VectorDB
    vectordb = Chroma.from_documents(
        split_docs, 
        embeddings, 
        persist_directory="./chroma_db"
    )
    vectordb.persist()

    # 5️⃣ Cấu hình mô hình Ollama
    llm = OllamaLLM(
        model=os.getenv("OLLAMA_MODEL", "llama3"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        top_k=int(os.getenv("LLM_TOP_K", "40")),
        repeat_penalty=float(os.getenv("LLM_REPEAT_PENALTY", "1.2")),
        num_ctx=int(os.getenv("LLM_NUM_CTX", "4096"))
    )

    # 6️⃣ Prompt mẫu
    template = """
    Bạn là trợ lý AI hiểu rõ nội quy công ty. Dựa trên thông tin dưới đây:
    {context}
    Hãy trả lời ngắn gọn, chính xác cho câu hỏi sau: {question}
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # 7️⃣ Tạo chuỗi hỏi–đáp
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt}
    )

    logger.info("🎯 QA chain đã sẵn sàng")
    return qa_chain
