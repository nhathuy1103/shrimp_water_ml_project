import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


def load_rag_db(persist_dir: str = "vector_db"):
    """
    Tạo / load vector database từ các file PDF về nuôi tôm.
    Nếu không tìm thấy PDF => trả về None, agent sẽ fallback sang LLM thuần.
    """

    pdf_files = [
        os.path.join("data", "yeu_to_moi_truong.pdf"),
        os.path.join("data", "benh_o_tom.pdf"),
    ]

    docs = []
    for path in pdf_files:
        if os.path.exists(path):
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        else:
            print(f"[RAG] WARNING: Không tìm thấy file PDF: {path}")

    if not docs:
        print("[RAG] Không có tài liệu PDF, agent sẽ chỉ dựa vào ML + kiến thức LLM.")
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    return vectordb