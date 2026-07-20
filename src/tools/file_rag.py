import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction
from openai import OpenAI
import fitz
from docx import Document

DATA_DIR = "data"
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5


class APIEmbeddingFunction(EmbeddingFunction[Documents]):
    """ChromaDB 自定义嵌入函数：调用 OpenAI 兼容 API 获取 embedding。

    通过环境变量配置：
    - EMBEDDING_MODEL: 嵌入模型名（默认 text-embedding-v4）
    - EMBEDDING_BASE_URL: API 地址（默认 DashScope 兼容端点）
    - EMBEDDING_API_KEY: API Key（默认复用 OPENAI_API_KEY）
    """

    def __init__(self):
        self._client = OpenAI(
            api_key=os.getenv("EMBEDDING_API_KEY", os.getenv("OPENAI_API_KEY", "")),
            base_url=os.getenv("EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
        self._model = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")

    def __call__(self, input: Documents) -> Embeddings:
        # 替换换行符可提升检索质量（OpenAI 建议做法）
        cleaned = [text.replace("\n", " ") for text in input]
        resp = self._client.embeddings.create(model=self._model, input=cleaned)
        return [d.embedding for d in resp.data]


def _get_chroma() -> chromadb.PersistentClient:
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR)


_embedding_fn: APIEmbeddingFunction | None = None


def _get_embedding_fn() -> APIEmbeddingFunction:
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = APIEmbeddingFunction()
    return _embedding_fn


def _get_collection(project_id: str):
    client = _get_chroma()
    name = f"rag_project_{project_id}"
    return client.get_or_create_collection(name=name, embedding_function=_get_embedding_fn())


# ---- file parsing ----

def parse_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return "\n".join(text_parts)


def parse_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join(p.text for p in doc.paragraphs)


# ---- chunking ----

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    paragraphs = text.split("\n")
    chunks = []
    current = ""

    for p in paragraphs:
        p = p.strip()
        if not p:
            continue

        if len(current) + len(p) > chunk_size:
            if current:
                chunks.append(current.strip())
                overlap_text = current[-overlap:] if len(current) > overlap else current
                current = overlap_text + "\n" + p
            else:
                for i in range(0, len(p), chunk_size - overlap):
                    chunk = p[i:i + chunk_size]
                    if chunk.strip():
                        chunks.append(chunk.strip())
                current = ""
        else:
            current += "\n" + p if current else p

    if current.strip():
        chunks.append(current.strip())

    return chunks


# ---- indexing ----

def index_document(project_id: str, file_path: str, filename: str) -> int:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        text = parse_pdf(file_path)
    elif ext in (".docx", ".doc"):
        text = parse_docx(file_path)
    else:
        raise ValueError(f"不支持的文件类型: {ext}")

    if not text.strip():
        raise ValueError("文件中未提取到文本内容")

    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("无法对文档进行分块")

    collection = _get_collection(project_id)

    ids = [f"{filename}_{i}" for i in range(len(chunks))]
    metadatas = [{"filename": filename, "chunk_index": i} for i in range(len(chunks))]

    collection.add(ids=ids, documents=chunks, metadatas=metadatas)

    return len(chunks)


# ---- retrieval ----

def search_chunks(project_id: str, query: str, k: int = TOP_K) -> list[dict]:
    try:
        collection = _get_collection(project_id)
        if collection.count() == 0:
            return []
        results = collection.query(query_texts=[query], n_results=k)
        docs = []
        if results["documents"] and results["documents"][0]:
            for i, doc_text in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                docs.append({
                    "content": doc_text,
                    "filename": meta.get("filename", "unknown"),
                    "chunk_index": meta.get("chunk_index", 0),
                })
        return docs
    except Exception:
        return []


# ---- file management ----

def get_project_files(project_id: str) -> list[dict]:
    proj_dir = os.path.join(UPLOAD_DIR, project_id)
    if not os.path.exists(proj_dir):
        return []

    files = []
    for f in os.listdir(proj_dir):
        fpath = os.path.join(proj_dir, f)
        if os.path.isfile(fpath):
            files.append({"filename": f, "size": os.path.getsize(fpath)})
    return files


def delete_project_index(project_id: str):
    try:
        client = _get_chroma()
        client.delete_collection(f"rag_project_{project_id}")
    except Exception:
        pass

    proj_dir = os.path.join(UPLOAD_DIR, project_id)
    if os.path.exists(proj_dir):
        shutil.rmtree(proj_dir)
