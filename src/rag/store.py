import logging
import os
import shutil
from pathlib import Path

import chromadb

from .config import CHROMA_DIR, UPLOAD_DIR, CHUNK_SIZE, TOP_K
from .embedding import _get_embedding_fn
from .parsing import parse_pdf, parse_docx
from .chunking import chunk_text, chunk_by_sections

logger = logging.getLogger(__name__)


def _get_chroma() -> chromadb.PersistentClient:
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR)


def _get_collection(project_id: str):
    client = _get_chroma()
    name = f"rag_project_{project_id}"
    return client.get_or_create_collection(name=name, embedding_function=_get_embedding_fn())


# ---- indexing ----

def index_document(project_id: str, file_path: str, filename: str) -> int:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        text = parse_pdf(file_path)
        chunks_with_meta = chunk_by_sections(text, max_chunk_size=CHUNK_SIZE)
        chunks = [c["content"] for c in chunks_with_meta]
        metadatas = [
            {"filename": filename, "chunk_index": i,
             "page": c["page"], "paragraph": c["paragraph"],
             "section": c["section"]}
            for i, c in enumerate(chunks_with_meta)
        ]
    elif ext in (".docx", ".doc"):
        text = parse_docx(file_path)
        chunks = chunk_text(text)
        metadatas = [{"filename": filename, "chunk_index": i, "page": 1, "paragraph": i + 1, "section": ""} for i in range(len(chunks))]
    else:
        raise ValueError(f"不支持的文件类型: {ext}")

    if not text.strip():
        raise ValueError("文件中未提取到文本内容")

    if not chunks:
        raise ValueError("无法对文档进行分块")

    collection = _get_collection(project_id)

    ids = [f"{filename}_{i}" for i in range(len(chunks))]

    collection.add(ids=ids, documents=chunks, metadatas=metadatas)

    return len(chunks)


# ---- retrieval ----

def search_chunks(project_id: str, query: str, k: int = TOP_K) -> list[dict]:
    try:
        collection = _get_collection(project_id)
        if collection.count() == 0:
            logger.info("RAG 检索: project_id=%s 集合为空，跳过检索", project_id)
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
                    "page": meta.get("page", 1),
                    "paragraph": meta.get("paragraph", 0),
                    "section": meta.get("section", ""),
                })
        logger.debug("RAG 检索: project_id=%s 命中 %d 个片段", project_id, len(docs))
        return docs
    except Exception:
        logger.exception("RAG 检索失败: project_id=%s, query=%s", project_id, query[:100])
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
