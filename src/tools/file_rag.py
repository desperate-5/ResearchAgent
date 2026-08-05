import logging
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

import chromadb
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction
from openai import OpenAI
import fitz
from docx import Document
import re

DATA_DIR = "data"
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma")

CHUNK_SIZE = 300
CHUNK_OVERLAP = 60
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
        # DashScope 嵌入 API 单次请求有 batch size 上限（不同账号/模型上限不同，实测 10 条），
        # 此处设为 8 留足余量，超过会自动分批调用
        self._batch_size = 8

    def __call__(self, input: Documents) -> Embeddings:
        # 替换换行符可提升检索质量（OpenAI 建议做法）
        cleaned = [text.replace("\n", " ") for text in input]
        embeddings = []
        for i in range(0, len(cleaned), self._batch_size):
            batch = cleaned[i:i + self._batch_size]
            resp = self._client.embeddings.create(model=self._model, input=batch)
            embeddings.extend(d.embedding for d in resp.data)
        return embeddings


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
    """解析 PDF 并插入 [PAGE:N] 页码标记，供 chunk_text_with_metadata 追踪页码。"""
    doc = fitz.open(file_path)
    text_parts = []
    for i, page in enumerate(doc):
        text_parts.append(f"[PAGE:{i + 1}]\n{page.get_text()}")
    doc.close()
    return "\n".join(text_parts)


def parse_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join(p.text for p in doc.paragraphs)


# ---- chunking ----

# 章节标题检测正则。匹配中文论文中常见的章节标题格式：
#   - 编号标题: 1.1 / 1.1.1 / 2.3.4 后跟文本
#   - 中文章节: 第X章 / 第X节
#   - 中文序号: 一、 / (一) / （一）
#   - 论文固定段落: 摘要/ABSTRACT/绪论/引言/结论/参考文献/致谢/目录 等
_SECTION_HEADING = re.compile(
    r'^('
    r'\d+(?:\.\d+)+[\s　]+[^\s].{1,60}'                # 1.1 / 1.1.1 标题
    r'|第[一二三四五六七八九十\d]+[章节][\s　]*[^\s]?.{0,60}'  # 第X章/第X节
    r'|[一二三四五六七八九十]+[\s　]*[、．.][^\s].{0,60}'     # 一、/二、标题
    r'|[（(][\d一二三四五六七八九十]+[）)][^\s]?.{0,60}'    # (一)/(1) 标题
    r'|^(?:摘要|ABSTRACT|Abstract'
    r'|关键词|关键字|KEYWORDS|Keywords'
    r'|绪论|引言|前言|背景|导论'
    r'|结论|总结|结束语|展望|讨论'
    r'|致谢|鸣谢|ACKNOWLEDGMENTS?'
    r'|参考文献|REFERENCES?|Bibliography'
    r'|附录|Appendix'
    r'|目[\s　]*录|目录'
    r')[\s　]*$'
    r')',
    re.IGNORECASE,
)


def _is_section_heading(line: str) -> bool:
    """判断一行文本是否为章节标题（不含 TOC 中的点线目录行）。"""
    if len(line) > 80:
        return False
    if "....." in line or "……" in line:
        return False
    return bool(_SECTION_HEADING.match(line))


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


def chunk_by_sections(text: str, max_chunk_size: int = 500) -> list[dict]:
    """按章节标题分块，每块上限 max_chunk_size 字符。

    期望输入文本中包含 [PAGE:N] 标记（由 parse_pdf 注入）。
    先检测章节标题作为边界，再将每节内容按 max_chunk_size 切分。
    每块保留所属章节标题作为上下文。

    Returns:
        list[dict]: [{"content": str, "page": int, "paragraph": int, "section": str}, ...]
    """
    page_marker = re.compile(r"^\[PAGE:(\d+)\]$")
    lines = text.split("\n")

    # ---- Phase 1: 定位所有章节边界 ----
    section_boundaries: list[int] = []
    section_titles: list[str] = []

    for i, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped or page_marker.match(stripped):
            continue
        if _is_section_heading(stripped):
            section_boundaries.append(i)
            section_titles.append(stripped)

    if not section_boundaries:
        # 无章节标题 → 整个文档视为一个"正文"节
        section_boundaries = [0]
        section_titles = ["正文"]

    # ---- Phase 2: 逐节收集内容并分块 ----
    chunks: list[dict] = []
    current_page = 1
    para_in_page = 0

    for si in range(len(section_boundaries)):
        sec_title = section_titles[si]
        start_line = section_boundaries[si]
        end_line = section_boundaries[si + 1] if si + 1 < len(section_boundaries) else len(lines)

        # 跳过标题行本身，从下一行开始收集内容
        section_texts: list[tuple[str, int, int]] = []  # (text, page, para_in_page)
        for j in range(start_line + 1, end_line):
            stripped = lines[j].strip()
            if not stripped:
                continue
            pm = page_marker.match(stripped)
            if pm:
                current_page = int(pm.group(1))
                para_in_page = 0
                continue
            para_in_page += 1
            section_texts.append((stripped, current_page, para_in_page))

        # 在节内按 max_chunk_size 切分
        buf = ""
        buf_page = section_texts[0][1] if section_texts else 1
        buf_para = section_texts[0][2] if section_texts else 1

        for txt, pg, pp in section_texts:
            if len(buf) + len(txt) > max_chunk_size:
                if buf:
                    chunks.append({
                        "content": buf.strip(),
                        "page": buf_page,
                        "paragraph": buf_para,
                        "section": sec_title,
                    })
                    buf = txt
                    buf_page = pg
                    buf_para = pp
                else:
                    # 单行超限：强制截断
                    for k in range(0, len(txt), max_chunk_size):
                        sub = txt[k:k + max_chunk_size].strip()
                        if sub:
                            chunks.append({
                                "content": sub,
                                "page": pg,
                                "paragraph": pp,
                                "section": sec_title,
                            })
                    buf = ""
            else:
                if not buf:
                    buf_page = pg
                    buf_para = pp
                buf += "\n" + txt if buf else txt

        if buf.strip():
            chunks.append({
                "content": buf.strip(),
                "page": buf_page,
                "paragraph": buf_para,
                "section": sec_title,
            })

    return chunks


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
