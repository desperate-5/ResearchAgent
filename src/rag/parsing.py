import fitz
from docx import Document


def parse_pdf(file_path: str) -> str:
    """解析 PDF 并插入 [PAGE:N] 页码标记，供 chunk_by_sections 追踪页码。"""
    doc = fitz.open(file_path)
    text_parts = []
    for i, page in enumerate(doc):
        text_parts.append(f"[PAGE:{i + 1}]\n{page.get_text()}")
    doc.close()
    return "\n".join(text_parts)


def parse_docx(file_path: str) -> str:
    doc = Document(file_path)
    return "\n".join(p.text for p in doc.paragraphs)
