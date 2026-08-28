from langchain_core.tools import tool

from ..context.tool_compression import truncate_rag_chunk
from ..rag.store import search_chunks


def make_rag_tool(project_id: str):
    """创建一个绑定到特定项目的 RAG 检索工具。每次请求动态生成，确保 project_id 正确。"""

    @tool
    def search_uploaded_docs(query: str) -> str:
        """搜索用户已上传的 PDF/Word 文档内容。当用户提到「我上传的文件」「这篇论文」「文档里」「文件里」时使用此工具。"""
        docs = search_chunks(project_id, query)
        if not docs:
            return "未在已上传的文档中找到相关内容。"

        lines = []
        for i, doc in enumerate(docs, 1):
            src = doc.get("filename", "未知文件")
            content = truncate_rag_chunk(doc.get("content", ""))
            page = doc.get("page", 1)
            para = doc.get("paragraph", 0)
            section = doc.get("section", "")
            pos_parts = []
            if section:
                pos_parts.append(f"章节: {section}")
            if page and para:
                pos_parts.append(f"第{page}页 第{para}段")
            line = f"{i}. [doc] {src}\n"
            line += f"   来源: {src}\n"
            if pos_parts:
                line += f"   附加: {' | '.join(pos_parts)}\n"
            line += f"   内容: {content}\n"
            lines.append(line)

        return "\n\n".join(lines)

    return search_uploaded_docs
