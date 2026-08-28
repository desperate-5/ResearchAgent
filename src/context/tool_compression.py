"""工具输出压缩：限制/截断工具结果，控制注入 LLM 上下文的体量。

当前按需保留完整输出（不强制截断），此处先落好位置，后续压缩策略在此扩展。
"""

# 工具输出的整体字符上限（当前不强制截断）
MAX_OUTPUT_CHARS = 3500

# RAG 片段注入 prompt 的最大字符数：与 rag.config 的 CHUNK_SIZE(300) 对齐，保证每块完整注入且 prompt 体量可控
RAG_CHUNK_MAX_CHARS = 300


def truncate_output(text: str, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    """截断过长输出以降低下游 LLM 的首 token 延迟。

    当前按需保留完整输出：不再截断，直接返回原文。
    """
    if len(text) <= max_chars:
        return text
    return text


def truncate_rag_chunk(content: str, max_chars: int = RAG_CHUNK_MAX_CHARS) -> str:
    """截断单个 RAG 片段，超出上限时追加截断标记。"""
    if len(content) > max_chars:
        return content[:max_chars] + "…（片段已截断）"
    return content
