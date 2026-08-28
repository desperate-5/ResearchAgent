"""工具输出来源解析（统一外壳格式）。

web_search / aminer_search_papers / search_uploaded_docs 三个工具统一输出：
    N. [type] title
       来源: xxx
       链接: xxx
       附加: xxx
       内容: xxx
"""
import hashlib
import re

_SOURCE_TYPE_MAP = {
    "web_search": "web",
    "aminer_search_papers": "paper",
    "search_uploaded_docs": "document",
}

_ENTRY_PATTERN = re.compile(
    r'\d+\.\s*\[(\w+)\]\s*(.+?)\n(.*?)(?=\n\d+\.\s*\[\w+\]|\Z)',
    re.DOTALL,
)


def parse_tool_sources(tool_name: str, output_text: str) -> list[dict]:
    """从工具输出文本中解析结构化来源信息（统一外壳格式）。"""
    sources: list[dict] = []
    for m in _ENTRY_PATTERN.finditer(output_text):
        entry_type = m.group(1).strip()
        title = m.group(2).strip()
        body = m.group(3)

        def _field(key: str) -> str:
            fm = re.search(rf'^\s*{key}:\s*(.*)', body, re.M)
            return fm.group(1).strip() if fm else ""

        source_name = _field("来源")
        url = _field("链接")
        extra = _field("附加")
        content = _field("内容")
        sid = hashlib.md5(f"{url}|{title}|{source_name}".encode()).hexdigest()[:12]
        src: dict = {
            "id": sid,
            "title": title,
            "url": url,
            "summary": content or extra,
            "source_type": _SOURCE_TYPE_MAP.get(tool_name, entry_type),
        }
        if entry_type == "web":
            src["published"] = _extract_publish_date(extra)
        elif entry_type == "paper":
            src["published"] = _extract_year(source_name)
        elif entry_type == "doc":
            section, page, para = _parse_doc_position(extra)
            pos_parts = []
            if section:
                pos_parts.append(section)
            if para:
                pos_parts.append(f"第{para}段")
            src["chunk_index"] = 0
            src["section"] = section
            src["page"] = page
            src["position"] = " | ".join(pos_parts)
            src["summary"] = content[:50].replace("\n", " ")
        sources.append(src)
    return sources


def _parse_doc_position(extra: str) -> tuple[str, int, int]:
    """从文档来源的「附加」字段解析 (章节, 页码, 段落)。"""
    section = ""
    page = 1
    para = 0
    if extra:
        sec_m = re.search(r'章节:\s*([^|]+)', extra)
        if sec_m:
            section = sec_m.group(1).strip()
        page_m = re.search(r'第(\d+)页', extra)
        if page_m:
            page = int(page_m.group(1))
        para_m = re.search(r'第(\d+)段', extra)
        if para_m:
            para = int(para_m.group(1))
    return section, page, para


def _extract_year(text: str) -> str:
    """从文本中提取 4 位年份字符串（供论文来源 published 字段使用）。"""
    if not text:
        return ""
    m = re.search(r"(19|20)\d{2}", text)
    return m.group(0) if m else ""


def _extract_publish_date(extra: str) -> str:
    """从 web 来源的「附加」字段提取发布日期（如「发布时间: 2024-06-15」）。"""
    if not extra:
        return ""
    m = re.search(r"发布时间[:\s]*([0-9]{4}[-/][0-9]{1,2}[-/][0-9]{1,2})", extra)
    if m:
        return m.group(1)
    m2 = re.search(r"((19|20)\d{2})年([0-9]{1,2})月?", extra)
    return m2.group(1) if m2 else ""
