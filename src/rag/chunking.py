import re

from .config import CHUNK_SIZE, CHUNK_OVERLAP

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
