"""规则信号评估（纯函数，零 LLM 依赖）。

域名白/黑名单、来源类型权重、URL 可达性、发布日期解析，
输出权威性 / 时效性维度分。所有函数可单测、可复现。
"""

import re
from datetime import datetime
from urllib.parse import urlparse

# ------------------------------------------------------------
# 域名规则表（可配置）
# ------------------------------------------------------------

# 加分域：预印本 / DOI / 学术搜索 / 期刊出版社官网
AUTHORITATIVE_DOMAINS = {
    "doi.org", "arxiv.org", "biorxiv.org", "medrxiv.org", "ssrn.com",
    "semanticscholar.org", "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov",
    "aminer.org", "aminer.cn",
    "nature.com", "science.org", "sciencemag.org", "pnas.org",
    "cell.com", "lancet.com", "nejm.org", "jamanetwork.com",
    "ieee.org", "acm.org", "springer.com", "springeropen.com",
    "elsevier.com", "sciencedirect.com", "wiley.com",
    "onlinelibrary.wiley.com", "tandfonline.com", "sagepub.com",
    "frontiersin.org", "plos.org", "mdpi.com", "hindawi.com",
    "iopscience.iop.org", "academic.oup.com", "cambridge.org",
    "degruyter.com", "emerald.com", "worldscientific.com",
}

# 权威顶级域后缀（官方机构 / 大学）
AUTHORITATIVE_TLDS = (
    ".edu", ".gov", ".mil", ".ac.uk", ".gov.uk", ".edu.au",
    ".edu.cn", ".gov.cn", ".ac.cn", ".cas.cn",
)

# 减分域：个人博客 / 自媒体 / 低质聚合 / 内容农场 / 问答平台
UNTRUSTED_DOMAINS = {
    "medium.com", "wordpress.com", "blogspot.com", "tumblr.com",
    "weebly.com", "wixsite.com", "substack.com", "github.io",
    "hashnode.dev", "notion.site",
    "csdn.net", "jianshu.com", "zhihu.com", "baijiahao.baidu.com",
    "toutiao.com", "360doc.com", "docin.com", "wenku.baidu.com",
    "diyifanwen.com", "reddit.com", "quora.com",
}

# 时效关键词：问题含这些词时，要求来源近 1-2 年
RECENCY_KEYWORDS = (
    "最新", "最近", "近年", "近期", "近一年", "近两年", "近三年",
    "进展", "前沿", "当前", "现状", "2025", "2026",
    "latest", "recent", "new", "state of the art", "sota",
)


def extract_domain(url: str) -> str:
    """从 URL 提取主域名（去掉 www 前缀，转小写）。"""
    if not url:
        return ""
    try:
        host = urlparse(url).hostname or ""
    except Exception:
        return ""
    host = host.lower().strip()
    if host.startswith("www."):
        host = host[4:]
    return host


def _match_any(domain: str, patterns) -> bool:
    """域名是否等于或属于某个 pattern（子域名匹配）。"""
    if not domain:
        return False
    return any(domain == p or domain.endswith("." + p) for p in patterns)


def is_authoritative(domain: str) -> bool:
    if not domain:
        return False
    d = domain.lower()
    if _match_any(d, AUTHORITATIVE_DOMAINS):
        return True
    return any(d.endswith(tld) for tld in AUTHORITATIVE_TLDS)


def is_untrusted(domain: str) -> bool:
    if not domain:
        return False
    return _match_any(domain.lower(), UNTRUSTED_DOMAINS)


def assess_authority(source: dict) -> int:
    """来源权威性打分（0-5），规则为主。

    基线：document(用户自身上传) 5 > paper 4 > web 3；
    域名白名单 +1（封顶 5），黑名单 -2（保底 0）。
    """
    stype = source.get("source_type", "web")
    url = source.get("url", "")
    domain = extract_domain(url)

    if stype == "document":
        score = 5
    elif stype == "paper":
        score = 4
    else:
        score = 3  # web 默认中性

    if domain:
        if is_authoritative(domain):
            score = min(5, score + 1)
        if is_untrusted(domain):
            score = max(0, score - 2)

    return score


def parse_year(source: dict) -> int | None:
    """尽力从来源中提取发表年份。优先 published 字段，其次 summary / title。"""
    pub = source.get("published", "")
    y = _extract_year_from_text(pub)
    if y is not None:
        return y
    for field in ("summary", "title"):
        y = _extract_year_from_text(source.get(field, ""))
        if y is not None:
            return y
    return None


def _extract_year_from_text(text: str) -> int | None:
    if not text:
        return None
    m = re.search(r"(19|20)\d{2}", str(text))
    if not m:
        return None
    y = int(m.group(0))
    if 1900 <= y <= datetime.now().year + 1:
        return y
    return None


def assess_timeliness(source: dict, user_query: str = "") -> int:
    """来源时效性打分（0-5）。

    无年份可判 → 中性 3；用户上传文档不苛责年份 → 4；
    问题含时效词 → 要求近 1-2 年；否则按年份缓慢衰减。
    """
    stype = source.get("source_type", "web")
    if stype == "document":
        return 4

    year = parse_year(source)
    if year is None:
        return 3

    age = max(0, datetime.now().year - year)
    query = (user_query or "").lower()
    require_recent = any(kw in query for kw in RECENCY_KEYWORDS)

    if require_recent:
        if age <= 1:
            return 5
        if age <= 2:
            return 4
        if age <= 3:
            return 3
        if age <= 5:
            return 2
        return 1

    if age <= 2:
        return 5
    if age <= 5:
        return 4
    if age <= 10:
        return 3
    return 2
