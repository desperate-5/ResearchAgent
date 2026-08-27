import os
import json
import time
import urllib.request
import urllib.parse

import jwt
from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool

AMINER_API_URL = os.getenv(
    "AMINER_API_URL",
    "https://datacenter.aminer.cn/gateway/open_platform/api/paper/qa/search",
)

AMINER_PAPER_INFO_URL = os.getenv(
    "AMINER_PAPER_INFO_URL",
    "https://datacenter.aminer.cn/gateway/open_platform/api/paper/info",
)

AMINER_PAPER_DETAIL_URL = os.getenv(
    "AMINER_PAPER_DETAIL_URL",
    "https://datacenter.aminer.cn/gateway/open_platform/api/paper/detail",
)

def _post_json(url: str, payload: dict, token: str, timeout: int = 8) -> dict:
    """向 AMiner 开放平台发送 POST JSON 请求，返回解析后的 JSON。"""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": token,
            "Content-Type": "application/json;charset=utf-8",
            "X-Platform": "openclaw",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_json(url: str, params: dict, token: str, timeout: int = 8) -> dict:
    """向 AMiner 开放平台发送 GET 请求（带查询参数），返回解析后的 JSON。"""
    query = urllib.parse.urlencode(params)
    req = urllib.request.Request(
        f"{url}?{query}" if query else url,
        headers={
            "Authorization": token,
            "X-Platform": "openclaw",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _generate_token() -> str:
    """使用 API Key + user_id 组装 JWT token（HS256，sign_type=SIGN）。"""
    signing_key = os.getenv("AMINER_API_KEY", "")
    user_id = os.getenv("AMINER_USER_ID", "")

    if not signing_key or not user_id:
        return ""

    now = int(time.time())
    payload = {
        "user_id": user_id,
        "exp": now + 7200,
        "timestamp": now,
    }
    headers = {
        "alg": "HS256",
        "sign_type": "SIGN",
    }
    return jwt.encode(payload, signing_key, algorithm="HS256", headers=headers)


@tool
def aminer_search_papers(query: str, count: int = 5) -> str:
    """搜索正式发表的学术论文（中英文均可）。当用户需要查找研究论文、文献综述、学术文章时使用。
    不适合搜索实时新闻、行业资讯、博客文章（这些用 web_search）。"""

    token = _generate_token()
    if not token:
        return "AMiner 学术搜索未配置（请在 .env 中设置 AMINER_API_KEY 和 AMINER_USER_ID）。"

    try:
        data = _post_json(AMINER_API_URL, {
            "use_topic": True,
            "query": query,
            "size": min(count, 10),
        }, token)
    except Exception as e:
        return f"AMiner 搜索请求失败: {e}"

    # paper/qa/search 返回: {"code":200, "success":true, "data":[{...}, ...]}
    papers = data.get("data", [])

    if not papers:
        return "未找到相关学术论文。"

    total = data.get("total", len(papers))

    # 优先付费 paper/detail 逐个获取完整摘要；拿不到（接口异常/无字段）再回退免费 paper/info 的 abstract_slice
    abstracts: dict[str, str] = {}
    paper_ids = [p.get("id", "") for p in papers if p.get("id")]

    def _extract_abstract(item: dict) -> str:
        for key in ("abstract", "abstract_zh", "abstract_slice"):
            val = item.get(key)
            if isinstance(val, str) and val:
                return val
        return ""

    for pid in paper_ids:
        try:
            detail = _get_json(AMINER_PAPER_DETAIL_URL, {"id": pid}, token)
            detail_papers = detail.get("data", [])
            if isinstance(detail_papers, dict):
                detail_papers = [detail_papers]
            for ip in detail_papers:
                ab = _extract_abstract(ip)
                if ab:
                    abstracts[pid] = ab
                    break
        except Exception:
            pass  # 单篇详情失败，交给下方批量回退

    missing_ids = [pid for pid in paper_ids if pid not in abstracts]
    if missing_ids:
        try:
            info = _post_json(AMINER_PAPER_INFO_URL, {"ids": missing_ids}, token)
            info_papers = info.get("data", [])
            if isinstance(info_papers, dict):
                info_papers = info_papers.get("items", []) or info_papers.get("papers", [])
            for ip in info_papers:
                ab = _extract_abstract(ip)
                if ab and ip.get("id") not in abstracts:
                    abstracts[ip.get("id")] = ab
        except Exception:
            pass  # 摘要获取失败不影响论文列表

    lines = [f"共找到 {total} 篇论文，以下是前 {len(papers)} 篇：\n"]

    for i, p in enumerate(papers, 1):
        title = p.get("title") or p.get("title_zh") or "无标题"
        first_author = p.get("first_author", "")
        venue_name = p.get("venue_name", "")
        year = p.get("year", "")
        citations = p.get("n_citation_bucket", "")
        doi = p.get("doi", "")

        line = f"{i}. [paper] {title}\n"
        if venue_name or year:
            line += f"   来源: {venue_name}{' (' + str(year) + ')' if year else ''}\n"
        if doi:
            line += f"   链接: https://doi.org/{doi}\n"
        extra_parts = []
        if first_author:
            extra_parts.append(f"第一作者: {first_author}")
        if citations:
            extra_parts.append(f"引用量: {citations}")
        if extra_parts:
            line += f"   附加: {' | '.join(extra_parts)}\n"

        abstract = abstracts.get(p.get("id", ""), "")
        if abstract:
            line += f"   内容: {abstract}\n"

        lines.append(line)

    return "\n".join(lines)
