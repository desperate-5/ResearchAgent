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
    "https://datacenter.aminer.cn/gateway/open_platform/api/paper/search/pro",
)


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

    params = {
        "keyword": query,
        "size": str(min(count, 10)),
    }
    url = f"{AMINER_API_URL}?{urllib.parse.urlencode(params)}"

    req = urllib.request.Request(
        url,
        headers={"Authorization": token},
    )

    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return f"AMiner 搜索请求失败: {e}"

    # 实际返回: {"code":200, "data":[{...}, ...], "total": N}
    papers = data.get("data", [])
    if isinstance(papers, dict):
        papers = papers.get("items", []) or papers.get("papers", [])

    if not papers:
        return "未找到相关学术论文。"

    total = data.get("total", len(papers))

    lines = [f"共找到 {total} 篇论文，以下是前 {len(papers)} 篇：\n"]

    for i, p in enumerate(papers, 1):
        title = p.get("title", "无标题")
        first_author = p.get("first_author", "")
        venue_name = p.get("venue_name", "")
        year = p.get("year", "")
        citations = p.get("n_citation_bucket", "")
        doi = p.get("doi", "")

        line = f"{i}. **{title}**\n"
        if first_author:
            line += f"   第一作者: {first_author}\n"
        if venue_name or year:
            line += f"   来源: {venue_name}{' (' + str(year) + ')' if year else ''}\n"
        if citations:
            line += f"   引用量: {citations}\n"
        if doi:
            line += f"   DOI: {doi}\n"

        lines.append(line)

    return "\n".join(lines)
