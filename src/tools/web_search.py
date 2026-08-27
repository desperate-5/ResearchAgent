import os
import json
import re
import urllib.request

from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool

BOCHA_API_URL = "https://api.bochaai.com/v1/web-search"


@tool
def web_search(query: str) -> str:
    """搜索互联网获取最新信息。当需要查找最新新闻、实时数据、或者超出知识截止日期的事实时使用。"""
    api_key = os.getenv("BOCHA_API_KEY", "")
    if not api_key:
        return "搜索服务未配置（缺少 BOCHA_API_KEY）。"

    body = json.dumps({
        "query": query,
        "count": 5,
        "freshness": "noLimit",
        "summary": True,
    }).encode("utf-8")

    req = urllib.request.Request(
        BOCHA_API_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return f"搜索请求失败: {e}"

    if data.get("code") != 200:
        return f"搜索失败: {data.get('message', '未知错误')}"

    web_pages = data.get("data", {}).get("webPages", {}).get("value", [])
    if not web_pages:
        return "未找到相关结果。"

    lines = []
    for i, r in enumerate(web_pages, 1):
        title = re.sub(r"\s+", " ", r.get("name", "无标题")).strip()
        url = r.get("url", "")
        summary = re.sub(r"\s+", " ", r.get("summary", "")).strip()
        snippet = re.sub(r"\s+", " ", r.get("snippet", "")).strip()
        site = r.get("siteName", "")
        date_crawled = r.get("dateLastCrawled", "")
        line = f"{i}. [web] {title}\n"
        if site:
            line += f"   来源: {site}\n"
        if url:
            line += f"   链接: {url}\n"
        if date_crawled:
            line += f"   附加: 发布时间: {date_crawled}\n"
        content = summary or snippet
        if content:
            line += f"   内容: {content}\n"
        lines.append(line)

    return "\n\n".join(lines)
