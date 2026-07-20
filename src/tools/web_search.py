import os
import json
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
        with urllib.request.urlopen(req, timeout=15) as resp:
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
        title = r.get("name", "无标题")
        url = r.get("url", "")
        summary = r.get("summary", "")
        site = r.get("siteName", "")
        lines.append(f"{i}. {title}\n   来源: {site} | URL: {url}\n   {summary}")

    return "\n\n".join(lines)
