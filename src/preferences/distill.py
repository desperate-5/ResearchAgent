"""会话蒸馏（观察式隐式 · 第③档）：从多轮对话里批量提炼用户未明说的习惯。

只在显式提取盲区工作：从没说出口的偏好、从没给过选择机会的偏好。
防坑铁律：每条候选必须引用用户原句，引用不出就不输出。
"""

from __future__ import annotations

import json
import os
import re
import sys

from langchain_openai import ChatOpenAI

from .models import KNOWN_DIMENSIONS, ObservedCandidate

DISTILL_PROMPT = """你是用户画像蒸馏器。从最近多轮对话中，找出用户**反复表现、但从未明说**的隐性偏好。

规则（防坑铁律）：
- 只提炼用户自己的话和选择，不要把助手回答质量误当成用户偏好。
- 每条候选必须能引用至少 3 句用户原话作为证据（evidence），引用不出就不输出。
- 关注跨轮重复模式：反复问同一个指标、反复要求多图、反复指定某领域/某方法等。
- 维度与取值必须来自下表，不要自造维度。

可识别的维度与取值：
- writing.sentence_style: concise | elaborate
- writing.figure_norm: tight | spacious
- writing.abstract_style: structured | narrative
- writing.lang: chinese | english
- literature.source_type: journal | conference
- literature.paper_type: review | experimental
- literature.preferred_language: chinese | english
- domain: 用户反复关注的学科/领域（自由文本）
- method: 用户反复偏好的研究方法（自由文本）

对话历史（用户/助手交替）：
{conversation}

{existing}

请只输出一个 JSON 数组（没有可靠的隐性偏好则输出 []），不要任何其他文字：
[{{"dimension": "method", "value": "实验对比", "evidence": ["用户原句1", "用户原句2", "用户原句3"]}}]"""


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "deepseek-chat"),
        api_key=os.getenv("OPENAI_API_KEY", "sk-xxx"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
        temperature=0.2,
        streaming=False,
        request_timeout=int(os.getenv("LLM_TIMEOUT", "60")),
    )


def _format_turns(history: list[dict]) -> str:
    lines = []
    for m in history:
        role = "用户" if m.get("role") == "user" else "助手"
        lines.append(f"{role}: {m.get('content', '')}")
    return "\n".join(lines)


def _parse_json_array(text: str) -> list[dict]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1:
        try:
            data = json.loads(text[start:end + 1])
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
        except json.JSONDecodeError:
            pass
    return []


def _normalize(raw: dict) -> ObservedCandidate | None:
    dim = str(raw.get("dimension", "")).strip()
    val = str(raw.get("value", "")).strip()
    ev = raw.get("evidence", [])
    if isinstance(ev, str):
        ev = [ev]
    ev = [str(e).strip() for e in ev if str(e).strip()]
    if dim not in KNOWN_DIMENSIONS:
        return None
    allowed = KNOWN_DIMENSIONS[dim]
    if allowed is not None:
        val = val.lower()
        if val not in allowed:
            return None
    if not val:
        return None
    if len(ev) < 3:  # 观察式需 ≥3 次重复才可信
        return None
    return ObservedCandidate(dimension=dim, value=val, evidence=ev[:5])


async def distill(history: list[dict], existing_summary: str = "") -> list[ObservedCandidate]:
    """蒸馏近 N 轮对话为隐性偏好候选（无可靠证据丢弃）。"""
    user_msgs = [m for m in history if m.get("role") == "user"]
    if len(user_msgs) < 3:
        return []
    try:
        llm = _get_llm()
        existing = f"已有摘要：\n{existing_summary}" if existing_summary else ""
        prompt = DISTILL_PROMPT.format(conversation=_format_turns(history), existing=existing)
        response = await llm.ainvoke(prompt)
        raw = _parse_json_array(response.content)
    except Exception as e:
        print(f"[DEBUG preferences.distill] 蒸馏失败: {e}", file=sys.stderr, flush=True)
        return []
    out = []
    for r in raw:
        c = _normalize(r)
        if c is not None:
            out.append(c)
    return out
