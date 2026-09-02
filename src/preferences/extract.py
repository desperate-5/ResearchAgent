"""陈述式显式偏好提取：规则预检（零 LLM 成本）+ LLM JSON 候选 + 校验。

信号档位：① 陈述式显式（用户主动、自发、没被问地表达偏好）→ 权重 +3。
"""

from __future__ import annotations

import json
import os
import re
import sys

from langchain_openai import ChatOpenAI

from .models import KNOWN_DIMENSIONS, PreferenceCandidate

# 规则预检：命中任一标记才进入 LLM 提取（未命中零 LLM 成本）
EXPLICIT_MARKERS = (
    "以后", "从现在起", "接下来", "尽量", "尽可能", "希望", "偏好", "更喜欢", "倾向于",
    "回答", "回复", "简洁", "简短", "精炼", "啰嗦", "详细", "详细点", "展开", "别太",
    "中文", "英文", "参考文献格式", "APA", "IEEE", "GB/T",
    "综述", "期刊", "会议论文", "对照组", "消融", "显著性检验",
)


EXPLICIT_PROMPT = """你是科研助手的偏好提取器。从用户的一句话中提取**显式表达**的研究/写作/检索偏好。

规则：
- 只提取用户明确说出、有意表达的偏好（显式信号），不要臆测未说出口的习惯。
- 每条候选必须能引用用户原句作为证据（evidence 字段），引用不出就不输出。
- 维度与取值必须来自下表，不要自造维度。

可识别的维度与取值：
- writing.sentence_style: concise（简洁/简短/精炼）| elaborate（详细/充分展开）
- writing.figure_norm: tight（图表紧凑）| spacious（图表宽松）
- writing.abstract_style: structured（结构化摘要）| narrative（叙述式摘要）
- writing.ref_format: GB/T 7714 | APA | IEEE
- writing.lang: chinese（中文）| english（英文）
- literature.source_type: journal（期刊）| conference（会议）
- literature.paper_type: review（综述）| experimental（实验）
- literature.preferred_language: chinese | english
- domain: 用户关注的学科/领域（自由文本，如"软件工程"）
- method: 用户偏好的研究方法（自由文本，如"实验对比"）

用户原句：
{text}

请只输出一个 JSON 数组（没有偏好则输出 []），不要任何其他文字：
[{{"dimension": "writing.sentence_style", "value": "concise", "evidence": "用户原句摘录"}}]"""


def rule_precheck(text: str) -> bool:
    """确定性预检：文本是否疑似显式偏好陈述（零 LLM 成本）。"""
    return bool(text) and any(m in text for m in EXPLICIT_MARKERS)


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "deepseek-chat"),
        api_key=os.getenv("OPENAI_API_KEY", "sk-xxx"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
        temperature=0.0,
        streaming=False,
        request_timeout=int(os.getenv("LLM_TIMEOUT", "60")),
    )


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


def normalize_candidate(raw: dict) -> PreferenceCandidate | None:
    """校验 + 归一化候选：维度必须在词汇表内，取值合法，否则丢弃。"""
    dim = str(raw.get("dimension", "")).strip()
    val = str(raw.get("value", "")).strip()
    evidence = str(raw.get("evidence", "")).strip()
    if dim not in KNOWN_DIMENSIONS:
        return None
    allowed = KNOWN_DIMENSIONS[dim]
    if allowed is not None:
        val = val.lower()
        if val not in allowed:
            return None
    if not val:
        return None
    return PreferenceCandidate(dimension=dim, value=val, evidence=evidence)


async def extract_explicit(text: str) -> list[PreferenceCandidate]:
    """从用户陈述中提取显式偏好候选。预检未命中返回空（零 LLM 成本）。"""
    if not rule_precheck(text):
        return []
    try:
        llm = _get_llm()
        response = await llm.ainvoke(EXPLICIT_PROMPT.format(text=text))
        raw = _parse_json_array(response.content)
    except Exception as e:
        print(f"[DEBUG preferences.extract] 显式提取失败: {e}", file=sys.stderr, flush=True)
        return []
    out = []
    for r in raw:
        c = normalize_candidate(r)
        if c is not None:
            out.append(c)
    return out
