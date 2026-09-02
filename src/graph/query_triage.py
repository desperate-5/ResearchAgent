"""检索前澄清门卫节点：规则预检 + LLM 分类，必要时 interrupt 让用户澄清。"""

import json
import os
import re
import sys

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt

from .state import AgentState
from .query_rules import rule_reject as _rule_reject
from .prompts import EXPLICIT_PLAN_KEYWORDS
from ..interaction.types import QueryClarificationPayload
from ..context.windowing import extract_user_query
from ..preferences.store import get_applied_hints


CLARIFY_PROMPT = """你是科研助手的检索意图澄清助手。判断用户的问题是否足够清晰、可以直接检索。

判断标准：
- 问题具体、明确、可直接用于检索 → ambiguous=false, invalid=false
- 完全无意义的随机字符串 / 乱码（如 "ndbajdkla566"、"asdhjqkw"、"gfdgfdgfd"）→ invalid=true（不检索、不生成方向）
- 问题模糊、有歧义、范围过宽、缺少关键信息（对象 / 目标 / 边界）→ ambiguous=true，并给出 3 个更具体的检索方向（每个方向须明确标注所属学科/领域，可独立检索）
- 纯寒暄 / 闲聊（你好、谢谢、再见）→ ambiguous=false（交给后续流程处理）

**概览型提问规则（重要）**：当用户用「想了解 / 介绍一下 / 现状 / 进展 / 入门 / 概述 / 有哪些」等表述，且主题可能横跨多个学科或可细分为多个方向时，必须返回 ambiguous=true。每个方向必须明确标注所属学科/领域，格式为「学科/领域：具体方向」，让用户一眼看出该方向属于哪个专业领域。

**How-to 提问规则（重要）**：当用户用「怎么做 / 如何做 / 怎样做 / 怎么实现 / 如何设计」等表述提问时，如果主题可横跨多个学科/领域（如 人机交互、机器人、人工智能、智能制造、数字孪生、智能体 等），必须返回 ambiguous=true，方向必须标注所属学科/领域（计算机、机械、电子、认知科学、人因工程、管理科学等），让用户确认是哪个专业的问题。只有主题明确属于单一领域的具体操作问题（如"Python list 怎么排序"）才返回 ambiguous=false。

示例：
- "我想了解一下人机协同" → {"ambiguous": true, "directions": ["人机交互(HCI)领域：人机协同的交互设计", "管理科学领域：人机协同的任务分配与组织", "机器人领域：人机协同的共享控制与安全"], "invalid": false}
- "最近 AI Agent 有什么进展" → {"ambiguous": true, "directions": ["软件工程领域：Agent 框架与架构", "认知科学领域：Agent 的规划与推理", "评测方法领域：Agent 基准与评估"], "invalid": false}
- "人机交互怎么做" → {"ambiguous": true, "directions": ["计算机科学与技术领域：人机交互系统与界面设计", "机械工程领域：人机交互的硬件设备与工效学", "认知科学与心理学领域：人机交互的认知模型与用户体验"], "invalid": false}
- "帮我查 2024 年 LLM Agent 记忆机制的综述论文" → {"ambiguous": false, "directions": [], "invalid": false}
- "ndbajdkla566" → {"ambiguous": false, "directions": [], "invalid": true}

用户问题：
{question}

请只输出一个 JSON 对象，不要添加任何其他文字：
{"ambiguous": false, "directions": [], "invalid": false}
或
{"ambiguous": true, "directions": ["方向1", "方向2", "方向3"], "invalid": false}
或
{"ambiguous": false, "directions": [], "invalid": true}"""


def _get_classify_llm() -> ChatOpenAI:
    """澄清分类专用 LLM。低温度提高判定稳定性。"""
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "deepseek-chat"),
        api_key=os.getenv("OPENAI_API_KEY", "sk-xxx"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
        temperature=0.0,
        streaming=False,
        request_timeout=int(os.getenv("LLM_TIMEOUT", "60")),
    )


BROAD_INTENT_PATTERNS = (
    # 概览型：想了解 / 介绍一下 / 现状 / 进展 ...
    "想了解", "了解一下", "了解下", "介绍一下", "介绍下",
    "概述", "现状", "进展", "入门", "有哪些", "涉及哪些", "哪些领域",
    # How-to 型：怎么做 / 如何做 ...（主题往往横跨多个学科，如"人机交互怎么做"）
    "怎么做", "如何做", "怎样做",
    "怎么实现", "如何实现", "怎样实现", "怎么设计", "如何设计", "怎样设计",
    "怎么开展", "如何开展", "怎样开展", "怎么进行", "如何进行", "怎样进行",
    "怎么研究", "如何研究", "怎样研究", "怎么构建", "如何构建", "怎样构建",
    "怎么开发", "如何开发", "怎样开发", "怎么搭建", "如何搭建", "怎样搭建",
)


def _has_broad_intent(text: str) -> bool:
    """概览型 / How-to 型提问的确定性信号（零 LLM 成本）。

    命中即强制澄清：这类提问主题往往横跨多个学科（人机交互、机器人、智能制造等），
    需要先让用户确认专业方向。How-to 模式只匹配"怎么做/如何做/怎么实现"等多字动宾
    组合，避免"怎么排序""怎么用"这类具体小问题被误触发。
    """
    return any(p in text for p in BROAD_INTENT_PATTERNS)


def _parse_json(text: str) -> dict:
    """从模型输出中提取 JSON 对象，失败时降级为不澄清。"""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        try:
            data = json.loads(text[start:end + 1])
            if isinstance(data, dict):
                directions = data.get("directions", [])
                if isinstance(directions, str):
                    directions = [directions]
                elif not isinstance(directions, list):
                    directions = []
                return {
                    "ambiguous": bool(data.get("ambiguous", False)),
                    "directions": [str(d) for d in directions if str(d).strip()],
                    "invalid": bool(data.get("invalid", False)),
                }
        except json.JSONDecodeError:
            pass
    return {"ambiguous": False, "directions": [], "invalid": False}


def _parse_json_array(text: str) -> list[str]:
    """从模型输出中提取 JSON 数组（方向补充用），失败返回空列表。"""
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1:
        try:
            data = json.loads(text[start:end + 1])
            if isinstance(data, list):
                return [str(d).strip() for d in data if str(d).strip()]
        except json.JSONDecodeError:
            pass
    return []


EXPAND_DIRECTIONS_PROMPT = """为下面的研究主题生成 {n} 个具体的检索方向，每个方向一句话、可独立检索，覆盖不同学科或领域，且每个方向必须明确标注所属学科/领域（格式：学科/领域：具体方向）。

主题：{topic}

已有方向（不要重复）：
{existing}

请只输出一个 JSON 数组，例如：
["领域1：方向1", "领域2：方向2"]"""


async def _expand_directions(raw: str, existing: list[str], target: int = 3) -> list[str]:
    """方向不足 target 时，请求 LLM 补足（去重）。失败则原样返回。"""
    result = list(existing)
    need = target - len(result)
    if need <= 0:
        return result
    try:
        llm = _get_classify_llm()
        existing_text = "\n".join(f"- {d}" for d in result) or "（无）"
        prompt = EXPAND_DIRECTIONS_PROMPT.format(n=need, topic=raw, existing=existing_text)
        response = await llm.ainvoke(prompt)
        for d in _parse_json_array(response.content):
            if d not in result:
                result.append(d)
                if len(result) >= target:
                    break
    except Exception as e:
        print(f"[DEBUG query_triage] 方向补充失败: {e}", file=sys.stderr, flush=True)
    return result


async def _ensure_directions(raw: str, directions: list[str], target: int = 3) -> list[str]:
    """保证方向数量达到 target：LLM 补足 + 确定性模板兜底。"""
    result = await _expand_directions(raw, [d for d in directions if d], target)
    templates = [
        f"「{raw}」的核心概念与理论基础",
        f"「{raw}」的研究现状与关键进展",
        f"「{raw}」的主要应用场景与方法",
    ]
    for t in templates:
        if len(result) >= target:
            break
        if t not in result:
            result.append(t)
    return result[:target]


def _resolve_choice(choice, raw: str) -> str:
    """把用户的选择解析为有效检索查询。"""
    if isinstance(choice, dict):
        if choice.get("use_original"):
            return raw
        if choice.get("selected_direction"):
            return choice["selected_direction"]
    return raw


async def _classify(raw: str, broad_hint: bool = False) -> dict:
    """LLM 判断问题是否模糊。失败时降级为不澄清，保证澄清绝不阻塞主流程。"""
    try:
        llm = _get_classify_llm()
        prompt = CLARIFY_PROMPT.format(question=raw)
        if broad_hint:
            prompt += "\n\n【特别提示】该提问属于概览型或 How-to 型提问（主题可能横跨多个学科/领域），请务必返回 ambiguous=true 并给出 3 个细分方向，每个方向须标注所属学科/领域（格式：学科/领域：具体方向）。"
        response = await llm.ainvoke(prompt)
        return _parse_json(response.content)
    except Exception as e:
        print(f"[DEBUG query_triage] 分类失败，直通检索: {e}", file=sys.stderr, flush=True)
        return {"ambiguous": False, "directions": [], "invalid": False}


INVALID_QUERY_RESPONSE = "抱歉，我无法理解您的输入，请重新输入一个更明确、具体的问题。"


def _has_explicit_plan_intent(text: str) -> bool:
    """明确的方案设计意图（零 LLM 成本）：命中则跳过检索前澄清，直接交给 supervisor 的 planner 路由。"""
    return any(p in text for p in EXPLICIT_PLAN_KEYWORDS)


def _tokens(text: str) -> set[str]:
    """把方向/领域文本拆成粗粒度 token 集合（用于重叠打分）。"""
    parts = re.split(r"[：:/\s、,，]+", (text or ""))
    toks = set()
    for p in parts:
        p = p.strip()
        if p:
            toks.add(p)
    for w in list(toks):
        for i in range(len(w) - 1):
            toks.add(w[i:i + 2])
    return toks


def _rank_by_domain(directions: list[str], domain: str) -> list[str]:
    """已生效的 domain 学习条目 → 澄清方向排序置顶（稳定排序，不改变方向集合）。"""
    if not domain:
        return directions
    dom_tokens = _tokens(domain)

    def score(d: str) -> int:
        return len(dom_tokens & _tokens(d))

    return sorted(directions, key=score, reverse=True)


async def query_triage_node(state: AgentState) -> dict:
    """检索前澄清门卫：规则预检 + LLM 分类，需要时中断让用户澄清。

    清晰问题直通；模糊问题中断让用户选方向；无效输入（乱码/废话）直接返回提示并
    结束本轮（不检索、不中断），由用户在主输入框重新提问。
    """
    raw = extract_user_query(state)

    # ① 规则预检：乱码 / 废话 / 空输入 → 直接提示重新输入，零检索、零中断
    if _rule_reject(raw):
        return {
            "query_invalid": True,
            "messages": [AIMessage(content=INVALID_QUERY_RESPONSE)],
        }

    # ② 明确方案设计意图（"方案设计 / 设计方案 / 研究方案"等）→ 跳过澄清，
    #    直接交给 supervisor 的 planner 路由；泛化的 How-to 问题仍走澄清
    if _has_explicit_plan_intent(raw):
        return {"effective_query": raw, "was_clarified": False}

    # ③ LLM 分类：清晰 → 直通；乱码/随机串（规则层漏网）→ 无效提示；
    #    模糊 / 概览型 / How-to 型 → 生成方向并中断
    broad_hint = _has_broad_intent(raw)
    verdict = await _classify(raw, broad_hint=broad_hint)
    if verdict.get("invalid"):
        return {
            "query_invalid": True,
            "messages": [AIMessage(content=INVALID_QUERY_RESPONSE)],
        }
    ambiguous = bool(verdict.get("ambiguous")) or broad_hint
    if not ambiguous:
        return {"effective_query": raw, "was_clarified": False}

    directions = await _ensure_directions(raw, verdict.get("directions", []))
    # M3：已生效的 domain 偏好 → 澄清方向排序置顶（不改变方向集合，仅重排）
    domain = get_applied_hints(state.get("project_id", "")).get("domain", "")
    if domain:
        directions = _rank_by_domain(directions, domain)
    choice = interrupt(QueryClarificationPayload(directions=directions).to_dict())
    return {"effective_query": _resolve_choice(choice, raw), "was_clarified": True}
