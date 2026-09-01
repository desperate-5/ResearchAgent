"""LLM 语义评估（两阶段调用）。

阶段 1：N 次并行调用，逐来源输出相关性分（每条来源 vs 用户问题）。
阶段 2：1 次全局调用，输出一致性分 + 缺口列表 + 质量小结。
解析失败时降级为中性分 / 空结果，不阻塞主流程。
"""

import asyncio
import json
import os
import re
import sys
import typing

from langchain_openai import ChatOpenAI

from .schemas import RelevanceResult, GlobalAssessment

NEUTRAL = 2.5


def get_llm(streaming: bool = False) -> ChatOpenAI:
    """评估专用 LLM。低温度提高评分稳定性。"""
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "deepseek-chat"),
        api_key=os.getenv("OPENAI_API_KEY", "sk-xxx"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
        temperature=0.0,
        streaming=streaming,
        request_timeout=int(os.getenv("LLM_TIMEOUT", "60")),
    )


RELEVANCE_PROMPT = """你是信息源相关性评估专家。判断下面这条来源内容与用户问题的相关程度，给出 0-5 分。

评分标准：
- 5：完全针对问题核心，直接可用
- 4：高度相关，覆盖主要方面
- 3：部分相关，仅覆盖问题的某一方面
- 2：弱相关，仅边缘沾边
- 1：基本无关
- 0：完全不相关

用户问题：
{question}

来源内容：
标题：{title}
类型：{source_type}
摘要：{summary}"""


GLOBAL_PROMPT = """你是信息源综合评估专家。基于给定的多条来源，完成四件事：

1. **一致性**：判断每条来源的结论与其他来源是否一致，逐条给出 0-5 分（5=与多数来源一致，0=与其他来源明显冲突）。
2. **缺口**：列出用户问题尚未被这些来源覆盖的关键子主题（每条一句话）。没有明显缺口则输出空列表。
3. **needs_refetch**：判断这批来源是否**明显不足**、需要立即补搜。判定标准（重要）：
   - 只有当前结果**无法回答用户问题的核心部分**（如关键子主题完全缺失、来源数量过少、相关性普遍偏低）时才为 true；
   - 检索结果大体可用、只是"还能补充更多细节 / 更多文献 / 更深层次"时，**必须为 false**（即使列出了少量可选补充主题）；
   - 不确定时取 false（宁可少补搜一次，节省时间）。
4. **小结**：用 1-2 句话概括这批来源的整体质量。

用户问题：
{question}

来源列表（编号 | 标题 | 一句话结论）：
{sources_text}"""


def _extract_json(text: str) -> str:
    """从模型输出中提取 JSON 子串（支持 markdown 代码块围栏）。"""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        return text[start:end + 1]
    return text


def _build_example(pydantic_cls) -> dict:
    """根据 Pydantic 模型字段生成示例 JSON，用于提示模型输出正确字段名。"""
    ex: dict = {}
    for name, field in pydantic_cls.model_fields.items():
        ann = field.annotation
        origin = typing.get_origin(ann)
        if origin is list:
            args = typing.get_args(ann)
            if args and hasattr(args[0], "model_fields"):
                ex[name] = [_build_example(args[0])]
            else:
                ex[name] = []
        elif ann in (int, float):
            ex[name] = 3
        elif ann is str:
            ex[name] = "一句话理由"
        elif ann is bool:
            ex[name] = False  # 布尔字段示例默认 false，避免引导模型输出 true（needs_refetch 同理）
        else:
            ex[name] = None
    return ex


def _coerce_fields(pydantic_cls, data: dict) -> dict:
    """字段名宽松映射：模型可能输出 score / relevance_score 等别名，映射回 schema 字段名。"""
    aliases = {
        "relevance": ("relevance", "score", "relevance_score", "rating"),
        "consistency": ("consistency", "score"),
        "source_number": ("source_number", "id", "source"),
        "summary": ("summary", "sum", "conclusion"),
        "gaps": ("gaps", "gap", "missing", "missing_topics"),
        "needs_refetch": ("needs_refetch", "need_refetch", "refetch", "need_retry", "should_refetch"),
        "reason": ("reason", "explanation", "why"),
    }
    fields = pydantic_cls.model_fields
    out: dict = {}
    for fname in fields:
        if fname in data:
            out[fname] = data[fname]
            continue
        for alias in aliases.get(fname, ()):
            if alias in data:
                out[fname] = data[alias]
                break
    return out


def _is_retryable(err: Exception) -> bool:
    """Connection 断连 / 限流（429）类错误可重试，其余直接失败。"""
    text = str(err).lower()
    return "connection" in text or "rate limit" in text or "429" in text or "timeout" in text


async def _ainvoke_with_retry(fn, retries: int = 2) -> object:
    """带指数退避的重试包装：Connection / 限流类错误最多重试 retries 次。"""
    for attempt in range(retries + 1):
        try:
            return await fn()
        except Exception as e:
            if attempt < retries and _is_retryable(e):
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            raise


async def _invoke_structured(llm, pydantic_cls, prompt: str):
    """结构化调用，带多级兜底：

    1. json_mode（response_format=json_object，思考模式模型通用）
    2. function_calling（部分模型不支持 json_mode 时兜底）
    3. 裸调用 + 手动 JSON 解析（含字段名宽松映射）
    全部失败时返回 None，由调用方降级为中性分。
    """
    methods = ["json_mode", "function_calling"]

    example = json.dumps(_build_example(pydantic_cls), ensure_ascii=False)
    schema_hint = f"\n\n请只输出一个 JSON 对象，字段名必须与以下结构完全一致：\n{example}"

    for method in methods:
        try:
            structured_llm = llm.with_structured_output(pydantic_cls, method=method)
            prompt_text = prompt + schema_hint if method == "json_mode" else prompt
            return await _ainvoke_with_retry(lambda: structured_llm.ainvoke(prompt_text))
        except Exception as e:
            print(f"[reviewer] structured output({method}) 失败: {e}", file=sys.stderr, flush=True)

    try:
        raw = await _ainvoke_with_retry(lambda: llm.ainvoke(prompt + schema_hint))
        data = json.loads(_extract_json(str(raw.content)))
        return pydantic_cls.model_validate(_coerce_fields(pydantic_cls, data))
    except Exception as e:
        print(f"[reviewer] 手动 JSON 解析失败: {e}", file=sys.stderr, flush=True)
        return None


async def assess_relevance(llm, sources: list[dict], user_query: str) -> dict[int, tuple[float, str]]:
    """阶段 1：N 次并行调用，逐来源输出相关性分。返回 {source_number: (score, reason)}。"""
    if not sources:
        return {}

    async def _one(source: dict):
        num = source.get("source_number", 0)
        prompt = RELEVANCE_PROMPT.format(
            question=user_query,
            title=source.get("title", ""),
            source_type=source.get("source_type", ""),
            summary=(source.get("summary") or "")[:500],
        )
        result = await _invoke_structured(llm, RelevanceResult, prompt)
        if result is None:
            return num, NEUTRAL, "解析失败，取中性分"
        return num, float(result.relevance), result.reason

    results = await asyncio.gather(*[_one(s) for s in sources])
    out: dict[int, tuple[float, str]] = {}
    for num, score, reason in results:
        out[num] = (score, reason)
    return out


async def assess_global(llm, sources: list[dict], user_query: str) -> GlobalAssessment:
    """阶段 2：1 次全局调用，输出一致性 + 缺口 + 小结。"""
    if not sources:
        return GlobalAssessment()

    lines = []
    for s in sources:
        num = s.get("source_number", "?")
        title = s.get("title", "")
        summary = (s.get("summary") or "").strip()
        one_line = summary.split("\n")[0][:120] if summary else ""
        lines.append(f"[{num}] {title} | {one_line}")
    sources_text = "\n".join(lines)

    prompt = GLOBAL_PROMPT.format(question=user_query, sources_text=sources_text)
    result = await _invoke_structured(llm, GlobalAssessment, prompt)
    if result is None:
        return GlobalAssessment()
    return result
