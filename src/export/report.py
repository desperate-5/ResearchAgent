import os
from datetime import datetime, timezone

from langchain_openai import ChatOpenAI

from ..memory.store import get_history, get_summary, get_project_sources, get_latest_plan
from ..projects.manager import get_project


REPORT_SYSTEM_PROMPT = """你是一个学术写作专家。你的任务是基于已有的研究资料，撰写一篇高质量、结构化的中文学术文章。

写作要求：
- 使用中文撰写，学术风格，专业、严谨、简洁
- 结构清晰：摘要 → 引言 → 相关研究 → 技术方案 → 分析与讨论 → 结论 → 参考文献
- 基于提供的资料内容，不要编造不存在的数据或结论
- 对于资料中不充分的部分，用「[需要进一步调研]」标注
- 使用 Markdown 格式，合理使用标题、列表、表格
- 文中引用时用 [N] 标注，与参考文献列表对应
- 如果提供了研究方案，要围绕方案展开论述"""


async def generate_report(project_id: str) -> str:
    """从对话历史 + 来源 + 方案生成结构化 Markdown 学术文章。"""
    project = get_project(project_id)
    project_name = project["name"] if project else "未知项目"

    history = get_history(project_id)
    summary = get_summary(project_id)
    sources = get_project_sources(project_id)
    latest_plan = get_latest_plan(project_id)

    transcript = _format_transcript(history)

    if not transcript.strip():
        return _empty_report(project_name)

    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "deepseek-chat"),
        api_key=os.getenv("OPENAI_API_KEY", "sk-xxx"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
        temperature=0.3,
    )

    user_prompt_parts = [f"## 项目名称\n{project_name}"]

    if summary:
        user_prompt_parts.append(f"## 历史对话摘要\n{summary}")

    if latest_plan:
        plan_prefix = "自定义" if latest_plan.get("is_custom") else "已选定"
        user_prompt_parts.append(
            f"## 当前研究方案\n{plan_prefix}方案：{latest_plan.get('plan_title', '')}\n{latest_plan.get('plan_detail', '')}"
        )

    if sources:
        src_lines = []
        for s in sources:
            sn = s.get("source_number", "?")
            src_lines.append(f"[{sn}] {s.get('title', '')} - {s.get('url', '')}")
        if src_lines:
            user_prompt_parts.append(f"## 参考文献来源\n" + "\n".join(src_lines))

    user_prompt_parts.append(f"## 完整对话记录\n{transcript}")
    user_prompt_parts.append("请基于以上资料撰写学术文章。")

    user_prompt = "\n\n".join(user_prompt_parts)

    try:
        response = await llm.ainvoke([
            {"role": "system", "content": REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ])
        body = response.content
    except Exception:
        body = _fallback_report(project_name, history, summary)

    # 拼装最终报告
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total_messages = len(history)

    header = f"""# 学术文章：{project_name}

> 生成时间：{now} | 对话轮次：{total_messages} 条 | 摘要状态：{"已有" if summary else "暂无"}

---

"""
    return header + body


# ---- 内部函数 ----

def _format_transcript(history: list[dict]) -> str:
    """将对话历史格式化为可读的对话记录。"""
    lines = []
    for i, msg in enumerate(history, 1):
        role_label = "用户" if msg["role"] == "user" else "助手"
        content = msg.get("content", "")
        if content:
            # 截断过长的消息
            if len(content) > 2000:
                content = content[:2000] + "\n...[内容过长已截断]"
            lines.append(f"### [{i}] {role_label}\n{content}")
    return "\n\n".join(lines)


def _empty_report(project_name: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""# 学术文章：{project_name}

> 生成时间：{now} | 状态：空项目

---

## 摘要

暂无对话记录，无法生成学术文章。

## 引言

暂无足够信息。

## 相关研究

暂无足够信息。

## 技术方案

暂无足够信息。

## 分析与讨论

暂无足够信息。

## 结论

暂无足够信息。

## 参考文献

暂无引用来源。
"""


def _fallback_report(project_name: str, history: list[dict], summary: str) -> str:
    """LLM 调用失败时的模板化回退报告。"""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    user_msgs = [m for m in history if m["role"] == "user"]
    assistant_msgs = [m for m in history if m["role"] == "assistant"]

    return f"""# 学术文章：{project_name}

> 生成时间：{now} | 模式：基础报告（LLM 生成失败，使用模板）

---

## 摘要

{summary if summary else "暂无摘要。"}

## 引言

{_extract_questions(user_msgs) if user_msgs else "暂无足够信息。"}

## 相关研究

{_extract_conclusions(assistant_msgs) if assistant_msgs else "暂无足够信息。"}

## 技术方案

{_extract_timeline(history)}

## 分析与讨论

暂无足够信息（模板模式无法自动生成分析内容，请查看原始对话记录）。

## 结论

暂无足够信息（模板模式无法自动生成结论，请查看原始对话记录）。

## 参考文献

暂无足够信息（模板模式无法自动提取引用，请查看原始对话记录）。
"""


def _extract_timeline(history: list[dict]) -> str:
    """从历史对话中提取时间线概览。"""
    if not history:
        return "暂无对话记录。"

    lines = ["按时间顺序的讨论主题：\n"]
    for i, msg in enumerate(history, 1):
        role = "用户" if msg["role"] == "user" else "助手"
        content = msg.get("content", "")
        # 取前 80 字作为摘要
        preview = content[:80].replace("\n", " ") + ("..." if len(content) > 80 else "")
        lines.append(f"- **[{i}] {role}**: {preview}")
    return "\n".join(lines)


def _extract_conclusions(messages: list[dict]) -> str:
    """从助手消息中简单提取可能包含结论的片段。"""
    lines = ["从对话中提取的助手回复要点：\n"]
    for i, msg in enumerate(messages[-10:], 1):  # 只取最近 10 条
        content = msg.get("content", "")
        preview = content[:200].replace("\n", " ") + ("..." if len(content) > 200 else "")
        lines.append(f"- {preview}")
    return "\n".join(lines)


def _extract_questions(messages: list[dict]) -> str:
    """提取用户提出的问题。"""
    lines = ["用户提出的问题：\n"]
    for msg in messages:
        content = msg.get("content", "")
        preview = content[:200].replace("\n", " ")
        if len(content) > 200:
            preview += "..."
        lines.append(f"- {preview}")
    return "\n".join(lines)
