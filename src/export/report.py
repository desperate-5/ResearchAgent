import os
from datetime import datetime, timezone

from langchain_openai import ChatOpenAI

from ..memory.store import get_history, get_summary
from ..projects.manager import get_project


REPORT_SYSTEM_PROMPT = """你是一个科研报告生成助手。根据用户提供的对话历史，生成一份结构化的 Markdown 研究报告。

报告必须使用中文，严格按以下四个章节组织：

## 一、研究脉络
- 按时间顺序梳理本次研究中讨论的主题演变
- 说明各阶段的研究重点和转折点
- 突出关键的研究思路推进

## 二、已确认结论
- 列出对话中已经明确的结论和发现
- 每条结论用要点列出，注明依据来源
- 区分"高置信度"和"中等置信度"

## 三、待解决问题
- 列出尚未解决或需要进一步研究的问题
- 说明每个问题的难度和优先级
- 如有可能，给出建议的研究方向

## 四、引用来源
- 汇总对话中提及的论文、文章、数据来源
- 尽可能给出完整信息（标题、作者、年份、链接）
- 标注每个来源与哪个结论相关

要求：
- 基于对话内容，不要编造不存在的信息
- 格式整洁，使用 Markdown 标题、列表、表格
- 如果对话中某部分内容不足，请注明"暂无足够信息"而不是编造"""


async def generate_report(project_id: str) -> str:
    """从对话历史生成结构化 Markdown 研究报告。"""
    project = get_project(project_id)
    project_name = project["name"] if project else "未知项目"

    history = get_history(project_id)
    summary = get_summary(project_id)

    # 将对话格式化为可读文本
    transcript = _format_transcript(history)

    if not transcript.strip():
        return _empty_report(project_name)

    # 调用 LLM 生成报告
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "deepseek-chat"),
        api_key=os.getenv("OPENAI_API_KEY", "sk-xxx"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
        temperature=0.3,
    )

    user_prompt = f"""## 项目名称
{project_name}

## 历史对话摘要
{summary if summary else "暂无"}

## 完整对话记录
{transcript}

请基于以上对话内容生成研究报告。"""

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

    header = f"""# 研究报告：{project_name}

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
    return f"""# 研究报告：{project_name}

> 生成时间：{now} | 状态：空项目

---

## 一、研究脉络

暂无对话记录，无法梳理研究脉络。

## 二、已确认结论

暂无足够信息。

## 三、待解决问题

暂无足够信息。

## 四、引用来源

暂无引用来源。
"""


def _fallback_report(project_name: str, history: list[dict], summary: str) -> str:
    """LLM 调用失败时的模板化回退报告。"""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # 统计基本信息
    user_msgs = [m for m in history if m["role"] == "user"]
    assistant_msgs = [m for m in history if m["role"] == "assistant"]

    return f"""# 研究报告：{project_name}

> 生成时间：{now} | 模式：基础报告（LLM 生成失败，使用模板）

---

## 一、研究脉络

{_extract_timeline(history)}

## 二、已确认结论

{_extract_conclusions(assistant_msgs) if assistant_msgs else "暂无足够信息。"}

## 三、待解决问题

{_extract_questions(user_msgs) if user_msgs else "暂无足够信息。"}

## 四、引用来源

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
