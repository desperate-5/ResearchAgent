"""对话摘要：压缩生成（写）与摘要注入（读）。

- compress_conversation: 后台压缩策略（阈值判断 + 窗口裁剪 + 生成 + 持久化）
- build_summary_injection: 读取项目最新摘要并格式化为注入上下文的文本
"""

import os

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from ..graph.prompts import COMPRESSION_TURN_THRESHOLD
from ..storage.records import get_summary, save_summary
from .windowing import count_turns, last_n_turns


COMPRESSION_PROMPT = """请将以下对话历史压缩为结构化摘要。保留所有关键信息，丢弃闲聊和冗余内容。

## 对话历史
{conversation}

{existing}

请严格按以下格式输出摘要（不超过 500 字）：

核心结论：[本轮研究达成的结论]
关键信息点：[重要的事实、数据、引用]
未解决问题：[用户尚未得到答案的问题]
用户偏好：[用户的习惯和偏好]"""


def _get_compression_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "deepseek-chat"),
        api_key=os.getenv("OPENAI_API_KEY", "sk-xxx"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
        temperature=0.3,
        streaming=False,
    )


def _format_messages(messages: list) -> str:
    lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"用户: {m.content}")
        elif isinstance(m, AIMessage):
            content = m.content or ""
            if m.tool_calls:
                tools = [t["name"] for t in m.tool_calls]
                content = f"[调用工具: {', '.join(tools)}] {content}"
            lines.append(f"助手: {content}")
        elif isinstance(m, SystemMessage):
            pass
        else:
            role = getattr(m, "role", "unknown")
            content = getattr(m, "content", "")
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


async def generate_summary(messages: list, existing_summary: str = "") -> str:
    """将消息列表压缩为结构化摘要。返回新摘要字符串。"""
    if not messages:
        return existing_summary or ""

    llm = _get_compression_llm()
    conversation = _format_messages(messages)

    existing = ""
    if existing_summary:
        existing = f"## 现有摘要\n{existing_summary}\n\n请将上方对话历史融合进现有摘要，更新各字段内容。"

    prompt = COMPRESSION_PROMPT.format(conversation=conversation, existing=existing)
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return response.content


def build_summary_injection(project_id: str) -> str:
    """读取项目最新对话摘要并格式化为注入文本；无摘要返回空字符串。"""
    summary = get_summary(project_id)
    if not summary:
        return ""
    return f"## 历史对话摘要\n{summary}"


async def compress_conversation(project_id: str, messages: list, existing_summary: str = "") -> None:
    """对话压缩策略：超过阈值时把旧消息压缩为摘要并持久化。

    内部吞掉异常，保证压缩失败不影响主流程（供后台任务调用）。
    """
    try:
        if count_turns(messages) <= COMPRESSION_TURN_THRESHOLD:
            return
        recent_msgs = last_n_turns(messages, 5)
        old_msgs = messages[: len(messages) - len(recent_msgs)]
        if not old_msgs:
            return
        new_summary = await generate_summary(old_msgs, existing_summary)
        save_summary(project_id, new_summary)
    except Exception:
        pass  # 压缩失败不影响主流程
