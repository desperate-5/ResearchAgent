"""对话窗口裁剪：按「轮」统计与切分消息列表。"""

from langchain_core.messages import SystemMessage, HumanMessage

from ..graph.prompts import MAX_CONTEXT_TURNS


def count_turns(messages: list) -> int:
    """统计对话轮数：一轮 = 一条用户消息。"""
    return sum(1 for m in messages if isinstance(m, HumanMessage))


def last_n_turns(messages: list, n: int) -> list:
    """取最近 n 轮的完整消息（按用户消息边界切分，保证轮次完整）。"""
    user_indices = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if len(user_indices) <= n:
        return messages
    return messages[user_indices[-n]:]


def get_recent_messages(state: dict) -> list:
    """获取最近的用户对话消息（不含 system message），最多 MAX_CONTEXT_TURNS 轮。"""
    all_messages = list(state["messages"])
    conv_msgs = [m for m in all_messages if not isinstance(m, SystemMessage)]
    return last_n_turns(conv_msgs, MAX_CONTEXT_TURNS)


def extract_user_query(state: dict) -> str:
    """从消息列表中提取最新的用户消息文本（作为检索 / 判断用的原始问题）。"""
    all_messages = list(state.get("messages", []))
    for m in reversed(all_messages):
        if hasattr(m, "type") and m.type == "human":
            content = m.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part.get("text", "")
    return ""
