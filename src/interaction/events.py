"""把 interrupt payload 翻译为前端 SSE 事件（纯函数）。"""


def to_sse_event(payload, message_index: int | None = None) -> dict | None:
    """把图节点的 interrupt payload 翻译为前端 SSE 事件字典。

    不认识或非交互 payload 返回 None。
    """
    if not isinstance(payload, dict):
        return None

    itype = payload.get("type", "")
    if itype == "query_clarification":
        ev = {
            "type": "query_clarification",
            "directions": payload.get("directions", []),
        }
    elif itype == "plan_options":
        ev = {
            "type": "plan_options",
            "options": payload.get("options", []),
        }
    else:
        return None

    if message_index is not None:
        ev["message_index"] = message_index
    return ev


def iter_interrupt_events(interrupts, message_index: int) -> list[dict]:
    """从 graph_state.interrupts 逐个翻译为 SSE 事件（跳过非交互项）。"""
    events = []
    for it in interrupts:
        val = getattr(it, "value", it)
        ev = to_sse_event(val, message_index)
        if ev:
            events.append(ev)
    return events
