"""把 /chat/resume 请求体解析为图节点 interrupt() 的返回值（纯函数）。"""


def parse_resume(body) -> dict:
    """把 resume 请求体解析为 interrupt() 的返回值 dict。

    兼容统一 ResumeRequest（含 type 字段）与旧 PlanResumeRequest 字段。
    """
    itype = (getattr(body, "type", None) or "plan_options").strip()

    if itype == "query_clarification":
        return {
            "selected_direction": getattr(body, "selected_direction", "") or "",
            "use_original": bool(getattr(body, "use_original", False)),
        }

    # 默认 plan_options
    return {
        "chosen_plan_id": getattr(body, "chosen_plan_id", "") or "",
        "custom_plan_text": getattr(body, "custom_plan_text", "") or "",
    }
