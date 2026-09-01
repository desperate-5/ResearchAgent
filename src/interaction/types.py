"""交互 payload 类型定义（纯数据）。"""

from dataclasses import dataclass, field


@dataclass
class QueryClarificationPayload:
    """检索前澄清：用户问题模糊时抛给前端的澄清请求（2~3 个检索方向）。"""

    directions: list[str] = field(default_factory=list)
    type: str = "query_clarification"

    def to_dict(self) -> dict:
        return {"type": self.type, "directions": self.directions}


@dataclass
class PlanOptionsPayload:
    """方案选择：planner 生成候选方案后抛给前端供选择。"""

    options: list[dict] = field(default_factory=list)
    type: str = "plan_options"

    def to_dict(self) -> dict:
        return {"type": self.type, "options": self.options}
