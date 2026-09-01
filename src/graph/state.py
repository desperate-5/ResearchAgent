from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    project_id: str
    summary: str
    system_prompt: str

    # 多 Agent 新增字段
    agent_outputs: dict         # {agent_name: output_text}  各子 agent 的输出
    next_agent: str             # supervisor 决定的下一个 agent 或 "FINISH"
    supervisor_log: list[dict]  # supervisor 的调度记录 [{next, reason}, ...]
    required_tools: list[str]   # 用户指定的工具列表，supervisor 需确保调度对应 agent
    reference_sources: list[dict]  # 本轮从工具输出中解析的来源（含全局编号）

    # reviewer 来源评级字段
    source_ratings: list[dict]   # reviewer 输出的来源可信度评级 [{source_number, credibility, reason}, ...]

    # reviewer 检索质量评估字段（功能点 1）
    source_assessments: list[dict]  # 每条来源评分卡 [{source_number, dimension_scores, score, credibility, evidence}]
    retrieval_gaps: list[str]       # 信息缺口子问题列表
    needs_refetch: bool             # reviewer 是否判定来源明显不足、需要补搜
    search_round: int               # 已补搜次数（0 表示首轮，每次补搜 +1）

    # planner 人机协同字段
    plan_options: list[dict]      # planner 生成的候选方案 [{id, title, description, pros, cons}, ...]
    chosen_plan_id: str            # 用户选择的预制方案 ID（空字符串表示未选预制方案）
    custom_plan_text: str          # 用户自定义的方案文本（未选预制方案时填写）

    # 人机交互字段
    effective_query: str           # 澄清后的有效检索查询（空 = 使用原始输入）
    was_clarified: bool            # 本轮是否经过了检索前澄清（用户选定了专业方向）
    has_prior_research: bool       # 项目是否已有历史研究内容（持久化来源 / 摘要）
    query_invalid: bool            # 输入是否无效（乱码/废话），为真时跳过检索直接结束
