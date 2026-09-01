"""评审结果的 Pydantic 模型定义。

包含 SourceAssessment / ReviewerOutput 等结构化类型，用于 with_structured_output 的强类型输出。
"""

from pydantic import BaseModel, Field


class DimensionScores(BaseModel):
    """四个评估维度的 0-5 分。权威性/时效性由规则信号给出，相关性/一致性由 LLM 给出。"""

    authority: float = Field(ge=0, le=5, description="权威性分（规则信号）")
    timeliness: float = Field(ge=0, le=5, description="时效性分（规则信号）")
    relevance: float = Field(ge=0, le=5, description="相关性分（LLM）")
    consistency: float = Field(ge=0, le=5, description="一致性分（LLM）")


class SourceAssessment(BaseModel):
    """单条来源的可解释评分卡。"""

    source_number: int
    dimension_scores: DimensionScores
    score: float
    credibility: str  # 高 / 中 / 低
    evidence: str = ""


class ReviewerOutput(BaseModel):
    """reviewer 的整体输出：评分卡 + 质量小结 + 缺口列表。"""

    assessments: list[SourceAssessment] = Field(default_factory=list)
    summary: str = ""
    gaps: list[str] = Field(default_factory=list)
    needs_refetch: bool = False  # 是否明显不足以回答问题、需要补搜（supervisor 据此门控）


# ------------------------------------------------------------
# LLM 结构化输出子模型（用于 with_structured_output）
# ------------------------------------------------------------

class RelevanceResult(BaseModel):
    """阶段 1：单条来源 vs 用户问题的相关性。"""

    relevance: int = Field(ge=0, le=5, description="相关性 0-5 分")
    reason: str = Field(default="", description="一句话理由")


class ConsistencyEntry(BaseModel):
    """阶段 2：单条来源与其他来源的结论一致性。"""

    source_number: int
    consistency: int = Field(ge=0, le=5, description="一致性 0-5 分")


class GlobalAssessment(BaseModel):
    """阶段 2：全局一致性 + 缺口列表 + 质量小结 + 是否需补搜。"""

    consistency: list[ConsistencyEntry] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    summary: str = ""
    needs_refetch: bool = False  # 这批来源是否明显不足、必须补搜
