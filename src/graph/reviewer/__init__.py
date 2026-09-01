"""评审节点领域逻辑包。

包含检索质量评估的实现：规则信号、评分聚合、LLM 语义评估。
对外暴露 assess_sources 编排入口，供 nodes.py 的 reviewer_node 调用。
"""

from .rules import assess_authority, assess_timeliness
from .scoring import compute_score, map_credibility, build_evidence, NEUTRAL
from .llm_assess import get_llm, assess_relevance, assess_global
from .schemas import (
    DimensionScores,
    SourceAssessment,
    ReviewerOutput,
    RelevanceResult,
    ConsistencyEntry,
    GlobalAssessment,
)


async def assess_sources(state: dict, user_query: str = "") -> ReviewerOutput:
    """检索质量评估编排入口：规则信号 + LLM 两阶段语义评估 → 评分卡 + 小结 + 缺口。

    - 权威性 / 时效性：纯规则信号（可复现、零 LLM 成本）
    - 相关性：阶段 1 N 次并行 LLM 调用
    - 一致性 + 缺口 + 小结：阶段 2 一次全局 LLM 调用
    """
    sources = state.get("reference_sources", [])
    if not sources:
        return ReviewerOutput()

    llm = get_llm()

    # 规则信号
    authority = {s.get("source_number"): assess_authority(s) for s in sources}
    timeliness = {s.get("source_number"): assess_timeliness(s, user_query) for s in sources}

    # LLM 阶段 1 / 阶段 2
    relevance = await assess_relevance(llm, sources, user_query)
    global_result = await assess_global(llm, sources, user_query)

    consistency = {e.source_number: float(e.consistency) for e in global_result.consistency}

    assessments: list[SourceAssessment] = []
    for s in sources:
        num = s.get("source_number", 0)
        dims = {
            "authority": float(authority.get(num, 3)),
            "timeliness": float(timeliness.get(num, 3)),
            "relevance": float(relevance.get(num, (NEUTRAL, ""))[0]),
            "consistency": float(consistency.get(num, NEUTRAL)),
        }
        score = compute_score(dims)
        assessments.append(SourceAssessment(
            source_number=num,
            dimension_scores=DimensionScores(**dims),
            score=score,
            credibility=map_credibility(score),
            evidence=build_evidence(s, dims),
        ))

    return ReviewerOutput(
        assessments=assessments,
        summary=global_result.summary,
        gaps=list(global_result.gaps),
        needs_refetch=bool(global_result.needs_refetch),
    )


__all__ = [
    "assess_sources",
    "DimensionScores",
    "SourceAssessment",
    "ReviewerOutput",
    "RelevanceResult",
    "ConsistencyEntry",
    "GlobalAssessment",
    "assess_authority",
    "assess_timeliness",
    "compute_score",
    "map_credibility",
    "build_evidence",
]
