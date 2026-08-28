"""来源可信度重排。"""


def rerank_sources(reference_sources: list[dict], source_assessments: list[dict]) -> list[dict]:
    """按可信度重排来源：高可信置顶、低可信沉底，无评分卡的来源取中性分保持原位。

    重排分 = (综合分 score + 相关性维度分 relevance) / 10，两者均 0-5。
    使用稳定排序，同分来源保持原相对顺序。
    """
    if not source_assessments:
        return list(reference_sources)

    by_num = {a.get("source_number"): a for a in source_assessments}

    def _rank(s: dict) -> float:
        a = by_num.get(s.get("source_number"))
        if not a:
            return 0.5
        score = float(a.get("score", 2.5))
        dims = a.get("dimension_scores") or {}
        relevance = float(dims.get("relevance", 2.5))
        return (score + relevance) / 10.0

    return sorted(reference_sources, key=_rank, reverse=True)
