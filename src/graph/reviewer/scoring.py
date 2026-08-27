"""评分聚合（纯函数，零 LLM 依赖）。

将各维度分加权聚合成综合分，映射到高/中/低可信度，并组装可解释评分卡。
"""

from .rules import extract_domain, is_authoritative, is_untrusted

# 维度权重：权威性 / 时效性 / 相关性 / 一致性（规则维度合计 0.5 托底）
DIMENSION_WEIGHTS = {
    "authority": 0.35,
    "timeliness": 0.15,
    "relevance": 0.30,
    "consistency": 0.20,
}

HIGH_THRESHOLD = 3.8
MID_THRESHOLD = 2.5

NEUTRAL = 2.5  # LLM 维度解析失败时的中性分


def compute_score(dim_scores: dict) -> float:
    """加权聚合综合分（0-5）。缺失维度取中性分。"""
    total = 0.0
    for dim, w in DIMENSION_WEIGHTS.items():
        total += w * float(dim_scores.get(dim, NEUTRAL))
    return round(total, 2)


def map_credibility(score: float) -> str:
    if score >= HIGH_THRESHOLD:
        return "高"
    if score >= MID_THRESHOLD:
        return "中"
    return "低"


def build_evidence(source: dict, dim_scores: dict) -> str:
    """组装可解释的证据描述，说明评分依据。"""
    parts: list[str] = []
    stype = source.get("source_type", "")
    if stype == "paper":
        parts.append("学术论文来源")
    elif stype == "document":
        parts.append("用户上传文档")

    domain = extract_domain(source.get("url", ""))
    if domain:
        if is_authoritative(domain):
            parts.append(f"域名 {domain} 权威白名单")
        elif is_untrusted(domain):
            parts.append(f"域名 {domain} 低可信")

    if dim_scores.get("relevance", NEUTRAL) >= 4:
        parts.append("与问题高度相关")
    elif dim_scores.get("relevance", NEUTRAL) <= 2:
        parts.append("与问题相关性偏低")

    if dim_scores.get("consistency", NEUTRAL) < 2.5:
        parts.append("与其他来源存在冲突")

    return "；".join(parts) or "无明显特征"
