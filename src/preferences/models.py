import json

from pydantic import BaseModel, Field


class LiteraturePref(BaseModel):
    """文献检索偏好"""
    source_type: str = Field(default="", description="journal / conference / both")
    year_start: int = Field(default=0, description="最早年份")
    year_end: int = Field(default=0, description="最晚年份")
    paper_type: str = Field(default="", description="review / experimental / both")
    min_citations: int = Field(default=0, description="最低引用量")
    preferred_venues: list[str] = Field(default_factory=list, description="偏好期刊/会议")
    preferred_language: str = Field(default="", description="chinese / english / both")


class WritingPref(BaseModel):
    """论文写作偏好"""
    sentence_style: str = Field(default="", description="concise / elaborate")
    figure_norm: str = Field(default="", description="tight / spacious")
    abstract_style: str = Field(default="", description="structured / narrative")
    ref_format: str = Field(default="GB/T 7714", description="APA / IEEE / GB/T 7714")
    lang: str = Field(default="chinese", description="chinese / english")


class ExperimentPref(BaseModel):
    """实验分析偏好"""
    metrics: list[str] = Field(default_factory=list, description="评估指标：accuracy, F1, AUC 等")
    require_control: bool = Field(default=False, description="需要对照组")
    significance_test: bool = Field(default=False, description="需要显著性检验")
    require_ablation: bool = Field(default=False, description="需要消融实验")


class PreferencesConfig(BaseModel):
    """项目完整的偏好配置"""
    literature: LiteraturePref = Field(default_factory=LiteraturePref)
    writing: WritingPref = Field(default_factory=WritingPref)
    experiment: ExperimentPref = Field(default_factory=ExperimentPref)


# ============================================================
# 偏好进化：学习层 / 证据层模型（§五 四层结构）
# ============================================================

# 学习层可识别的维度及其合法取值（None = 自由文本，如 domain/method）
KNOWN_DIMENSIONS: dict[str, set | None] = {
    "writing.sentence_style": {"concise", "elaborate"},
    "writing.figure_norm": {"tight", "spacious"},
    "writing.abstract_style": {"structured", "narrative"},
    "writing.ref_format": {"GB/T 7714", "APA", "IEEE"},
    "writing.lang": {"chinese", "english"},
    "literature.source_type": {"journal", "conference", "both"},
    "literature.paper_type": {"review", "experimental", "both"},
    "literature.preferred_language": {"chinese", "english", "both"},
    "domain": None,
    "method": None,
}


class PreferenceCandidate(BaseModel):
    """LLM 提取出的单条偏好候选（陈述式显式 / 选择式显式）。"""
    dimension: str
    value: str
    evidence: str = ""


class ObservedCandidate(BaseModel):
    """会话蒸馏产出的隐性偏好候选（观察式，带多条原句证据）。"""
    dimension: str
    value: str
    evidence: list[str] = Field(default_factory=list)


class ProfileItemRow(BaseModel):
    """学习层 profile_items 表的一行（Beta 账本：μ = a/(a+b)）。"""
    id: int | None = None
    scope: str = "global"          # global | project
    project_id: str = ""           # scope=project 时关联项目
    dimension: str
    value: str
    a: float = 1.0
    b: float = 1.0
    source: str = "explicit"       # explicit | choice | observed | manual
    applied: int = 0               # 当前是否生效
    user_locked: int = 0           # 手动层锁定保护
    last_seen: str = ""            # 衰减计时
    evidence_json: str = "[]"

    @property
    def mu(self) -> float:
        s = self.a + self.b
        return self.a / s if s > 0 else 0.0

    @property
    def evidence_list(self) -> list[str]:
        try:
            data = json.loads(self.evidence_json or "[]")
            if isinstance(data, list):
                return [str(x) for x in data]
        except Exception:
            pass
        return []
