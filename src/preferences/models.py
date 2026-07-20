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
    viz_tool: str = Field(default="", description="matplotlib / seaborn / plotly")
    significance_test: bool = Field(default=False, description="需要显著性检验")
    require_ablation: bool = Field(default=False, description="需要消融实验")


class ToolPref(BaseModel):
    """工具调用偏好"""
    prefer_python: bool = Field(default=False, description="优先用 Python 绘图")
    prefer_arxiv: bool = Field(default=False, description="优先检索 arXiv")
    avoid_cnki: bool = Field(default=False, description="避免使用知网")
    search_priority: str = Field(default="", description="arxiv / semantic_scholar / web")
    plot_library: str = Field(default="", description="matplotlib / seaborn / plotly")


class PreferencesConfig(BaseModel):
    """项目完整的偏好配置"""
    literature: LiteraturePref = Field(default_factory=LiteraturePref)
    writing: WritingPref = Field(default_factory=WritingPref)
    experiment: ExperimentPref = Field(default_factory=ExperimentPref)
    tool: ToolPref = Field(default_factory=ToolPref)
