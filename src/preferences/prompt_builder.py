from .models import PreferencesConfig, LiteraturePref, WritingPref, ExperimentPref, ToolPref


def build_preference_prompt(prefs: PreferencesConfig) -> str:
    """将结构化偏好配置转为自然语言指令，拼入系统提示词。"""
    sections = []

    lit = _format_literature(prefs.literature)
    if lit:
        sections.append(f"文献检索偏好：{lit}")

    wrt = _format_writing(prefs.writing)
    if wrt:
        sections.append(f"论文写作偏好：{wrt}")

    exp = _format_experiment(prefs.experiment)
    if exp:
        sections.append(f"实验分析偏好：{exp}")

    tool = _format_tool(prefs.tool)
    if tool:
        sections.append(f"工具偏好：{tool}")

    if not sections:
        return ""

    return "## 用户偏好\n" + "\n".join(sections)


# ---- 各分类的格式化函数 ----


def _format_literature(p: LiteraturePref) -> str:
    parts = []
    if p.source_type:
        label = {"journal": "期刊论文", "conference": "会议论文", "both": "期刊和会议论文"}
        parts.append(f"优先检索{label.get(p.source_type, p.source_type)}")
    if p.year_start and p.year_end:
        parts.append(f"时间范围 {p.year_start}-{p.year_end} 年")
    elif p.year_start:
        parts.append(f"不早于 {p.year_start} 年")
    elif p.year_end:
        parts.append(f"不晚于 {p.year_end} 年")
    if p.paper_type:
        label = {"review": "综述", "experimental": "实验论文", "both": "综述和实验论文"}
        parts.append(f"偏好{label.get(p.paper_type, p.paper_type)}")
    if p.min_citations:
        parts.append(f"引用量不低于 {p.min_citations}")
    if p.preferred_venues:
        parts.append(f"重点关注 {'、'.join(p.preferred_venues)}")
    if p.preferred_language:
        label = {"chinese": "中文", "english": "英文", "both": "中英文"}
        parts.append(f"语言偏好：{label.get(p.preferred_language, p.preferred_language)}")

    return "；".join(parts)


def _format_writing(p: WritingPref) -> str:
    parts = []
    if p.sentence_style:
        label = {"concise": "简洁精炼，避免冗余", "elaborate": "详细展开，充分解释"}
        parts.append(label.get(p.sentence_style, p.sentence_style))
    if p.figure_norm:
        label = {"tight": "图表紧凑，信息密度高", "spacious": "图表宽松，留白充足"}
        parts.append(label.get(p.figure_norm, p.figure_norm))
    if p.abstract_style:
        label = {"structured": "结构化摘要（目的-方法-结果-结论）", "narrative": "叙述式摘要"}
        parts.append(f"摘要风格：{label.get(p.abstract_style, p.abstract_style)}")
    if p.ref_format:
        parts.append(f"参考文献格式：{p.ref_format}")
    if p.lang:
        parts.append(f"回答语言：{'中文' if p.lang == 'chinese' else '英文'}")
    return "；".join(parts)


def _format_experiment(p: ExperimentPref) -> str:
    parts = []
    if p.metrics:
        parts.append(f"评估指标：{'、'.join(p.metrics)}")
    if p.require_control:
        parts.append("必须包含对照组")
    if p.viz_tool:
        parts.append(f"图表工具：{p.viz_tool}")
    if p.significance_test:
        parts.append("需要显著性检验")
    if p.require_ablation:
        parts.append("需要消融实验")
    return "；".join(parts)


def _format_tool(p: ToolPref) -> str:
    parts = []
    if p.prefer_python:
        parts.append("优先使用 Python 绘图")
    if p.prefer_arxiv:
        parts.append("优先检索 arXiv 学术论文库")
    if p.avoid_cnki:
        parts.append("避免使用知网检索")
    if p.search_priority:
        parts.append(f"搜索优先级：{p.search_priority}")
    if p.plot_library:
        parts.append(f"绑图库：{p.plot_library}")
    return "；".join(parts)
