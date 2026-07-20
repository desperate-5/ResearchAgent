import json
from ..memory.store import _get_conn
from .models import PreferencesConfig, LiteraturePref, WritingPref, ExperimentPref, ToolPref

# 反馈标签 → (偏好类别, 字段名, 新值)
FEEDBACK_RULES: dict[str, tuple[str, str, object]] = {
    # writing
    "太啰嗦": ("writing", "sentence_style", "concise"),
    "不够简洁": ("writing", "sentence_style", "concise"),
    "不够详细": ("writing", "sentence_style", "elaborate"),
    "太简略": ("writing", "sentence_style", "elaborate"),
    "图表太密": ("writing", "figure_norm", "spacious"),
    "图表太疏": ("writing", "figure_norm", "tight"),
    "摘要太抽象": ("writing", "abstract_style", "structured"),
    "需要结构化摘要": ("writing", "abstract_style", "structured"),
    "参考文献格式不对": ("writing", "ref_format", "GB/T 7714"),
    # literature
    "引用太旧": ("literature", "year_start", "recent"),
    "需要最新文献": ("literature", "year_start", "recent"),
    "需要中文文献": ("literature", "preferred_language", "chinese"),
    "需要英文文献": ("literature", "preferred_language", "english"),
    "优先会议论文": ("literature", "source_type", "conference"),
    "优先期刊论文": ("literature", "source_type", "journal"),
    "需要综述": ("literature", "paper_type", "review"),
    "需要实验论文": ("literature", "paper_type", "experimental"),
    # experiment
    "需要实验数据": ("experiment", "require_control", True),
    "需要对照组": ("experiment", "require_control", True),
    "需要显著性检验": ("experiment", "significance_test", True),
    "需要消融实验": ("experiment", "require_ablation", True),
    "需要统计分析": ("experiment", "significance_test", True),
    # tool
    "用 Python 画图": ("tool", "prefer_python", True),
    "优先 arxiv": ("tool", "prefer_arxiv", True),
    "少用知网": ("tool", "avoid_cnki", True),
    "优先学术搜索": ("tool", "prefer_arxiv", True),
    "用 matplotlib": ("tool", "plot_library", "matplotlib"),
    "用 seaborn": ("tool", "plot_library", "seaborn"),
    "用 plotly": ("tool", "plot_library", "plotly"),
}


def get_preferences(project_id: str) -> PreferencesConfig:
    """获取项目的偏好配置，不存在则返回默认值。"""
    conn = _get_conn()
    row = conn.execute(
        "SELECT literature, writing, experiment, tool "
        "FROM project_preferences WHERE project_id = ?",
        (project_id,),
    ).fetchone()
    conn.close()

    if row is None:
        return PreferencesConfig()

    def _parse(field: str, model_cls):
        raw = row[field] if row[field] else "{}"
        try:
            return model_cls(**json.loads(raw))
        except Exception:
            return model_cls()

    return PreferencesConfig(
        literature=_parse("literature", LiteraturePref),
        writing=_parse("writing", WritingPref),
        experiment=_parse("experiment", ExperimentPref),
        tool=_parse("tool", ToolPref),
    )


def save_preferences(project_id: str, prefs: PreferencesConfig):
    """保存或更新项目的偏好配置。"""
    conn = _get_conn()
    conn.execute(
        """INSERT INTO project_preferences (project_id, literature, writing, experiment, tool)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(project_id) DO UPDATE SET
           literature=excluded.literature,
           writing=excluded.writing,
           experiment=excluded.experiment,
           tool=excluded.tool""",
        (
            project_id,
            prefs.literature.model_dump_json(exclude_defaults=False),
            prefs.writing.model_dump_json(exclude_defaults=False),
            prefs.experiment.model_dump_json(exclude_defaults=False),
            prefs.tool.model_dump_json(exclude_defaults=False),
        ),
    )
    conn.commit()
    conn.close()


def apply_feedback(project_id: str, tag: str) -> PreferencesConfig | None:
    """根据反馈标签调整偏好。返回调整后的配置，无匹配规则时返回 None。"""
    if tag not in FEEDBACK_RULES:
        return None

    category, field, value = FEEDBACK_RULES[tag]
    prefs = get_preferences(project_id)

    cat_obj = getattr(prefs, category)

    if field == "year_start" and value == "recent":
        # "引用太旧" → 把 year_start 拉到近 3 年
        import datetime
        cat_obj.year_start = datetime.datetime.now().year - 3
    elif isinstance(value, list):
        current = getattr(cat_obj, field, [])
        if not isinstance(current, list):
            current = []
        for v in value:
            if v not in current:
                current.append(v)
        setattr(cat_obj, field, current)
    elif value == "toggle":
        current = getattr(cat_obj, field, False)
        setattr(cat_obj, field, not current)
    else:
        setattr(cat_obj, field, value)

    setattr(prefs, category, cat_obj)
    save_preferences(project_id, prefs)
    return prefs
