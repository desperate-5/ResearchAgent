import json
import os
import yaml
from .models import PreferencesConfig, LiteraturePref, WritingPref, ExperimentPref

PREFERENCES_FILE = os.path.join("data", "preferences.md")

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
}


def _generate_default_md() -> str:
    """生成带 YAML frontmatter 和中文说明的默认 preferences.md 内容。"""
    default_prefs = PreferencesConfig()
    yaml_block = yaml.dump(
        default_prefs.model_dump(exclude_defaults=False),
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    ).strip()

    return f"""---
{yaml_block}
---

# 科研助手偏好配置

本文件用于配置科研助手的行为偏好。编辑上方 YAML 区域即可定制助手行为。

## 文献检索偏好 (literature)

| 字段 | 类型 | 说明 |
|------|------|------|
| `source_type` | string | 文献来源：`journal`（期刊）、`conference`（会议）、`""`（不限） |
| `year_start` | int | 最早检索年份，0 表示不限 |
| `year_end` | int | 最晚检索年份，0 表示不限 |
| `paper_type` | string | 论文类型：`review`（综述）、`experimental`（实验）、`""`（不限） |
| `min_citations` | int | 最低引用量，0 表示不限 |
| `preferred_venues` | list | 偏好期刊/会议名称，如 `["NeurIPS", "ICML"]` |
| `preferred_language` | string | 语言偏好：`chinese`（中文）、`english`（英文）、`""`（不限） |

## 论文写作偏好 (writing)

| 字段 | 类型 | 说明 |
|------|------|------|
| `sentence_style` | string | 文风：`concise`（简洁）、`elaborate`（详细）、`""`（默认） |
| `figure_norm` | string | 图表密度：`tight`（紧凑）、`spacious`（宽松）、`""`（默认） |
| `abstract_style` | string | 摘要风格：`structured`（结构化）、`narrative`（叙述式）、`""`（默认） |
| `ref_format` | string | 参考文献格式：`GB/T 7714`、`APA`、`IEEE` |
| `lang` | string | 回答语言：`chinese`（中文）、`english`（英文） |

## 实验分析偏好 (experiment)

| 字段 | 类型 | 说明 |
|------|------|------|
| `metrics` | list | 关注指标，如 `["accuracy", "F1"]` |
| `require_control` | bool | 是否要求对照组 |
| `significance_test` | bool | 是否要求显著性检验 |
| `require_ablation` | bool | 是否要求消融实验 |
"""


def _read_preferences_file() -> tuple[dict, str]:
    """读取 preferences.md，返回 (yaml_dict, body_markdown)。
    文件不存在时自动创建并返回默认值。"""
    if not os.path.exists(PREFERENCES_FILE):
        md = _generate_default_md()
        os.makedirs(os.path.dirname(PREFERENCES_FILE), exist_ok=True)
        with open(PREFERENCES_FILE, "w", encoding="utf-8") as f:
            f.write(md)
        return PreferencesConfig().model_dump(exclude_defaults=False), ""

    with open(PREFERENCES_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    # 解析 YAML frontmatter（--- 分隔符之间）
    parts = content.split("---", 2)
    if len(parts) >= 3:
        yaml_str = parts[1].strip()
        body = parts[2].strip()
    elif len(parts) == 2:
        yaml_str = parts[1].strip()
        body = ""
    else:
        yaml_str = ""
        body = content.strip()

    try:
        yaml_dict = yaml.safe_load(yaml_str) or {}
    except yaml.YAMLError:
        yaml_dict = {}

    return yaml_dict, body


def _write_preferences_file(prefs: PreferencesConfig):
    """将 PreferencesConfig 写回 preferences.md，保留 markdown body 不变。"""
    _, body = _read_preferences_file()

    yaml_block = yaml.dump(
        prefs.model_dump(exclude_defaults=False),
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    ).strip()

    content = f"---\n{yaml_block}\n---\n\n{body}"
    with open(PREFERENCES_FILE, "w", encoding="utf-8") as f:
        f.write(content)


def get_preferences() -> PreferencesConfig:
    """获取全局用户偏好配置，不存在则返回默认值。"""
    yaml_dict, _ = _read_preferences_file()

    def _build(model_cls, key: str):
        data = yaml_dict.get(key, {})
        if not isinstance(data, dict):
            data = {}
        try:
            return model_cls(**data)
        except Exception:
            return model_cls()

    return PreferencesConfig(
        literature=_build(LiteraturePref, "literature"),
        writing=_build(WritingPref, "writing"),
        experiment=_build(ExperimentPref, "experiment"),
    )


def save_preferences(prefs: PreferencesConfig):
    """保存全局用户偏好配置到 preferences.md。"""
    _write_preferences_file(prefs)


def get_raw_preferences() -> str:
    """返回 preferences.md 的完整原始内容（用于设置编辑器）。"""
    if not os.path.exists(PREFERENCES_FILE):
        return _generate_default_md()
    with open(PREFERENCES_FILE, "r", encoding="utf-8") as f:
        return f.read()


def save_raw_preferences(raw_content: str) -> PreferencesConfig | None:
    """保存原始 markdown 内容到 preferences.md。
    返回解析后的 PreferencesConfig，YAML 格式错误时返回 None（但文件仍会保存）。"""
    with open(PREFERENCES_FILE, "w", encoding="utf-8") as f:
        f.write(raw_content)

    try:
        return get_preferences()
    except Exception:
        return None


def apply_feedback(tag: str) -> PreferencesConfig | None:
    """根据反馈标签调整全局偏好。返回调整后的配置，无匹配规则时返回 None。"""
    if tag not in FEEDBACK_RULES:
        return None

    category, field, value = FEEDBACK_RULES[tag]
    prefs = get_preferences()

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
    save_preferences(prefs)
    return prefs
