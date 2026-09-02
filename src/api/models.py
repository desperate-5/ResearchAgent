from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    project_id: str = Field(..., description="项目 ID")
    message: str = Field(..., description="用户当前消息")
    tools: list[str] = Field(default=[], description="用户选择的工具列表，如 ['web_search', 'aminer_search_papers']")


class CreateProjectRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200, description="项目名称")


class RenameProjectRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200, description="新项目名称")


class ProjectResponse(BaseModel):
    id: str
    name: str
    created_at: str
    updated_at: str


class HistoryMessage(BaseModel):
    role: str
    content: str
    created_at: str


class ProjectHistory(BaseModel):
    project_id: str
    messages: list[HistoryMessage]


# ---- 偏好 ----

from ..preferences.models import (  # noqa: E402
    LiteraturePref, WritingPref, ExperimentPref, PreferencesConfig,
)


class UpdatePreferencesRequest(BaseModel):
    literature: LiteraturePref | None = None
    writing: WritingPref | None = None
    experiment: ExperimentPref | None = None


# ---- 原始偏好文件 ----

class RawPreferencesRequest(BaseModel):
    content: str = Field(..., min_length=0, description="preferences.md 的完整原始内容")


# ---- 反馈 ----

class FeedbackRequest(BaseModel):
    type: str = Field(..., description="like / dislike")
    tag: str = Field(default="", description="快捷标签，如 '太啰嗦'、'需要英文文献'")
    comment: str = Field(default="", description="自由文本补充")


# ---- 人机交互恢复（通用） ----

class ResumeRequest(BaseModel):
    project_id: str = Field(..., description="项目 ID")
    type: str = Field(default="plan_options", description="交互类型：plan_options / query_clarification")
    # plan_options
    chosen_plan_id: str = Field(default="", description="用户选择的预制方案 ID")
    custom_plan_text: str = Field(default="", description="用户自定义的方案文本")
    plan_title: str = Field(default="", description="所选预制方案的标题（证据采集）")
    plan_type: str = Field(default="", description="所选方案的类别/类型（证据采集）")
    # query_clarification
    selected_direction: str = Field(default="", description="用户选择的澄清方向")
    use_original: bool = Field(default=False, description="是否按原始问题检索")
