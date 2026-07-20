from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    project_id: str = Field(..., description="项目 ID")
    message: str = Field(..., description="用户当前消息")
    tools: list[str] = Field(default=[], description="用户选择的工具列表，如 ['web_search', 'python_executor']")


class CreateProjectRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200, description="项目名称")


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
    LiteraturePref, WritingPref, ExperimentPref, ToolPref, PreferencesConfig,
)


class UpdatePreferencesRequest(BaseModel):
    literature: LiteraturePref | None = None
    writing: WritingPref | None = None
    experiment: ExperimentPref | None = None
    tool: ToolPref | None = None


# ---- 反馈 ----

class FeedbackRequest(BaseModel):
    project_id: str = Field(..., description="项目 ID")
    type: str = Field(..., description="like / dislike")
    tag: str = Field(default="", description="快捷标签，如 '太啰嗦'、'需要英文文献'")
    comment: str = Field(default="", description="自由文本补充")
