from pydantic import BaseModel, Field
from typing import Any


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None


class ChartSpec(BaseModel):
    chart_type: str
    title: str
    data: dict[str, Any]


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sql: str | None = None
    confidence: float
    charts: list[ChartSpec] = []
    table: list[dict[str, Any]] = []
    suggestions: list[str] = []
