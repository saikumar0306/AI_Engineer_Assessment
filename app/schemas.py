from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question must not be empty.")
        return value


class QueryResponse(BaseModel):
    question: str
    answer: str
    result: Any


class AnomalyRecord(BaseModel):
    ticket_id: str
    anomaly_type: str
    reason: str
    resolution_time_hrs: float | None = None
    priority: str | None = None
    status: str | None = None
    created_at: datetime | None = None


class AnomaliesResponse(BaseModel):
    total: int
    anomalies: list[AnomalyRecord]