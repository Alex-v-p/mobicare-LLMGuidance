from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class PatientVariables(BaseModel):
    values: Dict[str, Any] = Field(default_factory=dict)


class GenerationOptions(BaseModel):
    use_fake_rag: bool = True
    temperature: float = 0.2
    max_tokens: int = 256


class GuidanceRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    question: str
    patient: PatientVariables = Field(default_factory=PatientVariables)
    options: GenerationOptions = Field(default_factory=GenerationOptions)


class RetrievedContext(BaseModel):
    source_id: str
    title: str
    snippet: str


class InferenceRequest(BaseModel):
    request_id: str
    question: str
    patient_variables: Dict[str, Any] = Field(default_factory=dict)
    retrieved_context: List[RetrievedContext] = Field(default_factory=list)
    options: GenerationOptions = Field(default_factory=GenerationOptions)


class InferenceResponse(BaseModel):
    request_id: str
    status: str
    model: str
    answer: str
    retrieved_context: List[RetrievedContext] = Field(default_factory=list)
    used_variables: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ApiGuidanceResponse(BaseModel):
    request_id: str
    status: str
    answer: str
    model: str
    rag: List[RetrievedContext] = Field(default_factory=list)
    used_variables: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OllamaGenerateResponse(BaseModel):
    model: Optional[str] = None
    response: str = ""
    done: bool = True
    done_reason: Optional[str] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None


JobState = Literal["queued", "running", "completed", "failed", "not_found"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobAcceptedResponse(BaseModel):
    request_id: str
    status: Literal["queued"] = "queued"
    status_url: str


class JobRecord(BaseModel):
    request_id: str
    status: JobState
    request: InferenceRequest
    result: Optional[InferenceResponse] = None
    error: Optional[str] = None
    created_at: str = Field(default_factory=utc_now_iso)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    updated_at: str = Field(default_factory=utc_now_iso)


class ApiGuidanceJobStatus(BaseModel):
    request_id: str
    status: JobState
    answer: Optional[str] = None
    model: Optional[str] = None
    rag: List[RetrievedContext] = Field(default_factory=list)
    used_variables: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    updated_at: Optional[str] = None
