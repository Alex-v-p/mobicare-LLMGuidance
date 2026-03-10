from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, HttpUrl


def new_request_id() -> str:
    return f"req_{uuid4()}"


def new_job_id() -> str:
    return f"job_{uuid4()}"


class PatientVariables(BaseModel):
    values: Dict[str, Any] = Field(default_factory=dict)


class GenerationOptions(BaseModel):
    use_retrieval: bool = True
    top_k: int = 3
    temperature: float = 0.2
    max_tokens: int = 256
    use_example_response: bool = False
    retrieval_mode: Literal["dense", "hybrid"] = "hybrid"
    hybrid_dense_weight: float = 0.65
    hybrid_sparse_weight: float = 0.35
    use_graph_augmentation: bool = True
    graph_max_extra_nodes: int = 2
    enable_query_rewriting: bool = False
    enable_response_verification: bool = False
    enable_regeneration: bool = False
    max_regeneration_attempts: int = 1
    callback_url: Optional[HttpUrl] = None
    callback_headers: Dict[str, str] = Field(default_factory=dict)


class GuidanceRequest(BaseModel):
    request_id: str = Field(default_factory=new_request_id)
    question: str
    patient: PatientVariables = Field(default_factory=PatientVariables)
    options: GenerationOptions = Field(default_factory=GenerationOptions)


class RetrievedContext(BaseModel):
    source_id: str
    title: str
    snippet: str
    chunk_id: Optional[str] = None
    page_number: Optional[int] = None


class InferenceRequest(BaseModel):
    request_id: str
    question: str
    patient_variables: Dict[str, Any] = Field(default_factory=dict)
    retrieved_context: List[RetrievedContext] = Field(default_factory=list)
    options: GenerationOptions = Field(default_factory=GenerationOptions)


class VerificationResult(BaseModel):
    verdict: Literal["pass", "fail"]
    issues: List[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "low"


class InferenceResponse(BaseModel):
    request_id: str
    status: str
    model: str
    answer: str
    retrieved_context: List[RetrievedContext] = Field(default_factory=list)
    used_variables: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    verification: Optional[VerificationResult] = None


class ApiGuidanceResponse(BaseModel):
    request_id: str
    status: str
    answer: str
    model: str
    rag: List[RetrievedContext] = Field(default_factory=list)
    used_variables: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    verification: Optional[VerificationResult] = None


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
    job_id: str
    request_id: str
    status: Literal["queued"] = "queued"
    status_url: str


class JobRecord(BaseModel):
    job_id: str = Field(default_factory=new_job_id)
    request_id: str
    status: JobState
    request: InferenceRequest
    result: Optional[InferenceResponse] = None
    error: Optional[str] = None
    result_object_key: Optional[str] = None
    callback_attempts: int = 0
    callback_last_status: Optional[str] = None
    callback_last_error: Optional[str] = None
    worker_id: Optional[str] = None
    lease_expires_at: Optional[str] = None
    created_at: str = Field(default_factory=utc_now_iso)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    updated_at: str = Field(default_factory=utc_now_iso)


class ApiGuidanceJobStatus(BaseModel):
    job_id: str
    request_id: str
    status: JobState
    answer: Optional[str] = None
    model: Optional[str] = None
    rag: List[RetrievedContext] = Field(default_factory=list)
    used_variables: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    verification: Optional[VerificationResult] = None
    error: Optional[str] = None
    result_object_key: Optional[str] = None
    callback_attempts: int = 0
    callback_last_status: Optional[str] = None
    callback_last_error: Optional[str] = None
    worker_id: Optional[str] = None
    lease_expires_at: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    updated_at: Optional[str] = None
