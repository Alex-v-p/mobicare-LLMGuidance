from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from inference.clinical import ClinicalProfile
from shared.contracts.inference import InferenceRequest, RetrievedContext

class GuidanceRetriever(Protocol):
    async def retrieve_context(
        self,
        *,
        request: InferenceRequest,
        retrieval_query: str,
    ) -> tuple[list[RetrievedContext], dict[str, object]]: ...


@dataclass(slots=True)
class QueryRewriteResult:
    query: str
    rewritten: bool


@dataclass(slots=True)
class QueryPlan:
    effective_question: str
    base_query: str
    expanded_queries: list[str]
    clinical_profile: ClinicalProfile
    clusters: list[str]
    specialty_focus: str


@dataclass(slots=True)
class ContextAssessment:
    sufficient: bool
    confidence: str
    reasons: list[str]
    topical_terms: list[str]
    cluster_coverage: dict[str, int]
