from __future__ import annotations

from typing import Any

from inference.clinical.config_repository import load_drug_dosing_catalog_payload
from inference.domain.drug_dosing import DrugEvidence, DrugRecommendation
from inference.domain.drug_dosing.catalog import (
    default_agent as domain_default_agent,
    family_keywords as domain_family_keywords,
    family_priority as domain_family_priority,
    family_query_order as domain_family_query_order,
    family_query_template as domain_family_query_template,
)
from inference.domain.drug_dosing.context import select_grounded_rag_context
from inference.domain.drug_dosing.evidence import extract_grounded_drug_evidence as domain_extract_grounded_drug_evidence
from inference.domain.drug_dosing.query_builder import build_drug_retrieval_queries as domain_build_drug_retrieval_queries
from inference.domain.drug_dosing.recommendation_engine import (
    build_recommendations,
    build_safety_cautions,
    build_tradeoff_notes,
    select_visible_recommendations,
)
from inference.domain.drug_dosing.renderer import render_drug_dosing_answer, summarize_drug_dosing_warnings
from inference.domain.drug_dosing.snapshot import build_snapshot as domain_build_snapshot
from inference.domain.drug_dosing.verification import verify_grounded_payload
from shared.contracts.inference import RetrievedContext


# Compatibility wrappers keep the original import surface stable while the
# implementation lives in inference.domain.drug_dosing.

def _drug_dosing_catalog() -> dict[str, Any]:
    return load_drug_dosing_catalog_payload()



def _family_priority() -> dict[str, int]:
    return domain_family_priority(loader=load_drug_dosing_catalog_payload)



def _default_agent(family: str) -> str:
    return domain_default_agent(family, loader=load_drug_dosing_catalog_payload)



def _family_query_order() -> tuple[str, ...]:
    return domain_family_query_order(loader=load_drug_dosing_catalog_payload)



def _family_keywords() -> dict[str, set[str]]:
    return domain_family_keywords(loader=load_drug_dosing_catalog_payload)



def _family_query_template(family: str) -> str:
    return domain_family_query_template(family, loader=load_drug_dosing_catalog_payload)


# ---------- public builders ----------

def build_snapshot(patient_variables: dict[str, Any]) -> dict[str, Any]:
    return domain_build_snapshot(patient_variables, default_agent=_default_agent)



def build_drug_retrieval_queries(snapshot: dict[str, Any]) -> list[dict[str, str]]:
    return domain_build_drug_retrieval_queries(
        snapshot,
        family_query_order=_family_query_order,
        family_query_template=_family_query_template,
        default_agent=_default_agent,
    )



def extract_grounded_drug_evidence(
    *,
    retrieved_context: list[RetrievedContext],
    snapshot: dict[str, Any],
    family_contexts: dict[str, list[RetrievedContext]] | None = None,
) -> dict[str, DrugEvidence]:
    return domain_extract_grounded_drug_evidence(
        retrieved_context=retrieved_context,
        snapshot=snapshot,
        family_contexts=family_contexts,
        family_query_order=_family_query_order,
        family_keywords=_family_keywords,
        default_agent=_default_agent,
    )



def build_grounded_drug_dosing_payload(
    *,
    patient_variables: dict[str, Any],
    retrieved_context: list[RetrievedContext],
    retrieval_queries: list[str],
    family_contexts: dict[str, list[RetrievedContext]] | None = None,
) -> dict[str, Any]:
    snapshot = build_snapshot(patient_variables)
    evidence = extract_grounded_drug_evidence(
        retrieved_context=retrieved_context,
        snapshot=snapshot,
        family_contexts=family_contexts,
    )
    recommendations = build_recommendations(snapshot, evidence, default_agent=_default_agent)
    selected = select_visible_recommendations(recommendations, snapshot, family_priority=_family_priority)
    tradeoffs = build_tradeoff_notes(recommendations)
    safety_cautions = build_safety_cautions(recommendations)
    evidence_rows_used = {
        family: item.to_dict()
        for family, item in evidence.items()
        if item.source_chunk_ids
    }
    return {
        "mode": "drug_dosing_grounded",
        "recommendations": {family: item.to_dict() for family, item in recommendations.items()},
        "selected_recommendations": selected,
        "evidence_rows_used": evidence_rows_used,
        "tradeoffs": tradeoffs,
        "safety_cautions": safety_cautions,
        "inputs_used": snapshot,
        "retrieval_queries": retrieval_queries,
    }
