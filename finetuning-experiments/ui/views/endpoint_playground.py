from __future__ import annotations

import json
import time
from typing import Any

import streamlit as st

from adapters.gateway import GatewayClient
from adapters.guidance import GuidanceClient


DEFAULT_INGESTION_PAYLOAD = {
    "cleaning_strategy": "deep",
    "cleaning_params": {},
    "chunking_strategy": "page_index",
    "chunking_params": {},
    "embedding_model": "nomic-embed-text",
}

DEFAULT_STANDARD_GUIDANCE_PAYLOAD = {
    "question": "What therapy should be considered for symptomatic HFrEF despite ACE inhibitor and beta blocker?",
    "patient": {"values": {"age": 72, "ef": 32}},
    "options": {
        "top_k": 3,
        "temperature": 0.0,
        "max_tokens": 256,
        "retrieval_mode": "hybrid",
        "use_graph_augmentation": False,
        "enable_query_rewriting": False,
        "enable_response_verification": False,
        "enable_regeneration": False,
        "pipeline_variant": "standard",
    },
}

DEFAULT_DRUG_DOSING_PAYLOAD = {
    "question": "",
    "patient": {
        "values": {
            "age": 74,
            "gender": "male",
            "creatinine": 2.0,
            "egfr": 34,
            "potassium": 5.3,
            "blood_pressure_systolic": 95,
            "heart_rate": 108,
            "nyha": 3,
            "ef": 28,
            "DoseSpiro_prev": 25,
            "DoseBB_prev": 2.5,
            "RASDose_prev": 10,
            "ARNIDose_prev": 0,
            "SGLT2Dose_prev": 0,
            "Loop_dose_prev": 40
        }
    },
    "options": {
        "pipeline_variant": "drug_dosing",
        "llm_model": "qwen2.5:3b-instruct",
        "embedding_model": "qwen3-embedding:4b",
        "top_k": 3,
        "temperature": 0.0,
        "max_tokens": 256,
        "retrieval_mode": "hybrid",
        "use_graph_augmentation": False,
        "enable_query_rewriting": False,
        "enable_response_verification": True,
        "enable_regeneration": False
    }
}


def _parse_json(text: str) -> dict[str, Any]:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object.")
    return payload



def render() -> None:
    st.subheader("Endpoint playground")
    base_url = st.text_input("Gateway base URL", value="http://localhost:8000")
    timeout_seconds = st.number_input("HTTP timeout (seconds)", min_value=5, max_value=3600, value=60, step=5)

    ingest_tab, guidance_tab = st.tabs(["Ingestion", "Guidance"])

    with ingest_tab:
        ingestion_text = st.text_area("Ingestion payload", value=json.dumps(DEFAULT_INGESTION_PAYLOAD, indent=2), height=220)
        delete_first = st.checkbox("Delete collection before ingestion", value=False)
        if st.button("Run ingestion job"):
            try:
                payload = _parse_json(ingestion_text)
                client = GatewayClient(base_url=base_url, timeout_seconds=int(timeout_seconds))
                started = time.perf_counter()
                if delete_first:
                    st.write(client.delete_ingestion_collection())
                result = client.run_ingestion_and_wait(payload)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                st.success(f"Completed in {elapsed_ms:.2f} ms")
                st.json({"job_id": result.job_id, "status": result.status, "record": result.record})
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

    with guidance_tab:
        example_name = st.selectbox("Guidance example", ["Standard QA", "Drug dosing"], index=0)
        default_payload = DEFAULT_STANDARD_GUIDANCE_PAYLOAD if example_name == "Standard QA" else DEFAULT_DRUG_DOSING_PAYLOAD
        guidance_text = st.text_area("Guidance payload", value=json.dumps(default_payload, indent=2), height=340)
        if st.button("Run guidance job"):
            try:
                payload = _parse_json(guidance_text)
                client = GuidanceClient(base_url=base_url, timeout_seconds=int(timeout_seconds))
                started = time.perf_counter()
                result = client.run_guidance_and_wait(payload)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                st.success(f"Completed in {elapsed_ms:.2f} ms")
                st.json({"job_id": result.job_id, "status": result.status, "record": result.record})
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
