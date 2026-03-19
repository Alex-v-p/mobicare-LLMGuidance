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

DEFAULT_GUIDANCE_PAYLOAD = {
    "question": "What therapy should be considered for symptomatic HFrEF despite ACE inhibitor and beta blocker?",
    "patient_variables": {"age": 72, "ef": 32},
    "inference": {
        "top_k": 3,
        "temperature": 0.0,
        "max_tokens": 256,
        "retrieval_mode": "hybrid",
        "use_graph_augmentation": False,
        "enable_query_rewriting": False,
        "enable_response_verification": False,
        "enable_regeneration": False,
    },
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
        guidance_text = st.text_area("Guidance payload", value=json.dumps(DEFAULT_GUIDANCE_PAYLOAD, indent=2), height=280)
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
