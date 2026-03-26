from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import streamlit as st

from adapters.gateway import GatewayAPIResponse, GatewayClient
from adapters.guidance import GuidanceClient
from utils.gateway_auth import GatewayAuthContext, create_local_access_token, login_for_access_token, resolve_ssl_verify


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

CLINICAL_CONFIG_FILENAMES = {
    "marker_ranges": "marker_ranges.json",
    "drug_dosing_catalog": "drug_dosing_catalog.json",
}

ROOT = Path(__file__).resolve().parents[3]
CLINICAL_DEFAULTS_DIR = ROOT / "services" / "inference" / "src" / "inference" / "clinical"
DEFAULT_GATEWAY_BASE_URL = os.environ.get("PLAYGROUND_GATEWAY_BASE_URL") or os.environ.get("GATEWAY_BASE_URL") or "http://localhost:8000"


@st.cache_data(show_spinner=False)
def _load_default_clinical_payloads() -> dict[str, dict[str, Any]]:
    defaults: dict[str, dict[str, Any]] = {
        "marker_ranges": {
            "sodium": {
                "label": "Sodium",
                "bands": [
                    {"label": "low", "lt": 135},
                    {"label": "normal", "gte": 135, "lte": 145},
                    {"label": "high", "gt": 145},
                ],
            }
        },
        "drug_dosing_catalog": {
            "default_agents": {"beta_blocker": "bisoprolol"},
            "family_query_order": ["beta_blocker"],
            "family_priority": {"beta_blocker": 10},
            "families": {
                "beta_blocker": {
                    "keywords": ["beta blocker", "bisoprolol"],
                    "query_template": "beta blocker dosing in HFrEF",
                }
            },
        },
    }
    for config_name, filename in CLINICAL_CONFIG_FILENAMES.items():
        path = CLINICAL_DEFAULTS_DIR / filename
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload:
            defaults[config_name] = payload
    return defaults


def _parse_json(text: str) -> dict[str, Any]:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object.")
    return payload


def _payload_state_key(config_name: str) -> str:
    return f"clinical_payload_text::{config_name}"


def _etag_state_key(config_name: str) -> str:
    return f"clinical_etag::{config_name}"


def _checksum_state_key(config_name: str) -> str:
    return f"clinical_checksum::{config_name}"


def _versions_state_key(config_name: str) -> str:
    return f"clinical_versions::{config_name}"


def _rollback_state_key(config_name: str) -> str:
    return f"clinical_rollback_version::{config_name}"


def _last_result_state_key(config_name: str) -> str:
    return f"clinical_last_result::{config_name}"


def _ensure_clinical_state(config_name: str) -> None:
    defaults = _load_default_clinical_payloads()
    payload_key = _payload_state_key(config_name)
    if payload_key not in st.session_state:
        st.session_state[payload_key] = json.dumps(defaults[config_name], indent=2, ensure_ascii=False)
    st.session_state.setdefault(_etag_state_key(config_name), "")
    st.session_state.setdefault(_checksum_state_key(config_name), "")
    st.session_state.setdefault(_versions_state_key(config_name), [])
    st.session_state.setdefault(_rollback_state_key(config_name), "")
    st.session_state.setdefault(_last_result_state_key(config_name), None)


def _coerce_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _coerce_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_coerce_jsonable(item) for item in value]
    return value


def _store_clinical_response(config_name: str, response: GatewayAPIResponse) -> None:
    st.session_state[_last_result_state_key(config_name)] = {
        "status_code": response.status_code,
        "headers": response.headers,
        "body": _coerce_jsonable(response.body),
    }
    etag = response.headers.get("ETag") or (response.body.get("config") or {}).get("etag") or None
    checksum = response.headers.get("X-Content-SHA256") or (response.body.get("config") or {}).get("checksum_sha256") or None
    if etag is not None:
        st.session_state[_etag_state_key(config_name)] = etag
    if checksum is not None:
        st.session_state[_checksum_state_key(config_name)] = checksum

    body = response.body
    if isinstance(body, dict) and isinstance(body.get("payload"), dict) and body["payload"]:
        st.session_state[_payload_state_key(config_name)] = json.dumps(body["payload"], indent=2, ensure_ascii=False)
    versions = body.get("versions")
    if isinstance(versions, list):
        st.session_state[_versions_state_key(config_name)] = versions
        if versions and not st.session_state.get(_rollback_state_key(config_name)):
            st.session_state[_rollback_state_key(config_name)] = str(versions[0].get("version_id") or "")


def _render_clinical_result(config_name: str) -> None:
    result = st.session_state.get(_last_result_state_key(config_name))
    if not result:
        return
    st.caption(f"Last response status: {result['status_code']}")
    with st.expander("Last response headers", expanded=False):
        st.json(result["headers"])
    st.json(result["body"])


def render() -> None:
    st.subheader("Endpoint playground")
    base_url = st.text_input("Gateway base URL", value=DEFAULT_GATEWAY_BASE_URL)
    st.caption("This URL should point at the Nginx reverse-proxy front door, not directly at the API container.")
    timeout_seconds = st.number_input("HTTP timeout (seconds)", min_value=5, max_value=3600, value=60, step=5)
    verify_ssl_enabled = st.checkbox("Verify TLS certificates", value=not str(base_url).startswith("https://localhost"))
    ca_bundle_path = st.text_input(
        "Custom CA bundle path (optional)",
        value=os.environ.get("PLAYGROUND_GATEWAY_CA_BUNDLE_PATH", ""),
        help="Use this for mkcert or other custom roots. When set, requests will verify TLS against this CA bundle.",
    )
    verify_ssl = resolve_ssl_verify(verify_ssl_enabled, ca_bundle_path or None)

    auth_mode = st.selectbox(
        "Authentication",
        options=["None", "Bearer token", "Gateway login", "Generate local JWT (testing)"],
        help="Use a generated local JWT or exchange credentials at /auth/token when the gateway is secured.",
    )
    auth_token = ""
    if auth_mode == "Bearer token":
        auth_token = st.text_input("Bearer token", value=os.environ.get("PLAYGROUND_GATEWAY_AUTH_TOKEN", ""), type="password")
    elif auth_mode == "Gateway login":
        login_col1, login_col2 = st.columns(2)
        with login_col1:
            login_email = st.text_input("Login email", value=os.environ.get("PLAYGROUND_GATEWAY_AUTH_EMAIL", "tester@example.com"))
        with login_col2:
            login_password = st.text_input("Login password", value=os.environ.get("PLAYGROUND_GATEWAY_AUTH_PASSWORD", ""), type="password")
        if login_email and login_password:
            try:
                auth_token, expires_in = login_for_access_token(
                    base_url=base_url,
                    email=login_email,
                    password=login_password,
                    timeout_seconds=int(timeout_seconds),
                    verify_ssl=verify_ssl,
                )
                label = f"about {max(1, expires_in // 60)} minutes" if expires_in else "the configured expiry"
                st.caption(f"Fetched gateway token valid for {label}.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not fetch gateway token: {exc}")
                auth_token = ""
    elif auth_mode == "Generate local JWT (testing)":
        jwt_col1, jwt_col2 = st.columns(2)
        with jwt_col1:
            jwt_email = st.text_input("JWT email", value=os.environ.get("PLAYGROUND_GATEWAY_AUTH_EMAIL", "tester@example.com"))
            jwt_secret = st.text_input(
                "JWT secret",
                value=os.environ.get("PLAYGROUND_GATEWAY_JWT_SECRET", "replace-this-in-real-environments"),
                type="password",
            )
        with jwt_col2:
            jwt_issuer = st.text_input("JWT issuer", value=os.environ.get("PLAYGROUND_GATEWAY_JWT_ISSUER", "mobicare-llm-api"))
            jwt_audience = st.text_input("JWT audience", value=os.environ.get("PLAYGROUND_GATEWAY_JWT_AUDIENCE", "mobicare-gateway"))
            jwt_exp_minutes = st.number_input("JWT expiry (minutes)", min_value=1, max_value=1440, value=60, step=5)
        try:
            auth_token, expires_in = create_local_access_token(
                email=jwt_email,
                secret_key=jwt_secret,
                issuer=jwt_issuer,
                audience=jwt_audience,
                exp_minutes=int(jwt_exp_minutes),
            )
            st.caption(f"Generated local test token valid for {expires_in // 60} minutes.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not generate JWT: {exc}")
            auth_token = ""

    if auth_token:
        with st.expander("Resolved Authorization header", expanded=False):
            st.code(f"Bearer {auth_token[:32]}...", language=None)

    if st.button("Test current auth against /auth/example-protected"):
        try:
            auth_client = GatewayClient(
                base_url=base_url,
                timeout_seconds=int(timeout_seconds),
                auth_token=auth_token or None,
                verify_ssl=verify_ssl,
            )
            st.json(auth_client.test_authentication().body)
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))

    ingest_tab, guidance_tab, configs_tab = st.tabs(["Ingestion", "Guidance", "Clinical configs"])

    with ingest_tab:
        ingestion_text = st.text_area("Ingestion payload", value=json.dumps(DEFAULT_INGESTION_PAYLOAD, indent=2), height=220)
        delete_first = st.checkbox("Delete collection before ingestion", value=False)
        if st.button("Run ingestion job"):
            try:
                payload = _parse_json(ingestion_text)
                client = GatewayClient(
                    base_url=base_url,
                    timeout_seconds=int(timeout_seconds),
                    auth_token=auth_token or None,
                    verify_ssl=verify_ssl,
                )
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
                client = GuidanceClient(
                    base_url=base_url,
                    timeout_seconds=int(timeout_seconds),
                    auth_token=auth_token or None,
                    verify_ssl=verify_ssl,
                )
                started = time.perf_counter()
                result = client.run_guidance_and_wait(payload)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                st.success(f"Completed in {elapsed_ms:.2f} ms")
                st.json({"job_id": result.job_id, "status": result.status, "record": result.record})
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

    with configs_tab:
        config_name = st.selectbox(
            "Config name",
            options=["marker_ranges", "drug_dosing_catalog"],
            help="These map to the MinIO-backed clinical config endpoints exposed by the gateway API.",
        )
        _ensure_clinical_state(config_name)
        client = GatewayClient(
            base_url=base_url,
            timeout_seconds=int(timeout_seconds),
            auth_token=auth_token or None,
            verify_ssl=verify_ssl,
        )

        action_col1, action_col2, action_col3 = st.columns(3)
        if action_col1.button("List configs"):
            try:
                started = time.perf_counter()
                response = client.list_clinical_configs()
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                st.success(f"Listed configs in {elapsed_ms:.2f} ms")
                st.session_state[_last_result_state_key(config_name)] = {
                    "status_code": response.status_code,
                    "headers": response.headers,
                    "body": _coerce_jsonable(response.body),
                }
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

        if action_col2.button("Load current config"):
            try:
                started = time.perf_counter()
                response = client.get_clinical_config(config_name)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                _store_clinical_response(config_name, response)
                st.success(f"Loaded current config in {elapsed_ms:.2f} ms")
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

        if action_col3.button("List versions"):
            try:
                started = time.perf_counter()
                response = client.list_clinical_config_versions(config_name)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                _store_clinical_response(config_name, response)
                versions = response.body.get("versions") or []
                if versions:
                    st.session_state[_rollback_state_key(config_name)] = str(versions[0].get("version_id") or "")
                st.success(f"Loaded {len(versions)} versions in {elapsed_ms:.2f} ms")
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

        payload_text = st.text_area(
            "Config payload",
            key=_payload_state_key(config_name),
            height=360,
            help="Edit the JSON payload that will be sent to create or update operations.",
        )

        concurrency_col1, concurrency_col2 = st.columns(2)
        concurrency_col1.text_input(
            "If-Match (ETag)",
            key=_etag_state_key(config_name),
            help="Optional optimistic-lock token. Use the value returned by GET or a prior write.",
        )
        concurrency_col2.text_input(
            "X-Content-SHA256",
            key=_checksum_state_key(config_name),
            help="Optional checksum lock. Useful when you want to require an exact content match before mutating.",
        )

        operation_tab_create, operation_tab_update, operation_tab_delete, operation_tab_rollback = st.tabs(
            ["Create", "Update", "Delete", "Rollback"]
        )

        with operation_tab_create:
            st.caption("POST /clinical-configs/{config_name}")
            if st.button("Create config"):
                try:
                    payload = _parse_json(payload_text)
                    started = time.perf_counter()
                    response = client.create_clinical_config(
                        config_name,
                        payload,
                        expected_etag=st.session_state[_etag_state_key(config_name)] or None,
                        expected_checksum_sha256=st.session_state[_checksum_state_key(config_name)] or None,
                    )
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    _store_clinical_response(config_name, response)
                    st.success(f"Created config in {elapsed_ms:.2f} ms")
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))

        with operation_tab_update:
            st.caption("PUT /clinical-configs/{config_name}")
            if st.button("Update config"):
                try:
                    payload = _parse_json(payload_text)
                    started = time.perf_counter()
                    response = client.update_clinical_config(
                        config_name,
                        payload,
                        expected_etag=st.session_state[_etag_state_key(config_name)] or None,
                        expected_checksum_sha256=st.session_state[_checksum_state_key(config_name)] or None,
                    )
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    _store_clinical_response(config_name, response)
                    st.success(f"Updated config in {elapsed_ms:.2f} ms")
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))

        with operation_tab_delete:
            st.caption("DELETE /clinical-configs/{config_name}")
            st.warning("Delete archives the current live object first, then removes it from the live MinIO path.")
            if st.button("Delete config"):
                try:
                    started = time.perf_counter()
                    response = client.delete_clinical_config(
                        config_name,
                        expected_etag=st.session_state[_etag_state_key(config_name)] or None,
                        expected_checksum_sha256=st.session_state[_checksum_state_key(config_name)] or None,
                    )
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    _store_clinical_response(config_name, response)
                    st.success(f"Deleted config in {elapsed_ms:.2f} ms")
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))

        with operation_tab_rollback:
            st.caption("POST /clinical-configs/{config_name}/rollback")
            versions = st.session_state.get(_versions_state_key(config_name)) or []
            version_options = [str(item.get("version_id") or "") for item in versions if item.get("version_id")]
            if version_options:
                selected_version = st.selectbox(
                    "Known version IDs",
                    options=version_options,
                    index=0,
                    key=f"clinical_version_select::{config_name}",
                )
                if st.button("Use selected version ID"):
                    st.session_state[_rollback_state_key(config_name)] = selected_version
            st.text_input(
                "Rollback version_id",
                key=_rollback_state_key(config_name),
                help="Paste a version_id returned by the versions endpoint.",
            )
            if st.button("Rollback config"):
                try:
                    version_id = str(st.session_state[_rollback_state_key(config_name)] or "").strip()
                    if not version_id:
                        raise ValueError("Rollback version_id must not be empty.")
                    started = time.perf_counter()
                    response = client.rollback_clinical_config(
                        config_name,
                        version_id,
                        expected_etag=st.session_state[_etag_state_key(config_name)] or None,
                        expected_checksum_sha256=st.session_state[_checksum_state_key(config_name)] or None,
                    )
                    elapsed_ms = (time.perf_counter() - started) * 1000.0
                    _store_clinical_response(config_name, response)
                    st.success(f"Rolled back config in {elapsed_ms:.2f} ms")
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))

        _render_clinical_result(config_name)
