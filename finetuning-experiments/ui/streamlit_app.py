from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure project root is importable when Streamlit runs this file directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts.loader import list_run_artifacts, load_run_artifact

st.set_page_config(page_title="LLM Guidance Benchmarks", layout="wide")
st.title("LLM Guidance Benchmark Dashboard")

root_default = str(ROOT / "artifacts" / "runs")
root = st.sidebar.text_input("Run artifacts directory", value=root_default)
paths = list_run_artifacts(root)

if not paths:
    st.warning("No run artifacts found.")
    st.stop()

runs: list[dict] = []
for path in paths:
    artifact = load_run_artifact(path)
    runs.append(
        {
            "path": str(path),
            "run_id": artifact.get("run_id"),
            "label": artifact.get("label"),
            "datetime": artifact.get("datetime"),
            "hit@3": artifact.get("retrieval_summary", {}).get("hit_at_3", 0.0),
            "answer_similarity": artifact.get("generation_summary", {}).get(
                "average_answer_similarity", 0.0
            ),
            "exact_pass_rate": artifact.get("generation_summary", {}).get(
                "exact_pass_rate", 0.0
            ),
            "avg_latency": artifact.get("api_summary", {}).get("average", 0.0),
        }
    )

overview_df = pd.DataFrame(runs)
st.subheader("Run overview")
st.dataframe(overview_df, use_container_width=True)

selected_paths = st.multiselect(
    "Compare runs",
    options=[r["path"] for r in runs],
    default=[runs[0]["path"]],
)

selected_artifacts = [load_run_artifact(path) for path in selected_paths]

if selected_artifacts:
    compare_rows: list[dict] = []
    for artifact in selected_artifacts:
        compare_rows.append(
            {
                "run_id": artifact.get("run_id"),
                "label": artifact.get("label"),
                "hit@1": artifact.get("retrieval_summary", {}).get("hit_at_1", 0.0),
                "hit@3": artifact.get("retrieval_summary", {}).get("hit_at_3", 0.0),
                "mrr": artifact.get("retrieval_summary", {}).get("mrr", 0.0),
                "answer_similarity": artifact.get("generation_summary", {}).get(
                    "average_answer_similarity", 0.0
                ),
                "fact_recall": artifact.get("generation_summary", {}).get(
                    "average_required_fact_recall", 0.0
                ),
                "exact_pass_rate": artifact.get("generation_summary", {}).get(
                    "exact_pass_rate", 0.0
                ),
                "avg_latency": artifact.get("api_summary", {}).get("average", 0.0),
                "p95_latency": artifact.get("api_summary", {}).get("p95", 0.0),
            }
        )
    st.subheader("Comparison")
    st.dataframe(pd.DataFrame(compare_rows), use_container_width=True)

artifact = selected_artifacts[0]
per_case = artifact.get("per_case_results", [])

if not per_case:
    st.warning("This run has no per-case results.")
    st.stop()

case_options = [case.get("case_id") for case in per_case]
selected_case_id = st.selectbox("Per-case drilldown", options=case_options)
selected_case = next(case for case in per_case if case.get("case_id") == selected_case_id)

col1, col2 = st.columns(2)
with col1:
    st.markdown("### Case")
    st.write(selected_case.get("question"))
    st.markdown("### Gold passage")
    st.write(selected_case.get("gold_passage_text"))
    st.markdown("### Reference answer")
    st.write(selected_case.get("reference_answer"))

with col2:
    st.markdown("### Generated answer")
    st.write(selected_case.get("generated_answer"))
    st.markdown("### Scores")
    st.json(
        {
            "retrieval": selected_case.get("retrieval_scores", {}),
            "generation": selected_case.get("generation_scores", {}),
            "timings": selected_case.get("timings", {}),
        }
    )

st.markdown("### Retrieved chunks")
retrieved_df = pd.DataFrame(selected_case.get("retrieved_chunks", []))
if not retrieved_df.empty:
    st.dataframe(retrieved_df, use_container_width=True)
else:
    st.info("No retrieved chunks recorded.")

st.markdown("### Source match candidates")
source_df = pd.json_normalize(selected_case.get("source_match_candidates", []))
if not source_df.empty:
    st.dataframe(source_df, use_container_width=True)
else:
    st.info("No source mapping candidates recorded.")

with st.expander("Raw endpoint result"):
    st.code(
        json.dumps(
            selected_case.get("raw_endpoint_result", {}),
            indent=2,
            ensure_ascii=False,
        )
    )