from __future__ import annotations

import json

import altair as alt
import pandas as pd
import streamlit as st

from ui.views.common import build_case_dataframe, load_full_run, load_source_maps, safe_str


def render(run_df: pd.DataFrame, output_dir: str) -> None:
    st.subheader("Per-case analysis")
    if run_df.empty:
        st.info("No runs available.")
        return

    run_options = run_df[["run_id", "label", "artifact_path"]].copy()
    selected_run_id = st.selectbox(
        "Run",
        options=run_options["run_id"].tolist(),
        format_func=lambda run_id: f"{run_options.loc[run_options['run_id'] == run_id, 'label'].iloc[0]} | {run_id}",
    )
    artifact_path = run_options.loc[run_options["run_id"] == selected_run_id, "artifact_path"].iloc[0]
    artifact = load_full_run(artifact_path)
    if not artifact:
        st.warning("Full artifact not found for this run.")
        return

    cases = build_case_dataframe(artifact)
    if cases.empty:
        st.info("This run has no per-case results.")
        return

    filter1, filter2 = st.columns(2)
    with filter1:
        statuses = sorted(x for x in cases["status"].dropna().unique().tolist() if x)
        selected_statuses = st.multiselect("Status", statuses, default=statuses)
    with filter2:
        min_similarity = st.slider("Minimum answer similarity", 0.0, 1.0, 0.0, 0.01)

    filtered = cases.copy()
    if selected_statuses:
        filtered = filtered[filtered["status"].isin(selected_statuses)]
    filtered = filtered[filtered["answer_similarity"] >= min_similarity]

    st.dataframe(
        filtered[
            [
                "case_id",
                "status",
                "answerability",
                "retrieval_hit@3",
                "mrr",
                "answer_similarity",
                "fact_recall",
                "faithfulness",
                "hallucination_rate",
                "exact_pass",
                "warning_count",
                "total_latency_ms",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    scatter = (
        alt.Chart(filtered)
        .mark_circle(size=80)
        .encode(
            x=alt.X("retrieval_hit@3:Q"),
            y=alt.Y("answer_similarity:Q"),
            color="status:N",
            tooltip=["case_id", "question", "retrieval_hit@3", "answer_similarity", "total_latency_ms"],
        )
        .properties(title="Answer similarity vs retrieval hit@3")
        .interactive()
    )
    st.altair_chart(scatter, use_container_width=True)

    case_id = st.selectbox("Case drilldown", filtered["case_id"].tolist())
    case = next((item for item in artifact.get("per_case_results") or [] if safe_str(item.get("case_id")) == case_id), None)
    if not case:
        return

    case_left, case_right = st.columns(2)
    with case_left:
        st.markdown("### Prompted case")
        st.write(case.get("question"))
        st.markdown("### Gold passage")
        st.write(case.get("gold_passage_text"))
        st.markdown("### Reference answer")
        st.write(case.get("reference_answer"))
        st.markdown("### Required facts")
        st.write(case.get("required_facts") or [])
        st.markdown("### Forbidden facts")
        st.write(case.get("forbidden_facts") or [])
    with case_right:
        st.markdown("### Generated answer")
        st.write(case.get("generated_answer"))
        st.markdown("### Score breakdown")
        st.json(
            {
                "retrieval": case.get("retrieval_scores") or {},
                "generation": case.get("generation_scores") or {},
                "timings": case.get("timings") or {},
                "telemetry": case.get("telemetry") or {},
            }
        )

    source_maps = load_source_maps(output_dir, selected_run_id)
    source_assignments = source_maps.get("case_source_maps") or []
    assignment = next((item for item in source_assignments if safe_str(item.get("case_id")) == case_id), {})

    st.markdown("### Retrieved chunks")
    retrieved_df = pd.DataFrame(case.get("retrieved_chunks") or [])
    if not retrieved_df.empty:
        st.dataframe(retrieved_df, use_container_width=True, hide_index=True)
    else:
        st.info("No retrieved chunks recorded.")

    st.markdown("### Derived source list")
    st.json(case.get("source_list") or assignment.get("source_list") or {})

    with st.expander("Raw endpoint result"):
        st.code(json.dumps(case.get("raw_endpoint_result") or {}, indent=2, ensure_ascii=False))
