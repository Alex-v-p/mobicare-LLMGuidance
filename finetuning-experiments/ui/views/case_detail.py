from __future__ import annotations

import json

import altair as alt
import pandas as pd
import streamlit as st

from ui.views.common import build_case_dataframe, load_full_run, load_source_maps, safe_str


def _classify_bottleneck(row: pd.Series) -> str:
    quality_score = pd.to_numeric(pd.Series([row.get("quality_score")]), errors="coerce").iloc[0]
    fact_recall = pd.to_numeric(pd.Series([row.get("fact_recall")]), errors="coerce").iloc[0]
    if float(row.get("retrieval_hit@3", 0.0)) < 0.5:
        return "Retrieval bottleneck"
    if pd.notna(quality_score) and quality_score < 0.6:
        return "Generation bottleneck"
    if pd.notna(fact_recall) and fact_recall < 0.6:
        return "Generation bottleneck"
    if float(row.get("hallucination_rate", 0.0)) > 0.15:
        return "Grounding risk"
    if float(row.get("total_latency_ms", 0.0)) > 8000:
        return "Latency bottleneck"
    return "Healthy"


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

    cases = cases.copy()
    cases["bottleneck"] = cases.apply(_classify_bottleneck, axis=1)

    top1, top2, top3, top4 = st.columns(4)
    with top1:
        st.metric("Cases", len(cases))
    with top2:
        st.metric("Healthy cases", int((cases["bottleneck"] == "Healthy").sum()))
    with top3:
        st.metric("Retrieval bottlenecks", int((cases["bottleneck"] == "Retrieval bottleneck").sum()))
    with top4:
        st.metric("Generation bottlenecks", int((cases["bottleneck"] == "Generation bottleneck").sum()))

    filter1, filter2, filter3 = st.columns(3)
    with filter1:
        statuses = sorted(x for x in cases["status"].dropna().unique().tolist() if x)
        selected_statuses = st.multiselect("Status", statuses, default=statuses)
    with filter2:
        bottlenecks = sorted(x for x in cases["bottleneck"].dropna().unique().tolist() if x)
        selected_bottlenecks = st.multiselect("Bottleneck type", bottlenecks, default=bottlenecks)
    with filter3:
        min_similarity = st.slider("Minimum generation score", 0.0, 1.0, 0.0, 0.01)

    filtered = cases.copy()
    if selected_statuses:
        filtered = filtered[filtered["status"].isin(selected_statuses)]
    if selected_bottlenecks:
        filtered = filtered[filtered["bottleneck"].isin(selected_bottlenecks)]
    filtered = filtered[filtered["quality_score"].fillna(0.0) >= min_similarity]

    if filtered.empty:
        st.warning("Current filters removed all cases.")
        return

    st.markdown("### Case health table")
    st.dataframe(
        filtered[
            [
                "case_id",
                "status",
                "answerability",
                "evaluation_profile",
                "bottleneck",
                "retrieval_hit@3",
                "mrr",
                "quality_score",
                "deterministic_rubric_score",
                "llm_judge_score",
                "answer_similarity",
                "fact_recall",
                "faithfulness",
                "hallucination_rate",
                "exact_pass",
                "warning_count",
                "retrieved_chunk_count",
                "total_latency_ms",
            ]
        ].sort_values(["quality_score", "retrieval_hit@3"], ascending=[True, True]),
        use_container_width=True,
        hide_index=True,
    )

    left, right = st.columns(2)
    with left:
        scatter = (
            alt.Chart(filtered)
            .mark_circle(size=90)
            .encode(
                x=alt.X("retrieval_hit@3:Q", title="retrieval hit@3"),
                y=alt.Y("quality_score:Q", title="generation score"),
                color=alt.Color("bottleneck:N", title="Bottleneck"),
                size=alt.Size("total_latency_ms:Q", title="Latency (ms)"),
                tooltip=[
                    "case_id",
                    "question",
                    "evaluation_profile",
                    "bottleneck",
                    "retrieval_hit@3",
                    "quality_score",
                    "deterministic_rubric_score",
                    "llm_judge_score",
                    "answer_similarity",
                    "fact_recall",
                    "hallucination_rate",
                    "total_latency_ms",
                ],
            )
            .properties(title="Where each case breaks down")
            .interactive()
        )
        st.altair_chart(scatter, use_container_width=True)

    with right:
        bottleneck_counts = filtered.groupby("bottleneck").size().reset_index(name="count")
        bar = (
            alt.Chart(bottleneck_counts)
            .mark_bar()
            .encode(
                x=alt.X("bottleneck:N", sort="-y", title="Failure mode"),
                y=alt.Y("count:Q", title="Cases"),
                tooltip=["bottleneck", "count"],
            )
            .properties(title="Dominant failure modes")
        )
        st.altair_chart(bar, use_container_width=True)

    cohort_left, cohort_right = st.columns(2)
    with cohort_left:
        answerability_summary = (
            filtered.groupby("answerability")[["retrieval_hit@3", "quality_score", "deterministic_rubric_score", "llm_judge_score", "fact_recall", "total_latency_ms"]]
            .mean(numeric_only=True)
            .reset_index()
        )
        st.markdown("### Performance by answerability")
        st.dataframe(answerability_summary, use_container_width=True, hide_index=True)

    with cohort_right:
        hard_cases = filtered.sort_values(["quality_score", "retrieval_hit@3"], ascending=True).head(10)
        st.markdown("### Lowest-confidence cases")
        st.dataframe(
            hard_cases[["case_id", "bottleneck", "question", "quality_score", "retrieval_hit@3", "deterministic_rubric_score", "llm_judge_score", "fact_recall", "hallucination_rate"]],
            use_container_width=True,
            hide_index=True,
        )

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
