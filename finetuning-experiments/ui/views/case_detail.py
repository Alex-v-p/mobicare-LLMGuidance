from __future__ import annotations

import json

import altair as alt
import pandas as pd
import streamlit as st

from ui.views.common import build_case_dataframe, load_full_run, load_source_maps, safe_str


GENERATION_THRESHOLD = 0.6
HALLUCINATION_THRESHOLD = 0.15
LATENCY_THRESHOLD_MS = 8000


def _classify_bottleneck(row: pd.Series) -> str:
    if float(row.get("retrieval_hit@3", 0.0)) < 0.5:
        return "Retrieval bottleneck"
    if float(row.get("generation_score_display", 0.0)) < GENERATION_THRESHOLD:
        return "Generation bottleneck"
    if float(row.get("hallucination_rate", 0.0)) > HALLUCINATION_THRESHOLD:
        return "Grounding risk"
    if float(row.get("total_latency_ms", 0.0)) > LATENCY_THRESHOLD_MS:
        return "Latency bottleneck"
    return "Healthy"


def _prepare_case_frame(cases: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    frame = cases.copy()
    retrieval_axis = "weighted_relevance_display"
    if retrieval_axis not in frame.columns or frame[retrieval_axis].dropna().empty:
        retrieval_axis = "mrr" if "mrr" in frame.columns and frame["mrr"].dropna().any() else "retrieval_hit@3"

    frame["retrieval_strength"] = pd.to_numeric(frame[retrieval_axis], errors="coerce").fillna(0.0)
    frame["generation_strength"] = pd.to_numeric(frame["generation_score_display"], errors="coerce").fillna(0.0)
    frame["total_latency_ms"] = pd.to_numeric(frame["total_latency_ms"], errors="coerce").fillna(0.0)
    frame["hallucination_rate"] = pd.to_numeric(frame["hallucination_rate"], errors="coerce").fillna(0.0)

    frame["retrieval_band"] = pd.cut(
        frame["retrieval_strength"],
        bins=[-0.001, 0.05, 0.2, 0.4, 1.001],
        labels=["None", "Weak", "Moderate", "Strong"],
    )
    frame["generation_band"] = pd.cut(
        frame["generation_strength"],
        bins=[-0.001, 0.25, 0.5, 0.75, 1.001],
        labels=["Very low", "Low", "Good", "Strong"],
    )
    frame["latency_band"] = pd.cut(
        frame["total_latency_ms"],
        bins=[-0.001, 4000, 8000, 12000, float("inf")],
        labels=["Fast", "Moderate", "Slow", "Very slow"],
    )
    return frame, retrieval_axis


def _render_case_scatter(filtered: pd.DataFrame, retrieval_axis: str) -> None:
    axis_title = {
        "weighted_relevance_display": "Retrieval evidence quality",
        "mrr": "Retrieval ranking quality (MRR)",
        "retrieval_hit@3": "Retrieval hit@3",
    }.get(retrieval_axis, retrieval_axis)

    thresholds = pd.DataFrame({"x": [0.2], "y": [GENERATION_THRESHOLD]})
    scatter = alt.Chart(filtered).mark_circle(opacity=0.85).encode(
        x=alt.X("retrieval_strength:Q", title=axis_title, scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("generation_strength:Q", title="Generation quality", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("bottleneck:N", title="Bottleneck"),
        size=alt.Size("total_latency_ms:Q", title="Latency (ms)", scale=alt.Scale(range=[40, 800])),
        tooltip=[
            alt.Tooltip("case_id:N", title="Case"),
            alt.Tooltip("answerability:N", title="Answerability"),
            alt.Tooltip("evaluation_profile:N", title="Evaluation profile"),
            alt.Tooltip("bottleneck:N", title="Bottleneck"),
            alt.Tooltip("retrieval_strength:Q", title=axis_title, format=".3f"),
            alt.Tooltip("generation_strength:Q", title="Generation", format=".3f"),
            alt.Tooltip("hallucination_rate:Q", title="Hallucination", format=".3f"),
            alt.Tooltip("total_latency_ms:Q", title="Latency", format=".0f"),
            alt.Tooltip("question:N", title="Question"),
        ],
    )
    rules = alt.Chart(thresholds).mark_rule(strokeDash=[5, 5], opacity=0.35).encode(x="x:Q") + alt.Chart(thresholds).mark_rule(strokeDash=[5, 5], opacity=0.35).encode(y="y:Q")
    st.altair_chart((rules + scatter).properties(title="Case map: retrieval evidence vs generation quality").interactive(), use_container_width=True)


def _render_distribution_heatmap(filtered: pd.DataFrame) -> None:
    dist = (
        filtered.groupby(["retrieval_band", "generation_band"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    base = alt.Chart(dist).encode(
        x=alt.X("retrieval_band:N", title="Retrieval evidence band", sort=["None", "Weak", "Moderate", "Strong"]),
        y=alt.Y("generation_band:N", title="Generation band", sort=["Very low", "Low", "Good", "Strong"]),
        tooltip=["retrieval_band", "generation_band", "count"],
    )
    heatmap = base.mark_rect().encode(color=alt.Color("count:Q", title="Cases", scale=alt.Scale(scheme="blues")))
    text = base.mark_text(fontSize=12).encode(
        text="count:Q",
        color=alt.condition(alt.datum.count >= filtered.shape[0] * 0.08, alt.value("white"), alt.value("#d1d5db")),
    )
    st.altair_chart((heatmap + text).properties(title="Where cases cluster"), use_container_width=True)


def _render_bottleneck_bar(filtered: pd.DataFrame) -> None:
    bottleneck_counts = (
        filtered.groupby("bottleneck")
        .agg(count=("case_id", "size"), median_generation=("generation_strength", "median"), median_latency_ms=("total_latency_ms", "median"))
        .reset_index()
        .sort_values("count", ascending=False)
    )
    chart = alt.Chart(bottleneck_counts).mark_bar().encode(
        y=alt.Y("bottleneck:N", sort="-x", title=None),
        x=alt.X("count:Q", title="Cases"),
        color=alt.Color("median_generation:Q", title="Median generation", scale=alt.Scale(scheme="blues")),
        tooltip=[
            alt.Tooltip("bottleneck:N", title="Bottleneck"),
            alt.Tooltip("count:Q", title="Cases"),
            alt.Tooltip("median_generation:Q", title="Median generation", format=".3f"),
            alt.Tooltip("median_latency_ms:Q", title="Median latency", format=".0f"),
        ],
    ).properties(title="Dominant failure modes")
    st.altair_chart(chart, use_container_width=True)


def _render_answerability_chart(filtered: pd.DataFrame) -> None:
    answerability_summary = (
        filtered.groupby("answerability")[
            ["retrieval_strength", "generation_strength", "deterministic_rubric_score", "llm_judge_score", "total_latency_ms"]
        ]
        .mean(numeric_only=True)
        .reset_index()
        .melt(id_vars=["answerability"], var_name="metric", value_name="value")
    )
    chart = alt.Chart(answerability_summary).mark_bar().encode(
        x=alt.X("metric:N", title=None, axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("value:Q", title="Average value"),
        color=alt.Color("answerability:N", title="Answerability"),
        xOffset="answerability:N",
        tooltip=["answerability", "metric", alt.Tooltip("value:Q", format=".3f")],
    ).properties(title="Answerable vs unanswerable performance")
    st.altair_chart(chart, use_container_width=True)


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
    cases, retrieval_axis = _prepare_case_frame(cases)

    top1, top2, top3, top4, top5 = st.columns(5)
    with top1:
        st.metric("Cases", len(cases))
    with top2:
        st.metric("Healthy cases", int((cases["bottleneck"] == "Healthy").sum()))
    with top3:
        st.metric("Retrieval bottlenecks", int((cases["bottleneck"] == "Retrieval bottleneck").sum()))
    with top4:
        st.metric("Generation bottlenecks", int((cases["bottleneck"] == "Generation bottleneck").sum()))
    with top5:
        st.metric("Median latency", f"{cases['total_latency_ms'].median():.0f} ms")

    filter1, filter2, filter3, filter4 = st.columns(4)
    with filter1:
        statuses = sorted(x for x in cases["status"].dropna().unique().tolist() if x)
        selected_statuses = st.multiselect("Status", statuses, default=statuses)
    with filter2:
        bottlenecks = sorted(x for x in cases["bottleneck"].dropna().unique().tolist() if x)
        selected_bottlenecks = st.multiselect("Bottleneck type", bottlenecks, default=bottlenecks)
    with filter3:
        answerability = sorted(x for x in cases["answerability"].dropna().unique().tolist() if x)
        selected_answerability = st.multiselect("Answerability", answerability, default=answerability)
    with filter4:
        min_generation = st.slider("Minimum generation score", 0.0, 1.0, 0.0, 0.01)

    question_filter = st.text_input("Search question or case ID", placeholder="heart failure, case-042, ...")

    filtered = cases.copy()
    if selected_statuses:
        filtered = filtered[filtered["status"].isin(selected_statuses)]
    if selected_bottlenecks:
        filtered = filtered[filtered["bottleneck"].isin(selected_bottlenecks)]
    if selected_answerability:
        filtered = filtered[filtered["answerability"].isin(selected_answerability)]
    filtered = filtered[filtered["generation_strength"] >= min_generation]
    if question_filter.strip():
        needle = question_filter.strip().lower()
        filtered = filtered[
            filtered["case_id"].str.lower().str.contains(needle, na=False)
            | filtered["question"].str.lower().str.contains(needle, na=False)
        ]

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
                "retrieval_strength",
                "retrieval_hit@3",
                "mrr",
                "generation_strength",
                "deterministic_rubric_score",
                "llm_judge_score",
                "hallucination_rate",
                "warning_count",
                "retrieved_chunk_count",
                "total_latency_ms",
            ]
        ].sort_values(["generation_strength", "retrieval_strength"], ascending=[True, True]),
        use_container_width=True,
        hide_index=True,
    )

    left, right = st.columns([1.2, 0.8])
    with left:
        _render_case_scatter(filtered, retrieval_axis)
    with right:
        _render_distribution_heatmap(filtered)

    lower_left, lower_right = st.columns([0.9, 1.1])
    with lower_left:
        _render_bottleneck_bar(filtered)
    with lower_right:
        _render_answerability_chart(filtered)

    cohort_left, cohort_right = st.columns(2)
    with cohort_left:
        answerability_summary = (
            filtered.groupby("answerability")[["retrieval_strength", "generation_strength", "deterministic_rubric_score", "llm_judge_score", "total_latency_ms"]]
            .mean(numeric_only=True)
            .reset_index()
        )
        st.markdown("### Performance by answerability")
        st.dataframe(answerability_summary, use_container_width=True, hide_index=True)

    with cohort_right:
        hard_cases = filtered.sort_values(["generation_strength", "retrieval_strength"], ascending=True).head(10)
        st.markdown("### Lowest-confidence cases")
        st.dataframe(
            hard_cases[["case_id", "evaluation_profile", "bottleneck", "question", "retrieval_strength", "generation_strength", "deterministic_rubric_score", "llm_judge_score", "hallucination_rate"]],
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
