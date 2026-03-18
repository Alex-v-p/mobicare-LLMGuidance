from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

# Ensure project root is importable when Streamlit runs this file directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts.loader import list_run_artifacts, load_run_artifact


# -----------------------------
# App setup
# -----------------------------
st.set_page_config(page_title="LLM Guidance Benchmarks", layout="wide")
st.title("LLM Guidance Benchmark Dashboard")


# -----------------------------
# Helpers
# -----------------------------
def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def get_nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def metric_card(label: str, value: Any, help_text: str | None = None) -> None:
    st.metric(label=label, value=value, help=help_text)


def build_run_row(artifact: dict[str, Any], path: str) -> dict[str, Any]:
    retrieval = artifact.get("retrieval_summary", {})
    generation = artifact.get("generation_summary", {})
    api = artifact.get("api_summary", {})
    ingestion = artifact.get("ingestion_summary", {})

    return {
        "path": path,
        "run_id": safe_str(artifact.get("run_id")),
        "label": safe_str(artifact.get("label")),
        "datetime": safe_str(artifact.get("datetime")),
        "dataset_version": safe_str(artifact.get("dataset_version")),
        "documents_version": safe_str(artifact.get("documents_version")),
        "hit@1": safe_float(retrieval.get("hit_at_1")),
        "hit@3": safe_float(retrieval.get("hit_at_3")),
        "hit@5": safe_float(retrieval.get("hit_at_5")),
        "mrr": safe_float(retrieval.get("mrr")),
        "avg_answer_similarity": safe_float(generation.get("average_answer_similarity")),
        "avg_fact_recall": safe_float(generation.get("average_required_fact_recall")),
        "exact_pass_rate": safe_float(generation.get("exact_pass_rate")),
        "forbidden_violation_rate": safe_float(generation.get("forbidden_fact_violation_rate")),
        "avg_latency": safe_float(api.get("average")),
        "p95_latency": safe_float(api.get("p95")),
        "p99_latency": safe_float(api.get("p99")),
        "ingestion_duration": safe_float(ingestion.get("total_duration_seconds")),
        "chunks_created": safe_float(ingestion.get("chunks_created")),
        "vectors_upserted": safe_float(ingestion.get("vectors_upserted")),
        "case_count": len(artifact.get("per_case_results", [])),
    }


def per_case_table(artifact: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case in artifact.get("per_case_results", []):
        retrieval_scores = case.get("retrieval_scores", {}) or {}
        generation_scores = case.get("generation_scores", {}) or {}
        timings = case.get("timings", {}) or {}

        rows.append(
            {
                "case_id": safe_str(case.get("case_id")),
                "question": safe_str(case.get("question")),
                "answerability": safe_str(case.get("answerability")),
                "generated_answer": safe_str(case.get("generated_answer")),
                "retrieval_hit@1": safe_float(retrieval_scores.get("hit_at_1")),
                "retrieval_hit@3": safe_float(retrieval_scores.get("hit_at_3")),
                "mrr": safe_float(retrieval_scores.get("mrr")),
                "answer_similarity": safe_float(generation_scores.get("answer_similarity")),
                "fact_recall": safe_float(generation_scores.get("required_fact_recall")),
                "forbidden_violations": safe_float(generation_scores.get("forbidden_fact_violations")),
                "exact_pass": safe_float(generation_scores.get("exact_pass")),
                "total_latency_ms": safe_float(timings.get("total_latency_ms")),
                "retrieval_latency_ms": safe_float(timings.get("retrieval_latency_ms")),
                "generation_latency_ms": safe_float(timings.get("generation_latency_ms")),
                "warning_count": len(case.get("warnings", []) or []),
                "retrieved_chunk_count": len(case.get("retrieved_chunks", []) or []),
                "source_candidate_count": len(case.get("source_match_candidates", []) or []),
            }
        )
    return pd.DataFrame(rows)


def case_compare_table(selected_artifacts: list[dict[str, Any]], case_id: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for artifact in selected_artifacts:
        run_id = safe_str(artifact.get("run_id"))
        label = safe_str(artifact.get("label"))

        case = next(
            (c for c in artifact.get("per_case_results", []) if c.get("case_id") == case_id),
            None,
        )
        if not case:
            continue

        retrieval_scores = case.get("retrieval_scores", {}) or {}
        generation_scores = case.get("generation_scores", {}) or {}
        timings = case.get("timings", {}) or {}

        rows.append(
            {
                "run_id": run_id,
                "label": label,
                "case_id": case_id,
                "question": safe_str(case.get("question")),
                "generated_answer": safe_str(case.get("generated_answer")),
                "retrieval_hit@1": safe_float(retrieval_scores.get("hit_at_1")),
                "retrieval_hit@3": safe_float(retrieval_scores.get("hit_at_3")),
                "mrr": safe_float(retrieval_scores.get("mrr")),
                "answer_similarity": safe_float(generation_scores.get("answer_similarity")),
                "fact_recall": safe_float(generation_scores.get("required_fact_recall")),
                "forbidden_violations": safe_float(generation_scores.get("forbidden_fact_violations")),
                "exact_pass": safe_float(generation_scores.get("exact_pass")),
                "total_latency_ms": safe_float(timings.get("total_latency_ms")),
                "warning_count": len(case.get("warnings", []) or []),
                "retrieved_chunk_count": len(case.get("retrieved_chunks", []) or []),
            }
        )

    return pd.DataFrame(rows)


def melt_metrics(df: pd.DataFrame, id_col: str, metrics: list[str]) -> pd.DataFrame:
    available = [m for m in metrics if m in df.columns]
    if not available:
        return pd.DataFrame()
    return df.melt(
        id_vars=[id_col],
        value_vars=available,
        var_name="metric",
        value_name="value",
    )


def altair_bar(df: pd.DataFrame, x: str, y: str, color: str | None = None, title: str = ""):
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X(x, sort="-y"),
        y=alt.Y(y),
        tooltip=list(df.columns),
    )
    if color:
        chart = chart.encode(color=color)
    if title:
        chart = chart.properties(title=title)
    return chart.interactive()


def altair_line(df: pd.DataFrame, x: str, y: str, color: str | None = None, title: str = ""):
    chart = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X(x),
        y=alt.Y(y),
        tooltip=list(df.columns),
    )
    if color:
        chart = chart.encode(color=color)
    if title:
        chart = chart.properties(title=title)
    return chart.interactive()


# -----------------------------
# Load artifacts
# -----------------------------
root_default = str(ROOT / "artifacts" / "runs")
root = st.sidebar.text_input("Run artifacts directory", value=root_default)
paths = list_run_artifacts(root)

if not paths:
    st.warning("No run artifacts found.")
    st.stop()

artifacts_by_path: dict[str, dict[str, Any]] = {}
run_rows: list[dict[str, Any]] = []

for path in paths:
    artifact = load_run_artifact(path)
    path_str = str(path)
    artifacts_by_path[path_str] = artifact
    run_rows.append(build_run_row(artifact, path_str))

overview_df = pd.DataFrame(run_rows)
if not overview_df.empty and "datetime" in overview_df.columns:
    overview_df["datetime_parsed"] = pd.to_datetime(overview_df["datetime"], errors="coerce")
    overview_df = overview_df.sort_values(by="datetime_parsed", ascending=False)

run_options = overview_df["path"].tolist()
default_compare = run_options[: min(3, len(run_options))]
selected_paths = st.sidebar.multiselect(
    "Runs to compare",
    options=run_options,
    default=default_compare,
)
selected_artifacts = [artifacts_by_path[p] for p in selected_paths]

if not selected_artifacts:
    st.warning("Select at least one run.")
    st.stop()


# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab_compare, tab_case_compare, tab_run_detail = st.tabs(
    ["Overview", "Run comparison", "Case comparison", "Run detail"]
)


# -----------------------------
# Overview
# -----------------------------
with tab_overview:
    st.subheader("Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Runs loaded", len(overview_df))
    with col2:
        metric_card("Best hit@3", round(overview_df["hit@3"].max(), 4) if not overview_df.empty else 0.0)
    with col3:
        metric_card(
            "Best answer similarity",
            round(overview_df["avg_answer_similarity"].max(), 4) if not overview_df.empty else 0.0,
        )
    with col4:
        metric_card(
            "Lowest avg latency",
            round(overview_df["avg_latency"].min(), 2) if not overview_df.empty else 0.0,
        )

    st.markdown("### Run table")
    display_cols = [
        "run_id",
        "label",
        "datetime",
        "dataset_version",
        "documents_version",
        "hit@1",
        "hit@3",
        "hit@5",
        "mrr",
        "avg_answer_similarity",
        "avg_fact_recall",
        "exact_pass_rate",
        "avg_latency",
        "p95_latency",
        "chunks_created",
        "case_count",
    ]
    st.dataframe(overview_df[display_cols], use_container_width=True)

    metric_for_bar = st.selectbox(
        "Metric to visualize across runs",
        options=[
            "hit@1",
            "hit@3",
            "hit@5",
            "mrr",
            "avg_answer_similarity",
            "avg_fact_recall",
            "exact_pass_rate",
            "avg_latency",
            "p95_latency",
            "chunks_created",
        ],
        index=1,
        key="overview_metric_bar",
    )

    chart_df = overview_df[["label", "run_id", metric_for_bar]].copy()
    chart_df["run_label"] = chart_df["label"].fillna("") + " | " + chart_df["run_id"].fillna("")
    st.altair_chart(
        altair_bar(chart_df, x="run_label:N", y=f"{metric_for_bar}:Q", title=f"{metric_for_bar} by run"),
        use_container_width=True,
    )

    if overview_df["datetime_parsed"].notna().any():
        line_metric = st.selectbox(
            "Timeline metric",
            options=[
                "hit@3",
                "mrr",
                "avg_answer_similarity",
                "avg_fact_recall",
                "exact_pass_rate",
                "avg_latency",
                "p95_latency",
            ],
            index=0,
            key="overview_metric_line",
        )
        timeline_df = overview_df.dropna(subset=["datetime_parsed"]).copy()
        timeline_df["run_label"] = timeline_df["label"].fillna("") + " | " + timeline_df["run_id"].fillna("")
        st.altair_chart(
            altair_line(
                timeline_df,
                x="datetime_parsed:T",
                y=f"{line_metric}:Q",
                color="run_label:N",
                title=f"{line_metric} over time",
            ),
            use_container_width=True,
        )


# -----------------------------
# Run comparison
# -----------------------------
with tab_compare:
    st.subheader("Run comparison")

    compare_rows = []
    for artifact in selected_artifacts:
        compare_rows.append(
            {
                "run_id": safe_str(artifact.get("run_id")),
                "label": safe_str(artifact.get("label")),
                "hit@1": safe_float(get_nested(artifact, "retrieval_summary", "hit_at_1", default=0.0)),
                "hit@3": safe_float(get_nested(artifact, "retrieval_summary", "hit_at_3", default=0.0)),
                "hit@5": safe_float(get_nested(artifact, "retrieval_summary", "hit_at_5", default=0.0)),
                "mrr": safe_float(get_nested(artifact, "retrieval_summary", "mrr", default=0.0)),
                "avg_answer_similarity": safe_float(
                    get_nested(artifact, "generation_summary", "average_answer_similarity", default=0.0)
                ),
                "avg_fact_recall": safe_float(
                    get_nested(artifact, "generation_summary", "average_required_fact_recall", default=0.0)
                ),
                "exact_pass_rate": safe_float(
                    get_nested(artifact, "generation_summary", "exact_pass_rate", default=0.0)
                ),
                "avg_latency": safe_float(get_nested(artifact, "api_summary", "average", default=0.0)),
                "p95_latency": safe_float(get_nested(artifact, "api_summary", "p95", default=0.0)),
                "p99_latency": safe_float(get_nested(artifact, "api_summary", "p99", default=0.0)),
                "chunks_created": safe_float(get_nested(artifact, "ingestion_summary", "chunks_created", default=0.0)),
            }
        )

    compare_df = pd.DataFrame(compare_rows)
    compare_df["run_label"] = compare_df["label"].fillna("") + " | " + compare_df["run_id"].fillna("")
    st.dataframe(compare_df.drop(columns=["run_label"]), use_container_width=True)

    compare_metric_groups = {
        "Retrieval": ["hit@1", "hit@3", "hit@5", "mrr"],
        "Generation": ["avg_answer_similarity", "avg_fact_recall", "exact_pass_rate"],
        "Latency": ["avg_latency", "p95_latency", "p99_latency"],
        "Ingestion": ["chunks_created"],
    }

    group_choice = st.selectbox(
        "Metric group",
        options=list(compare_metric_groups.keys()),
        index=0,
    )
    melted = melt_metrics(compare_df, "run_label", compare_metric_groups[group_choice])

    if not melted.empty:
        grouped_chart = (
            alt.Chart(melted)
            .mark_bar()
            .encode(
                x=alt.X("run_label:N", title="Run"),
                y=alt.Y("value:Q", title="Value"),
                color=alt.Color("metric:N", title="Metric"),
                xOffset="metric:N",
                tooltip=["run_label", "metric", "value"],
            )
            .properties(title=f"{group_choice} metrics")
            .interactive()
        )
        st.altair_chart(grouped_chart, use_container_width=True)


# -----------------------------
# Case comparison across runs
# -----------------------------
with tab_case_compare:
    st.subheader("Case comparison across runs")

    all_case_ids = sorted(
        {
            safe_str(case.get("case_id"))
            for artifact in selected_artifacts
            for case in artifact.get("per_case_results", [])
        }
    )

    if not all_case_ids:
        st.info("No per-case results available in the selected runs.")
    else:
        selected_case_id = st.selectbox("Case ID", options=all_case_ids, key="cross_run_case_id")
        case_df = case_compare_table(selected_artifacts, selected_case_id)

        if case_df.empty:
            st.info("Selected case not found in the chosen runs.")
        else:
            st.dataframe(
                case_df[
                    [
                        "run_id",
                        "label",
                        "retrieval_hit@1",
                        "retrieval_hit@3",
                        "mrr",
                        "answer_similarity",
                        "fact_recall",
                        "forbidden_violations",
                        "exact_pass",
                        "total_latency_ms",
                        "warning_count",
                        "retrieved_chunk_count",
                    ]
                ],
                use_container_width=True,
            )

            metric_choice = st.selectbox(
                "Case comparison metric",
                options=[
                    "retrieval_hit@1",
                    "retrieval_hit@3",
                    "mrr",
                    "answer_similarity",
                    "fact_recall",
                    "forbidden_violations",
                    "exact_pass",
                    "total_latency_ms",
                    "warning_count",
                ],
                index=3,
                key="case_compare_metric",
            )

            case_df["run_label"] = case_df["label"].fillna("") + " | " + case_df["run_id"].fillna("")
            st.altair_chart(
                altair_bar(
                    case_df[["run_label", metric_choice]],
                    x="run_label:N",
                    y=f"{metric_choice}:Q",
                    title=f"{selected_case_id} - {metric_choice}",
                ),
                use_container_width=True,
            )

            st.markdown("### Generated answers by run")
            for _, row in case_df.iterrows():
                with st.expander(f'{row["label"]} | {row["run_id"]}'):
                    st.write(row["generated_answer"])


# -----------------------------
# Run detail
# -----------------------------
with tab_run_detail:
    st.subheader("Run detail")

    selected_run_path = st.selectbox(
        "Run",
        options=run_options,
        format_func=lambda p: f'{overview_df.loc[overview_df["path"] == p, "label"].iloc[0]} | '
        f'{overview_df.loc[overview_df["path"] == p, "run_id"].iloc[0]}',
        key="run_detail_path",
    )
    artifact = artifacts_by_path[selected_run_path]
    run_df = per_case_table(artifact)

    header_col1, header_col2, header_col3, header_col4 = st.columns(4)
    with header_col1:
        metric_card("Run ID", safe_str(artifact.get("run_id")))
    with header_col2:
        metric_card("Label", safe_str(artifact.get("label")))
    with header_col3:
        metric_card("Dataset", safe_str(artifact.get("dataset_version")))
    with header_col4:
        metric_card("Cases", len(artifact.get("per_case_results", [])))

    st.markdown("### Config")
    st.json(artifact.get("config", {}))

    summary_col1, summary_col2, summary_col3 = st.columns(3)
    with summary_col1:
        st.markdown("#### Retrieval summary")
        st.json(artifact.get("retrieval_summary", {}))
    with summary_col2:
        st.markdown("#### Generation summary")
        st.json(artifact.get("generation_summary", {}))
    with summary_col3:
        st.markdown("#### API summary")
        st.json(artifact.get("api_summary", {}))

    if run_df.empty:
        st.info("No per-case results in this run.")
    else:
        st.markdown("### Per-case metrics")

        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            answerability_filter = st.multiselect(
                "Answerability",
                options=sorted([x for x in run_df["answerability"].dropna().unique().tolist() if x]),
                default=sorted([x for x in run_df["answerability"].dropna().unique().tolist() if x]),
            )
        with filter_col2:
            min_similarity = st.slider(
                "Minimum answer similarity",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.01,
            )

        filtered_df = run_df.copy()
        if answerability_filter:
            filtered_df = filtered_df[filtered_df["answerability"].isin(answerability_filter)]
        filtered_df = filtered_df[filtered_df["answer_similarity"] >= min_similarity]

        st.dataframe(filtered_df, use_container_width=True)

        scatter_metric_x = st.selectbox(
            "Scatter X-axis",
            options=["retrieval_hit@3", "mrr", "answer_similarity", "fact_recall", "total_latency_ms"],
            index=0,
            key="scatter_x",
        )
        scatter_metric_y = st.selectbox(
            "Scatter Y-axis",
            options=["answer_similarity", "fact_recall", "total_latency_ms", "warning_count"],
            index=0,
            key="scatter_y",
        )

        scatter_chart = (
            alt.Chart(filtered_df)
            .mark_circle(size=90)
            .encode(
                x=alt.X(f"{scatter_metric_x}:Q"),
                y=alt.Y(f"{scatter_metric_y}:Q"),
                color=alt.Color("answerability:N"),
                tooltip=["case_id", "question", scatter_metric_x, scatter_metric_y],
            )
            .properties(title=f"{scatter_metric_y} vs {scatter_metric_x}")
            .interactive()
        )
        st.altair_chart(scatter_chart, use_container_width=True)

        case_options = filtered_df["case_id"].tolist()
        if case_options:
            selected_case_id = st.selectbox("Per-case drilldown", options=case_options, key="run_detail_case_id")
            selected_case = next(
                case for case in artifact.get("per_case_results", [])
                if safe_str(case.get("case_id")) == selected_case_id
            )

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