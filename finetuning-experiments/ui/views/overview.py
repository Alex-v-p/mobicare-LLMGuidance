from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from ui.views.common import metric_card


RUN_COLUMNS = [
    "run_id",
    "label",
    "datetime",
    "dataset_version",
    "documents_version",
    "chunking_strategy",
    "retrieval_mode",
    "llm_model",
    "hit@1",
    "hit@3",
    "mrr",
    "avg_answer_similarity",
    "avg_fact_recall",
    "exact_pass_rate",
    "avg_latency",
    "p95_latency",
    "chunks_created",
    "case_count",
]


def _bar(df: pd.DataFrame, x: str, y: str, title: str) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(x=alt.X(x, sort="-y"), y=alt.Y(y), tooltip=list(df.columns))
        .properties(title=title)
        .interactive()
    )


def _line(df: pd.DataFrame, x: str, y: str, color: str, title: str) -> alt.Chart:
    return (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(x=alt.X(x), y=alt.Y(y), color=color, tooltip=list(df.columns))
        .properties(title=title)
        .interactive()
    )


def render(df: pd.DataFrame) -> None:
    st.subheader("Run overview")
    if df.empty:
        st.info("No runs found.")
        return

    top1, top2, top3, top4 = st.columns(4)
    with top1:
        metric_card("Runs loaded", len(df))
    with top2:
        metric_card("Best hit@3", round(float(df["hit@3"].max()), 4))
    with top3:
        metric_card("Best answer similarity", round(float(df["avg_answer_similarity"].max()), 4))
    with top4:
        metric_card("Lowest p95 latency", round(float(df["p95_latency"].replace(0, pd.NA).dropna().min() or 0.0), 2))

    filter1, filter2, filter3 = st.columns(3)
    with filter1:
        datasets = sorted(x for x in df["dataset_version"].dropna().unique().tolist() if x)
        selected_datasets = st.multiselect("Dataset", datasets, default=datasets)
    with filter2:
        chunking = sorted(x for x in df["chunking_strategy"].dropna().unique().tolist() if x)
        selected_chunking = st.multiselect("Chunking strategy", chunking, default=chunking)
    with filter3:
        models = sorted(x for x in df["llm_model"].dropna().unique().tolist() if x)
        selected_models = st.multiselect("LLM model", models, default=models)

    filtered = df.copy()
    if selected_datasets:
        filtered = filtered[filtered["dataset_version"].isin(selected_datasets)]
    if selected_chunking:
        filtered = filtered[filtered["chunking_strategy"].isin(selected_chunking)]
    if selected_models:
        filtered = filtered[filtered["llm_model"].isin(selected_models)]

    st.dataframe(filtered[RUN_COLUMNS], use_container_width=True, hide_index=True)

    if filtered.empty:
        return

    metric = st.selectbox(
        "Overview chart metric",
        [
            "hit@1",
            "hit@3",
            "mrr",
            "avg_answer_similarity",
            "avg_fact_recall",
            "exact_pass_rate",
            "verification_pass_rate",
            "avg_latency",
            "p95_latency",
            "queue_delay_avg",
            "chunks_created",
            "avg_chunk_length",
        ],
        index=1,
    )
    chart_df = filtered[["run_id", "label", metric]].copy()
    chart_df["run_label"] = chart_df["label"].fillna("") + " | " + chart_df["run_id"].fillna("")
    st.altair_chart(_bar(chart_df, "run_label:N", f"{metric}:Q", f"{metric} by run"), use_container_width=True)

    timeline_df = filtered.dropna(subset=["datetime_parsed"]).copy()
    if not timeline_df.empty:
        timeline_metric = st.selectbox(
            "Timeline metric",
            [
                "hit@3",
                "avg_answer_similarity",
                "exact_pass_rate",
                "avg_latency",
                "p95_latency",
            ],
            index=0,
        )
        timeline_df["run_label"] = timeline_df["label"].fillna("") + " | " + timeline_df["run_id"].fillna("")
        st.altair_chart(
            _line(timeline_df, "datetime_parsed:T", f"{timeline_metric}:Q", "run_label:N", f"{timeline_metric} over time"),
            use_container_width=True,
        )
