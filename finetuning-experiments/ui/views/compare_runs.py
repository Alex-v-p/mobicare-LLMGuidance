from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st


COMPARE_COLUMNS = [
    "run_id",
    "label",
    "chunking_strategy",
    "retrieval_mode",
    "llm_model",
    "prompt_label",
    "hit@1",
    "hit@3",
    "mrr",
    "weighted_relevance",
    "avg_answer_similarity",
    "avg_fact_recall",
    "exact_pass_rate",
    "verification_pass_rate",
    "avg_latency",
    "p95_latency",
    "queue_delay_avg",
    "chunks_created",
]


def render(df: pd.DataFrame) -> None:
    st.subheader("Compare runs")
    if df.empty:
        st.info("Select at least one run.")
        return

    compare_df = df[COMPARE_COLUMNS].copy()
    compare_df["run_label"] = compare_df["label"].fillna("") + " | " + compare_df["run_id"].fillna("")
    st.dataframe(compare_df.drop(columns=["run_label"]), use_container_width=True, hide_index=True)

    baseline = compare_df.iloc[0]
    delta_rows = []
    numeric_cols = [
        "hit@1",
        "hit@3",
        "mrr",
        "weighted_relevance",
        "avg_answer_similarity",
        "avg_fact_recall",
        "exact_pass_rate",
        "verification_pass_rate",
        "avg_latency",
        "p95_latency",
        "queue_delay_avg",
        "chunks_created",
    ]
    for _, row in compare_df.iterrows():
        delta = {"run_id": row["run_id"], "label": row["label"]}
        for col in numeric_cols:
            delta[f"Δ {col}"] = float(row[col]) - float(baseline[col])
        delta_rows.append(delta)
    st.markdown("### Delta vs first selected run")
    st.dataframe(pd.DataFrame(delta_rows), use_container_width=True, hide_index=True)

    metric_groups = {
        "Retrieval": ["hit@1", "hit@3", "mrr", "weighted_relevance"],
        "Generation": ["avg_answer_similarity", "avg_fact_recall", "exact_pass_rate", "verification_pass_rate"],
        "Latency": ["avg_latency", "p95_latency", "queue_delay_avg"],
        "Ingestion": ["chunks_created"],
    }
    group_name = st.selectbox("Metric group", list(metric_groups.keys()))
    melted = compare_df.melt(
        id_vars=["run_label"],
        value_vars=metric_groups[group_name],
        var_name="metric",
        value_name="value",
    )
    chart = (
        alt.Chart(melted)
        .mark_bar()
        .encode(
            x=alt.X("run_label:N", title="Run"),
            y=alt.Y("value:Q", title="Value"),
            color="metric:N",
            xOffset="metric:N",
            tooltip=["run_label", "metric", "value"],
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)
