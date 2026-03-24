from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st


COMPARE_COLUMNS = [
    "run_id",
    "label",
    "chunking_strategy",
    "retrieval_mode",
    "pipeline_variant",
    "llm_model",
    "prompt_label",
    "query_rewriting",
    "verification",
    "graph_augmentation",
    "second_pass_mapping",
    "hit@1",
    "hit@3",
    "mrr",
    "weighted_relevance",
    "lenient_success_score",
    "context_diversity_score",
    "avg_answer_similarity",
    "avg_answer_quality",
    "avg_judge_score",
    "avg_fact_recall",
    "avg_faithfulness",
    "exact_pass_rate",
    "verification_pass_rate",
    "hallucination_rate",
    "avg_latency",
    "p95_latency",
    "queue_delay_avg",
    "api_failure_rate",
    "api_timeout_rate",
    "chunks_created",
]

BENEFIT_DIRECTION = {
    "hit@1": 1,
    "hit@3": 1,
    "mrr": 1,
    "weighted_relevance": 1,
    "lenient_success_score": 1,
    "context_diversity_score": 1,
    "avg_answer_similarity": 1,
    "avg_answer_quality": 1,
    "avg_judge_score": 1,
    "avg_fact_recall": 1,
    "avg_faithfulness": 1,
    "exact_pass_rate": 1,
    "verification_pass_rate": 1,
    "hallucination_rate": -1,
    "avg_latency": -1,
    "p95_latency": -1,
    "queue_delay_avg": -1,
    "api_failure_rate": -1,
    "api_timeout_rate": -1,
    "chunks_created": -1,
}


def _normalize_for_heatmap(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for metric in metrics:
        series = pd.to_numeric(df[metric], errors="coerce")
        if series.nunique(dropna=True) <= 1:
            norm = pd.Series([1.0] * len(series), index=series.index)
        else:
            if BENEFIT_DIRECTION.get(metric, 1) < 0:
                series = -series
            norm = (series - series.min()) / (series.max() - series.min())
        for idx, value in norm.items():
            rows.append(
                {
                    "run_label": df.loc[idx, "run_label"],
                    "metric": metric,
                    "normalized_value": float(value),
                    "raw_value": float(df.loc[idx, metric]),
                }
            )
    return pd.DataFrame(rows)


def _narrative(delta_df: pd.DataFrame, baseline_run_label: str) -> list[str]:
    insights: list[str] = []
    if delta_df.empty:
        return insights
    non_baseline = delta_df[delta_df["run_label"] != baseline_run_label].copy()
    if non_baseline.empty:
        return insights

    def best(metric: str, ascending: bool = False) -> pd.Series:
        return non_baseline.sort_values(metric, ascending=ascending).iloc[0]

    retrieval_winner = best("Δ hit@3", ascending=False)
    generation_winner = best("Δ avg_answer_similarity", ascending=False)
    latency_winner = best("Δ p95_latency", ascending=True)
    risk_winner = best("Δ api_failure_rate", ascending=True)

    insights.append(
        f"**Retrieval lift:** `{retrieval_winner['run_label']}` improves hit@3 by **{retrieval_winner['Δ hit@3']:+.3f}** vs baseline."
    )
    insights.append(
        f"**Generation lift:** `{generation_winner['run_label']}` changes answer similarity by **{generation_winner['Δ avg_answer_similarity']:+.3f}**."
    )
    insights.append(
        f"**Latency movement:** `{latency_winner['run_label']}` changes p95 latency by **{latency_winner['Δ p95_latency']:+.1f} ms**. Negative is better here."
    )
    insights.append(
        f"**Operational risk:** `{risk_winner['run_label']}` changes failure rate by **{risk_winner['Δ api_failure_rate']:+.3f}**. Negative is better."
    )
    return insights


def render(df: pd.DataFrame) -> None:
    st.subheader("Compare runs")
    if df.empty:
        st.info("Select at least one run.")
        return

    compare_df = df[COMPARE_COLUMNS].copy()
    compare_df["run_label"] = compare_df["label"].fillna("") + " | " + compare_df["run_id"].fillna("")
    st.dataframe(compare_df.drop(columns=["run_label"]), use_container_width=True, hide_index=True)

    baseline_label = st.selectbox("Baseline run", compare_df["run_label"].tolist(), index=0)
    baseline = compare_df.loc[compare_df["run_label"] == baseline_label].iloc[0]

    numeric_cols = list(BENEFIT_DIRECTION.keys())
    delta_rows = []
    for _, row in compare_df.iterrows():
        delta = {"run_id": row["run_id"], "label": row["label"], "run_label": row["run_label"]}
        wins = 0
        losses = 0
        for col in numeric_cols:
            delta_value = float(row[col]) - float(baseline[col])
            delta[f"Δ {col}"] = delta_value
            if row["run_label"] != baseline_label:
                adjusted = delta_value * BENEFIT_DIRECTION[col]
                if adjusted > 0:
                    wins += 1
                elif adjusted < 0:
                    losses += 1
        delta["wins_vs_baseline"] = wins
        delta["losses_vs_baseline"] = losses
        delta_rows.append(delta)
    delta_df = pd.DataFrame(delta_rows)

    st.markdown("### Change vs baseline")
    st.dataframe(delta_df.drop(columns=["run_label"]), use_container_width=True, hide_index=True)

    st.markdown("### What changed")
    for line in _narrative(delta_df, baseline_label):
        st.markdown(f"- {line}")

    left, right = st.columns([1.1, 0.9])
    with left:
        metric_groups = {
            "Retrieval": ["hit@1", "hit@3", "mrr", "weighted_relevance"],
            "Generation": ["avg_answer_similarity", "avg_fact_recall", "avg_faithfulness", "exact_pass_rate", "verification_pass_rate", "hallucination_rate"],
            "Latency / reliability": ["avg_latency", "p95_latency", "queue_delay_avg", "api_failure_rate", "api_timeout_rate"],
            "Pipeline complexity": ["chunks_created"],
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
            .properties(title=f"{group_name} metrics")
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)

    with right:
        diagnostic_cols = ["hit@3", "lenient_success_score", "avg_answer_quality", "avg_judge_score", "p95_latency", "api_failure_rate", "hallucination_rate"]
        heatmap_df = _normalize_for_heatmap(compare_df, diagnostic_cols)
        heatmap = (
            alt.Chart(heatmap_df)
            .mark_rect()
            .encode(
                x=alt.X("metric:N", title="Metric"),
                y=alt.Y("run_label:N", title="Run"),
                color=alt.Color("normalized_value:Q", title="Relative strength"),
                tooltip=["run_label", "metric", "raw_value", "normalized_value"],
            )
            .properties(title="Relative strength heatmap")
        )
        st.altair_chart(heatmap, use_container_width=True)

    st.markdown("### Trade-off map")
    tradeoff = (
        alt.Chart(compare_df)
        .mark_circle(size=170)
        .encode(
            x=alt.X("p95_latency:Q", title="p95 latency (ms)"),
            y=alt.Y("avg_answer_quality:Q", title="Answer quality"),
            color=alt.Color("hit@3:Q", title="hit@3"),
            shape=alt.Shape("retrieval_mode:N", title="Retrieval mode"),
            tooltip=[
                "run_label",
                "chunking_strategy",
                "retrieval_mode",
                "pipeline_variant",
                "llm_model",
                "prompt_label",
                "hit@3",
                "avg_answer_quality",
                "avg_judge_score",
                "avg_fact_recall",
                "p95_latency",
                "api_failure_rate",
            ],
        )
        .interactive()
    )
    st.altair_chart(tradeoff, use_container_width=True)
