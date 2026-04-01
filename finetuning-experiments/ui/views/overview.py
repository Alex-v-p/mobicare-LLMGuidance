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
    "prompt_label",
    "hit@1",
    "hit@3",
    "mrr",
    "avg_answer_similarity",
    "avg_deterministic_rubric",
    "avg_llm_judge_score",
    "avg_effective_generation_score",
    "exact_pass_rate",
    "verification_pass_rate",
    "avg_latency",
    "p95_latency",
    "chunks_created",
    "case_count",
]

HIGHER_IS_BETTER = {
    "hit@1",
    "hit@3",
    "hit@5",
    "mrr",
    "weighted_relevance",
    "lenient_success_score",
    "context_diversity_score",
    "soft_ndcg",
    "avg_answer_similarity",
    "avg_answer_quality",
    "avg_deterministic_rubric",
    "avg_judge_score",
    "avg_llm_judge_score",
    "avg_effective_generation_score",
    "exact_pass_rate",
    "verification_pass_rate",
    "api_completion_rate",
    "page_coverage_ratio",
    "normalized.strict_success_rate",
    "normalized.avg_answer_similarity",
}

LOWER_IS_BETTER = {
    "avg_latency",
    "p50_latency",
    "p95_latency",
    "p99_latency",
    "api_failure_rate",
    "api_timeout_rate",
    "queue_delay_avg",
    "execution_duration_avg",
    "forbidden_violation_rate",
    "hallucination_rate",
    "normalized.avg_latency",
}


def _build_scorecard(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "hit@3",
        "mrr",
        "avg_answer_similarity",
        "avg_deterministic_rubric",
        "avg_llm_judge_score",
        "avg_effective_generation_score",
        "exact_pass_rate",
        "verification_pass_rate",
        "avg_latency",
        "p95_latency",
        "api_failure_rate",
        "hallucination_rate",
    ]
    available = [metric for metric in metrics if metric in df.columns]
    if not available:
        return df.assign(composite_score=0.0)

    scored = df.copy()
    for metric in available:
        series = pd.to_numeric(scored[metric], errors="coerce")
        if series.nunique(dropna=True) <= 1:
            scored[f"{metric}__norm"] = 1.0
            continue
        if metric in LOWER_IS_BETTER:
            series = -series
        scored[f"{metric}__norm"] = (series - series.min()) / (series.max() - series.min())

    norm_cols = [f"{metric}__norm" for metric in available]
    scored["composite_score"] = scored[norm_cols].mean(axis=1).round(4)
    scored["efficiency_score"] = (
        scored[[c for c in ["hit@3__norm", "avg_effective_generation_score__norm", "p95_latency__norm"] if c in scored.columns]]
        .mean(axis=1)
        .round(4)
    )
    return scored


def _summarize_patterns(df: pd.DataFrame) -> list[str]:
    findings: list[str] = []
    if df.empty:
        return findings

    ranked = df.sort_values("composite_score", ascending=False)
    best = ranked.iloc[0]
    findings.append(
        f"**Top overall run:** `{best['run_id']}` ({best.get('label') or 'unlabeled'}) with composite score **{best['composite_score']:.3f}**, hit@3 **{best.get('hit@3', 0.0):.3f}**, and effective generation score **{best.get('avg_effective_generation_score', 0.0):.3f}**."
    )

    if len(df) > 1:
        fastest = df.sort_values("p95_latency", ascending=True).iloc[0]
        strongest_retrieval = df.sort_values("hit@3", ascending=False).iloc[0]
        strongest_generation = df.sort_values("avg_effective_generation_score", ascending=False).iloc[0]
        findings.append(
            f"**Best trade-off on speed:** `{fastest['run_id']}` has the lowest p95 latency at **{fastest.get('p95_latency', 0.0):.1f} ms**."
        )
        findings.append(
            f"**Best retrieval:** `{strongest_retrieval['run_id']}` leads hit@3 at **{strongest_retrieval.get('hit@3', 0.0):.3f}**; **best generation:** `{strongest_generation['run_id']}` leads effective generation score at **{strongest_generation.get('avg_effective_generation_score', 0.0):.3f}**."
        )

    for dimension in ["chunking_strategy", "retrieval_mode", "llm_model"]:
        if dimension in df.columns and df[dimension].nunique(dropna=True) > 1:
            grouped = (
                df.groupby(dimension, dropna=False)[["composite_score", "hit@3", "avg_effective_generation_score", "p95_latency"]]
                .mean(numeric_only=True)
                .sort_values("composite_score", ascending=False)
                .reset_index()
            )
            leader = grouped.iloc[0]
            findings.append(
                f"**Strongest {dimension.replace('_', ' ')}:** `{leader[dimension] or 'unspecified'}` averages composite **{leader['composite_score']:.3f}**, hit@3 **{leader['hit@3']:.3f}**, effective generation **{leader['avg_effective_generation_score']:.3f}**, p95 latency **{leader['p95_latency']:.1f} ms**."
            )
    return findings


def _pareto_frontier(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    if df.empty:
        return df
    frontier = []
    ordered = df.sort_values(x_col, ascending=True)
    best_y = float("-inf")
    for _, row in ordered.iterrows():
        y_value = float(row[y_col])
        if y_value >= best_y:
            frontier.append(row)
            best_y = y_value
    return pd.DataFrame(frontier)


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
        metric_card("Best generation", round(float(df["avg_effective_generation_score"].max()), 4))
    with top4:
        valid_p95 = df["p95_latency"].replace(0, pd.NA).dropna()
        metric_card("Lowest p95 latency", round(float(valid_p95.min()), 2) if not valid_p95.empty else 0.0)

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

    if filtered.empty:
        st.warning("Current filters removed all runs.")
        return

    scored = _build_scorecard(filtered)
    scored["run_label"] = scored["label"].fillna("") + " | " + scored["run_id"].fillna("")

    st.markdown("### Unified leaderboard")
    leaderboard_columns = [
        "run_id",
        "label",
        "composite_score",
        "efficiency_score",
        "hit@3",
        "mrr",
        "avg_deterministic_rubric",
        "avg_llm_judge_score",
        "avg_effective_generation_score",
        "p95_latency",
        "api_failure_rate",
        "hallucination_rate",
        "chunking_strategy",
        "retrieval_mode",
        "llm_model",
    ]
    st.dataframe(
        scored[leaderboard_columns].sort_values(["composite_score", "efficiency_score"], ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### What the data is saying")
    for finding in _summarize_patterns(scored):
        st.markdown(f"- {finding}")

    left, right = st.columns([1.15, 1])
    with left:
        frontier = _pareto_frontier(scored[["run_label", "hit@3", "p95_latency"]].dropna(), "p95_latency", "hit@3")
        scatter = (
            alt.Chart(scored)
            .mark_circle(size=130)
            .encode(
                x=alt.X("p95_latency:Q", title="p95 latency (ms)"),
                y=alt.Y("hit@3:Q", title="hit@3"),
                color=alt.Color("avg_effective_generation_score:Q", title="Generation"),
                tooltip=[
                    "run_label",
                    "chunking_strategy",
                    "retrieval_mode",
                    "llm_model",
                    "hit@3",
                    "mrr",
                    "avg_deterministic_rubric",
                    "avg_llm_judge_score",
                    "avg_effective_generation_score",
                    "p95_latency",
                    "api_failure_rate",
                ],
            )
            .properties(title="Retrieval vs latency, colored by generation score")
            .interactive()
        )
        if not frontier.empty:
            frontier_chart = (
                alt.Chart(frontier)
                .mark_line(point=True)
                .encode(x="p95_latency:Q", y="hit@3:Q", tooltip=["run_label", "p95_latency", "hit@3"])
            )
            st.altair_chart(scatter + frontier_chart, use_container_width=True)
        else:
            st.altair_chart(scatter, use_container_width=True)

    with right:
        diagnostic_metrics = [
            "hit@3",
            "mrr",
            "avg_deterministic_rubric",
            "avg_llm_judge_score",
            "avg_effective_generation_score",
            "p95_latency",
            "api_failure_rate",
        ]
        melted = scored.melt(
            id_vars=["run_label"],
            value_vars=[metric for metric in diagnostic_metrics if metric in scored.columns],
            var_name="metric",
            value_name="value",
        )
        chart = (
            alt.Chart(melted)
            .mark_bar()
            .encode(
                x=alt.X("metric:N", title="Metric"),
                y=alt.Y("value:Q", title="Value"),
                color="run_label:N",
                xOffset="run_label:N",
                tooltip=["run_label", "metric", "value"],
            )
            .properties(title="Key benchmark metrics")
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)
