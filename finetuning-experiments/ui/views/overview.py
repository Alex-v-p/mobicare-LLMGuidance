from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from ui.views.common import metric_card
from ui.views.visuals import OVERVIEW_HEATMAP_METRICS, attach_run_labels, build_metric_long_frame, metric_label, pareto_frontier, score_runs


LEADERBOARD_COLUMNS = [
    "rank",
    "label",
    "run_id",
    "composite_score",
    "efficiency_score",
    "retrieval_index",
    "generation_index",
    "operations_index",
    "hit@3",
    "mrr",
    "avg_effective_generation_score",
    "p95_latency",
    "api_failure_rate",
    "llm_model",
    "chunking_strategy",
    "retrieval_mode",
]


def _summarize_patterns(df: pd.DataFrame) -> list[str]:
    findings: list[str] = []
    if df.empty:
        return findings

    ranked = df.sort_values(["composite_score", "efficiency_score"], ascending=False)
    best = ranked.iloc[0]
    findings.append(
        f"**Top overall run:** `{best['run_id']}` pairs composite **{best['composite_score']:.3f}** with hit@3 **{best.get('hit@3', 0.0):.3f}**, effective generation **{best.get('avg_effective_generation_score', 0.0):.3f}**, and p95 latency **{best.get('p95_latency', 0.0):.1f} ms**."
    )

    if len(df) > 1:
        fastest = df.sort_values("p95_latency", ascending=True).iloc[0]
        best_retrieval = df.sort_values("hit@3", ascending=False).iloc[0]
        best_generation = df.sort_values("avg_effective_generation_score", ascending=False).iloc[0]
        findings.append(
            f"**Fastest run:** `{fastest['run_id']}` has the lowest p95 latency at **{fastest.get('p95_latency', 0.0):.1f} ms**."
        )
        findings.append(
            f"**Retrieval leader:** `{best_retrieval['run_id']}` leads hit@3 at **{best_retrieval.get('hit@3', 0.0):.3f}**; **generation leader:** `{best_generation['run_id']}` leads effective generation at **{best_generation.get('avg_effective_generation_score', 0.0):.3f}**."
        )

    frontier = pareto_frontier(df, "p95_latency", "avg_effective_generation_score")
    if not frontier.empty:
        findings.append(f"**Pareto-efficient runs:** {', '.join(f'`{run_id}`' for run_id in frontier['run_id'].tolist())}.")

    for dimension in ["llm_model", "chunking_strategy", "retrieval_mode"]:
        if dimension not in df.columns or df[dimension].nunique(dropna=True) <= 1:
            continue
        grouped = (
            df.groupby(dimension, dropna=False)[["composite_score", "hit@3", "avg_effective_generation_score", "p95_latency"]]
            .mean(numeric_only=True)
            .sort_values("composite_score", ascending=False)
            .reset_index()
        )
        leader = grouped.iloc[0]
        findings.append(
            f"**Best {dimension.replace('_', ' ')}:** `{leader[dimension] or 'unspecified'}` averages composite **{leader['composite_score']:.3f}**, hit@3 **{leader['hit@3']:.3f}**, generation **{leader['avg_effective_generation_score']:.3f}**, p95 latency **{leader['p95_latency']:.1f} ms**."
        )
    return findings


def _render_tradeoff_chart(scored: pd.DataFrame) -> None:
    frontier = pareto_frontier(scored, "p95_latency", "avg_effective_generation_score")
    median_latency = float(scored["p95_latency"].median()) if scored["p95_latency"].notna().any() else 0.0
    median_generation = float(scored["avg_effective_generation_score"].median()) if scored["avg_effective_generation_score"].notna().any() else 0.0
    label_df = scored.sort_values(["composite_score", "efficiency_score"], ascending=False).head(min(4, len(scored)))

    base = alt.Chart(scored)
    scatter = base.mark_circle(opacity=0.9).encode(
        x=alt.X("p95_latency:Q", title="p95 latency (ms)", scale=alt.Scale(zero=False)),
        y=alt.Y("avg_effective_generation_score:Q", title="Effective generation", scale=alt.Scale(domain=[0, 1])),
        size=alt.Size("hit@3:Q", title="Hit@3", scale=alt.Scale(range=[120, 800])),
        color=alt.Color("composite_score:Q", title="Composite", scale=alt.Scale(scheme="blues")),
        shape=alt.Shape("llm_model:N", title="LLM model"),
        tooltip=[
            alt.Tooltip("label:N", title="Label"),
            alt.Tooltip("run_id:N", title="Run ID"),
            alt.Tooltip("chunking_strategy:N", title="Chunking"),
            alt.Tooltip("retrieval_mode:N", title="Retrieval"),
            alt.Tooltip("llm_model:N", title="LLM"),
            alt.Tooltip("hit@3:Q", title="Hit@3", format=".3f"),
            alt.Tooltip("mrr:Q", title="MRR", format=".3f"),
            alt.Tooltip("avg_effective_generation_score:Q", title="Generation", format=".3f"),
            alt.Tooltip("p95_latency:Q", title="p95 latency", format=".1f"),
            alt.Tooltip("composite_score:Q", title="Composite", format=".3f"),
        ],
    )

    median_rules = (
        alt.Chart(pd.DataFrame({"median_latency": [median_latency], "median_generation": [median_generation]}))
        .mark_rule(strokeDash=[5, 5], opacity=0.35)
        .encode(x="median_latency:Q")
    ) + (
        alt.Chart(pd.DataFrame({"median_latency": [median_latency], "median_generation": [median_generation]}))
        .mark_rule(strokeDash=[5, 5], opacity=0.35)
        .encode(y="median_generation:Q")
    )

    frontier_chart = alt.Chart(frontier).mark_line(point=True, strokeDash=[6, 4], opacity=0.8).encode(
        x="p95_latency:Q",
        y="avg_effective_generation_score:Q",
        tooltip=[
            alt.Tooltip("run_id:N", title="Run ID"),
            alt.Tooltip("p95_latency:Q", title="p95 latency", format=".1f"),
            alt.Tooltip("avg_effective_generation_score:Q", title="Generation", format=".3f"),
        ],
    )

    labels = alt.Chart(label_df).mark_text(align="left", dx=8, dy=-8).encode(
        x="p95_latency:Q",
        y="avg_effective_generation_score:Q",
        text="short_run_label:N",
    )

    chart = (median_rules + scatter + frontier_chart + labels).properties(title="Trade-off frontier: quality vs latency").interactive()
    st.altair_chart(chart, use_container_width=True)
    st.caption("Bubble size tracks retrieval hit@3. Dashed lines show the filtered medians. The dashed frontier highlights Pareto-efficient runs.")


def _render_dimension_breakdown(scored: pd.DataFrame) -> None:
    candidate_dimensions = [
        column
        for column in ["llm_model", "chunking_strategy", "retrieval_mode", "dataset_version"]
        if column in scored.columns and scored[column].nunique(dropna=True) > 1
    ]
    if not candidate_dimensions:
        st.info("Current filters only include one value per configuration dimension, so there is nothing useful to aggregate here yet.")
        return

    dimension = st.selectbox("Aggregate runs by", candidate_dimensions, key="overview_dimension")
    grouped = (
        scored.groupby(dimension, dropna=False)[["composite_score", "hit@3", "avg_effective_generation_score", "p95_latency"]]
        .mean(numeric_only=True)
        .reset_index()
        .sort_values("composite_score", ascending=False)
    )
    grouped[dimension] = grouped[dimension].replace("", "unspecified")

    chart = alt.Chart(grouped).mark_bar().encode(
        y=alt.Y(f"{dimension}:N", sort="-x", title=None),
        x=alt.X("composite_score:Q", title="Average composite score"),
        color=alt.Color("avg_effective_generation_score:Q", title="Avg generation", scale=alt.Scale(scheme="blues")),
        tooltip=[
            alt.Tooltip(f"{dimension}:N", title=dimension.replace("_", " ").title()),
            alt.Tooltip("composite_score:Q", title="Composite", format=".3f"),
            alt.Tooltip("hit@3:Q", title="Hit@3", format=".3f"),
            alt.Tooltip("avg_effective_generation_score:Q", title="Generation", format=".3f"),
            alt.Tooltip("p95_latency:Q", title="p95 latency", format=".1f"),
        ],
    ).properties(title=f"Average performance by {dimension.replace('_', ' ')}")
    st.altair_chart(chart, use_container_width=True)


def _render_metric_heatmap(scored: pd.DataFrame) -> None:
    heatmap_df = build_metric_long_frame(scored, OVERVIEW_HEATMAP_METRICS, normalized=True)
    if heatmap_df.empty:
        st.info("No diagnostic metrics are available for the current filters.")
        return

    run_order = scored.sort_values(["composite_score", "efficiency_score"], ascending=False)["plot_run_label"].tolist()
    metric_order = [metric_label(metric) for metric in OVERVIEW_HEATMAP_METRICS if metric in heatmap_df["metric"].unique()]

    base = alt.Chart(heatmap_df).encode(
        x=alt.X("metric_label:N", sort=metric_order, title=None, axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("plot_run_label:N", sort=run_order, title=None),
        tooltip=[
            alt.Tooltip("plot_run_label:N", title="Run"),
            alt.Tooltip("metric_label:N", title="Metric"),
            alt.Tooltip("display_value:N", title="Raw value"),
            alt.Tooltip("normalized_value:Q", title="Relative strength", format=".2f"),
        ],
    )
    heatmap = base.mark_rect().encode(
        color=alt.Color("normalized_value:Q", title="Relative strength", scale=alt.Scale(domain=[0, 1], scheme="blues"))
    )
    text = base.mark_text(fontSize=11).encode(
        text="display_value:N",
        color=alt.condition(alt.datum.normalized_value >= 0.55, alt.value("white"), alt.value("#d1d5db")),
    )
    st.altair_chart((heatmap + text).properties(title="Run profile heatmap"), use_container_width=True)


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

    filter1, filter2, filter3, filter4 = st.columns(4)
    with filter1:
        datasets = sorted(x for x in df["dataset_version"].dropna().unique().tolist() if x)
        selected_datasets = st.multiselect("Dataset", datasets, default=datasets)
    with filter2:
        chunking = sorted(x for x in df["chunking_strategy"].dropna().unique().tolist() if x)
        selected_chunking = st.multiselect("Chunking strategy", chunking, default=chunking)
    with filter3:
        models = sorted(x for x in df["llm_model"].dropna().unique().tolist() if x)
        selected_models = st.multiselect("LLM model", models, default=models)
    with filter4:
        retrieval_modes = sorted(x for x in df["retrieval_mode"].dropna().unique().tolist() if x)
        selected_retrieval_modes = st.multiselect("Retrieval mode", retrieval_modes, default=retrieval_modes)

    filtered = df.copy()
    if selected_datasets:
        filtered = filtered[filtered["dataset_version"].isin(selected_datasets)]
    if selected_chunking:
        filtered = filtered[filtered["chunking_strategy"].isin(selected_chunking)]
    if selected_models:
        filtered = filtered[filtered["llm_model"].isin(selected_models)]
    if selected_retrieval_modes:
        filtered = filtered[filtered["retrieval_mode"].isin(selected_retrieval_modes)]

    if filtered.empty:
        st.warning("Current filters removed all runs.")
        return

    scored = score_runs(filtered)
    leaderboard = scored.sort_values(["composite_score", "efficiency_score"], ascending=False).reset_index(drop=True)
    leaderboard["rank"] = leaderboard.index + 1

    st.markdown("### Unified leaderboard")
    st.dataframe(leaderboard[LEADERBOARD_COLUMNS], use_container_width=True, hide_index=True)

    st.markdown("### What the data is saying")
    for finding in _summarize_patterns(leaderboard):
        st.markdown(f"- {finding}")

    left, right = st.columns([1.25, 0.95])
    with left:
        _render_tradeoff_chart(leaderboard)
    with right:
        _render_dimension_breakdown(leaderboard)

    st.markdown("### Diagnostic metric profile")
    _render_metric_heatmap(leaderboard)
