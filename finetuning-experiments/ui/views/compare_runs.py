from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from ui.views.visuals import (
    COMPARE_METRIC_GROUPS,
    COMPARE_PRIORITY_METRICS,
    DETAIL_FRAME_COLUMNS,
    SUMMARY_FRAME_COLUMNS,
    attach_run_labels,
    build_delta_frames,
    metric_label,
    pareto_frontier,
    score_runs,
)


COMPARE_COLUMNS = [
    "run_id",
    "label",
    "chunking_strategy",
    "retrieval_mode",
    "llm_model",
    "prompt_label",
    "query_rewriting",
    "verification",
    "graph_augmentation",
    "second_pass_mapping",
    "hit@1",
    "hit@3",
    "mrr",
    "strict_hit@3",
    "strict_mrr",
    "weighted_relevance_display",
    "lenient_success_score",
    "context_diversity_score",
    "avg_answer_similarity",
    "avg_deterministic_rubric",
    "avg_llm_judge_score",
    "avg_effective_generation_score",
    "exact_pass_rate",
    "grounded_fact_pass_rate",
    "verification_pass_rate",
    "verification_alignment_rate",
    "hallucination_rate",
    "avg_latency",
    "p95_latency",
    "queue_delay_avg",
    "api_failure_rate",
    "api_timeout_rate",
    "chunks_created",
]


def _coerce_summary_frame(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df is None or summary_df.empty:
        return pd.DataFrame(columns=SUMMARY_FRAME_COLUMNS)
    return summary_df.reindex(columns=SUMMARY_FRAME_COLUMNS)


def _coerce_detail_frame(detail_df: pd.DataFrame) -> pd.DataFrame:
    if detail_df is None or detail_df.empty:
        return pd.DataFrame(columns=DETAIL_FRAME_COLUMNS)
    return detail_df.reindex(columns=DETAIL_FRAME_COLUMNS)


SUMMARY_MERGE_COLUMNS = [
    "wins_vs_baseline",
    "losses_vs_baseline",
    "ties_vs_baseline",
    "net_improvement",
    "biggest_gain",
    "biggest_loss",
]


def _merge_summary_columns(compare_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    summary_subset_columns = ["run_label", *SUMMARY_MERGE_COLUMNS]
    summary_subset = _coerce_summary_frame(summary_df)[summary_subset_columns]

    # Guard against re-merging into a frame that already contains these fields.
    # Without dropping them first, pandas adds _x/_y suffixes and downstream column
    # selection fails with KeyError.
    base_compare_df = compare_df.drop(columns=[column for column in SUMMARY_MERGE_COLUMNS if column in compare_df.columns])
    merged = base_compare_df.merge(summary_subset, on="run_label", how="left")

    numeric_defaults = {
        "wins_vs_baseline": 0,
        "losses_vs_baseline": 0,
        "ties_vs_baseline": 0,
        "net_improvement": 0,
    }
    text_defaults = {
        "biggest_gain": "—",
        "biggest_loss": "—",
    }
    for column, default in numeric_defaults.items():
        merged[column] = pd.to_numeric(merged.get(column), errors="coerce").fillna(default)
    for column, default in text_defaults.items():
        merged[column] = merged.get(column).fillna(default)
    return merged


def _narrative(summary_df: pd.DataFrame, detail_df: pd.DataFrame, baseline_run_label: str) -> list[str]:
    insights: list[str] = []
    if summary_df.empty or detail_df.empty:
        return insights

    non_baseline_summary = summary_df[~summary_df["is_baseline"]].copy()
    if non_baseline_summary.empty:
        return insights

    best_overall = non_baseline_summary.sort_values(["net_improvement", "composite_score"], ascending=False).iloc[0]
    insights.append(
        f"**Best overall challenger:** `{best_overall['run_label']}` wins on **{int(best_overall['wins_vs_baseline'])}** metrics and loses on **{int(best_overall['losses_vs_baseline'])}** vs baseline `{baseline_run_label}`."
    )

    for metric in ["hit@3", "avg_effective_generation_score", "p95_latency", "api_failure_rate"]:
        metric_rows = detail_df[(detail_df["metric"] == metric) & (~detail_df["is_baseline"])].copy()
        if metric_rows.empty:
            continue
        best_row = metric_rows.sort_values("benefit_delta", ascending=False).iloc[0]
        label = {
            "hit@3": "Retrieval lift",
            "avg_effective_generation_score": "Generation lift",
            "p95_latency": "Latency movement",
            "api_failure_rate": "Reliability movement",
        }[metric]
        insights.append(f"**{label}:** `{best_row['run_label']}` changes {metric_label(metric).lower()} by **{best_row['delta_display']}** vs baseline.")
    return insights


def _render_summary_table(compare_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    merged = _merge_summary_columns(compare_df, summary_df)
    columns = [
        "label",
        "run_id",
        "composite_score",
        "efficiency_score",
        "hit@3",
        "avg_effective_generation_score",
        "p95_latency",
        "wins_vs_baseline",
        "losses_vs_baseline",
        "net_improvement",
        "biggest_gain",
        "biggest_loss",
    ]
    for column in columns:
        if column not in merged.columns:
            merged[column] = "—"
    st.dataframe(
        merged[columns].sort_values(["net_improvement", "composite_score"], ascending=False),
        use_container_width=True,
        hide_index=True,
    )


def _render_delta_heatmap(detail_df: pd.DataFrame, run_order: list[str]) -> None:
    heatmap_df = detail_df.copy()
    metric_order = [metric_label(metric) for metric in COMPARE_PRIORITY_METRICS if metric in heatmap_df["metric"].unique()]
    base = alt.Chart(heatmap_df).encode(
        x=alt.X("metric_label:N", sort=metric_order, title=None, axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("plot_run_label:N", sort=run_order, title=None),
        tooltip=[
            alt.Tooltip("plot_run_label:N", title="Run"),
            alt.Tooltip("metric_label:N", title="Metric"),
            alt.Tooltip("raw_display:N", title="Run value"),
            alt.Tooltip("baseline_display:N", title="Baseline value"),
            alt.Tooltip("delta_display:N", title="Delta"),
            alt.Tooltip("benefit_delta:Q", title="Benefit-adjusted delta", format="+.3f"),
        ],
    )
    heatmap = base.mark_rect().encode(
        color=alt.Color(
            "benefit_delta:Q",
            title="Better ↔ Worse",
            scale=alt.Scale(scheme="redblue", domainMid=0),
        )
    )
    text = base.mark_text(fontSize=11).encode(
        text="delta_display:N",
        color=alt.condition(alt.datum.benefit_delta > 0.05, alt.value("white"), alt.value("#d1d5db")),
    )
    st.altair_chart((heatmap + text).properties(title="Metric-by-metric delta vs baseline"), use_container_width=True)


def _render_delta_strip(detail_df: pd.DataFrame) -> None:
    group_name = st.selectbox("Delta view", list(COMPARE_METRIC_GROUPS.keys()), key="compare_metric_group")
    metric_group = [metric for metric in COMPARE_METRIC_GROUPS[group_name] if metric in detail_df["metric"].unique()]
    subset = detail_df[(detail_df["metric"].isin(metric_group)) & (~detail_df["is_baseline"])].copy()
    if subset.empty:
        st.info("There are no non-baseline deltas to plot for this metric group.")
        return

    metric_order = [metric_label(metric) for metric in metric_group]
    zero_rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[5, 5], opacity=0.45).encode(x="x:Q")
    points = alt.Chart(subset).mark_circle(size=130).encode(
        x=alt.X("benefit_delta:Q", title="Improvement vs baseline (benefit-adjusted)", axis=alt.Axis(format="+.3f")),
        y=alt.Y("metric_label:N", sort=metric_order, title=None),
        color=alt.Color("plot_run_label:N", title="Run"),
        tooltip=[
            alt.Tooltip("plot_run_label:N", title="Run"),
            alt.Tooltip("metric_label:N", title="Metric"),
            alt.Tooltip("delta_display:N", title="Raw delta"),
            alt.Tooltip("benefit_delta:Q", title="Benefit-adjusted delta", format="+.3f"),
        ],
    )
    st.altair_chart((zero_rule + points).properties(title=f"{group_name} deltas"), use_container_width=True)


def _render_net_improvement(summary_df: pd.DataFrame) -> None:
    subset = summary_df[~summary_df["is_baseline"]].copy()
    if subset.empty:
        st.info("Add more than one run to compare improvements vs baseline.")
        return

    chart = alt.Chart(subset).mark_bar().encode(
        y=alt.Y("plot_run_label:N", sort="-x", title=None),
        x=alt.X("net_improvement:Q", title="Net improvement count"),
        color=alt.condition(alt.datum.net_improvement >= 0, alt.value("#2563eb"), alt.value("#dc2626")),
        tooltip=[
            alt.Tooltip("plot_run_label:N", title="Run"),
            alt.Tooltip("wins_vs_baseline:Q", title="Wins"),
            alt.Tooltip("losses_vs_baseline:Q", title="Losses"),
            alt.Tooltip("ties_vs_baseline:Q", title="Ties"),
            alt.Tooltip("net_improvement:Q", title="Net improvement"),
            alt.Tooltip("biggest_gain:N", title="Biggest gain"),
            alt.Tooltip("biggest_loss:N", title="Biggest loss"),
        ],
    ).properties(title="How often each run beats the baseline")
    st.altair_chart(chart, use_container_width=True)


def _render_tradeoff(compare_df: pd.DataFrame, baseline_label: str) -> None:
    frontier = pareto_frontier(compare_df, "p95_latency", "avg_effective_generation_score")
    baseline_df = compare_df[compare_df["run_label"] == baseline_label]

    base = alt.Chart(compare_df)
    scatter = base.mark_circle(opacity=0.9).encode(
        x=alt.X("p95_latency:Q", title="p95 latency (ms)", scale=alt.Scale(zero=False)),
        y=alt.Y("avg_effective_generation_score:Q", title="Effective generation", scale=alt.Scale(domain=[0, 1])),
        size=alt.Size("hit@3:Q", title="Hit@3", scale=alt.Scale(range=[180, 900])),
        color=alt.Color("net_improvement:Q", title="Net improvement", scale=alt.Scale(scheme="redblue", domainMid=0)),
        tooltip=[
            alt.Tooltip("label:N", title="Label"),
            alt.Tooltip("run_id:N", title="Run ID"),
            alt.Tooltip("hit@3:Q", title="Hit@3", format=".3f"),
            alt.Tooltip("avg_effective_generation_score:Q", title="Generation", format=".3f"),
            alt.Tooltip("p95_latency:Q", title="p95 latency", format=".1f"),
            alt.Tooltip("wins_vs_baseline:Q", title="Wins"),
            alt.Tooltip("losses_vs_baseline:Q", title="Losses"),
            alt.Tooltip("net_improvement:Q", title="Net"),
        ],
    )
    labels = alt.Chart(compare_df).mark_text(align="left", dx=8, dy=-8).encode(
        x="p95_latency:Q",
        y="avg_effective_generation_score:Q",
        text="short_run_label:N",
    )
    frontier_chart = alt.Chart(frontier).mark_line(point=True, strokeDash=[6, 4], opacity=0.8).encode(
        x="p95_latency:Q",
        y="avg_effective_generation_score:Q",
    )
    baseline_ring = alt.Chart(baseline_df).mark_circle(size=520, filled=False, strokeWidth=2.5).encode(
        x="p95_latency:Q",
        y="avg_effective_generation_score:Q",
    )
    st.altair_chart((scatter + frontier_chart + labels + baseline_ring).properties(title="Trade-off map for selected runs").interactive(), use_container_width=True)
    st.caption("Bubble size reflects hit@3. Color shows how many metrics each run improves vs the chosen baseline. The outlined point is the baseline.")


def render(df: pd.DataFrame) -> None:
    st.subheader("Compare runs")
    if df.empty:
        st.info("Select at least one run.")
        return

    compare_df = df[[column for column in COMPARE_COLUMNS if column in df.columns]].copy()
    compare_df = score_runs(compare_df)
    compare_df = attach_run_labels(compare_df)

    baseline_label = st.selectbox("Baseline run", compare_df["run_label"].tolist(), index=0)
    summary_df, detail_df = build_delta_frames(compare_df, baseline_label, COMPARE_PRIORITY_METRICS)
    summary_df = _coerce_summary_frame(summary_df)
    detail_df = _coerce_detail_frame(detail_df)
    compare_df = _merge_summary_columns(compare_df, summary_df)

    st.markdown("### Comparison summary")
    _render_summary_table(compare_df, summary_df)

    if detail_df.empty:
        st.info("No comparable metrics were found for the selected runs yet, so baseline deltas are unavailable for this selection.")
        st.markdown("### Trade-off map")
        _render_tradeoff(compare_df, baseline_label)
        return

    st.markdown("### What changed")
    narrative_lines = _narrative(summary_df, detail_df, baseline_label)
    if narrative_lines:
        for line in narrative_lines:
            st.markdown(f"- {line}")
    else:
        st.caption("There is not enough comparable data yet to generate a narrative summary.")

    left, right = st.columns([1.15, 0.85])
    with left:
        _render_delta_strip(detail_df)
    with right:
        _render_net_improvement(summary_df)

    st.markdown("### Delta heatmap")
    run_order = summary_df.sort_values(["is_baseline", "net_improvement", "composite_score"], ascending=[False, False, False])["plot_run_label"].tolist()
    if not run_order:
        run_order = compare_df.sort_values(["composite_score", "efficiency_score"], ascending=False)["plot_run_label"].tolist()
    _render_delta_heatmap(detail_df, run_order)

    st.markdown("### Trade-off map")
    _render_tradeoff(compare_df, baseline_label)

    with st.expander("Raw metric deltas"):
        wide_delta = (
            detail_df.pivot(index="plot_run_label", columns="metric_label", values="delta_display")
            .reset_index()
            .rename(columns={"plot_run_label": "run"})
        )
        st.dataframe(wide_delta, use_container_width=True, hide_index=True)
