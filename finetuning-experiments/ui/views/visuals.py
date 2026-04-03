from __future__ import annotations

from typing import Any

import pandas as pd


METRIC_METADATA: dict[str, dict[str, Any]] = {
    "hit@1": {"label": "Hit@1", "higher_is_better": True, "group": "Retrieval", "format": ".3f"},
    "hit@3": {"label": "Hit@3", "higher_is_better": True, "group": "Retrieval", "format": ".3f"},
    "hit@5": {"label": "Hit@5", "higher_is_better": True, "group": "Retrieval", "format": ".3f"},
    "mrr": {"label": "MRR", "higher_is_better": True, "group": "Retrieval", "format": ".3f"},
    "strict_hit@3": {"label": "Strict hit@3", "higher_is_better": True, "group": "Retrieval", "format": ".3f"},
    "strict_mrr": {"label": "Strict MRR", "higher_is_better": True, "group": "Retrieval", "format": ".3f"},
    "weighted_relevance": {"label": "Weighted relevance", "higher_is_better": True, "group": "Retrieval", "format": ".3f"},
    "weighted_relevance_display": {"label": "Evidence quality", "higher_is_better": True, "group": "Retrieval", "format": ".3f"},
    "lenient_success_score": {"label": "Lenient success", "higher_is_better": True, "group": "Retrieval", "format": ".3f"},
    "context_diversity_score": {"label": "Context diversity", "higher_is_better": True, "group": "Retrieval", "format": ".3f"},
    "avg_answer_similarity": {"label": "Answer similarity", "higher_is_better": True, "group": "Generation", "format": ".3f"},
    "avg_deterministic_rubric": {"label": "Deterministic rubric", "higher_is_better": True, "group": "Generation", "format": ".3f"},
    "avg_llm_judge_score": {"label": "LLM judge", "higher_is_better": True, "group": "Generation", "format": ".3f"},
    "avg_effective_generation_score": {"label": "Effective generation", "higher_is_better": True, "group": "Generation", "format": ".3f"},
    "exact_pass_rate": {"label": "Exact pass rate", "higher_is_better": True, "group": "Generation", "format": ".3f"},
    "grounded_fact_pass_rate": {"label": "Grounded fact pass", "higher_is_better": True, "group": "Generation", "format": ".3f"},
    "verification_pass_rate": {"label": "Verification pass", "higher_is_better": True, "group": "Generation", "format": ".3f"},
    "verification_alignment_rate": {"label": "Verification alignment", "higher_is_better": True, "group": "Generation", "format": ".3f"},
    "hallucination_rate": {"label": "Hallucination rate", "higher_is_better": False, "group": "Generation", "format": ".3f"},
    "avg_latency": {"label": "Avg latency (ms)", "higher_is_better": False, "group": "Operations", "format": ".1f"},
    "p50_latency": {"label": "p50 latency (ms)", "higher_is_better": False, "group": "Operations", "format": ".1f"},
    "p95_latency": {"label": "p95 latency (ms)", "higher_is_better": False, "group": "Operations", "format": ".1f"},
    "p99_latency": {"label": "p99 latency (ms)", "higher_is_better": False, "group": "Operations", "format": ".1f"},
    "queue_delay_avg": {"label": "Queue delay (ms)", "higher_is_better": False, "group": "Operations", "format": ".1f"},
    "api_failure_rate": {"label": "API failure rate", "higher_is_better": False, "group": "Operations", "format": ".3f"},
    "api_timeout_rate": {"label": "API timeout rate", "higher_is_better": False, "group": "Operations", "format": ".3f"},
    "chunks_created": {"label": "Chunks created", "higher_is_better": False, "group": "Pipeline", "format": ".0f"},
}

SCORECARD_METRICS = [
    "hit@3",
    "mrr",
    "weighted_relevance_display",
    "avg_answer_similarity",
    "avg_deterministic_rubric",
    "avg_llm_judge_score",
    "avg_effective_generation_score",
    "exact_pass_rate",
    "grounded_fact_pass_rate",
    "verification_pass_rate",
    "verification_alignment_rate",
    "p95_latency",
    "api_failure_rate",
    "hallucination_rate",
]

OVERVIEW_HEATMAP_METRICS = [
    "hit@3",
    "mrr",
    "weighted_relevance_display",
    "avg_deterministic_rubric",
    "avg_llm_judge_score",
    "avg_effective_generation_score",
    "p95_latency",
    "api_failure_rate",
]

COMPARE_METRIC_GROUPS: dict[str, list[str]] = {
    "Retrieval": ["hit@1", "hit@3", "mrr", "strict_hit@3", "strict_mrr", "weighted_relevance_display", "lenient_success_score", "context_diversity_score"],
    "Generation": ["avg_answer_similarity", "avg_deterministic_rubric", "avg_llm_judge_score", "avg_effective_generation_score", "exact_pass_rate", "grounded_fact_pass_rate", "verification_pass_rate", "verification_alignment_rate", "hallucination_rate"],
    "Operations": ["avg_latency", "p95_latency", "queue_delay_avg", "api_failure_rate", "api_timeout_rate"],
    "Pipeline": ["chunks_created"],
}

COMPARE_PRIORITY_METRICS = [metric for metrics in COMPARE_METRIC_GROUPS.values() for metric in metrics]

SUMMARY_FRAME_COLUMNS = [
    "run_id",
    "label",
    "run_label",
    "plot_run_label",
    "composite_score",
    "efficiency_score",
    "wins_vs_baseline",
    "losses_vs_baseline",
    "ties_vs_baseline",
    "net_improvement",
    "biggest_gain",
    "biggest_gain_value",
    "biggest_loss",
    "biggest_loss_value",
    "is_baseline",
]

DETAIL_FRAME_COLUMNS = [
    "run_label",
    "plot_run_label",
    "metric",
    "metric_label",
    "metric_group",
    "raw_value",
    "baseline_value",
    "delta",
    "benefit_delta",
    "delta_display",
    "raw_display",
    "baseline_display",
    "is_baseline",
]


def empty_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=SUMMARY_FRAME_COLUMNS)


def empty_detail_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=DETAIL_FRAME_COLUMNS)


def metric_label(metric: str) -> str:
    return str(METRIC_METADATA.get(metric, {}).get("label") or metric.replace("_", " ").title())


def metric_group(metric: str) -> str:
    return str(METRIC_METADATA.get(metric, {}).get("group") or "Other")


def is_higher_better(metric: str) -> bool:
    return bool(METRIC_METADATA.get(metric, {}).get("higher_is_better", True))


def metric_direction(metric: str) -> int:
    return 1 if is_higher_better(metric) else -1


def format_metric_value(metric: str, value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    fmt = str(METRIC_METADATA.get(metric, {}).get("format") or ".3f")
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return str(value)


def format_metric_delta(metric: str, value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    fmt = str(METRIC_METADATA.get(metric, {}).get("format") or ".3f")
    try:
        return format(float(value), f"+{fmt}")
    except (TypeError, ValueError):
        return str(value)


def _truncate(text: str, limit: int = 28) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def attach_run_labels(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    label_series = frame.get("label", pd.Series(index=frame.index, dtype="object")).fillna("")
    run_series = frame.get("run_id", pd.Series(index=frame.index, dtype="object")).fillna("")
    frame["run_label"] = label_series.where(label_series != "", run_series)
    frame["selector_label"] = frame["run_label"] + " | " + run_series
    frame["short_run_label"] = frame["run_label"].map(lambda value: _truncate(value, 30))
    frame["plot_run_label"] = frame["short_run_label"] + " · " + run_series.astype(str).str[-6:]
    return frame


def available_metrics(df: pd.DataFrame, metrics: list[str]) -> list[str]:
    available: list[str] = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        series = pd.to_numeric(df[metric], errors="coerce")
        if series.notna().sum() == 0:
            continue
        available.append(metric)
    return available


def normalize_metric_series(series: pd.Series, metric: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if valid.empty:
        return pd.Series([float("nan")] * len(numeric), index=numeric.index)
    if valid.nunique(dropna=True) <= 1:
        return pd.Series([1.0 if pd.notna(value) else float("nan") for value in numeric], index=numeric.index)
    adjusted = numeric * metric_direction(metric)
    return (adjusted - adjusted.min()) / (adjusted.max() - adjusted.min())


def score_runs(df: pd.DataFrame, metrics: list[str] | None = None) -> pd.DataFrame:
    frame = attach_run_labels(df)
    metrics = available_metrics(frame, metrics or SCORECARD_METRICS)
    if not metrics:
        frame["composite_score"] = 0.0
        frame["efficiency_score"] = 0.0
        frame["retrieval_index"] = 0.0
        frame["generation_index"] = 0.0
        frame["operations_index"] = 0.0
        return frame

    for metric in metrics:
        frame[f"{metric}__norm"] = normalize_metric_series(frame[metric], metric)

    retrieval_norms = [f"{metric}__norm" for metric in metrics if metric_group(metric) == "Retrieval"]
    generation_norms = [f"{metric}__norm" for metric in metrics if metric_group(metric) == "Generation"]
    operations_norms = [f"{metric}__norm" for metric in metrics if metric_group(metric) == "Operations"]
    efficiency_norms = [f"{metric}__norm" for metric in metrics if metric in {"hit@3", "avg_effective_generation_score", "p95_latency"}]

    frame["retrieval_index"] = frame[retrieval_norms].mean(axis=1).fillna(0.0).round(4) if retrieval_norms else 0.0
    frame["generation_index"] = frame[generation_norms].mean(axis=1).fillna(0.0).round(4) if generation_norms else 0.0
    frame["operations_index"] = frame[operations_norms].mean(axis=1).fillna(0.0).round(4) if operations_norms else 0.0
    frame["composite_score"] = frame[[f"{metric}__norm" for metric in metrics]].mean(axis=1).fillna(0.0).round(4)
    frame["efficiency_score"] = frame[efficiency_norms].mean(axis=1).fillna(0.0).round(4) if efficiency_norms else frame["composite_score"]
    return frame


def pareto_frontier(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    frontier_rows: list[pd.Series] = []
    filtered = df[[x_col, y_col]].dropna()
    if filtered.empty:
        return df.iloc[0:0]
    ordered = df.dropna(subset=[x_col, y_col]).sort_values(x_col, ascending=True)
    best_y = float("-inf")
    for _, row in ordered.iterrows():
        y_value = float(row[y_col])
        if y_value >= best_y:
            frontier_rows.append(row)
            best_y = y_value
    if not frontier_rows:
        return ordered.iloc[0:0]
    return pd.DataFrame(frontier_rows)


def build_metric_long_frame(df: pd.DataFrame, metrics: list[str], normalized: bool = False) -> pd.DataFrame:
    frame = attach_run_labels(df)
    rows: list[dict[str, Any]] = []
    for metric in available_metrics(frame, metrics):
        normalized_series = normalize_metric_series(frame[metric], metric)
        for idx, row in frame.iterrows():
            raw_value = pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0]
            if pd.isna(raw_value):
                continue
            rows.append(
                {
                    "run_label": row.get("run_label"),
                    "plot_run_label": row.get("plot_run_label"),
                    "metric": metric,
                    "metric_label": metric_label(metric),
                    "metric_group": metric_group(metric),
                    "raw_value": float(raw_value),
                    "display_value": format_metric_value(metric, raw_value),
                    "normalized_value": float(normalized_series.loc[idx]) if pd.notna(normalized_series.loc[idx]) else float("nan"),
                    "chart_value": float(normalized_series.loc[idx]) if normalized else float(raw_value),
                }
            )
    return pd.DataFrame(rows)


def build_delta_frames(df: pd.DataFrame, baseline_run_label: str, metrics: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = attach_run_labels(df)
    metrics = available_metrics(frame, metrics)
    if frame.empty or not metrics:
        return empty_summary_frame(), empty_detail_frame()

    baseline = frame.loc[frame["run_label"] == baseline_run_label]
    if baseline.empty:
        baseline = frame.iloc[[0]]
    baseline_row = baseline.iloc[0]

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        wins = 0
        losses = 0
        ties = 0
        biggest_gain_metric = None
        biggest_gain_value = float("-inf")
        biggest_loss_metric = None
        biggest_loss_value = float("inf")

        for metric in metrics:
            raw_value = pd.to_numeric(pd.Series([row.get(metric)]), errors="coerce").iloc[0]
            baseline_value = pd.to_numeric(pd.Series([baseline_row.get(metric)]), errors="coerce").iloc[0]
            if pd.isna(raw_value) or pd.isna(baseline_value):
                continue
            delta = float(raw_value - baseline_value)
            benefit_delta = delta * metric_direction(metric)
            if row["run_label"] != baseline_row["run_label"]:
                if benefit_delta > 1e-9:
                    wins += 1
                elif benefit_delta < -1e-9:
                    losses += 1
                else:
                    ties += 1
            if benefit_delta > biggest_gain_value:
                biggest_gain_metric = metric
                biggest_gain_value = benefit_delta
            if benefit_delta < biggest_loss_value:
                biggest_loss_metric = metric
                biggest_loss_value = benefit_delta
            detail_rows.append(
                {
                    "run_label": row["run_label"],
                    "plot_run_label": row["plot_run_label"],
                    "metric": metric,
                    "metric_label": metric_label(metric),
                    "metric_group": metric_group(metric),
                    "raw_value": float(raw_value),
                    "baseline_value": float(baseline_value),
                    "delta": delta,
                    "benefit_delta": benefit_delta,
                    "delta_display": format_metric_delta(metric, delta),
                    "raw_display": format_metric_value(metric, raw_value),
                    "baseline_display": format_metric_value(metric, baseline_value),
                    "is_baseline": row["run_label"] == baseline_row["run_label"],
                }
            )

        summary_rows.append(
            {
                "run_id": row.get("run_id"),
                "label": row.get("label"),
                "run_label": row["run_label"],
                "plot_run_label": row["plot_run_label"],
                "composite_score": row.get("composite_score", 0.0),
                "efficiency_score": row.get("efficiency_score", 0.0),
                "wins_vs_baseline": wins,
                "losses_vs_baseline": losses,
                "ties_vs_baseline": ties,
                "net_improvement": wins - losses,
                "biggest_gain": metric_label(biggest_gain_metric) if biggest_gain_metric else "—",
                "biggest_gain_value": 0.0 if biggest_gain_metric is None or biggest_gain_value == float("-inf") else biggest_gain_value,
                "biggest_loss": metric_label(biggest_loss_metric) if biggest_loss_metric else "—",
                "biggest_loss_value": 0.0 if biggest_loss_metric is None or biggest_loss_value == float("inf") else biggest_loss_value,
                "is_baseline": row["run_label"] == baseline_row["run_label"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)
    if summary_df.empty:
        summary_df = empty_summary_frame()
    else:
        summary_df = summary_df.reindex(columns=SUMMARY_FRAME_COLUMNS)
    if detail_df.empty:
        detail_df = empty_detail_frame()
    else:
        detail_df = detail_df.reindex(columns=DETAIL_FRAME_COLUMNS)
    return summary_df, detail_df
