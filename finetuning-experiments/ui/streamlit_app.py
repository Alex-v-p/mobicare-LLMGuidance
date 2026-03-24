from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure project root is importable when Streamlit runs this file directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.views import campaign_detail, case_detail, compare_runs, endpoint_playground, overview
from ui.views.common import DEFAULT_RUNS_ROOT, build_runs_dataframe, discover_campaigns, discover_runs


st.set_page_config(page_title="LLM Guidance Benchmarks", layout="wide")
st.title("LLM Guidance Experiment Dashboard")

root = st.sidebar.text_input("Run artifacts directory", value=str(DEFAULT_RUNS_ROOT))
runs, _ = discover_runs(root)
run_df = build_runs_dataframe(runs)
campaigns = discover_campaigns(root)

if run_df.empty:
    st.warning("No run artifacts found.")
    st.stop()

run_df = run_df.copy()
run_df["selector_label"] = run_df["label"].fillna("") + " | " + run_df["run_id"].fillna("")
selector_options = run_df["run_id"].tolist()
default_compare = selector_options[: min(3, len(selector_options))]
selected_run_ids = st.sidebar.multiselect(
    "Runs to compare",
    options=selector_options,
    default=default_compare,
    format_func=lambda run_id: run_df.loc[run_df["run_id"] == run_id, "selector_label"].iloc[0],
)
selected_df = run_df[run_df["run_id"].isin(selected_run_ids)].copy()

sidebar_col1, sidebar_col2 = st.sidebar.columns(2)
with sidebar_col1:
    st.metric("Run summaries", len(run_df))
with sidebar_col2:
    st.metric("Campaigns", len(campaigns))

summary_only_count = int(run_df["summary_path"].notna().sum())
full_artifact_count = int(run_df["artifact_path"].notna().sum())
st.sidebar.caption(f"Summary-first loading enabled: {summary_only_count} summaries, {full_artifact_count} full artifacts")


tab_overview, tab_compare, tab_case, tab_campaign, tab_playground = st.tabs(
    ["Overview", "Compare runs", "Per-case analysis", "Campaign detail", "Endpoint playground"]
)

with tab_overview:
    overview.render(run_df)

with tab_compare:
    compare_runs.render(selected_df)

with tab_case:
    case_detail.render(run_df, root)

with tab_campaign:
    campaign_detail.render(campaigns)

with tab_playground:
    endpoint_playground.render()
