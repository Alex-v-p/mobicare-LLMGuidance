from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st


def render(campaigns: list[dict]) -> None:
    st.subheader("Campaign detail")
    if not campaigns:
        st.info("No campaign artifacts found.")
        return

    options = [item.get("label") or item.get("campaign_id") for item in campaigns]
    selected = st.selectbox("Campaign", options)
    campaign = next(item for item in campaigns if (item.get("label") or item.get("campaign_id")) == selected)

    top1, top2, top3, top4 = st.columns(4)
    with top1:
        st.metric("Completed runs", int(campaign.get("run_count") or 0))
    with top2:
        st.metric("Failed runs", int(campaign.get("failed_run_count") or 0))
    with top3:
        st.metric("Groups", len(((campaign.get("campaign_summary") or {}).get("groups") or {})))
    with top4:
        st.metric("Artifact", campaign.get("artifact_path", ""))

    st.markdown("### Campaign summary")
    st.json(campaign.get("campaign_summary") or {})

    runs_df = pd.DataFrame(campaign.get("runs") or [])
    if not runs_df.empty:
        st.markdown("### Runs")
        st.dataframe(runs_df, use_container_width=True, hide_index=True)

        averages = (campaign.get("campaign_summary") or {}).get("average_normalized_metrics") or {}
        if averages:
            avg_df = pd.DataFrame(
                [{"metric": key, "value": value} for key, value in averages.items()]
            ).sort_values(by="value", ascending=False)
            chart = (
                alt.Chart(avg_df.head(15))
                .mark_bar()
                .encode(x=alt.X("metric:N", sort="-y"), y=alt.Y("value:Q"), tooltip=["metric", "value"])
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)

    failed_df = pd.DataFrame(campaign.get("failed_runs") or [])
    if not failed_df.empty:
        st.markdown("### Failed runs")
        st.dataframe(failed_df, use_container_width=True, hide_index=True)
