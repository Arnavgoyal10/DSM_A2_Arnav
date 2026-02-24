import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

MEDIA_ENRICHED_CSV = BASE_DIR / "stage4_media_enriched.csv"
IMPACT_CSV = BASE_DIR / "emdat_events_impact.csv"
MEDIA_EVENTS_CSV = BASE_DIR / "reliefweb_media_events.csv"
EVENT_SUMMARY_JSON = BASE_DIR / "stage4_event_summary.json"

"""
print(
    "Defining load_data: read the enriched media, impact, and summary JSON files "
    "produced in earlier stages into pandas DataFrames and dicts for visualization."
)
"""


def load_data() -> Dict[str, Any]:
    media_df = pd.read_csv(
        MEDIA_ENRICHED_CSV, parse_dates=["publication_date", "event_date"]
    )
    impact_df = pd.read_csv(IMPACT_CSV)
    media_events_df = pd.read_csv(MEDIA_EVENTS_CSV, parse_dates=["publication_date"])

    if EVENT_SUMMARY_JSON.exists():
        summary = json.loads(EVENT_SUMMARY_JSON.read_text(encoding="utf-8"))
    else:
        summary = {}

    return {
        "media_enriched": media_df,
        "impact": impact_df,
        "media_events": media_events_df,
        "summary": summary,
    }


"""
print(
    "Defining build_dual_timeline_figure: aggregate daily news volume per event label "
    "and return a Plotly line chart overlaying Event A vs Event B over time."
)
"""


def build_dual_timeline_figure(media_enriched: pd.DataFrame) -> go.Figure:
    df = media_enriched.copy()
    df = df.dropna(subset=["publication_date"])
    df["pub_date"] = df["publication_date"].dt.date

    daily = (
        df.groupby(["event_label", "pub_date"], as_index=False)
        .size()
        .rename(columns={"size": "daily_report_count"})
    )

    fig = px.line(
        daily,
        x="pub_date",
        y="daily_report_count",
        color="event_label",
        markers=True,
        title="Dual Timeline – Daily News Volume by Event",
        labels={"pub_date": "Date", "daily_report_count": "Number of reports"},
    )
    fig.update_layout(legend_title_text="Event")
    return fig


"""
print(
    "Defining build_resilience_radar_figure: compute magnitude, population exposure, "
    "media coverage, and vulnerability proxy per event and plot them on a radar chart."
)
"""


def build_resilience_radar_figure(
    impact_df: pd.DataFrame, media_events_df: pd.DataFrame
) -> go.Figure:
    coverage_counts = media_events_df["event_label"].value_counts().to_dict()

    metrics: List[Dict[str, Any]] = []
    for _, row in impact_df.iterrows():
        label = row["event_label"]
        magnitude = row.get("magnitude", None)
        population_exposed = row.get("total_affected", None)
        total_reports = coverage_counts.get(label, 0)
        econ_per_capita = row.get("economic_damage_per_capita", None)

        metrics.append(
            {
                "event_label": label,
                "Magnitude": float(magnitude) if pd.notna(magnitude) else 0.0,
                "Population exposure": (
                    float(population_exposed) if pd.notna(population_exposed) else 0.0
                ),
                "Media coverage": float(total_reports),
                "Vulnerability proxy": (
                    float(econ_per_capita) if pd.notna(econ_per_capita) else 0.0
                ),
            }
        )

    radar_df = pd.DataFrame(metrics)

    axes = ["Magnitude", "Population exposure", "Media coverage", "Vulnerability proxy"]
    norm_df = radar_df.copy()
    for col in axes:
        col_max = norm_df[col].max()
        if col_max > 0:
            norm_df[col] = norm_df[col] / col_max
        else:
            norm_df[col] = 0.0

    categories = axes
    fig = go.Figure()
    for _, row in norm_df.iterrows():
        values = [row[c] for c in categories]
        values.append(values[0])
        cats_closed = categories + [categories[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=cats_closed,
                fill="toself",
                name=row["event_label"],
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Resilience Radar – Relative Magnitude, Exposure, Coverage, Vulnerability",
    )
    return fig


"""
print(
    "Defining main: Streamlit app entry point that loads the data, renders the dual "
    "timeline and resilience radar, and exposes simple controls to inspect each event."
)
"""


def main():
    st.set_page_config(page_title="Disaster Pulse Dashboard", layout="wide")

    st.title("Disaster Pulse Dashboard")
    st.markdown(
        """
This dashboard compares **Event A (Indonesia 2018)** and **Event B (Myanmar 2025)**
across media response and impact dimensions.
"""
    )

    data = load_data()
    media_enriched = data["media_enriched"]
    impact_df = data["impact"]
    media_events_df = data["media_events"]
    summary = data["summary"]

    st.subheader("Dual Timeline – Media Volume Over Time")
    fig_timeline = build_dual_timeline_figure(media_enriched)
    st.plotly_chart(fig_timeline, use_container_width=True)

    st.subheader("Resilience Radar – Magnitude, Exposure, Coverage, Vulnerability")
    fig_radar = build_resilience_radar_figure(impact_df, media_events_df)
    st.plotly_chart(fig_radar, use_container_width=True)

    st.sidebar.title("Event Details")
    event_labels = impact_df["event_label"].tolist()
    selected = st.sidebar.selectbox("Select event", event_labels)

    st.sidebar.markdown("**Impact metrics (EM-DAT/HDX):**")
    impact_row = impact_df.set_index("event_label").loc[selected].to_dict()
    for k, v in impact_row.items():
        st.sidebar.write(f"- {k}: {v}")

    if summary and selected in summary:
        st.sidebar.markdown("**Temporal & Sentiment Summary:**")
        ev = summary[selected]
        temporal = ev.get("temporal") or {}
        sentiment = ev.get("sentiment") or {}
        st.sidebar.write(f"- Event date: {temporal.get('event_date')}")
        st.sidebar.write(f"- Media peak date: {temporal.get('media_peak_date')}")
        st.sidebar.write(f"- ΔT (days): {temporal.get('delta_t_days')}")
        st.sidebar.write(
            f"- Mean headline sentiment: {sentiment.get('headline_sentiment_mean')}"
        )
        st.sidebar.write(
            f"- Sentiment volatility (std): {sentiment.get('headline_sentiment_std')}"
        )

        st.sidebar.markdown("**Top organizations/entities:**")
        top_orgs = (ev.get("top_entities") or {}).get("org_like_top20") or []
        for text, count in top_orgs[:10]:
            st.sidebar.write(f"- {text} ({count} mentions)")

        ec = ev.get("entity_classification") or {}
        if ec.get("ngo") or ec.get("government") or ec.get("private"):
            st.sidebar.markdown("**Entities by type (NGO / Govt / Private):**")
            for label, key in [
                ("NGOs", "ngo"),
                ("Government", "government"),
                ("Private", "private"),
            ]:
                items = ec.get(key) or []
                if items:
                    st.sidebar.caption(label)
                    for text, count in items[:5]:
                        st.sidebar.write(f"  - {text} ({count})")

        ex = ev.get("extracted_impact") or {}
        if ex.get("deaths") or ex.get("losses") or ex.get("relief_funds"):
            st.sidebar.markdown("**Extracted from text (deaths / losses / funds):**")
            for key, title in [
                ("deaths", "Deaths"),
                ("losses", "Losses"),
                ("relief_funds", "Relief funds"),
            ]:
                blob = ex.get(key) or {}
                total = blob.get("total_mentions", 0)
                top = (blob.get("top_values") or [])[:3]
                if total or top:
                    st.sidebar.caption(f"{title}: {total} mentions")
                    for val, cnt in top:
                        st.sidebar.write(f"  - {val} ({cnt})")


if __name__ == "__main__":
    print(
        "Launching Stage 6 Disaster Pulse dashboard with Streamlit.\n"
        "Run this script via: streamlit run stage6_dashboard.py"
    )
    main()
