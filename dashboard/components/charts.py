"""Interactive Plotly visualizations for SOC Overview, Analytics, and Threat Map."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dashboard.components.cards import apply_soc_plotly_theme
from src.detection.threat_scoring import THREAT_COLORS


def plot_attack_class_donut(df_logs: pd.DataFrame) -> go.Figure:
    """Donut chart (hole=0.45) of traffic breakdown by predicted attack class."""
    counts = df_logs["attack_class"].value_counts().reset_index()
    counts.columns = ["attack_class", "count"]
    fig = px.pie(
        counts,
        names="attack_class",
        values="count",
        hole=0.45,
        title="Traffic Distribution by Predicted Class",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return apply_soc_plotly_theme(fig, height=360)


def plot_threat_level_donut(df_logs: pd.DataFrame) -> go.Figure:
    """Donut chart (hole=0.55) of detections by threat severity tier."""
    counts = df_logs["threat_level"].value_counts().reset_index()
    counts.columns = ["threat_level", "count"]
    fig = px.pie(
        counts,
        names="threat_level",
        values="count",
        hole=0.55,
        title="Alert Distribution by Threat Tier",
        color="threat_level",
        color_discrete_map=THREAT_COLORS,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return apply_soc_plotly_theme(fig, height=360)


def plot_timeline_area(df_logs: pd.DataFrame) -> go.Figure:
    """1-minute bucketed time-series area chart grouped by threat level."""
    work = df_logs.copy()
    work["ts_parsed"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work.dropna(subset=["ts_parsed"])
    if work.empty:
        fig = go.Figure()
        fig.update_layout(title="Detection Timeline (No parseable timestamps)")
        return apply_soc_plotly_theme(fig, height=350)

    work["minute"] = work["ts_parsed"].dt.floor("min")
    grouped = work.groupby(["minute", "threat_level"]).size().reset_index(name="count")
    fig = px.area(
        grouped,
        x="minute",
        y="count",
        color="threat_level",
        color_discrete_map=THREAT_COLORS,
        title="Detection Volume Over Time (1-Minute Buckets)",
    )
    return apply_soc_plotly_theme(fig, height=350)


def plot_confidence_histogram(df_logs: pd.DataFrame) -> go.Figure:
    """Stacked 40-bin histogram of prediction confidence partitioned by threat level."""
    fig = px.histogram(
        df_logs,
        x="confidence",
        color="threat_level",
        nbins=40,
        barmode="stack",
        color_discrete_map=THREAT_COLORS,
        title="Confidence Score Distribution by Threat Tier",
    )
    return apply_soc_plotly_theme(fig, height=350)


def plot_top_attacking_ips(df_logs: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of top-15 attacking source IPs colored by max confidence."""
    attacks = df_logs[df_logs["attack_class"].str.upper() != "BENIGN"].copy()
    if attacks.empty:
        fig = go.Figure()
        fig.update_layout(title="Top Attacking Source IPs (No attacks in view)")
        return apply_soc_plotly_theme(fig, height=380)

    agg = (
        attacks.groupby("src_ip")
        .agg(
            attack_count=("attack_class", "count"),
            max_confidence=("confidence", "max"),
        )
        .reset_index()
        .sort_values("attack_count", ascending=False)
        .head(15)
    )
    agg = agg.sort_values("attack_count", ascending=True)
    fig = px.bar(
        agg,
        x="attack_count",
        y="src_ip",
        orientation="h",
        color="max_confidence",
        color_continuous_scale="RdYlGn_r",
        range_color=[0.4, 1.0],
        title="Top 15 Attacking Source IPs (Length=Volume, Color=Max Confidence)",
    )
    return apply_soc_plotly_theme(fig, height=400)


def plot_top_destination_ports(df_logs: pd.DataFrame) -> go.Figure:
    """Bar chart of top-12 targeted destination ports across all flows."""
    ports = df_logs["dst_port"].astype(str).value_counts().head(12).reset_index()
    ports.columns = ["dst_port", "flows"]
    fig = px.bar(
        ports,
        x="dst_port",
        y="flows",
        color="flows",
        color_continuous_scale="Blues",
        title="Top 12 Targeted Destination Ports",
    )
    return apply_soc_plotly_theme(fig, height=400)


def plot_attack_port_heatmap(df_logs: pd.DataFrame) -> go.Figure:
    """Heatmap cross-tabulating attack classes against top-8 destination ports."""
    if df_logs.empty:
        return apply_soc_plotly_theme(go.Figure(), height=360)
    top_ports = df_logs["dst_port"].value_counts().head(8).index.tolist()
    sub = df_logs[df_logs["dst_port"].isin(top_ports)].copy()
    sub["dst_port_str"] = sub["dst_port"].astype(str)
    piv = pd.pivot_table(
        sub,
        index="attack_class",
        columns="dst_port_str",
        values="confidence",
        aggfunc="count",
        fill_value=0,
    )
    fig = px.imshow(
        piv,
        color_continuous_scale="Reds",
        aspect="auto",
        text_auto=True,
        title="Attack Category vs Top-8 Destination Ports Matrix",
    )
    return apply_soc_plotly_theme(fig, height=380)
