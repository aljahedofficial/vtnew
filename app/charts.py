"""Plotly chart builders for the UI."""

from __future__ import annotations

from typing import Dict, List

import plotly.graph_objects as go


def build_radar_chart(metrics: Dict[str, float]) -> go.Figure:
    labels = list(metrics.keys())
    values = list(metrics.values())

    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=values + values[:1],
                theta=labels + labels[:1],
                fill="toself",
                line_color="#2563eb",
            )
        ]
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(l=20, r=20, t=30, b=20),
        polar=dict(radialaxis=dict(range=[0, 100], showticklabels=True)),
    )
    return fig


def build_line_chart(sentence_lengths: Dict[str, List[int]]) -> go.Figure:
    fig = go.Figure()
    for label, values in sentence_lengths.items():
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(values) + 1)),
                y=values,
                mode="lines+markers",
                name=label,
            )
        )
    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Sentence",
        yaxis_title="Words",
    )
    return fig


def build_bar_chart(components: Dict[str, float]) -> go.Figure:
    labels = list(components.keys())
    values = list(components.values())
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color="#2563eb",
            )
        ]
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        yaxis=dict(range=[0, 100]),
    )
    return fig


def build_pie_chart(categories: Dict[str, int]) -> go.Figure:
    total = sum(categories.values())
    if total == 0:
        labels = ["None"]
        values = [1]
    else:
        labels = list(categories.keys())
        values = list(categories.values())
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.45,
            )
        ]
    )
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))
    return fig


def build_gauge_chart(score: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2563eb"},
                "steps": [
                    {"range": [0, 40], "color": "#fca5a5"},
                    {"range": [40, 60], "color": "#fdba74"},
                    {"range": [60, 80], "color": "#fde047"},
                    {"range": [80, 100], "color": "#86efac"},
                ],
            },
        )
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    return fig
