"""Plotly chart builders for the UI."""

from __future__ import annotations

from typing import Dict, List

import plotly.graph_objects as go


def build_radar_chart(
    original_metrics: Dict[str, float],
    edited_metrics: Dict[str, float],
    metric_standards: Dict[str, Dict[str, float]] | None = None,
) -> go.Figure:
    labels = list(edited_metrics.keys())
    edited_values = [edited_metrics[label] for label in labels]
    original_values = [original_metrics.get(label, 0) for label in labels]

    def format_standard(value: object) -> str:
        if isinstance(value, (int, float)):
            return f"{value:.2f}"
        return "--"

    customdata = []
    for label in labels:
        human_value = "--"
        ai_value = "--"
        if metric_standards and label in metric_standards:
            standards = metric_standards.get(label, {})
            human_value = format_standard(standards.get("human"))
            ai_value = format_standard(standards.get("ai"))
        customdata.append([human_value, ai_value])

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=original_values + original_values[:1],
            theta=labels + labels[:1],
            customdata=customdata + customdata[:1],
            fill="toself",
            fillcolor="rgba(22, 163, 74, 0.45)",
            line=dict(color="#16a34a", width=2),
            name="Original",
            showlegend=True,
            hovertemplate=(
                "<b>%{theta}</b><br>"
                "Original: %{r:.2f}<br>"
                "Human: %{customdata[0]} | AI: %{customdata[1]}"
                "<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=edited_values + edited_values[:1],
            theta=labels + labels[:1],
            customdata=customdata + customdata[:1],
            fill="toself",
            fillcolor="rgba(239, 68, 68, 0.45)",
            line=dict(color="#ef4444", width=2),
            name="AI-Edited",
            showlegend=True,
            hovertemplate=(
                "<b>%{theta}</b><br>"
                "AI-Edited: %{r:.2f}<br>"
                "Human: %{customdata[0]} | AI: %{customdata[1]}"
                "<extra></extra>"
            ),
        )
    )

    if metric_standards and labels:
        human_values = [metric_standards.get(label, {}).get("human", 0) for label in labels]
        ai_values = [metric_standards.get(label, {}).get("ai", 0) for label in labels]
        fig.add_trace(
            go.Scatterpolar(
                r=human_values + human_values[:1],
                theta=labels + labels[:1],
                mode="lines",
                line=dict(color="#16a34a", width=2, dash="dash"),
                name="Human standard",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatterpolar(
                r=ai_values + ai_values[:1],
                theta=labels + labels[:1],
                mode="lines",
                line=dict(color="#ef4444", width=2, dash="dot"),
                name="AI standard",
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.06,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
        margin=dict(l=20, r=20, t=30, b=20),
        polar=dict(radialaxis=dict(range=[0, 100], showticklabels=True)),
    )
    return fig


def build_line_chart(sentence_lengths: Dict[str, List[int]]) -> go.Figure:
    fig = go.Figure()
    for label, values in sentence_lengths.items():
        color = "#2563eb"
        label_lower = label.lower()
        if label_lower in {"original", "ai source"}:
            color = "#16a34a"
        elif label_lower in {"edited", "writer rewrite", "ai-edited"}:
            color = "#ef4444"
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(values) + 1)),
                y=values,
                mode="lines+markers",
                name=label,
                line=dict(color=color, shape="spline"),
                marker=dict(color=color),
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
                x=values,
                y=labels,
                orientation="h",
                marker_color="#2563eb",
            )
        ]
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(range=[0, 100]),
    )
    return fig


def build_metric_standards_chart(metric_standards: Dict[str, Dict[str, float]]) -> go.Figure:
    labels = list(metric_standards.keys())
    human_values = [metric_standards[label].get("human", 0) for label in labels]
    ai_values = [metric_standards[label].get("ai", 0) for label in labels]

    fig = go.Figure(
        data=[
            go.Bar(
                x=human_values,
                y=labels,
                orientation="h",
                name="Human Standard",
                marker_color="#16a34a",
            ),
            go.Bar(
                x=ai_values,
                y=labels,
                orientation="h",
                name="AI Standard",
                marker_color="#ef4444",
            ),
        ]
    )
    fig.update_layout(
        barmode="group",
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def build_pie_chart(categories: Dict[str, int], colors: List[str] | None = None) -> go.Figure:
    total = sum(categories.values())
    if total == 0:
        labels = ["None"]
        values = [1]
    else:
        labels = list(categories.keys())
        values = list(categories.values())
    pie = go.Pie(
        labels=labels,
        values=values,
        hole=0.45,
    )
    if colors:
        pie.marker = dict(colors=colors)
    fig = go.Figure(data=[pie])
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


def build_mini_gauge(score: float, bar_color: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge",
            value=score,
            gauge={
                "axis": {"range": [0, 100], "visible": False},
                "bar": {"color": bar_color},
                "steps": [
                    {"range": [0, 100], "color": "rgba(15, 23, 42, 0.06)"},
                ],
            },
        )
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=80)
    return fig


def render_latex_to_image(latex_str: str) -> bytes:
    """Render a LaTeX string to a PNG image using Plotly/Kaleido."""
    import plotly.io as pio
    
    # Ensure it has $ for plotly MathJax if not already there
    if not latex_str.startswith("$"):
        latex_str = f"${latex_str}$"
        
    fig = go.Figure()
    fig.add_annotation(
        text=latex_str,
        showarrow=False,
        font=dict(size=44, color="black"),
        xref="paper", yref="paper",
        x=0.5, y=0.5
    )
    fig.update_layout(
        width=1000,
        height=280,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return pio.to_image(fig, format="png", scale=2)
