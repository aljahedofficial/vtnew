from __future__ import annotations

from collections import Counter
from pathlib import Path
import difflib
import html
import json
import math
import re
from typing import Dict, List, Tuple

import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

try:
	from streamlit_local_storage import LocalStorage
except ImportError:
	LocalStorage = None

try:
	from docx import Document
except ImportError:
	Document = None

try:
	from pypdf import PdfReader
except ImportError:
	PdfReader = None

from app.analysis import CalibrationStandards, analyze_multitext
from app.charts import (
	build_bar_chart,
	build_gauge_chart,
	build_line_chart,
	build_metric_standards_chart,
	build_pie_chart,
	build_radar_chart,
)
try:
	from app.charts import build_mini_gauge
except ImportError:
	def build_mini_gauge(score: float, bar_color: str) -> go.Figure:
		return build_gauge_chart(score)
from app.state import (
	AI_TEXT_KEY,
	CALIBRATION_KEY,
	HUMAN_TEXT_KEY,
	PARAPHRASE_TEXT_KEY,
	PROMPT_TEXT_KEY,
	SOURCE_TEXT_KEY,
)


ASSETS_DIR = Path(__file__).parent / "assets"
LOGO_PATH = ASSETS_DIR / "logo.svg"
FAVICON_PATH = ASSETS_DIR / "favicon.svg"

THEMES: Dict[str, Dict[str, str]] = {
	"Modern SaaS Blue": {
		"bg": "#F8FAFC",
		"surface": "#FFFFFF",
		"secondary": "#EEF2F7",
		"text": "#0F172A",
		"text_secondary": "#475569",
		"text_muted": "#94A3B8",
		"accent": "#2563EB",
		"accent_hover": "#1D4ED8",
		"accent_subtle": "#93C5FD",
		"success": "#16A34A",
		"warning": "#F59E0B",
		"danger": "#DC2626",
		"divider": "#E2E8F0",
		"chart_original": "#16A34A",
		"chart_edited": "#DC2626",
		"chart_negotiated": "#2563EB",
	},
	"Dark Professional": {
		"bg": "#0B1120",
		"surface": "#111827",
		"secondary": "#1F2933",
		"text": "#E5E7EB",
		"text_secondary": "#9CA3AF",
		"text_muted": "#6B7280",
		"accent": "#38BDF8",
		"accent_hover": "#0EA5E9",
		"accent_subtle": "#22D3EE",
		"success": "#34D399",
		"warning": "#FBBF24",
		"danger": "#F87171",
		"divider": "#1F2933",
		"chart_original": "#34D399",
		"chart_edited": "#F87171",
		"chart_negotiated": "#38BDF8",
	},
	"Neutral Corporate": {
		"bg": "#F5F7FA",
		"surface": "#FFFFFF",
		"secondary": "#F5F7FA",
		"text": "#111827",
		"text_secondary": "#374151",
		"text_muted": "#6B7280",
		"accent": "#374151",
		"accent_hover": "#4B5563",
		"accent_subtle": "#E5E7EB",
		"success": "#0F766E",
		"warning": "#B45309",
		"danger": "#991B1B",
		"divider": "#E5E7EB",
		"chart_original": "#0F766E",
		"chart_edited": "#991B1B",
		"chart_negotiated": "#1D4ED8",
	},
	"Productivity Green": {
		"bg": "#F7FDF9",
		"surface": "#FFFFFF",
		"secondary": "#ECFDF5",
		"text": "#064E3B",
		"text_secondary": "#065F46",
		"text_muted": "#6EE7B7",
		"accent": "#16A34A",
		"accent_hover": "#15803D",
		"accent_subtle": "#22C55E",
		"success": "#16A34A",
		"warning": "#F97316",
		"danger": "#DC2626",
		"divider": "#D1FAE5",
		"chart_original": "#16A34A",
		"chart_edited": "#DC2626",
		"chart_negotiated": "#2563EB",
	},
	"Elegant Tech Purple": {
		"bg": "#FAFAFF",
		"surface": "#FFFFFF",
		"secondary": "#F3F4FF",
		"text": "#1E1B4B",
		"text_secondary": "#4338CA",
		"text_muted": "#A5B4FC",
		"accent": "#6366F1",
		"accent_hover": "#4F46E5",
		"accent_subtle": "#A78BFA",
		"success": "#22C55E",
		"warning": "#F59E0B",
		"danger": "#EF4444",
		"divider": "#E0E7FF",
		"chart_original": "#22C55E",
		"chart_edited": "#EF4444",
		"chart_negotiated": "#6366F1",
	},
}

METRICS = [
	{
		"key": "burstiness",
		"label": "Burstiness",
		"caption": "Rhythm & Variation",
		"description": "Measures sentence length variation.",
	},
	{
		"key": "lexical_diversity",
		"label": "Lexical Diversity",
		"caption": "Vocabulary Range",
		"description": "Measures vocabulary richness and repetition.",
	},
	{
		"key": "syntactic_complexity",
		"label": "Syntactic Complexity",
		"caption": "Sentence Structure",
		"description": "Measures subordination and structural depth.",
	},
	{
		"key": "ai_ism_likelihood",
		"label": "AI-ism Likelihood",
		"caption": "Formulaic Phrases",
		"description": "Detects AI-like phrases and patterns.",
	},
	{
		"key": "function_word_ratio",
		"label": "Function Word Ratio",
		"caption": "Connector Words",
		"description": "Measures density of small connector words.",
	},
	{
		"key": "discourse_marker_density",
		"label": "Discourse Markers",
		"caption": "Signposting",
		"description": "Measures discourse markers like however or therefore.",
	},
	{
		"key": "information_density",
		"label": "Information Density",
		"caption": "Content Substance",
		"description": "Measures ratio of content words to total words.",
	},
	{
		"key": "epistemic_hedging",
		"label": "Epistemic Hedging",
		"caption": "Uncertainty Markers",
		"description": "Measures hedging language and caution markers.",
	},
]

PRESET_SETS = [
	{
		"name": "Preset Set 1",
		"human": (
			"I drafted this paragraph after reviewing my notes and outlining the main idea. "
			"The goal is to explain the concept in plain language and keep the flow natural. "
			"I focused on clarity and avoided adding extra claims that I could not support."
		),
		"source": (
			"The source material argues that consistent feedback loops improve learning outcomes. "
			"It highlights that short, timely responses help learners adjust their approach. "
			"The author also notes that feedback is most effective when tied to specific actions."
		),
		"prompt": (
			"Rewrite the source material in a concise academic paragraph. Maintain key claims, "
			"use formal tone, and avoid bullet points."
		),
		"ai": (
			"Feedback loops are central to learning because they enable rapid adjustment and reinforcement. "
			"Short, timely responses are especially effective when they target specific behaviors. "
			"Accordingly, well-designed feedback mechanisms improve outcomes and sustain progress."
		),
		"paraphrase": (
			"Learning improves when people receive quick, specific feedback that connects to what they just did. "
			"The source emphasizes that targeted responses help learners correct course and stay engaged. "
			"This makes feedback loops a practical tool for steady improvement."
		),
	},
	{
		"name": "Preset Set 2",
		"human": (
			"I wrote this section to summarize the discussion in a way that sounds like me. "
			"I kept the sentences short and varied so the paragraph feels conversational. "
			"The point is to show the logic without turning it into a list."
		),
		"source": (
			"The report describes how urban green spaces reduce heat and improve air quality. "
			"It notes measurable temperature drops near parks and tree-lined streets. "
			"The authors conclude that planning policy should prioritize canopy coverage."
		),
		"prompt": (
			"Compose an academic summary of the source material. Keep formal tone, preserve evidence, "
			"and avoid introducing new data."
		),
		"ai": (
			"Urban green spaces mitigate heat and improve air quality by increasing canopy coverage. "
			"Studies report measurable temperature reductions near parks and tree corridors. "
			"Therefore, planning policy should prioritize tree cover to enhance environmental outcomes."
		),
		"paraphrase": (
			"The report shows that parks and tree-lined streets cool nearby areas and help clean the air. "
			"Measured temperature drops support the case for more canopy in city design. "
			"It argues that policy should treat tree cover as a core planning priority."
		),
	},
	{
		"name": "Preset Set 3",
		"human": (
			"This paragraph captures my initial interpretation of the findings, not a final conclusion. "
			"I wanted the summary to sound grounded and careful, so I avoided dramatic language. "
			"The phrasing reflects how I would explain it in a draft."
		),
		"source": (
			"The study compares remote and in-person teams and finds similar productivity levels. "
			"It reports higher satisfaction among remote teams when communication norms are clear. "
			"The authors recommend structured check-ins to reduce ambiguity."
		),
		"prompt": (
			"Summarize the findings in a formal paragraph. Emphasize comparison, highlight the "
			"recommendation, and keep it concise."
		),
		"ai": (
			"The study finds that remote and in-person teams show comparable productivity outcomes. "
			"Remote teams report higher satisfaction when communication norms are explicit. "
			"Consequently, the authors recommend structured check-ins to minimize ambiguity."
		),
		"paraphrase": (
			"The research shows no major productivity gap between remote and in-person teams. "
			"Satisfaction improves for remote groups when communication rules are spelled out. "
			"It recommends scheduled check-ins to keep expectations clear."
		),
	},
	{
		"name": "Preset Set 4",
		"human": (
			"I am outlining the argument in a straightforward way so the reasoning stays easy to follow. "
			"I kept the word choice simple and avoided over-explaining. "
			"This is the kind of paragraph I would write before a full revision."
		),
		"source": (
			"The article reviews data privacy practices and warns about opaque data sharing. "
			"It identifies consent fatigue as a factor that reduces meaningful user choice. "
			"The authors call for clearer disclosures and limited default data collection."
		),
		"prompt": (
			"Write a formal summary that keeps the cautionary tone. Do not add new sources."
		),
		"ai": (
			"The article cautions that opaque data sharing undermines user control and privacy. "
			"It links consent fatigue to reduced meaningful choice in digital settings. "
			"Accordingly, the authors call for clearer disclosures and restrained default collection."
		),
		"paraphrase": (
			"The piece warns that unclear data sharing weakens user control and privacy. "
			"It connects consent fatigue with less meaningful choice for users. "
			"The authors argue for clearer disclosures and tighter default collection rules."
		),
	},
]


def load_css() -> None:
	css_path = ASSETS_DIR / "styles.css"
	if css_path.exists():
		st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def apply_theme(theme_name: str) -> None:
	theme = THEMES.get(theme_name, THEMES["Neutral Corporate"])
	style = "\n".join(
		[
			":root {",
			f"  --vt-bg: {theme['bg']};",
			f"  --vt-surface: {theme['surface']};",
			f"  --vt-secondary: {theme['secondary']};",
			f"  --vt-text: {theme['text']};",
			f"  --vt-text-secondary: {theme['text_secondary']};",
			f"  --vt-text-muted: {theme['text_muted']};",
			f"  --vt-accent: {theme['accent']};",
			f"  --vt-accent-hover: {theme['accent_hover']};",
			f"  --vt-accent-subtle: {theme['accent_subtle']};",
			f"  --vt-success: {theme['success']};",
			f"  --vt-warning: {theme['warning']};",
			f"  --vt-danger: {theme['danger']};",
			f"  --vt-divider: {theme['divider']};",
			f"  --vt-chart-original: {theme['chart_original']};",
			f"  --vt-chart-edited: {theme['chart_edited']};",
			f"  --vt-chart-negotiated: {theme['chart_negotiated']};",
			"}",
		]
	)
	st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)


def word_count(text: str) -> int:
	return len([w for w in text.split() if w.strip()])


def format_metric(value: object) -> str:
	if isinstance(value, (int, float)):
		return f"{value:.2f}"
	return str(value)


def _plotly_download_config() -> Dict[str, object]:
	return {
		"toImageButtonOptions": {
			"format": "png",
			"filename": "voicetracer-chart",
			"height": 600,
			"width": 900,
			"scale": 2,
		},
	}


_DIFF_TOKEN_RE = re.compile(r"\s+|[^\s]+")
_WORD_RE = re.compile(r"\b\w+\b")


def _tokenize_for_diff(text: str) -> List[str]:
	return _DIFF_TOKEN_RE.findall(text)


def _build_diff_markup(original_text: str, edited_text: str) -> Tuple[str, str]:
	original_tokens = _tokenize_for_diff(original_text)
	edited_tokens = _tokenize_for_diff(edited_text)
	matcher = difflib.SequenceMatcher(a=original_tokens, b=edited_tokens)
	original_output: List[str] = []
	edited_output: List[str] = []

	def wrap_span(segment: str, class_name: str) -> str:
		if not segment:
			return ""
		return f"<span class=\"{class_name}\">{segment}</span>"

	for tag, i1, i2, j1, j2 in matcher.get_opcodes():
		original_segment = "".join(html.escape(token) for token in original_tokens[i1:i2])
		edited_segment = "".join(html.escape(token) for token in edited_tokens[j1:j2])
		if tag == "equal":
			original_output.append(original_segment)
			edited_output.append(edited_segment)
		elif tag == "delete":
			original_output.append(wrap_span(original_segment, "vt-diff-remove"))
		elif tag == "insert":
			edited_output.append(wrap_span(edited_segment, "vt-diff-add"))
		elif tag == "replace":
			original_output.append(wrap_span(original_segment, "vt-diff-remove"))
			edited_output.append(wrap_span(edited_segment, "vt-diff-add"))

	return "".join(original_output), "".join(edited_output)


def _build_similarity_highlight(
	base_text: str,
	rewrite_text: str,
	highlight_class: str,
) -> str:
	if not base_text or not rewrite_text:
		return html.escape(rewrite_text or "")

	base_words = [w.lower() for w in _WORD_RE.findall(base_text)]
	rewrite_words = [w.lower() for w in _WORD_RE.findall(rewrite_text)]
	base_word_set = set(base_words)

	def build_ngrams(words: List[str], size: int) -> set[str]:
		return {
			" ".join(words[idx : idx + size])
			for idx in range(len(words) - size + 1)
		}

	base_fourgrams = build_ngrams(base_words, 4)
	base_trigrams = build_ngrams(base_words, 3)
	base_bigrams = build_ngrams(base_words, 2)
	word_hits = [False] * len(rewrite_words)
	phrase_hits = [False] * len(rewrite_words)

	for idx in range(len(rewrite_words) - 4 + 1):
		ngram = " ".join(rewrite_words[idx : idx + 4])
		if ngram in base_fourgrams:
			for offset in range(4):
				word_hits[idx + offset] = True
				phrase_hits[idx + offset] = True

	for size, ngrams in ((3, base_trigrams), (2, base_bigrams)):
		if not ngrams:
			continue
		for idx in range(len(rewrite_words) - size + 1):
			ngram = " ".join(rewrite_words[idx : idx + size])
			if ngram in ngrams:
				for offset in range(size):
					word_hits[idx + offset] = True

	for idx, word in enumerate(rewrite_words):
		if word in base_word_set:
			word_hits[idx] = True

	output: List[str] = []
	word_index = 0
	for token in _DIFF_TOKEN_RE.findall(rewrite_text):
		if _WORD_RE.fullmatch(token):
			escaped = html.escape(token)
			if word_index < len(word_hits) and word_hits[word_index]:
				classes = [highlight_class]
				if phrase_hits[word_index]:
					classes.append("vt-highlight-phrase")
				class_attr = " ".join(classes)
				output.append(f"<span class=\"{class_attr}\">{escaped}</span>")
			else:
				output.append(escaped)
			word_index += 1
		else:
			output.append(html.escape(token))

	return "".join(output)


def _render_similarity_panels(source_text: str, ai_text: str, rewrite_text: str) -> None:
	source_base_markup = _build_similarity_highlight(
		rewrite_text,
		source_text,
		"vt-highlight-source",
	)
	source_rewrite_markup = _build_similarity_highlight(
		source_text,
		rewrite_text,
		"vt-highlight-source",
	)
	ai_base_markup = _build_similarity_highlight(
		rewrite_text,
		ai_text,
		"vt-highlight-ai",
	)
	ai_rewrite_markup = _build_similarity_highlight(
		ai_text,
		rewrite_text,
		"vt-highlight-ai",
	)
	left_col, right_col = st.columns(2)
	with left_col:
		st.markdown(
			f"""
			<div class="vt-compare-wrapper">
			  <div class="vt-compare-label">Source vs rewrite (cosine)</div>
			  <div class="vt-similarity-panel">
			    <div class="vt-similarity-block">
			      <div class="vt-compare-label">Source</div>
			      <div class="vt-compare-text">{source_base_markup}</div>
			    </div>
			    <div class="vt-similarity-block">
			      <div class="vt-compare-label">Rewrite</div>
			      <div class="vt-compare-text">{source_rewrite_markup}</div>
			    </div>
			  </div>
			</div>
			""",
			unsafe_allow_html=True,
		)
	with right_col:
		st.markdown(
			f"""
			<div class="vt-compare-wrapper">
			  <div class="vt-compare-label">AI vs rewrite (cosine)</div>
			  <div class="vt-similarity-panel">
			    <div class="vt-similarity-block">
			      <div class="vt-compare-label">AI Source</div>
			      <div class="vt-compare-text">{ai_base_markup}</div>
			    </div>
			    <div class="vt-similarity-block">
			      <div class="vt-compare-label">Rewrite</div>
			      <div class="vt-compare-text">{ai_rewrite_markup}</div>
			    </div>
			  </div>
			</div>
			""",
			unsafe_allow_html=True,
		)


def _render_similarity_metric(label: str, score: float, bar_color: str) -> None:
	label_col, gauge_col = st.columns([3, 1])
	with label_col:
		st.metric(label, f"{score * 100:.1f}%")
	with gauge_col:
		st.plotly_chart(
			build_mini_gauge(score * 100, bar_color),
			use_container_width=True,
			config={"displayModeBar": False},
		)


def render_comparison_panel(
	original_text: str,
	edited_text: str,
	panel_id: str,
	panel_height: int = 280,
) -> None:
	original_markup, edited_markup = _build_diff_markup(original_text, edited_text)
	frame_height = panel_height + 70
	html_block = f"""
	<style>
	  .vt-compare-grid {{
	    display: grid;
	    grid-template-columns: repeat(2, minmax(0, 1fr));
	    gap: 12px;
	    font-family: inherit;
	  }}
	  .vt-compare-wrapper {{
	    display: flex;
	    flex-direction: column;
	    gap: 6px;
	  }}
	  .vt-compare-label {{
	    font-size: 0.8rem;
	    color: #64748b;
	    font-weight: 600;
	  }}
	  .vt-compare-panel {{
	    border: 1px solid #e2e8f0;
	    border-radius: 8px;
	    background: #ffffff;
	    padding: 12px;
	    height: {panel_height}px;
	    overflow: auto;
	  }}
	  .vt-compare-text {{
	    white-space: pre-wrap;
	    font-family: "Courier New", Courier, monospace;
	    font-size: 0.85rem;
	    line-height: 1.4;
	    color: #0f172a;
	  }}
	  .vt-diff-add {{
	    background: rgba(34, 197, 94, 0.2);
	    border-radius: 4px;
	    padding: 0 2px;
	  }}
	  .vt-diff-remove {{
	    background: rgba(239, 68, 68, 0.2);
	    border-radius: 4px;
	    padding: 0 2px;
	    text-decoration: line-through;
	  }}
	</style>
	<div class=\"vt-compare-grid\" id=\"{panel_id}\">
	  <div class=\"vt-compare-wrapper\">
	    <div class=\"vt-compare-label\">AI Source</div>
	    <div class=\"vt-compare-panel\" id=\"{panel_id}-left\">
	      <div class=\"vt-compare-text\">{original_markup}</div>
	    </div>
	  </div>
	  <div class=\"vt-compare-wrapper\">
	    <div class=\"vt-compare-label\">Writer Rewrite</div>
	    <div class=\"vt-compare-panel\" id=\"{panel_id}-right\">
	      <div class=\"vt-compare-text\">{edited_markup}</div>
	    </div>
	  </div>
	</div>
	<script>
	(function() {{
	  const left = document.getElementById("{panel_id}-left");
	  const right = document.getElementById("{panel_id}-right");
	  if (!left || !right) return;
	  let syncing = false;
	  const syncScroll = (source, target) => () => {{
	    if (syncing) return;
	    syncing = true;
	    target.scrollTop = source.scrollTop;
	    target.scrollLeft = source.scrollLeft;
	    syncing = false;
	  }};
	  left.addEventListener("scroll", syncScroll(left, right));
	  right.addEventListener("scroll", syncScroll(right, left));
	}})();
	</script>
	"""
	components.html(html_block, height=frame_height, scrolling=False)


def render_ai_ism_panel(
	title: str,
	categories: Dict[str, int],
	metric_results: Dict[str, "MetricResult"] | None,
	colors: List[str],
) -> None:
	st.markdown(f"<div class='vt-muted'>{title}</div>", unsafe_allow_html=True)
	total = sum(categories.values()) if categories else 0
	if total == 0:
		st.info("No AI-isms found.")
		h_count = 0
		markers = 0
		if metric_results:
			h_metric = metric_results.get("epistemic_hedging")
			discourse_metric = metric_results.get("discourse_marker_density")
			if h_metric:
				h_count = h_metric.details.get("hedge_count", 0)
			if discourse_metric:
				markers = discourse_metric.details.get("total_markers", 0)
		stats = [
			("Cliches", categories.get("cliches", 0)),
			("Connectors", categories.get("connectors", 0)),
			("Generic openers", categories.get("generic_openers", 0)),
			("Hedges", h_count),
			("Transition words", markers),
		]
		cols = st.columns(3)
		for idx, (label, value) in enumerate(stats):
			cols[idx % 3].metric(label, value)
		return
	st.plotly_chart(
		build_pie_chart(categories, colors=colors),
		use_container_width=True,
		config=_plotly_download_config(),
	)


def _estimate_projected_score(
	analysis: "AnalysisResult",
	metric_label: str,
	original_score: float,
	edited_score: float,
) -> float:
	weights = {
		"Burstiness": 0.25,
		"Lexical Diversity": 0.20,
		"Syntactic Complexity": 0.20,
		"Epistemic Hedging": 0.25,
	}
	delta = original_score - edited_score
	weight = weights.get(metric_label, 0.08)
	projected = analysis.score + (delta * weight)
	return max(0.0, min(100.0, projected))


def _top_repeated_words(text: str, limit: int = 4) -> List[str]:
	stopwords = {
		"the",
		"and",
		"that",
		"with",
		"from",
		"this",
		"there",
		"their",
		"which",
		"into",
		"for",
		"are",
		"was",
		"were",
		"has",
		"have",
		"had",
		"but",
		"not",
		"you",
		"your",
		"they",
		"them",
		"its",
		"can",
		"may",
		"might",
		"would",
		"could",
	}
	words = re.findall(r"[A-Za-z']+", text.lower())
	filtered = [word for word in words if word not in stopwords and len(word) > 3]
	counts = Counter(filtered)
	return [word for word, _ in counts.most_common(limit) if counts[word] > 2]


def _build_repair_suggestions(
	metric_label: str,
	analysis: "AnalysisResult",
	original_text: str,
	edited_text: str,
) -> List[str]:
	suggestions: List[str] = []
	if metric_label == "AI-ism Likelihood":
		phrases = [phrase.get("phrase", "") for phrase in analysis.ai_ism_phrases[:3]]
		phrases = [phrase for phrase in phrases if phrase]
		if phrases:
			suggestions.append(
				"Rewrite or remove these phrases: " + ", ".join(phrases)
			)
			suggestions.append("Use direct verbs and concrete nouns over scaffolding clauses.")
		else:
			suggestions.append("Reduce template phrases and tighten transitions between ideas.")
	elif metric_label == "Burstiness":
		lengths = analysis.sentence_lengths.get("Writer Rewrite", [])
		if lengths:
			shortest = min(lengths)
			longest = max(lengths)
			suggestions.append(
				f"Split the longest sentence (~{longest} words) or merge very short ones (~{shortest} words)."
			)
		suggestions.append("Mix short, medium, and long sentences to restore rhythm.")
	elif metric_label == "Lexical Diversity":
		repeats = _top_repeated_words(edited_text)
		if repeats:
			suggestions.append("Replace repeated words: " + ", ".join(repeats))
		suggestions.append("Vary verbs and adjectives that appear in consecutive sentences.")
	elif metric_label == "Syntactic Complexity":
		suggestions.append("Add one subordinate clause to key sentences to reflect your original structure.")
		suggestions.append("Balance simple statements with one or two layered sentences per paragraph.")
	elif metric_label == "Function Word Ratio":
		suggestions.append("Trim filler words such as 'that', 'just', and 'really'.")
		suggestions.append("Prefer precise nouns and verbs over connector-heavy phrasing.")
	elif metric_label == "Discourse Markers":
		details = analysis.metric_results["discourse_marker_density"].details
		marker_count = details.get("total_markers", 0)
		suggestions.append(f"Reduce transition markers (current count: {marker_count}).")
		suggestions.append("Keep one transition per paragraph, and let sentence order do the work.")
	elif metric_label == "Information Density":
		details = analysis.metric_results["information_density"].details
		content_words = details.get("content_words", 0)
		suggestions.append(f"Add concrete nouns or numbers to raise content words (now {content_words}).")
		suggestions.append("Replace abstract fillers with specific examples or observations.")
	elif metric_label == "Epistemic Hedging":
		details = analysis.metric_results["epistemic_hedging"].details
		headges = details.get("hedge_count", 0)
		suggestions.append(f"Reintroduce cautious wording where appropriate (hedges: {hedges}).")
		suggestions.append("Balance certainty with qualifiers like 'likely' or 'suggests'.")
	else:
		suggestions.append("Review highlighted changes and restore any phrasing that sounds less like you.")
	return suggestions


def word_count_notice(label: str, count: int) -> None:
	if count == 0:
		return
	if count < 200:
		st.warning(f"{label} is short ({count} words). Aim for 200-500 words.")
	elif count > 500:
		st.info(f"{label} is long ({count} words). Results may take longer to read.")


def read_uploaded_text(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> str:
	if uploaded_file is None:
		return ""
	name = (uploaded_file.name or "").lower()
	data = uploaded_file.read()
	if name.endswith(".txt"):
		return data.decode("utf-8", errors="replace")
	if name.endswith(".docx"):
		if Document is None:
			raise RuntimeError("python-docx is not installed.")
		doc = Document(uploaded_file)
		return "\n".join(p.text for p in doc.paragraphs if p.text)
	if name.endswith(".pdf"):
		if PdfReader is None:
			raise RuntimeError("pypdf is not installed.")
		reader = PdfReader(uploaded_file)
		pages = [page.extract_text() or "" for page in reader.pages]
		return "\n".join(pages).strip()
	return data.decode("utf-8", errors="replace")


def init_state() -> None:
	default_standards = CalibrationStandards.DEFAULTS
	if "theme_name" not in st.session_state:
		st.session_state.theme_name = "Neutral Corporate"
	if "prompt_text" not in st.session_state:
		st.session_state.prompt_text = ""
	if "use_default_standards" not in st.session_state:
		st.session_state.use_default_standards = True
	if "calibration" not in st.session_state:
		st.session_state.calibration = {
			key: {"human": value["human"], "ai": value["ai"]}
			for key, value in default_standards.items()
		}
	if "calibration_loaded" not in st.session_state:
		st.session_state.calibration_loaded = False
	if "analysis" not in st.session_state:
		st.session_state.analysis = None
	if "human_text" not in st.session_state:
		st.session_state.human_text = ""
	if "source_text" not in st.session_state:
		st.session_state.source_text = ""
	if "ai_text" not in st.session_state:
		st.session_state.ai_text = ""
	if "paraphrase_text" not in st.session_state:
		st.session_state.paraphrase_text = ""
	if "human_file_name" not in st.session_state:
		st.session_state.human_file_name = None
	if "source_file_name" not in st.session_state:
		st.session_state.source_file_name = None
	if "ai_file_name" not in st.session_state:
		st.session_state.ai_file_name = None
	if "paraphrase_file_name" not in st.session_state:
		st.session_state.paraphrase_file_name = None
	if "repair_metric" not in st.session_state:
		st.session_state.repair_metric = METRICS[0]["label"]
	if "page" not in st.session_state:
		st.session_state.page = "Upload & Configuration"
	if "calibration_toggle_prev" not in st.session_state:
		st.session_state.calibration_toggle_prev = "Use default standards"
	if "defaults_reset_notice" not in st.session_state:
		st.session_state.defaults_reset_notice = False
	if "analysis_notice" not in st.session_state:
		st.session_state.analysis_notice = False
	if "auto_load_notice" not in st.session_state:
		st.session_state.auto_load_notice = ""
	if "auto_load_notice_kind" not in st.session_state:
		st.session_state.auto_load_notice_kind = "info"
	if "storage_op_index" not in st.session_state:
		st.session_state.storage_op_index = 0


def _next_storage_key(prefix: str) -> str:
	if "storage_op_index" not in st.session_state:
		st.session_state.storage_op_index = 0
	st.session_state.storage_op_index += 1
	return f"{prefix}_{st.session_state.storage_op_index}"


def save_human(local_storage: LocalStorage | None) -> None:
	if local_storage:
		local_storage.setItem(HUMAN_TEXT_KEY, st.session_state.human_text)
	st.session_state.analysis = None


def save_ai(local_storage: LocalStorage | None) -> None:
	if local_storage:
		local_storage.setItem(AI_TEXT_KEY, st.session_state.ai_text)
	st.session_state.analysis = None


def save_paraphrase(local_storage: LocalStorage | None) -> None:
	if local_storage:
		local_storage.setItem(PARAPHRASE_TEXT_KEY, st.session_state.paraphrase_text)
	st.session_state.analysis = None


def save_prompt(local_storage: LocalStorage | None) -> None:
	if local_storage:
		local_storage.setItem(PROMPT_TEXT_KEY, st.session_state.prompt_text)
	st.session_state.analysis = None


def save_source(local_storage: LocalStorage | None) -> None:
	if local_storage:
		local_storage.setItem(SOURCE_TEXT_KEY, st.session_state.source_text)
	st.session_state.analysis = None


def clear_human(local_storage: LocalStorage | None) -> None:
	st.session_state.human_text = ""
	if local_storage:
		try:
			local_storage.deleteItem(HUMAN_TEXT_KEY, key=_next_storage_key("delete_human"))
		except KeyError:
			pass
	st.session_state.analysis = None



def clear_source(local_storage: LocalStorage | None) -> None:
	st.session_state.source_text = ""
	st.session_state.source_file_name = None
	if local_storage:
		try:
			local_storage.deleteItem(SOURCE_TEXT_KEY, key=_next_storage_key("delete_source"))
		except KeyError:
			pass
	st.session_state.analysis = None


def clear_ai(local_storage: LocalStorage | None) -> None:
	st.session_state.ai_text = ""
	if local_storage:
		try:
			local_storage.deleteItem(AI_TEXT_KEY, key=_next_storage_key("delete_ai"))
		except KeyError:
			pass
	st.session_state.analysis = None


def clear_paraphrase(local_storage: LocalStorage | None) -> None:
	st.session_state.paraphrase_text = ""
	if local_storage:
		try:
			local_storage.deleteItem(
				PARAPHRASE_TEXT_KEY,
				key=_next_storage_key("delete_paraphrase"),
			)
		except KeyError:
			pass
	st.session_state.analysis = None


def auto_load_text(
	local_storage: LocalStorage | None,
	storage_key: str,
	state_key: str,
	label: str,
	file_name_key: str | None = None,
) -> None:
	if not local_storage:
		st.session_state.auto_load_notice = "Autosave is unavailable because streamlit-local-storage is missing."
		st.session_state.auto_load_notice_kind = "error"
		return
	stored = local_storage.getItem(storage_key)
	if not stored:
		st.session_state.auto_load_notice = f"No saved {label} text found."
		st.session_state.auto_load_notice_kind = "info"
		return
	if not isinstance(stored, str):
		stored = str(stored)
	st.session_state[state_key] = stored
	if file_name_key:
		st.session_state[file_name_key] = None
	st.session_state.analysis = None
	st.session_state.auto_load_notice = f"Auto loaded saved {label} text."
	st.session_state.auto_load_notice_kind = "success"


def _coerce_calibration(data: object) -> Dict[str, Dict[str, float]] | None:
	if not isinstance(data, dict):
		return None
	coerced: Dict[str, Dict[str, float]] = {}
	for key, defaults in CalibrationStandards.DEFAULTS.items():
		entry = data.get(key)
		if not isinstance(entry, dict):
			return None
		human = entry.get("human")
		ai = entry.get("ai")
		if not isinstance(human, (int, float)) or not isinstance(ai, (int, float)):
			return None
		coerced[key] = {"human": float(human), "ai": float(ai)}
	return coerced


def _set_storage_item(
	local_storage: LocalStorage | None,
	item_key: str,
	value: str,
	key_prefix: str,
) -> None:
	if not local_storage:
		return
	op_key = _next_storage_key(key_prefix)
	try:
		local_storage.setItem(item_key, value, key=op_key)
	except TypeError:
		local_storage.setItem(item_key, value)


def apply_preset(local_storage: LocalStorage | None, preset_index: int) -> None:
	if preset_index < 0 or preset_index >= len(PRESET_SETS):
		st.session_state.auto_load_notice = "Preset selection is invalid."
		st.session_state.auto_load_notice_kind = "error"
		return
	preset = PRESET_SETS[preset_index]
	st.session_state.human_text = preset["human"]
	st.session_state.source_text = preset["source"]
	st.session_state.ai_text = preset["ai"]
	st.session_state.paraphrase_text = preset["paraphrase"]
	st.session_state.prompt_text = preset["prompt"]
	st.session_state.human_file_name = None
	st.session_state.source_file_name = None
	st.session_state.ai_file_name = None
	st.session_state.paraphrase_file_name = None
	_set_storage_item(local_storage, HUMAN_TEXT_KEY, preset["human"], "preset_human")
	_set_storage_item(local_storage, SOURCE_TEXT_KEY, preset["source"], "preset_source")
	_set_storage_item(local_storage, AI_TEXT_KEY, preset["ai"], "preset_ai")
	_set_storage_item(local_storage, PARAPHRASE_TEXT_KEY, preset["paraphrase"], "preset_paraphrase")
	_set_storage_item(local_storage, PROMPT_TEXT_KEY, preset["prompt"], "preset_prompt")
	st.session_state.analysis = None
	st.session_state.auto_load_notice = f"Loaded {preset['name']} into all sections."
	st.session_state.auto_load_notice_kind = "success"


def load_calibration(local_storage: LocalStorage | None) -> None:
	if not local_storage or st.session_state.calibration_loaded:
		return
	stored = local_storage.getItem(CALIBRATION_KEY)
	if not stored:
		st.session_state.calibration_loaded = True
		return
	try:
		data = json.loads(stored)
	except (TypeError, json.JSONDecodeError):
		st.session_state.calibration_loaded = True
		return
	sanitized = _coerce_calibration(data)
	if sanitized:
		st.session_state.calibration = sanitized
		st.session_state.use_default_standards = False
		st.session_state.calibration_toggle = "Adjust thresholds"
	st.session_state.calibration_loaded = True


def save_calibration(local_storage: LocalStorage | None) -> None:
	if not local_storage:
		return
	local_storage.setItem(CALIBRATION_KEY, json.dumps(st.session_state.calibration))


def run_analysis() -> None:
	custom_standards = None
	if not st.session_state.use_default_standards:
		custom_standards = st.session_state.calibration
	human_text = st.session_state.human_text
	source_text = st.session_state.source_text
	ai_text = st.session_state.ai_text
	paraphrase_text = st.session_state.paraphrase_text
	if not isinstance(human_text, str):
		human_text = str(human_text or "")
	if not isinstance(source_text, str):
		source_text = str(source_text or "")
	if not isinstance(ai_text, str):
		ai_text = str(ai_text or "")
	if not isinstance(paraphrase_text, str):
		paraphrase_text = str(paraphrase_text or "")
	st.session_state.analysis = analyze_multitext(
		human_text,
		source_text,
		ai_text,
		paraphrase_text,
		custom_standards=custom_standards,
	)


def run_analysis_with_notice() -> None:
	run_analysis()
	st.session_state.analysis_notice = True


def render_header() -> None:
	header_left, header_right = st.columns([1, 6], gap="small")
	with header_left:
		if LOGO_PATH.exists():
			st.image(str(LOGO_PATH), width=64)
		else:
			st.markdown("<div class='vt-logo-fallback'>VT</div>", unsafe_allow_html=True)
	with header_right:
		st.markdown(
			"""
			<div class="vt-header">
			  <div>
			    <div class="vt-title">VoiceTracer</div>
			    <div class="vt-tagline">Preserve Your Voice. Navigate AI with Autonomy.</div>
			  </div>
			</div>
			""",
			unsafe_allow_html=True,
		)


def render_upload_screen(local_storage: LocalStorage | None) -> None:
	st.markdown(
		"""
		<div class="vt-hero">
		  <div class="vt-compass">&#x1F9ED;</div>
		  <div class="vt-hero-title">Trace Voice Across AI and Human Rewrites</div>
		  <div class="vt-hero-sub">Analyze AI mimicry, source similarity, and voice preservation in one dashboard.</div>
		</div>
		""",
		unsafe_allow_html=True,
	)

	st.markdown("<div class='vt-section-title'>Configuration</div>", unsafe_allow_html=True)
	prompt_status = "Provided" if st.session_state.prompt_text.strip() else "Missing"
	config_left, config_right = st.columns([2, 1], gap="large")
	with config_left:
		st.selectbox(
			"Genre",
			options=["Academic Writing"],
			index=0,
			help="Genre is locked for this release.",
			disabled=True,
		)
		st.radio(
			"Calibration",
			options=["Use default standards", "Adjust thresholds"],
			index=0 if st.session_state.use_default_standards else 1,
			horizontal=True,
			key="calibration_toggle",
		)
		st.session_state.use_default_standards = (
			st.session_state.calibration_toggle == "Use default standards"
		)
		if (
			st.session_state.calibration_toggle_prev != st.session_state.calibration_toggle
			and st.session_state.use_default_standards
		):
			st.session_state.defaults_reset_notice = True
		st.session_state.calibration_toggle_prev = st.session_state.calibration_toggle
		if st.session_state.use_default_standards:
			for key, defaults in CalibrationStandards.DEFAULTS.items():
				st.session_state.calibration[key] = {
					"human": defaults["human"],
					"ai": defaults["ai"],
				}
				st.session_state[f"inline_human_{key}"] = defaults["human"]
				st.session_state[f"inline_ai_{key}"] = defaults["ai"]
			if local_storage:
				try:
					local_storage.deleteItem(
						CALIBRATION_KEY,
						key=_next_storage_key("delete_calibration"),
					)
				except KeyError:
					pass
			if st.session_state.defaults_reset_notice:
				st.success("Default thresholds restored. The next analysis uses standard settings.")
				st.session_state.defaults_reset_notice = False
		if not st.session_state.use_default_standards:
			st.info("Adjusting thresholds affects the next analysis run.")
			st.caption("Custom thresholds active.")
			for metric in METRICS:
				key = metric["key"]
				label = metric["label"]
				defaults = CalibrationStandards.DEFAULTS[key]
				current = st.session_state.calibration.get(key, defaults)
				with st.expander(f"{label} thresholds", expanded=False):
					human_value = st.slider(
						"Human Standard",
						min_value=0.0,
						max_value=100.0,
						value=float(current["human"]),
						step=0.01,
						key=f"inline_human_{key}",
					)
					ai_value = st.slider(
						"AI Standard",
						min_value=0.0,
						max_value=100.0,
						value=float(current["ai"]),
						step=0.01,
						key=f"inline_ai_{key}",
					)
					st.session_state.calibration[key] = {
						"human": human_value,
						"ai": ai_value,
					}
			if local_storage:
				save_calibration(local_storage)
			both_present = (
				bool(st.session_state.human_text.strip())
				and bool(st.session_state.source_text.strip())
				and bool(st.session_state.ai_text.strip())
				and bool(st.session_state.paraphrase_text.strip())
				and bool(st.session_state.prompt_text.strip())
			)
			st.button(
				"Apply thresholds & rerun analysis",
				disabled=not both_present,
				on_click=run_analysis if both_present else None,
			)
	with config_right:
		st.markdown(
			f"""
			<div class="vt-card vt-subtle">
			  <div class="vt-card-title">Step Summary</div>
			  <div class="vt-card-caption">Provide all four sections to unlock analysis.</div>
			  <div class="vt-metric-rows">
			    <div class="vt-metric-row"><span>Genre</span><span>Academic Writing</span></div>
			    <div class="vt-metric-row"><span>Prompt</span><span>{prompt_status}</span></div>
			  </div>
			</div>
			""",
			unsafe_allow_html=True,
		)

	st.markdown("<div class='vt-section-title'>Text Inputs</div>", unsafe_allow_html=True)
	if st.session_state.auto_load_notice:
		notice = st.session_state.auto_load_notice
		kind = st.session_state.auto_load_notice_kind
		if kind == "success":
			st.success(notice)
		elif kind == "error":
			st.error(notice)
		else:
			st.info(notice)
		st.session_state.auto_load_notice = ""
		st.session_state.auto_load_notice_kind = "info"
	preset_options = [preset["name"] for preset in PRESET_SETS]
	preset_col_left, preset_col_right = st.columns([3, 1])
	with preset_col_left:
		selected_preset = st.selectbox("Preset set", options=preset_options, index=0)
	with preset_col_right:
		st.button(
			"Load Preset",
			use_container_width=True,
			on_click=apply_preset,
			args=(local_storage, preset_options.index(selected_preset)),
		)
	section_top_left, section_top_right = st.columns(2, gap="large")

	with section_top_left:
		st.markdown("<div class='vt-card-title'>Section 1: Human Baseline (Unassisted)</div>", unsafe_allow_html=True)
		upload_human = st.file_uploader(
			"Upload unedited human text",
			type=["pdf", "docx", "txt"],
			key="human_file",
		)
		if upload_human and upload_human.name != st.session_state.human_file_name:
			try:
				st.session_state.human_text = read_uploaded_text(upload_human)
				st.session_state.human_file_name = upload_human.name
				save_human(local_storage)
				st.success(f"Loaded {upload_human.name}")
			except RuntimeError as exc:
				st.error(str(exc))
			except Exception:
				st.error("Could not read the file. Try a plain text export.")

		st.text_area(
			"Paste or write your unassisted text",
			key="human_text",
			height=220,
			placeholder="Write or paste your raw human text here...",
			on_change=lambda: save_human(local_storage),
		)
		human_wc = word_count(st.session_state.human_text)
		st.caption(f"Word count: {human_wc}")
		word_count_notice("Human baseline", human_wc)
		human_actions_left, human_actions_right = st.columns(2)
		with human_actions_left:
			st.button(
				"Auto Load",
				key="auto_load_human",
				use_container_width=True,
				on_click=auto_load_text,
				args=(local_storage, HUMAN_TEXT_KEY, "human_text", "human baseline", "human_file_name"),
			)
		with human_actions_right:
			st.button("Clear Section 1", on_click=lambda: clear_human(local_storage), use_container_width=True)

	with section_top_right:
		st.markdown("<div class='vt-card-title'>Section 2: Source Material (Upload Only)</div>", unsafe_allow_html=True)
		upload_source = st.file_uploader(
			"Upload source material",
			type=["pdf", "docx", "txt"],
			key="source_file",
		)
		if upload_source and upload_source.name != st.session_state.source_file_name:
			try:
				st.session_state.source_text = read_uploaded_text(upload_source)
				st.session_state.source_file_name = upload_source.name
				save_source(local_storage)
				st.success(f"Loaded {upload_source.name}")
			except RuntimeError as exc:
				st.error(str(exc))
			except Exception:
				st.error("Could not read the file. Try a plain text export.")
		st.text_area(
			"Source content preview",
			value=st.session_state.source_text,
			height=220,
			disabled=True,
		)
		source_wc = word_count(st.session_state.source_text)
		if source_wc:
			st.caption(f"Word count: {source_wc}")
		source_actions_left, source_actions_right = st.columns(2)
		with source_actions_left:
			st.button(
				"Auto Load",
				key="auto_load_source",
				use_container_width=True,
				on_click=auto_load_text,
				args=(local_storage, SOURCE_TEXT_KEY, "source_text", "source material", "source_file_name"),
			)
		with source_actions_right:
			st.button("Clear Section 2", on_click=lambda: clear_source(local_storage), use_container_width=True)

	section_bottom_left, section_bottom_right = st.columns(2, gap="large")

	with section_bottom_left:
		st.markdown("<div class='vt-card-title'>Section 3: AI-Generated Draft</div>", unsafe_allow_html=True)
		st.text_input(
			"Enter the prompt you used to generate the content (required)",
			key="prompt_text",
			placeholder="Paste the exact prompt used with the AI...",
			on_change=lambda: save_prompt(local_storage),
		)
		if not st.session_state.prompt_text.strip():
			st.warning("Prompt is required for report generation.")
		upload_ai = st.file_uploader(
			"Upload AI-generated text",
			type=["pdf", "docx", "txt"],
			key="ai_file",
		)
		if upload_ai and upload_ai.name != st.session_state.ai_file_name:
			try:
				st.session_state.ai_text = read_uploaded_text(upload_ai)
				st.session_state.ai_file_name = upload_ai.name
				save_ai(local_storage)
				st.success(f"Loaded {upload_ai.name}")
			except RuntimeError as exc:
				st.error(str(exc))
			except Exception:
				st.error("Could not read the file. Try a plain text export.")

		st.text_area(
			"Paste AI-generated draft",
			key="ai_text",
			height=220,
			placeholder="Paste your AI-generated text here...",
			on_change=lambda: save_ai(local_storage),
		)
		ai_wc = word_count(st.session_state.ai_text)
		st.caption(f"Word count: {ai_wc}")
		word_count_notice("AI draft", ai_wc)
		ai_actions_left, ai_actions_right = st.columns(2)
		with ai_actions_left:
			st.button(
				"Auto Load",
				key="auto_load_ai",
				use_container_width=True,
				on_click=auto_load_text,
				args=(local_storage, AI_TEXT_KEY, "ai_text", "AI draft", "ai_file_name"),
			)
		with ai_actions_right:
			st.button("Clear Section 3", on_click=lambda: clear_ai(local_storage), use_container_width=True)

	with section_bottom_right:
		st.markdown("<div class='vt-card-title'>Section 4: Writer Paraphrase (Your Rewrite)</div>", unsafe_allow_html=True)
		upload_paraphrase = st.file_uploader(
			"Upload your paraphrase",
			type=["pdf", "docx", "txt"],
			key="paraphrase_file",
		)
		if upload_paraphrase and upload_paraphrase.name != st.session_state.paraphrase_file_name:
			try:
				st.session_state.paraphrase_text = read_uploaded_text(upload_paraphrase)
				st.session_state.paraphrase_file_name = upload_paraphrase.name
				save_paraphrase(local_storage)
				st.success(f"Loaded {upload_paraphrase.name}")
			except RuntimeError as exc:
				st.error(str(exc))
			except Exception:
				st.error("Could not read the file. Try a plain text export.")

		st.text_area(
			"Paste or write your paraphrase",
			key="paraphrase_text",
			height=220,
			placeholder="Paste or write your rewrite here...",
			on_change=lambda: save_paraphrase(local_storage),
		)
		paraphrase_wc = word_count(st.session_state.paraphrase_text)
		st.caption(f"Word count: {paraphrase_wc}")
		word_count_notice("Paraphrase", paraphrase_wc)
		paraphrase_actions_left, paraphrase_actions_right = st.columns(2)
		with paraphrase_actions_left:
			st.button(
				"Auto Load",
				key="auto_load_paraphrase",
				use_container_width=True,
				on_click=auto_load_text,
				args=(local_storage, PARAPHRASE_TEXT_KEY, "paraphrase_text", "paraphrase", "paraphrase_file_name"),
			)
		with paraphrase_actions_right:
			st.button("Clear Section 4", on_click=lambda: clear_paraphrase(local_storage), use_container_width=True)

	all_present = (
		bool(st.session_state.human_text.strip())
		and bool(st.session_state.source_text.strip())
		and bool(st.session_state.ai_text.strip())
		and bool(st.session_state.paraphrase_text.strip())
		and bool(st.session_state.prompt_text.strip())
	)
	if not all_present:
		st.warning("Add all four texts and the AI prompt to enable analysis.")
	st.button(
		"Run Analysis",
		disabled=not all_present,
		on_click=run_analysis_with_notice if all_present else None,
		use_container_width=True,
	)
	if st.session_state.analysis_notice:
		st.success("Analysis Complete. Go to next step.")
		st.session_state.analysis_notice = False

	st.markdown(
		"<div class='vt-footer'>Version 0.9.2 | Thesis Citation | Privacy</div>",
		unsafe_allow_html=True,
	)


def render_dashboard_screen() -> None:
	analysis = st.session_state.analysis
	if not analysis:
		st.info("Run analysis to view the dashboard.")
		return

	st.markdown("<div class='vt-section-title'>Analysis View</div>", unsafe_allow_html=True)
	section = st.radio(
		"",
		options=["Executive Summary", "Metric Deep-Dive", "Visual Evidence"],
		horizontal=True,
	)

	if section == "Executive Summary":
		st.markdown("<div class='vt-section-title'>Executive Summary</div>", unsafe_allow_html=True)
		st.markdown(
			f"""
			<div class="vt-card vt-subtle">
			  <div class="vt-card-title">Voice Preservation Score (AI Source vs Writer Rewrite)</div>
			  <div class="vt-card-value">{format_metric(analysis.score)}</div>
			  <div class="vt-card-caption">{analysis.classification}</div>
			</div>
			""",
			unsafe_allow_html=True,
		)
		st.markdown("<div class='vt-card-caption'>Voice Preservation Gauge</div>", unsafe_allow_html=True)
		st.plotly_chart(
			build_gauge_chart(analysis.score),
			use_container_width=True,
			config=_plotly_download_config(),
		)
		st.markdown("<div class='vt-section-title'>Component Breakdown</div>", unsafe_allow_html=True)
		components = {
			"Authenticity Markers": analysis.components.get("Authenticity", 0.0),
			"Lexical Identity": analysis.components.get("Lexical", 0.0),
			"Structural Identity": analysis.components.get("Structural", 0.0),
			"Stylistic Identity": analysis.components.get("Stylistic", 0.0),
			"Voice Consistency": analysis.consistency_score,
		}
		for label, value in components.items():
			st.markdown(f"<div class='vt-card-caption'>{label}</div>", unsafe_allow_html=True)
			st.progress(min(max(int(value), 0), 100))
		st.markdown("<div class='vt-section-title'>Quick Stats</div>", unsafe_allow_html=True)
		stats = [
			("Word Delta", analysis.word_delta, "AI source vs rewrite"),
			("Sentence Delta", analysis.sentence_delta, "Structure change"),
			("AI-isms", analysis.ai_ism_total, "Writer rewrite phrases"),
		]
		for title, value, caption in stats:
			st.markdown(
				f"""
				<div class="vt-card vt-subtle">
				  <div class="vt-card-title">{title}</div>
				  <div class="vt-card-value">{value}</div>
				  <div class="vt-card-caption">{caption}</div>
				</div>
				""",
				unsafe_allow_html=True,
			)
		st.markdown("<div class='vt-section-title'>Similarity Index</div>", unsafe_allow_html=True)
		ai_similarity_cosine = getattr(analysis, "ai_similarity_cosine", 0.0)
		ai_similarity_ngram = getattr(analysis, "ai_similarity_ngram", 0.0)
		source_similarity_cosine = getattr(analysis, "source_similarity_cosine", 0.0)
		source_similarity_ngram = getattr(analysis, "source_similarity_ngram", 0.0)
		sim_left, sim_right = st.columns(2)
		with sim_left:
			_render_similarity_metric("Source vs rewrite (cosine)", source_similarity_cosine, "#16a34a")
			_render_similarity_metric("Source vs rewrite (n-gram)", source_similarity_ngram, "#16a34a")
		with sim_right:
			_render_similarity_metric("AI vs rewrite (cosine)", ai_similarity_cosine, "#ef4444")
			_render_similarity_metric("AI vs rewrite (n-gram)", ai_similarity_ngram, "#ef4444")
		st.markdown(
			"<div class='vt-muted'>Highlighted matches are based on overlapping phrases and shared words.</div>",
			unsafe_allow_html=True,
		)
		_render_similarity_panels(
			st.session_state.source_text,
			st.session_state.ai_text,
			st.session_state.paraphrase_text,
		)
		if analysis.score < 80 and st.button("Open Repair Preview", use_container_width=True):
			st.session_state.page = "Repair Preview"
			st.rerun()

	elif section == "Metric Deep-Dive":
		st.markdown("<div class='vt-section-title'>Metric Deep-Dive</div>", unsafe_allow_html=True)
		st.plotly_chart(
			build_radar_chart(
				analysis.metrics_original,
				analysis.metrics_edited,
				analysis.metric_standards,
			),
			use_container_width=True,
			config=_plotly_download_config(),
		)
		for metric in METRICS:
			label = metric["label"]
			key = metric["key"]
			original_value = analysis.metrics_original.get(label, 0.0)
			edited_value = analysis.metrics_edited.get(label, 0.0)
			delta = edited_value - original_value
			verdict = analysis.metric_results[key].verdict
			standards = analysis.metric_standards.get(label, {})
			human_value = standards.get("human", 0.0)
			ai_value = standards.get("ai", 0.0)
			st.markdown(
				f"""
				<div class="vt-card vt-subtle">
				  <div class="vt-card-title">{label}</div>
				  <div class="vt-metric-header">
				    <span>AI Source: {format_metric(original_value)}</span>
				    <span>Writer Rewrite: {format_metric(edited_value)}</span>
				    <span>Delta: {format_metric(delta)}</span>
				    <span class="vt-badge">{verdict}</span>
				  </div>
				  <div class="vt-card-caption">{metric['caption']}</div>
				</div>
				""",
				unsafe_allow_html=True,
			)
			with st.expander("Details"):
				mini = go.Figure(
					data=[
						go.Bar(
							x=[original_value, edited_value, human_value, ai_value],
							y=["AI Source", "Writer Rewrite", "Human Std", "AI Std"],
							orientation="h",
							marker=dict(
								color=[
									"#16a34a",
									"#ef4444",
									"rgba(22, 163, 74, 0.35)",
									"rgba(239, 68, 68, 0.35)",
								]
							),
						)
					]
				)
				mini.update_layout(
					margin=dict(l=20, r=20, t=10, b=10),
					xaxis=dict(range=[0, 100]),
					height=200,
				)
				st.plotly_chart(
					mini,
					use_container_width=True,
					config=_plotly_download_config(),
				)
				st.markdown(
					f"""
					<div class="vt-muted">{metric['description']}</div>
					<div class="vt-muted">Rewrite shows a {format_metric(abs(delta))} point change vs AI source.</div>
					""",
					unsafe_allow_html=True,
				)

	else:
		st.markdown("<div class='vt-section-title'>Visual Evidence</div>", unsafe_allow_html=True)
		st.caption("Sentence rhythm across the text (green: AI source, red: writer rewrite).")
		st.plotly_chart(
			build_line_chart(analysis.sentence_lengths),
			use_container_width=True,
			config=_plotly_download_config(),
		)
		st.caption("Human vs AI standard thresholds for each metric.")
		st.plotly_chart(
			build_metric_standards_chart(analysis.metric_standards),
			use_container_width=True,
			config=_plotly_download_config(),
		)
		st.caption("AI-ism category distribution (AI source vs writer rewrite).")
		original_ai_ism = getattr(analysis, "ai_ism_categories_original", analysis.ai_ism_categories)
		pie_left, pie_right = st.columns(2)
		with pie_left:
			render_ai_ism_panel(
				"AI Source",
				original_ai_ism,
				getattr(analysis, "metric_results_original", None),
				colors=["#16a34a", "#22c55e", "#4ade80", "#86efac", "#bbf7d0"],
			)
		with pie_right:
			render_ai_ism_panel(
				"Writer Rewrite",
				analysis.ai_ism_categories,
				analysis.metric_results,
				colors=["#ef4444", "#f97316", "#f59e0b", "#facc15", "#fb7185"],
			)
		with st.expander("AI-isms Detected in Writer Rewrite"):
			if analysis.ai_ism_phrases:
				for phrase in analysis.ai_ism_phrases:
					st.markdown(
						f"- {phrase.get('phrase', '')} ({phrase.get('category', 'n/a')})",
					)
			else:
				st.markdown("- No AI-isms detected.")
		st.markdown("<div class='vt-section-title'>Text Comparison</div>", unsafe_allow_html=True)
		st.caption("Scroll either panel to sync. Differences are highlighted.")
		render_comparison_panel(
			st.session_state.ai_text,
			st.session_state.paraphrase_text,
			panel_id="analysis-compare",
			panel_height=320,
		)


def render_repair_preview() -> None:
	analysis = st.session_state.analysis
	st.markdown("<div class='vt-section-title'>Repair Preview</div>", unsafe_allow_html=True)
	if not analysis:
		st.info("Run analysis to open repair preview.")
		return
	compromised = [
		metric["label"]
		for metric in METRICS
		if analysis.metric_results[metric["key"]].verdict == "Compromised"
	]
	options = compromised or [m["label"] for m in METRICS]
	metric_focus = st.selectbox("Current Focus", options=options)
	st.session_state.repair_metric = metric_focus
	st.markdown(
		f"<div class='vt-muted'>Negotiating voice alignment for: {metric_focus}</div>",
		unsafe_allow_html=True,
	)
	original_score = analysis.metrics_original.get(metric_focus, 0.0)
	edited_score = analysis.metrics_edited.get(metric_focus, 0.0)
	projected_score = _estimate_projected_score(
		analysis,
		metric_focus,
		original_score,
		edited_score,
	)
	score_left, score_mid, score_right = st.columns(3)
	with score_left:
		st.metric("AI source score", f"{original_score:.1f}")
	with score_mid:
		delta_metric = edited_score - original_score
		st.metric("Writer rewrite score", f"{edited_score:.1f}", delta=f"{delta_metric:+.1f}")
	with score_right:
		delta_overall = projected_score - analysis.score
		st.metric("Projected overall (estimate)", f"{projected_score:.1f}", delta=f"{delta_overall:+.1f}")
	st.caption("Projected overall score is an estimate based on component weighting.")
	st.markdown("<div class='vt-section-title'>Similarity Snapshot</div>", unsafe_allow_html=True)
	ai_similarity_cosine = getattr(analysis, "ai_similarity_cosine", 0.0)
	ai_similarity_ngram = getattr(analysis, "ai_similarity_ngram", 0.0)
	source_similarity_cosine = getattr(analysis, "source_similarity_cosine", 0.0)
	source_similarity_ngram = getattr(analysis, "source_similarity_ngram", 0.0)
	sim_left, sim_right = st.columns(2)
	with sim_left:
		_render_similarity_metric("Source vs rewrite (cosine)", source_similarity_cosine, "#16a34a")
		_render_similarity_metric("Source vs rewrite (n-gram)", source_similarity_ngram, "#16a34a")
	with sim_right:
		_render_similarity_metric("AI vs rewrite (cosine)", ai_similarity_cosine, "#ef4444")
		_render_similarity_metric("AI vs rewrite (n-gram)", ai_similarity_ngram, "#ef4444")
	st.markdown(
		"<div class='vt-muted'>Highlighted matches are based on overlapping phrases and shared words.</div>",
		unsafe_allow_html=True,
	)
	_render_similarity_panels(
		st.session_state.source_text,
		st.session_state.ai_text,
		st.session_state.paraphrase_text,
	)
	st.markdown("<div class='vt-section-title'>Panel Layout</div>", unsafe_allow_html=True)
	st.caption("Use the slider to adjust panel height. The radio selects which panel to view.")
	panel_height = st.slider("Panel height", min_value=200, max_value=520, value=300, step=20, key="panel_height")
	active_panel = st.radio(
		"View",
		options=["AI Source", "Writer Rewrite", "Your Choice"],
		horizontal=True,
	)
	panel_col = st.columns(1)[0]
	with panel_col:
		if active_panel == "AI Source":
			st.markdown("<div class='vt-card vt-subtle'><div class='vt-card-title'>AI Source</div></div>", unsafe_allow_html=True)
			st.text_area("AI Source", value=st.session_state.ai_text, height=panel_height, disabled=True)
		elif active_panel == "Writer Rewrite":
			st.markdown("<div class='vt-card vt-subtle'><div class='vt-card-title'>Writer Rewrite</div></div>", unsafe_allow_html=True)
			st.text_area("Writer Rewrite", value=st.session_state.paraphrase_text, height=panel_height, disabled=True)
		else:
			st.markdown("<div class='vt-card vt-subtle'><div class='vt-card-title'>Your Choice</div></div>", unsafe_allow_html=True)
			choice = st.radio("Negotiated Options", options=["Option A", "Option B", "Option C", "Custom"], horizontal=True)
			custom_text = ""
			if choice == "Custom":
				custom_text = st.text_area("Custom Rewrite", height=max(180, panel_height - 80))
			st.button("Apply Selection", use_container_width=True)
			if choice == "Custom" and custom_text.strip():
				custom_standards = None
				if not st.session_state.use_default_standards:
					custom_standards = st.session_state.calibration
				try:
					custom_analysis = analyze_multitext(
						st.session_state.human_text,
						st.session_state.source_text,
						st.session_state.ai_text,
						custom_text,
						custom_standards=custom_standards,
					)
					delta_custom = custom_analysis.score - analysis.score
					st.metric(
						"Custom overall score",
						f"{custom_analysis.score:.1f}",
						delta=f"{delta_custom:+.1f}",
					)
				except ValueError:
					st.warning("Custom text is too short to score reliably.")

	st.markdown("<div class='vt-section-title'>Repair Suggestions</div>", unsafe_allow_html=True)
	suggestions = _build_repair_suggestions(
		metric_focus,
		analysis,
		st.session_state.ai_text,
		st.session_state.paraphrase_text,
	)
	for suggestion in suggestions:
		st.markdown(f"- {suggestion}")
	st.markdown(
		"""
		<div class="vt-card vt-subtle">
		  <div class="vt-card-title">Annotation</div>
		  <div class="vt-card-caption">AI changed language that affects this metric. Review and negotiate the change.</div>
		</div>
		""",
		unsafe_allow_html=True,
	)
	st.markdown(
		"<div class='vt-metric-rows'><span>Skip to Next Issue</span> | <span>Accept Rewrite</span> | <span>Restore AI Source</span> | <span>Generate Final Text</span></div>",
		unsafe_allow_html=True,
	)


def render_calibration(local_storage: LocalStorage | None) -> None:
	st.markdown("<div class='vt-section-title'>Calibration Panel</div>", unsafe_allow_html=True)
	if st.session_state.use_default_standards:
		st.info("Enable 'Adjust thresholds' to edit calibration values.")
	disabled = st.session_state.use_default_standards
	for metric in METRICS:
		key = metric["key"]
		label = metric["label"]
		defaults = CalibrationStandards.DEFAULTS[key]
		current = st.session_state.calibration.get(key, defaults)
		with st.expander(label):
			human_value = st.slider(
				"Human Standard",
				min_value=0.0,
				max_value=100.0,
				value=float(current["human"]),
				step=0.01,
				key=f"inline_human_{key}",
				disabled=disabled,
			)
			ai_value = st.slider(
				"AI Standard",
				min_value=0.0,
				max_value=100.0,
				value=float(current["ai"]),
				step=0.01,
				key=f"inline_ai_{key}",
				disabled=disabled,
			)
			st.selectbox(
				"Sensitivity",
				options=["Strict", "Moderate", "Permissive"],
				index=1,
				key=f"sensitivity_{key}",
				disabled=disabled,
			)
			st.session_state.calibration[key] = {"human": human_value, "ai": ai_value}
			if st.button(f"Reset {label}", key=f"reset_{key}"):
				st.session_state.calibration[key] = {
					"human": defaults["human"],
					"ai": defaults["ai"],
				}
				st.session_state[f"inline_human_{key}"] = defaults["human"]
				st.session_state[f"inline_ai_{key}"] = defaults["ai"]
				if local_storage:
					save_calibration(local_storage)
				st.rerun()
	st.markdown("<div class='vt-section-title'>Impact Preview</div>", unsafe_allow_html=True)
	st.markdown(
		"<div class='vt-muted'>Adjusting thresholds will update verdicts on the next analysis run.</div>",
		unsafe_allow_html=True,
	)
	st.button("Save as Custom Profile")
	st.button("Export Calibration Settings")
	if st.button("Reset All"):
		for key, defaults in CalibrationStandards.DEFAULTS.items():
			st.session_state.calibration[key] = {
				"human": defaults["human"],
				"ai": defaults["ai"],
			}
			st.session_state[f"inline_human_{key}"] = defaults["human"]
			st.session_state[f"inline_ai_{key}"] = defaults["ai"]
		if local_storage:
			save_calibration(local_storage)
		st.rerun()


def render_documentation_export() -> None:
	st.markdown("<div class='vt-section-title'>Documentation Export</div>", unsafe_allow_html=True)
	st.selectbox("Report Type", options=["PDF", "Word", "Excel", "JSON"])
	st.markdown("<div class='vt-section-title'>Sections to Include</div>", unsafe_allow_html=True)
	for section in [
		"Executive Summary",
		"Full Metric Analysis",
		"Visualizations",
		"Repair Preview Decisions",
		"AI Source vs Writer Rewrite Comparison",
		"Authorship Documentation Statement",
	]:
		st.checkbox(section, value=True)
	st.markdown("<div class='vt-section-title'>Authorship Documentation Statement</div>", unsafe_allow_html=True)
	st.text_area(
		"Statement",
		value=(
			"This document certifies that the user engaged in AI-assisted writing with systematic voice "
			"preservation monitoring using VoiceTracer."
		),
		height=140,
	)
	st.button("Generate Report")
	st.button("Download")
	st.button("Email to Advisor")


def render_settings() -> None:
	st.markdown("<div class='vt-section-title'>Settings</div>", unsafe_allow_html=True)
	st.selectbox("Default Theme", options=list(THEMES.keys()), index=list(THEMES.keys()).index(st.session_state.theme_name))
	st.selectbox("Default Genre", options=["Academic Writing"])
	st.selectbox("Default Calibration Strictness", options=["Strict", "Moderate", "Permissive"], index=1)
	st.checkbox("Email notifications", value=False)
	st.markdown("<div class='vt-section-title'>About</div>", unsafe_allow_html=True)
	st.markdown("<div class='vt-muted'>Version info | Methodology documentation | Contact</div>", unsafe_allow_html=True)


page_icon = str(FAVICON_PATH) if FAVICON_PATH.exists() else "🧭"
st.set_page_config(page_title="VoiceTracer", layout="wide", page_icon=page_icon)
init_state()
load_css()
apply_theme(st.session_state.theme_name)

local_storage = LocalStorage() if LocalStorage else None
load_calibration(local_storage)
stored_human = local_storage.getItem(HUMAN_TEXT_KEY) if local_storage else ""
stored_ai = local_storage.getItem(AI_TEXT_KEY) if local_storage else ""
stored_paraphrase = local_storage.getItem(PARAPHRASE_TEXT_KEY) if local_storage else ""
stored_prompt = local_storage.getItem(PROMPT_TEXT_KEY) if local_storage else ""
stored_source = local_storage.getItem(SOURCE_TEXT_KEY) if local_storage else ""
if not st.session_state.human_text:
	st.session_state.human_text = stored_human or ""
if not st.session_state.source_text:
	st.session_state.source_text = stored_source or ""
if not st.session_state.ai_text:
	st.session_state.ai_text = stored_ai or ""
if not st.session_state.paraphrase_text:
	st.session_state.paraphrase_text = stored_paraphrase or ""
if not st.session_state.prompt_text:
	st.session_state.prompt_text = stored_prompt or ""

render_header()

with st.sidebar:
	st.markdown("<div class='vt-section-title'>Navigation</div>", unsafe_allow_html=True)
	page = st.radio(
		"",
		options=[
			"Upload & Configuration",
			"Analysis Dashboard",
			"Repair Preview",
			"Calibration",
			"Documentation Export",
			"Settings",
		],
		index=[
			"Upload & Configuration",
			"Analysis Dashboard",
			"Repair Preview",
			"Calibration",
			"Documentation Export",
			"Settings",
		].index(st.session_state.page),
	)
	st.session_state.page = page
	st.markdown("<div class='vt-section-title'>Theme</div>", unsafe_allow_html=True)
	selected_theme = st.selectbox("Theme", options=list(THEMES.keys()), index=list(THEMES.keys()).index(st.session_state.theme_name))
	if selected_theme != st.session_state.theme_name:
		st.session_state.theme_name = selected_theme
		st.rerun()

if LocalStorage is None:
	st.warning("Autosave is unavailable because streamlit-local-storage is missing.")
if Document is None:
	st.info("DOCX upload requires python-docx.")
if PdfReader is None:
	st.info("PDF upload requires pypdf.")

nav_pages = [
	"Upload & Configuration",
	"Analysis Dashboard",
	"Repair Preview",
	"Calibration",
	"Documentation Export",
	"Settings",
]

if st.session_state.page == "Upload & Configuration":
	render_upload_screen(local_storage)
elif st.session_state.page == "Analysis Dashboard":
	render_dashboard_screen()
elif st.session_state.page == "Repair Preview":
	render_repair_preview()
elif st.session_state.page == "Calibration":
	render_calibration(local_storage)

if st.button("Next step", use_container_width=True, key="next_step_global"):
	current_index = nav_pages.index(st.session_state.page)
	next_index = min(current_index + 1, len(nav_pages) - 1)
	if next_index != current_index:
		st.session_state.page = nav_pages[next_index]
		st.rerun()
elif st.session_state.page == "Documentation Export":
	render_documentation_export()
elif st.session_state.page == "Settings":
	render_settings()
