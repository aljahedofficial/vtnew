from __future__ import annotations

from collections import Counter
from pathlib import Path
import difflib
import html
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

from app.analysis import CalibrationStandards, analyze_texts
from app.charts import (
	build_bar_chart,
	build_gauge_chart,
	build_line_chart,
	build_pie_chart,
	build_radar_chart,
)
from app.state import EDITED_TEXT_KEY, ORIGINAL_TEXT_KEY


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


_DIFF_TOKEN_RE = re.compile(r"\s+|[^\s]+")


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


def render_comparison_panel(
	original_text: str,
	edited_text: str,
	panel_id: str,
	panel_height: int = 280,
) -> None:
	original_markup, edited_markup = _build_diff_markup(original_text, edited_text)
	html_block = f"""
	<div class=\"vt-compare-grid\" id=\"{panel_id}\">
	  <div class=\"vt-compare-wrapper\">
	    <div class=\"vt-compare-label\">Original</div>
	    <div class=\"vt-compare-panel\" id=\"{panel_id}-left\">
	      <div class=\"vt-compare-text\">{original_markup}</div>
	    </div>
	  </div>
	  <div class=\"vt-compare-wrapper\">
	    <div class=\"vt-compare-label\">AI-Edited</div>
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
	components.html(html_block, height=panel_height, scrolling=False)


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
		lengths = analysis.sentence_lengths.get("Edited", [])
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
		st.session_state.prompt_text = "Fix grammar and fluency"
	if "use_default_standards" not in st.session_state:
		st.session_state.use_default_standards = True
	if "calibration" not in st.session_state:
		st.session_state.calibration = {
			key: {"human": value["human"], "ai": value["ai"]}
			for key, value in default_standards.items()
		}
	if "analysis" not in st.session_state:
		st.session_state.analysis = None
	if "original_text" not in st.session_state:
		st.session_state.original_text = ""
	if "edited_text" not in st.session_state:
		st.session_state.edited_text = ""
	if "original_file_name" not in st.session_state:
		st.session_state.original_file_name = None
	if "edited_file_name" not in st.session_state:
		st.session_state.edited_file_name = None
	if "repair_metric" not in st.session_state:
		st.session_state.repair_metric = METRICS[0]["label"]
	if "page" not in st.session_state:
		st.session_state.page = "Upload & Configuration"


def save_original(local_storage: LocalStorage | None) -> None:
	if local_storage:
		local_storage.setItem(ORIGINAL_TEXT_KEY, st.session_state.original_text)
	st.session_state.analysis = None


def save_edited(local_storage: LocalStorage | None) -> None:
	if local_storage:
		local_storage.setItem(EDITED_TEXT_KEY, st.session_state.edited_text)
	st.session_state.analysis = None


def clear_original(local_storage: LocalStorage | None) -> None:
	st.session_state.original_text = ""
	if local_storage:
		local_storage.deleteItem(ORIGINAL_TEXT_KEY)
	st.session_state.analysis = None


def clear_edited(local_storage: LocalStorage | None) -> None:
	st.session_state.edited_text = ""
	if local_storage:
		local_storage.deleteItem(EDITED_TEXT_KEY)
	st.session_state.analysis = None


def run_analysis() -> None:
	custom_standards = None
	if not st.session_state.use_default_standards:
		custom_standards = st.session_state.calibration
	original_text = st.session_state.original_text
	edited_text = st.session_state.edited_text
	if not isinstance(original_text, str):
		original_text = str(original_text or "")
	if not isinstance(edited_text, str):
		edited_text = str(edited_text or "")
	st.session_state.analysis = analyze_texts(
		original_text,
		edited_text,
		custom_standards=custom_standards,
	)


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
		  <div class="vt-hero-title">Compare Your Original and AI-Edited Text</div>
		  <div class="vt-hero-sub">Measure voice preservation across 8 stylistic dimensions.</div>
		</div>
		""",
		unsafe_allow_html=True,
	)

	st.markdown("<div class='vt-section-title'>Configuration</div>", unsafe_allow_html=True)
	config_left, config_right = st.columns([2, 1], gap="large")
	with config_left:
		st.selectbox(
			"Genre",
			options=["Academic Writing"],
			index=0,
			help="Genre is locked for this release.",
			disabled=True,
		)
		prompt = st.text_input(
			"Prompt Declaration",
			value=st.session_state.prompt_text,
			help="Default: Fix grammar and fluency",
		)
		st.session_state.prompt_text = prompt
		if prompt.strip() != "Fix grammar and fluency":
			st.warning("Prompt changed. This may alter the AI editing profile.")
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
	with config_right:
		st.markdown(
			"""
			<div class="vt-card vt-subtle">
			  <div class="vt-card-title">Step Summary</div>
			  <div class="vt-card-caption">Upload both drafts to unlock analysis.</div>
			  <div class="vt-metric-rows">
			    <div class="vt-metric-row"><span>Genre</span><span>Academic Writing</span></div>
			    <div class="vt-metric-row"><span>Prompt</span><span>Fix grammar and fluency</span></div>
			  </div>
			</div>
			""",
			unsafe_allow_html=True,
		)

	st.markdown("<div class='vt-section-title'>Text Upload</div>", unsafe_allow_html=True)
	left_col, right_col = st.columns(2, gap="large")

	with left_col:
		upload_original = st.file_uploader(
			"Original Draft",
			type=["pdf", "docx", "txt"],
			key="original_file",
		)
		if upload_original and upload_original.name != st.session_state.original_file_name:
			try:
				st.session_state.original_text = read_uploaded_text(upload_original)
				st.session_state.original_file_name = upload_original.name
				save_original(local_storage)
				st.success(f"Loaded {upload_original.name}")
			except RuntimeError as exc:
				st.error(str(exc))
			except Exception:
				st.error("Could not read the file. Try a plain text export.")

		st.text_area(
			"Paste Original Draft",
			key="original_text",
			height=240,
			placeholder="Paste your original text here...",
			on_change=lambda: save_original(local_storage),
		)
		original_wc = word_count(st.session_state.original_text)
		st.caption(f"Word count: {original_wc}")
		word_count_notice("Original draft", original_wc)
		st.button("Clear Original", on_click=lambda: clear_original(local_storage), use_container_width=True)

	with right_col:
		upload_edited = st.file_uploader(
			"AI-Edited Version",
			type=["pdf", "docx", "txt"],
			key="edited_file",
		)
		if upload_edited and upload_edited.name != st.session_state.edited_file_name:
			try:
				st.session_state.edited_text = read_uploaded_text(upload_edited)
				st.session_state.edited_file_name = upload_edited.name
				save_edited(local_storage)
				st.success(f"Loaded {upload_edited.name}")
			except RuntimeError as exc:
				st.error(str(exc))
			except Exception:
				st.error("Could not read the file. Try a plain text export.")

		st.text_area(
			"Paste AI-Edited Version",
			key="edited_text",
			height=240,
			placeholder="Paste your AI-edited text here...",
			on_change=lambda: save_edited(local_storage),
		)
		edited_wc = word_count(st.session_state.edited_text)
		st.caption(f"Word count: {edited_wc}")
		word_count_notice("AI-edited text", edited_wc)
		st.button("Clear AI-Edited", on_click=lambda: clear_edited(local_storage), use_container_width=True)

	both_present = bool(st.session_state.original_text.strip()) and bool(
		st.session_state.edited_text.strip()
	)
	if not both_present:
		st.warning("Add both texts to enable analysis.")
	st.button(
		"Analyze Voice Preservation",
		disabled=not both_present,
		on_click=run_analysis if both_present else None,
		use_container_width=True,
	)

	st.markdown(
		"<div class='vt-footer'>Version 0.9 | Thesis Citation | Privacy</div>",
		unsafe_allow_html=True,
	)


def render_dashboard_screen() -> None:
	analysis = st.session_state.analysis
	if not analysis:
		st.info("Run analysis to view the dashboard.")
		return

	left_col, center_col, right_col = st.columns([1, 1.4, 1], gap="large")

	with left_col:
		st.markdown("<div class='vt-section-title'>Executive Summary</div>", unsafe_allow_html=True)
		st.markdown(
			f"""
			<div class="vt-card vt-subtle">
			  <div class="vt-card-title">Voice Preservation Score</div>
			  <div class="vt-card-value">{format_metric(analysis.score)}</div>
			  <div class="vt-card-caption">{analysis.classification}</div>
			</div>
			""",
			unsafe_allow_html=True,
		)
		st.plotly_chart(build_gauge_chart(analysis.score), use_container_width=True)
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
			("Word Delta", analysis.word_delta, "Original vs edited"),
			("Sentence Delta", analysis.sentence_delta, "Structure change"),
			("AI-isms", analysis.ai_ism_total, "Formulaic phrases"),
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
		if analysis.score < 80 and st.button("Open Repair Preview", use_container_width=True):
			st.session_state.page = "Repair Preview"
			st.rerun()

	with center_col:
		st.markdown("<div class='vt-section-title'>Metric Deep-Dive</div>", unsafe_allow_html=True)
		st.plotly_chart(
			build_radar_chart(
				analysis.metrics_original,
				analysis.metrics_edited,
				analysis.metric_standards,
			),
			use_container_width=True,
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
				    <span>Original: {format_metric(original_value)}</span>
				    <span>Edited: {format_metric(edited_value)}</span>
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
							y=["Original", "AI-Edited", "Human Std", "AI Std"],
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
				st.plotly_chart(mini, use_container_width=True)
				st.markdown(
					f"""
					<div class="vt-muted">{metric['description']}</div>
					<div class="vt-muted">Edited text shows a {format_metric(abs(delta))} point change vs original.</div>
					""",
					unsafe_allow_html=True,
				)

	with right_col:
		st.markdown("<div class='vt-section-title'>Visual Evidence</div>", unsafe_allow_html=True)
		st.caption("Sentence rhythm across the text (green: original, red: AI-edited).")
		st.plotly_chart(build_line_chart(analysis.sentence_lengths), use_container_width=True)
		st.caption("Component scores used for the final score.")
		st.plotly_chart(build_bar_chart(analysis.components), use_container_width=True)
		st.caption("AI-ism category distribution.")
		st.plotly_chart(build_pie_chart(analysis.ai_ism_categories), use_container_width=True)
		with st.expander("AI-isms Detected"):
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
			st.session_state.original_text,
			st.session_state.edited_text,
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
		f"<div class='vt-muted'>Negotiating voice preservation for: {metric_focus}</div>",
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
		st.metric("Original score", f"{original_score:.1f}")
	with score_mid:
		delta_metric = edited_score - original_score
		st.metric("AI-Edited score", f"{edited_score:.1f}", delta=f"{delta_metric:+.1f}")
	with score_right:
		delta_overall = projected_score - analysis.score
		st.metric("Projected overall (estimate)", f"{projected_score:.1f}", delta=f"{delta_overall:+.1f}")
	st.caption("Projected overall score is an estimate based on component weighting.")
	col_left, col_mid, col_right = st.columns([1, 1, 1.3], gap="large")
	with col_left:
		st.markdown("<div class='vt-card vt-subtle'><div class='vt-card-title'>Your Original Voice</div></div>", unsafe_allow_html=True)
		st.text_area("Original", value=st.session_state.original_text, height=260, disabled=True)
	with col_mid:
		st.markdown("<div class='vt-card vt-subtle'><div class='vt-card-title'>AI Edit</div></div>", unsafe_allow_html=True)
		st.text_area("AI Edit", value=st.session_state.edited_text, height=260, disabled=True)
	with col_right:
		st.markdown("<div class='vt-card vt-subtle'><div class='vt-card-title'>Your Choice</div></div>", unsafe_allow_html=True)
		choice = st.radio("Negotiated Options", options=["Option A", "Option B", "Option C", "Custom"], horizontal=True)
		custom_text = ""
		if choice == "Custom":
			custom_text = st.text_area("Custom Rewrite", height=180)
		st.button("Apply Selection", use_container_width=True)
		if choice == "Custom" and custom_text.strip():
			custom_standards = None
			if not st.session_state.use_default_standards:
				custom_standards = st.session_state.calibration
			try:
				custom_analysis = analyze_texts(
					st.session_state.original_text,
					custom_text,
					custom_standards=custom_standards,
				)
				delta_custom = custom_analysis.score - analysis.score
				st.metric("Custom overall score", f"{custom_analysis.score:.1f}", delta=f"{delta_custom:+.1f}")
			except ValueError:
				st.warning("Custom text is too short to score reliably.")

	st.markdown("<div class='vt-section-title'>Repair Suggestions</div>", unsafe_allow_html=True)
	suggestions = _build_repair_suggestions(
		metric_focus,
		analysis,
		st.session_state.original_text,
		st.session_state.edited_text,
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
		"<div class='vt-metric-rows'><span>Skip to Next Issue</span> | <span>Accept All AI Edits</span> | <span>Restore All Original</span> | <span>Generate Final Text</span></div>",
		unsafe_allow_html=True,
	)


def render_calibration() -> None:
	st.markdown("<div class='vt-section-title'>Calibration Panel</div>", unsafe_allow_html=True)
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
				key=f"human_{key}",
			)
			ai_value = st.slider(
				"AI Standard",
				min_value=0.0,
				max_value=100.0,
				value=float(current["ai"]),
				step=0.01,
				key=f"ai_{key}",
			)
			st.selectbox(
				"Sensitivity",
				options=["Strict", "Moderate", "Permissive"],
				index=1,
				key=f"sensitivity_{key}",
			)
			st.session_state.calibration[key] = {"human": human_value, "ai": ai_value}
			if st.button(f"Reset {label}", key=f"reset_{key}"):
				st.session_state.calibration[key] = {
					"human": defaults["human"],
					"ai": defaults["ai"],
				}
				st.rerun()
	st.markdown("<div class='vt-section-title'>Impact Preview</div>", unsafe_allow_html=True)
	st.markdown(
		"<div class='vt-muted'>Adjusting thresholds will update verdicts on the next analysis run.</div>",
		unsafe_allow_html=True,
	)
	st.button("Save as Custom Profile")
	st.button("Export Calibration Settings")
	st.button("Reset All")


def render_documentation_export() -> None:
	st.markdown("<div class='vt-section-title'>Documentation Export</div>", unsafe_allow_html=True)
	st.selectbox("Report Type", options=["PDF", "Word", "Excel", "JSON"])
	st.markdown("<div class='vt-section-title'>Sections to Include</div>", unsafe_allow_html=True)
	for section in [
		"Executive Summary",
		"Full Metric Analysis",
		"Visualizations",
		"Repair Preview Decisions",
		"Original vs Final Text Comparison",
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
stored_original = local_storage.getItem(ORIGINAL_TEXT_KEY) if local_storage else ""
stored_edited = local_storage.getItem(EDITED_TEXT_KEY) if local_storage else ""
if not st.session_state.original_text:
	st.session_state.original_text = stored_original or ""
if not st.session_state.edited_text:
	st.session_state.edited_text = stored_edited or ""

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

if st.session_state.page == "Upload & Configuration":
	render_upload_screen(local_storage)
elif st.session_state.page == "Analysis Dashboard":
	render_dashboard_screen()
elif st.session_state.page == "Repair Preview":
	render_repair_preview()
elif st.session_state.page == "Calibration":
	render_calibration()
elif st.session_state.page == "Documentation Export":
	render_documentation_export()
elif st.session_state.page == "Settings":
	render_settings()
