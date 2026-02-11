from pathlib import Path

import streamlit as st

try:
	from streamlit_local_storage import LocalStorage
except ImportError:
	LocalStorage = None

from app.analysis import analyze_texts
from app.charts import (
	build_bar_chart,
	build_gauge_chart,
	build_line_chart,
	build_pie_chart,
	build_radar_chart,
)
from app.state import EDITED_TEXT_KEY, ORIGINAL_TEXT_KEY


def load_css() -> None:
	css_path = Path(__file__).parent / "assets" / "styles.css"
	if css_path.exists():
		st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def word_count(text: str) -> int:
	return len([w for w in text.split() if w.strip()])


st.set_page_config(page_title="VoiceTracer", layout="wide")
load_css()

st.title("VoiceTracer")
st.caption("Preserve Your Voice. Navigate AI with Autonomy.")

local_storage = LocalStorage() if LocalStorage else None

stored_original = local_storage.getItem(ORIGINAL_TEXT_KEY) if local_storage else ""
stored_edited = local_storage.getItem(EDITED_TEXT_KEY) if local_storage else ""

if "original_text" not in st.session_state:
	st.session_state.original_text = stored_original or ""
if "edited_text" not in st.session_state:
	st.session_state.edited_text = stored_edited or ""
if "analysis" not in st.session_state:
	st.session_state.analysis = None


def save_original() -> None:
	if local_storage:
		local_storage.setItem(ORIGINAL_TEXT_KEY, st.session_state.original_text)
	st.session_state.analysis = None


def save_edited() -> None:
	if local_storage:
		local_storage.setItem(EDITED_TEXT_KEY, st.session_state.edited_text)
	st.session_state.analysis = None


def clear_original() -> None:
	st.session_state.original_text = ""
	if local_storage:
		local_storage.deleteItem(ORIGINAL_TEXT_KEY)
	st.session_state.analysis = None


def clear_edited() -> None:
	st.session_state.edited_text = ""
	if local_storage:
		local_storage.deleteItem(EDITED_TEXT_KEY)
	st.session_state.analysis = None


def run_analysis() -> None:
	st.session_state.analysis = analyze_texts(
		st.session_state.original_text,
		st.session_state.edited_text,
	)


if LocalStorage is None:
	st.warning("Autosave is unavailable because streamlit-local-storage is missing.")

st.markdown("<div class='vt-section-title'>Upload</div>", unsafe_allow_html=True)
st.markdown(
	"<div class='vt-muted'>Paste your original draft and the AI-edited version.</div>",
	unsafe_allow_html=True,
)

left_col, right_col = st.columns(2, gap="large")

with left_col:
	st.text_area(
		"Original Draft",
		key="original_text",
		height=240,
		placeholder="Paste your original text here...",
		on_change=save_original,
	)
	st.caption(f"Word count: {word_count(st.session_state.original_text)}")
	st.button("Clear Original", on_click=clear_original, use_container_width=True)

with right_col:
	st.text_area(
		"AI-Edited Version",
		key="edited_text",
		height=240,
		placeholder="Paste your AI-edited text here...",
		on_change=save_edited,
	)
	st.caption(f"Word count: {word_count(st.session_state.edited_text)}")
	st.button("Clear AI-Edited", on_click=clear_edited, use_container_width=True)

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

st.divider()

st.markdown("<div class='vt-section-title'>Analysis Summary</div>", unsafe_allow_html=True)

summary_col, stats_col = st.columns([1, 2], gap="large")

with summary_col:
	analysis = st.session_state.analysis
	st.markdown(
		f"""
		<div class="vt-card vt-subtle">
		  <div class="vt-card-title">Voice Preservation Score</div>
		  <div class="vt-card-value">{analysis.score if analysis else '--'}</div>
		  <div class="vt-card-caption">{analysis.classification if analysis else 'Awaiting analysis'}</div>
		</div>
		""",
		unsafe_allow_html=True,
	)

with stats_col:
	stats = [
		(
			"Word Delta",
			analysis.word_delta if analysis else "--",
			"Original vs edited",
		),
		(
			"Sentence Delta",
			analysis.sentence_delta if analysis else "--",
			"Structure change",
		),
		(
			"AI-isms",
			analysis.ai_ism_total if analysis else "--",
			"Formulaic phrases",
		),
	]
	stat_cols = st.columns(3, gap="medium")
	for col, (title, value, caption) in zip(stat_cols, stats):
		with col:
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

st.markdown("<div class='vt-section-title'>Metric Cards</div>", unsafe_allow_html=True)

metric_items = [
	("Burstiness", "Sentence rhythm"),
	("Lexical Diversity", "Vocabulary range"),
	("Syntactic Complexity", "Structure depth"),
	("AI-ism Likelihood", "Formulaic phrasing"),
	("Function Word Ratio", "Connector density"),
	("Discourse Markers", "Signposting"),
	("Information Density", "Content ratio"),
	("Epistemic Hedging", "Uncertainty markers"),
]

metric_cols = st.columns(4, gap="medium")
for idx, (title, caption) in enumerate(metric_items):
	with metric_cols[idx % 4]:
		metric_value = "--"
		if analysis:
			metric_value = analysis.metrics.get(title, "--")
		st.markdown(
			f"""
			<div class="vt-card vt-subtle">
			  <div class="vt-card-title">{title}</div>
			  <div class="vt-card-value">{metric_value}</div>
			  <div class="vt-card-caption">{caption}</div>
			</div>
			""",
			unsafe_allow_html=True,
		)

st.divider()

st.markdown("<div class='vt-section-title'>Visualizations</div>", unsafe_allow_html=True)

if analysis:
	chart_left, chart_right = st.columns(2, gap="large")
	with chart_left:
		st.plotly_chart(build_gauge_chart(analysis.score), use_container_width=True)
		st.plotly_chart(build_radar_chart(analysis.metrics), use_container_width=True)
	with chart_right:
		st.plotly_chart(build_line_chart(analysis.sentence_lengths), use_container_width=True)
		st.plotly_chart(build_bar_chart(analysis.components), use_container_width=True)
	st.plotly_chart(build_pie_chart(analysis.ai_ism_categories), use_container_width=True)

else:
	st.info("Run analysis to view charts.")
