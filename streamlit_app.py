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


ASSETS_DIR = Path(__file__).parent / "assets"
LOGO_PATH = ASSETS_DIR / "logo.svg"
FAVICON_PATH = ASSETS_DIR / "favicon.svg"


def load_css() -> None:
	css_path = ASSETS_DIR / "styles.css"
	if css_path.exists():
		st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def word_count(text: str) -> int:
	return len([w for w in text.split() if w.strip()])


def format_metric(value: object) -> str:
	if isinstance(value, (int, float)):
		return f"{value:.2f}"
	return str(value)


def word_count_notice(label: str, count: int) -> None:
	if count == 0:
		return
	if count < 200:
		st.warning(f"{label} is short ({count} words). Aim for 200-500 words.")
	elif count > 500:
		st.info(f"{label} is long ({count} words). Results may take longer to read.")


page_icon = str(FAVICON_PATH) if FAVICON_PATH.exists() else "🧭"
st.set_page_config(page_title="VoiceTracer", layout="wide", page_icon=page_icon)
load_css()

header_left, header_right = st.columns([1, 6], gap="small")
with header_left:
	if LOGO_PATH.exists():
		st.image(str(LOGO_PATH), width=64)
	else:
		st.markdown("<div class='vt-logo-fallback'>VT</div>", unsafe_allow_html=True)
with header_right:
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
	original_wc = word_count(st.session_state.original_text)
	st.caption(f"Word count: {original_wc}")
	word_count_notice("Original draft", original_wc)
	st.button("Clear Original", on_click=clear_original, use_container_width=True)

with right_col:
	st.text_area(
		"AI-Edited Version",
		key="edited_text",
		height=240,
		placeholder="Paste your AI-edited text here...",
		on_change=save_edited,
	)
	edited_wc = word_count(st.session_state.edited_text)
	st.caption(f"Word count: {edited_wc}")
	word_count_notice("AI-edited text", edited_wc)
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
	help="Runs the analysis on the two texts.",
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
		  <div class="vt-card-value">{format_metric(analysis.score) if analysis else '--'}</div>
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
			metric_value = format_metric(analysis.metrics.get(title, "--"))
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

st.markdown("<div class='vt-section-title'>Metric Notes</div>", unsafe_allow_html=True)
st.markdown(
	"""
<div class="vt-muted">
<ul>
  <li>Burstiness: variation in sentence length.</li>
  <li>Lexical Diversity: vocabulary range and repetition.</li>
  <li>Syntactic Complexity: presence of subordination.</li>
  <li>AI-ism Likelihood: formulaic phrases often seen in AI edits.</li>
  <li>Function Word Ratio: density of connector words.</li>
  <li>Discourse Markers: signposting such as however, therefore.</li>
  <li>Information Density: content word ratio.</li>
  <li>Epistemic Hedging: cautious language (might, suggests).</li>
</ul>
</div>
""",
	unsafe_allow_html=True,
)

st.divider()

st.markdown("<div class='vt-section-title'>Visualizations</div>", unsafe_allow_html=True)
st.caption("Charts summarize how the AI edit shifts your voice across metrics.")

if analysis:
	chart_left, chart_right = st.columns(2, gap="large")
	with chart_left:
		st.caption("Gauge: overall voice preservation score.")
		st.plotly_chart(build_gauge_chart(analysis.score), use_container_width=True)
		st.caption("Radar: eight stylistic dimensions (0-100).")
		st.plotly_chart(build_radar_chart(analysis.metrics), use_container_width=True)
	with chart_right:
		st.caption("Line: sentence length rhythm across the text.")
		st.plotly_chart(build_line_chart(analysis.sentence_lengths), use_container_width=True)
		st.caption("Bar: component scores used for the final score.")
		st.plotly_chart(build_bar_chart(analysis.components), use_container_width=True)
	st.caption("Pie: AI-ism category distribution.")
	st.plotly_chart(build_pie_chart(analysis.ai_ism_categories), use_container_width=True)

else:
	st.info("Run analysis to view charts.")
