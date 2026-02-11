# VoiceTracer WebApp Implementation Plan

## Overview
Single-page webapp focused on fast reload, minimal UI, and strong data visualization. No backend, no auth, no user accounts. Client-side autosave keeps text input between reloads. Streamlit is the initial deployment target, but the plan keeps the logic and UI modular for future migration.

## Product Goals
- Measure voice preservation between original and AI-edited text.
- Visualize results clearly with charts.
- Provide a clean, minimal, fast experience.
- Keep the app free to deploy.

## Scope
In scope:
- Single-page analysis flow (Upload -> Analysis).
- Autosave input in browser storage.
- Clear buttons for both text areas.
- Charts: radar, line, bar, pie, gauge.
- Static assets: logo, icons, CSS, fonts.

Out of scope (for now):
- Auth, user accounts, saved projects, audit trails.
- Paid hosting or backend services.

## Tech Stack (Phase 1)
- Streamlit (UI framework)
- Plotly (charts: line, bar, pie, radar, gauge)
- streamlit-local-storage (client-side autosave)
- Pure Python for computation

## App Architecture
### UI Layer
- Streamlit layout with clear sections and simple interaction.
- Minimal CSS for spacing and card style.
- Subtle transition only for hover and score gauge.

### Logic Layer
- Lightweight analysis functions (placeholder first, then real metrics).
- Consistent data schema for charts.
- Readable, testable functions separated from UI.

### Persistence Layer
- Client-side localStorage for autosave.
- No server-side storage.

## Repository Structure
- streamlit_app.py: main entry
- app/analysis.py: text processing and metrics
- app/charts.py: Plotly chart builders
- app/state.py: autosave helpers
- assets/styles.css: minimal styles
- assets/logo.svg: logo
- .streamlit/config.toml: theme tokens
- requirements.txt: dependencies
- docs/IMPLEMENTATION.md: plan
- docs/TASKS.md: task list

## UI Sections
1. Upload
	- Original vs AI-edited inputs
	- Word counts
	- Clear buttons
2. Analysis Summary
	- Voice Preservation Score
	- Classification label
3. Visualizations
	- Radar chart (8 metrics)
	- Line chart (sentence rhythm)
	- Bar chart (component scores)
	- Pie chart (AI-isms distribution)
4. Notes
	- Short explanatory copy per metric

## Autosave
- Save original and edited text into localStorage on each input change.
- Restore values on app load.
- Clear buttons reset values and delete localStorage keys.

## Accessibility
- WCAG-friendly contrast in default theme.
- Chart legends and labels visible by default.
- Clear headings and input labels.

## Deployment Options
Primary (free): Streamlit Community Cloud.
Alternatives (free tiers):
- Hugging Face Spaces (Streamlit or Gradio)
- Render (free tier, limited)

## Future Expansion (Phase 2+)
- Full Repair Preview workflow.
- Exportable reports.
- Calibration panel.
- Optional backend for collaboration.
