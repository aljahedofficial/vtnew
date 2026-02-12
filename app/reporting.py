"""Reporting and export utilities for VoiceTracer."""

import json
from dataclasses import asdict
from io import BytesIO
import datetime
from typing import Dict, List, Any

from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from app.analysis import AnalysisResult, MetricResult

class ReportGenerator:
    """Generates various report formats from AnalysisResult."""

    @staticmethod
    def to_dict(analysis: AnalysisResult) -> Dict[str, Any]:
        """Convert AnalysisResult (and nested dataclasses) to a dictionary."""
        return asdict(analysis)

    @classmethod
    def generate_json(cls, analysis: AnalysisResult) -> bytes:
        """Generate a JSON report."""
        data = cls.to_dict(analysis)
        data["generated_at"] = datetime.datetime.now().isoformat()
        return json.dumps(data, indent=4).encode("utf-8")

    @classmethod
    def generate_docx(cls, analysis: AnalysisResult, sections_to_include: List[str], statement: str) -> bytes:
        """Generate a Word document report."""
        doc = Document()
        
        # Header
        doc.add_heading("VoiceTracer Analysis Report", 0)
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p.add_run(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}").italic = True

        # Executive Summary
        if "Executive Summary" in sections_to_include:
            doc.add_heading("Executive Summary", level=1)
            doc.add_paragraph(f"Overall Voice Preservation Score: {analysis.score:.1f}")
            doc.add_paragraph(f"Classification: {analysis.classification}")
            doc.add_paragraph(f"Consistency Score: {analysis.consistency_score:.2f}")

        # Full Metric Analysis
        if "Full Metric Analysis" in sections_to_include:
            doc.add_heading("Detailed Metric Analysis", level=1)
            table = doc.add_table(rows=1, cols=4)
            table.style = 'Table Grid'
            hdr_cells = table.rows[0].cells
            hdr_cells[0].text = 'Metric'
            hdr_cells[1].text = 'Raw Value'
            hdr_cells[2].text = 'Score'
            hdr_cells[3].text = 'Verdict'

            for key, result in analysis.metric_results.items():
                row_cells = table.add_row().cells
                row_cells[0].text = result.name
                row_cells[1].text = f"{result.raw_value:.4f}"
                row_cells[2].text = f"{result.normalized_score:.4f}"
                row_cells[3].text = result.verdict

        # Comparison
        if "AI Source vs Writer Rewrite Comparison" in sections_to_include:
            doc.add_heading("Comparison Statistics", level=1)
            doc.add_paragraph(f"Original Word Count: {analysis.original_word_count}")
            doc.add_paragraph(f"Edited Word Count: {analysis.edited_word_count}")
            doc.add_paragraph(f"Word Delta: {analysis.word_delta:+d}")
            
            doc.add_heading("Similarity Scores", level=2)
            doc.add_paragraph(f"AI vs Rewrite (Cosine): {analysis.ai_similarity_cosine:.4f}")
            doc.add_paragraph(f"AI vs Rewrite (N-gram): {analysis.ai_similarity_ngram:.4f}")
            doc.add_paragraph(f"Source vs Rewrite (Cosine): {analysis.source_similarity_cosine:.4f}")
            doc.add_paragraph(f"Source vs Rewrite (N-gram): {analysis.source_similarity_ngram:.4f}")

        # Authorship Statement
        if "Authorship Documentation Statement" in sections_to_include:
            doc.add_heading("Authorship Statement", level=1)
            doc.add_paragraph(statement)

        # Save to Stream
        target = BytesIO()
        doc.save(target)
        return target.getvalue()

    @classmethod
    def generate_excel(cls, analysis: AnalysisResult) -> bytes:
        """Generate a basic CSV/Excel compatible export."""
        # Since pandas is not in requirements, we'll generate a CSV string
        import csv
        from io import StringIO
        
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(["Metric", "AI Source", "Writer Rewrite", "Delta", "Verdict"])
        
        for name, original in analysis.metrics_original.items():
            edited = analysis.metrics_edited.get(name, 0.0)
            verdict = analysis.metric_results.get(name.lower().replace(" ", "_"), None)
            verdict_str = verdict.verdict if verdict else "N/A"
            writer.writerow([name, original, edited, edited - original, verdict_str])
            
        return output.getvalue().encode("utf-8")
