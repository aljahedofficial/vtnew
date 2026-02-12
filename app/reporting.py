"""Reporting and export utilities for VoiceTracer."""

import json
from dataclasses import asdict
from io import BytesIO
import datetime
from typing import Dict, List, Any, Optional

try:
    from docx import Document
    from docx.shared import Inches, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH
except ImportError:
    Document = None

try:
    from fpdf import FPDF
except ImportError:
    FPDF = None

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
    def generate_docx(
        cls, 
        analysis: AnalysisResult, 
        sections_to_include: List[str], 
        statement: str,
        negotiated_text: Optional[str] = None,
        images: Optional[Dict[str, bytes]] = None
    ) -> bytes:
        """Generate a Word document report."""
        if not Document:
            return b"Error: python-docx not installed."
            
        doc = Document()
        
        # Header
        doc.add_heading("VoiceTracer Analysis Report", 0)
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p.add_run(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}").italic = True

        # Executive Summary
        if "Executive Summary" in sections_to_include:
            doc.add_heading("Executive Summary", level=1)
            doc.add_paragraph(f"Overall Voice Preservation Score: {analysis.score:.1f}/100")
            doc.add_paragraph(f"Classification: {analysis.classification}")
            doc.add_paragraph(f"Consistency Score: {analysis.consistency_score:.2f}")

        # Visualizations
        if "Visualizations" in sections_to_include and images:
            doc.add_heading("Visualizations", level=1)
            for name, img_bytes in images.items():
                doc.add_heading(name, level=2)
                img_stream = BytesIO(img_bytes)
                doc.add_picture(img_stream, width=Inches(6))

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

        # Repair Preview Decisions
        if "Repair Preview Decisions" in sections_to_include and negotiated_text:
            doc.add_heading("Repair Preview Decisions", level=1)
            doc.add_heading("Final Negotiated Rewrite", level=2)
            doc.add_paragraph(negotiated_text)

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
    def generate_pdf(
        cls, 
        analysis: AnalysisResult, 
        sections_to_include: List[str], 
        statement: str,
        negotiated_text: Optional[str] = None,
        images: Optional[Dict[str, bytes]] = None
    ) -> bytes:
        """Generate a PDF document report using fpdf2."""
        if not FPDF:
            # If FPDF is not available, we can't generate a PDF.
            # But on Streamlit Cloud, it should be installed via requirements.txt.
            return b"Error: fpdf2 not installed."

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        # Title
        pdf.set_font("Helvetica", "B", 24)
        pdf.cell(0, 20, "VoiceTracer Analysis Report", 0, 1, "C")
        
        # Date
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 10, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, "R")
        pdf.ln(10)

        # Executive Summary
        if "Executive Summary" in sections_to_include:
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "Executive Summary", 0, 1)
            pdf.set_font("Helvetica", "", 12)
            pdf.cell(0, 10, f"Overall Voice Preservation Score: {analysis.score:.1f}/100", 0, 1)
            pdf.cell(0, 10, f"Classification: {analysis.classification}", 0, 1)
            pdf.cell(0, 10, f"Consistency Score: {analysis.consistency_score:.2f}", 0, 1)
            pdf.ln(5)

        # Visualizations
        if "Visualizations" in sections_to_include and images:
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "Visualizations", 0, 1)
            for name, img_bytes in images.items():
                pdf.set_font("Helvetica", "B", 12)
                pdf.cell(0, 10, name, 0, 1)
                img_stream = BytesIO(img_bytes)
                # FPDF can't take BytesIO directly in all versions, let's use a temporary name or check
                # For fpdf2, it can handle some stream-like objects or we can save it to a tmp file.
                # Actually, fpdf2 can handle io.BytesIO if we pass it as 'name'
                pdf.image(img_stream, x=10, w=190) 
                pdf.ln(5)

        # Full Metric Analysis
        if "Full Metric Analysis" in sections_to_include:
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "Detailed Metric Analysis", 0, 1)
            
            # Table Header
            pdf.set_font("Helvetica", "B", 10)
            col_widths = [60, 40, 40, 40]
            headers = ["Metric", "Raw Value", "Score", "Verdict"]
            for i in range(len(headers)):
                pdf.cell(col_widths[i], 10, headers[i], 1)
            pdf.ln()

            # Table Rows
            pdf.set_font("Helvetica", "", 10)
            for key, result in analysis.metric_results.items():
                pdf.cell(col_widths[0], 10, result.name, 1)
                pdf.cell(col_widths[1], 10, f"{result.raw_value:.3f}", 1)
                pdf.cell(col_widths[2], 10, f"{result.normalized_score:.3f}", 1)
                pdf.cell(col_widths[3], 10, result.verdict, 1)
                pdf.ln()
            pdf.ln(5)

        # Repair Preview Decisions
        if "Repair Preview Decisions" in sections_to_include and negotiated_text:
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "Repair Preview Decisions", 0, 1)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 10, "Final Negotiated Rewrite:", 0, 1)
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 10, negotiated_text)
            pdf.ln(5)

        # Comparison
        if "AI Source vs Writer Rewrite Comparison" in sections_to_include:
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "Comparison Statistics", 0, 1)
            pdf.set_font("Helvetica", "", 12)
            pdf.cell(0, 10, f"Original Word Count: {analysis.original_word_count}", 0, 1)
            pdf.cell(0, 10, f"Edited Word Count: {analysis.edited_word_count}", 0, 1)
            pdf.cell(0, 10, f"Word Delta: {analysis.word_delta:+d}", 0, 1)
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 10, "Similarity Scores:", 0, 1)
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(0, 10, f"AI vs Rewrite (Cosine): {analysis.ai_similarity_cosine:.4f}", 0, 1)
            pdf.cell(0, 10, f"AI vs Rewrite (N-gram): {analysis.ai_similarity_ngram:.4f}", 0, 1)
            pdf.cell(0, 10, f"Source vs Rewrite (Cosine): {analysis.source_similarity_cosine:.4f}", 0, 1)
            pdf.cell(0, 10, f"Source vs Rewrite (N-gram): {analysis.source_similarity_ngram:.4f}", 0, 1)
            pdf.ln(5)

        # Authorship Statement
        if "Authorship Documentation Statement" in sections_to_include:
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "Authorship Statement", 0, 1)
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 8, statement)

        return bytes(pdf.output())

    @classmethod
    def generate_excel(cls, analysis: AnalysisResult) -> bytes:
        """Generate a basic CSV/Excel compatible export."""
        import csv
        from io import StringIO
        
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(["Metric", "AI Source", "Writer Rewrite", "Delta", "Verdict"])
        
        for name, original in analysis.metrics_original.items():
            edited = analysis.metrics_edited.get(name, 0.0)
            # Find the result object for name
            verdict_str = "N/A"
            for res in analysis.metric_results.values():
                if res.name == name:
                    verdict_str = res.verdict
                    break
            writer.writerow([name, original, edited, edited - original, verdict_str])
            
        return output.getvalue().encode("utf-8")
