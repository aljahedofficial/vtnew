"""Reporting and export utilities for VoiceTracer."""

import json
from dataclasses import asdict
from io import BytesIO
import datetime
from typing import Dict, List, Any, Optional

try:
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
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
            doc.add_paragraph(f"Overall Voice Preservation Score: {analysis.score:.2f}/100")
            doc.add_paragraph(f"Classification: {analysis.classification}")
            doc.add_paragraph(f"Consistency Score: {analysis.consistency_score:.2f}")

        # Data Only Section (Requested detailed data)
        # Note: We'll include this if "Executive Summary" or "Full Metric Analysis" is checked, 
        # as it contains the raw data points.
        doc.add_heading("Detailed Analysis Data", level=1)
        
        # Component Breakdown Table
        doc.add_heading("Component Breakdown", level=2)
        comp_table = doc.add_table(rows=1, cols=2)
        comp_table.style = 'Table Grid'
        comp_table.rows[0].cells[0].text = "Component"
        comp_table.rows[0].cells[1].text = "Score"
        
        components = [
            ("Authenticity Markers", analysis.components.get("Authenticity", 0.0)),
            ("Lexical Identity", analysis.components.get("Lexical", 0.0)),
            ("Structural Identity", analysis.components.get("Structural", 0.0)),
            ("Stylistic Identity", analysis.components.get("Stylistic", 0.0)),
            ("Voice Consistency", analysis.consistency_score),
        ]
        for name, val in components:
            row = comp_table.add_row().cells
            row[0].text = name
            row[1].text = f"{val:.2f}"

        # Quick Stats Table
        doc.add_heading("Quick Stats", level=2)
        stats_table = doc.add_table(rows=1, cols=2)
        stats_table.style = 'Table Grid'
        stats_table.rows[0].cells[0].text = "Metric"
        stats_table.rows[0].cells[1].text = "Value"
        
        stats = [
            ("Word Delta", str(analysis.word_delta)),
            ("Sentence Delta", str(analysis.sentence_delta)),
            ("AI-isms Total", str(analysis.ai_ism_total)),
            ("Original Word Count", str(analysis.original_word_count)),
            ("Rewrite Word Count", str(analysis.edited_word_count)),
            ("Original Sentence Count", str(analysis.original_sentence_count)),
            ("Rewrite Sentence Count", str(analysis.edited_sentence_count)),
        ]
        for name, val in stats:
            row = stats_table.add_row().cells
            row[0].text = name
            row[1].text = val

        # Similarity Index
        doc.add_heading("Similarity Index", level=2)
        sim_table = doc.add_table(rows=1, cols=2)
        sim_table.style = 'Table Grid'
        sim_table.rows[0].cells[0].text = "Metric"
        sim_table.rows[0].cells[1].text = "Score"
        
        sims = [
            ("Source vs Rewrite (Cosine)", f"{analysis.source_similarity_cosine * 100:.1f}%"),
            ("Source vs Rewrite (N-gram)", f"{analysis.source_similarity_ngram * 100:.1f}%"),
            ("AI vs Rewrite (Cosine)", f"{analysis.ai_similarity_cosine * 100:.1f}%"),
            ("AI vs Rewrite (N-gram)", f"{analysis.ai_similarity_ngram * 100:.1f}%"),
        ]
        for name, val in sims:
            row = sim_table.add_row().cells
            row[0].text = name
            row[1].text = val

        # Visualizations
        if "Visualizations" in sections_to_include and images:
            doc.add_heading("Visualizations", level=1)
            # Order them logically
            order = ["Voice Identity Spectrum", "Sentence Length Variation", "Metric Standards", "AI Source AI-isms", "Writer Rewrite AI-isms"]
            for name in order:
                if name in images:
                    doc.add_heading(name, level=2)
                    img_stream = BytesIO(images[name])
                    doc.add_picture(img_stream, width=Inches(5.5))

        # Full Metric Analysis
        if "Full Metric Analysis" in sections_to_include:
            doc.add_heading("Full Metric Deep-Dive", level=1)
            table = doc.add_table(rows=1, cols=4)
            table.style = 'Table Grid'
            hdr_cells = table.rows[0].cells
            hdr_cells[0].text = 'Metric'
            hdr_cells[1].text = 'AI Source'
            hdr_cells[2].text = 'Rewrite'
            hdr_cells[3].text = 'Verdict'

            # Define metrics to show
            for label, original_val in analysis.metrics_original.items():
                edited_val = analysis.metrics_edited.get(label, 0.0)
                # Find verdict
                verdict = "N/A"
                for res in analysis.metric_results.values():
                    if res.name == label:
                        verdict = res.verdict
                        break
                
                row_cells = table.add_row().cells
                row_cells[0].text = label
                row_cells[1].text = f"{original_val:.4f}"
                row_cells[2].text = f"{edited_val:.4f}"
                row_cells[3].text = verdict

        # Repair Preview Decisions
        if "Repair Preview Decisions" in sections_to_include and negotiated_text:
            doc.add_heading("Repair Preview Decisions", level=1)
            doc.add_heading("Final Negotiated Rewrite", level=2)
            doc.add_paragraph(negotiated_text)

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
            pdf.cell(0, 10, f"Overall Voice Preservation Score: {analysis.score:.2f}/100", 0, 1)
            pdf.cell(0, 10, f"Classification: {analysis.classification}", 0, 1)
            pdf.cell(0, 10, f"Consistency Score: {analysis.consistency_score:.2f}", 0, 1)
            pdf.ln(5)

        # Data Only Sections
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Detailed Analysis Data", 0, 1)
        pdf.ln(2)

        # Component Breakdown
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "Component Breakdown", 0, 1)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(100, 10, "Component", 1)
        pdf.cell(40, 10, "Score", 1)
        pdf.ln()
        pdf.set_font("Helvetica", "", 10)
        components = [
            ("Authenticity Markers", analysis.components.get("Authenticity", 0.0)),
            ("Lexical Identity", analysis.components.get("Lexical", 0.0)),
            ("Structural Identity", analysis.components.get("Structural", 0.0)),
            ("Stylistic Identity", analysis.components.get("Stylistic", 0.0)),
            ("Voice Consistency", analysis.consistency_score),
        ]
        for name, val in components:
            pdf.cell(100, 10, name, 1)
            pdf.cell(40, 10, f"{val:.2f}", 1)
            pdf.ln()
        pdf.ln(5)

        # Quick Stats
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "Quick Stats", 0, 1)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(100, 10, "Metric", 1)
        pdf.cell(40, 10, "Value", 1)
        pdf.ln()
        pdf.set_font("Helvetica", "", 10)
        stats = [
            ("Word Delta", str(analysis.word_delta)),
            ("Sentence Delta", str(analysis.sentence_delta)),
            ("AI-isms Total", str(analysis.ai_ism_total)),
            ("Original Word Count", str(analysis.original_word_count)),
            ("Rewrite Word Count", str(analysis.edited_word_count)),
            ("Original Sentence Count", str(analysis.original_sentence_count)),
            ("Rewrite Sentence Count", str(analysis.edited_sentence_count)),
        ]
        for name, val in stats:
            pdf.cell(100, 10, name, 1)
            pdf.cell(40, 10, val, 1)
            pdf.ln()
        pdf.ln(5)

        # Similarity Index
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "Similarity Index", 0, 1)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(100, 10, "Metric", 1)
        pdf.cell(40, 10, "Score", 1)
        pdf.ln()
        pdf.set_font("Helvetica", "", 10)
        sims = [
            ("Source vs Rewrite (Cosine)", f"{analysis.source_similarity_cosine * 100:.1f}%"),
            ("Source vs Rewrite (N-gram)", f"{analysis.source_similarity_ngram * 100:.1f}%"),
            ("AI vs Rewrite (Cosine)", f"{analysis.ai_similarity_cosine * 100:.1f}%"),
            ("AI vs Rewrite (N-gram)", f"{analysis.ai_similarity_ngram * 100:.1f}%"),
        ]
        for name, val in sims:
            pdf.cell(100, 10, name, 1)
            pdf.cell(40, 10, val, 1)
            pdf.ln()
        pdf.ln(5)

        # Visualizations
        if "Visualizations" in sections_to_include and images:
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "Visualizations", 0, 1)
            order = ["Voice Identity Spectrum", "Sentence Length Variation", "Metric Standards", "AI Source AI-isms", "Writer Rewrite AI-isms"]
            for name in order:
                if name in images:
                    pdf.set_font("Helvetica", "B", 12)
                    pdf.cell(0, 10, name, 0, 1)
                    img_stream = BytesIO(images[name])
                    pdf.image(img_stream, x=10, w=180) 
                    pdf.ln(5)

        # Full Metric Analysis
        if "Full Metric Analysis" in sections_to_include:
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "Detailed Metric Deep-Dive", 0, 1)
            
            # Table Header
            pdf.set_font("Helvetica", "B", 10)
            col_widths = [60, 30, 30, 40]
            headers = ["Metric", "AI Source", "Rewrite", "Verdict"]
            for i in range(len(headers)):
                pdf.cell(col_widths[i], 10, headers[i], 1)
            pdf.ln()

            # Table Rows
            pdf.set_font("Helvetica", "", 10)
            for label, original_val in analysis.metrics_original.items():
                edited_val = analysis.metrics_edited.get(label, 0.0)
                verdict = "N/A"
                for res in analysis.metric_results.values():
                    if res.name == label:
                        verdict = res.verdict
                        break
                
                pdf.cell(col_widths[0], 10, label, 1)
                pdf.cell(col_widths[1], 10, f"{original_val:.3f}", 1)
                pdf.cell(col_widths[2], 10, f"{edited_val:.3f}", 1)
                pdf.cell(col_widths[3], 10, verdict, 1)
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
            verdict_str = "N/A"
            for res in analysis.metric_results.values():
                if res.name == name:
                    verdict_str = res.verdict
                    break
            writer.writerow([name, original, edited, edited - original, verdict_str])
            
        return output.getvalue().encode("utf-8")
