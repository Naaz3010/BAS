"""
reports.py
Export reports in PDF, Excel, CSV, and PowerPoint formats, summarizing
KPIs, key tables, and the (AI or rule-based) executive summary.
Charts are not embedded; the interactive charts live in the app.
"""

import io
from xml.sax.saxutils import escape

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils.dataframe import dataframe_to_rows

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor


def format_value(v) -> str:
    """Human-friendly KPI formatting (1234567.891 -> 1,234,567.89)."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:,.2f}"
    return str(v)


def _clean_line(line: str) -> str:
    """Strip markdown bold and escape characters that ReportLab would treat as markup."""
    return escape(line.replace("**", "").strip())


def _fit_table(df: pd.DataFrame, max_cols: int = 8, max_rows: int = 15, max_chars: int = 18):
    """Limit a dataframe so it fits a portrait page: first N columns, short cells."""
    sub = df.iloc[:, :max_cols].head(max_rows)
    header = [str(c)[:max_chars] for c in sub.columns]
    rows = [[(f"{v:,.2f}" if isinstance(v, float) else str(v))[:max_chars] for v in r]
            for r in sub.itertuples(index=False)]
    return [header] + rows


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def export_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------------
# EXCEL
# ---------------------------------------------------------------------------

def export_excel(sheets: dict) -> bytes:
    """sheets: {sheet_name: dataframe}"""
    wb = Workbook()
    wb.remove(wb.active)

    header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    header_font = Font(color="FFFFFF", bold=True)

    for name, df in sheets.items():
        ws = wb.create_sheet(title=name[:31])
        for row in dataframe_to_rows(df, index=False, header=True):
            ws.append(row)
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
        for col_cells in ws.columns:
            length = max(len(str(c.value)) if c.value is not None else 0 for c in col_cells)
            ws.column_dimensions[col_cells[0].column_letter].width = min(max(length + 2, 10), 40)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def export_pdf(title: str, kpis: dict, summary_text: str, tables: dict = None) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], textColor=colors.HexColor("#1F4E78"))
    heading_style = ParagraphStyle("Heading", parent=styles["Heading2"], textColor=colors.HexColor("#1F4E78"))

    story = [Paragraph(escape(title), title_style), Spacer(1, 16)]

    story.append(Paragraph("Key Metrics", heading_style))
    kpi_data = [["Metric", "Value"]] + [[k.replace("_", " ").title(), format_value(v)] for k, v in kpis.items()]
    kpi_table = Table(kpi_data, colWidths=[220, 220])
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E78")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
    ]))
    story += [kpi_table, Spacer(1, 20)]

    story.append(Paragraph("Executive Summary", heading_style))
    for line in summary_text.split("\n"):
        if line.strip():
            story.append(Paragraph(_clean_line(line), styles["Normal"]))
    story.append(Spacer(1, 20))

    if tables:
        for name, df in tables.items():
            story.append(Paragraph(escape(str(name)), heading_style))
            data = _fit_table(df)
            t = Table(data, repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E78")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
            ]))
            story += [t, Spacer(1, 16)]

    doc.build(story)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# POWERPOINT
# ---------------------------------------------------------------------------

def export_pptx(title: str, kpis: dict, summary_text: str, top_products: pd.DataFrame = None) -> bytes:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    navy = RGBColor(0x1F, 0x4E, 0x78)

    # Title slide
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    box = slide.shapes.add_textbox(Inches(1), Inches(2.8), Inches(11.3), Inches(1.5))
    tf = box.text_frame
    tf.text = title
    tf.paragraphs[0].font.size = Pt(40)
    tf.paragraphs[0].font.bold = True
    tf.paragraphs[0].font.color.rgb = navy

    # KPI slide
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12), Inches(0.8))
    title_box.text_frame.text = "Key Performance Indicators"
    title_box.text_frame.paragraphs[0].font.size = Pt(30)
    title_box.text_frame.paragraphs[0].font.bold = True
    title_box.text_frame.paragraphs[0].font.color.rgb = navy

    rows = len(kpis) + 1
    table_shape = slide.shapes.add_table(rows, 2, Inches(1), Inches(1.3), Inches(11), Inches(0.5 * rows))
    table = table_shape.table
    table.cell(0, 0).text = "Metric"
    table.cell(0, 1).text = "Value"
    for i, (k, v) in enumerate(kpis.items(), start=1):
        table.cell(i, 0).text = k.replace("_", " ").title()
        table.cell(i, 1).text = format_value(v)

    # Summary slide
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12), Inches(0.8))
    title_box.text_frame.text = "Executive Summary"
    title_box.text_frame.paragraphs[0].font.size = Pt(30)
    title_box.text_frame.paragraphs[0].font.bold = True
    title_box.text_frame.paragraphs[0].font.color.rgb = navy

    body = slide.shapes.add_textbox(Inches(0.6), Inches(1.3), Inches(12), Inches(5.8))
    tf = body.text_frame
    tf.word_wrap = True
    first = True
    for line in summary_text.split("\n"):
        clean = line.replace("**", "").replace("- ", "• ").strip()
        if not clean:
            continue
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        p.text = clean
        p.font.size = Pt(16)
        first = False

    # Top products slide (optional)
    if top_products is not None and not top_products.empty:
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12), Inches(0.8))
        title_box.text_frame.text = "Top Products"
        title_box.text_frame.paragraphs[0].font.size = Pt(30)
        title_box.text_frame.paragraphs[0].font.bold = True
        title_box.text_frame.paragraphs[0].font.color.rgb = navy

        sample = top_products.head(10)
        rows, cols = sample.shape[0] + 1, sample.shape[1]
        table_shape = slide.shapes.add_table(rows, cols, Inches(0.5), Inches(1.3), Inches(12.3), Inches(0.4 * rows))
        table = table_shape.table
        for j, col in enumerate(sample.columns):
            table.cell(0, j).text = str(col)
        for i, (_, row) in enumerate(sample.iterrows(), start=1):
            for j, val in enumerate(row):
                table.cell(i, j).text = str(val)

    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()
