"""Render the Report Agent's five sections plus charts into a PDF (reportlab)."""
from __future__ import annotations

import html
import re
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

SECTIONS = ["Executive Summary", "Methodology", "Findings", "Limitations", "Recommendations"]
ACCENT = colors.HexColor("#2a78d6")

_styles = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=_styles["BodyText"], fontSize=10, leading=14)
H1 = ParagraphStyle("h1", parent=_styles["Heading1"], textColor=colors.HexColor("#0b0b0b"))
H2 = ParagraphStyle("h2", parent=_styles["Heading2"], textColor=ACCENT, spaceBefore=12)
CAPTION = ParagraphStyle("cap", parent=BODY, fontSize=8.5, textColor=colors.HexColor("#52514e"))


def _inline(text: str) -> str:
    """Escape for reportlab and turn **bold** / *italic* markdown into tags."""
    t = html.escape(text.strip(), quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<i>\1</i>", t)
    return t


def markdown_to_flowables(text: str) -> list:
    """Paragraphs separated by blank lines; lines starting with '-', '*' or '1.' become bullets."""
    flow, bullets = [], []

    def flush():
        if bullets:
            flow.append(ListFlowable([ListItem(Paragraph(b, BODY), leftIndent=12) for b in bullets],
                                     bulletType="bullet", start="•", leftIndent=12))
            bullets.clear()

    para: list[str] = []
    for line in (text or "").splitlines() + [""]:
        m = re.match(r"^\s*(?:[-*•]|\d+[.)])\s+(.*)", line)
        if m:
            if para:
                flow.append(Paragraph(_inline(" ".join(para)), BODY))
                para = []
            bullets.append(_inline(m.group(1)))
        elif not line.strip():
            if para:
                flow.append(Paragraph(_inline(" ".join(para)), BODY))
                para = []
            flush()
        else:
            if line.lstrip().startswith("#"):
                line = line.lstrip("# ")
            flush()
            para.append(line)
    flush()
    return flow


def _image(path: str, max_w: float = 6.2 * inch, max_h: float = 4.3 * inch) -> Image:
    img = Image(path)
    ratio = min(max_w / img.imageWidth, max_h / img.imageHeight)
    img.drawWidth, img.drawHeight = img.imageWidth * ratio, img.imageHeight * ratio
    return img


def _metrics_table(scenarios: dict) -> Table:
    first = next(iter(scenarios.values()))
    keys = list(first["test_scores"].keys())
    rows = [["Scenario", "Selected model", *[k.upper() for k in keys]]]
    for name, s in scenarios.items():
        rows.append([name, s["selected_model"], *[f"{s['test_scores'][k]:.3f}" for k in keys]])
    rows.append(["Baseline", first["baseline"]["description"], *[f"{first['baseline'][k]:.3f}" for k in keys]])
    t = Table(rows, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
        ("FONT", (0, 1), (-1, -1), "Helvetica", 8.5),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f1ee")]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d0cfca")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def caption_for(chart_path: str) -> str:
    name = Path(chart_path).stem
    for prefix, text in [
        ("target_distribution", "How the target values are spread across the dataset."),
        ("correlation_heatmap", "Pairwise correlations; dark red and dark blue are strong positive and negative relationships."),
        ("model_comparison", "Validation scores of the three candidate models (higher is better)."),
        ("scenario_comparison", "Final test-set score for each scenario against a naive baseline."),
        ("feature_importance", "How much the test score drops when each feature is shuffled; longer bars matter more."),
        ("predicted_vs_actual", "Each dot is a test-set row; dots on the dashed line were predicted exactly."),
        ("confusion_matrix", "Rows are the true class, columns the predicted class; the diagonal is correct."),
    ]:
        if name.startswith(prefix):
            return text
    return ""


def build_pdf(
    out_path: str | Path,
    title: str,
    goal: str,
    sections: dict[str, str],
    scenarios: dict,
    charts: list[str],
) -> str:
    out_path = Path(out_path)
    doc = SimpleDocTemplate(str(out_path), pagesize=LETTER, leftMargin=0.8 * inch, rightMargin=0.8 * inch,
                            topMargin=0.7 * inch, bottomMargin=0.7 * inch, title=title)
    story: list = [
        Paragraph(_inline(title), H1),
        Paragraph(f"Generated {datetime.now():%B %d, %Y} by the AI data team "
                  "(Verify Agent → Data Science Agent → Report Agent).", CAPTION),
        Spacer(1, 6),
        Paragraph(f"<b>Goal:</b> {_inline(goal)}", BODY),
    ]
    for sec in SECTIONS:
        story.append(Paragraph(sec, H2))
        story.extend(markdown_to_flowables(sections.get(sec, "")))
        if sec == "Findings" and scenarios:
            story += [Spacer(1, 6), Paragraph("<b>Test-set results</b>", BODY), Spacer(1, 3), _metrics_table(scenarios)]
    if charts:
        story += [PageBreak(), Paragraph("Charts", H2)]
        for c in charts:
            if Path(c).exists():
                story.append(KeepTogether([_image(c), Paragraph(caption_for(c), CAPTION), Spacer(1, 14)]))
    doc.build(story)
    return str(out_path)
