"""
utils/pdf_report.py
Feature 4 – PDF report generation using reportlab.
Produces a multi-page PDF: cover, stats, opportunities table, annotations.
"""
import io
from datetime import date
from typing import Dict, Optional

import pandas as pd

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, PageBreak, Image,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False


def _hex_to_rl_color(hex_color: str):
    """Convert #rrggbb to reportlab Color."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16)/255, int(h[2:4],16)/255, int(h[4:6],16)/255
    return colors.Color(r, g, b)


DARK_BG   = _hex_to_rl_color("#1e2233") if REPORTLAB_OK else None
BLUE      = _hex_to_rl_color("#4a9eff") if REPORTLAB_OK else None
GREEN     = _hex_to_rl_color("#4ae090") if REPORTLAB_OK else None
LIGHT_ROW = _hex_to_rl_color("#252840") if REPORTLAB_OK else None


def generate_pdf(
    drug_name:   str,
    opp_df:      pd.DataFrame,
    annotations: Dict,
    weights:     Dict,
    network_png: Optional[bytes] = None,
) -> bytes:
    """
    Build and return a multi-page PDF as bytes.
    Falls back to a minimal text PDF if reportlab is missing.
    """
    if not REPORTLAB_OK:
        # Minimal fallback
        lines = [
            f"RepurposeIQ Report – {drug_name}",
            f"Date: {date.today()}",
            "",
            "Top Repurposing Opportunities:",
        ]
        if opp_df is not None and not opp_df.empty:
            for _, row in opp_df.head(20).iterrows():
                lines.append(f"  {row['disease']}  Score: {row['composite_score']}")
        return "\n".join(lines).encode()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("Title", parent=styles["Title"],
                                  fontSize=24, textColor=colors.HexColor("#4a9eff"),
                                  spaceAfter=12, alignment=TA_CENTER)
    h2_style    = ParagraphStyle("H2", parent=styles["Heading2"],
                                  fontSize=14, textColor=colors.HexColor("#4ae090"),
                                  spaceBefore=16, spaceAfter=6)
    body_style  = ParagraphStyle("Body", parent=styles["Normal"],
                                  fontSize=9, textColor=colors.HexColor("#c8d0e8"),
                                  leading=13)
    caption_style = ParagraphStyle("Caption", parent=styles["Normal"],
                                    fontSize=8, textColor=colors.HexColor("#8892a4"),
                                    leading=11, alignment=TA_CENTER)

    story = []

    # ── Cover page ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1.5*inch))
    story.append(Paragraph("RepurposeIQ", title_style))
    story.append(Paragraph("Drug Repurposing Intelligence Report", h2_style))
    story.append(Spacer(1, 0.3*inch))

    cover_data = [
        ["Drug analysed:", drug_name],
        ["Report date:",   str(date.today())],
        ["Scoring weights:",
         f"Target overlap {weights.get('to',0.35)*100:.0f}%  ·  "
         f"Network proximity {weights.get('np',0.40)*100:.0f}%  ·  "
         f"Expression reversal {weights.get('er',0.25)*100:.0f}%"],
    ]
    cover_table = Table(cover_data, colWidths=[2*inch, 4.5*inch])
    cover_table.setStyle(TableStyle([
        ("FONTNAME",    (0,0),(-1,-1), "Helvetica"),
        ("FONTSIZE",    (0,0),(-1,-1), 10),
        ("FONTNAME",    (0,0),(0,-1),  "Helvetica-Bold"),
        ("TEXTCOLOR",   (0,0),(-1,-1), colors.HexColor("#c8d0e8")),
        ("TEXTCOLOR",   (1,0),(1,-1),  colors.HexColor("#ffffff")),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.HexColor("#1e2233"),colors.HexColor("#252840")]),
        ("TOPPADDING",  (0,0),(-1,-1), 6),
        ("BOTTOMPADDING",(0,0),(-1,-1),6),
        ("LEFTPADDING", (0,0),(-1,-1), 8),
    ]))
    story.append(cover_table)
    story.append(PageBreak())

    # ── Summary stats ─────────────────────────────────────────────────────────
    story.append(Paragraph("Summary Statistics", h2_style))
    if opp_df is not None and not opp_df.empty:
        high_conf = int((opp_df["composite_score"] > 0.70).sum())
        top_row   = opp_df.iloc[0]
        avg_ai    = opp_df["ai_plausibility"].mean()
        stats_data = [
            ["Metric", "Value"],
            ["Total pairs analysed",    str(len(opp_df))],
            ["High-confidence pairs",   str(high_conf)],
            ["Top composite score",     f"{top_row['composite_score']:.2f} ({top_row['disease']})"],
            ["Avg. AI plausibility",    f"{avg_ai:.1f} / 10"],
            ["Annotated pairs",         str(len(annotations))],
        ]
        stats_table = Table(stats_data, colWidths=[3.5*inch, 3*inch])
        stats_table.setStyle(TableStyle([
            ("BACKGROUND",  (0,0),(-1,0),  colors.HexColor("#4a9eff")),
            ("TEXTCOLOR",   (0,0),(-1,0),  colors.white),
            ("FONTNAME",    (0,0),(-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",    (0,0),(-1,-1), 9),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#1e2233"),colors.HexColor("#252840")]),
            ("TEXTCOLOR",   (0,1),(-1,-1), colors.HexColor("#c8d0e8")),
            ("TOPPADDING",  (0,0),(-1,-1), 5),
            ("BOTTOMPADDING",(0,0),(-1,-1),5),
            ("LEFTPADDING", (0,0),(-1,-1), 8),
            ("GRID",        (0,0),(-1,-1), 0.3, colors.HexColor("#2a2d45")),
        ]))
        story.append(stats_table)
    story.append(Spacer(1, 0.2*inch))

    # ── Network image (if provided) ───────────────────────────────────────────
    if network_png:
        story.append(Paragraph("Protein Interaction Network", h2_style))
        img_buf = io.BytesIO(network_png)
        img = Image(img_buf, width=6*inch, height=3*inch)
        story.append(img)
        story.append(Paragraph("Protein interaction network for top-ranked drug-disease pair.", caption_style))
        story.append(Spacer(1, 0.2*inch))

    story.append(PageBreak())

    # ── Opportunities table (top 20) ─────────────────────────────────────────
    story.append(Paragraph(f"Top Repurposing Opportunities — {drug_name}", h2_style))
    if opp_df is not None and not opp_df.empty:
        top20 = opp_df.head(20)
        tbl_header = ["#","Disease","Category","Score","TO","NP","ER","AI","Annotation"]
        tbl_data = [tbl_header]
        for idx, row in top20.iterrows():
            ann_key = f"{drug_name}_{row['disease']}"
            ann = annotations.get(ann_key, {})
            ann_tag = ann.get("status","") or ""
            tbl_data.append([
                str(idx),
                row["disease"][:28],
                row["category"][:14],
                f"{row['composite_score']:.2f}",
                f"{row['target_overlap']:.2f}",
                f"{row['network_proximity']:.2f}",
                f"{row['expression_reversal']:.2f}",
                f"{row['ai_plausibility']:.1f}",
                ann_tag[:16],
            ])
        col_w = [0.3,2.0,1.1,0.6,0.5,0.5,0.5,0.5,1.0]
        col_w = [w*inch for w in col_w]
        opp_table = Table(tbl_data, colWidths=col_w, repeatRows=1)
        opp_table.setStyle(TableStyle([
            ("BACKGROUND",  (0,0),(-1,0),  colors.HexColor("#4a9eff")),
            ("TEXTCOLOR",   (0,0),(-1,0),  colors.white),
            ("FONTNAME",    (0,0),(-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",    (0,0),(-1,-1), 7),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#1e2233"),colors.HexColor("#252840")]),
            ("TEXTCOLOR",   (0,1),(-1,-1), colors.HexColor("#c8d0e8")),
            ("TOPPADDING",  (0,0),(-1,-1), 4),
            ("BOTTOMPADDING",(0,0),(-1,-1),4),
            ("LEFTPADDING", (0,0),(-1,-1), 4),
            ("GRID",        (0,0),(-1,-1), 0.3, colors.HexColor("#2a2d45")),
        ]))
        story.append(opp_table)

    # ── Annotations ──────────────────────────────────────────────────────────
    if annotations:
        story.append(PageBreak())
        story.append(Paragraph("Researcher Annotations", h2_style))
        ann_data = [["Pair","Status","Note"]]
        for key, ann in annotations.items():
            ann_data.append([key.replace("_"," → "), ann.get("status",""), ann.get("note","")[:60]])
        ann_table = Table(ann_data, colWidths=[2.5*inch, 1.2*inch, 2.8*inch])
        ann_table.setStyle(TableStyle([
            ("BACKGROUND",  (0,0),(-1,0),  colors.HexColor("#4ae090")),
            ("TEXTCOLOR",   (0,0),(-1,0),  colors.HexColor("#0f1117")),
            ("FONTNAME",    (0,0),(-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",    (0,0),(-1,-1), 8),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#1e2233"),colors.HexColor("#252840")]),
            ("TEXTCOLOR",   (0,1),(-1,-1), colors.HexColor("#c8d0e8")),
            ("TOPPADDING",  (0,0),(-1,-1), 4),
            ("BOTTOMPADDING",(0,0),(-1,-1),4),
            ("LEFTPADDING", (0,0),(-1,-1), 6),
            ("GRID",        (0,0),(-1,-1), 0.3, colors.HexColor("#2a2d45")),
        ]))
        story.append(ann_table)

    doc.build(story)
    buf.seek(0)
    return buf.read()
