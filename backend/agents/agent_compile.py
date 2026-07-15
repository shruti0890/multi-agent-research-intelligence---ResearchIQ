import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import datetime
from google import genai
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

from schemas import ProjectReportState  # type: ignore

# Load environment variables
load_dotenv()

# Configure Gemini using the new google.genai SDK
_gemini_client = None
def _get_client():
    global _gemini_client
    if _gemini_client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment.")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client

GEMINI_MODEL = "gemini-1.5-flash"


def _build_styles():
    """Build and return all custom paragraph styles for the PDF report."""
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CoverTitle', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=26, leading=32,
        textColor=colors.HexColor('#1A365D'), spaceAfter=15
    )
    subtitle_style = ParagraphStyle(
        'CoverSubtitle', parent=styles['Normal'],
        fontName='Helvetica', fontSize=16, leading=20,
        textColor=colors.HexColor('#2B6CB0'), spaceAfter=8
    )
    meta_style = ParagraphStyle(
        'CoverMeta', parent=styles['Normal'],
        fontName='Helvetica', fontSize=11, leading=15,
        textColor=colors.HexColor('#4A5568'), spaceAfter=8
    )
    h1_style = ParagraphStyle(
        'SectionHeading', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=17, leading=22,
        textColor=colors.HexColor('#2B6CB0'),
        spaceBefore=18, spaceAfter=10, keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'SubHeading', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=12, leading=16,
        textColor=colors.HexColor('#2D3748'),
        spaceBefore=10, spaceAfter=6, keepWithNext=True
    )
    body_style = ParagraphStyle(
        'Body', parent=styles['Normal'],
        fontName='Helvetica', fontSize=9, leading=13,
        textColor=colors.HexColor('#2D3748'), spaceAfter=6
    )
    bullet_style = ParagraphStyle(
        'Bullet', parent=styles['Normal'],
        fontName='Helvetica', fontSize=9, leading=13,
        textColor=colors.HexColor('#2D3748'),
        leftIndent=14, spaceAfter=3,
        bulletIndent=4
    )
    disclaimer_style = ParagraphStyle(
        'Disclaimer', parent=styles['Normal'],
        fontName='Helvetica-Oblique', fontSize=8, leading=11,
        textColor=colors.HexColor('#718096')
    )
    tag_style = ParagraphStyle(
        'Tag', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=8, leading=10,
        textColor=colors.HexColor('#2B6CB0')
    )
    return {
        'title': title_style, 'subtitle': subtitle_style, 'meta': meta_style,
        'h1': h1_style, 'h2': h2_style, 'body': body_style,
        'bullet': bullet_style, 'disclaimer': disclaimer_style, 'tag': tag_style
    }


def compile_final_report(state: ProjectReportState, output_dir: str = ".") -> str:
    """
    Synthesizes executive summary and generates a highly-structured PDF report.
    Returns the absolute path to the generated PDF file.
    """
    today_str = datetime.date.today().strftime("%B %d, %Y")

    # --- Gemini: Generate Executive Summary ---
    summary_narrative = (
        f"This report presents a multi-agent synthesis of academic research and patent intelligence "
        f"for the topic: '{state.topic}'. Automated agents have scanned six literature databases, "
        f"identified critical research gaps, and audited active patent claims to provide actionable "
        f"guidance for researchers and product teams."
    )

    prompt = f"""
    You are a Lead Research Director compiling a final executive summary.
    Review the following structured inputs:
    
    Research Papers:
    {json.dumps([p.model_dump() for p in state.research.papers], indent=2)}
    
    Identified Gaps:
    {json.dumps([g.model_dump() for g in state.gaps.gaps], indent=2)}
    
    Active Patents:
    {json.dumps([pat.model_dump() for pat in state.patents.patents], indent=2)}
    
    Write a 2-paragraph cohesive Executive Summary that:
    1. Highlights the main trend in the research papers for the topic '{state.topic}'.
    2. Explains how the proposed method resolves the critical gaps without infringing on the active patents.
    
    Keep it concise, professional, and actionable. Respond ONLY with the text of the Executive Summary.
    """
    
    try:
        client = _get_client()
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        summary_narrative = response.text.strip()
    except Exception as e:
        print(f"Error calling Gemini in Agent 4 compiler: {e}")

    # --- Build PDF ---
    filename = f"report_{state.topic.lower().replace(' ', '_')}.pdf"
    file_path = os.path.join(output_dir, filename)
    
    doc = SimpleDocTemplate(
        file_path, pagesize=letter,
        rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54
    )
    
    S = _build_styles()
    story = []

    # ================================================================
    # PAGE 1: COVER PAGE
    # ================================================================
    story.append(Spacer(1, 80))
    story.append(Paragraph("RESEARCH INTELLIGENCE &amp; PATENT BRIEF", S['title']))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"Topic Focus: {state.topic}", S['subtitle']))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2B6CB0'), spaceAfter=20))
    story.append(Spacer(1, 120))

    # Cover stats summary table
    num_papers = len(state.research.papers)
    num_gaps = len(state.gaps.gaps)
    num_patents = len(state.patents.patents)

    stats_data = [
        [
            Paragraph(f"<b><font size=22 color='#2B6CB0'>{num_papers}</font></b><br/><font size=9 color='#718096'>Papers Analyzed</font>", S['body']),
            Paragraph(f"<b><font size=22 color='#DD6B20'>{num_gaps}</font></b><br/><font size=9 color='#718096'>Research Gaps</font>", S['body']),
            Paragraph(f"<b><font size=22 color='#E53E3E'>{num_patents}</font></b><br/><font size=9 color='#718096'>Patents Audited</font>", S['body']),
        ]
    ]
    stats_table = Table(stats_data, colWidths=[168, 168, 168])
    stats_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#E2E8F0')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F7FAFC')),
        ('TOPPADDING', (0, 0), (-1, -1), 14),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 14),
    ]))
    story.append(stats_table)
    story.append(Spacer(1, 40))

    story.append(Paragraph(f"<b>Generated By:</b> ResearchIQ Multi-Agent Engine", S['meta']))
    story.append(Paragraph(f"<b>Date:</b> {today_str}", S['meta']))
    story.append(Spacer(1, 30))
    story.append(Paragraph(
        "<i>Disclaimer: This report represents an AI-generated synthesis of public academic articles and patent databases. "
        "It is designed solely to assist research orientation and understanding. All source materials must be consulted directly "
        "before citing or implementing findings.</i>",
        S['disclaimer']
    ))
    story.append(PageBreak())

    # ================================================================
    # PAGE 2: EXECUTIVE SUMMARY
    # ================================================================
    story.append(Paragraph("1. Executive Summary", S['h1']))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#CBD5E0'), spaceAfter=10))

    # Wrap summary in a styled callout box
    exec_box = Table([[Paragraph(summary_narrative, S['body'])]], colWidths=[504])
    exec_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#EBF8FF')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#90CDF4')),
        ('TOPPADDING', (0, 0), (-1, -1), 14),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 14),
        ('LEFTPADDING', (0, 0), (-1, -1), 16),
        ('RIGHTPADDING', (0, 0), (-1, -1), 16),
    ]))
    story.append(exec_box)
    story.append(Spacer(1, 20))

    # ================================================================
    # SECTION 2: ACADEMIC RESEARCH LANDSCAPE
    # ================================================================
    story.append(Paragraph("2. Academic Research Landscape", S['h1']))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#CBD5E0'), spaceAfter=10))
    story.append(Paragraph(
        f"The following {num_papers} papers represent the top-ranked relevant publications from multi-platform API searches "
        f"(arXiv, Semantic Scholar, Crossref, OpenAlex, Europe PMC, DOAJ):",
        S['body']
    ))
    story.append(Spacer(1, 8))

    # Rich expanded paper table with abstract snippet + source URL
    table_data = [[
        Paragraph("<b>Paper</b>", S['body']),
        Paragraph("<b>Rank</b>", S['body']),
        Paragraph("<b>NotebookLM Summary &amp; Algorithmic Approach</b>", S['body']),
    ]]

    for paper in state.research.papers:
        short_title = paper.title[:55] + "..." if len(paper.title) > 55 else paper.title
        authors_str = ", ".join(paper.authors[:2]) + (" et al." if len(paper.authors) > 2 else "")

        rank_color = "#38A169" if paper.relevance_rank == "High" else ("#DD6B20" if paper.relevance_rank == "Medium" else "#E53E3E")
        score_pct = int(paper.relevance_score * 100)

        summary_cell = (
            f"<b><font color='#2B6CB0'>{short_title}</font></b><br/>"
            f"<font size=7 color='#718096'>{authors_str} · {paper.year}</font><br/><br/>"
            f"<b>Summary:</b> {paper.notebook_summary[:120]}...<br/>"
            f"<b>Method:</b> <font color='#2D6A4F'>{paper.technical_execution[:100]}...</font><br/>"
            f"<b>Datasets:</b> {', '.join(paper.datasets[:3])}"
        )

        rank_cell = (
            f"<b><font color='{rank_color}'>{paper.relevance_rank}</font></b><br/>"
            f"<font size=7 color='#718096'>Score: {score_pct}%</font>"
        )

        table_data.append([
            Paragraph(summary_cell, S['body']),
            Paragraph(rank_cell, S['body']),
            Paragraph(
                f"<a href='{paper.url}'><font color='#2B6CB0'>🔗 View Paper</font></a>" if paper.url else "–",
                S['body']
            ),
        ])

    t = Table(table_data, colWidths=[270, 60, 174])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#EDF2F7')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F7FAFC')]),
    ]))
    story.append(t)
    story.append(PageBreak())

    # ================================================================
    # SECTION 2b: PER-PAPER DEEP ANALYSIS
    # ================================================================
    story.append(Paragraph("3. Detailed Paper-by-Paper Analysis", S['h1']))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#CBD5E0'), spaceAfter=10))
    story.append(Paragraph(
        "The following section provides a structured deep analysis of each paper, covering "
        "its problem statement, proposed solution, methodology, results, challenges, and future directions.",
        S['body']
    ))
    story.append(Spacer(1, 10))

    for paper_idx, paper in enumerate(state.research.papers, 1):
        rank_color = "#38A169" if paper.relevance_rank == "High" else ("#DD6B20" if paper.relevance_rank == "Medium" else "#E53E3E")
        authors_str = ", ".join(paper.authors[:3]) + (" et al." if len(paper.authors) > 3 else "")

        # Paper header bar
        header_html = (
            f"<b><font size=11 color='#1A365D'>Paper {paper_idx}: {paper.title}</font></b><br/>"
            f"<font size=8 color='#718096'>{authors_str} · {paper.year} · "
            f"<font color='{rank_color}'>{paper.relevance_rank} Relevance</font> · Score: {int(paper.relevance_score * 100)}%</font>"
        )
        header_table = Table([[Paragraph(header_html, S['body'])]], colWidths=[504])
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#EBF8FF')),
            ('BOX', (0, 0), (-1, -1), 1.5, colors.HexColor('#90CDF4')),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ]))
        story.append(header_table)
        story.append(Spacer(1, 4))

        # 6-row structured detail grid
        detail_rows = [
            ["🔴 Problem", paper.problem_statement],
            ["🟢 Solution", paper.proposed_solution],
            ["⚙️ Methodology", paper.methodology],
            ["📊 Results", paper.results],
            ["⚠️ Challenges", paper.challenges],
            ["🔭 Future Work", paper.future_outcomes],
        ]

        detail_table_data = []
        for label, content in detail_rows:
            detail_table_data.append([
                Paragraph(f"<b>{label}</b>", S['body']),
                Paragraph(content if content and content != "Not extracted" else "Not available in abstract.", S['body']),
            ])

        detail_table = Table(detail_table_data, colWidths=[110, 394])
        detail_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#CBD5E0')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F7FAFC')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('ROWBACKGROUNDS', (1, 0), (1, -1), [colors.white, colors.HexColor('#FAFAFA')]),
        ]))
        story.append(detail_table)
        story.append(Spacer(1, 14))

    story.append(PageBreak())


    papers_table = Table(table_data, colWidths=[280, 65, 159])
    papers_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#EBF4FF')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#CBD5E0')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F7FAFC')]),
    ]))
    story.append(papers_table)
    story.append(PageBreak())

    # ================================================================
    # PAGE 3: RESEARCH GAP ANALYSIS
    # ================================================================
    story.append(Paragraph("3. Research Gap Analysis", S['h1']))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#CBD5E0'), spaceAfter=10))
    story.append(Paragraph("Key weaknesses and unresolved limitations identified in current published literature:", S['body']))
    story.append(Spacer(1, 10))

    for idx, gap in enumerate(state.gaps.gaps, 1):
        severity_color = "#E53E3E" if gap.severity == "Critical" else ("#DD6B20" if gap.severity == "Moderate" else "#38A169")
        bg_color = "#FFF5F5" if gap.severity == "Critical" else ("#FFFAF0" if gap.severity == "Moderate" else "#F0FFF4")

        evidence_lines = "".join([f"• {ep}<br/>" for ep in gap.evidence_papers])
        gap_html = (
            f"<b>Gap #{idx} — <font color='{severity_color}'>[{gap.severity}]</font></b><br/><br/>"
            f"<b>Description:</b> {gap.description}<br/><br/>"
            f"<b>Why It Matters:</b> {gap.why_it_matters}<br/><br/>"
            f"<b>Evidence Papers:</b><br/>{evidence_lines}"
        )

        gap_table = Table([[Paragraph(gap_html, S['body'])]], colWidths=[504])
        gap_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(bg_color)),
            ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor('#E2E8F0')),
            ('LINELEFT', (0, 0), (-1, -1), 5, colors.HexColor(severity_color)),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('LEFTPADDING', (0, 0), (-1, -1), 16),
            ('RIGHTPADDING', (0, 0), (-1, -1), 14),
        ]))
        story.append(gap_table)
        story.append(Spacer(1, 10))

    story.append(Spacer(1, 12))
    story.append(Paragraph("4. Proposed Novel Methodology", S['h1']))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#CBD5E0'), spaceAfter=10))
    story.append(Paragraph(f"<b>Title:</b> {state.gaps.proposed_method.title}", S['h2']))
    story.append(Spacer(1, 6))

    # Novelty score bar (visual indicator)
    score = state.gaps.proposed_method.novelty_score
    filled = int(score / 5)  # max 20 blocks
    empty = 20 - filled
    bar_html = (
        f"<b>Novelty Score:</b> "
        f"<font color='#2B6CB0'>{'█' * filled}</font>"
        f"<font color='#CBD5E0'>{'░' * empty}</font>"
        f" <b>{score}/100</b>"
    )
    story.append(Paragraph(bar_html, S['body']))
    story.append(Spacer(1, 8))

    # Methodology steps as structured bullet cards
    steps_html = "<b>Execution Architecture:</b><br/>"
    for step_idx, step in enumerate(state.gaps.proposed_method.approach.split("\n"), 1):
        if step.strip():
            steps_html += f"&nbsp;&nbsp;<b>{step_idx}.</b> {step.strip()}<br/>"

    steps_html += f"<br/><b>Rationale:</b> <i>{state.gaps.proposed_method.rationale}</i>"

    method_table = Table([[Paragraph(steps_html, S['body'])]], colWidths=[504])
    method_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#EBF8FF')),
        ('BOX', (0, 0), (-1, -1), 1.5, colors.HexColor('#60A5FA')),
        ('TOPPADDING', (0, 0), (-1, -1), 14),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 14),
        ('LEFTPADDING', (0, 0), (-1, -1), 16),
        ('RIGHTPADDING', (0, 0), (-1, -1), 14),
    ]))
    story.append(method_table)
    story.append(PageBreak())

    # ================================================================
    # PAGE 4: PATENT LANDSCAPE
    # ================================================================
    story.append(Paragraph("5. Intellectual Property Landscape", S['h1']))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#CBD5E0'), spaceAfter=10))
    story.append(Paragraph(
        "Evaluation of active patents and FTO (Freedom to Operate) checks against the proposed methodology:",
        S['body']
    ))
    story.append(Spacer(1, 8))

    # Expanded patent table with 5 columns
    patent_headers = [[
        Paragraph("<b>Patent ID</b>", S['body']),
        Paragraph("<b>Assignee</b>", S['body']),
        Paragraph("<b>Relevance</b>", S['body']),
        Paragraph("<b>FTO</b>", S['body']),
        Paragraph("<b>Design-Around Strategy</b>", S['body']),
    ]]
    pt_data = patent_headers

    for pat in state.patents.patents:
        fto_color = "#E53E3E" if pat.fto_rating == "Alert" else ("#DD6B20" if pat.fto_rating == "Caution" else "#38A169")
        rel_color = "#E53E3E" if pat.relevance == "Prior Art" else ("#DD6B20" if pat.relevance == "Overlap" else "#38A169")

        title_snippet = pat.title[:35] + "..." if len(pat.title) > 35 else pat.title

        pt_data.append([
            Paragraph(
                f"<b>{pat.patent_id}</b><br/>"
                f"<font size=7 color='#718096'>{title_snippet}</font><br/>"
                f"<a href='{pat.url or ''}'><font size=7 color='#2B6CB0'>View Patent</font></a>",
                S['body']
            ),
            Paragraph(pat.assignee, S['body']),
            Paragraph(f"<b><font color='{rel_color}'>{pat.relevance}</font></b>", S['body']),
            Paragraph(f"<b><font color='{fto_color}'>{pat.fto_rating}</font></b>", S['body']),
            Paragraph(pat.design_around_strategy, S['body']),
        ])

    pt = Table(pt_data, colWidths=[95, 80, 65, 50, 214])
    pt.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#EBF4FF')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#CBD5E0')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F7FAFC')]),
    ]))
    story.append(pt)
    story.append(Spacer(1, 20))

    # White Space Opportunities as a structured list
    story.append(Paragraph("6. Unpatented White Space Opportunities", S['h1']))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#CBD5E0'), spaceAfter=10))
    story.append(Paragraph(
        "The following areas have been identified as commercially viable innovation opportunities "
        "with no active patent coverage:",
        S['body']
    ))
    story.append(Spacer(1, 8))

    for opp in state.patents.white_space_opportunities:
        opp_row = Table(
            [[Paragraph(f"💡 {opp}", S['body'])]],
            colWidths=[504]
        )
        opp_row.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F0FFF4')),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#9AE6B4')),
            ('LINELEFT', (0, 0), (-1, -1), 4, colors.HexColor('#38A169')),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 14),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ]))
        story.append(opp_row)
        story.append(Spacer(1, 6))

    # Build Document
    doc.build(story)
    
    return os.path.abspath(file_path)
