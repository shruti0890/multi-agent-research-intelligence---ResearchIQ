import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import datetime
import html
from google.genai import types
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from dotenv import load_dotenv

import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt

from schemas import ProjectReportState, GeminiQuotaExhaustedError, GeminiError
from agents.agent_utils import execute_gemini_with_retry, get_gemini_model, get_gemini_client  # type: ignore

load_dotenv(override=True)

GEMINI_MODEL = get_gemini_model()
_get_client = get_gemini_client

# Quota exhaustion detection (same logic as other agents)
_QUOTA_SIGNALS = [
    "resource_exhausted",
    "generate_content_free_tier",
    "free-tier limit",
    "free_tier",
    "generaterequestsperday",
    "quotavalue",
    "requests_per_day",
    "per_day",
    "daily quota",
    "daily limit",
]


def _is_quota_exhausted(error_msg: str) -> bool:
    msg = error_msg.lower()
    return any(sig in msg for sig in _QUOTA_SIGNALS)


def _gemini_generate_with_retry(
    client,
    model: str,
    prompt: str,
    config,
    max_retries: int = 3,
    agent_label: str = "Agent4",
) -> str:
    """Thin wrapper around centralized execute_gemini_with_retry for backward compatibility."""
    return execute_gemini_with_retry(
        prompt=prompt,
        config=config,
        model=model,
        max_retries=max_retries,
        agent_label=agent_label,
        client=client,
    )





def _esc(val) -> str:
    """Helper to safely escape dynamic text for ReportLab Paragraph markup."""
    if val is None:
        return ""
    return html.escape(str(val))


def _build_styles():
    """Build and return custom typography styles matching the McKinsey design system."""
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CoverTitle', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=24, leading=30,
        textColor=colors.HexColor('#0F172A'), spaceAfter=15
    )
    subtitle_style = ParagraphStyle(
        'CoverSubtitle', parent=styles['Normal'],
        fontName='Helvetica', fontSize=14, leading=18,
        textColor=colors.HexColor('#0F766E'), spaceAfter=8
    )
    meta_style = ParagraphStyle(
        'CoverMeta', parent=styles['Normal'],
        fontName='Helvetica', fontSize=10, leading=14,
        textColor=colors.HexColor('#475569'), spaceAfter=8
    )
    h1_style = ParagraphStyle(
        'SectionHeading', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=15, leading=20,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=18, spaceAfter=8, keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'SubSectionHeading', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=12, leading=16,
        textColor=colors.HexColor('#0F766E'),
        spaceBefore=12, spaceAfter=6, keepWithNext=True
    )
    body_style = ParagraphStyle(
        'BodyDark', parent=styles['Normal'],
        fontName='Helvetica', fontSize=8.5, leading=12,
        textColor=colors.HexColor('#1E293B'), spaceAfter=4
    )
    body_bold = ParagraphStyle(
        'BodyDarkBold', parent=body_style,
        fontName='Helvetica-Bold'
    )
    callout_style = ParagraphStyle(
        'CalloutText', parent=styles['Normal'],
        fontName='Helvetica', fontSize=8.5, leading=12,
        textColor=colors.HexColor('#0F172A')
    )
    bullet_style = ParagraphStyle(
        'CustomBullet', parent=body_style,
        leftIndent=15, firstLineIndent=-10, spaceAfter=4
    )
    footer_style = ParagraphStyle(
        'FooterText', parent=styles['Normal'],
        fontName='Helvetica', fontSize=7.5, leading=10,
        textColor=colors.HexColor('#64748B')
    )
    disclaimer_style = ParagraphStyle(
        'DisclaimerText', parent=styles['Normal'],
        fontName='Helvetica-Oblique', fontSize=7.5, leading=10,
        textColor=colors.HexColor('#94A3B8')
    )

    return {
        'title': title_style,
        'subtitle': subtitle_style,
        'meta': meta_style,
        'h1': h1_style,
        'h2': h2_style,
        'body': body_style,
        'body_bold': body_bold,
        'callout': callout_style,
        'bullet': bullet_style,
        'footer': footer_style,
        'disclaimer': disclaimer_style
    }


def _generate_charts(state: ProjectReportState, output_dir: str) -> dict:
    """Generates visual analytics charts and returns their file paths."""
    os.makedirs(output_dir, exist_ok=True)
    paths = {}
    
    # Theme colors
    primary_color = '#0F172A'
    secondary_color = '#0F766E'
    accent_red = '#EF4444'
    accent_orange = '#F59E0B'
    accent_green = '#10B981'
    
    # 1. Paper Relevance & Innovation Comparison Chart
    try:
        papers = state.research.papers[:5] # Top 5 papers
        titles = [p.title[:20] + "..." if len(p.title) > 20 else p.title for p in papers]
        relevance_scores = [int(p.relevance_score * 100) for p in papers]
        innovation_scores = [p.innovation_score for p in papers]
        
        x = range(len(titles))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(5.5, 2.6))
        ax.bar([i - width/2 for i in x], relevance_scores, width, label='Relevance Score', color=primary_color)
        ax.bar([i + width/2 for i in x], innovation_scores, width, label='Innovation Score', color=secondary_color)
        
        ax.set_title('Relevance & Innovation Comparison (Top 5 Papers)', fontsize=9, fontweight='bold', color=primary_color)
        ax.set_xticks(x)
        ax.set_xticklabels(titles, rotation=12, ha='right', fontsize=7.5)
        ax.set_ylim(0, 110)
        ax.legend(fontsize=7)
        plt.tight_layout()
        
        path = os.path.join(output_dir, 'chart_paper_comparison.png')
        plt.savefig(path, dpi=200)
        plt.close()
        paths['paper_comparison'] = path
    except Exception as e:
        print(f"Error generating paper comparison chart: {e}")
        
    # 2. Gap Severity Distribution Donut Chart
    try:
        severities = [p.gap_severity for p in state.research.papers]
        counts = {
            'Critical': severities.count('Critical'),
            'Moderate': severities.count('Moderate'),
            'Low': severities.count('Low')
        }
        labels = [f"{k} ({v})" for k, v in counts.items() if v > 0]
        sizes = [v for v in counts.values() if v > 0]
        color_map = {'Critical': accent_red, 'Moderate': accent_orange, 'Low': accent_green}
        colors_list = [color_map[k] for k in counts.keys() if counts[k] > 0]
        
        fig, ax = plt.subplots(figsize=(4, 2.2))
        ax.pie(sizes, labels=labels, colors=colors_list, autopct='%1.0f%%', startangle=90, 
               textprops={'fontsize': 7.5}, wedgeprops=dict(width=0.4, edgecolor='w'))
        ax.set_title('Research Gap Severity Distribution', fontsize=9, fontweight='bold', color=primary_color)
        plt.tight_layout()
        
        path = os.path.join(output_dir, 'chart_gap_distribution.png')
        plt.savefig(path, dpi=200)
        plt.close()
        paths['gap_distribution'] = path
    except Exception as e:
        print(f"Error generating gap distribution chart: {e}")
        
    # 3. Patent Relevance Scores Chart
    try:
        patents = state.patents.patents[:5] # Top 5 patents
        pids = [pat.patent_id for pat in patents]
        scores = [pat.relevance_score for pat in patents]
        
        fig, ax = plt.subplots(figsize=(5.5, 2.4))
        bars = ax.barh(pids, scores, color=secondary_color, height=0.45)
        ax.set_title('Patent Relevance Score & Ranking (Top 5)', fontsize=9, fontweight='bold', color=primary_color)
        ax.set_xlabel('Relevance Score (0-100)', fontsize=7.5)
        ax.set_xlim(0, 110)
        ax.tick_params(axis='both', which='major', labelsize=7.5)
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 2, bar.get_y() + bar.get_height()/2, f'{int(width)}', 
                    va='center', ha='left', fontsize=7.5, color=primary_color, fontweight='bold')
        plt.tight_layout()
        
        path = os.path.join(output_dir, 'chart_patent_scores.png')
        plt.savefig(path, dpi=200)
        plt.close()
        paths['patent_scores'] = path
    except Exception as e:
        print(f"Error generating patent scores chart: {e}")
        
    return paths


def compile_final_report(state: ProjectReportState, output_dir: str = ".") -> str:
    """
    Generates a professional McKinsey/BCG Research Intelligence Report.
    Returns the absolute path to the generated PDF file.
    """
    today_str = datetime.date.today().strftime("%B %d, %Y")
    
    # ── Step 1: Query Gemini for Executive Summary & Strategic Recs ──────
    summary_narrative = (
        f"This report presents a multi-agent synthesis of academic research and patent intelligence "
        f"for the topic: '{state.topic}'."
    )
    recs_narrative = "• Align research objectives with unpatented white spaces.\n• Focus on solving scalability constraints identified in current methods."
    lite_papers = []
    for p in state.research.papers:
        lite_papers.append({
            "title": p.title,
            "year": p.year,
            "relevance_rank": p.relevance_rank,
            "gap_description": p.gap_description,
            "gap_severity": p.gap_severity
        })
        
    lite_patents = []
    for pat in state.patents.patents:
        lite_patents.append({
            "patent_id": pat.patent_id,
            "title": pat.title,
            "relevance": pat.relevance,
            "fto_rating": pat.fto_rating
        })

    prompt = f"""
    You are a Lead Research Director compiling a final Executive Summary and Strategic Recommendations.
    Review the following structured inputs:
    
    Research Papers Summary:
    {json.dumps(lite_papers, indent=2)}
    
    Active Patents Summary:
    {json.dumps(lite_patents, indent=2)}
    
    Write a JSON response with exactly two keys:
    1. "summary": A cohesive 2-paragraph Executive Summary summarizing trends in the papers for '{state.topic}' and how the proposed method solves the gaps.
    2. "recommendations": A list of 4 highly strategic, actionable R&D/IP recommendations.
    
    Respond ONLY with a valid JSON block matching this structure:
    {{
      "summary": "...",
      "recommendations": ["...", "...", "...", "..."]
    }}
    """
    
    try:
        response_text = _gemini_generate_with_retry(
            client=_get_client(),
            model=GEMINI_MODEL,
            prompt=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
            max_retries=3,
            agent_label="Agent4",
        )
        data = json.loads(response_text.strip())
        summary_narrative = data.get("summary", summary_narrative)
        recs_narrative = "\n".join([f"• {r}" for r in data.get("recommendations", [])])
        compiler_gemini_ok = True
        state.compiler_status = "success"
    except Exception as e:
        err_str = str(e)
        if _is_quota_exhausted(err_str.lower()):
            print(f"[Gemini] Agent4 QUOTA EXHAUSTED: {err_str[:160]}. Using fallback narrative.")
            state.gemini_quota_exhausted = True
        else:
            print(f"[Gemini] Agent4 error: {err_str[:160]}. Using fallback narrative.")
        compiler_gemini_ok = False
        state.compiler_status = "fallback_due_to_gemini"
        
    # Generate Matplotlib charts
    chart_paths = _generate_charts(state, output_dir)

    # Determine patent analysis status for the report
    patent_status = getattr(state.patents, "patent_analysis_status", "success")
    patent_status_display = {
        "success": "✅ Gemini patent synthesis: completed",
        "retrieval_success_synthesis_failed": "⚠️ Gemini patent synthesis unavailable — raw patent data preserved",
        "retrieval_failed": "❌ Patent retrieval failed",
    }.get(patent_status, patent_status)

    
    # Setup document
    clean_topic_slug = "".join(c if c.isalnum() else "_" for c in state.topic.lower()).strip("_")
    filename = f"report_{clean_topic_slug}.pdf"
    file_path = os.path.join(output_dir, filename)
    doc = SimpleDocTemplate(
        file_path, pagesize=letter,
        rightMargin=45, leftMargin=45, topMargin=45, bottomMargin=45
    )
    
    S = _build_styles()
    story = []
    
    # ================================================================
    # PAGE 1: COVER PAGE
    # ================================================================
    story.append(Spacer(1, 40))
    story.append(Paragraph("RESEARCH INTELLIGENCE &amp; PATENT BRIEF", S['title']))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"Topic Focus: {_esc(state.topic)}", S['subtitle']))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0F766E'), spaceAfter=15))
    story.append(Spacer(1, 100))
    
    num_papers = len(state.research.papers)
    num_gaps = len(state.research.papers)
    num_patents = len(state.patents.patents)
    
    stats_data = [
        [
            Paragraph(f"<b><font size=20 color='#0F172A'>{num_papers}</font></b><br/><font size=8.5 color='#64748B'>Papers Analyzed</font>", S['body']),
            Paragraph(f"<b><font size=20 color='#0F766E'>{num_gaps}</font></b><br/><font size=8.5 color='#64748B'>Research Gaps</font>", S['body']),
            Paragraph(f"<b><font size=20 color='#EF4444'>{num_patents}</font></b><br/><font size=8.5 color='#64748B'>Patents Audited</font>", S['body']),
        ]
    ]
    stats_table = Table(stats_data, colWidths=[174, 174, 174])
    stats_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor('#E2E8F0')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('TOPPADDING', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
    ]))
    story.append(stats_table)
    story.append(Spacer(1, 120))
    
    story.append(Paragraph(f"<b>Generated By:</b> ResearchIQ Multi-Agent Engine", S['meta']))
    story.append(Paragraph(f"<b>Date:</b> {_esc(today_str)}", S['meta']))
    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "<i>Disclaimer: This report represents an AI-generated synthesis of public academic articles and patent databases. "
        "It is designed solely to assist research orientation and understanding. All source materials must be consulted directly "
        "before citing or implementing findings.</i>",
        S['disclaimer']
    ))
    story.append(PageBreak())
    
    # ================================================================
    # PAGE 2: EXECUTIVE SUMMARY & STRATEGIC RECOMMENDATIONS
    # ================================================================
    story.append(Paragraph("1. Executive Summary", S['h1']))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#E2E8F0'), spaceAfter=8))

    # ── Gemini-unavailable banner ────────────────────────────────────────────
    if not compiler_gemini_ok:
        banner_text = (
            "<b>⚠️ Gemini AI Unavailable — Fallback Mode</b><br/>"
            "The Gemini daily API quota was exhausted during this run. "
            "The Executive Summary and Strategic Recommendations below were generated using "
            "a deterministic fallback (based on paper abstracts and gap descriptions), "
            "NOT by Gemini. Patent data was preserved from live database retrieval."
        )
        banner_para = Paragraph(_esc(banner_text), S['body'])
        banner_table = Table([[banner_para]], colWidths=[522])
        banner_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#FFFBEB')),
            ('BOX', (0, 0), (-1, -1), 1.5, colors.HexColor('#D97706')),
            ('LINELEFT', (0, 0), (-1, -1), 5, colors.HexColor('#D97706')),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('LEFTPADDING', (0, 0), (-1, -1), 14),
            ('RIGHTPADDING', (0, 0), (-1, -1), 14),
        ]))
        story.append(banner_table)
        story.append(Spacer(1, 8))
    # ── Patent synthesis status line ─────────────────────────────────────────
    story.append(Paragraph(_esc(patent_status_display), S['body']))
    story.append(Spacer(1, 8))

    exec_box = Table([[Paragraph(_esc(summary_narrative), S['body'])]], colWidths=[522])
    exec_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#E2E8F0')),
        ('LINELEFT', (0, 0), (-1, -1), 4, colors.HexColor('#0F766E')),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('LEFTPADDING', (0, 0), (-1, -1), 14),
        ('RIGHTPADDING', (0, 0), (-1, -1), 14),
    ]))
    story.append(exec_box)
    story.append(Spacer(1, 12))
    
    # Executive summary metrics deck
    top_papers = state.research.papers[:3]
    top_gaps = [p for p in state.research.papers if p.gap_severity == 'Critical'][:3]
    if not top_gaps:
        top_gaps = state.research.papers[:3]
    top_patents = state.patents.patents[:3]
    
    deck_data = [
        [
            Paragraph("<b>🏆 Top 3 Relevant Papers</b>", S['body_bold']),
            Paragraph("<b>🔴 Top Opportunity Gaps</b>", S['body_bold']),
            Paragraph("<b>🛡️ Top Patent Matches</b>", S['body_bold'])
        ],
        [
            Paragraph("<br/>".join([f"{idx+1}. {_esc(p.title[:28])}... ({int(p.relevance_score*100)}%)" for idx, p in enumerate(top_papers)]), S['body']),
            Paragraph("<br/>".join([f"{idx+1}. {_esc(p.gap_description[:28])}... ({_esc(p.gap_severity)})" for idx, p in enumerate(top_gaps)]), S['body']),
            Paragraph("<br/>".join([f"{idx+1}. {_esc(pat.patent_id)} ({pat.relevance_score}%)" for idx, pat in enumerate(top_patents)]), S['body'])
        ]
    ]
    deck_table = Table(deck_data, colWidths=[174, 174, 174])
    deck_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F8FAFC')),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(deck_table)
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("Strategic Recommendations", S['h2']))
    for rec in recs_narrative.split("\n"):
        if rec.strip():
            story.append(Paragraph(_esc(rec.strip()), S['bullet']))
            
    story.append(PageBreak())
    
    # ================================================================
    # PAGE 3: COMPARATIVE ANALYSIS
    # ================================================================
    story.append(Paragraph("2. Comparative Research Analysis", S['h1']))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#E2E8F0'), spaceAfter=10))
    story.append(Paragraph(
        "Comparative grid mapping the methodologies, performance metrics, and research relevance scores across the analyzed papers:",
        S['body']
    ))
    story.append(Spacer(1, 8))
    
    table_headers = [
        Paragraph("<b>Paper Name</b>", S['body_bold']),
        Paragraph("<b>Methodology</b>", S['body_bold']),
        Paragraph("<b>Datasets</b>", S['body_bold']),
        Paragraph("<b>Performance</b>", S['body_bold']),
        Paragraph("<b>Innov. Score</b>", S['body_bold']),
        Paragraph("<b>Relevance Score</b>", S['body_bold']),
        Paragraph("<b>Gap Severity</b>", S['body_bold'])
    ]
    comp_data = [table_headers]
    for p in state.research.papers:
        short_title = p.title[:35] + "..." if len(p.title) > 35 else p.title
        short_method = p.technical_execution[:30] + "..." if len(p.technical_execution) > 30 else p.technical_execution
        short_datasets = ", ".join(p.datasets[:2])
        short_datasets = short_datasets[:25] + "..." if len(short_datasets) > 25 else short_datasets
        
        # Parse performance snippet from results
        perf = p.results[:30] + "..." if len(p.results) > 30 else p.results
        
        sev_color = '#EF4444' if p.gap_severity == 'Critical' else ('#F59E0B' if p.gap_severity == 'Moderate' else '#10B981')
        
        comp_data.append([
            Paragraph(f"<font color='#0F766E'>{_esc(short_title)}</font>", S['body']),
            Paragraph(_esc(short_method), S['body']),
            Paragraph(_esc(short_datasets), S['body']),
            Paragraph(_esc(perf), S['body']),
            Paragraph(f"<b>{p.innovation_score}/100</b>", S['body']),
            Paragraph(f"<b>{int(p.relevance_score * 100)}%</b>", S['body']),
            Paragraph(f"<font color='{sev_color}'><b>{_esc(p.gap_severity)}</b></font>", S['body'])
        ])
        
    comp_table = Table(comp_data, colWidths=[90, 85, 80, 85, 55, 65, 62])
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F8FAFC')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
    ]))
    story.append(comp_table)
    story.append(PageBreak())
    
    # ================================================================
    # PAGE 4-6: DETAILED PAPER ANALYSIS
    # ================================================================
    story.append(Paragraph("3. Detailed Literature Vetting", S['h1']))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#E2E8F0'), spaceAfter=10))
    
    for idx, p in enumerate(state.research.papers, 1):
        sev_color = '#EF4444' if p.gap_severity == 'Critical' else ('#F59E0B' if p.gap_severity == 'Moderate' else '#10B981')
        
        header_html = (
            f"<b><font size=10 color='#0F172A'>Paper {idx}: {_esc(p.title)}</font></b><br/>"
            f"<font size=7.5 color='#64748B'>Authors: {_esc(', '.join(p.authors[:2]))} ({p.year}) | Relevance: {int(p.relevance_score*100)}% | Innovation: {p.innovation_score}/100</font>"
        )
        
        header_table = Table([[Paragraph(header_html, S['body'])]], colWidths=[522])
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F0FDF4') if p.gap_severity == 'Low' else (colors.HexColor('#FFFBEB') if p.gap_severity == 'Moderate' else colors.HexColor('#FEF2F2'))),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#D1FAE5') if p.gap_severity == 'Low' else (colors.HexColor('#FEF3C7') if p.gap_severity == 'Moderate' else colors.HexColor('#FEE2E2'))),
            ('LINELEFT', (0, 0), (-1, -1), 4, colors.HexColor(sev_color)),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ]))
        story.append(header_table)
        story.append(Spacer(1, 3))

        # ── Coverage type badge ──────────────────────────────────────────────
        # Explicitly distinguishes full-paper fact sheets from abstract-only fallbacks.
        coverage_type = getattr(p, 'coverage_type', 'unavailable')
        full_text_src = getattr(p, 'full_text_source', 'none') or 'none'
        ft_words = getattr(p, 'full_text_word_count', 0) or 0

        if coverage_type == 'full_paper':
            cov_text = (
                f"<font color='#10B981'><b>✅ Coverage: Full paper</b></font>  "
                f"<font color='#64748B' size=7>Source: {_esc(full_text_src)} | "
                f"{ft_words:,} words acquired</font>"
            )
            cov_bg = colors.HexColor('#F0FDF4')
            cov_border = colors.HexColor('#D1FAE5')
        elif coverage_type == 'partial_paper':
            cov_text = (
                f"<font color='#F59E0B'><b>⚠️ Coverage: Partial paper</b></font>  "
                f"<font color='#64748B' size=7>Some full-text content could not be extracted. "
                f"Source: {_esc(full_text_src)} | {ft_words:,} words</font>"
            )
            cov_bg = colors.HexColor('#FFFBEB')
            cov_border = colors.HexColor('#FEF3C7')
        elif coverage_type == 'abstract_only':
            cov_text = (
                f"<font color='#EF4444'><b>⚠️ Coverage: Abstract only</b></font>  "
                f"<font color='#64748B' size=7>Full text was unavailable. "
                f"Gap analysis for this paper is based on abstract evidence only.</font>"
            )
            cov_bg = colors.HexColor('#FEF2F2')
            cov_border = colors.HexColor('#FEE2E2')
        else:
            cov_text = (
                f"<font color='#94A3B8'><b>Coverage: Unavailable</b></font>  "
                f"<font color='#64748B' size=7>No text was available for this paper.</font>"
            )
            cov_bg = colors.HexColor('#F8FAFC')
            cov_border = colors.HexColor('#E2E8F0')

        cov_table = Table([[Paragraph(cov_text, S['body'])]], colWidths=[522])
        cov_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), cov_bg),
            ('BOX', (0, 0), (-1, -1), 0.5, cov_border),
            ('LINELEFT', (0, 0), (-1, -1), 3,
             colors.HexColor('#10B981') if coverage_type == 'full_paper'
             else colors.HexColor('#F59E0B') if coverage_type == 'partial_paper'
             else colors.HexColor('#EF4444') if coverage_type == 'abstract_only'
             else colors.HexColor('#94A3B8')),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ]))
        story.append(cov_table)
        story.append(Spacer(1, 4))

        detail_rows = [
            ["🔴 Problem Statement", p.problem_statement],
            ["🟢 Proposed Solution", p.proposed_solution],
            ["⚙️ Methodology & Models", p.methodology],
            ["📊 Results & Metrics", p.results],
            ["⚠️ Technical Challenges", p.challenges],
            ["🔭 Future Outcomes", p.future_outcomes],
            ["💡 Research Significance", p.research_significance],
            ["🚨 Specific Research Gap", f"<b>[{_esc(p.gap_severity)} severity]</b>: {_esc(p.gap_description)}"],
            ["🔗 Gap Impact & Opportunities", f"<b>Impact:</b> {_esc(p.gap_impact)}<br/><b>Opportunity:</b> {_esc(p.gap_opportunity)}<br/><b>Future Scope:</b> {_esc(p.gap_future_scope)}"]
        ]
        
        detail_data = []
        for label, content in detail_rows:
            display_content = content if content and content != "Not extracted" else "Not available in abstract."
            # Only escape if not already containing bold/break tags formatted above
            if label not in ("🚨 Specific Research Gap", "🔗 Gap Impact & Opportunities"):
                display_content = _esc(display_content)
            detail_data.append([
                Paragraph(f"<b>{_esc(label)}</b>", S['body_bold']),
                Paragraph(display_content, S['body']),
            ])
            
        detail_table = Table(detail_data, colWidths=[120, 402])
        detail_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F8FAFC')),
            ('ROWBACKGROUNDS', (1, 0), (1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
        ]))
        story.append(detail_table)
        story.append(Spacer(1, 10))
        
        # PageBreak every 2 papers to avoid layout overflow
        if idx % 2 == 0 and idx < len(state.research.papers):
            story.append(PageBreak())
            
    story.append(PageBreak())
    
    # ================================================================
    # PAGE 7: VISUAL ANALYTICS
    # ================================================================
    story.append(Paragraph("4. Visual Analytics Suite", S['h1']))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#E2E8F0'), spaceAfter=10))
    story.append(Paragraph(
        "Visualizations summarizing research relevance scores, gap severity ratios, and patent portfolio rankings:",
        S['body']
    ))
    story.append(Spacer(1, 8))
    
    # Incorporate matplotlib images
    if 'paper_comparison' in chart_paths:
        story.append(Paragraph("<b>Figure 1: Research Relevance & Innovation Scores</b>", S['body_bold']))
        story.append(Image(chart_paths['paper_comparison'], width=5.2*inch, height=2.46*inch))
        story.append(Spacer(1, 10))
        
    charts_row_data = []
    if 'gap_distribution' in chart_paths:
        charts_row_data.append(Image(chart_paths['gap_distribution'], width=2.4*inch, height=1.32*inch))
    if 'patent_scores' in chart_paths:
        charts_row_data.append(Image(chart_paths['patent_scores'], width=2.8*inch, height=1.22*inch))
        
    if charts_row_data:
        charts_table = Table([charts_row_data], colWidths=[260, 262])
        charts_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ]))
        story.append(charts_table)
        
    story.append(PageBreak())
    
    # ================================================================
    # PAGE 8: PATENT LANDSCAPE & WHITE SPACE
    # ================================================================
    story.append(Paragraph("5. Intellectual Property Landscape", S['h1']))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#E2E8F0'), spaceAfter=10))
    story.append(Paragraph(
        "Evaluation of active patents and FTO (Freedom to Operate) checks sorted by relevance score:",
        S['body']
    ))
    story.append(Spacer(1, 8))
    
    patent_headers = [
        Paragraph("<b>Patent ID</b>", S['body_bold']),
        Paragraph("<b>Assignee</b>", S['body_bold']),
        Paragraph("<b>FTO Status</b>", S['body_bold']),
        Paragraph("<b>Rel. Score</b>", S['body_bold']),
        Paragraph("<b>Rank</b>", S['body_bold']),
        Paragraph("<b>Match Explanation &amp; Design-Around Strategy</b>", S['body_bold'])
    ]
    pt_data = [patent_headers]
    for pat in state.patents.patents:
        fto_color = '#EF4444' if pat.fto_rating == 'Alert' else ('#F59E0B' if pat.fto_rating == 'Caution' else '#10B981')
        
        pt_data.append([
            Paragraph(f"<b>{_esc(pat.patent_id)}</b><br/><font size=7 color='#64748B'>{_esc(pat.title[:20])}...</font>", S['body']),
            Paragraph(_esc(pat.assignee), S['body']),
            Paragraph(f"<font color='{fto_color}'><b>{_esc(pat.fto_rating)}</b></font>", S['body']),
            Paragraph(f"<b>{pat.relevance_score}/100</b>", S['body']),
            Paragraph(f"<b>#{pat.rank}</b>", S['body']),
            Paragraph(f"<b>Match:</b> {_esc(pat.match_explanation)}<br/><b>Strategy:</b> {_esc(pat.design_around_strategy)}", S['body'])
        ])
        
    pt_table = Table(pt_data, colWidths=[65, 75, 60, 50, 40, 232])
    pt_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F8FAFC')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
    ]))
    story.append(pt_table)
    story.append(Spacer(1, 15))
    
    # White Space opportunities
    story.append(Paragraph("6. Unpatented Innovation Opportunities", S['h2']))
    for opp in state.patents.white_space_opportunities:
        story.append(Paragraph(f"💡 {_esc(opp)}", S['bullet']))

    # ── Compression Statistics Section ────────────────────────────────────
    papers_with_compression = [
        p for p in state.research.papers
        if getattr(p, 'original_tokens', 0) > 0
    ]
    if papers_with_compression:
        story.append(PageBreak())
        story.append(Paragraph("7. Extractive Compression Statistics", S['h1']))
        story.append(Paragraph(
            "ResearchIQ uses a section-aware TextRank extractive compression pipeline. "
            "Every sentence sent to Gemini is verbatim from the original paper. "
            "No RAG, no embeddings, no LLM-generated summaries are used in the compression stage.",
            S['body']
        ))
        story.append(Spacer(1, 8))

        comp_data = [[
            Paragraph("<b>Paper</b>", S['body']),
            Paragraph("<b>Orig. Tokens</b>", S['body']),
            Paragraph("<b>Comp. Tokens</b>", S['body']),
            Paragraph("<b>Reduction</b>", S['body']),
            Paragraph("<b>Section Coverage</b>", S['body']),
        ]]
        for p in papers_with_compression:
            reduction = getattr(p, 'compression_ratio', 0.0) * 100
            coverage = getattr(p, 'section_coverage', 0.0) * 100
            orig_tok = getattr(p, 'original_tokens', 0)
            comp_tok = getattr(p, 'compressed_tokens', 0)

            # Color code reduction
            if reduction >= 60:
                red_color = '#10B981'
            elif reduction >= 30:
                red_color = '#F59E0B'
            else:
                red_color = '#64748B'

            comp_data.append([
                Paragraph(f"{_esc(p.title[:40])}{'...' if len(p.title) > 40 else ''}", S['body']),
                Paragraph(f"{orig_tok:,}", S['body']),
                Paragraph(f"{comp_tok:,}", S['body']),
                Paragraph(f"<font color='{red_color}'><b>{reduction:.1f}%</b></font>", S['body']),
                Paragraph(f"{coverage:.0f}%", S['body']),
            ])

        comp_table = Table(comp_data, colWidths=[220, 70, 70, 70, 92])
        comp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F8FAFC')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
        ]))
        story.append(comp_table)
        story.append(Spacer(1, 8))
        story.append(Paragraph(
            "Note: Faithfulness = 100% means every selected sentence was verified verbatim "
            "in the original paper source. Compression ratio = 1 − (compressed_tokens / original_tokens).",
            S['disclaimer']
        ))

    # Build document
    doc.build(story)

    compiler_status = "success" if compiler_gemini_ok else "fallback_due_to_gemini"
    return os.path.abspath(file_path), compiler_status
