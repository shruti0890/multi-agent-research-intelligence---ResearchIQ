import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import datetime
from google import genai
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

from schemas import ProjectReportState  # type: ignore

load_dotenv()

_gemini_client = None
def _get_client():
    global _gemini_client
    if _gemini_client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment.")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client

GEMINI_MODEL = "gemini-2.5-flash"


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
        'SubHeading', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=11, leading=14,
        textColor=colors.HexColor('#0F766E'),
        spaceBefore=10, spaceAfter=4, keepWithNext=True
    )
    body_style = ParagraphStyle(
        'Body', parent=styles['Normal'],
        fontName='Helvetica', fontSize=8.5, leading=12,
        textColor=colors.HexColor('#334155'), spaceAfter=5
    )
    body_bold = ParagraphStyle(
        'BodyBold', parent=body_style,
        fontName='Helvetica-Bold'
    )
    bullet_style = ParagraphStyle(
        'Bullet', parent=styles['Normal'],
        fontName='Helvetica', fontSize=8.5, leading=12,
        textColor=colors.HexColor('#334155'),
        leftIndent=12, spaceAfter=3,
        bulletIndent=4
    )
    disclaimer_style = ParagraphStyle(
        'Disclaimer', parent=styles['Normal'],
        fontName='Helvetica-Oblique', fontSize=7.5, leading=10,
        textColor=colors.HexColor('#64748B')
    )
    tag_style = ParagraphStyle(
        'Tag', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=7.5, leading=9,
        textColor=colors.HexColor('#0F766E')
    )
    return {
        'title': title_style, 'subtitle': subtitle_style, 'meta': meta_style,
        'h1': h1_style, 'h2': h2_style, 'body': body_style, 'body_bold': body_bold,
        'bullet': bullet_style, 'disclaimer': disclaimer_style, 'tag': tag_style
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
    
    prompt = f"""
    You are a Lead Research Director compiling a final Executive Summary and Strategic Recommendations.
    Review the following structured inputs:
    
    Research Papers:
    {json.dumps([p.model_dump() for p in state.research.papers], indent=2)}
    
    Active Patents:
    {json.dumps([pat.model_dump() for pat in state.patents.patents], indent=2)}
    
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
        client = _get_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        data = json.loads(response.text.strip())
        summary_narrative = data.get("summary", summary_narrative)
        recs_narrative = "\n".join([f"• {r}" for r in data.get("recommendations", [])])
    except Exception as e:
        print(f"Error calling Gemini in compiler: {e}")
        
    # Generate Matplotlib charts
    chart_paths = _generate_charts(state, output_dir)
    
    # Setup document
    filename = f"report_{state.topic.lower().replace(' ', '_')}.pdf"
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
    story.append(Paragraph(f"Topic Focus: {state.topic}", S['subtitle']))
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
    story.append(Paragraph(f"<b>Date:</b> {today_str}", S['meta']))
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
    
    exec_box = Table([[Paragraph(summary_narrative, S['body'])]], colWidths=[522])
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
            Paragraph("<br/>".join([f"{idx+1}. {p.title[:28]}... ({int(p.relevance_score*100)}%)" for idx, p in enumerate(top_papers)]), S['body']),
            Paragraph("<br/>".join([f"{idx+1}. {p.gap_description[:28]}... ({p.gap_severity})" for idx, p in enumerate(top_gaps)]), S['body']),
            Paragraph("<br/>".join([f"{idx+1}. {pat.patent_id} ({pat.relevance_score}%)" for idx, pat in enumerate(top_patents)]), S['body'])
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
            story.append(Paragraph(rec.strip(), S['bullet']))
            
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
            Paragraph(f"<font color='#0F766E'>{short_title}</font>", S['body']),
            Paragraph(short_method, S['body']),
            Paragraph(short_datasets, S['body']),
            Paragraph(perf, S['body']),
            Paragraph(f"<b>{p.innovation_score}/100</b>", S['body']),
            Paragraph(f"<b>{int(p.relevance_score * 100)}%</b>", S['body']),
            Paragraph(f"<font color='{sev_color}'><b>{p.gap_severity}</b></font>", S['body'])
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
            f"<b><font size=10 color='#0F172A'>Paper {idx}: {p.title}</font></b><br/>"
            f"<font size=7.5 color='#64748B'>Authors: {', '.join(p.authors[:2])} ({p.year}) | Relevance: {int(p.relevance_score*100)}% | Innovation: {p.innovation_score}/100</font>"
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
        story.append(Spacer(1, 4))
        
        detail_rows = [
            ["🔴 Problem Statement", p.problem_statement],
            ["🟢 Proposed Solution", p.proposed_solution],
            ["⚙️ Methodology & Models", p.methodology],
            ["📊 Results & Metrics", p.results],
            ["⚠️ Technical Challenges", p.challenges],
            ["🔭 Future Outcomes", p.future_outcomes],
            ["💡 Research Significance", p.research_significance],
            ["🚨 Specific Research Gap", f"<b>[{p.gap_severity} severity]</b>: {p.gap_description}"],
            ["🔗 Gap Impact & Opportunities", f"<b>Impact:</b> {p.gap_impact}<br/><b>Opportunity:</b> {p.gap_opportunity}<br/><b>Future Scope:</b> {p.gap_future_scope}"]
        ]
        
        detail_data = []
        for label, content in detail_rows:
            detail_data.append([
                Paragraph(f"<b>{label}</b>", S['body_bold']),
                Paragraph(content if content and content != "Not extracted" else "Not available in abstract.", S['body']),
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
            Paragraph(f"<b>{pat.patent_id}</b><br/><font size=7 color='#64748B'>{pat.title[:20]}...</font>", S['body']),
            Paragraph(pat.assignee, S['body']),
            Paragraph(f"<font color='{fto_color}'><b>{pat.fto_rating}</b></font>", S['body']),
            Paragraph(f"<b>{pat.relevance_score}/100</b>", S['body']),
            Paragraph(f"<b>#{pat.rank}</b>", S['body']),
            Paragraph(f"<b>Match:</b> {pat.match_explanation}<br/><b>Strategy:</b> {pat.design_around_strategy}", S['body'])
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
        story.append(Paragraph(f"💡 {opp}", S['bullet']))
        
    # Build document
    doc.build(story)
    
    return os.path.abspath(file_path)
