import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

REPORTS_DIR = os.path.join('static', 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

def generate_pdf_report(report_data):
    """
    Generates a PDF report containing patient details, MRI prediction results,
    Grad-CAM visualizations, AI explanations, and doctor notes.
    """
    report_id = report_data.get('report_id', 'N/A')
    pdf_filename = f"brain_scan_report_{report_id}.pdf"
    pdf_path = os.path.join(REPORTS_DIR, pdf_filename)
    
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        textColor=colors.HexColor('#312E81'),
        spaceAfter=10
    )
    
    section_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=colors.HexColor('#4338CA'),
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#1E293B'),
        leading=14
    )
    
    disclaimer_style = ParagraphStyle(
        'DisclaimerText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        textColor=colors.HexColor('#64748B'),
        leading=11
    )

    story = []

    # Title & Header
    story.append(Paragraph("🧠 BrainScan-AI Diagnostic Report", title_style))
    story.append(Paragraph(f"<b>Report ID:</b> #{report_id} | <b>Date:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", body_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#E2E8F0'), spaceAfter=15))

    # Patient & Model Overview Table
    patient_info = [
        [Paragraph("<b>Patient Name:</b>", body_style), Paragraph(str(report_data.get('patient_name', 'N/A')), body_style),
         Paragraph("<b>Selected Architecture:</b>", body_style), Paragraph(str(report_data.get('model_name', 'N/A')), body_style)],
        [Paragraph("<b>Predicted Category:</b>", body_style), Paragraph(f"<font color='#4338CA'><b>{report_data.get('prediction', 'N/A')}</b></font>", body_style),
         Paragraph("<b>Confidence Level:</b>", body_style), Paragraph(f"<b>{report_data.get('confidence', 0):.2f}%</b>", body_style)]
    ]
    
    info_table = Table(patient_info, colWidths=[120, 150, 130, 130])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F8FAFC')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#E2E8F0')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 15))

    # Class Probabilities Section
    story.append(Paragraph("Class Probability Breakdown", section_style))
    prob_data = [["Category", "Probability Confidence"]]
    probabilities = report_data.get('probabilities', {})
    for cat, prob in probabilities.items():
        prob_data.append([cat, f"{prob:.2f}%"])
        
    prob_table = Table(prob_data, colWidths=[250, 280])
    prob_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#EEF2FF')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#312E81')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#CBD5E1')),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 15))

    # Image & Grad-CAM Visualizations Side-by-Side
    story.append(Paragraph("Radiological & Grad-CAM Heatmap Visualizations", section_style))
    img_cells = []
    
    orig_img_path = report_data.get('image_path')
    gradcam_img_path = report_data.get('gradcam_path')

    if orig_img_path and os.path.exists(orig_img_path):
        img_cells.append(RLImage(orig_img_path, width=180, height=180))
    else:
        img_cells.append(Paragraph("Original Scan Image Not Available", body_style))

    if gradcam_img_path and os.path.exists(gradcam_img_path):
        img_cells.append(RLImage(gradcam_img_path, width=180, height=180))
    else:
        img_cells.append(Paragraph("Grad-CAM Image Not Available", body_style))

    viz_table = Table([[img_cells[0], img_cells[1]]], colWidths=[265, 265])
    viz_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(viz_table)
    story.append(Spacer(1, 15))

    # AI Assistant Summary
    if report_data.get('ai_explanation'):
        story.append(Paragraph("AI Diagnostic Summary", section_style))
        explanation_clean = report_data['ai_explanation'].replace('\n', '<br/>')
        story.append(Paragraph(explanation_clean, body_style))
        story.append(Spacer(1, 15))

    # Doctor Review & Comments
    if report_data.get('doctor_comment'):
        story.append(Paragraph("Specialist Medical Comments", section_style))
        doctor_comment_clean = report_data['doctor_comment'].replace('\n', '<br/>')
        story.append(Paragraph(f"<b>Status:</b> {report_data.get('review_status', 'Reviewed')}<br/><b>Doctor Notes:</b> {doctor_comment_clean}", body_style))
        story.append(Spacer(1, 15))

    # Footer Disclaimer
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#E2E8F0'), spaceBefore=10, spaceAfter=10))
    story.append(Paragraph(
        "<b>Medical Disclaimer:</b> BrainScan-AI is a deep learning computer-assisted diagnostic framework designed to aid clinical review. "
        "It is NOT a replacement for qualified physician evaluation, histological verification, or formal medical diagnosis.",
        disclaimer_style
    ))

    # Build Document
    doc.build(story)
    return os.path.join('static', 'reports', pdf_filename).replace('\\', '/')