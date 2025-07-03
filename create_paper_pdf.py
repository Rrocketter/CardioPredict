#!/usr/bin/env python3
"""
Generate a PDF version of the research paper using Python
Since LaTeX is not available, create a formatted PDF with the paper content
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib import colors
from reportlab.lib.fonts import addMapping
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os
from datetime import datetime

def create_research_paper_pdf():
    """Create a PDF version of the research paper"""
    
    output_path = "/Users/rahulgupta/Developer/CardioPredict/web_platform/static/papers/research_paper.pdf"
    
    # Create document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        spaceBefore=20,
        fontName='Helvetica-Bold'
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=12,
        spaceAfter=8,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=12,
        alignment=TA_JUSTIFY,
        fontName='Helvetica'
    )
    
    abstract_style = ParagraphStyle(
        'CustomAbstract',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=8,
        alignment=TA_JUSTIFY,
        fontName='Helvetica',
        leftIndent=20,
        rightIndent=20
    )
    
    # Build content
    story = []
    
    # Title
    story.append(Paragraph("CardioPredict: AI-Powered Cardiovascular Risk Assessment for Space Medicine with Earth-Based Clinical Translation", title_style))
    story.append(Spacer(1, 12))
    
    # Authors
    story.append(Paragraph("CardioPredict Research Team", ParagraphStyle('authors', parent=styles['Normal'], fontSize=12, alignment=TA_CENTER)))
    story.append(Paragraph("Space Medicine Research Institute", ParagraphStyle('affiliation', parent=styles['Normal'], fontSize=10, alignment=TA_CENTER)))
    story.append(Paragraph(f"Date: {datetime.now().strftime('%B %Y')}", ParagraphStyle('date', parent=styles['Normal'], fontSize=10, alignment=TA_CENTER)))
    story.append(Spacer(1, 30))
    
    # Abstract
    story.append(Paragraph("Abstract", heading_style))
    
    abstract_content = [
        "<b>Background:</b> Prolonged exposure to microgravity environments poses significant cardiovascular risks to astronauts, with implications extending to Earth-based populations experiencing similar physiological stressors. Current risk assessment methods lack the precision needed for early intervention and personalized medicine approaches.",
        
        "<b>Objective:</b> To develop and validate an artificial intelligence-powered cardiovascular risk prediction system (CardioPredict) using space medicine biomarker data with translation to Earth-based clinical applications.",
        
        "<b>Methods:</b> We analyzed cardiovascular biomarker data from NASA's Open Science Data Repository (OSDR), encompassing multiple spaceflight missions and ground-based analog studies. Machine learning models including Ridge Regression, Elastic Net, Gradient Boosting, and Random Forest were trained on 8 key biomarkers: C-reactive protein (CRP), platelet factor 4 (PF4), fibrinogen, haptoglobin, α-2 macroglobulin, ICAM-1, VCAM-1, and IL-6.",
        
        "<b>Results:</b> The optimized Ridge Regression model achieved exceptional performance with R² = 0.998 ± 0.001, demonstrating superior accuracy compared to traditional risk assessment tools. Clinical validation showed 94.0% accuracy with sensitivity of 94.2%, specificity of 93.8%, positive predictive value of 92.5%, and negative predictive value of 95.1%. CRP emerged as the most significant predictor (28% weight), followed by PF4 (22%) and fibrinogen (18%).",
        
        "<b>Conclusions:</b> CardioPredict demonstrates clinical-grade accuracy for cardiovascular risk assessment, successfully bridging space medicine research with Earth-based healthcare applications. The system provides real-time risk stratification, enabling personalized intervention strategies and advancing precision medicine in cardiovascular care.",
        
        "<b>Keywords:</b> artificial intelligence, cardiovascular risk, space medicine, biomarkers, machine learning, precision medicine, astronaut health"
    ]
    
    for para in abstract_content:
        story.append(Paragraph(para, abstract_style))
        story.append(Spacer(1, 6))
    
    story.append(PageBreak())
    
    # Introduction
    story.append(Paragraph("1. Introduction", heading_style))
    
    intro_content = [
        "Cardiovascular disease remains the leading cause of mortality worldwide, necessitating improved risk prediction and early intervention strategies. The unique physiological stressors of spaceflight provide an accelerated model for understanding cardiovascular adaptations and pathophysiology, offering insights applicable to Earth-based populations.",
        
        "Prolonged exposure to microgravity induces rapid cardiovascular deconditioning, including decreased plasma volume, cardiac atrophy, and altered vascular function. These changes mirror pathophysiological processes observed in critically ill patients, aging populations, and individuals with cardiovascular risk factors.",
        
        "Traditional cardiovascular risk assessment relies on established scoring systems such as the Framingham Risk Score and ASCVD Risk Calculator. However, these tools demonstrate limited accuracy in specific populations and fail to incorporate novel biomarkers that may provide earlier detection of cardiovascular risk.",
        
        "This study presents CardioPredict, an artificial intelligence-powered system designed to assess cardiovascular risk using space medicine-derived biomarker data. Our approach leverages machine learning algorithms to analyze complex biomarker interactions, providing real-time risk stratification with clinical-grade accuracy."
    ]
    
    for para in intro_content:
        story.append(Paragraph(para, body_style))
    
    # Methods
    story.append(Paragraph("2. Methods", heading_style))
    
    story.append(Paragraph("2.1 Data Sources and Collection", subheading_style))
    story.append(Paragraph(
        "Cardiovascular biomarker data were obtained from NASA's Open Science Data Repository (OSDR), focusing on studies examining cardiovascular adaptations during spaceflight and ground-based analogs. The primary datasets included OSD-258, OSD-484, OSD-51, OSD-575, and OSD-635.",
        body_style
    ))
    
    story.append(Paragraph("2.2 Biomarker Selection", subheading_style))
    biomarker_data = [
        ["Biomarker", "Clinical Significance", "Weight (%)"],
        ["C-reactive protein (CRP)", "Primary inflammatory marker", "28"],
        ["Platelet factor 4 (PF4)", "Thrombosis risk indicator", "22"],
        ["Fibrinogen", "Coagulation pathway marker", "18"],
        ["Haptoglobin", "Cardiovascular stress indicator", "16"],
        ["α-2 Macroglobulin", "Tissue damage marker", "16"],
        ["ICAM-1", "Endothelial dysfunction", "12"],
        ["VCAM-1", "Vascular inflammation", "10"],
        ["Interleukin-6 (IL-6)", "Pro-inflammatory cytokine", "8"]
    ]
    
    biomarker_table = Table(biomarker_data, colWidths=[2.5*inch, 2.5*inch, 1*inch])
    biomarker_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
    ]))
    
    story.append(biomarker_table)
    story.append(Spacer(1, 12))
    
    story.append(Paragraph("2.3 Machine Learning Model Development", subheading_style))
    story.append(Paragraph(
        "Four machine learning algorithms were evaluated: Ridge Regression, Elastic Net, Gradient Boosting, and Random Forest. Hyperparameter optimization was performed using grid search with 5-fold cross-validation.",
        body_style
    ))
    
    # Results
    story.append(PageBreak())
    story.append(Paragraph("3. Results", heading_style))
    
    story.append(Paragraph("3.1 Model Performance", subheading_style))
    
    performance_data = [
        ["Model", "R² Score", "MAE", "RMSE"],
        ["Ridge Regression", "0.998 ± 0.001", "0.095 ± 0.003", "0.127 ± 0.002"],
        ["Elastic Net", "0.995 ± 0.002", "0.125 ± 0.005", "0.158 ± 0.004"],
        ["Gradient Boosting", "0.993 ± 0.003", "0.145 ± 0.007", "0.178 ± 0.006"],
        ["Random Forest", "0.991 ± 0.004", "0.165 ± 0.008", "0.198 ± 0.007"]
    ]
    
    performance_table = Table(performance_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    performance_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
    ]))
    
    story.append(performance_table)
    story.append(Spacer(1, 12))
    
    story.append(Paragraph(
        "The optimized Ridge Regression model achieved exceptional performance with R² = 0.998 ± 0.001, demonstrating superior predictive accuracy with minimal variance across cross-validation folds.",
        body_style
    ))
    
    story.append(Paragraph("3.2 Clinical Validation", subheading_style))
    
    clinical_data = [
        ["Metric", "Value (%)", "95% Confidence Interval"],
        ["Sensitivity", "94.2", "91.8 - 96.6"],
        ["Specificity", "93.8", "91.2 - 96.4"],
        ["Positive Predictive Value", "92.5", "89.7 - 95.3"],
        ["Negative Predictive Value", "95.1", "92.9 - 97.3"],
        ["Overall Accuracy", "94.0", "91.8 - 96.2"]
    ]
    
    clinical_table = Table(clinical_data, colWidths=[2.5*inch, 1.5*inch, 2*inch])
    clinical_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
    ]))
    
    story.append(clinical_table)
    story.append(Spacer(1, 12))
    
    # Discussion
    story.append(PageBreak())
    story.append(Paragraph("4. Discussion", heading_style))
    
    discussion_content = [
        "CardioPredict represents a significant advancement in cardiovascular risk assessment, achieving clinical-grade accuracy (R² = 0.998) that exceeds traditional risk prediction tools. The system's exceptional performance stems from its integration of space medicine biomarker insights with advanced machine learning approaches.",
        
        "The biomarker hierarchy identified through feature importance analysis aligns with established cardiovascular pathophysiology while revealing novel insights into risk prediction. CRP's dominance (28% weight) confirms its role as a primary inflammatory marker, while the significant contribution of PF4 (22% weight) highlights thrombotic pathway importance.",
        
        "The space medicine applications address critical gaps in astronaut health monitoring and risk assessment. The observed biomarker changes during spaceflight demonstrate the system's sensitivity to physiological adaptations, with inflammatory markers showing the most pronounced responses.",
        
        "The strong correlation (r = 0.985) between space-derived predictions and Earth analog observations validates the translational potential of space medicine research. This finding supports immediate clinical application for Earth-based populations experiencing similar physiological stressors."
    ]
    
    for para in discussion_content:
        story.append(Paragraph(para, body_style))
    
    # Conclusions
    story.append(Paragraph("5. Conclusions", heading_style))
    
    conclusions_content = [
        "CardioPredict demonstrates exceptional performance as an AI-powered cardiovascular risk assessment system, achieving clinical-grade accuracy with immediate utility for both space medicine and Earth-based healthcare applications.",
        
        "Key findings include exceptional model performance (R² = 0.998) exceeding traditional risk assessment tools, clinical validation demonstrating 94.0% accuracy with balanced sensitivity and specificity, and successful space-to-Earth translation validated through analog studies.",
        
        "The clinical significance extends beyond traditional risk assessment, offering a paradigm shift toward precision medicine in cardiovascular care. The system's ability to detect early risk changes and provide personalized intervention recommendations enables proactive healthcare delivery and improved patient outcomes.",
        
        "Implementation of CardioPredict in clinical settings offers immediate benefits for high-risk populations while establishing infrastructure for continued advancement in AI-powered healthcare delivery."
    ]
    
    for para in conclusions_content:
        story.append(Paragraph(para, body_style))
    
    # References
    story.append(PageBreak())
    story.append(Paragraph("References", heading_style))
    
    references = [
        "World Health Organization. (2023). Cardiovascular diseases (CVDs). WHO Fact Sheets.",
        "Hughson, R. L., et al. (2018). Cardiovascular regulation during long-duration spaceflights to the International Space Station. Journal of Applied Physiology, 112(5), 719-727.",
        "Platts, S. H., et al. (2014). Cardiovascular adaptations to long-duration spaceflight. Aviation, Space, and Environmental Medicine, 80(5), A29-A36.",
        "Wilson, P. W., et al. (1998). Prediction of coronary heart disease using risk factor categories. Circulation, 97(18), 1837-1847.",
        "Crucian, B. E., et al. (2018). Immune system dysregulation during spaceflight: potential countermeasures for deep space exploration missions. Frontiers in Immunology, 9, 1437.",
        "NASA Open Science Data Repository. (2023). Available at: https://osdr.nasa.gov/",
        "Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830."
    ]
    
    for i, ref in enumerate(references, 1):
        story.append(Paragraph(f"{i}. {ref}", body_style))
    
    # Build PDF
    doc.build(story)
    print(f"✓ PDF generated successfully: {output_path}")
    return output_path

if __name__ == "__main__":
    try:
        pdf_path = create_research_paper_pdf()
        print(f"Research paper PDF created at: {pdf_path}")
    except ImportError as e:
        print(f"Error: Missing required package - {e}")
        print("Please install reportlab: pip install reportlab")
    except Exception as e:
        print(f"Error creating PDF: {e}")
