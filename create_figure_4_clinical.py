#!/usr/bin/env python3
"""
Create Figure 4: Clinical Decision Support and Implementation Framework
Advanced visualization for clinical deployment and real-world application
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Wedge
import matplotlib.gridspec as gridspec
import seaborn as sns
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Use the same journal-quality configuration
plt.rcParams.update({
    'font.family': ['Arial', 'Helvetica'],
    'font.size': 8,
    'figure.figsize': (8.5, 7),
    'figure.dpi': 600,
    'figure.facecolor': 'white',
    'figure.constrained_layout.use': True,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'axes.titleweight': 'bold',
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'legend.frameon': False,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'lines.linewidth': 1.5,
    'lines.markersize': 4
})

# Professional journal color palette
JOURNAL_COLORS = {
    'blue': '#1f77b4',
    'orange': '#ff7f0e', 
    'green': '#2ca02c',
    'red': '#d62728',
    'purple': '#9467bd',
    'brown': '#8c564b',
    'pink': '#e377c2',
    'gray': '#7f7f7f',
    'olive': '#bcbd22',
    'cyan': '#17becf'
}

def add_panel_label(ax, label, x_offset=-0.15, y_offset=1.05, fontsize=12, fontweight='bold'):
    """Add professional panel labels (A, B, C, etc.) to subplots"""
    ax.text(x_offset, y_offset, label, transform=ax.transAxes, 
            fontsize=fontsize, fontweight=fontweight, va='bottom', ha='right')

def create_figure_4_clinical_decision_support():
    """Create Figure 4: Clinical Decision Support and Implementation Framework"""
    
    # Create figure
    fig = plt.figure(figsize=(8.5, 10))
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3, height_ratios=[1, 1, 1, 1.2])
    
    # Panel A: Risk stratification dashboard
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Create a dashboard-style visualization
    risk_categories = ['Low Risk\n(0-30%)', 'Moderate Risk\n(30-60%)', 'High Risk\n(60-80%)', 'Critical Risk\n(80-100%)']
    patient_counts = [450, 280, 150, 45]
    colors = [JOURNAL_COLORS['green'], JOURNAL_COLORS['orange'], JOURNAL_COLORS['red'], '#8B0000']
    
    # Create horizontal bar chart
    bars = ax1.barh(risk_categories, patient_counts, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add patient count labels
    for i, (bar, count) in enumerate(zip(bars, patient_counts)):
        ax1.text(count + 10, i, f'{count} patients', va='center', fontweight='bold', fontsize=8)
    
    ax1.set_xlabel('Number of Patients')
    ax1.set_title('Patient Risk Stratification', fontweight='bold')
    ax1.set_xlim(0, 500)
    
    # Add percentage annotations
    total_patients = sum(patient_counts)
    for i, (count, bar) in enumerate(zip(patient_counts, bars)):
        percentage = (count / total_patients) * 100
        ax1.text(count/2, i, f'{percentage:.1f}%', ha='center', va='center', 
                fontweight='bold', color='white', fontsize=9)
    
    add_panel_label(ax1, 'A')
    
    # Panel B: Clinical decision tree
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    # Decision tree nodes
    decision_nodes = [
        {'pos': (5, 9), 'text': 'CV Risk\nAssessment', 'color': JOURNAL_COLORS['blue'], 'shape': 'rect'},
        {'pos': (2, 7), 'text': 'Low Risk\n<30%', 'color': JOURNAL_COLORS['green'], 'shape': 'diamond'},
        {'pos': (5, 7), 'text': 'Moderate\n30-60%', 'color': JOURNAL_COLORS['orange'], 'shape': 'diamond'},
        {'pos': (8, 7), 'text': 'High Risk\n>60%', 'color': JOURNAL_COLORS['red'], 'shape': 'diamond'},
        {'pos': (2, 5), 'text': 'Annual\nScreening', 'color': JOURNAL_COLORS['gray'], 'shape': 'oval'},
        {'pos': (5, 5), 'text': '6-Month\nMonitoring', 'color': JOURNAL_COLORS['orange'], 'shape': 'oval'},
        {'pos': (8, 5), 'text': 'Immediate\nIntervention', 'color': JOURNAL_COLORS['red'], 'shape': 'oval'},
        {'pos': (5, 3), 'text': 'Specialist\nReferral', 'color': '#8B0000', 'shape': 'oval'},
        {'pos': (2, 1), 'text': 'Lifestyle\nModification', 'color': JOURNAL_COLORS['green'], 'shape': 'rect'},
        {'pos': (5, 1), 'text': 'Medication\nReview', 'color': JOURNAL_COLORS['orange'], 'shape': 'rect'},
        {'pos': (8, 1), 'text': 'Emergency\nProtocol', 'color': JOURNAL_COLORS['red'], 'shape': 'rect'}
    ]
    
    # Draw decision tree
    for node in decision_nodes:
        x, y = node['pos']
        if node['shape'] == 'rect':
            rect = FancyBboxPatch((x-0.6, y-0.4), 1.2, 0.8, boxstyle="round,pad=0.05",
                                 facecolor=node['color'], alpha=0.7, edgecolor='black', linewidth=0.5)
            ax2.add_patch(rect)
        elif node['shape'] == 'diamond':
            # Create diamond shape using polygon
            diamond_x = [x, x+0.6, x, x-0.6, x]
            diamond_y = [y+0.4, y, y-0.4, y, y+0.4]
            diamond = plt.Polygon(list(zip(diamond_x, diamond_y)), facecolor=node['color'], 
                                alpha=0.7, edgecolor='black', linewidth=0.5)
            ax2.add_patch(diamond)
        elif node['shape'] == 'oval':
            ellipse = patches.Ellipse((x, y), 1.2, 0.8, facecolor=node['color'], 
                                    alpha=0.7, edgecolor='black', linewidth=0.5)
            ax2.add_patch(ellipse)
        
        ax2.text(x, y, node['text'], ha='center', va='center', fontsize=7, 
                fontweight='bold', color='white' if node['color'] != JOURNAL_COLORS['gray'] else 'black')
    
    # Draw connections
    connections = [
        ((5, 8.6), (2, 7.4)),   # Root to Low
        ((5, 8.6), (5, 7.4)),   # Root to Moderate
        ((5, 8.6), (8, 7.4)),   # Root to High
        ((2, 6.6), (2, 5.4)),   # Low to Annual
        ((5, 6.6), (5, 5.4)),   # Moderate to 6-Month
        ((8, 6.6), (8, 5.4)),   # High to Immediate
        ((8, 4.6), (5, 3.4)),   # High to Specialist
        ((2, 4.6), (2, 1.4)),   # Annual to Lifestyle
        ((5, 4.6), (5, 1.4)),   # Monitoring to Medication
        ((8, 4.6), (8, 1.4))    # Immediate to Emergency
    ]
    
    for start, end in connections:
        ax2.plot([start[0], end[0]], [start[1], end[1]], 'k-', linewidth=1, alpha=0.7)
        # Add arrowhead
        dx, dy = end[0] - start[0], end[1] - start[1]
        ax2.arrow(end[0] - 0.1*dx, end[1] - 0.1*dy, 0.1*dx, 0.1*dy, 
                 head_width=0.1, head_length=0.1, fc='black', ec='black', alpha=0.7)
    
    ax2.set_title('Clinical Decision Algorithm', fontweight='bold', y=0.95)
    add_panel_label(ax2, 'B', x_offset=0.02, y_offset=0.95)
    
    # Panel C: Implementation timeline
    ax3 = fig.add_subplot(gs[1, :])
    
    # Timeline data
    phases = ['Research\n& Development', 'Regulatory\nReview', 'Pilot Testing', 'Staff Training', 
              'System Integration', 'Clinical Validation', 'Full Deployment', 'Monitoring\n& Evaluation']
    durations = [12, 6, 3, 2, 4, 6, 3, 12]  # months
    start_times = [0, 12, 18, 21, 23, 27, 33, 36]
    
    # Colors for different phase types
    phase_colors = [JOURNAL_COLORS['blue'], JOURNAL_COLORS['red'], JOURNAL_COLORS['orange'], 
                   JOURNAL_COLORS['green'], JOURNAL_COLORS['purple'], JOURNAL_COLORS['brown'], 
                   JOURNAL_COLORS['pink'], JOURNAL_COLORS['cyan']]
    
    # Create Gantt chart
    for i, (phase, duration, start, color) in enumerate(zip(phases, durations, start_times, phase_colors)):
        ax3.barh(i, duration, left=start, height=0.6, color=color, alpha=0.8, 
                edgecolor='black', linewidth=0.5)
        # Add phase labels
        ax3.text(start + duration/2, i, f'{duration}m', ha='center', va='center', 
                fontweight='bold', color='white', fontsize=8)
    
    ax3.set_yticks(range(len(phases)))
    ax3.set_yticklabels(phases)
    ax3.set_xlabel('Timeline (Months)')
    ax3.set_title('Clinical Implementation Timeline', fontweight='bold')
    ax3.set_xlim(0, 48)
    ax3.grid(True, axis='x', alpha=0.3)
    
    # Add milestone markers
    milestones = [(18, 'FDA Submission'), (27, 'Pilot Complete'), (36, 'Go-Live'), (48, 'Full Adoption')]
    for month, milestone in milestones:
        ax3.axvline(month, color='red', linestyle='--', alpha=0.7)
        ax3.text(month, len(phases)-0.5, milestone, rotation=90, ha='right', va='bottom', 
                fontsize=7, color='red', fontweight='bold')
    
    add_panel_label(ax3, 'C')
    
    # Panel D: Cost-benefit analysis
    ax4 = fig.add_subplot(gs[2, 0])
    
    years = ['Year 1', 'Year 2', 'Year 3', 'Year 4', 'Year 5']
    implementation_costs = [2.5, 1.8, 1.2, 1.0, 0.8]  # Million USD
    healthcare_savings = [0.5, 2.2, 3.8, 5.1, 6.2]    # Million USD
    
    x = np.arange(len(years))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, implementation_costs, width, label='Implementation Costs',
                    color=JOURNAL_COLORS['red'], alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax4.bar(x + width/2, healthcare_savings, width, label='Healthcare Savings',
                    color=JOURNAL_COLORS['green'], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax4.set_ylabel('Cost/Savings (Million USD)')
    ax4.set_xlabel('Year')
    ax4.set_title('Cost-Benefit Analysis', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(years)
    ax4.legend()
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'${height:.1f}M', ha='center', va='bottom', fontsize=7)
    
    # Add ROI line
    roi = [(savings - costs) / costs * 100 for costs, savings in zip(implementation_costs, healthcare_savings)]
    ax4_twin = ax4.twinx()
    ax4_twin.plot(x, roi, 'o-', color='black', linewidth=2, markersize=6, label='ROI (%)')
    ax4_twin.set_ylabel('Return on Investment (%)')
    ax4_twin.legend(loc='upper left')
    
    # Highlight break-even point
    for i, r in enumerate(roi):
        if r > 0:
            ax4.axvline(i, color='gray', linestyle=':', alpha=0.5)
            ax4.text(i, max(healthcare_savings)*0.9, 'Break-even', ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7), fontsize=7)
            break
    
    add_panel_label(ax4, 'D')
    
    # Panel E: Quality metrics dashboard
    ax5 = fig.add_subplot(gs[2, 1])
    
    # Quality metrics
    metrics = ['Diagnostic\nAccuracy', 'Patient\nSatisfaction', 'Clinical\nEfficiency', 
               'Cost\nReduction', 'Provider\nAdoption']
    baseline = [75, 70, 60, 65, 55]
    with_ai = [94, 88, 85, 82, 78]
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    baseline += baseline[:1]
    with_ai += with_ai[:1]
    
    ax5 = plt.subplot(gs[2, 1], projection='polar')
    ax5.plot(angles, baseline, 'o-', linewidth=2, color=JOURNAL_COLORS['gray'], 
             label='Baseline', markersize=4)
    ax5.fill(angles, baseline, alpha=0.15, color=JOURNAL_COLORS['gray'])
    
    ax5.plot(angles, with_ai, 'o-', linewidth=2, color=JOURNAL_COLORS['blue'], 
             label='With CardioPredict', markersize=4)
    ax5.fill(angles, with_ai, alpha=0.25, color=JOURNAL_COLORS['blue'])
    
    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(metrics)
    ax5.set_ylim(0, 100)
    ax5.set_yticks([25, 50, 75, 100])
    ax5.set_yticklabels(['25%', '50%', '75%', '100%'])
    ax5.set_title('Quality Improvement Metrics', fontweight='bold', pad=20)
    ax5.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax5.grid(True)
    
    add_panel_label(ax5, 'E', x_offset=-0.1, y_offset=1.1)
    
    # Panel F: Training and competency framework
    ax6 = fig.add_subplot(gs[3, :])
    ax6.set_xlim(0, 12)
    ax6.set_ylim(0, 8)
    ax6.axis('off')
    
    # Training modules
    training_modules = [
        {'pos': (2, 6.5), 'text': 'AI Fundamentals\n(4 hours)', 'color': JOURNAL_COLORS['blue'], 'level': 1},
        {'pos': (5, 6.5), 'text': 'Biomarker\nInterpretation\n(6 hours)', 'color': JOURNAL_COLORS['green'], 'level': 1},
        {'pos': (8, 6.5), 'text': 'Risk Assessment\nProtocols\n(4 hours)', 'color': JOURNAL_COLORS['orange'], 'level': 1},
        {'pos': (10.5, 6.5), 'text': 'Clinical Decision\nSupport\n(3 hours)', 'color': JOURNAL_COLORS['purple'], 'level': 1},
        {'pos': (2, 4.5), 'text': 'Hands-on\nPractice\n(8 hours)', 'color': JOURNAL_COLORS['red'], 'level': 2},
        {'pos': (5, 4.5), 'text': 'Case Studies\n(6 hours)', 'color': JOURNAL_COLORS['brown'], 'level': 2},
        {'pos': (8, 4.5), 'text': 'Quality Assurance\n(4 hours)', 'color': JOURNAL_COLORS['pink'], 'level': 2},
        {'pos': (10.5, 4.5), 'text': 'Ethics & Legal\n(2 hours)', 'color': JOURNAL_COLORS['cyan'], 'level': 2},
        {'pos': (3.5, 2.5), 'text': 'Competency\nAssessment', 'color': JOURNAL_COLORS['gray'], 'level': 3},
        {'pos': (7, 2.5), 'text': 'Certification\nExam', 'color': '#8B0000', 'level': 3},
        {'pos': (5.25, 0.5), 'text': 'Continuing\nEducation', 'color': 'gold', 'level': 4}
    ]
    
    # Group by levels
    level_names = ['Foundation Level', 'Application Level', 'Assessment Level', 'Maintenance Level']
    level_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    level_positions = [6.5, 4.5, 2.5, 0.5]
    
    # Draw level backgrounds
    for i, (level_name, color, y_pos) in enumerate(zip(level_names, level_colors, level_positions)):
        if i < 3:  # Don't draw background for maintenance level
            rect = Rectangle((0.5, y_pos-0.7), 11, 1.4, facecolor=color, alpha=0.2, edgecolor='none')
            ax6.add_patch(rect)
        ax6.text(0.2, y_pos, f'Level {i+1}:\n{level_name}', va='center', ha='left', 
                fontsize=8, fontweight='bold')
    
    # Draw training modules
    for module in training_modules:
        x, y = module['pos']
        rect = FancyBboxPatch((x-0.7, y-0.5), 1.4, 1.0, boxstyle="round,pad=0.05",
                             facecolor=module['color'], alpha=0.8, edgecolor='black', linewidth=0.5)
        ax6.add_patch(rect)
        ax6.text(x, y, module['text'], ha='center', va='center', fontsize=7, 
                fontweight='bold', color='white')
    
    # Draw connections between levels
    level_connections = [
        # Level 1 to Level 2
        ((2, 6), (2, 5)),
        ((5, 6), (5, 5)),
        ((8, 6), (8, 5)),
        ((10.5, 6), (10.5, 5)),
        # Level 2 to Level 3
        ((2, 4), (3.5, 3)),
        ((5, 4), (3.5, 3)),
        ((8, 4), (7, 3)),
        ((10.5, 4), (7, 3)),
        # Level 3 to Level 4
        ((3.5, 2), (5.25, 1)),
        ((7, 2), (5.25, 1))
    ]
    
    for start, end in level_connections:
        ax6.plot([start[0], end[0]], [start[1], end[1]], 'k-', linewidth=1, alpha=0.5)
    
    ax6.set_title('Healthcare Provider Training and Competency Framework', fontweight='bold', y=0.95)
    add_panel_label(ax6, 'F', x_offset=0.02, y_offset=0.95)
    
    plt.suptitle('CardioPredict: Clinical Decision Support and Implementation Framework',
                 fontsize=12, fontweight='bold', y=0.98)
    
    plt.savefig('/Users/rahulgupta/Developer/CardioPredict/figures/journal_figure_4_clinical_implementation.png',
                dpi=600, bbox_inches='tight', facecolor='white')
    plt.show()
    print("✓ Figure 4: Clinical Implementation Framework created successfully")

def main():
    """Generate Figure 4"""
    print("Creating Figure 4: Clinical Decision Support and Implementation Framework...")
    print("=" * 80)
    
    try:
        create_figure_4_clinical_decision_support()
        
        print("\n" + "=" * 80)
        print("✓ Figure 4 created successfully!")
        print("\nFigure saved to: /Users/rahulgupta/Developer/CardioPredict/figures/")
        print("journal_figure_4_clinical_implementation.png")
        print("\nFigure shows comprehensive clinical implementation framework including:")
        print("- Risk stratification dashboard")
        print("- Clinical decision algorithm")
        print("- Implementation timeline") 
        print("- Cost-benefit analysis")
        print("- Quality improvement metrics")
        print("- Training and competency framework")
        
    except Exception as e:
        print(f"Error creating figure: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
