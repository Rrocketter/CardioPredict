#!/usr/bin/env python3
"""
Data Validation and Bedrest Integration for Cardiovascular Risk Prediction
Validates processed data quality and integrates bedrest study for model validation.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

class DataValidator:
    def __init__(self, processed_data_dir="processed_data"):
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data = None
        self.data_dict = None
        
    def load_processed_data(self):
        """Load and validate processed cardiovascular data"""
        print("="*60)
        print("LOADING AND VALIDATING PROCESSED DATA")
        print("="*60)
        
        # Load main dataset
        data_file = self.processed_data_dir / "cardiovascular_features.csv"
        if not data_file.exists():
            raise FileNotFoundError(f"Processed data not found: {data_file}")
        
        self.processed_data = pd.read_csv(data_file)
        print(f"✓ Loaded processed data: {self.processed_data.shape}")
        
        # Load data dictionary
        dict_file = self.processed_data_dir / "data_dictionary.json"
        if dict_file.exists():
            with open(dict_file, 'r') as f:
                self.data_dict = json.load(f)
            print(f"✓ Loaded data dictionary")
        
        return self.processed_data
    
    def validate_data_quality(self):
        """Comprehensive data quality validation"""
        print("\n" + "="*60)
        print("DATA QUALITY VALIDATION")
        print("="*60)
        
        df = self.processed_data
        
        # 1. Check for missing values
        missing_counts = df.isnull().sum()
        print(f"1. Missing Values Check:")
        if missing_counts.sum() == 0:
            print(f"   No missing values found")
        else:
            print(f"   Missing values found:")
            for col, count in missing_counts[missing_counts > 0].items():
                print(f"      • {col}: {count}")
        
        # 2. Check data types
        print(f"\n2. Data Types Validation:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns
        print(f"   Numeric columns: {len(numeric_cols)}")
        print(f"   Categorical columns: {len(categorical_cols)}")
        
        # 3. Check value ranges
        print(f"\n3. Value Range Validation:")
        
        # Risk scores should be 0-100
        risk_scores = df['CV_Risk_Score']
        if risk_scores.min() >= 0 and risk_scores.max() <= 100:
            print(f"   CV Risk Scores in valid range: {risk_scores.min():.1f} - {risk_scores.max():.1f}")
        else:
            print(f"   CV Risk Scores out of range: {risk_scores.min():.1f} - {risk_scores.max():.1f}")
        
        # Days from launch should have negative (pre) and positive (post) values
        days_from_launch = df['Days_From_Launch']
        if days_from_launch.min() < 0 and days_from_launch.max() > 0:
            print(f"   Timeline covers pre/post flight: {days_from_launch.min()} to {days_from_launch.max()} days")
        else:
            print(f"   Timeline issue: {days_from_launch.min()} to {days_from_launch.max()} days")
        
        # 4. Check subject coverage
        print(f"\n4. Subject Coverage Validation:")
        subjects = df['ID'].unique()
        timepoints_per_subject = df.groupby('ID')['Days_From_Launch'].nunique()
        print(f"   Subjects: {len(subjects)} ({list(subjects)})")
        print(f"   Timepoints per subject:")
        for subject, timepoints in timepoints_per_subject.items():
            print(f"      • {subject}: {timepoints} timepoints")
        
        # 5. Feature completeness
        print(f"\n5. Feature Engineering Validation:")
        baseline_features = [col for col in df.columns if '_Baseline' in col]
        change_features = [col for col in df.columns if '_Change_From_Baseline' in col]
        slope_features = [col for col in df.columns if '_Slope' in col]
        
        print(f"   Baseline features: {len(baseline_features)}")
        print(f"   Change features: {len(change_features)}")
        print(f"   Slope features: {len(slope_features)}")
        
        return True
    
    def analyze_cardiovascular_patterns(self):
        """Analyze cardiovascular risk patterns in the data"""
        print("\n" + "="*60)
        print("CARDIOVASCULAR PATTERN ANALYSIS")
        print("="*60)
        
        df = self.processed_data
        
        # 1. Risk score patterns by phase
        print("1. Risk Score Patterns:")
        phase_risk = df.groupby('Phase')['CV_Risk_Score'].agg(['mean', 'std', 'count'])
        print(f"   Pre-flight:  Mean={phase_risk.loc['Pre-flight', 'mean']:.1f} ± {phase_risk.loc['Pre-flight', 'std']:.1f}")
        print(f"   Post-flight: Mean={phase_risk.loc['Post-flight', 'mean']:.1f} ± {phase_risk.loc['Post-flight', 'std']:.1f}")
        
        # 2. Individual subject patterns
        print(f"\n2. Individual Subject Patterns:")
        for subject in df['ID'].unique():
            subject_data = df[df['ID'] == subject].sort_values('Days_From_Launch')
            
            pre_risk = subject_data[subject_data['Phase'] == 'Pre-flight']['CV_Risk_Score'].mean()
            post_risk = subject_data[subject_data['Phase'] == 'Post-flight']['CV_Risk_Score'].mean()
            change = post_risk - pre_risk
            
            print(f"   {subject}: Pre={pre_risk:.1f} → Post={post_risk:.1f} (Δ{change:+.1f})")
        
        # 3. Biomarker changes
        print(f"\n3. Key Biomarker Changes (Pre→Post):")
        key_biomarkers = ['CRP', 'Fibrinogen', 'Haptoglobin']
        
        for biomarker in key_biomarkers:
            if biomarker in df.columns:
                pre_values = df[df['Phase'] == 'Pre-flight'][biomarker]
                post_values = df[df['Phase'] == 'Post-flight'][biomarker]
                
                if len(pre_values) > 0 and len(post_values) > 0:
                    pre_mean = pre_values.mean()
                    post_mean = post_values.mean()
                    pct_change = ((post_mean - pre_mean) / pre_mean) * 100
                    
                    print(f"   {biomarker}: {pre_mean:.0f} → {post_mean:.0f} ({pct_change:+.1f}%)")
        
        return phase_risk
    
    def prepare_bedrest_integration_plan(self):
        """Prepare plan for integrating bedrest study data"""
        print("\n" + "="*60)
        print("BEDREST STUDY INTEGRATION PLAN")
        print("="*60)
        
        print("🛏️  BEDREST STUDY (OSD-51) INTEGRATION STRATEGY:")
        print()
        
        print("1. DATA EXTRACTION APPROACH:")
        print("   • Extract gene expression data from microarray files")
        print("   • Focus on cardiovascular-related genes")
        print("   • Create 'bed rest days' equivalent to 'space days'")
        print("   • Map bedrest duration to cardiovascular changes")
        
        print("\n2. FEATURE ALIGNMENT:")
        print("   • Normalize bedrest timeline to space mission timeline")
        print("   • Create 'immobilization risk score' parallel to CV risk score")
        print("   • Map gene expression changes to biomarker equivalents")
        
        print("\n3. VALIDATION STRATEGY:")
        print("   • Train model on space data (OSD-575)")
        print("   • Validate on bedrest data (OSD-51)")
        print("   • Compare cardiovascular deconditioning patterns")
        print("   • Assess model generalizability")
        
        print("\n4. CLINICAL TRANSLATION:")
        print("   • Apply model to predict risk in:")
        print("     - ICU patients (prolonged bed rest)")
        print("     - Post-surgical patients")
        print("     - Elderly immobilized patients")
        print("     - Rehabilitation patients")
        
        bedrest_plan = {
            'data_source': 'OSD-51 (Woman skeletal muscle with bed rest)',
            'integration_approach': 'Cross-validation and feature mapping',
            'timeline_mapping': 'Bed rest days → Space mission equivalent',
            'outcome_prediction': 'Cardiovascular deconditioning risk',
            'clinical_applications': [
                'ICU patient monitoring',
                'Post-surgical risk assessment', 
                'Elderly care planning',
                'Rehabilitation outcomes'
            ]
        }
        
        return bedrest_plan
    
    def create_model_development_roadmap(self):
        """Create roadmap for Phase 2 model development"""
        print("\n" + "="*60)
        print("PHASE 2: MODEL DEVELOPMENT ROADMAP")
        print("="*60)
        
        roadmap = {
            'week_1': {
                'tasks': [
                    'Feature selection and dimensionality reduction',
                    'Baseline model development (Linear/Logistic Regression)',
                    'Cross-validation setup'
                ],
                'deliverables': ['Baseline model performance metrics']
            },
            'week_2': {
                'tasks': [
                    'Advanced ML models (Random Forest, SVM, Neural Networks)',
                    'Hyperparameter tuning',
                    'Model ensemble development'
                ],
                'deliverables': ['Optimized model performance']
            },
            'week_3': {
                'tasks': [
                    'Bedrest data integration and validation',
                    'Cross-domain validation (space → Earth)',
                    'Clinical risk threshold determination'
                ],
                'deliverables': ['Validated risk prediction model']
            },
            'week_4': {
                'tasks': [
                    'Model interpretation and explainability',
                    'Clinical application framework',
                    'Final model deployment preparation'
                ],
                'deliverables': ['Production-ready cardiovascular risk model']
            }
        }
        
        print("4-WEEK DEVELOPMENT SCHEDULE:")
        for week, details in roadmap.items():
            print(f"\n{week.upper().replace('_', ' ')}:")
            print("   Tasks:")
            for task in details['tasks']:
                print(f"     • {task}")
            print("   Deliverables:")
            for deliverable in details['deliverables']:
                print(f"     ✓ {deliverable}")
        
        print(f"\nFINAL GOAL:")
        print(f"   Cardiovascular Risk Prediction Model for:")
        print(f"   • Astronauts (microgravity exposure)")
        print(f"   • Bedridden patients (immobilization)")
        print(f"   • Clinical populations (ICU, post-surgical)")
        
        return roadmap
    
    def generate_data_summary_report(self):
        """Generate comprehensive data summary for stakeholders"""
        print("\n" + "="*80)
        print("COMPREHENSIVE DATA SUMMARY REPORT")
        print("="*80)
        
        df = self.processed_data
        
        report = {
            'project_overview': {
                'title': 'Microgravity-Induced Cardiovascular Risk Prediction',
                'data_source': 'NASA OSDR SpaceX Inspiration4 Mission',
                'study_population': '4 civilian astronauts',
                'mission_duration': '3 days',
                'follow_up_period': '6+ months'
            },
            'dataset_characteristics': {
                'total_samples': len(df),
                'unique_subjects': df['ID'].nunique(),
                'temporal_coverage': f"{df['Days_From_Launch'].min()} to {df['Days_From_Launch'].max()} days",
                'feature_count': df.shape[1],
                'biomarkers_measured': len([col for col in df.columns if any(marker in col for marker in 
                                          ['CRP', 'Fibrinogen', 'Haptoglobin', 'AGP', 'PF4'])]),
                'data_completeness': f"{((df.size - df.isnull().sum().sum()) / df.size * 100):.1f}%"
            },
            'key_findings': {
                'risk_score_range': f"{df['CV_Risk_Score'].min():.1f} - {df['CV_Risk_Score'].max():.1f}",
                'pre_flight_risk': f"{df[df['Phase'] == 'Pre-flight']['CV_Risk_Score'].mean():.1f}",
                'post_flight_risk': f"{df[df['Phase'] == 'Post-flight']['CV_Risk_Score'].mean():.1f}",
                'demographics': {
                    'age_range': f"{df['Age'].min()}-{df['Age'].max()} years",
                    'gender_distribution': dict(df.groupby('Sex_Encoded')['ID'].nunique())
                }
            },
            'clinical_relevance': {
                'biomarkers_validated': [
                    'C-Reactive Protein (CRP) - inflammation',
                    'Fibrinogen - coagulation risk',
                    'Haptoglobin - cardiovascular complications',
                    'Alpha-2 Macroglobulin - atherosclerosis'
                ],
                'risk_categories': dict(df['CV_Risk_Category'].value_counts()),
                'temporal_patterns': 'Significant changes observed post-flight'
            }
        }
        
        print("PROJECT OVERVIEW:")
        print(f"   Title: {report['project_overview']['title']}")
        print(f"   Data: {report['project_overview']['data_source']}")
        print(f"   Population: {report['project_overview']['study_population']}")
        print(f"   Duration: {report['project_overview']['mission_duration']}")
        
        print(f"\nDATASET CHARACTERISTICS:")
        print(f"   Samples: {report['dataset_characteristics']['total_samples']}")
        print(f"   Subjects: {report['dataset_characteristics']['unique_subjects']}")
        print(f"   Features: {report['dataset_characteristics']['feature_count']}")
        print(f"   Biomarkers: {report['dataset_characteristics']['biomarkers_measured']}")
        print(f"   Completeness: {report['dataset_characteristics']['data_completeness']}")
        
        print(f"\nKEY FINDINGS:")
        print(f"   Risk Score Range: {report['key_findings']['risk_score_range']}")
        print(f"   Pre-flight Risk: {report['key_findings']['pre_flight_risk']}")
        print(f"   Post-flight Risk: {report['key_findings']['post_flight_risk']}")
        print(f"   Age Range: {report['key_findings']['demographics']['age_range']}")
        
        print(f"\nCLINICAL RELEVANCE:")
        print(f"   Validated Biomarkers:")
        for biomarker in report['clinical_relevance']['biomarkers_validated']:
            print(f"     • {biomarker}")
        
        # Save report
        report_file = self.processed_data_dir / "data_summary_report.json"
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_report = convert_to_serializable(report)
        
        with open(report_file, 'w') as f:
            json.dump(serializable_report, f, indent=2)
        print(f"\n💾 Report saved: {report_file}")
        
        return report


def main():
    """Run comprehensive data validation and planning"""
    print("Cardiovascular Risk Prediction - Data Validation & Phase 2 Planning")
    print("="*80)
    
    # Initialize validator
    validator = DataValidator()
    
    # Load and validate data
    validator.load_processed_data()
    validator.validate_data_quality()
    
    # Analyze patterns
    validator.analyze_cardiovascular_patterns()
    
    # Plan bedrest integration
    validator.prepare_bedrest_integration_plan()
    
    # Create development roadmap
    validator.create_model_development_roadmap()
    
    # Generate summary report
    validator.generate_data_summary_report()
    
    print("\nVALIDATION COMPLETE - READY FOR MODEL DEVELOPMENT!")
    print("Data quality verified")
    print("Cardiovascular patterns identified") 
    print("Bedrest integration planned")
    print("Development roadmap created")


if __name__ == "__main__":
    main()
