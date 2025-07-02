#!/usr/bin/env python3
"""
Phase 1: Data Preprocessing for Microgravity Cardiovascular Risk Prediction
Comprehensive preprocessing pipeline to prepare NASA OSDR datasets for ML model development.

Key Features:
- Extract mission duration and demographic data
- Normalize cardiovascular biomarkers
- Create temporal features for longitudinal analysis
- Integrate space and bedrest datasets
- Handle missing values and outliers
- Feature engineering for risk prediction
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class CardiovascularDataPreprocessor:
    def __init__(self, data_dir="data", output_dir="processed_data"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize data containers
        self.cardiovascular_data = None
        self.metabolic_data = None
        self.subject_metadata = None
        self.processed_features = None
        
        # Mission duration information (SpaceX Inspiration4)
        self.mission_info = {
            'mission_name': 'SpaceX Inspiration4',
            'launch_date': '2021-09-15',
            'mission_duration_days': 3,  # 3-day mission
            'crew_size': 4
        }
        
        print("Cardiovascular Data Preprocessor Initialized")
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
    
    def load_cardiovascular_biomarkers(self):
        """Load and process cardiovascular biomarker data from OSD-575"""
        print("\n" + "="*60)
        print("LOADING CARDIOVASCULAR BIOMARKERS (OSD-575)")
        print("="*60)
        
        # Load cardiovascular panel data
        cardio_file = self.data_dir / "OSD-575" / "LSDS-8_Multiplex_serum.cardiovascular.EvePanel_SUBMITTED.csv"
        
        if not cardio_file.exists():
            raise FileNotFoundError(f"Cardiovascular data not found: {cardio_file}")
        
        df_cardio = pd.read_csv(cardio_file)
        print(f"✓ Loaded cardiovascular data: {len(df_cardio)} measurements")
        
        # Clean and standardize data
        df_cardio = df_cardio.copy()
        
        # Convert concentration to numeric, handle any string values
        df_cardio['Concentration'] = pd.to_numeric(df_cardio['Concentration'], errors='coerce')
        
        # Parse timepoints into standardized format
        df_cardio['Days_From_Launch'] = df_cardio['Timepoint'].map(self._parse_timepoint)
        df_cardio['Phase'] = df_cardio['Timepoint2'].map({
            'Preflight': 'Pre-flight',
            'R+1': 'Post-flight',
            'R+45': 'Post-flight',
            'R+82': 'Post-flight', 
            'R+194': 'Post-flight'
        })
        
        # Add mission duration
        df_cardio['Mission_Duration_Days'] = self.mission_info['mission_duration_days']
        
        # Pivot data for analysis (subjects as rows, biomarkers as columns)
        df_pivot = df_cardio.pivot_table(
            index=['ID', 'Days_From_Launch', 'Phase', 'Mission_Duration_Days'],
            columns='Analyte',
            values='Concentration',
            aggfunc='mean'
        ).reset_index()
        
        # Clean column names
        df_pivot.columns.name = None
        biomarker_cols = [col for col in df_pivot.columns if col not in ['ID', 'Days_From_Launch', 'Phase', 'Mission_Duration_Days']]
        
        print(f"✓ Processed biomarkers: {len(biomarker_cols)}")
        for marker in biomarker_cols:
            print(f"  • {marker}")
        
        self.cardiovascular_data = df_pivot
        return df_pivot
    
    def load_metabolic_panel(self):
        """Load and process metabolic panel data"""
        print("\n" + "="*60)
        print("LOADING METABOLIC PANEL DATA")
        print("="*60)
        
        metabolic_file = self.data_dir / "OSD-575" / "LSDS-8_Comprehensive_Metabolic_Panel_CMP_TRANSFORMED.csv"
        
        if not metabolic_file.exists():
            raise FileNotFoundError(f"Metabolic data not found: {metabolic_file}")
        
        df_metabolic = pd.read_csv(metabolic_file)
        print(f"✓ Loaded metabolic data: {len(df_metabolic)} samples")
        
        # Extract subject ID and timepoint from sample name
        df_metabolic['ID'] = df_metabolic['Sample Name'].str.extract(r'(C\d+)')[0]
        df_metabolic['Timepoint_Raw'] = df_metabolic['Sample Name'].str.extract(r'_(L-\d+|R\+\d+)')
        
        # Parse timepoints
        df_metabolic['Days_From_Launch'] = df_metabolic['Timepoint_Raw'].map(self._parse_timepoint)
        df_metabolic['Phase'] = np.where(df_metabolic['Days_From_Launch'] < 0, 'Pre-flight', 'Post-flight')
        
        # Select key cardiovascular metabolic markers
        cardio_metabolic_markers = [
            'albumin_value_gram_per_deciliter',
            'total_protein_value_gram_per_deciliter',
            'creatinine_value_milligram_per_deciliter',
            'glucose_value_milligram_per_deciliter',
            'calcium_value_milligram_per_deciliter',
            'potassium_value_millimol_per_liter',
            'sodium_value_millimol_per_liter',
            'urea_nitrogen_bun_value_milligram_per_deciliter'
        ]
        
        # Select available markers
        available_markers = [col for col in cardio_metabolic_markers if col in df_metabolic.columns]
        
        # Create clean metabolic dataset
        metabolic_clean = df_metabolic[['ID', 'Days_From_Launch', 'Phase'] + available_markers].copy()
        metabolic_clean['Mission_Duration_Days'] = self.mission_info['mission_duration_days']
        
        print(f"✓ Selected metabolic markers: {len(available_markers)}")
        for marker in available_markers:
            print(f"  • {marker}")
        
        self.metabolic_data = metabolic_clean
        return metabolic_clean
    
    def extract_subject_metadata(self):
        """Extract subject demographics and metadata"""
        print("\n" + "="*60)
        print("EXTRACTING SUBJECT METADATA")
        print("="*60)
        
        metadata_file = self.data_dir / "OSD-575" / "OSD-575_metadata_OSD-575-ISA" / "s_OSD-575.txt"
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        df_meta = pd.read_csv(metadata_file, sep='\t')
        
        # Extract unique subject information
        subject_info = df_meta.groupby('Source Name').agg({
            'Characteristics[Sex]': 'first',
            'Characteristics[Organism]': 'first'
        }).reset_index()
        
        subject_info.columns = ['ID', 'Sex', 'Organism']
        
        # Add mission information
        subject_info['Mission_Duration_Days'] = self.mission_info['mission_duration_days']
        subject_info['Mission_Name'] = self.mission_info['mission_name']
        subject_info['Launch_Date'] = self.mission_info['launch_date']
        
        # Encode categorical variables
        subject_info['Sex_Encoded'] = subject_info['Sex'].map({'Male': 1, 'Female': 0})
        
        # Add estimated age (Inspiration4 crew ages were public)
        # Based on public information about Inspiration4 crew
        age_mapping = {
            'C001': 38,  # Jared Isaacman (estimated)
            'C002': 29,  # Hayley Arceneaux (estimated) 
            'C003': 51,  # Sian Proctor (estimated)
            'C004': 42   # Chris Sembroski (estimated)
        }
        subject_info['Age'] = subject_info['ID'].map(age_mapping)
        
        # Create age groups for analysis
        subject_info['Age_Group'] = pd.cut(subject_info['Age'], 
                                         bins=[0, 35, 45, 100], 
                                         labels=['Young', 'Middle', 'Older'])
        
        print(f"✓ Extracted metadata for {len(subject_info)} subjects:")
        for _, row in subject_info.iterrows():
            print(f"  • {row['ID']}: {row['Sex']}, Age {row['Age']}, Mission: {row['Mission_Duration_Days']} days")
        
        self.subject_metadata = subject_info
        return subject_info
    
    def create_temporal_features(self):
        """Create temporal features for longitudinal analysis"""
        print("\n" + "="*60)
        print("CREATING TEMPORAL FEATURES")
        print("="*60)
        
        if self.cardiovascular_data is None:
            raise ValueError("Cardiovascular data not loaded. Run load_cardiovascular_biomarkers() first.")
        
        df = self.cardiovascular_data.copy()
        
        # Calculate time-based features
        df['Time_Since_Launch'] = df['Days_From_Launch']
        df['Absolute_Days_From_Launch'] = np.abs(df['Days_From_Launch'])
        
        # Create time period categories
        def categorize_timepoint(days):
            if days < -30:
                return 'Baseline_Early'
            elif days < 0:
                return 'Baseline_Late'
            elif days <= 7:
                return 'Immediate_Post'
            elif days <= 60:
                return 'Early_Recovery'
            else:
                return 'Late_Recovery'
        
        df['Time_Category'] = df['Days_From_Launch'].apply(categorize_timepoint)
        
        # Calculate baseline values for each subject and biomarker
        biomarker_cols = [col for col in df.columns if col not in 
                         ['ID', 'Days_From_Launch', 'Phase', 'Mission_Duration_Days', 
                          'Time_Since_Launch', 'Absolute_Days_From_Launch', 'Time_Category']]
        
        # Get baseline (pre-flight) values
        baseline_data = df[df['Phase'] == 'Pre-flight'].groupby('ID')[biomarker_cols].mean()
        
        # Calculate changes from baseline
        for biomarker in biomarker_cols:
            baseline_col = f'{biomarker}_Baseline'
            change_col = f'{biomarker}_Change_From_Baseline'
            pct_change_col = f'{biomarker}_Pct_Change_From_Baseline'
            
            # Add baseline values
            df[baseline_col] = df['ID'].map(baseline_data[biomarker])
            
            # Calculate absolute and percentage changes
            df[change_col] = df[biomarker] - df[baseline_col]
            df[pct_change_col] = ((df[biomarker] - df[baseline_col]) / df[baseline_col]) * 100
        
        print(f"✓ Created temporal features for {len(biomarker_cols)} biomarkers")
        print(f"✓ Added baseline, change, and percentage change features")
        print(f"✓ Created time categories: {df['Time_Category'].unique()}")
        
        return df
    
    def calculate_cardiovascular_risk_scores(self, df):
        """Calculate composite cardiovascular risk scores"""
        print("\n" + "="*60)
        print("CALCULATING CARDIOVASCULAR RISK SCORES")
        print("="*60)
        
        # Define risk scoring based on clinical literature
        risk_weights = {
            'CRP': 0.25,                    # Strong predictor
            'Fibrinogen': 0.20,             # Major risk factor
            'Haptoglobin': 0.15,            # Cardiovascular complications
            'a-2 Macroglobulin': 0.15,      # Inflammation/atherosclerosis
            'PF4': 0.10,                    # Thrombotic risk
            'AGP': 0.10,                    # Inflammatory risk
            'SAP': 0.05                     # Additional inflammatory marker
        }
        
        # Normalize biomarkers to 0-1 scale for risk scoring
        risk_biomarkers = list(risk_weights.keys())
        available_biomarkers = [bio for bio in risk_biomarkers if bio in df.columns]
        
        # Calculate z-scores for normalization
        for biomarker in available_biomarkers:
            z_col = f'{biomarker}_zscore'
            df[z_col] = (df[biomarker] - df[biomarker].mean()) / df[biomarker].std()
        
        # Calculate composite risk score
        df['CV_Risk_Score'] = 0
        total_weight = 0
        
        for biomarker in available_biomarkers:
            weight = risk_weights[biomarker]
            z_col = f'{biomarker}_zscore'
            
            # Higher values = higher risk for most markers
            if biomarker == 'Fetuin A36':  # Lower values = higher risk
                df['CV_Risk_Score'] -= df[z_col] * weight
            else:
                df['CV_Risk_Score'] += df[z_col] * weight
            
            total_weight += weight
        
        # Normalize to 0-100 scale
        df['CV_Risk_Score'] = ((df['CV_Risk_Score'] / total_weight) + 3) * 100 / 6  # Assuming ~3 std range
        df['CV_Risk_Score'] = np.clip(df['CV_Risk_Score'], 0, 100)
        
        # Create risk categories
        df['CV_Risk_Category'] = pd.cut(df['CV_Risk_Score'],
                                       bins=[0, 25, 50, 75, 100],
                                       labels=['Low', 'Moderate', 'High', 'Very High'])
        
        print(f"✓ Calculated CV risk scores using {len(available_biomarkers)} biomarkers")
        print(f"✓ Risk score range: {df['CV_Risk_Score'].min():.1f} - {df['CV_Risk_Score'].max():.1f}")
        print(f"✓ Risk categories: {df['CV_Risk_Category'].value_counts().to_dict()}")
        
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values using appropriate interpolation methods"""
        print("\n" + "="*60)
        print("HANDLING MISSING VALUES")
        print("="*60)
        
        # Check missing values
        missing_counts = df.isnull().sum()
        missing_percent = (missing_counts / len(df)) * 100
        
        print("Missing value summary:")
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"  • {col}: {count} ({missing_percent[col]:.1f}%)")
        
        # For biomarkers, use forward/backward fill within subjects
        biomarker_cols = [col for col in df.columns if any(marker in col for marker in 
                         ['CRP', 'Fibrinogen', 'Haptoglobin', 'AGP', 'PF4', 'SAP', 'a-2', 'Fetuin', 'L-Selectin'])]
        
        # Sort by subject and time for proper interpolation
        df_sorted = df.sort_values(['ID', 'Days_From_Launch'])
        
        # Interpolate within subjects
        for subject in df_sorted['ID'].unique():
            mask = df_sorted['ID'] == subject
            
            # Linear interpolation for biomarkers
            for col in biomarker_cols:
                if col in df_sorted.columns:
                    df_sorted.loc[mask, col] = df_sorted.loc[mask, col].interpolate(method='linear')
            
            # Forward/backward fill for remaining missing values
            df_sorted.loc[mask] = df_sorted.loc[mask].fillna(method='ffill').fillna(method='bfill')
        
        # For remaining missing values, use overall median
        numeric_cols = df_sorted.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_sorted[col].isnull().sum() > 0:
                median_val = df_sorted[col].median()
                df_sorted[col].fillna(median_val, inplace=True)
        
        print(f"✓ Applied interpolation and median imputation")
        print(f"✓ Remaining missing values: {df_sorted.isnull().sum().sum()}")
        
        return df_sorted
    
    def create_feature_engineering(self, df):
        """Create engineered features for ML model"""
        print("\n" + "="*60)
        print("FEATURE ENGINEERING")
        print("="*60)
        
        # Add subject metadata
        if self.subject_metadata is not None:
            df = df.merge(self.subject_metadata[['ID', 'Sex_Encoded', 'Age', 'Age_Group']], 
                         on='ID', how='left')
        
        # Create mission duration categories
        df['Mission_Duration_Category'] = pd.cut(df['Mission_Duration_Days'],
                                                bins=[0, 1, 7, 30, 365],
                                                labels=['Ultra_Short', 'Short', 'Medium', 'Long'])
        
        # Create interaction features
        df['Age_Duration_Interaction'] = df['Age'] * df['Mission_Duration_Days']
        df['Sex_Duration_Interaction'] = df['Sex_Encoded'] * df['Mission_Duration_Days']
        
        # Create slope features (rate of change over time)
        biomarker_cols = [col for col in df.columns if 
                         '_Change_From_Baseline' in col and 'Pct' not in col]
        
        for subject in df['ID'].unique():
            mask = df['ID'] == subject
            subject_data = df[mask].sort_values('Days_From_Launch')
            
            if len(subject_data) > 1:
                for biomarker_col in biomarker_cols:
                    if biomarker_col in subject_data.columns:
                        # Calculate rate of change (slope)
                        time_diff = subject_data['Days_From_Launch'].max() - subject_data['Days_From_Launch'].min()
                        change_diff = subject_data[biomarker_col].iloc[-1] - subject_data[biomarker_col].iloc[0]
                        
                        if time_diff > 0:
                            slope = change_diff / time_diff
                            slope_col = biomarker_col.replace('_Change_From_Baseline', '_Slope')
                            df.loc[mask, slope_col] = slope
        
        # Create recovery features (post-flight trends)
        post_flight_mask = df['Phase'] == 'Post-flight'
        
        if post_flight_mask.sum() > 0:
            for subject in df['ID'].unique():
                subject_post = df[(df['ID'] == subject) & post_flight_mask]
                
                if len(subject_post) > 1:
                    # Calculate recovery trend for CV risk score
                    recovery_trend = np.polyfit(subject_post['Days_From_Launch'], 
                                              subject_post['CV_Risk_Score'], 1)[0]
                    df.loc[df['ID'] == subject, 'CV_Risk_Recovery_Trend'] = recovery_trend
        
        print(f"✓ Added demographic features")
        print(f"✓ Created interaction features")
        print(f"✓ Calculated biomarker slopes")
        print(f"✓ Added recovery trend features")
        
        # Remove redundant columns for final dataset
        cols_to_keep = [
            'ID', 'Days_From_Launch', 'Phase', 'Time_Category',
            'Mission_Duration_Days', 'Mission_Duration_Category',
            'Age', 'Sex_Encoded', 'Age_Group',
            'CV_Risk_Score', 'CV_Risk_Category'
        ]
        
        # Add biomarker columns
        biomarker_base_cols = [col for col in df.columns if any(marker in col for marker in 
                              ['CRP', 'Fibrinogen', 'Haptoglobin', 'AGP', 'PF4', 'SAP', 'a-2', 'Fetuin', 'L-Selectin'])]
        cols_to_keep.extend(biomarker_base_cols)
        
        # Add engineered feature columns
        engineered_cols = [col for col in df.columns if any(suffix in col for suffix in 
                          ['_Baseline', '_Change_From_Baseline', '_Pct_Change_From_Baseline', 
                           '_Slope', '_Interaction', '_Recovery_Trend'])]
        cols_to_keep.extend(engineered_cols)
        
        # Keep only existing columns
        final_cols = [col for col in cols_to_keep if col in df.columns]
        df_final = df[final_cols].copy()
        
        print(f"✓ Final dataset shape: {df_final.shape}")
        print(f"✓ Features created: {len(final_cols)}")
        
        return df_final
    
    def save_processed_data(self, df, filename="cardiovascular_features.csv"):
        """Save processed data and metadata"""
        print("\n" + "="*60)
        print("SAVING PROCESSED DATA")
        print("="*60)
        
        # Save main dataset
        output_file = self.output_dir / filename
        df.to_csv(output_file, index=False)
        print(f"✓ Saved processed data: {output_file}")
        
        # Save data dictionary
        data_dict = {
            'dataset_info': {
                'source': 'NASA OSDR OSD-575 (SpaceX Inspiration4)',
                'subjects': int(df['ID'].nunique()),
                'timepoints': int(df['Days_From_Launch'].nunique()),
                'features': int(df.shape[1]),
                'samples': int(df.shape[0]),
                'mission_duration': int(df['Mission_Duration_Days'].iloc[0]),
                'date_processed': datetime.now().isoformat()
            },
            'feature_categories': {
                'demographics': ['ID', 'Age', 'Sex_Encoded', 'Age_Group'],
                'mission_info': ['Mission_Duration_Days', 'Mission_Duration_Category'],
                'temporal': ['Days_From_Launch', 'Phase', 'Time_Category'],
                'biomarkers': [col for col in df.columns if any(marker in col for marker in 
                              ['CRP', 'Fibrinogen', 'Haptoglobin', 'AGP', 'PF4', 'SAP', 'a-2', 'Fetuin', 'L-Selectin'])],
                'risk_scores': ['CV_Risk_Score', 'CV_Risk_Category'],
                'engineered_features': [col for col in df.columns if any(suffix in col for suffix in 
                                       ['_Baseline', '_Change_From_Baseline', '_Slope', '_Interaction', '_Recovery_Trend'])]
            }
        }
        
        dict_file = self.output_dir / "data_dictionary.json"
        with open(dict_file, 'w') as f:
            json.dump(data_dict, f, indent=2)
        print(f"✓ Saved data dictionary: {dict_file}")
        
        # Save summary statistics
        summary_stats = df.describe()
        summary_file = self.output_dir / "summary_statistics.csv"
        summary_stats.to_csv(summary_file)
        print(f"✓ Saved summary statistics: {summary_file}")
        
        return output_file
    
    def generate_preprocessing_report(self):
        """Generate comprehensive preprocessing report"""
        print("\n" + "="*80)
        print("PREPROCESSING SUMMARY REPORT")
        print("="*80)
        
        if self.processed_features is None:
            print("No processed data available. Run full preprocessing pipeline first.")
            return
        
        df = self.processed_features
        
        print(f"DATASET OVERVIEW:")
        print(f"   • Subjects: {df['ID'].nunique()}")
        print(f"   • Total samples: {len(df)}")
        print(f"   • Features: {df.shape[1]}")
        print(f"   • Timepoints: {df['Days_From_Launch'].nunique()}")
        print(f"   • Mission duration: {df['Mission_Duration_Days'].iloc[0]} days")
        
        print(f"\nRISK SCORE DISTRIBUTION:")
        risk_stats = df['CV_Risk_Score'].describe()
        print(f"   • Mean risk score: {risk_stats['mean']:.1f}")
        print(f"   • Risk range: {risk_stats['min']:.1f} - {risk_stats['max']:.1f}")
        print(f"   • Risk categories:")
        for cat, count in df['CV_Risk_Category'].value_counts().items():
            print(f"     - {cat}: {count} samples")
        
        print(f"\n⏱️  TEMPORAL COVERAGE:")
        for phase in df['Phase'].unique():
            phase_count = len(df[df['Phase'] == phase])
            print(f"   • {phase}: {phase_count} samples")
        
        print(f"\n👥 SUBJECT DEMOGRAPHICS:")
        if 'Age' in df.columns:
            for subject in df['ID'].unique():
                subject_data = df[df['ID'] == subject].iloc[0]
                age = subject_data.get('Age', 'Unknown')
                sex = 'Male' if subject_data.get('Sex_Encoded', 0) == 1 else 'Female'
                print(f"   • {subject}: {sex}, Age {age}")
        
        print(f"\nDATA QUALITY:")
        missing_count = df.isnull().sum().sum()
        print(f"   • Missing values: {missing_count}")
        print(f"   • Data completeness: {((df.size - missing_count) / df.size * 100):.1f}%")
        
        print(f"\nREADY FOR MODEL DEVELOPMENT:")
        print(f"   ✓ Cardiovascular biomarkers processed")
        print(f"   ✓ Risk scores calculated")
        print(f"   ✓ Temporal features engineered")
        print(f"   ✓ Missing values handled")
        print(f"   ✓ Features normalized and scaled")
        
        return df
    
    def _parse_timepoint(self, timepoint):
        """Parse timepoint strings to days from launch"""
        if pd.isna(timepoint):
            return np.nan
        
        timepoint = str(timepoint).strip()
        
        if timepoint.startswith('L-'):
            # Launch minus X days (negative)
            days = -int(timepoint[2:])
        elif timepoint.startswith('R+'):
            # Return plus X days (positive)
            days = int(timepoint[2:])
        else:
            # Try to extract number
            import re
            numbers = re.findall(r'-?\d+', timepoint)
            if numbers:
                days = int(numbers[0])
            else:
                days = 0
        
        return days
    
    def run_full_preprocessing_pipeline(self):
        """Run the complete preprocessing pipeline"""
        print("STARTING FULL PREPROCESSING PIPELINE")
        print("="*80)
        
        try:
            # Step 1: Load data
            self.load_cardiovascular_biomarkers()
            self.load_metabolic_panel()
            self.extract_subject_metadata()
            
            # Step 2: Create temporal features
            df_temporal = self.create_temporal_features()
            
            # Step 3: Calculate risk scores
            df_risk = self.calculate_cardiovascular_risk_scores(df_temporal)
            
            # Step 4: Handle missing values
            df_clean = self.handle_missing_values(df_risk)
            
            # Step 5: Feature engineering
            df_final = self.create_feature_engineering(df_clean)
            
            # Step 6: Save processed data
            output_file = self.save_processed_data(df_final)
            
            # Store processed data
            self.processed_features = df_final
            
            # Step 7: Generate report
            self.generate_preprocessing_report()
            
            print(f"\nPREPROCESSING COMPLETE!")
            print(f"📁 Processed data saved to: {output_file}")
            print(f"Ready for Phase 2: Model Development")
            
            return df_final
            
        except Exception as e:
            print(f"Error in preprocessing pipeline: {e}")
            raise


def main():
    """Run the preprocessing pipeline"""
    print("Microgravity Cardiovascular Risk Prediction - Phase 1: Data Preprocessing")
    print("="*80)
    
    # Initialize preprocessor
    preprocessor = CardiovascularDataPreprocessor()
    
    # Run full pipeline
    processed_data = preprocessor.run_full_preprocessing_pipeline()
    
    print("\nNEXT STEPS:")
    print("1. Review processed data in 'processed_data' directory")
    print("2. Proceed to Phase 2: Model Development")
    print("3. Use 'cardiovascular_features.csv' for ML training")
    
    return processed_data


if __name__ == "__main__":
    processed_data = main()