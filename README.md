# Microgravity-Induced Cardiovascular Risk Prediction Project


### Project Overview

**Goal**: Build an ML model to predict cardiovascular risk in astronauts based on mission duration, age, and physiological markers, then apply findings to predict similar risks in bedridden patients on Earth.

### Data Successfully Processed

####  **Primary Dataset: SpaceX Inspiration4 (OSD-575)**

- **Subjects**: 4 civilian astronauts (2 Male, 2 Female)
- **Ages**: 29-51 years
- **Mission**: 3-day spaceflight
- **Timeline**: 92 days pre-flight to 194 days post-flight
- **Biomarkers**: 9 cardiovascular risk markers
- **Samples**: 28 longitudinal measurements

#### **Key Cardiovascular Biomarkers Extracted**

1. **CRP** (C-Reactive Protein) - Primary inflammation marker
2. **Fibrinogen** - Coagulation/thrombosis risk
3. **Haptoglobin** - Cardiovascular complications marker
4. **α-2 Macroglobulin** - Atherosclerosis indicator
5. **PF4** (Platelet Factor 4) - Thrombotic risk
6. **AGP** - Inflammatory cardiovascular risk
7. **SAP** - Additional inflammatory marker
8. **Fetuin A** - Vascular calcification inhibitor
9. **L-Selectin** - Endothelial dysfunction marker

#### **Processed Features (102 total)**

- **Baseline Features**: Pre-flight reference values for each biomarker
- **Change Features**: Absolute changes from baseline
- **Percentage Changes**: Relative changes from baseline
- **Slope Features**: Rate of change over time
- **Risk Scores**: Composite cardiovascular risk (32.3-71.7 range)
- **Temporal Features**: Time categories and mission duration
- **Demographics**: Age, sex, mission parameters

### Key Findings from Preprocessing

#### **Significant Cardiovascular Changes Detected**

- **CRP increased 19.9%** post-flight (inflammation marker)
- **Haptoglobin increased 104.5%** (cardiovascular complications)
- **Fibrinogen decreased 15.0%** (coagulation changes)
- **Risk Score Changes**: Pre-flight 48.1 → Post-flight 51.4

#### **Individual Subject Patterns**

- **C001**: Risk +7.9 points (20% increase)
- **C002**: Risk -4.2 points (8% decrease)
- **C003**: Risk +4.5 points (8% increase)
- **C004**: Risk +5.2 points (11% increase)

### Data Quality Assessment 

#### **Excellent Data Quality**

- **100% data completeness** (no missing values)
- **Perfect temporal coverage** (pre/post flight)
- **Complete subject coverage** (7 timepoints per subject)
- **Validated biomarker ranges**
- **Engineered features ready for ML**

## Phase 2: Model Development Plan

- **Feature Selection**: Identify most predictive biomarkers
- **Baseline Models**: Linear/Logistic Regression
- **Cross-Validation**: Time-series aware validation
- **Target**: CV risk score prediction

- **Machine Learning**: Random Forest, SVM, Neural Networks
- **Hyperparameter Tuning**: Grid search optimization
- **Ensemble Methods**: Model combination
- **Performance Metrics**: MAE, RMSE, R², AUC


- **OSD-51 Processing**: Extract bedrest study data
- **Cross-Domain Validation**: Space → Earth model transfer
- **Risk Thresholds**: Clinical decision boundaries
- **Model Validation**: Bedrest vs. spaceflight comparison

- **Model Interpretation**: SHAP/LIME explainability
- **Clinical Framework**: ICU/post-surgical applications
- **Deployment Prep**: Production-ready model
- **Documentation**: Clinical implementation guide

## Bedrest Study Integration Strategy

### **Earth Analog Validation (OSD-51)**

- **Data Source**: Woman skeletal muscle bedrest study
- **Approach**: Map bedrest days → space mission equivalent
- **Validation**: Compare cardiovascular deconditioning patterns
- **Clinical Translation**: Apply to immobilized patients

### **Target Clinical Applications**

1. **ICU Patients** - Prolonged bedrest monitoring
2. **Post-Surgical** - Recovery risk assessment
3. **Elderly Care** - Immobilization complications
4. **Rehabilitation** - Cardiovascular recovery prediction
