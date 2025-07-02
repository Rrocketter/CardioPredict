#!/usr/bin/env python3
"""
ml model stuff for cardiovascular risk prediction
week 1 - just basic feature selection and baseline models

trying to predict heart problems in space with earth data too
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression, RFE, RFECV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import joblib

class CardiovascularRiskMLPipeline:
    def __init__(self, processed_data_dir="processed_data", models_dir="models"):
        self.processed_data_dir = Path(processed_data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize data containers
        self.data = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.scaler = None
        self.models = {}
        self.feature_selector = None
        self.selected_features = None
        
        print("Cardiovascular Risk ML Pipeline Initialized")
        print(f"Models will be saved to: {self.models_dir}")
    
    def load_processed_data(self):
        # just load the processed heart data
        print("\n" + "="*60)
        print("LOADING PROCESSED DATA")
        print("="*60)
        
        data_file = self.processed_data_dir / "cardiovascular_features.csv"
        if not data_file.exists():
            raise FileNotFoundError(f"Processed data not found: {data_file}")
        
        self.data = pd.read_csv(data_file)
        print(f"✓ Loaded data: {self.data.shape}")
        
        # Display basic info
        print(f"  • Subjects: {self.data['ID'].nunique()}")
        print(f"  • Timepoints: {self.data['Days_From_Launch'].nunique()}")
        print(f"  • Features: {self.data.shape[1]}")
        print(f"  • Target range: {self.data['CV_Risk_Score'].min():.1f} - {self.data['CV_Risk_Score'].max():.1f}")
        
        return self.data
    
    def prepare_features_and_target(self):
        # just get features and target ready
        print("\n" + "="*60)
        print("PREPARING FEATURES AND TARGET")
        print("="*60)
        
        # Define target variable
        target_col = 'CV_Risk_Score'
        self.y = self.data[target_col].values
        
        # Define features to exclude (non-predictive or target-related)
        exclude_cols = [
            'ID',  # Subject identifier
            'CV_Risk_Score',  # Target variable
            'CV_Risk_Category',  # Derived from target
            'Time_Category',  # Categorical version of temporal info
            'Phase',  # Categorical version of temporal info
            'Age_Group',  # Categorical version of age
            'Mission_Duration_Category'  # Categorical version of mission duration
        ]
        
        # Select feature columns
        feature_cols = [col for col in self.data.columns if col not in exclude_cols]
        self.X = self.data[feature_cols].values
        self.feature_names = feature_cols
        
        print(f"✓ Feature matrix shape: {self.X.shape}")
        print(f"✓ Target vector shape: {self.y.shape}")
        print(f"✓ Features selected: {len(self.feature_names)}")
        
        # Handle any remaining missing values
        if np.isnan(self.X).any():
            print("handling remaining missing values...")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            self.X = imputer.fit_transform(self.X)
            print("missing values imputed")
        
        # Check for infinite values
        if np.isinf(self.X).any():
            print("handling infinite values...")
            self.X = np.nan_to_num(self.X, nan=0.0, posinf=1e10, neginf=-1e10)
            print("infinite values handled")
        
        return self.X, self.y
    
    def perform_feature_selection(self, k_best=20):
        # pick the best features somehow
        print("\n" + "="*60)
        print("FEATURE SELECTION ANALYSIS")
        print("="*60)
        
        # 1. Statistical feature selection (K-Best)
        print("1. Statistical Feature Selection (K-Best):")
        selector_kbest = SelectKBest(score_func=f_regression, k=k_best)
        X_kbest = selector_kbest.fit_transform(self.X, self.y)
        
        # Get selected feature names and scores
        selected_indices = selector_kbest.get_support(indices=True)
        kbest_features = [self.feature_names[i] for i in selected_indices]
        kbest_scores = selector_kbest.scores_[selected_indices]
        
        print(f"   ✓ Selected {len(kbest_features)} features")
        print("   Top 10 features by F-score:")
        for i, (feature, score) in enumerate(sorted(zip(kbest_features, kbest_scores), 
                                                   key=lambda x: x[1], reverse=True)[:10]):
            print(f"     {i+1:2d}. {feature}: {score:.2f}")
        
        # 2. Recursive Feature Elimination with Cross-Validation
        print("\n2. Recursive Feature Elimination (RFE):")
        estimator = RandomForestRegressor(n_estimators=50, random_state=42)
        selector_rfe = RFECV(estimator, step=1, cv=3, scoring='r2', n_jobs=-1)
        X_rfe = selector_rfe.fit_transform(self.X, self.y)
        
        rfe_features = [self.feature_names[i] for i in range(len(self.feature_names)) 
                       if selector_rfe.support_[i]]
        
        print(f"   ✓ Selected {len(rfe_features)} features")
        print(f"   ✓ Optimal number of features: {selector_rfe.n_features_}")
        # Handle different scikit-learn versions
        if hasattr(selector_rfe, 'cv_results_'):
            print(f"   ✓ Cross-validation score: {selector_rfe.cv_results_['mean_test_score'].max():.3f}")
        else:
            print(f"   ✓ RFE feature selection completed")
        
        # 3. Feature importance from Random Forest
        print("\n3. Random Forest Feature Importance:")
        rf_temp = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_temp.fit(self.X, self.y)
        
        feature_importance = list(zip(self.feature_names, rf_temp.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print("   Top 10 features by importance:")
        for i, (feature, importance) in enumerate(feature_importance[:10]):
            print(f"     {i+1:2d}. {feature}: {importance:.4f}")
        
        # 4. Combine selection methods (intersection approach)
        print("\n4. Combined Feature Selection:")
        
        # Get top features from each method
        top_kbest = set(sorted(zip(kbest_features, kbest_scores), 
                              key=lambda x: x[1], reverse=True)[:15])
        top_kbest_names = {x[0] for x in top_kbest}
        
        top_rfe = set(rfe_features[:15] if len(rfe_features) >= 15 else rfe_features)
        
        top_rf = {x[0] for x in feature_importance[:15]}
        
        # Features that appear in at least 2 methods
        consensus_features = []
        for feature in self.feature_names:
            count = sum([
                feature in top_kbest_names,
                feature in top_rfe,
                feature in top_rf
            ])
            if count >= 2:
                consensus_features.append(feature)
        
        # If consensus is too small, take union of top features
        if len(consensus_features) < 10:
            consensus_features = list(top_kbest_names.union(top_rfe).union(top_rf))[:20]
        
        print(f"   ✓ Consensus features (appearing in ≥2 methods): {len(consensus_features)}")
        
        # Store selected features
        self.selected_features = consensus_features
        self.feature_selector = {
            'kbest_features': kbest_features,
            'rfe_features': rfe_features,
            'rf_importance': feature_importance,
            'consensus_features': consensus_features
        }
        
        # Create final feature matrix
        feature_indices = [self.feature_names.index(f) for f in consensus_features]
        self.X_selected = self.X[:, feature_indices]
        
        print(f"\nFinal selected features: {len(consensus_features)}")
        print("Selected features:")
        for i, feature in enumerate(consensus_features, 1):
            print(f"  {i:2d}. {feature}")
        
        return self.X_selected
    
    def setup_cross_validation(self):
        # setup cross validation for time series stuff
        print("\n" + "="*60)
        print("CROSS-VALIDATION SETUP")
        print("="*60)
        
        # Use TimeSeriesSplit for temporal data
        # This ensures future data is not used to predict past
        n_splits = min(5, len(self.data) // 4)  # Adjust based on data size
        
        self.cv_splitter = TimeSeriesSplit(n_splits=n_splits)
        
        print(f"✓ Time-series cross-validation configured")
        print(f"  • Number of splits: {n_splits}")
        print(f"  • Validation strategy: Temporal ordering preserved")
        
        # Visualize the splits
        print(f"\n  Cross-validation splits:")
        for i, (train_idx, val_idx) in enumerate(self.cv_splitter.split(self.X_selected)):
            train_days = self.data.iloc[train_idx]['Days_From_Launch']
            val_days = self.data.iloc[val_idx]['Days_From_Launch']
            print(f"    Fold {i+1}: Train days [{train_days.min():4.0f} to {train_days.max():4.0f}] "
                  f"→ Val days [{val_days.min():4.0f} to {val_days.max():4.0f}]")
        
        return self.cv_splitter
    
    def train_baseline_models(self):
        # train some basic ml models
        print("\n" + "="*60)
        print("BASELINE MODEL TRAINING")
        print("="*60)
        
        # Scale features for algorithms that require it
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.X_selected)
        
        # Define baseline models
        baseline_models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Elastic Net': ElasticNet(alpha=1.0, l1_ratio=0.5),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        
        # Train and evaluate each model
        results = {}
        
        for name, model in baseline_models.items():
            print(f"\n{name}:")
            
            # Use scaled data for linear models, original for tree-based
            X_train = X_scaled if 'Regression' in name or 'Net' in name else self.X_selected
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, X_train, self.y, 
                                      cv=self.cv_splitter, 
                                      scoring='r2',
                                      n_jobs=-1)
            
            # Calculate additional metrics
            cv_mae = -cross_val_score(model, X_train, self.y,
                                    cv=self.cv_splitter,
                                    scoring='neg_mean_absolute_error',
                                    n_jobs=-1)
            
            cv_rmse = np.sqrt(-cross_val_score(model, X_train, self.y,
                                             cv=self.cv_splitter,
                                             scoring='neg_mean_squared_error',
                                             n_jobs=-1))
            
            # Store results
            results[name] = {
                'r2_mean': cv_scores.mean(),
                'r2_std': cv_scores.std(),
                'mae_mean': cv_mae.mean(),
                'mae_std': cv_mae.std(),
                'rmse_mean': cv_rmse.mean(),
                'rmse_std': cv_rmse.std(),
                'model': model
            }
            
            # Print results
            print(f"   R² Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            print(f"   MAE:      {cv_mae.mean():.2f} ± {cv_mae.std():.2f}")
            print(f"   RMSE:     {cv_rmse.mean():.2f} ± {cv_rmse.std():.2f}")
            
            # Train final model on all data for storage
            model.fit(X_train, self.y)
            
        # Store models and results
        self.models = results
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['r2_mean'])
        best_score = results[best_model_name]['r2_mean']
        
        print(f"\nBEST BASELINE MODEL: {best_model_name}")
        print(f"   R² Score: {best_score:.3f}")
        print(f"   MAE:      {results[best_model_name]['mae_mean']:.2f}")
        print(f"   RMSE:     {results[best_model_name]['rmse_mean']:.2f}")
        
        return results
    
    def hyperparameter_optimization(self):
        # tune hyperparameters for better models
        print("\n" + "="*60)
        print("HYPERPARAMETER OPTIMIZATION")
        print("="*60)
        
        # Get top 2 models for optimization
        sorted_models = sorted(self.models.items(), 
                             key=lambda x: x[1]['r2_mean'], 
                             reverse=True)[:2]
        
        optimized_models = {}
        
        for model_name, model_info in sorted_models:
            print(f"\nOptimizing {model_name}:")
            
            base_model = model_info['model']
            
            # Define parameter grids
            if 'Ridge' in model_name:
                param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
            elif 'Lasso' in model_name:
                param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0]}
            elif 'Elastic' in model_name:
                param_grid = {
                    'alpha': [0.1, 1.0, 10.0],
                    'l1_ratio': [0.1, 0.5, 0.9]
                }
            elif 'Random Forest' in model_name:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }
            else:  # Linear Regression
                param_grid = {}
            
            if param_grid:
                # Prepare data
                X_train = (self.scaler.transform(self.X_selected) 
                          if 'Regression' in model_name or 'Net' in model_name 
                          else self.X_selected)
                
                # Grid search
                grid_search = GridSearchCV(
                    base_model.__class__(),
                    param_grid,
                    cv=self.cv_splitter,
                    scoring='r2',
                    n_jobs=-1
                )
                
                grid_search.fit(X_train, self.y)
                
                # Store optimized model
                optimized_models[model_name] = {
                    'model': grid_search.best_estimator_,
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_,
                    'cv_results': grid_search.cv_results_
                }
                
                print(f"   Best parameters: {grid_search.best_params_}")
                print(f"   Best CV score: {grid_search.best_score_:.3f}")
                print(f"   Improvement: {grid_search.best_score_ - model_info['r2_mean']:+.3f}")
            else:
                optimized_models[model_name] = model_info
        
        # Update models with optimized versions
        self.optimized_models = optimized_models
        
        return optimized_models
    
    def analyze_feature_importance(self):
        # see which features matter most
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        # Get best model
        if hasattr(self, 'optimized_models'):
            best_model_name = max(self.optimized_models.keys(), 
                                key=lambda k: self.optimized_models[k].get('best_score', 0))
            best_model = self.optimized_models[best_model_name]['model']
        else:
            best_model_name = max(self.models.keys(), key=lambda k: self.models[k]['r2_mean'])
            best_model = self.models[best_model_name]['model']
        
        print(f"Analyzing feature importance from: {best_model_name}")
        
        # Extract feature importance based on model type
        if hasattr(best_model, 'feature_importances_'):
            # Tree-based models
            importances = best_model.feature_importances_
            importance_type = "Feature Importance"
        elif hasattr(best_model, 'coef_'):
            # Linear models
            importances = np.abs(best_model.coef_)
            importance_type = "Coefficient Magnitude"
        else:
            print("   Model doesn't support feature importance analysis")
            return
        
        # Create importance dataframe
        feature_importance_df = pd.DataFrame({
            'Feature': self.selected_features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print(f"\nTop 10 Features by {importance_type}:")
        for i, (_, row) in enumerate(feature_importance_df.head(10).iterrows(), 1):
            print(f"  {i:2d}. {row['Feature']:<40} {row['Importance']:.4f}")
        
        # Categorize features by type
        feature_categories = {
            'Biomarkers': [],
            'Temporal': [],
            'Demographics': [],
            'Engineered': []
        }
        
        for feature in feature_importance_df.head(15)['Feature']:
            if any(marker in feature for marker in ['CRP', 'Fibrinogen', 'Haptoglobin', 'AGP', 'PF4']):
                feature_categories['Biomarkers'].append(feature)
            elif 'Days_From_Launch' in feature or 'Time' in feature:
                feature_categories['Temporal'].append(feature)
            elif 'Age' in feature or 'Sex' in feature:
                feature_categories['Demographics'].append(feature)
            else:
                feature_categories['Engineered'].append(feature)
        
        print(f"\nFeature Categories in Top 15:")
        for category, features in feature_categories.items():
            if features:
                print(f"  {category}: {len(features)} features")
                for feature in features[:3]:  # Show top 3
                    print(f"    • {feature}")
        
        return feature_importance_df
    
    def save_models_and_results(self):
        # save everything
        print("\n" + "="*60)
        print("SAVING MODELS AND RESULTS")
        print("="*60)
        
        # Save scaler
        scaler_path = self.models_dir / "feature_scaler.joblib"
        joblib.dump(self.scaler, scaler_path)
        print(f"✓ Saved feature scaler: {scaler_path}")
        
        # Save feature selector info
        selector_path = self.models_dir / "feature_selection.json"
        with open(selector_path, 'w') as f:
            # Convert to serializable format
            serializable_selector = {
                'selected_features': self.selected_features,
                'consensus_features': self.feature_selector['consensus_features'],
                'kbest_features': self.feature_selector['kbest_features'],
                'rfe_features': self.feature_selector['rfe_features']
            }
            json.dump(serializable_selector, f, indent=2)
        print(f"✓ Saved feature selection info: {selector_path}")
        
        # Save models
        if hasattr(self, 'optimized_models'):
            models_to_save = self.optimized_models
            results_file = "optimized_models_results.json"
        else:
            models_to_save = self.models
            results_file = "baseline_models_results.json"
        
        # Save individual models
        for name, model_info in models_to_save.items():
            model_filename = name.lower().replace(' ', '_') + '_model.joblib'
            model_path = self.models_dir / model_filename
            
            if 'model' in model_info:
                joblib.dump(model_info['model'], model_path)
                print(f"✓ Saved {name}: {model_path}")
        
        # Save results summary
        results_summary = {}
        for name, info in models_to_save.items():
            results_summary[name] = {
                'r2_score': float(info.get('best_score', info.get('r2_mean', 0))),
                'mae': float(info.get('mae_mean', 0)),
                'rmse': float(info.get('rmse_mean', 0))
            }
            if 'best_params' in info:
                results_summary[name]['best_params'] = info['best_params']
        
        results_path = self.models_dir / results_file
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        print(f"✓ Saved results summary: {results_path}")
        
        return results_summary
    
    def generate_ml_report(self):
        # make a report about how everything went
        print("\n" + "="*80)
        print("MACHINE LEARNING DEVELOPMENT REPORT")
        print("="*80)
        
        # Get best model info
        if hasattr(self, 'optimized_models'):
            best_model_name = max(self.optimized_models.keys(), 
                                key=lambda k: self.optimized_models[k].get('best_score', 0))
            best_score = self.optimized_models[best_model_name].get('best_score', 0)
            models_source = self.optimized_models
        else:
            best_model_name = max(self.models.keys(), key=lambda k: self.models[k]['r2_mean'])
            best_score = self.models[best_model_name]['r2_mean']
            models_source = self.models
        
        print(f"DATASET SUMMARY:")
        print(f"   • Total samples: {len(self.data)}")
        print(f"   • Features selected: {len(self.selected_features)}")
        print(f"   • Target variable: CV_Risk_Score")
        print(f"   • Target range: {self.y.min():.1f} - {self.y.max():.1f}")
        
        print(f"\nMODEL PERFORMANCE:")
        print(f"   • Best model: {best_model_name}")
        print(f"   • R² Score: {best_score:.3f}")
        
        if best_model_name in models_source:
            model_info = models_source[best_model_name]
            if 'mae_mean' in model_info:
                print(f"   • MAE: {model_info['mae_mean']:.2f}")
                print(f"   • RMSE: {model_info['rmse_mean']:.2f}")
        
        print(f"\nFEATURE INSIGHTS:")
        print(f"   • Original features: {len(self.feature_names)}")
        print(f"   • Selected features: {len(self.selected_features)}")
        print(f"   • Feature reduction: {(1 - len(self.selected_features)/len(self.feature_names))*100:.1f}%")
        
        # Performance interpretation
        print(f"\nPERFORMANCE INTERPRETATION:")
        if best_score >= 0.8:
            performance_level = "Excellent"
            interpretation = "Model explains >80% of variance - ready for clinical validation"
        elif best_score >= 0.6:
            performance_level = "Good"
            interpretation = "Model shows strong predictive power - suitable for further development"
        elif best_score >= 0.4:
            performance_level = "Moderate"
            interpretation = "Model shows promise - consider feature engineering or more data"
        else:
            performance_level = "Poor"
            interpretation = "Model needs significant improvement - review features and approach"
        
        print(f"   • Performance level: {performance_level}")
        print(f"   • Interpretation: {interpretation}")
        
        print(f"\nNEXT STEPS:")
        if best_score >= 0.6:
            print(f"   ✓ Ready for Week 2: Advanced models and ensemble methods")
            print(f"   ✓ Consider Neural Networks and Gradient Boosting")
            print(f"   ✓ Proceed with bedrest data integration")
        else:
            print(f"   • Focus on feature engineering improvement")
            print(f"   • Consider additional biomarker interactions")
            print(f"   • Review temporal feature construction")
        
        print(f"\nCLINICAL RELEVANCE:")
        print(f"   • Cardiovascular risk prediction capability established")
        print(f"   • Model suitable for longitudinal monitoring")
        print(f"   • Ready for Earth analog validation (bedrest study)")
        
        return {
            'best_model': best_model_name,
            'best_score': best_score,
            'performance_level': performance_level,
            'selected_features_count': len(self.selected_features)
        }
    
    def run_week1_pipeline(self):
        # just run all the ml stuff for week 1
        print("STARTING WEEK 1: ML MODEL DEVELOPMENT")
        print("="*80)
        
        try:
            # Step 1: Load data
            self.load_processed_data()
            
            # Step 2: Prepare features
            self.prepare_features_and_target()
            
            # Step 3: Feature selection
            self.perform_feature_selection()
            
            # Step 4: Cross-validation setup
            self.setup_cross_validation()
            
            # Step 5: Train baseline models
            self.train_baseline_models()
            
            # Step 6: Hyperparameter optimization
            self.hyperparameter_optimization()
            
            # Step 7: Feature importance analysis
            self.analyze_feature_importance()
            
            # Step 8: Save results
            self.save_models_and_results()
            
            # Step 9: Generate report
            report = self.generate_ml_report()
            
            print(f"\nWEEK 1 COMPLETE!")
            print(f"Baseline models trained and optimized")
            print(f"Feature selection completed")
            print(f"Models saved for future use")
            print(f"Best model: {report['best_model']} (R² = {report['best_score']:.3f})")
            
            return report
            
        except Exception as e:
            print(f"Error in Week 1 pipeline: {e}")
            raise


def main():
    # just run week 1 stuff
    print("Cardiovascular Risk Prediction - Week 1: ML Model Development")
    print("="*80)
    
    # start ml pipeline
    ml_pipeline = CardiovascularRiskMLPipeline()
    
    # run all the week 1 stuff
    report = ml_pipeline.run_week1_pipeline()
    
    print("\nREADY FOR WEEK 2:")
    print("• Advanced ML models (Neural Networks, Gradient Boosting)")
    print("• Ensemble methods")
    print("• Bedrest data integration")
    print("• Cross-domain validation")
    
    return ml_pipeline, report


if __name__ == "__main__":
    ml_pipeline, report = main()
