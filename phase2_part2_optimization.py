#!/usr/bin/env python3
"""


This module implements hyperparameter optimization and ensemble methods:
- Extensive hyperparameter tuning for promising models
- Advanced ensemble techniques (Voting, Bagging, Stacking)
- Feature engineering enhancements
- Model interpretability analysis

Building on Part 1 results: Sklearn GB (R² = 0.661) vs Baseline Elastic Net (R² = 0.770)
Goal: Beat the baseline through optimization and ensembles
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Advanced ML imports
from sklearn.model_selection import (
    TimeSeriesSplit, cross_val_score, GridSearchCV, RandomizedSearchCV,
    ParameterGrid, validation_curve
)
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor, VotingRegressor, BaggingRegressor,
    ExtraTreesRegressor, AdaBoostRegressor
)
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import joblib
from scipy import stats

# Advanced tree-based models with proper error handling
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("XGBoost available")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    print("LightGBM available")
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
    print("CatBoost available")
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available")

class AdvancedMLOptimizer:
    def __init__(self, processed_data_dir="processed_data", models_dir="models"):
        self.processed_data_dir = Path(processed_data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Load data and setup from Part 1
        self.load_data_and_setup()
        
        # Initialize containers for Part 2
        self.optimized_models = {}
        self.ensemble_models = {}
        self.feature_engineered_data = {}
        
        print("Advanced ML Optimizer Initialized (Part 2/3)")
        print(f"Building on Part 1 results")
    
    def load_data_and_setup(self):
        """Load data and setup from previous phases"""
        print("\n" + "="*70)
        print("LOADING DATA AND PREVIOUS RESULTS")
        print("="*70)
        
        # Load processed data
        data_file = self.processed_data_dir / "cardiovascular_features.csv"
        self.data = pd.read_csv(data_file)
        
        # Load Week 1 selected features
        feature_file = self.models_dir / "feature_selection.json"
        with open(feature_file, 'r') as f:
            feature_info = json.load(f)
        self.selected_features = feature_info['consensus_features']
        
        # Prepare feature matrix
        self.y = self.data['CV_Risk_Score'].values
        self.X = self.data[self.selected_features].values
        
        # Handle missing/infinite values
        if np.isnan(self.X).any():
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            self.X = imputer.fit_transform(self.X)
        
        if np.isinf(self.X).any():
            self.X = np.nan_to_num(self.X, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # Setup cross-validation
        n_splits = min(5, len(self.data) // 6)
        self.cv_splitter = TimeSeriesSplit(n_splits=n_splits)
        
        # Load baseline results
        self.baseline_score = 0.770  # From Week 1 Elastic Net
        
        print(f"✓ Data loaded: {self.X.shape}")
        print(f"✓ Selected features: {len(self.selected_features)}")
        print(f"✓ Baseline to beat: R² = {self.baseline_score:.3f}")
    
    def create_advanced_features(self):
        """Create advanced feature engineering"""
        print("\n" + "="*70)
        print("ADVANCED FEATURE ENGINEERING")
        print("="*70)
        
        feature_sets = {}
        
        # 1. Original features
        feature_sets['original'] = self.X
        print(f"1. Original features: {self.X.shape[1]} features")
        
        # 2. Polynomial features (interaction terms)
        print("2. Creating polynomial interaction features...")
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_poly = poly.fit_transform(self.X)
        
        # Select only interaction terms (not squared terms for interpretability)
        feature_names = poly.get_feature_names_out(self.selected_features)
        interaction_mask = ['*' in name for name in feature_names]
        X_interactions = X_poly[:, interaction_mask]
        
        # Combine original + interactions
        X_enhanced = np.hstack([self.X, X_interactions])
        feature_sets['enhanced'] = X_enhanced
        print(f"   ✓ Enhanced features: {X_enhanced.shape[1]} features")
        print(f"   ✓ Added {X_interactions.shape[1]} interaction terms")
        
        # 3. Statistical features (moving averages, slopes) for temporal data
        if 'Days_From_Launch' in self.data.columns:
            print("3. Creating temporal statistical features...")
            
            # Group by ID and create temporal features
            temporal_features = []
            
            for subject_id in self.data['ID'].unique():
                subject_data = self.data[self.data['ID'] == subject_id].copy()
                subject_data = subject_data.sort_values('Days_From_Launch')
                
                # Calculate slopes for key biomarkers
                if len(subject_data) > 1:
                    days = subject_data['Days_From_Launch'].values
                    
                    for feature in ['CRP', 'Fibrinogen_mg_dl']:  # Key biomarkers
                        if feature in subject_data.columns:
                            values = subject_data[feature].values
                            if len(values) > 1 and not np.isnan(values).all():
                                slope, _, _, _, _ = stats.linregress(days, values)
                                temporal_features.append(slope)
                            else:
                                temporal_features.append(0)
                        else:
                            temporal_features.append(0)
                else:
                    temporal_features.extend([0, 0])  # Add zeros for missing slopes
            
            # Convert to array and reshape
            temporal_array = np.array(temporal_features).reshape(-1, 2)
            
            # Combine with enhanced features
            if temporal_array.shape[0] == X_enhanced.shape[0]:
                X_temporal = np.hstack([X_enhanced, temporal_array])
                feature_sets['temporal'] = X_temporal
                print(f"   ✓ Temporal features: {X_temporal.shape[1]} features")
            else:
                feature_sets['temporal'] = X_enhanced
                print("   Temporal feature mismatch - using enhanced features")
        else:
            feature_sets['temporal'] = feature_sets['enhanced']
        
        # 4. Scaled versions
        scaler = RobustScaler()
        feature_sets['scaled'] = scaler.fit_transform(feature_sets['temporal'])
        print(f"4. Scaled features: {feature_sets['scaled'].shape[1]} features")
        
        self.feature_sets = feature_sets
        print(f"\nCreated {len(feature_sets)} feature sets")
        
        return feature_sets
    
    def optimize_gradient_boosting_models(self):
        """Extensive hyperparameter optimization for gradient boosting models"""
        print("\n" + "="*70)
        print("GRADIENT BOOSTING HYPERPARAMETER OPTIMIZATION")
        print("="*70)
        
        optimization_results = {}
        
        # 1. Scikit-learn Gradient Boosting (extensive tuning)
        print("1. Optimizing Scikit-learn Gradient Boosting:")
        
        gb_param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.15],
            'max_depth': [3, 4, 5, 6, 8],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        gb_model = GradientBoostingRegressor(random_state=42)
        
        # Use RandomizedSearchCV for efficiency
        gb_search = RandomizedSearchCV(
            gb_model, gb_param_grid,
            n_iter=50,  # Try 50 combinations
            cv=self.cv_splitter,
            scoring='r2',
            n_jobs=-1,
            random_state=42
        )
        
        gb_search.fit(self.X, self.y)
        
        optimization_results['GB_Sklearn_Optimized'] = {
            'model': gb_search.best_estimator_,
            'best_params': gb_search.best_params_,
            'best_score': gb_search.best_score_,
            'feature_set': 'original'
        }
        
        print(f"   ✓ Best R² Score: {gb_search.best_score_:.3f}")
        print(f"   ✓ Best parameters: {gb_search.best_params_}")
        
        # 2. XGBoost optimization (if available)
        if XGBOOST_AVAILABLE:
            print("\n2. Optimizing XGBoost:")
            
            xgb_param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 4, 5, 6],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            xgb_model = xgb.XGBRegressor(random_state=42, verbosity=0)
            
            xgb_search = RandomizedSearchCV(
                xgb_model, xgb_param_grid,
                n_iter=30,
                cv=self.cv_splitter,
                scoring='r2',
                n_jobs=-1,
                random_state=42
            )
            
            xgb_search.fit(self.X, self.y)
            
            optimization_results['XGBoost_Optimized'] = {
                'model': xgb_search.best_estimator_,
                'best_params': xgb_search.best_params_,
                'best_score': xgb_search.best_score_,
                'feature_set': 'original'
            }
            
            print(f"   ✓ Best R² Score: {xgb_search.best_score_:.3f}")
            print(f"   ✓ Best parameters: {xgb_search.best_params_}")
        
        # 3. LightGBM optimization (if available)
        if LIGHTGBM_AVAILABLE:
            print("\n3. Optimizing LightGBM:")
            
            lgb_param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 4, 5, 6],
                'min_child_samples': [10, 20, 30],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            lgb_model = lgb.LGBMRegressor(random_state=42, verbosity=-1)
            
            lgb_search = RandomizedSearchCV(
                lgb_model, lgb_param_grid,
                n_iter=30,
                cv=self.cv_splitter,
                scoring='r2',
                n_jobs=-1,
                random_state=42
            )
            
            lgb_search.fit(self.X, self.y)
            
            optimization_results['LightGBM_Optimized'] = {
                'model': lgb_search.best_estimator_,
                'best_params': lgb_search.best_params_,
                'best_score': lgb_search.best_score_,
                'feature_set': 'original'
            }
            
            print(f"   ✓ Best R² Score: {lgb_search.best_score_:.3f}")
            print(f"   ✓ Best parameters: {lgb_search.best_params_}")
        
        # Find best optimized model
        if optimization_results:
            best_model = max(optimization_results.keys(), 
                           key=lambda k: optimization_results[k]['best_score'])
            best_score = optimization_results[best_model]['best_score']
            
            print(f"\nBEST OPTIMIZED MODEL: {best_model}")
            print(f"   R² Score: {best_score:.3f}")
            
            # Compare with baseline
            improvement = best_score - self.baseline_score
            print(f"   Baseline comparison: {improvement:+.3f}")
            
            if improvement > 0:
                print("   SUCCESS: Beat the baseline!")
            else:
                print("   📉 Still below baseline - trying ensembles...")
        
        self.optimized_models.update(optimization_results)
        return optimization_results
    
    def optimize_neural_networks(self):
        """Optimize neural networks with better hyperparameters"""
        print("\n" + "="*70)
        print("NEURAL NETWORK OPTIMIZATION")
        print("="*70)
        
        nn_results = {}
        
        # Simplified architectures for small dataset
        nn_configs = [
            {
                'name': 'Simple_NN_Optimized',
                'hidden_layer_sizes': [(20,), (30,), (50,)],
                'activation': ['relu', 'tanh'],
                'solver': ['lbfgs', 'adam'],
                'alpha': [0.0001, 0.001, 0.01, 0.1],
                'max_iter': [2000, 3000, 5000]  # Increased iterations
            }
        ]
        
        # Use scaled features for neural networks
        X_scaled = RobustScaler().fit_transform(self.X)
        
        for config in nn_configs:
            print(f"\nOptimizing {config['name']}:")
            
            # Create parameter grid
            param_grid = {
                'hidden_layer_sizes': config['hidden_layer_sizes'],
                'activation': config['activation'],
                'solver': config['solver'],
                'alpha': config['alpha'],
                'max_iter': config['max_iter']
            }
            
            nn_model = MLPRegressor(random_state=42)
            
            # Randomized search
            nn_search = RandomizedSearchCV(
                nn_model, param_grid,
                n_iter=20,  # Limited iterations for speed
                cv=self.cv_splitter,
                scoring='r2',
                n_jobs=-1,
                random_state=42
            )
            
            try:
                nn_search.fit(X_scaled, self.y)
                
                nn_results[config['name']] = {
                    'model': nn_search.best_estimator_,
                    'best_params': nn_search.best_params_,
                    'best_score': nn_search.best_score_,
                    'feature_set': 'scaled'
                }
                
                print(f"   ✓ Best R² Score: {nn_search.best_score_:.3f}")
                print(f"   ✓ Best architecture: {nn_search.best_params_['hidden_layer_sizes']}")
                
            except Exception as e:
                print(f"   Error optimizing {config['name']}: {e}")
        
        self.optimized_models.update(nn_results)
        return nn_results
    
    def create_ensemble_models(self):
        """Create advanced ensemble models combining different approaches"""
        print("\n" + "="*70)
        print("ADVANCED ENSEMBLE METHODS")
        print("="*70)
        
        ensemble_results = {}
        
        # Load Week 1 best models for ensemble
        week1_models = {}
        
        # Try to load Week 1 models
        try:
            elastic_net_path = self.models_dir / "elastic_net_model.joblib"
            if elastic_net_path.exists():
                week1_models['elastic_net'] = joblib.load(elastic_net_path)
                print("✓ Loaded Week 1 Elastic Net model")
        except:
            # Create a simple Elastic Net as fallback
            week1_models['elastic_net'] = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
            week1_models['elastic_net'].fit(StandardScaler().fit_transform(self.X), self.y)
            print("✓ Created fallback Elastic Net model")
        
        # Get best optimized models from current session
        current_best = {}
        if self.optimized_models:
            # Get top 2 optimized models
            sorted_models = sorted(self.optimized_models.items(), 
                                 key=lambda x: x[1]['best_score'], 
                                 reverse=True)[:2]
            
            for name, info in sorted_models:
                current_best[name] = info['model']
        
        # 1. Voting Ensemble (Soft voting for regression)
        if len(current_best) >= 2:
            print("\n1. Creating Voting Ensemble:")
            
            estimators = []
            
            # Add Week 1 model with preprocessing
            week1_pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', week1_models['elastic_net'])
            ])
            estimators.append(('elastic_net', week1_pipeline))
            
            # Add current best models
            for name, model in current_best.items():
                estimators.append((name.lower(), model))
            
            voting_ensemble = VotingRegressor(estimators=estimators)
            
            # Cross-validate ensemble
            cv_scores = cross_val_score(
                voting_ensemble, self.X, self.y,
                cv=self.cv_splitter,
                scoring='r2',
                n_jobs=-1
            )
            
            ensemble_results['Voting_Ensemble'] = {
                'model': voting_ensemble,
                'r2_mean': cv_scores.mean(),
                'r2_std': cv_scores.std(),
                'n_estimators': len(estimators)
            }
            
            # Train final ensemble
            voting_ensemble.fit(self.X, self.y)
            
            print(f"   ✓ Voting Ensemble R² Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            print(f"   ✓ Number of base models: {len(estimators)}")
        
        # 2. Bagging Ensemble with different base models
        print("\n2. Creating Bagging Ensembles:")
        
        base_models = [
            ('GB_Bagging', GradientBoostingRegressor(random_state=42)),
            ('ExtraTrees', ExtraTreesRegressor(n_estimators=100, random_state=42)),
        ]
        
        for name, base_model in base_models:
            bagging_model = BaggingRegressor(
                estimator=base_model,  # Updated parameter name
                n_estimators=10,
                random_state=42,
                n_jobs=-1
            )
            
            cv_scores = cross_val_score(
                bagging_model, self.X, self.y,
                cv=self.cv_splitter,
                scoring='r2',
                n_jobs=-1
            )
            
            ensemble_results[name] = {
                'model': bagging_model,
                'r2_mean': cv_scores.mean(),
                'r2_std': cv_scores.std()
            }
            
            # Train final model
            bagging_model.fit(self.X, self.y)
            
            print(f"   ✓ {name} R² Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        # 3. Weighted Average Ensemble
        if len(current_best) >= 2:
            print("\n3. Creating Weighted Average Ensemble:")
            
            # Get individual model predictions
            model_predictions = {}
            model_scores = {}
            
            for name, info in self.optimized_models.items():
                if 'model' in info:
                    model = info['model']
                    
                    # Get cross-validated predictions
                    cv_preds = []
                    cv_scores_local = []
                    
                    for train_idx, val_idx in self.cv_splitter.split(self.X):
                        X_train, X_val = self.X[train_idx], self.X[val_idx]
                        y_train, y_val = self.y[train_idx], self.y[val_idx]
                        
                        model.fit(X_train, y_train)
                        val_pred = model.predict(X_val)
                        
                        cv_preds.extend(val_pred)
                        cv_scores_local.append(r2_score(y_val, val_pred))
                    
                    model_predictions[name] = np.array(cv_preds)
                    model_scores[name] = np.mean(cv_scores_local)
            
            if len(model_predictions) >= 2:
                # Calculate weights based on performance
                total_score = sum(max(0, score) for score in model_scores.values())
                
                if total_score > 0:
                    weights = {name: max(0, score) / total_score 
                             for name, score in model_scores.items()}
                    
                    # Create weighted predictions
                    weighted_pred = np.zeros_like(list(model_predictions.values())[0])
                    for name, pred in model_predictions.items():
                        weighted_pred += weights[name] * pred
                    
                    # Calculate weighted ensemble score
                    weighted_score = r2_score(self.y, weighted_pred)
                    
                    ensemble_results['Weighted_Average'] = {
                        'r2_mean': weighted_score,
                        'weights': weights,
                        'base_models': list(model_predictions.keys())
                    }
                    
                    print(f"   ✓ Weighted Average R² Score: {weighted_score:.3f}")
                    print(f"   ✓ Model weights: {weights}")
        
        # Find best ensemble
        if ensemble_results:
            best_ensemble = max(ensemble_results.keys(), 
                              key=lambda k: ensemble_results[k].get('r2_mean', 0))
            best_score = ensemble_results[best_ensemble]['r2_mean']
            
            print(f"\nBEST ENSEMBLE: {best_ensemble}")
            print(f"   R² Score: {best_score:.3f}")
            
            # Compare with baseline
            improvement = best_score - self.baseline_score
            print(f"   Baseline comparison: {improvement:+.3f}")
            
            if improvement > 0:
                print("   SUCCESS: Ensemble beat the baseline!")
            else:
                print("   📉 Ensemble still below baseline")
        
        self.ensemble_models.update(ensemble_results)
        return ensemble_results
    
    def run_advanced_ml_part2(self):
        """Run Part 2 of advanced ML development"""
        print("STARTING WEEK 2: ADVANCED ML DEVELOPMENT (PART 2/3)")
        print("="*80)
        
        try:
            # Step 1: Advanced feature engineering
            self.create_advanced_features()
            
            # Step 2: Optimize gradient boosting models
            gb_results = self.optimize_gradient_boosting_models()
            
            # Step 3: Optimize neural networks
            nn_results = self.optimize_neural_networks()
            
            # Step 4: Create ensemble models
            ensemble_results = self.create_ensemble_models()
            
            # Step 5: Generate comprehensive results
            summary = self.generate_part2_summary()
            
            print(f"\nPART 2/3 COMPLETE!")
            print(f"Hyperparameter optimization: {len(gb_results) + len(nn_results)} models")
            print(f"Ensemble methods: {len(ensemble_results)} ensembles")
            print(f"Ready for Part 3: Model interpretability & deployment")
            
            return {
                'optimized_models': len(self.optimized_models),
                'ensemble_models': len(self.ensemble_models),
                'best_performance': summary.get('best_score', 0)
            }
            
        except Exception as e:
            print(f"Error in Advanced ML Part 2: {e}")
            raise
    
    def generate_part2_summary(self):
        """Generate comprehensive Part 2 summary"""
        print("\n" + "="*80)
        print("ADVANCED ML PART 2 SUMMARY")
        print("="*80)
        
        all_models = {**self.optimized_models, **self.ensemble_models}
        
        if not all_models:
            print("No models completed successfully")
            return {'best_score': 0}
        
        # Find overall best model
        best_model_name = None
        best_score = -np.inf
        
        for name, info in all_models.items():
            score = info.get('best_score', info.get('r2_mean', 0))
            if score > best_score:
                best_score = score
                best_model_name = name
        
        print(f"OVERALL BEST MODEL: {best_model_name}")
        print(f"   R² Score: {best_score:.3f}")
        
        # Compare with baseline
        baseline_improvement = best_score - self.baseline_score
        print(f"\nBASELINE COMPARISON:")
        print(f"   Week 1 Baseline: Elastic Net (R² = {self.baseline_score:.3f})")
        print(f"   Week 2 Best: {best_model_name} (R² = {best_score:.3f})")
        
        if baseline_improvement > 0:
            print(f"   SUCCESS: +{baseline_improvement:.3f} improvement ({baseline_improvement/self.baseline_score*100:.1f}%)")
            status = "ADVANCED MODELS SUCCESSFUL"
        else:
            print(f"   📉 Gap: {baseline_improvement:.3f} (baseline still better)")
            status = "BASELINE STILL SUPERIOR"
        
        # Model breakdown
        optimized_count = len(self.optimized_models)
        ensemble_count = len(self.ensemble_models)
        
        print(f"\nMODEL SUMMARY:")
        print(f"   Optimized Models: {optimized_count}")
        print(f"   Ensemble Models: {ensemble_count}")
        print(f"   Total Advanced Models: {len(all_models)}")
        
        # Top 3 models
        sorted_models = sorted(all_models.items(), 
                             key=lambda x: x[1].get('best_score', x[1].get('r2_mean', 0)), 
                             reverse=True)[:3]
        
        print(f"\n🥇 TOP 3 MODELS:")
        for i, (name, info) in enumerate(sorted_models, 1):
            score = info.get('best_score', info.get('r2_mean', 0))
            print(f"   {i}. {name}: R² = {score:.3f}")
        
        print(f"\nCLINICAL ASSESSMENT:")
        if best_score >= 0.8:
            clinical_level = "EXCELLENT - Ready for clinical deployment"
        elif best_score >= 0.7:
            clinical_level = "GOOD - Suitable for clinical validation"
        elif best_score >= 0.6:
            clinical_level = "MODERATE - Needs improvement"
        else:
            clinical_level = "POOR - Significant work needed"
        
        print(f"   Performance Level: {clinical_level}")
        print(f"   Status: {status}")
        
        return {
            'best_model': best_model_name,
            'best_score': best_score,
            'baseline_improvement': baseline_improvement,
            'status': status,
            'clinical_level': clinical_level
        }


def main():
    """Run Advanced ML Development Part 2"""
    print("Cardiovascular Risk Prediction - Week 2: Advanced ML Development (Part 2/3)")
    print("="*90)
    
    # Initialize advanced ML optimizer
    optimizer = AdvancedMLOptimizer()
    
    # Run Part 2 of advanced ML development
    results = optimizer.run_advanced_ml_part2()
    
    print("\nREADY FOR PART 3/3:")
    print("• Model interpretability and feature importance")
    print("• Clinical validation metrics")
    print("• Model deployment preparation")
    print("• Final performance comparison")
    
    return optimizer, results


if __name__ == "__main__":
    optimizer, results = main()
