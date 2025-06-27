#!/usr/bin/env python3
"""

This module implements advanced machine learning models including:
- Neural Networks (Multi-layer Perceptron)
- Gradient Boosting (XGBoost, LightGBM, CatBoost)
- Advanced preprocessing and feature engineering
- Model ensemble preparation

Building on strong baseline (Elastic Net R² = 0.770)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Advanced ML imports
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
import joblib

# Advanced tree-based models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available - will skip XGBoost models")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM not available - will skip LightGBM models")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️  CatBoost not available - will skip CatBoost models")

# Deep learning alternative (if tensorflow not available)
from sklearn.ensemble import GradientBoostingRegressor

class AdvancedCardiovascularRiskML:
    def __init__(self, processed_data_dir="processed_data", models_dir="models"):
        self.processed_data_dir = Path(processed_data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize data containers
        self.data = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.selected_features = None
        self.scalers = {}
        self.advanced_models = {}
        self.preprocessing_pipelines = {}
        
        # Load baseline results from Week 1
        self.baseline_results = self.load_baseline_results()
        
        print("� Advanced Cardiovascular Risk ML Pipeline Initialized")
        print(f"Building on Week 1 baseline: {self.baseline_results.get('best_model', 'Unknown')}")
        print(f"Advanced models will be saved to: {self.models_dir}")
    
    def load_baseline_results(self):
        """Load Week 1 baseline results for comparison"""
        try:
            results_file = self.models_dir / "optimized_models_results.json"
            if not results_file.exists():
                results_file = self.models_dir / "baseline_models_results.json"
            
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                # Find best baseline model
                best_model = max(results.keys(), key=lambda k: results[k]['r2_score'])
                best_score = results[best_model]['r2_score']
                
                print(f"✓ Loaded Week 1 results: {best_model} (R² = {best_score:.3f})")
                return {'best_model': best_model, 'best_score': best_score, 'all_results': results}
            else:
                print("⚠️  No baseline results found - will establish new baseline")
                return {'best_model': 'None', 'best_score': 0.0}
        except Exception as e:
            print(f"⚠️  Error loading baseline results: {e}")
            return {'best_model': 'None', 'best_score': 0.0}
    
    def load_processed_data_and_features(self):
        """Load processed data and selected features from Week 1"""
        print("\n" + "="*70)
        print("LOADING DATA AND WEEK 1 FEATURES")
        print("="*70)
        
        # Load processed data
        data_file = self.processed_data_dir / "cardiovascular_features.csv"
        if not data_file.exists():
            raise FileNotFoundError(f"Processed data not found: {data_file}")
        
        self.data = pd.read_csv(data_file)
        
        # Load selected features from Week 1
        feature_file = self.models_dir / "feature_selection.json"
        if feature_file.exists():
            with open(feature_file, 'r') as f:
                feature_info = json.load(f)
            self.selected_features = feature_info['consensus_features']
            print(f"✓ Loaded Week 1 selected features: {len(self.selected_features)}")
        else:
            print("⚠️  No feature selection from Week 1 found - will perform new selection")
            self.selected_features = None
        
        print(f"✓ Loaded data: {self.data.shape}")
        print(f"  • Subjects: {self.data['ID'].nunique()}")
        print(f"  • Timepoints: {self.data['Days_From_Launch'].nunique()}")
        
        return self.data
    
    def prepare_advanced_features(self):
        """Prepare features with advanced preprocessing options"""
        print("\n" + "="*70)
        print("ADVANCED FEATURE PREPARATION")
        print("="*70)
        
        # Define target
        self.y = self.data['CV_Risk_Score'].values
        
        # Use Week 1 selected features if available, otherwise select all relevant features
        if self.selected_features:
            # Validate that selected features exist in data
            available_features = [f for f in self.selected_features if f in self.data.columns]
            if len(available_features) != len(self.selected_features):
                missing = set(self.selected_features) - set(available_features)
                print(f"⚠️  Missing features from Week 1: {missing}")
            
            self.selected_features = available_features
            feature_cols = self.selected_features
        else:
            # Select features manually
            exclude_cols = [
                'ID', 'CV_Risk_Score', 'CV_Risk_Category', 
                'Time_Category', 'Phase', 'Age_Group', 'Mission_Duration_Category'
            ]
            feature_cols = [col for col in self.data.columns if col not in exclude_cols]
            self.selected_features = feature_cols
        
        # Create feature matrix
        self.X = self.data[feature_cols].values
        self.feature_names = feature_cols
        
        print(f"✓ Feature matrix: {self.X.shape}")
        print(f"✓ Selected features: {len(self.selected_features)}")
        
        # Handle missing values
        if np.isnan(self.X).any():
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            self.X = imputer.fit_transform(self.X)
            print("✓ Missing values imputed")
        
        # Handle infinite values
        if np.isinf(self.X).any():
            self.X = np.nan_to_num(self.X, nan=0.0, posinf=1e10, neginf=-1e10)
            print("✓ Infinite values handled")
        
        return self.X, self.y
    
    def create_advanced_preprocessing_pipelines(self):
        """Create multiple preprocessing pipelines for different model types"""
        print("\n" + "="*70)
        print("ADVANCED PREPROCESSING PIPELINES")
        print("="*70)
        
        # 1. Standard Scaling (for Neural Networks)
        print("1. Standard Scaling Pipeline (Neural Networks):")
        self.scalers['standard'] = StandardScaler()
        X_standard = self.scalers['standard'].fit_transform(self.X)
        print(f"   ✓ Standard scaled features: mean≈0, std≈1")
        
        # 2. Robust Scaling (for outlier-resistant models)
        print("2. Robust Scaling Pipeline (Outlier-resistant):")
        self.scalers['robust'] = RobustScaler()
        X_robust = self.scalers['robust'].fit_transform(self.X)
        print(f"   ✓ Robust scaled features: median=0, IQR-based scaling")
        
        # 3. MinMax Scaling (for bounded algorithms)
        print("3. MinMax Scaling Pipeline (Bounded algorithms):")
        self.scalers['minmax'] = MinMaxScaler()
        X_minmax = self.scalers['minmax'].fit_transform(self.X)
        print(f"   ✓ MinMax scaled features: range [0,1]")
        
        # 4. Quantile Transformation (for non-linear relationships)
        print("4. Quantile Transformation (Non-linear relationships):")
        self.scalers['quantile'] = QuantileTransformer(output_distribution='normal')
        X_quantile = self.scalers['quantile'].fit_transform(self.X)
        print(f"   ✓ Quantile transformed features: Gaussian-like distribution")
        
        # Store preprocessed data
        self.preprocessed_data = {
            'standard': X_standard,
            'robust': X_robust,
            'minmax': X_minmax,
            'quantile': X_quantile,
            'original': self.X  # Keep original for tree-based models
        }
        
        print(f"\n✅ Created {len(self.preprocessed_data)} preprocessing pipelines")
        return self.preprocessed_data
    
    def setup_advanced_cross_validation(self):
        """Setup advanced cross-validation with temporal awareness"""
        print("\n" + "="*70)
        print("ADVANCED CROSS-VALIDATION SETUP")
        print("="*70)
        
        # Time-series split with more sophisticated approach
        n_splits = min(5, len(self.data) // 6)  # More conservative splits for advanced models
        self.cv_splitter = TimeSeriesSplit(n_splits=n_splits)
        
        # Add temporal features for better validation
        if 'Days_From_Launch' in self.data.columns:
            days = self.data['Days_From_Launch'].values
            print(f"✓ Temporal range: {days.min():.0f} to {days.max():.0f} days")
            
            # Analyze temporal distribution
            pre_flight = np.sum(days < 0)
            in_flight = np.sum((days >= 0) & (days <= 365))  # Assuming max 1 year missions
            post_flight = np.sum(days > 365)
            
            print(f"  • Pre-flight samples: {pre_flight}")
            print(f"  • In-flight samples: {in_flight}")
            print(f"  • Post-flight samples: {post_flight}")
        
        print(f"✓ Advanced time-series CV: {n_splits} splits")
        print(f"  • Strategy: Temporal order preserved")
        print(f"  • Gap handling: Automatic")
        
        return self.cv_splitter
    
    def train_neural_networks(self):
        """Train Multi-layer Perceptron neural networks with different architectures"""
        print("\n" + "="*70)
        print("NEURAL NETWORK TRAINING")
        print("="*70)
        
        # Different NN architectures to test
        nn_architectures = {
            'Small NN': {
                'hidden_layer_sizes': (50,),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.001,
                'max_iter': 1000,
                'random_state': 42
            },
            'Medium NN': {
                'hidden_layer_sizes': (100, 50),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.001,
                'max_iter': 1000,
                'random_state': 42
            },
            'Deep NN': {
                'hidden_layer_sizes': (100, 50, 25),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.001,
                'max_iter': 1000,
                'random_state': 42
            },
            'Wide NN': {
                'hidden_layer_sizes': (200, 100),
                'activation': 'relu',
                'solver': 'lbfgs',  # Better for small datasets
                'alpha': 0.001,
                'max_iter': 1000,
                'random_state': 42
            }
        }
        
        nn_results = {}
        
        # Test different preprocessing for NNs
        preprocessing_methods = ['standard', 'robust', 'quantile']
        
        for prep_method in preprocessing_methods:
            print(f"\n--- Neural Networks with {prep_method.upper()} preprocessing ---")
            X_prep = self.preprocessed_data[prep_method]
            
            for nn_name, nn_params in nn_architectures.items():
                print(f"\n{nn_name} ({prep_method} preprocessing):")
                
                # Create model
                model = MLPRegressor(**nn_params)
                
                # Cross-validation
                try:
                    cv_scores = cross_val_score(
                        model, X_prep, self.y,
                        cv=self.cv_splitter,
                        scoring='r2',
                        n_jobs=-1
                    )
                    
                    cv_mae = -cross_val_score(
                        model, X_prep, self.y,
                        cv=self.cv_splitter,
                        scoring='neg_mean_absolute_error',
                        n_jobs=-1
                    )
                    
                    cv_rmse = np.sqrt(-cross_val_score(
                        model, X_prep, self.y,
                        cv=self.cv_splitter,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1
                    ))
                    
                    # Store results
                    model_key = f"{nn_name}_{prep_method}"
                    nn_results[model_key] = {
                        'model': model,
                        'preprocessing': prep_method,
                        'architecture': nn_params['hidden_layer_sizes'],
                        'r2_mean': cv_scores.mean(),
                        'r2_std': cv_scores.std(),
                        'mae_mean': cv_mae.mean(),
                        'mae_std': cv_mae.std(),
                        'rmse_mean': cv_rmse.mean(),
                        'rmse_std': cv_rmse.std()
                    }
                    
                    print(f"   R² Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
                    print(f"   MAE:      {cv_mae.mean():.2f} ± {cv_mae.std():.2f}")
                    print(f"   RMSE:     {cv_rmse.mean():.2f} ± {cv_rmse.std():.2f}")
                    
                    # Train final model
                    model.fit(X_prep, self.y)
                    
                except Exception as e:
                    print(f"   ❌ Error training {nn_name}: {e}")
                    continue
        
        # Find best NN model
        if nn_results:
            best_nn = max(nn_results.keys(), key=lambda k: nn_results[k]['r2_mean'])
            best_score = nn_results[best_nn]['r2_mean']
            
            print(f"\n� BEST NEURAL NETWORK: {best_nn}")
            print(f"   R² Score: {best_score:.3f}")
            print(f"   Architecture: {nn_results[best_nn]['architecture']}")
            print(f"   Preprocessing: {nn_results[best_nn]['preprocessing']}")
            
            # Compare with baseline
            baseline_score = self.baseline_results.get('best_score', 0.0)
            improvement = best_score - baseline_score
            print(f"   Improvement over baseline: {improvement:+.3f}")
        
        self.advanced_models.update(nn_results)
        return nn_results
    
    def train_gradient_boosting_models(self):
        """Train gradient boosting models (XGBoost, LightGBM, CatBoost, Sklearn)"""
        print("\n" + "="*70)
        print("GRADIENT BOOSTING MODELS")
        print("="*70)
        
        boosting_results = {}
        
        # 1. Scikit-learn Gradient Boosting (always available)
        print("1. Scikit-learn Gradient Boosting Regressor:")
        gb_sklearn = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        try:
            cv_scores = cross_val_score(
                gb_sklearn, self.X, self.y,
                cv=self.cv_splitter,
                scoring='r2',
                n_jobs=-1
            )
            
            cv_mae = -cross_val_score(
                gb_sklearn, self.X, self.y,
                cv=self.cv_splitter,
                scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
            
            cv_rmse = np.sqrt(-cross_val_score(
                gb_sklearn, self.X, self.y,
                cv=self.cv_splitter,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            ))
            
            boosting_results['Gradient_Boosting_Sklearn'] = {
                'model': gb_sklearn,
                'r2_mean': cv_scores.mean(),
                'r2_std': cv_scores.std(),
                'mae_mean': cv_mae.mean(),
                'mae_std': cv_mae.std(),
                'rmse_mean': cv_rmse.mean(),
                'rmse_std': cv_rmse.std()
            }
            
            print(f"   R² Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            print(f"   MAE:      {cv_mae.mean():.2f} ± {cv_mae.std():.2f}")
            print(f"   RMSE:     {cv_rmse.mean():.2f} ± {cv_rmse.std():.2f}")
            
            # Train final model
            gb_sklearn.fit(self.X, self.y)
            
        except Exception as e:
            print(f"   ❌ Error with Sklearn GB: {e}")
        
        # 2. XGBoost (if available)
        if XGBOOST_AVAILABLE:
            print("\n2. XGBoost Regressor:")
            
            xgb_model = xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                eval_metric='rmse',
                verbosity=0
            )
            
            try:
                cv_scores = cross_val_score(
                    xgb_model, self.X, self.y,
                    cv=self.cv_splitter,
                    scoring='r2',
                    n_jobs=-1
                )
                
                cv_mae = -cross_val_score(
                    xgb_model, self.X, self.y,
                    cv=self.cv_splitter,
                    scoring='neg_mean_absolute_error',
                    n_jobs=-1
                )
                
                cv_rmse = np.sqrt(-cross_val_score(
                    xgb_model, self.X, self.y,
                    cv=self.cv_splitter,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                ))
                
                boosting_results['XGBoost'] = {
                    'model': xgb_model,
                    'r2_mean': cv_scores.mean(),
                    'r2_std': cv_scores.std(),
                    'mae_mean': cv_mae.mean(),
                    'mae_std': cv_mae.std(),
                    'rmse_mean': cv_rmse.mean(),
                    'rmse_std': cv_rmse.std()
                }
                
                print(f"   R² Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
                print(f"   MAE:      {cv_mae.mean():.2f} ± {cv_mae.std():.2f}")
                print(f"   RMSE:     {cv_rmse.mean():.2f} ± {cv_rmse.std():.2f}")
                
                # Train final model
                xgb_model.fit(self.X, self.y)
                
            except Exception as e:
                print(f"   ❌ Error with XGBoost: {e}")
        
        # 3. LightGBM (if available)
        if LIGHTGBM_AVAILABLE:
            print("\n3. LightGBM Regressor:")
            
            lgb_model = lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbosity=-1
            )
            
            try:
                cv_scores = cross_val_score(
                    lgb_model, self.X, self.y,
                    cv=self.cv_splitter,
                    scoring='r2',
                    n_jobs=-1
                )
                
                cv_mae = -cross_val_score(
                    lgb_model, self.X, self.y,
                    cv=self.cv_splitter,
                    scoring='neg_mean_absolute_error',
                    n_jobs=-1
                )
                
                cv_rmse = np.sqrt(-cross_val_score(
                    lgb_model, self.X, self.y,
                    cv=self.cv_splitter,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                ))
                
                boosting_results['LightGBM'] = {
                    'model': lgb_model,
                    'r2_mean': cv_scores.mean(),
                    'r2_std': cv_scores.std(),
                    'mae_mean': cv_mae.mean(),
                    'mae_std': cv_mae.std(),
                    'rmse_mean': cv_rmse.mean(),
                    'rmse_std': cv_rmse.std()
                }
                
                print(f"   R² Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
                print(f"   MAE:      {cv_mae.mean():.2f} ± {cv_mae.std():.2f}")
                print(f"   RMSE:     {cv_rmse.mean():.2f} ± {cv_rmse.std():.2f}")
                
                # Train final model
                lgb_model.fit(self.X, self.y)
                
            except Exception as e:
                print(f"   ❌ Error with LightGBM: {e}")
        
        # 4. CatBoost (if available)
        if CATBOOST_AVAILABLE:
            print("\n4. CatBoost Regressor:")
            
            cat_model = cb.CatBoostRegressor(
                iterations=200,
                learning_rate=0.1,
                depth=6,
                random_state=42,
                verbose=False
            )
            
            try:
                cv_scores = cross_val_score(
                    cat_model, self.X, self.y,
                    cv=self.cv_splitter,
                    scoring='r2',
                    n_jobs=-1
                )
                
                cv_mae = -cross_val_score(
                    cat_model, self.X, self.y,
                    cv=self.cv_splitter,
                    scoring='neg_mean_absolute_error',
                    n_jobs=-1
                )
                
                cv_rmse = np.sqrt(-cross_val_score(
                    cat_model, self.X, self.y,
                    cv=self.cv_splitter,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                ))
                
                boosting_results['CatBoost'] = {
                    'model': cat_model,
                    'r2_mean': cv_scores.mean(),
                    'r2_std': cv_scores.std(),
                    'mae_mean': cv_mae.mean(),
                    'mae_std': cv_mae.std(),
                    'rmse_mean': cv_rmse.mean(),
                    'rmse_std': cv_rmse.std()
                }
                
                print(f"   R² Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
                print(f"   MAE:      {cv_mae.mean():.2f} ± {cv_mae.std():.2f}")
                print(f"   RMSE:     {cv_rmse.mean():.2f} ± {cv_rmse.std():.2f}")
                
                # Train final model
                cat_model.fit(self.X, self.y)
                
            except Exception as e:
                print(f"   ❌ Error with CatBoost: {e}")
        
        # Find best boosting model
        if boosting_results:
            best_boosting = max(boosting_results.keys(), key=lambda k: boosting_results[k]['r2_mean'])
            best_score = boosting_results[best_boosting]['r2_mean']
            
            print(f"\n� BEST BOOSTING MODEL: {best_boosting}")
            print(f"   R² Score: {best_score:.3f}")
            
            # Compare with baseline
            baseline_score = self.baseline_results.get('best_score', 0.0)
            improvement = best_score - baseline_score
            print(f"   Improvement over baseline: {improvement:+.3f}")
        
        self.advanced_models.update(boosting_results)
        return boosting_results
    
    def run_advanced_ml_part1(self):
        """Run first third of advanced ML development"""
        print("� STARTING WEEK 2: ADVANCED ML DEVELOPMENT (PART 1/3)")
        print("="*80)
        
        try:
            # Step 1: Load data and features
            self.load_processed_data_and_features()
            
            # Step 2: Advanced feature preparation
            self.prepare_advanced_features()
            
            # Step 3: Create preprocessing pipelines
            self.create_advanced_preprocessing_pipelines()
            
            # Step 4: Setup advanced cross-validation
            self.setup_advanced_cross_validation()
            
            # Step 5: Train neural networks
            nn_results = self.train_neural_networks()
            
            # Step 6: Train gradient boosting models
            boosting_results = self.train_gradient_boosting_models()
            
            # Step 7: Preliminary results summary
            self.generate_part1_summary()
            
            print(f"\n� PART 1/3 COMPLETE!")
            print(f"✅ Neural Networks trained: {len(nn_results)} models")
            print(f"✅ Gradient Boosting trained: {len(boosting_results)} models")
            print(f"✅ Ready for Part 2: Hyperparameter optimization & ensembles")
            
            return {
                'neural_networks': nn_results,
                'gradient_boosting': boosting_results,
                'total_models': len(self.advanced_models)
            }
            
        except Exception as e:
            print(f"❌ Error in Advanced ML Part 1: {e}")
            raise
    
    def generate_part1_summary(self):
        """Generate summary of Part 1 results"""
        print("\n" + "="*80)
        print("ADVANCED ML PART 1 SUMMARY")
        print("="*80)
        
        if not self.advanced_models:
            print("No models trained successfully")
            return
        
        # Get all results
        all_results = self.advanced_models
        
        # Find overall best model
        best_model_name = max(all_results.keys(), key=lambda k: all_results[k]['r2_mean'])
        best_score = all_results[best_model_name]['r2_mean']
        best_model_info = all_results[best_model_name]
        
        print(f"� BEST ADVANCED MODEL SO FAR: {best_model_name}")
        print(f"   R² Score: {best_score:.3f} ± {best_model_info['r2_std']:.3f}")
        print(f"   MAE:      {best_model_info['mae_mean']:.2f} ± {best_model_info['mae_std']:.2f}")
        print(f"   RMSE:     {best_model_info['rmse_mean']:.2f} ± {best_model_info['rmse_std']:.2f}")
        
        # Compare with baseline
        baseline_score = self.baseline_results.get('best_score', 0.0)
        baseline_name = self.baseline_results.get('best_model', 'Unknown')
        
        print(f"\n� COMPARISON WITH WEEK 1 BASELINE:")
        print(f"   Week 1 Best: {baseline_name} (R² = {baseline_score:.3f})")
        print(f"   Week 2 Best: {best_model_name} (R² = {best_score:.3f})")
        
        improvement = best_score - baseline_score
        if improvement > 0:
            print(f"   � IMPROVEMENT: +{improvement:.3f} ({improvement/baseline_score*100:.1f}%)")
        else:
            print(f"   � Performance: {improvement:.3f} (baseline still better)")
        
        # Model type breakdown
        nn_models = [k for k in all_results.keys() if 'NN' in k]
        boosting_models = [k for k in all_results.keys() if any(boost in k for boost in ['XGBoost', 'LightGBM', 'CatBoost', 'Gradient_Boosting'])]
        
        print(f"\n� MODEL BREAKDOWN:")
        print(f"   Neural Networks: {len(nn_models)} models")
        print(f"   Gradient Boosting: {len(boosting_models)} models")
        print(f"   Total Advanced Models: {len(all_results)}")
        
        # Top 3 models
        top_3 = sorted(all_results.items(), key=lambda x: x[1]['r2_mean'], reverse=True)[:3]
        print(f"\n� TOP 3 MODELS:")
        for i, (name, results) in enumerate(top_3, 1):
            print(f"   {i}. {name}: R² = {results['r2_mean']:.3f}")
        
        print(f"\n� NEXT STEPS (PART 2/3):")
        print(f"   • Hyperparameter optimization for top models")
        print(f"   • Advanced ensemble methods")
        print(f"   • Feature importance analysis")
        print(f"   • Model interpretability")
        
        return {
            'best_model': best_model_name,
            'best_score': best_score,
            'improvement': improvement,
            'total_models': len(all_results)
        }


def main():
    """Run Advanced ML Development Part 1"""
    print("Cardiovascular Risk Prediction - Week 2: Advanced ML Development (Part 1/3)")
    print("="*90)
    
    # Initialize advanced ML pipeline
    advanced_ml = AdvancedCardiovascularRiskML()
    
    # Run Part 1 of advanced ML development
    results = advanced_ml.run_advanced_ml_part1()
    
    print("\n� READY FOR PART 2/3:")
    print("• Hyperparameter optimization")
    print("• Advanced ensemble methods")
    print("• Model stacking and blending")
    print("• Feature importance analysis")
    
    return advanced_ml, results


if __name__ == "__main__":
    advanced_ml, results = main()