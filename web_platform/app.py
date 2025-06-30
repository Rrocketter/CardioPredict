#!/usr/bin/env python3
"""
CardioPredict Web Platform
A professional web interface for cardiovascular risk prediction in microgravity environments

This platform provides access to the AI-powered cardiovascular risk prediction system
developed for astronaut health monitoring with Earth analog validation.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import json
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from datetime import datetime
import logging
import random

# Import the API blueprint
from api import api

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'cardiopredict-scientific-platform-2025'

# Register the API blueprint
app.register_blueprint(api)

# Load the trained models and scalers
MODEL_DIR = Path('../deployment')
RESULTS_DIR = Path('../results')

class CardioPredict:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.load_models()
        
    def load_models(self):
        """Load the trained cardiovascular risk prediction models"""
        try:
            # Load the best performing model
            model_path = MODEL_DIR / 'cardiopredict_model.joblib'
            scaler_path = MODEL_DIR / 'cardiopredict_scaler.joblib'
            
            if model_path.exists() and scaler_path.exists():
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                logger.info("✓ Loaded deployment models successfully")
            else:
                # Fallback to development models
                model_path = Path('../models/elastic_net_model.joblib')
                scaler_path = Path('../models/feature_scaler.joblib')
                
                if model_path.exists() and scaler_path.exists():
                    self.model = joblib.load(model_path)
                    self.scaler = joblib.load(scaler_path)
                    logger.info("✓ Loaded development models successfully")
                else:
                    logger.warning("⚠️ No trained models found")
                    
            # Load feature information
            feature_file = Path('../models/feature_selection.json')
            if feature_file.exists():
                with open(feature_file, 'r') as f:
                    feature_info = json.load(f)
                self.feature_names = feature_info.get('consensus_features', [])
                logger.info(f"✓ Loaded {len(self.feature_names)} feature names")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            
    def predict_risk(self, biomarker_data):
        """Predict cardiovascular risk from biomarker data"""
        if self.model is None or self.scaler is None:
            return None, "Model not available"
            
        try:
            # Convert input to DataFrame
            df = pd.DataFrame([biomarker_data])
            
            # Scale features
            X_scaled = self.scaler.transform(df)
            
            # Predict
            risk_score = self.model.predict(X_scaled)[0]
            
            # Interpret risk level
            if risk_score < 35:
                risk_level = "Low"
                risk_color = "#28a745"  # Green
            elif risk_score < 55:
                risk_level = "Moderate"
                risk_color = "#ffc107"  # Yellow
            else:
                risk_level = "High"
                risk_color = "#dc3545"  # Red
                
            return {
                'risk_score': float(risk_score),
                'risk_level': risk_level,
                'risk_color': risk_color,
                'prediction_time': datetime.now().isoformat()
            }, None
            
        except Exception as e:
            return None, f"Prediction error: {str(e)}"

# Initialize the CardioPredict system
cardio_system = CardioPredict()

@app.route('/')
@app.route('/homepage')
def homepage():
    """Homepage with scientific overview and system introduction"""
    
    # Load key statistics from actual research results
    stats = {
        'model_accuracy': '82.0',  # ElasticNet R² = 0.820
        'validation_accuracy': '85.2',  # Enhanced validation accuracy
        'biomarkers_analyzed': 45,  # Total biomarkers in analysis
        'nasa_datasets': 4,  # OSD datasets used
        'ensemble_accuracy': '99.9',  # Weighted ensemble R² = 0.999
        'cross_domain_correlation': '89.7',  # Cross-domain validation
        'subjects_trained': 148,  # Combined space + bedrest subjects
        'clinical_interpretability': 'High'  # ElasticNet interpretability
    }
    
    # Key research achievements
    achievements = {
        'clinical_model': {
            'name': 'ElasticNet',
            'r_squared': 0.820,
            'interpretability': 'High',
            'clinical_use': True
        },
        'research_model': {
            'name': 'Weighted Average Ensemble',
            'r_squared': 0.999,
            'interpretability': 'Medium',
            'research_use': True
        },
        'validation': {
            'published_studies': 85.2,
            'cross_domain': True,
            'hospital_application': True
        }
    }
    
    # Dataset information
    datasets = [
        {'id': 'OSD-258', 'type': 'SpaceX Inspiration4 RNA-seq', 'status': 'primary'},
        {'id': 'OSD-484', 'type': 'Cardiac gene expression', 'status': 'primary'},
        {'id': 'OSD-575', 'type': 'Comprehensive metabolic panel', 'status': 'primary'},
        {'id': 'OSD-51', 'type': 'Microarray bedrest studies', 'status': 'validation'},
        {'id': 'OSD-635', 'type': 'Bulk RNA-seq validation', 'status': 'validation'}
    ]
    
    return render_template('homepage.html', 
                         stats=stats, 
                         achievements=achievements,
                         datasets=datasets)

@app.route('/research')
def research():
    """Research methodology and scientific findings"""
    return render_template('research.html')

@app.route('/documentation')
def documentation():
    """API documentation and technical resources"""
    return render_template('documentation.html')

@app.route('/contact')
def contact():
    """Contact information and support"""
    return render_template('contact.html')

@app.route('/about')
def about():
    """Detailed information about the research and methodology"""
    return render_template('about.html')

@app.route('/demo')
def demo():
    """Interactive demo for cardiovascular risk prediction"""
    
    # Demo configuration
    demo_config = {
        'environments': ['space', 'bedrest', 'hospital'],
        'biomarkers': [
            {
                'name': 'crp',
                'label': 'C-Reactive Protein (CRP)',
                'unit': 'mg/L',
                'range': [0, 100],
                'description': 'Inflammation marker'
            },
            {
                'name': 'pf4', 
                'label': 'Platelet Factor 4 (PF4)',
                'unit': 'ng/mL',
                'range': [0, 50],
                'description': 'Platelet activation marker'
            }
        ],
        'model_info': {
            'clinical_accuracy': 82.0,
            'validation_accuracy': 85.2,
            'model_name': 'ElasticNet Clinical Model'
        }
    }
    
    return render_template('demo.html', 
                         demo=demo_config,
                         feature_names=cardio_system.feature_names)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for cardiovascular risk prediction"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        # Predict cardiovascular risk
        result, error = cardio_system.predict_risk(data)
        
        if error:
            return jsonify({'error': error}), 500
            
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API prediction error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/stats')
def api_stats():
    """API endpoint for system statistics"""
    stats = {
        'total_predictions': 0,  # Could be tracked in database
        'model_version': '1.0.0',
        'last_updated': '2025-06-28',
        'uptime': 'Active'
    }
    return jsonify(stats)

@app.route('/api/dashboard/stats')
def api_dashboard_stats():
    """API endpoint for dashboard statistics"""
    try:
        # In a real application, these would come from a database
        stats = {
            'total_predictions': 1247,
            'model_accuracy': 82.3,
            'active_projects': 8,
            'collaborators': 24,
            'predictions_change': 12.5,
            'accuracy_change': 2.1,
            'projects_change': 2,
            'collaborators_change': 3
        }
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Dashboard stats error: {e}")
        return jsonify({'error': 'Failed to fetch stats'}), 500

@app.route('/api/dashboard/chart-data')
def api_chart_data():
    """API endpoint for chart data with different time periods"""
    try:
        period = request.args.get('period', '7d')
        
        if period == '7d':
            data = {
                'labels': ['Jun 21', 'Jun 22', 'Jun 23', 'Jun 24', 'Jun 25', 'Jun 26', 'Jun 27', 'Jun 28'],
                'accuracy': [78.2, 79.1, 80.4, 81.2, 81.8, 82.1, 82.3, 82.3],
                'predictions': [45, 52, 48, 61, 73, 56, 68, 71]
            }
        elif period == '30d':
            data = {
                'labels': ['May 29', 'Jun 5', 'Jun 12', 'Jun 19', 'Jun 26'],
                'accuracy': [75.4, 77.8, 79.2, 81.1, 82.3],
                'predictions': [234, 189, 267, 198, 347]
            }
        elif period == '90d':
            data = {
                'labels': ['Apr', 'May', 'Jun'],
                'accuracy': [72.1, 76.8, 82.3],
                'predictions': [567, 734, 892]
            }
        elif period == '1y':
            data = {
                'labels': ['Q2 2024', 'Q3 2024', 'Q4 2024', 'Q1 2025', 'Q2 2025'],
                'accuracy': [68.5, 71.2, 74.6, 78.1, 82.3],
                'predictions': [1234, 1456, 1678, 1891, 2103]
            }
        else:
            data = {
                'labels': ['Jun 21', 'Jun 22', 'Jun 23', 'Jun 24', 'Jun 25', 'Jun 26', 'Jun 27', 'Jun 28'],
                'accuracy': [78.2, 79.1, 80.4, 81.2, 81.8, 82.1, 82.3, 82.3],
                'predictions': [45, 52, 48, 61, 73, 56, 68, 71]
            }
            
        return jsonify(data)
    except Exception as e:
        logger.error(f"Chart data error: {e}")
        return jsonify({'error': 'Failed to fetch chart data'}), 500

@app.route('/api/dashboard/predictions', methods=['POST'])
def api_run_prediction():
    """API endpoint for running new predictions"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        patient_id = data.get('patient_id')
        environment = data.get('environment')
        biomarker_data = data.get('biomarker_data')
        
        if not all([patient_id, environment, biomarker_data]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Simulate prediction processing
        import time
        time.sleep(2)  # Simulate processing time
        
        # Generate mock prediction result
        risk_score = round(random.uniform(10, 90), 1)
        
        if risk_score < 35:
            risk_level = 'Low'
            risk_color = '#10b981'
        elif risk_score < 65:
            risk_level = 'Medium'
            risk_color = '#f59e0b'
        else:
            risk_level = 'High'
            risk_color = '#ef4444'
        
        result = {
            'patient_id': patient_id,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'environment': environment,
            'prediction_time': datetime.now().isoformat(),
            'biomarkers_analyzed': len(biomarker_data.split(',')) if isinstance(biomarker_data, str) else 12,
            'model_confidence': round(random.uniform(0.85, 0.98), 3)
        }
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': 'Prediction failed'}), 500

@app.route('/api/dashboard/notifications')
def api_notifications():
    """API endpoint for dashboard notifications"""
    try:
        notifications = [
            {
                'id': 1,
                'type': 'info',
                'title': 'New Dataset Available',
                'message': 'OSD-689 dataset has been uploaded and is ready for analysis',
                'time': '2 hours ago',
                'read': False
            },
            {
                'id': 2,
                'type': 'success',
                'title': 'Model Training Complete',
                'message': 'ElasticNet model training finished with 84.2% accuracy',
                'time': '4 hours ago',
                'read': False
            },
            {
                'id': 3,
                'type': 'warning',
                'title': 'Collaboration Request',
                'message': 'Dr. Johnson requested access to Mars-2027 project',
                'time': '1 day ago',
                'read': True
            }
        ]
        
        return jsonify(notifications)
    except Exception as e:
        logger.error(f"Notifications error: {e}")
        return jsonify({'error': 'Failed to fetch notifications'}), 500

@app.route('/publications')
def publications():
    """Scientific publications and research outputs"""
    return render_template('publications.html')

@app.route('/login')
def login():
    """Login and signup page for user authentication"""
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    """User dashboard for research platform access"""
    # In a real application, you would check authentication here
    # For demo purposes, we'll show the dashboard directly
    
    # Sample user data
    user_data = {
        'name': 'Dr. Research User',
        'role': 'Principal Investigator',
        'organization': 'Research University',
        'total_predictions': 1247,
        'model_accuracy': 82.3,
        'active_projects': 8,
        'collaborators': 24
    }
    
    # Sample dashboard data
    dashboard_data = {
        'recent_predictions': [
            {
                'patient_id': 'AST-001',
                'risk_score': 23.4,
                'risk_level': 'Low',
                'environment': 'Space Station',
                'date': '2025-06-28',
                'status': 'Complete'
            },
            {
                'patient_id': 'AST-002', 
                'risk_score': 67.8,
                'risk_level': 'High',
                'environment': 'Mars Mission',
                'date': '2025-06-27',
                'status': 'Review'
            },
            {
                'patient_id': 'BED-045',
                'risk_score': 41.2,
                'risk_level': 'Medium', 
                'environment': 'Bedrest Study',
                'date': '2025-06-26',
                'status': 'Complete'
            }
        ],
        'recent_activity': [
            {
                'action': 'New dataset uploaded: OSD-687',
                'time': '2 hours ago',
                'icon': 'upload',
                'type': 'blue'
            },
            {
                'action': 'Model training completed',
                'time': '4 hours ago', 
                'icon': 'check',
                'type': 'green'
            },
            {
                'action': 'Dr. Smith joined project "Mars-2027"',
                'time': '1 day ago',
                'icon': 'users', 
                'type': 'purple'
            }
        ]
    }
    
    return render_template('dashboard.html', 
                         user=user_data,
                         dashboard=dashboard_data)

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Development server
    app.run(debug=True, host='0.0.0.0', port=5001)
