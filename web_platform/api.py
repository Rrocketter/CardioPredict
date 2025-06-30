"""
CardioPredict API Module
Handles all backend API endpoints for the dashboard
"""

from flask import Blueprint, jsonify, request, session
import json
import numpy as np
from datetime import datetime, timedelta
import random

# Create API blueprint
api = Blueprint('api', __name__, url_prefix='/api/v1')

# Mock data generators for realistic responses
def generate_prediction_data():
    """Generate realistic cardiovascular prediction data"""
    return {
        'prediction_id': f'PRED-{random.randint(1000, 9999)}',
        'risk_score': round(random.uniform(10, 90), 1),
        'risk_level': random.choice(['Low', 'Medium', 'High']),
        'biomarkers': {
            'crp': round(random.uniform(0.5, 8.0), 2),
            'pf4': round(random.uniform(2.0, 15.0), 2),
            'tnf_alpha': round(random.uniform(1.0, 25.0), 2),
            'il6': round(random.uniform(0.5, 10.0), 2),
            'troponin': round(random.uniform(0.01, 0.5), 3)
        },
        'environment': random.choice(['Space Station', 'Mars Mission', 'Lunar Gateway', 'Bedrest Study', 'Clinical']),
        'confidence': round(random.uniform(75, 98), 1),
        'timestamp': datetime.now().isoformat()
    }

def generate_experiment_data():
    """Generate realistic experiment data"""
    status_options = ['Running', 'Completed', 'Failed', 'Paused']
    return {
        'experiment_id': f'EXP-{random.randint(100, 999)}',
        'name': random.choice([
            'Microgravity Cardiac Adaptation',
            'Bedrest Biomarker Analysis', 
            'Deep Space Radiation Effects',
            'Lunar Surface Cardiovascular Study',
            'Mars Mission Risk Assessment'
        ]),
        'status': random.choice(status_options),
        'progress': random.randint(0, 100),
        'accuracy': round(random.uniform(0.75, 0.98), 3),
        'loss': round(random.uniform(0.01, 0.15), 4),
        'epochs_completed': random.randint(50, 1000),
        'epochs_total': random.randint(500, 1500),
        'start_time': (datetime.now() - timedelta(hours=random.randint(1, 72))).isoformat(),
        'estimated_completion': (datetime.now() + timedelta(hours=random.randint(1, 24))).isoformat()
    }

# Dashboard Statistics Endpoints
@api.route('/stats/overview')
def get_dashboard_stats():
    """Get overall dashboard statistics"""
    return jsonify({
        'total_predictions': random.randint(1200, 1500),
        'model_accuracy': round(random.uniform(80, 95), 1),
        'active_projects': random.randint(6, 12),
        'collaborators': random.randint(20, 30),
        'datasets_processed': random.randint(45, 65),
        'experiments_running': random.randint(2, 8),
        'success_rate': round(random.uniform(92, 99), 1),
        'uptime': round(random.uniform(98, 100), 2)
    })

@api.route('/stats/chart-data')
def get_chart_data():
    """Get data for dashboard charts"""
    # Generate time series data for the last 30 days
    dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30, 0, -1)]
    
    return jsonify({
        'accuracy_trend': {
            'labels': dates,
            'data': [round(random.uniform(75, 95), 1) for _ in dates]
        },
        'prediction_volume': {
            'labels': dates,
            'data': [random.randint(20, 80) for _ in dates]
        },
        'biomarker_distribution': {
            'labels': ['CRP', 'PF4', 'TNF-α', 'IL-6', 'Troponin'],
            'data': [random.randint(15, 35) for _ in range(5)]
        },
        'environment_comparison': {
            'labels': ['Space Station', 'Mars Mission', 'Lunar Gateway', 'Bedrest', 'Clinical'],
            'risk_scores': [round(random.uniform(20, 80), 1) for _ in range(5)]
        }
    })

# Prediction Endpoints
@api.route('/predictions', methods=['GET'])
def get_predictions():
    """Get recent predictions with pagination"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    
    # Generate mock predictions
    predictions = [generate_prediction_data() for _ in range(per_page)]
    
    return jsonify({
        'predictions': predictions,
        'pagination': {
            'page': page,
            'per_page': per_page,
            'total': random.randint(1000, 2000),
            'pages': random.randint(100, 200)
        }
    })

@api.route('/predictions', methods=['POST'])
def create_prediction():
    """Create a new cardiovascular risk prediction"""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    # Validate required fields
    required_fields = ['patient_id', 'biomarkers', 'environment']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    # Generate prediction
    prediction = generate_prediction_data()
    prediction['patient_id'] = data['patient_id']
    prediction['environment'] = data['environment']
    
    # Simulate risk calculation based on biomarkers
    biomarker_values = data.get('biomarkers', {})
    risk_factors = []
    
    if biomarker_values.get('crp', 0) > 3.0:
        risk_factors.append('elevated_crp')
    if biomarker_values.get('tnf_alpha', 0) > 15.0:
        risk_factors.append('elevated_tnf')
    if biomarker_values.get('troponin', 0) > 0.1:
        risk_factors.append('elevated_troponin')
    
    # Adjust risk score based on factors
    base_risk = random.uniform(20, 40)
    risk_multiplier = 1 + (len(risk_factors) * 0.3)
    prediction['risk_score'] = min(round(base_risk * risk_multiplier, 1), 95.0)
    
    if prediction['risk_score'] < 30:
        prediction['risk_level'] = 'Low'
    elif prediction['risk_score'] < 65:
        prediction['risk_level'] = 'Medium'
    else:
        prediction['risk_level'] = 'High'
    
    prediction['risk_factors'] = risk_factors
    
    return jsonify(prediction), 201

# Experiment Endpoints
@api.route('/experiments')
def get_experiments():
    """Get all experiments"""
    experiments = [generate_experiment_data() for _ in range(random.randint(3, 8))]
    
    return jsonify({
        'experiments': experiments,
        'summary': {
            'total': len(experiments),
            'running': len([e for e in experiments if e['status'] == 'Running']),
            'completed': len([e for e in experiments if e['status'] == 'Completed']),
            'failed': len([e for e in experiments if e['status'] == 'Failed'])
        }
    })

@api.route('/experiments', methods=['POST'])
def create_experiment():
    """Create a new experiment"""
    data = request.get_json()
    
    experiment = generate_experiment_data()
    experiment['name'] = data.get('name', 'New Experiment')
    experiment['status'] = 'Running'
    experiment['progress'] = 0
    experiment['start_time'] = datetime.now().isoformat()
    
    return jsonify(experiment), 201

@api.route('/experiments/<experiment_id>/status', methods=['PUT'])
def update_experiment_status(experiment_id):
    """Update experiment status (pause, resume, stop)"""
    data = request.get_json()
    action = data.get('action')
    
    if action not in ['pause', 'resume', 'stop']:
        return jsonify({'error': 'Invalid action'}), 400
    
    status_map = {
        'pause': 'Paused',
        'resume': 'Running', 
        'stop': 'Stopped'
    }
    
    return jsonify({
        'experiment_id': experiment_id,
        'status': status_map[action],
        'timestamp': datetime.now().isoformat()
    })

# Model Endpoints
@api.route('/models')
def get_models():
    """Get available models"""
    models = [
        {
            'model_id': 'cardio-predict-v2.1',
            'name': 'CardioPredict ML Model v2.1',
            'type': 'Random Forest',
            'accuracy': 0.892,
            'precision': 0.847,
            'recall': 0.823,
            'f1_score': 0.835,
            'auc_roc': 0.912,
            'training_date': '2025-06-15T10:30:00Z',
            'status': 'Active',
            'description': 'Advanced cardiovascular risk prediction model for space medicine applications'
        },
        {
            'model_id': 'micrograv-adapt-v1.3',
            'name': 'Microgravity Adaptation Predictor',
            'type': 'Neural Network',
            'accuracy': 0.876,
            'precision': 0.832,
            'recall': 0.801,
            'f1_score': 0.816,
            'auc_roc': 0.894,
            'training_date': '2025-06-20T14:15:00Z',
            'status': 'Active',
            'description': 'Specialized model for predicting cardiovascular adaptation to microgravity environments'
        },
        {
            'model_id': 'bedrest-bio-v1.0',
            'name': 'Bedrest Biomarker Model',
            'type': 'Gradient Boosting',
            'accuracy': 0.853,
            'precision': 0.819,
            'recall': 0.787,
            'f1_score': 0.803,
            'auc_roc': 0.878,
            'training_date': '2025-06-10T09:45:00Z',
            'status': 'Training',
            'description': 'Model optimized for bedrest study biomarker analysis'
        }
    ]
    
    return jsonify({'models': models})

# Dataset Endpoints
@api.route('/datasets')
def get_datasets():
    """Get available datasets"""
    datasets = [
        {
            'dataset_id': 'OSD-258',
            'name': 'NASA OSD-258',
            'type': 'RNA-seq',
            'size': '2.4 GB',
            'samples': 847,
            'features': 20531,
            'last_updated': '2025-06-25T08:30:00Z',
            'status': 'Active',
            'description': 'SpaceX Inspiration4 RNA sequencing data for cardiovascular gene expression analysis'
        },
        {
            'dataset_id': 'OSD-484',
            'name': 'NASA OSD-484',
            'type': 'Microarray',
            'size': '1.8 GB',
            'samples': 624,
            'features': 54675,
            'last_updated': '2025-06-22T14:20:00Z',
            'status': 'Active',
            'description': 'Cardiac gene expression profiles from microgravity exposure studies'
        },
        {
            'dataset_id': 'OSD-575',
            'name': 'Bedrest Study BRS-01',
            'type': 'Clinical',
            'size': '945 MB',
            'samples': 156,
            'features': 847,
            'last_updated': '2025-06-28T11:15:00Z',
            'status': 'Processing',
            'description': '70-day bedrest study biomarker and physiological data'
        },
        {
            'dataset_id': 'OSD-635',
            'name': 'Multi-omics Space Study',
            'type': 'Multi-omics',
            'size': '4.2 GB',
            'samples': 1234,
            'features': 15789,
            'last_updated': '2025-06-20T16:45:00Z',
            'status': 'Active',
            'description': 'Comprehensive multi-omics dataset from long-duration spaceflight studies'
        }
    ]
    
    return jsonify({'datasets': datasets})

# Team Endpoints
@api.route('/team')
def get_team():
    """Get team members"""
    team_members = [
        {
            'user_id': 'sarah.chen',
            'name': 'Dr. Sarah Chen',
            'role': 'Principal Investigator',
            'department': 'Space Medicine',
            'avatar': 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=64&h=64&fit=crop&crop=face',
            'status': 'online',
            'projects': 4,
            'experiments': 23,
            'joined': '2024-01-15T00:00:00Z'
        },
        {
            'user_id': 'michael.rodriguez',
            'name': 'Dr. Michael Rodriguez',
            'role': 'Data Scientist',
            'department': 'Machine Learning',
            'avatar': 'https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=64&h=64&fit=crop&crop=face',
            'status': 'online',
            'projects': 6,
            'experiments': 45,
            'joined': '2024-03-20T00:00:00Z'
        },
        {
            'user_id': 'emily.watson',
            'name': 'Dr. Emily Watson',
            'role': 'Cardiologist',
            'department': 'Clinical Research',
            'avatar': 'https://images.unsplash.com/photo-1494790108755-2616b612b647?w=64&h=64&fit=crop&crop=face',
            'status': 'away',
            'projects': 3,
            'experiments': 18,
            'joined': '2024-02-10T00:00:00Z'
        }
    ]
    
    return jsonify({'team_members': team_members})

# Notification Endpoints
@api.route('/notifications')
def get_notifications():
    """Get user notifications"""
    notifications = [
        {
            'id': 'notif-001',
            'type': 'experiment_complete',
            'title': 'Experiment Completed',
            'message': 'Microgravity Cardiac Adaptation experiment has finished with 94.2% accuracy',
            'timestamp': (datetime.now() - timedelta(hours=2)).isoformat(),
            'read': False,
            'priority': 'high'
        },
        {
            'id': 'notif-002',
            'type': 'data_upload',
            'title': 'Dataset Updated',
            'message': 'NASA OSD-687 dataset has been successfully uploaded and processed',
            'timestamp': (datetime.now() - timedelta(hours=4)).isoformat(),
            'read': False,
            'priority': 'medium'
        },
        {
            'id': 'notif-003',
            'type': 'team_update',
            'title': 'Team Member Added',
            'message': 'Dr. Smith has joined the Mars-2027 project team',
            'timestamp': (datetime.now() - timedelta(days=1)).isoformat(),
            'read': True,
            'priority': 'low'
        }
    ]
    
    return jsonify({'notifications': notifications})

# API Health Check
@api.route('/health')
def health_check():
    """API health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'services': {
            'database': 'connected',
            'ml_service': 'running',
            'data_pipeline': 'active'
        }
    })

# Error handlers
@api.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@api.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500
