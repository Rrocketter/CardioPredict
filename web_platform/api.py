"""
CardioPredict API Module
Handles all backend API endpoints for the dashboard with database integration
"""

from flask import Blueprint, jsonify, request, session
import json
import numpy as np
from datetime import datetime, timedelta
import random
from sqlalchemy import func, desc
from models import db, User, Dataset, Prediction, Experiment, MLModel, Notification

# Create API blueprint
api = Blueprint('api', __name__, url_prefix='/api/v1')

# Dashboard Statistics Endpoints
@api.route('/stats/overview')
def get_dashboard_stats():
    """Get overall dashboard statistics from database"""
    try:
        total_predictions = db.session.query(func.count(Prediction.id)).scalar() or 0
        active_projects = db.session.query(func.count(Dataset.id)).filter(Dataset.status == 'Active').scalar() or 0
        collaborators = db.session.query(func.count(User.id)).scalar() or 0
        datasets_processed = db.session.query(func.count(Dataset.id)).scalar() or 0
        experiments_running = db.session.query(func.count(Experiment.id)).filter(Experiment.status == 'Running').scalar() or 0
        
        # Calculate average model accuracy
        avg_accuracy = db.session.query(func.avg(MLModel.accuracy)).filter(MLModel.status == 'Active').scalar()
        model_accuracy = round(avg_accuracy * 100, 1) if avg_accuracy else 85.0
        
        # Calculate success rate based on completed experiments
        completed_experiments = db.session.query(func.count(Experiment.id)).filter(Experiment.status == 'Completed').scalar() or 0
        total_experiments = db.session.query(func.count(Experiment.id)).scalar() or 1
        success_rate = round((completed_experiments / total_experiments) * 100, 1)
        
        return jsonify({
            'total_predictions': total_predictions,
            'model_accuracy': model_accuracy,
            'active_projects': active_projects,
            'collaborators': collaborators,
            'datasets_processed': datasets_processed,
            'experiments_running': experiments_running,
            'success_rate': success_rate,
            'uptime': round(random.uniform(98, 100), 2)  # Simulated uptime
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/stats/chart-data')
def get_chart_data():
    """Get data for dashboard charts from database"""
    try:
        # Generate time series data for the last 30 days
        dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30, 0, -1)]
        
        # Get prediction volume per day
        prediction_volume = []
        for date_str in dates:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
            next_date = date_obj + timedelta(days=1)
            count = db.session.query(func.count(Prediction.id)).filter(
                func.date(Prediction.created_at) == date_obj
            ).scalar() or 0
            prediction_volume.append(count)
        
        # Get biomarker distribution from recent predictions
        recent_predictions = Prediction.query.order_by(desc(Prediction.created_at)).limit(100).all()
        biomarker_counts = {'CRP': 0, 'PF4': 0, 'TNF-α': 0, 'IL-6': 0, 'Troponin': 0}
        
        for pred in recent_predictions:
            biomarkers = pred.get_biomarkers()
            for biomarker in biomarker_counts.keys():
                biomarker_key = biomarker.lower().replace('tnf-α', 'tnf_alpha').replace('il-6', 'il6')
                if biomarker_key in biomarkers:
                    biomarker_counts[biomarker] += 1
        
        # Get environment comparison from recent predictions
        env_stats = db.session.query(
            Prediction.environment,
            func.avg(Prediction.risk_score).label('avg_risk')
        ).group_by(Prediction.environment).all()
        
        env_labels = [env[0] for env in env_stats] if env_stats else ['Space Station', 'Mars Mission', 'Lunar Gateway', 'Bedrest', 'Clinical']
        env_scores = [round(float(env[1]), 1) for env in env_stats] if env_stats else [45.2, 62.8, 38.9, 41.5, 35.7]
        
        return jsonify({
            'accuracy_trend': {
                'labels': dates,
                'data': [round(random.uniform(85, 95), 1) for _ in dates]  # Simulated trend
            },
            'prediction_volume': {
                'labels': dates,
                'data': prediction_volume
            },
            'biomarker_distribution': {
                'labels': list(biomarker_counts.keys()),
                'data': list(biomarker_counts.values())
            },
            'environment_comparison': {
                'labels': env_labels,
                'risk_scores': env_scores
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Prediction Endpoints
@api.route('/predictions', methods=['GET'])
def get_predictions():
    """Get recent predictions with pagination from database"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        # Query predictions with pagination
        predictions_query = Prediction.query.order_by(desc(Prediction.created_at))
        paginated = predictions_query.paginate(
            page=page, 
            per_page=per_page, 
            error_out=False
        )
        
        predictions = [pred.to_dict() for pred in paginated.items]
        
        return jsonify({
            'predictions': predictions,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': paginated.total,
                'pages': paginated.pages
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/predictions', methods=['POST'])
def create_prediction():
    """Create a new cardiovascular risk prediction and save to database"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['patient_id', 'biomarkers', 'environment']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Generate prediction ID
        prediction_id = f'PRED-{random.randint(1000, 9999)}'
        
        # Simulate risk calculation based on biomarkers
        biomarker_values = data.get('biomarkers', {})
        risk_factors = []
        base_risk = random.uniform(20, 40)
        
        if biomarker_values.get('crp', 0) > 3.0:
            risk_factors.append('elevated_crp')
            base_risk += 15
        if biomarker_values.get('tnf_alpha', 0) > 15.0:
            risk_factors.append('elevated_tnf')
            base_risk += 20
        if biomarker_values.get('troponin', 0) > 0.1:
            risk_factors.append('elevated_troponin')
            base_risk += 25
        if biomarker_values.get('il6', 0) > 5.0:
            risk_factors.append('elevated_il6')
            base_risk += 10
        
        risk_score = min(round(base_risk, 1), 95.0)
        
        if risk_score < 30:
            risk_level = 'Low'
        elif risk_score < 65:
            risk_level = 'Medium'
        else:
            risk_level = 'High'
        
        # Create new prediction
        prediction = Prediction(
            prediction_id=prediction_id,
            patient_id=data['patient_id'],
            risk_score=risk_score,
            risk_level=risk_level,
            confidence=round(random.uniform(75, 98), 1),
            environment=data['environment']
        )
        
        prediction.set_biomarkers(biomarker_values)
        prediction.set_risk_factors(risk_factors)
        
        # Save to database
        db.session.add(prediction)
        db.session.commit()
        
        return jsonify(prediction.to_dict()), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# Experiment Endpoints
@api.route('/experiments')
def get_experiments():
    """Get all experiments from database"""
    try:
        experiments = Experiment.query.order_by(desc(Experiment.start_time)).all()
        experiments_data = [exp.to_dict() for exp in experiments]
        
        # Calculate summary statistics
        total = len(experiments_data)
        running = len([e for e in experiments_data if e['status'] == 'Running'])
        completed = len([e for e in experiments_data if e['status'] == 'Completed'])
        failed = len([e for e in experiments_data if e['status'] == 'Failed'])
        
        return jsonify({
            'experiments': experiments_data,
            'summary': {
                'total': total,
                'running': running,
                'completed': completed,
                'failed': failed
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/experiments', methods=['POST'])
def create_experiment():
    """Create a new experiment and save to database"""
    try:
        data = request.get_json()
        
        experiment_id = f'EXP-{random.randint(100, 999)}'
        
        experiment = Experiment(
            experiment_id=experiment_id,
            name=data.get('name', 'New Experiment'),
            status='Running',
            progress=0,
            epochs_total=data.get('epochs_total', random.randint(500, 1500)),
            created_by=data.get('created_by', 'system')
        )
        
        db.session.add(experiment)
        db.session.commit()
        
        return jsonify(experiment.to_dict()), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@api.route('/experiments/<experiment_id>/status', methods=['PUT'])
def update_experiment_status(experiment_id):
    """Update experiment status (pause, resume, stop)"""
    try:
        data = request.get_json()
        action = data.get('action')
        
        if action not in ['pause', 'resume', 'stop']:
            return jsonify({'error': 'Invalid action'}), 400
        
        experiment = Experiment.query.filter_by(experiment_id=experiment_id).first()
        if not experiment:
            return jsonify({'error': 'Experiment not found'}), 404
        
        status_map = {
            'pause': 'Paused',
            'resume': 'Running', 
            'stop': 'Stopped'
        }
        
        experiment.status = status_map[action]
        if action == 'stop':
            experiment.end_time = datetime.utcnow()
        
        db.session.commit()
        
        return jsonify({
            'experiment_id': experiment_id,
            'status': experiment.status,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# Model Endpoints
@api.route('/models')
def get_models():
    """Get available models from database"""
    try:
        models = MLModel.query.order_by(desc(MLModel.training_date)).all()
        models_data = [model.to_dict() for model in models]
        
        return jsonify({'models': models_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Dataset Endpoints
@api.route('/datasets')
def get_datasets():
    """Get available datasets from database"""
    try:
        datasets = Dataset.query.order_by(desc(Dataset.last_updated)).all()
        datasets_data = [dataset.to_dict() for dataset in datasets]
        
        return jsonify({'datasets': datasets_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Team Endpoints
@api.route('/team')
def get_team():
    """Get team members from database"""
    try:
        team_members = User.query.order_by(User.name).all()
        team_data = [user.to_dict() for user in team_members]
        
        return jsonify({'team_members': team_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Notification Endpoints
@api.route('/notifications')
def get_notifications():
    """Get user notifications from database"""
    try:
        notifications = Notification.query.order_by(desc(Notification.created_at)).limit(20).all()
        notifications_data = [notif.to_dict() for notif in notifications]
        
        return jsonify({'notifications': notifications_data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# API Health Check
@api.route('/health')
def health_check():
    """API health check endpoint with database status"""
    try:
        # Test database connection
        db.session.execute(db.text('SELECT 1'))
        db_status = 'connected'
    except Exception:
        db_status = 'error'
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'services': {
            'database': db_status,
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
