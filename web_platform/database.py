"""
Database initialization and sample data generation for CardioPredict
"""

import os
import random
from datetime import datetime, timedelta
from models import db, User, Dataset, Prediction, Experiment, MLModel, Notification

def init_database(app):
    """Initialize database with sample data"""
    with app.app_context():
        # Create all tables
        db.create_all()
        
        # Check if data already exists
        if User.query.count() > 0:
            print("Database already initialized with data")
            return
        
        print("Initializing database with sample data...")
        
        # Create sample users
        create_sample_users()
        
        # Create sample datasets
        create_sample_datasets()
        
        # Create sample ML models
        create_sample_models()
        
        # Create sample predictions
        create_sample_predictions()
        
        # Create sample experiments
        create_sample_experiments()
        
        # Create sample notifications
        create_sample_notifications()
        
        # Commit all changes
        db.session.commit()
        print("Database initialized successfully!")

def create_sample_users():
    """Create sample team members"""
    users = [
        {
            'user_id': 'sarah.chen',
            'name': 'Dr. Sarah Chen',
            'role': 'Principal Investigator',
            'department': 'Space Medicine',
            'avatar': 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=64&h=64&fit=crop&crop=face',
            'status': 'online',
            'projects': 4,
            'experiments': 23,
            'joined': datetime(2024, 1, 15)
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
            'joined': datetime(2024, 3, 20)
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
            'joined': datetime(2024, 2, 10)
        },
        {
            'user_id': 'james.parker',
            'name': 'Dr. James Parker',
            'role': 'Bioinformatics Specialist',
            'department': 'Computational Biology',
            'avatar': 'https://images.unsplash.com/photo-1560250097-0b93528c311a?w=64&h=64&fit=crop&crop=face',
            'status': 'offline',
            'projects': 5,
            'experiments': 32,
            'joined': datetime(2024, 4, 5)
        }
    ]
    
    for user_data in users:
        user = User(**user_data)
        db.session.add(user)

def create_sample_datasets():
    """Create sample datasets"""
    datasets = [
        {
            'dataset_id': 'OSD-258',
            'name': 'NASA OSD-258 - SpaceX Inspiration4',
            'type': 'RNA-seq',
            'size': '2.4 GB',
            'samples': 847,
            'features': 20531,
            'status': 'Active',
            'description': 'SpaceX Inspiration4 RNA sequencing data for cardiovascular gene expression analysis',
            'file_path': '/data/OSD-258/',
            'last_updated': datetime.now() - timedelta(days=5)
        },
        {
            'dataset_id': 'OSD-484',
            'name': 'NASA OSD-484 - Cardiac Microgravity',
            'type': 'Microarray',
            'size': '1.8 GB',
            'samples': 624,
            'features': 54675,
            'status': 'Active',
            'description': 'Cardiac gene expression profiles from microgravity exposure studies',
            'file_path': '/data/OSD-484/',
            'last_updated': datetime.now() - timedelta(days=8)
        },
        {
            'dataset_id': 'OSD-575',
            'name': 'Bedrest Study BRS-01',
            'type': 'Clinical',
            'size': '945 MB',
            'samples': 156,
            'features': 847,
            'status': 'Processing',
            'description': '70-day bedrest study biomarker and physiological data',
            'file_path': '/data/OSD-575/',
            'last_updated': datetime.now() - timedelta(days=2)
        },
        {
            'dataset_id': 'OSD-635',
            'name': 'Multi-omics Space Study',
            'type': 'Multi-omics',
            'size': '4.2 GB',
            'samples': 1234,
            'features': 15789,
            'status': 'Active',
            'description': 'Comprehensive multi-omics dataset from long-duration spaceflight studies',
            'file_path': '/data/OSD-635/',
            'last_updated': datetime.now() - timedelta(days=10)
        },
        {
            'dataset_id': 'OSD-51',
            'name': 'ISS Expedition Cardiac Data',
            'type': 'Physiological',
            'size': '1.2 GB',
            'samples': 298,
            'features': 1247,
            'status': 'Active',
            'description': 'International Space Station cardiovascular monitoring data',
            'file_path': '/data/OSD-51/',
            'last_updated': datetime.now() - timedelta(days=15)
        }
    ]
    
    for dataset_data in datasets:
        dataset = Dataset(**dataset_data)
        db.session.add(dataset)

def create_sample_models():
    """Create sample ML models"""
    models = [
        {
            'model_id': 'cardio-predict-v2.1',
            'name': 'CardioPredict ML Model v2.1',
            'model_type': 'Random Forest',
            'accuracy': 0.892,
            'precision': 0.847,
            'recall': 0.823,
            'f1_score': 0.835,
            'auc_roc': 0.912,
            'status': 'Active',
            'description': 'Advanced cardiovascular risk prediction model for space medicine applications',
            'file_path': '/models/cardio_predict_v2.1.joblib',
            'training_date': datetime(2025, 6, 15, 10, 30)
        },
        {
            'model_id': 'micrograv-adapt-v1.3',
            'name': 'Microgravity Adaptation Predictor',
            'model_type': 'Neural Network',
            'accuracy': 0.876,
            'precision': 0.832,
            'recall': 0.801,
            'f1_score': 0.816,
            'auc_roc': 0.894,
            'status': 'Active',
            'description': 'Specialized model for predicting cardiovascular adaptation to microgravity environments',
            'file_path': '/models/micrograv_adapt_v1.3.pkl',
            'training_date': datetime(2025, 6, 20, 14, 15)
        },
        {
            'model_id': 'bedrest-bio-v1.0',
            'name': 'Bedrest Biomarker Model',
            'model_type': 'Gradient Boosting',
            'accuracy': 0.853,
            'precision': 0.819,
            'recall': 0.787,
            'f1_score': 0.803,
            'auc_roc': 0.878,
            'status': 'Training',
            'description': 'Model optimized for bedrest study biomarker analysis',
            'file_path': '/models/bedrest_bio_v1.0.joblib',
            'training_date': datetime(2025, 6, 10, 9, 45)
        }
    ]
    
    for model_data in models:
        model = MLModel(**model_data)
        db.session.add(model)

def create_sample_predictions():
    """Create sample cardiovascular predictions"""
    environments = ['Space Station', 'Mars Mission', 'Lunar Gateway', 'Bedrest Study', 'Clinical']
    
    for i in range(30):
        # Generate realistic biomarker values
        biomarkers = {
            'crp': round(random.uniform(0.5, 8.0), 2),
            'pf4': round(random.uniform(2.0, 15.0), 2),
            'tnf_alpha': round(random.uniform(1.0, 25.0), 2),
            'il6': round(random.uniform(0.5, 10.0), 2),
            'troponin': round(random.uniform(0.01, 0.5), 3)
        }
        
        # Calculate risk score based on biomarkers
        risk_factors = []
        base_risk = random.uniform(20, 40)
        
        if biomarkers['crp'] > 3.0:
            risk_factors.append('elevated_crp')
            base_risk += 15
        if biomarkers['tnf_alpha'] > 15.0:
            risk_factors.append('elevated_tnf')
            base_risk += 20
        if biomarkers['troponin'] > 0.1:
            risk_factors.append('elevated_troponin')
            base_risk += 25
        if biomarkers['il6'] > 5.0:
            risk_factors.append('elevated_il6')
            base_risk += 10
        
        risk_score = min(base_risk, 95.0)
        
        if risk_score < 30:
            risk_level = 'Low'
        elif risk_score < 65:
            risk_level = 'Medium'
        else:
            risk_level = 'High'
        
        prediction = Prediction(
            prediction_id=f'PRED-{1000 + i}',
            patient_id=f'PAT-{random.randint(100, 999)}',
            risk_score=round(risk_score, 1),
            risk_level=risk_level,
            confidence=round(random.uniform(75, 98), 1),
            environment=random.choice(environments),
            created_at=datetime.now() - timedelta(days=random.randint(1, 30))
        )
        
        prediction.set_biomarkers(biomarkers)
        prediction.set_risk_factors(risk_factors)
        
        db.session.add(prediction)

def create_sample_experiments():
    """Create sample experiments"""
    experiment_names = [
        'Microgravity Cardiac Adaptation',
        'Bedrest Biomarker Analysis',
        'Deep Space Radiation Effects',
        'Lunar Surface Cardiovascular Study',
        'Mars Mission Risk Assessment',
        'ISS Crew Health Monitoring',
        'Spaceflight Countermeasures Study'
    ]
    
    statuses = ['Running', 'Completed', 'Failed', 'Paused']
    
    for i, name in enumerate(experiment_names):
        status = random.choice(statuses)
        progress = random.randint(0, 100) if status == 'Running' else (100 if status == 'Completed' else random.randint(10, 80))
        
        experiment = Experiment(
            experiment_id=f'EXP-{100 + i}',
            name=name,
            status=status,
            progress=progress,
            accuracy=round(random.uniform(0.75, 0.98), 3) if progress > 20 else None,
            loss=round(random.uniform(0.01, 0.15), 4) if progress > 20 else None,
            epochs_completed=random.randint(50, 1000) if progress > 0 else 0,
            epochs_total=random.randint(500, 1500),
            start_time=datetime.now() - timedelta(hours=random.randint(1, 72)),
            estimated_completion=datetime.now() + timedelta(hours=random.randint(1, 24)) if status == 'Running' else None,
            created_by=random.choice(['sarah.chen', 'michael.rodriguez', 'emily.watson'])
        )
        
        db.session.add(experiment)

def create_sample_notifications():
    """Create sample notifications"""
    notifications = [
        {
            'notification_id': 'notif-001',
            'notification_type': 'experiment_complete',
            'title': 'Experiment Completed',
            'message': 'Microgravity Cardiac Adaptation experiment has finished with 94.2% accuracy',
            'priority': 'high',
            'read': False,
            'created_at': datetime.now() - timedelta(hours=2)
        },
        {
            'notification_id': 'notif-002',
            'notification_type': 'data_upload',
            'title': 'Dataset Updated',
            'message': 'NASA OSD-687 dataset has been successfully uploaded and processed',
            'priority': 'medium',
            'read': False,
            'created_at': datetime.now() - timedelta(hours=4)
        },
        {
            'notification_id': 'notif-003',
            'notification_type': 'team_update',
            'title': 'Team Member Added',
            'message': 'Dr. Smith has joined the Mars-2027 project team',
            'priority': 'low',
            'read': True,
            'created_at': datetime.now() - timedelta(days=1)
        },
        {
            'notification_id': 'notif-004',
            'notification_type': 'model_update',
            'title': 'Model Training Complete',
            'message': 'CardioPredict v2.1 has finished training with improved accuracy of 89.2%',
            'priority': 'high',
            'read': False,
            'created_at': datetime.now() - timedelta(hours=6)
        },
        {
            'notification_id': 'notif-005',
            'notification_type': 'system_alert',
            'title': 'System Maintenance',
            'message': 'Scheduled maintenance will occur tomorrow from 2:00-4:00 AM UTC',
            'priority': 'medium',
            'read': True,
            'created_at': datetime.now() - timedelta(hours=12)
        }
    ]
    
    for notif_data in notifications:
        notification = Notification(**notif_data)
        db.session.add(notification)
