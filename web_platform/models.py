"""
Database models for CardioPredict Web Platform
Comprehensive SQLAlchemy models for all application data
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone
import json

db = SQLAlchemy()

class User(db.Model):
    """User model for team members and researchers"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(50), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    role = db.Column(db.String(100), nullable=False)
    department = db.Column(db.String(100), nullable=False)
    avatar = db.Column(db.Text)
    status = db.Column(db.String(20), default='offline')
    projects = db.Column(db.Integer, default=0)
    experiments = db.Column(db.Integer, default=0)
    joined = db.Column(db.DateTime, default=datetime.utcnow)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        return {
            'user_id': self.user_id,
            'name': self.name,
            'role': self.role,
            'department': self.department,
            'avatar': self.avatar,
            'status': self.status,
            'projects': self.projects,
            'experiments': self.experiments,
            'joined': self.joined.isoformat() if self.joined else None
        }

class Dataset(db.Model):
    """Dataset model for research datasets"""
    __tablename__ = 'datasets'
    
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.String(50), unique=True, nullable=False)
    name = db.Column(db.String(200), nullable=False)
    type = db.Column(db.String(50), nullable=False)
    size = db.Column(db.String(20))
    samples = db.Column(db.Integer)
    features = db.Column(db.Integer)
    status = db.Column(db.String(20), default='Active')
    description = db.Column(db.Text)
    file_path = db.Column(db.String(500))
    last_updated = db.Column(db.DateTime, default=datetime.utcnow)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'dataset_id': self.dataset_id,
            'name': self.name,
            'type': self.type,
            'size': self.size,
            'samples': self.samples,
            'features': self.features,
            'status': self.status,
            'description': self.description,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        }

class Prediction(db.Model):
    """Prediction model for cardiovascular risk predictions"""
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    prediction_id = db.Column(db.String(50), unique=True, nullable=False)
    patient_id = db.Column(db.String(100))
    risk_score = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float)
    environment = db.Column(db.String(100))
    biomarkers = db.Column(db.Text)  # JSON string
    risk_factors = db.Column(db.Text)  # JSON string
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def get_biomarkers(self):
        """Get biomarkers as dictionary"""
        if self.biomarkers:
            return json.loads(self.biomarkers)
        return {}
    
    def set_biomarkers(self, biomarkers_dict):
        """Set biomarkers from dictionary"""
        self.biomarkers = json.dumps(biomarkers_dict)
    
    def get_risk_factors(self):
        """Get risk factors as list"""
        if self.risk_factors:
            return json.loads(self.risk_factors)
        return []
    
    def set_risk_factors(self, risk_factors_list):
        """Set risk factors from list"""
        self.risk_factors = json.dumps(risk_factors_list)
    
    def to_dict(self):
        return {
            'prediction_id': self.prediction_id,
            'patient_id': self.patient_id,
            'risk_score': self.risk_score,
            'risk_level': self.risk_level,
            'confidence': self.confidence,
            'environment': self.environment,
            'biomarkers': self.get_biomarkers(),
            'risk_factors': self.get_risk_factors(),
            'timestamp': self.created_at.isoformat() if self.created_at else None
        }

class Experiment(db.Model):
    """Experiment model for ML experiments"""
    __tablename__ = 'experiments'
    
    id = db.Column(db.Integer, primary_key=True)
    experiment_id = db.Column(db.String(50), unique=True, nullable=False)
    name = db.Column(db.String(200), nullable=False)
    status = db.Column(db.String(20), default='Running')
    progress = db.Column(db.Integer, default=0)
    accuracy = db.Column(db.Float)
    loss = db.Column(db.Float)
    epochs_completed = db.Column(db.Integer, default=0)
    epochs_total = db.Column(db.Integer)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)
    estimated_completion = db.Column(db.DateTime)
    created_by = db.Column(db.String(50))  # user_id
    
    def to_dict(self):
        return {
            'experiment_id': self.experiment_id,
            'name': self.name,
            'status': self.status,
            'progress': self.progress,
            'accuracy': self.accuracy,
            'loss': self.loss,
            'epochs_completed': self.epochs_completed,
            'epochs_total': self.epochs_total,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'estimated_completion': self.estimated_completion.isoformat() if self.estimated_completion else None
        }

class MLModel(db.Model):
    """Model for tracking ML models"""
    __tablename__ = 'ml_models'
    
    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.String(50), unique=True, nullable=False)
    name = db.Column(db.String(200), nullable=False)
    model_type = db.Column(db.String(100), nullable=False)
    accuracy = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    auc_roc = db.Column(db.Float)
    status = db.Column(db.String(20), default='Active')
    description = db.Column(db.Text)
    file_path = db.Column(db.String(500))
    training_date = db.Column(db.DateTime, default=datetime.utcnow)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'model_id': self.model_id,
            'name': self.name,
            'type': self.model_type,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'auc_roc': self.auc_roc,
            'status': self.status,
            'description': self.description,
            'training_date': self.training_date.isoformat() if self.training_date else None
        }

class Notification(db.Model):
    """Notification model for user notifications"""
    __tablename__ = 'notifications'
    
    id = db.Column(db.Integer, primary_key=True)
    notification_id = db.Column(db.String(50), unique=True, nullable=False)
    user_id = db.Column(db.String(50))  # Can be null for system-wide notifications
    notification_type = db.Column(db.String(50), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    message = db.Column(db.Text, nullable=False)
    priority = db.Column(db.String(20), default='medium')
    read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.notification_id,
            'type': self.notification_type,
            'title': self.title,
            'message': self.message,
            'priority': self.priority,
            'read': self.read,
            'timestamp': self.created_at.isoformat() if self.created_at else None
        }
