"""
Phase 3 Model Extensions for CardioPredict Platform
Authentication, ML versioning, and advanced features
"""

from datetime import datetime, timedelta
import json
import uuid
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from models import db, User

# Authentication and Session Management
class UserSession(db.Model):
    """User session tracking for security and analytics"""
    __tablename__ = 'user_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(100), unique=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    
    # Session details
    login_time = db.Column(db.DateTime, default=datetime.now)
    last_activity = db.Column(db.DateTime, default=datetime.now)
    logout_time = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    
    # Security tracking
    ip_address = db.Column(db.String(45))  # IPv6 support
    user_agent = db.Column(db.Text)
    device_fingerprint = db.Column(db.String(100))
    location = db.Column(db.String(100))
    
    # Relationships
    user = db.relationship('User', backref=db.backref('sessions', lazy=True))
    
    def to_dict(self):
        return {
            'session_id': self.session_id,
            'user_id': self.user.user_id,
            'login_time': self.login_time.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'logout_time': self.logout_time.isoformat() if self.logout_time else None,
            'is_active': self.is_active,
            'ip_address': self.ip_address,
            'location': self.location,
            'device_info': self.user_agent[:100] if self.user_agent else None
        }

# Association table for user roles
user_roles_association = db.Table('user_role_assignments',
    db.Column('user_id', db.Integer, db.ForeignKey('users.id'), primary_key=True),
    db.Column('role_id', db.Integer, db.ForeignKey('user_roles.id'), primary_key=True),
    db.Column('assigned_at', db.DateTime, default=datetime.now)
)

class UserRole(db.Model):
    """User roles for RBAC system"""
    __tablename__ = 'user_roles'
    
    id = db.Column(db.Integer, primary_key=True)
    role_name = db.Column(db.String(50), unique=True, nullable=False)
    description = db.Column(db.Text)
    
    # Permissions
    permissions = db.Column(db.Text)  # JSON string of permissions
    is_system_role = db.Column(db.Boolean, default=False)
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    
    def get_permissions(self):
        """Get permissions as list"""
        if self.permissions:
            return json.loads(self.permissions)
        return []
    
    def set_permissions(self, permissions_list):
        """Set permissions from list"""
        self.permissions = json.dumps(permissions_list)
    
    def to_dict(self):
        return {
            'id': self.id,
            'role_name': self.role_name,
            'description': self.description,
            'permissions': self.get_permissions(),
            'is_system_role': self.is_system_role,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

# Add roles relationship to User model
# User.roles = db.relationship('UserRole', secondary=user_roles_association, backref='users')

class APIKey(db.Model):
    """API key management for external integrations"""
    __tablename__ = 'api_keys'
    
    id = db.Column(db.Integer, primary_key=True)
    key_id = db.Column(db.String(50), unique=True, nullable=False)
    key_hash = db.Column(db.String(255), nullable=False)  # Hashed API key
    
    # Key details
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    
    # Ownership and permissions
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'))
    
    # Access control
    permissions = db.Column(db.Text)  # JSON string of allowed endpoints/actions
    rate_limit = db.Column(db.Integer, default=1000)  # Requests per hour
    
    # Status and lifecycle
    is_active = db.Column(db.Boolean, default=True)
    expires_at = db.Column(db.DateTime)
    last_used = db.Column(db.DateTime)
    usage_count = db.Column(db.Integer, default=0)
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    user = db.relationship('User', backref=db.backref('api_keys', lazy=True))
    project = db.relationship('Project', backref=db.backref('api_keys', lazy=True))
    
    def get_permissions(self):
        """Get permissions as list"""
        if self.permissions:
            return json.loads(self.permissions)
        return []
    
    def set_permissions(self, permissions_list):
        """Set permissions from list"""
        self.permissions = json.dumps(permissions_list)
    
    def check_key(self, provided_key):
        """Verify provided API key against stored hash"""
        return check_password_hash(self.key_hash, provided_key)
    
    def is_expired(self):
        """Check if API key is expired"""
        if self.expires_at:
            return datetime.now() > self.expires_at
        return False
    
    def to_dict(self, include_sensitive=False):
        result = {
            'key_id': self.key_id,
            'name': self.name,
            'description': self.description,
            'user_id': self.user.user_id,
            'project_id': self.project.project_id if self.project else None,
            'permissions': self.get_permissions(),
            'rate_limit': self.rate_limit,
            'is_active': self.is_active,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'usage_count': self.usage_count,
            'created_at': self.created_at.isoformat()
        }
        
        if include_sensitive:
            result['key_hash'] = self.key_hash
        
        return result

# Advanced ML Models
class MLModelVersion(db.Model):
    """Model versioning for ML lifecycle management"""
    __tablename__ = 'ml_model_versions'
    
    id = db.Column(db.Integer, primary_key=True)
    version_id = db.Column(db.String(50), unique=True, nullable=False)
    
    # Model reference
    model_id = db.Column(db.Integer, db.ForeignKey('ml_models.id'), nullable=False)
    
    # Version details
    version_number = db.Column(db.String(20), nullable=False)  # e.g., "1.2.3"
    version_name = db.Column(db.String(100))
    description = db.Column(db.Text)
    
    # Model artifacts
    model_path = db.Column(db.String(500))
    config = db.Column(db.Text)  # JSON string of model configuration
    parameters = db.Column(db.Text)  # JSON string of model parameters
    
    # Performance metrics
    accuracy = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    auc_score = db.Column(db.Float)
    custom_metrics = db.Column(db.Text)  # JSON string for additional metrics
    
    # Training details
    training_data_size = db.Column(db.Integer)
    training_time = db.Column(db.Integer)  # Training time in seconds
    hyperparameters = db.Column(db.Text)  # JSON string
    
    # Deployment status
    status = db.Column(db.String(20), default='trained')  # trained, testing, deployed, retired
    is_active = db.Column(db.Boolean, default=False)
    deployment_date = db.Column(db.DateTime)
    retirement_date = db.Column(db.DateTime)
    
    # Metadata
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    model = db.relationship('MLModel', backref=db.backref('versions', lazy=True))
    creator = db.relationship('User', foreign_keys=[created_by])
    
    def get_config(self):
        """Get model configuration as dictionary"""
        if self.config:
            return json.loads(self.config)
        return {}
    
    def set_config(self, config_dict):
        """Set model configuration from dictionary"""
        self.config = json.dumps(config_dict)
    
    def get_parameters(self):
        """Get model parameters as dictionary"""
        if self.parameters:
            return json.loads(self.parameters)
        return {}
    
    def set_parameters(self, params_dict):
        """Set model parameters from dictionary"""
        self.parameters = json.dumps(params_dict)
    
    def get_custom_metrics(self):
        """Get custom metrics as dictionary"""
        if self.custom_metrics:
            return json.loads(self.custom_metrics)
        return {}
    
    def set_custom_metrics(self, metrics_dict):
        """Set custom metrics from dictionary"""
        self.custom_metrics = json.dumps(metrics_dict)
    
    def get_hyperparameters(self):
        """Get hyperparameters as dictionary"""
        if self.hyperparameters:
            return json.loads(self.hyperparameters)
        return {}
    
    def set_hyperparameters(self, hyperparams_dict):
        """Set hyperparameters from dictionary"""
        self.hyperparameters = json.dumps(hyperparams_dict)
    
    def to_dict(self):
        return {
            'version_id': self.version_id,
            'model_id': self.model.model_id,
            'version_number': self.version_number,
            'version_name': self.version_name,
            'description': self.description,
            'model_path': self.model_path,
            'config': self.get_config(),
            'parameters': self.get_parameters(),
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'auc_score': self.auc_score,
            'custom_metrics': self.get_custom_metrics(),
            'training_data_size': self.training_data_size,
            'training_time': self.training_time,
            'hyperparameters': self.get_hyperparameters(),
            'status': self.status,
            'is_active': self.is_active,
            'deployment_date': self.deployment_date.isoformat() if self.deployment_date else None,
            'retirement_date': self.retirement_date.isoformat() if self.retirement_date else None,
            'created_by': self.creator.user_id if self.creator else None,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

class FeatureImportance(db.Model):
    """Feature importance tracking for model interpretability"""
    __tablename__ = 'feature_importance'
    
    id = db.Column(db.Integer, primary_key=True)
    importance_id = db.Column(db.String(50), unique=True, nullable=False)
    
    # Model reference
    model_version_id = db.Column(db.Integer, db.ForeignKey('ml_model_versions.id'), nullable=False)
    
    # Feature details
    feature_name = db.Column(db.String(100), nullable=False)
    importance_score = db.Column(db.Float, nullable=False)
    importance_type = db.Column(db.String(50))  # permutation, shap, gain, etc.
    
    # Analysis details
    analysis_method = db.Column(db.String(50))  # The method used to calculate importance
    confidence_interval = db.Column(db.Text)  # JSON string [lower, upper]
    
    # Metadata
    calculated_at = db.Column(db.DateTime, default=datetime.now)
    
    # Relationships
    model_version = db.relationship('MLModelVersion', backref=db.backref('feature_importances', lazy=True))
    
    def get_confidence_interval(self):
        """Get confidence interval as list"""
        if self.confidence_interval:
            return json.loads(self.confidence_interval)
        return None
    
    def set_confidence_interval(self, interval_list):
        """Set confidence interval from list [lower, upper]"""
        self.confidence_interval = json.dumps(interval_list)
    
    def to_dict(self):
        return {
            'importance_id': self.importance_id,
            'model_version_id': self.model_version.version_id,
            'feature_name': self.feature_name,
            'importance_score': self.importance_score,
            'importance_type': self.importance_type,
            'analysis_method': self.analysis_method,
            'confidence_interval': self.get_confidence_interval(),
            'calculated_at': self.calculated_at.isoformat()
        }

class ModelDriftDetection(db.Model):
    """Model drift detection and monitoring"""
    __tablename__ = 'model_drift_detection'
    
    id = db.Column(db.Integer, primary_key=True)
    drift_id = db.Column(db.String(50), unique=True, nullable=False)
    
    # Model reference
    model_version_id = db.Column(db.Integer, db.ForeignKey('ml_model_versions.id'), nullable=False)
    
    # Drift analysis
    drift_type = db.Column(db.String(50))  # data_drift, concept_drift, prediction_drift
    drift_score = db.Column(db.Float)  # 0-1 score indicating drift severity
    threshold = db.Column(db.Float)  # Threshold used for detection
    is_drift_detected = db.Column(db.Boolean, default=False)
    
    # Analysis details
    analysis_period_start = db.Column(db.DateTime)
    analysis_period_end = db.Column(db.DateTime)
    baseline_data_size = db.Column(db.Integer)
    current_data_size = db.Column(db.Integer)
    
    # Drift metrics
    statistical_tests = db.Column(db.Text)  # JSON string of test results
    feature_drift_scores = db.Column(db.Text)  # JSON string of per-feature drift scores
    
    # Recommendations
    recommendation = db.Column(db.Text)
    action_required = db.Column(db.String(50))  # none, retrain, investigate, alert
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.now)
    
    # Relationships
    model_version = db.relationship('MLModelVersion', backref=db.backref('drift_detections', lazy=True))
    
    def get_statistical_tests(self):
        """Get statistical test results as dictionary"""
        if self.statistical_tests:
            return json.loads(self.statistical_tests)
        return {}
    
    def set_statistical_tests(self, tests_dict):
        """Set statistical test results from dictionary"""
        self.statistical_tests = json.dumps(tests_dict)
    
    def get_feature_drift_scores(self):
        """Get feature drift scores as dictionary"""
        if self.feature_drift_scores:
            return json.loads(self.feature_drift_scores)
        return {}
    
    def set_feature_drift_scores(self, scores_dict):
        """Set feature drift scores from dictionary"""
        self.feature_drift_scores = json.dumps(scores_dict)
    
    def to_dict(self):
        return {
            'drift_id': self.drift_id,
            'model_version_id': self.model_version.version_id,
            'drift_type': self.drift_type,
            'drift_score': self.drift_score,
            'threshold': self.threshold,
            'is_drift_detected': self.is_drift_detected,
            'analysis_period_start': self.analysis_period_start.isoformat() if self.analysis_period_start else None,
            'analysis_period_end': self.analysis_period_end.isoformat() if self.analysis_period_end else None,
            'baseline_data_size': self.baseline_data_size,
            'current_data_size': self.current_data_size,
            'statistical_tests': self.get_statistical_tests(),
            'feature_drift_scores': self.get_feature_drift_scores(),
            'recommendation': self.recommendation,
            'action_required': self.action_required,
            'created_at': self.created_at.isoformat()
        }

# Real-time Collaboration Models
class RealtimeEvent(db.Model):
    """Real-time events for WebSocket notifications"""
    __tablename__ = 'realtime_events'
    
    id = db.Column(db.Integer, primary_key=True)
    event_id = db.Column(db.String(50), unique=True, nullable=False)
    
    # Event details
    event_type = db.Column(db.String(50), nullable=False)  # comment_added, experiment_updated, etc.
    entity_type = db.Column(db.String(50))  # project, experiment, prediction, etc.
    entity_id = db.Column(db.String(50))
    
    # Event data
    event_data = db.Column(db.Text)  # JSON string of event-specific data
    message = db.Column(db.Text)  # Human-readable message
    
    # Targeting
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))  # Specific user (optional)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'))  # Project scope (optional)
    broadcast = db.Column(db.Boolean, default=False)  # Broadcast to all users
    
    # Status
    is_processed = db.Column(db.Boolean, default=False)
    processed_at = db.Column(db.DateTime)
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.now)
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    
    # Relationships
    user = db.relationship('User', foreign_keys=[user_id], backref=db.backref('targeted_events', lazy=True))
    project = db.relationship('Project', backref=db.backref('events', lazy=True))
    creator = db.relationship('User', foreign_keys=[created_by])
    
    def get_event_data(self):
        """Get event data as dictionary"""
        if self.event_data:
            return json.loads(self.event_data)
        return {}
    
    def set_event_data(self, data_dict):
        """Set event data from dictionary"""
        self.event_data = json.dumps(data_dict)
    
    def to_dict(self):
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'entity_type': self.entity_type,
            'entity_id': self.entity_id,
            'event_data': self.get_event_data(),
            'message': self.message,
            'user_id': self.user.user_id if self.user else None,
            'project_id': self.project.project_id if self.project else None,
            'broadcast': self.broadcast,
            'is_processed': self.is_processed,
            'processed_at': self.processed_at.isoformat() if self.processed_at else None,
            'created_at': self.created_at.isoformat(),
            'created_by': self.creator.user_id if self.creator else None
        }

# Background Processing Models
class BackgroundJob(db.Model):
    """Background job tracking for Celery tasks"""
    __tablename__ = 'background_jobs'
    
    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.String(50), unique=True, nullable=False)
    celery_task_id = db.Column(db.String(100))  # Celery task UUID
    
    # Job details
    job_type = db.Column(db.String(50), nullable=False)  # model_training, report_generation, etc.
    job_name = db.Column(db.String(200))
    description = db.Column(db.Text)
    
    # Job parameters
    parameters = db.Column(db.Text)  # JSON string of job parameters
    
    # Status tracking
    status = db.Column(db.String(20), default='pending')  # pending, running, completed, failed, cancelled
    progress = db.Column(db.Integer, default=0)  # 0-100
    
    # Timing
    created_at = db.Column(db.DateTime, default=datetime.now)
    started_at = db.Column(db.DateTime)
    completed_at = db.Column(db.DateTime)
    estimated_duration = db.Column(db.Integer)  # Estimated duration in seconds
    
    # Results and errors
    result = db.Column(db.Text)  # JSON string of job results
    error_message = db.Column(db.Text)
    retry_count = db.Column(db.Integer, default=0)
    max_retries = db.Column(db.Integer, default=3)
    
    # Relationships
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'))
    
    user = db.relationship('User', backref=db.backref('background_jobs', lazy=True))
    project = db.relationship('Project', backref=db.backref('background_jobs', lazy=True))
    
    def get_parameters(self):
        """Get job parameters as dictionary"""
        if self.parameters:
            return json.loads(self.parameters)
        return {}
    
    def set_parameters(self, params_dict):
        """Set job parameters from dictionary"""
        self.parameters = json.dumps(params_dict)
    
    def get_result(self):
        """Get job result as dictionary"""
        if self.result:
            return json.loads(self.result)
        return {}
    
    def set_result(self, result_dict):
        """Set job result from dictionary"""
        self.result = json.dumps(result_dict)
    
    def to_dict(self):
        return {
            'job_id': self.job_id,
            'celery_task_id': self.celery_task_id,
            'job_type': self.job_type,
            'job_name': self.job_name,
            'description': self.description,
            'parameters': self.get_parameters(),
            'status': self.status,
            'progress': self.progress,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'estimated_duration': self.estimated_duration,
            'result': self.get_result(),
            'error_message': self.error_message,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'created_by': self.user.user_id if self.user else None,
            'project_id': self.project.project_id if self.project else None
        }
