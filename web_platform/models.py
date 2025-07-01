"""
Database models for CardioPredict Web Platform
Comprehensive SQLAlchemy models for all application data
Phase 2: Advanced Features and Project Management
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timezone
import json

db = SQLAlchemy()

# Association table for many-to-many relationship between Users and Projects
project_members = db.Table('project_members',
    db.Column('user_id', db.Integer, db.ForeignKey('users.id'), primary_key=True),
    db.Column('project_id', db.Integer, db.ForeignKey('projects.id'), primary_key=True),
    db.Column('role', db.String(50), default='member'),
    db.Column('joined_date', db.DateTime, default=datetime.utcnow),
    db.Column('permissions', db.Text)  # JSON string for role-specific permissions
)

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
    
    # Phase 2: Enhanced relationships
    email = db.Column(db.String(120), unique=True)
    phone = db.Column(db.String(20))
    timezone = db.Column(db.String(50), default='UTC')
    preferences = db.Column(db.Text)  # JSON string for user preferences
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    project_memberships = db.relationship('Project', secondary=project_members, back_populates='team_members')
    comments = db.relationship('Comment', backref='author', lazy='dynamic')
    audit_logs = db.relationship('AuditLog', backref='user', lazy='dynamic')
    created_experiments = db.relationship('Experiment', backref='creator', lazy='dynamic')
    
    def get_preferences(self):
        """Get user preferences as dictionary"""
        if self.preferences:
            return json.loads(self.preferences)
        return {}
    
    def set_preferences(self, preferences_dict):
        """Set user preferences from dictionary"""
        self.preferences = json.dumps(preferences_dict)
    
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
            'joined': self.joined.isoformat() if self.joined else None,
            'email': self.email,
            'phone': self.phone,
            'timezone': self.timezone,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'is_active': self.is_active
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
    
    # Phase 2: Enhanced relationships and metadata
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'))
    uploaded_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    uploader = db.relationship('User', foreign_keys=[uploaded_by])
    
    # Dataset metadata
    version = db.Column(db.String(20), default='1.0')
    tags = db.Column(db.Text)  # JSON array of tags
    data_metadata = db.Column(db.Text)  # JSON string for additional metadata
    access_level = db.Column(db.String(20), default='internal')  # public, internal, restricted
    
    def get_tags(self):
        """Get tags as list"""
        if self.tags:
            return json.loads(self.tags)
        return []
    
    def set_tags(self, tags_list):
        """Set tags from list"""
        self.tags = json.dumps(tags_list)
    
    def get_metadata(self):
        """Get metadata as dictionary"""
        if self.data_metadata:
            return json.loads(self.data_metadata)
        return {}
    
    def set_metadata(self, metadata_dict):
        """Set metadata from dictionary"""
        self.data_metadata = json.dumps(metadata_dict)
    
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
            'version': self.version,
            'access_level': self.access_level,
            'tags': self.get_tags(),
            'project': self.project.to_dict() if self.project else None,
            'uploader': self.uploader.to_dict() if self.uploader else None,
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
    
    # Phase 2: Enhanced experiment tracking
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'))
    creator_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    
    # Experiment configuration and results
    config = db.Column(db.Text)  # JSON string of experiment configuration
    results = db.Column(db.Text)  # JSON string of detailed results
    logs = db.Column(db.Text)  # Experiment execution logs
    
    # Resource usage tracking
    cpu_hours = db.Column(db.Float)
    memory_peak = db.Column(db.Float)  # Peak memory usage in GB
    gpu_hours = db.Column(db.Float)
    
    # Version and reproducibility
    code_version = db.Column(db.String(50))
    environment = db.Column(db.Text)  # JSON string of environment details
    
    def get_config(self):
        """Get experiment configuration as dictionary"""
        if self.config:
            return json.loads(self.config)
        return {}
    
    def set_config(self, config_dict):
        """Set experiment configuration from dictionary"""
        self.config = json.dumps(config_dict)
    
    def get_results(self):
        """Get experiment results as dictionary"""
        if self.results:
            return json.loads(self.results)
        return {}
    
    def set_results(self, results_dict):
        """Set experiment results from dictionary"""
        self.results = json.dumps(results_dict)
    
    def get_environment(self):
        """Get environment details as dictionary"""
        if self.environment:
            return json.loads(self.environment)
        return {}
    
    def set_environment(self, env_dict):
        """Set environment details from dictionary"""
        self.environment = json.dumps(env_dict)
    
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
            'estimated_completion': self.estimated_completion.isoformat() if self.estimated_completion else None,
            'project': self.project.to_dict() if self.project else None,
            'creator': self.creator.to_dict() if self.creator else None,
            'cpu_hours': self.cpu_hours,
            'memory_peak': self.memory_peak,
            'gpu_hours': self.gpu_hours,
            'code_version': self.code_version
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

# Phase 2: Advanced Models for Project Management and Collaboration

class Project(db.Model):
    """Project model for managing research projects"""
    __tablename__ = 'projects'
    
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.String(50), unique=True, nullable=False)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    status = db.Column(db.String(20), default='active')  # active, completed, on_hold, cancelled
    priority = db.Column(db.String(20), default='medium')  # low, medium, high, critical
    
    # Project metadata
    start_date = db.Column(db.DateTime)
    end_date = db.Column(db.DateTime)
    budget = db.Column(db.Float)
    funding_source = db.Column(db.String(200))
    
    # Relationships
    lead_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    lead = db.relationship('User', foreign_keys=[lead_id])
    team_members = db.relationship('User', secondary=project_members, back_populates='project_memberships')
    
    # Project tracking
    progress = db.Column(db.Integer, default=0)  # 0-100 percentage
    milestones_completed = db.Column(db.Integer, default=0)
    milestones_total = db.Column(db.Integer, default=0)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Related entities
    experiments = db.relationship('Experiment', backref='project', lazy='dynamic')
    datasets = db.relationship('Dataset', backref='project', lazy='dynamic')
    reports = db.relationship('Report', backref='project', lazy='dynamic')
    comments = db.relationship('Comment', backref='project', lazy='dynamic')
    
    def to_dict(self):
        return {
            'project_id': self.project_id,
            'name': self.name,
            'description': self.description,
            'status': self.status,
            'priority': self.priority,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'budget': self.budget,
            'funding_source': self.funding_source,
            'progress': self.progress,
            'milestones_completed': self.milestones_completed,
            'milestones_total': self.milestones_total,
            'lead': self.lead.to_dict() if self.lead else None,
            'team_size': len(self.team_members),
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Comment(db.Model):
    """Comment model for collaborative discussions"""
    __tablename__ = 'comments'
    
    id = db.Column(db.Integer, primary_key=True)
    comment_id = db.Column(db.String(50), unique=True, nullable=False)
    content = db.Column(db.Text, nullable=False)
    
    # Polymorphic relationships - comments can be on different entities
    entity_type = db.Column(db.String(50), nullable=False)  # prediction, experiment, dataset, project
    entity_id = db.Column(db.String(50), nullable=False)
    
    # User and project relationships
    author_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'))
    
    # Threading support
    parent_id = db.Column(db.Integer, db.ForeignKey('comments.id'))
    replies = db.relationship('Comment', backref=db.backref('parent', remote_side=[id]), lazy='dynamic')
    
    # Metadata
    is_edited = db.Column(db.Boolean, default=False)
    is_pinned = db.Column(db.Boolean, default=False)
    tags = db.Column(db.Text)  # JSON array of tags
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def get_tags(self):
        """Get tags as list"""
        if self.tags:
            return json.loads(self.tags)
        return []
    
    def set_tags(self, tags_list):
        """Set tags from list"""
        self.tags = json.dumps(tags_list)
    
    def to_dict(self):
        return {
            'comment_id': self.comment_id,
            'content': self.content,
            'entity_type': self.entity_type,
            'entity_id': self.entity_id,
            'author': self.author.to_dict() if self.author else None,
            'parent_id': self.parent.comment_id if self.parent else None,
            'reply_count': self.replies.count(),
            'is_edited': self.is_edited,
            'is_pinned': self.is_pinned,
            'tags': self.get_tags(),
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class AuditLog(db.Model):
    """Audit log model for tracking system changes"""
    __tablename__ = 'audit_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    log_id = db.Column(db.String(50), unique=True, nullable=False)
    
    # Action details
    action = db.Column(db.String(50), nullable=False)  # create, update, delete, view
    entity_type = db.Column(db.String(50), nullable=False)
    entity_id = db.Column(db.String(50), nullable=False)
    
    # User and session info
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.Text)
    session_id = db.Column(db.String(100))
    
    # Change details
    old_values = db.Column(db.Text)  # JSON string of old values
    new_values = db.Column(db.Text)  # JSON string of new values
    changes_summary = db.Column(db.Text)
    
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def get_old_values(self):
        """Get old values as dictionary"""
        if self.old_values:
            return json.loads(self.old_values)
        return {}
    
    def set_old_values(self, values_dict):
        """Set old values from dictionary"""
        self.old_values = json.dumps(values_dict)
    
    def get_new_values(self):
        """Get new values as dictionary"""
        if self.new_values:
            return json.loads(self.new_values)
        return {}
    
    def set_new_values(self, values_dict):
        """Set new values from dictionary"""
        self.new_values = json.dumps(values_dict)
    
    def to_dict(self):
        return {
            'log_id': self.log_id,
            'action': self.action,
            'entity_type': self.entity_type,
            'entity_id': self.entity_id,
            'user': self.user.to_dict() if self.user else None,
            'ip_address': self.ip_address,
            'changes_summary': self.changes_summary,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }

class Report(db.Model):
    """Report model for generated analytics and reports"""
    __tablename__ = 'reports'
    
    id = db.Column(db.Integer, primary_key=True)
    report_id = db.Column(db.String(50), unique=True, nullable=False)
    name = db.Column(db.String(200), nullable=False)
    report_type = db.Column(db.String(50), nullable=False)  # analytics, summary, compliance, custom
    
    # Report content
    parameters = db.Column(db.Text)  # JSON string of report parameters
    results = db.Column(db.Text)  # JSON string of report results
    file_path = db.Column(db.String(500))  # Path to generated report file
    
    # Scheduling and automation
    is_scheduled = db.Column(db.Boolean, default=False)
    schedule_expression = db.Column(db.String(100))  # Cron-like expression
    next_run = db.Column(db.DateTime)
    
    # Relationships
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'))
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    creator = db.relationship('User', foreign_keys=[created_by])
    
    status = db.Column(db.String(20), default='pending')  # pending, generating, completed, failed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    
    def get_parameters(self):
        """Get report parameters as dictionary"""
        if self.parameters:
            return json.loads(self.parameters)
        return {}
    
    def set_parameters(self, params_dict):
        """Set report parameters from dictionary"""
        self.parameters = json.dumps(params_dict)
    
    def get_results(self):
        """Get report results as dictionary"""
        if self.results:
            return json.loads(self.results)
        return {}
    
    def set_results(self, results_dict):
        """Set report results from dictionary"""
        self.results = json.dumps(results_dict)
    
    def to_dict(self):
        return {
            'report_id': self.report_id,
            'name': self.name,
            'report_type': self.report_type,
            'status': self.status,
            'is_scheduled': self.is_scheduled,
            'schedule_expression': self.schedule_expression,
            'next_run': self.next_run.isoformat() if self.next_run else None,
            'project': self.project.to_dict() if self.project else None,
            'creator': self.creator.to_dict() if self.creator else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }

class Workflow(db.Model):
    """Workflow model for automated processes and pipelines"""
    __tablename__ = 'workflows'
    
    id = db.Column(db.Integer, primary_key=True)
    workflow_id = db.Column(db.String(50), unique=True, nullable=False)
    name = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    
    # Workflow definition
    definition = db.Column(db.Text)  # JSON string of workflow steps
    trigger_type = db.Column(db.String(50))  # manual, scheduled, event
    trigger_config = db.Column(db.Text)  # JSON string of trigger configuration
    
    # Status and tracking
    status = db.Column(db.String(20), default='active')  # active, paused, disabled
    version = db.Column(db.String(20), default='1.0')
    
    # Relationships
    project_id = db.Column(db.Integer, db.ForeignKey('projects.id'))
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'))
    creator = db.relationship('User', foreign_keys=[created_by])
    
    # Execution tracking
    last_run = db.Column(db.DateTime)
    next_run = db.Column(db.DateTime)
    run_count = db.Column(db.Integer, default=0)
    success_count = db.Column(db.Integer, default=0)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def get_definition(self):
        """Get workflow definition as dictionary"""
        if self.definition:
            return json.loads(self.definition)
        return {}
    
    def set_definition(self, definition_dict):
        """Set workflow definition from dictionary"""
        self.definition = json.dumps(definition_dict)
    
    def get_trigger_config(self):
        """Get trigger configuration as dictionary"""
        if self.trigger_config:
            return json.loads(self.trigger_config)
        return {}
    
    def set_trigger_config(self, config_dict):
        """Set trigger configuration from dictionary"""
        self.trigger_config = json.dumps(config_dict)
    
    def to_dict(self):
        return {
            'workflow_id': self.workflow_id,
            'name': self.name,
            'description': self.description,
            'status': self.status,
            'version': self.version,
            'trigger_type': self.trigger_type,
            'project': self.project.to_dict() if self.project else None,
            'creator': self.creator.to_dict() if self.creator else None,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'next_run': self.next_run.isoformat() if self.next_run else None,
            'run_count': self.run_count,
            'success_count': self.success_count,
            'success_rate': (self.success_count / self.run_count * 100) if self.run_count > 0 else 0,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
