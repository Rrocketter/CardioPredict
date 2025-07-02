"""
Phase 3 API Extensions for CardioPredict Platform
Authentication, real-time features, advanced ML, and production capabilities
"""

from flask import Blueprint, jsonify, request, session, current_app
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, create_refresh_token, get_jwt_identity, get_jwt
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import json
import uuid
import secrets
import hashlib
from functools import wraps
from sqlalchemy import func, desc, and_, or_
from models import (db, User, Dataset, Prediction, Experiment, MLModel, Notification,
                   Project, Comment, AuditLog, Report, Workflow, project_members)
from models_phase3 import (UserSession, UserRole, APIKey, MLModelVersion, FeatureImportance,
                          ModelDriftDetection, RealtimeEvent, BackgroundJob, user_roles_association)

# Create Phase 3 API blueprint
api_v3 = Blueprint('api_v3', __name__, url_prefix='/api/v3')

# JWT Configuration
jwt = JWTManager()

# Rate limiting configuration (would use Redis in production)
request_counts = {}

def rate_limit(max_requests=100, window=3600):
    """Rate limiting decorator"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = request.environ.get('REMOTE_ADDR', 'unknown')
            current_time = datetime.now()
            window_start = current_time - timedelta(seconds=window)
            
            # Clean old entries
            if client_ip in request_counts:
                request_counts[client_ip] = [
                    timestamp for timestamp in request_counts[client_ip]
                    if timestamp > window_start
                ]
            else:
                request_counts[client_ip] = []
            
            # Check rate limit
            if len(request_counts[client_ip]) >= max_requests:
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'message': f'Maximum {max_requests} requests per {window} seconds allowed'
                }), 429
            
            # Record request
            request_counts[client_ip].append(current_time)
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def require_permissions(required_permissions):
    """Permission checking decorator"""
    def decorator(f):
        @wraps(f)
        @jwt_required()
        def decorated_function(*args, **kwargs):
            current_user_id = get_jwt_identity()
            user = User.query.filter_by(user_id=current_user_id).first()
            
            if not user:
                return jsonify({'error': 'User not found'}), 404
            
            # Check if user has required permissions
            user_permissions = get_user_permissions(user)
            
            for permission in required_permissions:
                if permission not in user_permissions:
                    return jsonify({
                        'error': 'Insufficient permissions',
                        'required': required_permissions,
                        'user_permissions': user_permissions
                    }), 403
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# Authentication Endpoints
@api_v3.route('/auth/register', methods=['POST'])
@rate_limit(max_requests=5, window=3600)  # 5 registrations per hour
def register():
    """Register a new user"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['user_id', 'name', 'email', 'password', 'role']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Check if user already exists
        existing_user = User.query.filter(
            or_(User.user_id == data['user_id'], User.email == data['email'])
        ).first()
        
        if existing_user:
            return jsonify({'error': 'User already exists'}), 409
        
        # Create new user
        user = User(
            user_id=data['user_id'],
            name=data['name'],
            email=data['email'],
            role=data['role'],
            department=data.get('department', ''),
            phone=data.get('phone', ''),
            timezone=data.get('timezone', 'UTC'),
            is_active=True
        )
        
        # Set password hash
        user.password_hash = generate_password_hash(data['password'])
        
        # Set preferences
        default_preferences = {
            'theme': 'light',
            'notifications_email': True,
            'notifications_desktop': False,
            'default_dashboard': 'overview',
            'timezone_display': data.get('timezone', 'UTC')
        }
        user.set_preferences(default_preferences)
        
        db.session.add(user)
        db.session.commit()
        
        # Assign default role
        default_role = UserRole.query.filter_by(role_name='researcher').first()
        if default_role:
            db.session.execute(
                user_roles_association.insert().values(
                    user_id=user.id,
                    role_id=default_role.id
                )
            )
            db.session.commit()
        
        # Create access and refresh tokens
        access_token = create_access_token(identity=user.user_id)
        refresh_token = create_refresh_token(identity=user.user_id)
        
        # Log registration
        log_audit_action('create', 'user', user.user_id, user.user_id)
        
        return jsonify({
            'message': 'User registered successfully',
            'user': user.to_dict(),
            'access_token': access_token,
            'refresh_token': refresh_token
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@api_v3.route('/auth/login', methods=['POST'])
@rate_limit(max_requests=10, window=900)  # 10 login attempts per 15 minutes
def login():
    """User login"""
    try:
        data = request.get_json()
        
        if not data.get('user_id') or not data.get('password'):
            return jsonify({'error': 'User ID and password required'}), 400
        
        user = User.query.filter_by(user_id=data['user_id']).first()
        
        if not user or not user.password_hash or not check_password_hash(user.password_hash, data['password']):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        if not user.is_active:
            return jsonify({'error': 'Account is deactivated'}), 401
        
        # Create session
        session_id = str(uuid.uuid4())
        user_session = UserSession(
            session_id=session_id,
            user_id=user.id,
            ip_address=request.environ.get('REMOTE_ADDR'),
            user_agent=request.environ.get('HTTP_USER_AGENT'),
            device_fingerprint=data.get('device_fingerprint')
        )
        
        db.session.add(user_session)
        
        # Update last login
        user.last_login = datetime.now()
        db.session.commit()
        
        # Create tokens
        access_token = create_access_token(identity=user.user_id)
        refresh_token = create_refresh_token(identity=user.user_id)
        
        # Log login
        log_audit_action('login', 'user', user.user_id, user.user_id)
        
        return jsonify({
            'message': 'Login successful',
            'user': user.to_dict(),
            'access_token': access_token,
            'refresh_token': refresh_token,
            'session_id': session_id,
            'permissions': get_user_permissions(user)
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@api_v3.route('/auth/logout', methods=['POST'])
@jwt_required()
def logout():
    """User logout"""
    try:
        current_user_id = get_jwt_identity()
        session_id = request.get_json().get('session_id')
        
        if session_id:
            user_session = UserSession.query.filter_by(session_id=session_id).first()
            if user_session:
                user_session.is_active = False
                user_session.logout_time = datetime.now()
                db.session.commit()
        
        # Log logout
        log_audit_action('logout', 'user', current_user_id, current_user_id)
        
        return jsonify({'message': 'Logout successful'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_v3.route('/auth/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh():
    """Refresh access token"""
    try:
        current_user_id = get_jwt_identity()
        new_token = create_access_token(identity=current_user_id)
        
        return jsonify({'access_token': new_token})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_v3.route('/auth/profile', methods=['GET'])
@jwt_required()
def get_profile():
    """Get user profile"""
    try:
        current_user_id = get_jwt_identity()
        user = User.query.filter_by(user_id=current_user_id).first()
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        profile_data = user.to_dict()
        profile_data['permissions'] = get_user_permissions(user)
        profile_data['active_sessions'] = len([
            s for s in user.sessions if s.is_active
        ])
        
        return jsonify(profile_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_v3.route('/auth/profile', methods=['PUT'])
@jwt_required()
def update_profile():
    """Update user profile"""
    try:
        current_user_id = get_jwt_identity()
        user = User.query.filter_by(user_id=current_user_id).first()
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        data = request.get_json()
        
        # Update allowed fields
        allowed_fields = ['name', 'email', 'phone', 'timezone', 'department']
        for field in allowed_fields:
            if field in data:
                setattr(user, field, data[field])
        
        # Update preferences
        if 'preferences' in data:
            current_prefs = user.get_preferences()
            current_prefs.update(data['preferences'])
            user.set_preferences(current_prefs)
        
        user.updated_at = datetime.now()
        db.session.commit()
        
        # Log profile update
        log_audit_action('update', 'user', user.user_id, user.user_id)
        
        return jsonify({
            'message': 'Profile updated successfully',
            'user': user.to_dict()
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# API Key Management
@api_v3.route('/auth/api-keys', methods=['GET'])
@jwt_required()
def get_api_keys():
    """Get user's API keys"""
    try:
        current_user_id = get_jwt_identity()
        user = User.query.filter_by(user_id=current_user_id).first()
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        api_keys = APIKey.query.filter_by(user_id=user.id).all()
        keys_data = [key.to_dict() for key in api_keys]
        
        return jsonify({'api_keys': keys_data})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_v3.route('/auth/api-keys', methods=['POST'])
@jwt_required()
def create_api_key():
    """Create new API key"""
    try:
        current_user_id = get_jwt_identity()
        user = User.query.filter_by(user_id=current_user_id).first()
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        data = request.get_json()
        
        # Generate API key
        raw_key = secrets.token_urlsafe(32)
        key_hash = generate_password_hash(raw_key)
        
        api_key = APIKey(
            key_id=f"KEY-{uuid.uuid4().hex[:8].upper()}",
            key_hash=key_hash,
            name=data['name'],
            description=data.get('description', ''),
            user_id=user.id,
            rate_limit=data.get('rate_limit', 1000)
        )
        
        # Set permissions
        if data.get('permissions'):
            api_key.set_permissions(data['permissions'])
        
        # Set expiration
        if data.get('expires_in_days'):
            api_key.expires_at = datetime.now() + timedelta(days=data['expires_in_days'])
        
        db.session.add(api_key)
        db.session.commit()
        
        # Log API key creation
        log_audit_action('create', 'api_key', api_key.key_id, user.user_id)
        
        return jsonify({
            'message': 'API key created successfully',
            'api_key': api_key.to_dict(),
            'key': raw_key  # Only returned once!
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# Advanced ML Endpoints
@api_v3.route('/ml/models/<model_id>/versions', methods=['GET'])
@jwt_required()
@require_permissions(['ml.read'])
def get_model_versions(model_id):
    """Get all versions of a model"""
    try:
        model = MLModel.query.filter_by(model_id=model_id).first()
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        versions = MLModelVersion.query.filter_by(model_id=model.id)\
                                     .order_by(desc(MLModelVersion.created_at)).all()
        
        versions_data = [version.to_dict() for version in versions]
        
        return jsonify({
            'model_id': model_id,
            'versions': versions_data,
            'total_versions': len(versions_data),
            'active_version': next((v for v in versions_data if v['is_active']), None)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_v3.route('/ml/models/<model_id>/versions', methods=['POST'])
@jwt_required()
@require_permissions(['ml.write'])
def create_model_version(model_id):
    """Create a new model version"""
    try:
        current_user_id = get_jwt_identity()
        user = User.query.filter_by(user_id=current_user_id).first()
        model = MLModel.query.filter_by(model_id=model_id).first()
        
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        data = request.get_json()
        
        # Generate version ID
        version_id = f"VER-{model_id}-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
        
        model_version = MLModelVersion(
            version_id=version_id,
            model_id=model.id,
            version_number=data['version_number'],
            version_name=data.get('version_name'),
            description=data.get('description'),
            model_path=data.get('model_path'),
            accuracy=data.get('accuracy'),
            precision=data.get('precision'),
            recall=data.get('recall'),
            f1_score=data.get('f1_score'),
            auc_score=data.get('auc_score'),
            training_data_size=data.get('training_data_size'),
            training_time=data.get('training_time'),
            created_by=user.id
        )
        
        # Set configuration and parameters
        if data.get('config'):
            model_version.set_config(data['config'])
        if data.get('parameters'):
            model_version.set_parameters(data['parameters'])
        if data.get('hyperparameters'):
            model_version.set_hyperparameters(data['hyperparameters'])
        if data.get('custom_metrics'):
            model_version.set_custom_metrics(data['custom_metrics'])
        
        db.session.add(model_version)
        db.session.commit()
        
        # Log version creation
        log_audit_action('create', 'model_version', version_id, user.user_id)
        
        return jsonify(model_version.to_dict()), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@api_v3.route('/ml/models/versions/<version_id>/deploy', methods=['POST'])
@jwt_required()
@require_permissions(['ml.deploy'])
def deploy_model_version(version_id):
    """Deploy a model version"""
    try:
        current_user_id = get_jwt_identity()
        model_version = MLModelVersion.query.filter_by(version_id=version_id).first()
        
        if not model_version:
            return jsonify({'error': 'Model version not found'}), 404
        
        # Deactivate current active version
        current_active = MLModelVersion.query.filter_by(
            model_id=model_version.model_id,
            is_active=True
        ).first()
        
        if current_active:
            current_active.is_active = False
            current_active.retirement_date = datetime.now()
        
        # Activate new version
        model_version.is_active = True
        model_version.status = 'deployed'
        model_version.deployment_date = datetime.now()
        
        db.session.commit()
        
        # Create deployment event
        create_realtime_event(
            event_type='model_deployed',
            entity_type='model_version',
            entity_id=version_id,
            message=f"Model version {model_version.version_number} deployed",
            broadcast=True,
            created_by=current_user_id
        )
        
        # Log deployment
        log_audit_action('deploy', 'model_version', version_id, current_user_id)
        
        return jsonify({
            'message': 'Model version deployed successfully',
            'version': model_version.to_dict()
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@api_v3.route('/ml/models/versions/<version_id>/feature-importance', methods=['GET'])
@jwt_required()
@require_permissions(['ml.read'])
def get_feature_importance(version_id):
    """Get feature importance for a model version"""
    try:
        model_version = MLModelVersion.query.filter_by(version_id=version_id).first()
        if not model_version:
            return jsonify({'error': 'Model version not found'}), 404
        
        feature_importances = FeatureImportance.query.filter_by(
            model_version_id=model_version.id
        ).order_by(desc(FeatureImportance.importance_score)).all()
        
        importances_data = [fi.to_dict() for fi in feature_importances]
        
        # Calculate statistics
        if importances_data:
            total_importance = sum(fi['importance_score'] for fi in importances_data)
            for fi in importances_data:
                fi['percentage'] = round((fi['importance_score'] / total_importance) * 100, 2)
        
        return jsonify({
            'version_id': version_id,
            'feature_importances': importances_data,
            'total_features': len(importances_data),
            'analysis_methods': list(set(fi['analysis_method'] for fi in importances_data))
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_v3.route('/ml/models/versions/<version_id>/drift-detection', methods=['POST'])
@jwt_required()
@require_permissions(['ml.monitor'])
def perform_drift_detection(version_id):
    """Perform drift detection on a model version"""
    try:
        current_user_id = get_jwt_identity()
        model_version = MLModelVersion.query.filter_by(version_id=version_id).first()
        
        if not model_version:
            return jsonify({'error': 'Model version not found'}), 404
        
        data = request.get_json()
        
        # Create background job for drift detection
        job_id = f"DRIFT-{version_id}-{uuid.uuid4().hex[:8].upper()}"
        
        job = BackgroundJob(
            job_id=job_id,
            job_type='drift_detection',
            job_name=f"Drift Detection - {model_version.version_number}",
            description=f"Performing drift detection analysis for model version {version_id}",
            status='pending',
            created_by=User.query.filter_by(user_id=current_user_id).first().id
        )
        
        job.set_parameters({
            'model_version_id': version_id,
            'analysis_period_days': data.get('analysis_period_days', 30),
            'drift_threshold': data.get('drift_threshold', 0.1),
            'methods': data.get('methods', ['ks_test', 'psi'])
        })
        
        db.session.add(job)
        db.session.commit()
        
        # In a real implementation, this would trigger a Celery task
        # For now, we'll simulate the analysis
        simulate_drift_detection(job.id, model_version.id)
        
        return jsonify({
            'message': 'Drift detection analysis started',
            'job_id': job_id,
            'estimated_duration': 300  # 5 minutes
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# Real-time Events
@api_v3.route('/realtime/events', methods=['GET'])
@jwt_required()
def get_realtime_events():
    """Get recent real-time events for the user"""
    try:
        current_user_id = get_jwt_identity()
        user = User.query.filter_by(user_id=current_user_id).first()
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        # Get events for the user
        hours = request.args.get('hours', 24, type=int)
        since = datetime.now() - timedelta(hours=hours)
        
        events = RealtimeEvent.query.filter(
            and_(
                RealtimeEvent.created_at >= since,
                or_(
                    RealtimeEvent.user_id == user.id,
                    RealtimeEvent.broadcast == True,
                    RealtimeEvent.project_id.in_([p.id for p in user.projects])
                )
            )
        ).order_by(desc(RealtimeEvent.created_at)).limit(50).all()
        
        events_data = [event.to_dict() for event in events]
        
        return jsonify({
            'events': events_data,
            'total': len(events_data),
            'since': since.isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Background Jobs Management
@api_v3.route('/jobs', methods=['GET'])
@jwt_required()
def get_background_jobs():
    """Get user's background jobs"""
    try:
        current_user_id = get_jwt_identity()
        user = User.query.filter_by(user_id=current_user_id).first()
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        status = request.args.get('status')
        job_type = request.args.get('type')
        
        query = BackgroundJob.query.filter_by(created_by=user.id)
        
        if status:
            query = query.filter(BackgroundJob.status == status)
        if job_type:
            query = query.filter(BackgroundJob.job_type == job_type)
        
        jobs = query.order_by(desc(BackgroundJob.created_at)).limit(50).all()
        jobs_data = [job.to_dict() for job in jobs]
        
        return jsonify({
            'jobs': jobs_data,
            'total': len(jobs_data)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_v3.route('/jobs/<job_id>', methods=['GET'])
@jwt_required()
def get_job_status(job_id):
    """Get job status and details"""
    try:
        job = BackgroundJob.query.filter_by(job_id=job_id).first()
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        
        # Check if user has access to this job
        current_user_id = get_jwt_identity()
        user = User.query.filter_by(user_id=current_user_id).first()
        
        if job.created_by != user.id:
            return jsonify({'error': 'Access denied'}), 403
        
        return jsonify(job.to_dict())
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Helper Functions
def get_user_permissions(user):
    """Get all permissions for a user"""
    permissions = set()
    
    # Get roles and their permissions
    user_roles = db.session.query(UserRole).join(
        user_roles_association,
        UserRole.id == user_roles_association.c.role_id
    ).filter(user_roles_association.c.user_id == user.id).all()
    
    for role in user_roles:
        permissions.update(role.get_permissions())
    
    return list(permissions)

def log_audit_action(action, entity_type, entity_id, user_id=None, changes_summary=None):
    """Enhanced audit logging"""
    try:
        audit_log = AuditLog(
            log_id=f"AUDIT-{uuid.uuid4().hex[:8].upper()}",
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            changes_summary=changes_summary or f"{action.title()} {entity_type}",
            ip_address=request.environ.get('REMOTE_ADDR', 'unknown'),
            user_agent=request.environ.get('HTTP_USER_AGENT', 'unknown')
        )
        
        if user_id:
            user = User.query.filter_by(user_id=user_id).first()
            if user:
                audit_log.user = user
        
        db.session.add(audit_log)
        db.session.commit()
        
    except Exception as e:
        print(f"Audit logging failed: {e}")

def create_realtime_event(event_type, entity_type, entity_id, message, 
                         user_id=None, project_id=None, broadcast=False, 
                         created_by=None, event_data=None):
    """Create a real-time event"""
    try:
        event = RealtimeEvent(
            event_id=f"EVENT-{uuid.uuid4().hex[:8].upper()}",
            event_type=event_type,
            entity_type=entity_type,
            entity_id=entity_id,
            message=message,
            broadcast=broadcast
        )
        
        if user_id:
            user = User.query.filter_by(user_id=user_id).first()
            if user:
                event.user_id = user.id
        
        if project_id:
            project = Project.query.filter_by(project_id=project_id).first()
            if project:
                event.project_id = project.id
        
        if created_by:
            creator = User.query.filter_by(user_id=created_by).first()
            if creator:
                event.created_by = creator.id
        
        if event_data:
            event.set_event_data(event_data)
        
        db.session.add(event)
        db.session.commit()
        
        return event
        
    except Exception as e:
        print(f"Failed to create real-time event: {e}")
        return None

def simulate_drift_detection(job_id, model_version_id):
    """Simulate drift detection analysis (would be a Celery task in production)"""
    try:
        job = BackgroundJob.query.get(job_id)
        model_version = MLModelVersion.query.get(model_version_id)
        
        if not job or not model_version:
            return
        
        # Update job status
        job.status = 'running'
        job.started_at = datetime.now()
        job.progress = 0
        db.session.commit()
        
        # Simulate analysis (in real implementation, this would be actual drift detection)
        import time
        import random
        
        for i in range(5):
            time.sleep(1)  # Simulate processing time
            job.progress = (i + 1) * 20
            db.session.commit()
        
        # Create drift detection result
        drift_id = f"DRIFT-{uuid.uuid4().hex[:8].upper()}"
        drift_detection = ModelDriftDetection(
            drift_id=drift_id,
            model_version_id=model_version.id,
            drift_type='data_drift',
            drift_score=round(random.uniform(0.05, 0.15), 3),
            threshold=0.1,
            is_drift_detected=random.choice([True, False]),
            analysis_period_start=datetime.now() - timedelta(days=30),
            analysis_period_end=datetime.now(),
            baseline_data_size=1000,
            current_data_size=950,
            recommendation="Continue monitoring. No immediate action required.",
            action_required="monitor"
        )
        
        # Set mock statistical tests and feature drift scores
        drift_detection.set_statistical_tests({
            'ks_test': {'p_value': 0.15, 'statistic': 0.08},
            'psi': {'score': 0.12, 'threshold': 0.1}
        })
        
        drift_detection.set_feature_drift_scores({
            'crp': 0.05,
            'tnf_alpha': 0.12,
            'troponin': 0.03,
            'il6': 0.08,
            'pf4': 0.15
        })
        
        db.session.add(drift_detection)
        
        # Complete job
        job.status = 'completed'
        job.progress = 100
        job.completed_at = datetime.now()
        job.set_result({
            'drift_detection_id': drift_id,
            'drift_detected': drift_detection.is_drift_detected,
            'drift_score': drift_detection.drift_score,
            'action_required': drift_detection.action_required
        })
        
        db.session.commit()
        
        # Create real-time event
        create_realtime_event(
            event_type='drift_analysis_completed',
            entity_type='model_version',
            entity_id=model_version.version_id,
            message=f"Drift analysis completed for {model_version.version_number}",
            event_data={'job_id': job.job_id, 'drift_detected': drift_detection.is_drift_detected}
        )
        
    except Exception as e:
        # Mark job as failed
        if job:
            job.status = 'failed'
            job.error_message = str(e)
            job.completed_at = datetime.now()
            db.session.commit()
        
        print(f"Drift detection simulation failed: {e}")

# Role and Permission Management
@api_v3.route('/admin/roles', methods=['GET'])
@jwt_required()
@require_permissions(['admin.read'])
def get_roles():
    """Get all user roles"""
    try:
        roles = UserRole.query.all()
        roles_data = [role.to_dict() for role in roles]
        
        return jsonify({'roles': roles_data})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_v3.route('/admin/users/<user_id>/roles', methods=['POST'])
@jwt_required()
@require_permissions(['admin.users'])
def assign_user_role(user_id):
    """Assign role to user"""
    try:
        user = User.query.filter_by(user_id=user_id).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        data = request.get_json()
        role = UserRole.query.filter_by(role_name=data['role_name']).first()
        
        if not role:
            return jsonify({'error': 'Role not found'}), 404
        
        # Check if assignment already exists
        existing = db.session.query(user_roles_association).filter_by(
            user_id=user.id,
            role_id=role.id
        ).first()
        
        if existing:
            return jsonify({'error': 'Role already assigned'}), 409
        
        # Assign role
        db.session.execute(
            user_roles_association.insert().values(
                user_id=user.id,
                role_id=role.id
            )
        )
        db.session.commit()
        
        current_user_id = get_jwt_identity()
        log_audit_action('assign_role', 'user', user_id, current_user_id,
                        f"Assigned role {role.role_name}")
        
        return jsonify({'message': 'Role assigned successfully'})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
