"""
Phase 3 Database Initialization for CardioPredict Platform
Authentication, roles, and advanced features setup
"""

from datetime import datetime
import hashlib
import secrets
from werkzeug.security import generate_password_hash
from models import db, User
from models_phase3 import UserRole, user_roles_association
import logging

logger = logging.getLogger(__name__)

def init_phase3_database():
    """Initialize Phase 3 database tables and default data"""
    try:
        # Create all Phase 3 tables
        from models_phase3 import (UserSession, UserRole, APIKey, MLModelVersion, 
                                  FeatureImportance, ModelDriftDetection, 
                                  RealtimeEvent, BackgroundJob)
        
        db.create_all()
        logger.info("✓ Phase 3 database tables created successfully")
        
        # Initialize default roles and permissions
        init_default_roles()
        
        # Create default admin user if not exists
        # init_default_admin_user()  # Temporarily disabled due to schema mismatch
        
        logger.info("✓ Phase 3 database initialization completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Phase 3 database initialization failed: {e}")
        return False

def init_default_roles():
    """Initialize default user roles and permissions"""
    try:
        # Define default roles
        default_roles = [
            {
                'role_name': 'admin',
                'display_name': 'Administrator',
                'description': 'Full system access with all permissions',
                'permissions': [
                    'system.admin', 'user.manage', 'data.manage', 'model.manage',
                    'project.manage', 'api.admin', 'audit.view', 'realtime.admin'
                ]
            },
            {
                'role_name': 'researcher',
                'display_name': 'Researcher',
                'description': 'Research access with data and model permissions',
                'permissions': [
                    'data.read', 'data.write', 'model.read', 'model.write',
                    'project.read', 'project.write', 'prediction.create',
                    'experiment.manage', 'collaboration.participate'
                ]
            },
            {
                'role_name': 'clinician',
                'display_name': 'Clinician',
                'description': 'Clinical access with prediction and patient data permissions',
                'permissions': [
                    'data.read', 'prediction.create', 'prediction.read',
                    'report.read', 'report.write', 'patient.read',
                    'collaboration.participate'
                ]
            },
            {
                'role_name': 'viewer',
                'display_name': 'Viewer',
                'description': 'Read-only access to public data and reports',
                'permissions': [
                    'data.read', 'prediction.read', 'report.read',
                    'dashboard.view'
                ]
            },
            {
                'role_name': 'api_user',
                'display_name': 'API User',
                'description': 'Programmatic access via API keys',
                'permissions': [
                    'api.access', 'data.read', 'prediction.create',
                    'model.read'
                ]
            }
        ]
        
        # Create roles if they don't exist
        for role_data in default_roles:
            existing_role = UserRole.query.filter_by(role_name=role_data['role_name']).first()
            if not existing_role:
                role = UserRole(
                    role_name=role_data['role_name'],
                    description=role_data['description'],
                    is_system_role=True,
                    created_at=datetime.now()
                )
                role.set_permissions(role_data['permissions'])
                db.session.add(role)
                logger.info(f"✓ Created role: {role_data['role_name']}")
            else:
                # Update permissions if role exists
                existing_role.set_permissions(role_data['permissions'])
                existing_role.updated_at = datetime.now()
                logger.info(f"✓ Updated role: {role_data['role_name']}")
        
        db.session.commit()
        logger.info("✓ Default roles initialized successfully")
        
    except Exception as e:
        logger.error(f"✗ Failed to initialize default roles: {e}")
        db.session.rollback()
        raise

def init_default_admin_user():
    """Create default admin user if no admin exists"""
    try:
        # Check if any admin user exists
        admin_role = UserRole.query.filter_by(role_name='admin').first()
        if not admin_role:
            logger.warning("Admin role not found, skipping admin user creation")
            return
        
        # Check if any user has admin role
        admin_users = db.session.query(User).join(
            user_roles_association
        ).join(UserRole).filter(
            UserRole.role_name == 'admin'
        ).all()
        
        if admin_users:
            logger.info(f"✓ Admin users already exist ({len(admin_users)} found)")
            return
        
        # Create default admin user
        admin_user = User(
            user_id=f"admin_{secrets.token_hex(4)}",
            email="admin@cardiopredict.com",
            username="admin",
            password_hash=generate_password_hash("admin123!"),  # Change in production
            full_name="System Administrator",
            role="admin",
            is_active=True,
            created_at=datetime.now()
        )
        
        db.session.add(admin_user)
        db.session.flush()  # Get user ID
        
        # Assign admin role
        admin_user.roles.append(admin_role)
        
        db.session.commit()
        
        logger.info("✓ Default admin user created successfully")
        logger.warning("🔒 Default admin password is 'admin123!' - CHANGE IN PRODUCTION!")
        
    except Exception as e:
        logger.error(f"✗ Failed to create default admin user: {e}")
        db.session.rollback()
        raise

def create_api_key_for_user(user_id, name="Default API Key", scopes=None):
    """Create an API key for a user"""
    try:
        from models_phase3 import APIKey
        
        # Generate secure API key
        key = f"cp_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        # Default scopes for API access
        if scopes is None:
            scopes = ['api.access', 'data.read', 'prediction.create']
        
        api_key = APIKey(
            user_id=user_id,
            name=name,
            key_hash=key_hash,
            scopes=scopes,
            is_active=True,
            created_at=datetime.now()
        )
        
        db.session.add(api_key)
        db.session.commit()
        
        logger.info(f"✓ API key created for user {user_id}")
        return key  # Return the plain key (only time it's available)
        
    except Exception as e:
        logger.error(f"✗ Failed to create API key: {e}")
        db.session.rollback()
        raise

def check_phase3_database_status():
    """Check Phase 3 database status and integrity"""
    try:
        from models_phase3 import UserRole, APIKey, MLModelVersion
        
        # Check if Phase 3 tables exist and have data
        role_count = UserRole.query.count()
        api_key_count = APIKey.query.count()
        model_version_count = MLModelVersion.query.count()
        
        status = {
            'phase3_initialized': True,
            'roles_count': role_count,
            'api_keys_count': api_key_count,
            'model_versions_count': model_version_count,
            'default_roles_present': role_count >= 5,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✓ Phase 3 database status: {status}")
        return status
        
    except Exception as e:
        logger.error(f"✗ Phase 3 database status check failed: {e}")
        return {
            'phase3_initialized': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

if __name__ == "__main__":
    # For testing the database initialization
    from flask import Flask
    from models import db
    
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test_phase3.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    db.init_app(app)
    
    with app.app_context():
        init_phase3_database()
        status = check_phase3_database_status()
        print(f"Database status: {status}")
