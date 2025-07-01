import os
from datetime import timedelta

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'cardiopredict-scientific-platform-2025'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///cardiopredict.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Session configuration
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # File upload configuration
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Security headers
    SEND_FILE_MAX_AGE_DEFAULT = 31536000  # 1 year
    
    # Phase 3: JWT Configuration
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or 'cardiopredict-jwt-secret-2025'
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)
    JWT_ALGORITHM = 'HS256'
    
    # Phase 3: WebSocket Configuration
    SOCKETIO_ASYNC_MODE = 'threading'
    SOCKETIO_CORS_ALLOWED_ORIGINS = "*"  # Configure for production
    
    # Phase 3: Celery Configuration
    CELERY_BROKER_URL = os.environ.get('CELERY_BROKER_URL') or 'redis://localhost:6379/0'
    CELERY_RESULT_BACKEND = os.environ.get('CELERY_RESULT_BACKEND') or 'redis://localhost:6379/0'
    
    # Phase 3: Rate Limiting
    RATELIMIT_STORAGE_URL = os.environ.get('REDIS_URL') or 'redis://localhost:6379/1'
    RATELIMIT_DEFAULT = "1000 per hour"
    
    # Phase 3: API Configuration
    API_VERSION = "3.0"
    API_PAGINATION_DEFAULT = 25
    API_PAGINATION_MAX = 100
    
    # Phase 3: Model Configuration
    ML_MODEL_STORAGE = os.environ.get('MODEL_STORAGE_PATH') or '../models'
    MODEL_VERSIONING_ENABLED = True
    DRIFT_DETECTION_THRESHOLD = 0.1
    
    # Phase 3: Monitoring
    METRICS_ENABLED = True
    AUDIT_LOG_ENABLED = True
    
class DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///cardiopredict_dev.db'
    
    # Development-specific settings
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24)  # Longer for development
    SOCKETIO_CORS_ALLOWED_ORIGINS = "*"
    RATELIMIT_ENABLED = False  # Disable rate limiting in development

class ProductionConfig(Config):
    DEBUG = False
    # Use PostgreSQL for production if available
    if os.environ.get('DATABASE_URL'):
        SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL').replace('postgres://', 'postgresql://')
    
    # Production security settings
    SOCKETIO_CORS_ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', '').split(',')
    RATELIMIT_ENABLED = True
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(minutes=15)  # Shorter for production
    
    # Production performance settings
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 10,
        'pool_recycle': 120,
        'pool_pre_ping': True
    }

class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(minutes=5)
    RATELIMIT_ENABLED = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
