#!/usr/bin/env python3
"""
Deployment Verification Script for CardioPredict Platform
This script verifies that all components are ready for production deployment on Render.
"""

import os
import sys
import importlib
import subprocess
import json
from pathlib import Path

def check_file_exists(filepath):
    """Check if a file exists and return its status."""
    return Path(filepath).exists()

def check_imports():
    """Check if all required packages can be imported."""
    required_packages = {
        'Flask': 'flask',
        'Flask-SQLAlchemy': 'flask_sqlalchemy',
        'Flask-JWT-Extended': 'flask_jwt_extended',
        'Flask-SocketIO': 'flask_socketio',
        'Celery': 'celery',
        'Redis': 'redis',
        'psutil': 'psutil',
        'gunicorn': 'gunicorn',
        'cryptography': 'cryptography',
    }
    
    import_status = {}
    for package_name, import_name in required_packages.items():
        try:
            importlib.import_module(import_name)
            import_status[package_name] = "Available"
        except ImportError as e:
            import_status[package_name] = f"Missing: {e}"
    
    return import_status

def check_environment_variables():
    """Check for required environment variables."""
    required_env_vars = {
        'SECRET_KEY': 'Application secret key',
        'JWT_SECRET_KEY': 'JWT secret key',
        'DATABASE_URL': 'Database connection URL (optional for SQLite)',
        'REDIS_URL': 'Redis connection URL (optional for local Redis)',
    }
    
    env_status = {}
    for var, description in required_env_vars.items():
        value = os.getenv(var)
        if value:
            env_status[var] = f"Set (length: {len(value)})"
        else:
            env_status[var] = f"Not set - {description}"
    
    return env_status

def check_application_files():
    """Check if all required application files exist."""
    base_path = Path("/Users/rahulgupta/Developer/CardioPredict/web_platform")
    required_files = [
        'app.py',
        'api.py',
        'api_phase2.py',
        'api_phase3.py',
        'models.py',
        'models_phase3.py',
        'config.py',
        'database_phase3.py',
        'websocket_server.py',
        'celery_tasks.py',
        'requirements.txt',
        'templates/index.html',
        'static/script.js',
        'static/styles.css',
    ]
    
    file_status = {}
    for file_path in required_files:
        full_path = base_path / file_path
        if full_path.exists():
            file_status[file_path] = f"Exists ({full_path.stat().st_size} bytes)"
        else:
            file_status[file_path] = "Missing"
    
    return file_status

def test_flask_app_creation():
    """Test if the Flask app can be created without errors."""
    try:
        # Add the web_platform directory to Python path
        sys.path.insert(0, "/Users/rahulgupta/Developer/CardioPredict/web_platform")
        
        # Try to import the app
        from app import app
        
        # Check if app was created successfully
        if app:
            return "Flask app imports successfully"
        else:
            return "Flask app import returned None"
            
    except Exception as e:
        return f"Flask app import failed: {str(e)}"

def check_requirements_file():
    """Verify requirements.txt has all necessary packages."""
    req_file = Path("/Users/rahulgupta/Developer/CardioPredict/web_platform/requirements.txt")
    if not req_file.exists():
        return {"status": "requirements.txt not found"}
    
    with open(req_file, 'r') as f:
        content = f.read()
    
    critical_packages = [
        'Flask',
        'Flask-SQLAlchemy',
        'Flask-JWT-Extended',
        'Flask-SocketIO',
        'celery',
        'redis',
        'gunicorn',
        'psutil'
    ]
    
    missing_packages = []
    for package in critical_packages:
        if package.lower() not in content.lower():
            missing_packages.append(package)
    
    if missing_packages:
        return {"status": f"Missing packages: {', '.join(missing_packages)}"}
    else:
        return {"status": "All critical packages present"}

def generate_render_deployment_config():
    """Generate render.yaml configuration for deployment."""
    render_config = {
        "services": [
            {
                "type": "web",
                "name": "cardiopredict-web",
                "env": "python",
                "buildCommand": "pip install -r requirements.txt",
                "startCommand": "gunicorn -w 4 -b 0.0.0.0:$PORT app:app",
                "envVars": [
                    {
                        "key": "PYTHON_VERSION",
                        "value": "3.9.16"
                    },
                    {
                        "key": "SECRET_KEY",
                        "generateValue": True
                    },
                    {
                        "key": "JWT_SECRET_KEY",
                        "generateValue": True
                    }
                ]
            }
        ]
    }
    
    config_path = Path("/Users/rahulgupta/Developer/CardioPredict/render.yaml")
    try:
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(render_config, f, default_flow_style=False)
        return f"Generated render.yaml at {config_path}"
    except ImportError:
        # Fallback to JSON if PyYAML not available
        with open(config_path.with_suffix('.json'), 'w') as f:
            json.dump(render_config, f, indent=2)
        return f"Generated render.json at {config_path.with_suffix('.json')}"

def main():
    """Run all deployment verification checks."""
    print("CardioPredict Deployment Verification")
    print("=" * 50)
    
    # Check imports
    print("\n📦 Package Import Status:")
    import_status = check_imports()
    for package, status in import_status.items():
        print(f"  {package}: {status}")
    
    # Check environment variables
    print("\nEnvironment Variables:")
    env_status = check_environment_variables()
    for var, status in env_status.items():
        print(f"  {var}: {status}")
    
    # Check application files
    print("\n📁 Application Files:")
    file_status = check_application_files()
    for file_path, status in file_status.items():
        print(f"  {file_path}: {status}")
    
    # Check requirements.txt
    print("\nRequirements File:")
    req_status = check_requirements_file()
    print(f"  {req_status['status']}")
    
    # Test Flask app creation
    print("\n🌐 Flask Application:")
    app_status = test_flask_app_creation()
    print(f"  {app_status}")
    
    # Generate Render config
    print("\n☁️  Render Deployment Config:")
    render_status = generate_render_deployment_config()
    print(f"  {render_status}")
    
    print("\n" + "=" * 50)
    print("Deployment verification complete!")
    print("\n📝 Next Steps for Render Deployment:")
    print("1. Push your code to a GitHub repository")
    print("2. Connect your GitHub repo to Render")
    print("3. Set environment variables in Render dashboard:")
    print("   - SECRET_KEY (generate a strong secret)")
    print("   - JWT_SECRET_KEY (generate another strong secret)")
    print("   - DATABASE_URL (optional, will use SQLite by default)")
    print("   - REDIS_URL (optional, will use in-memory fallback)")
    print("4. Deploy using the requirements.txt file")
    print("5. Test all endpoints after deployment")

if __name__ == "__main__":
    main()
