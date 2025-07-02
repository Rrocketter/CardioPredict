#!/usr/bin/env python3
"""
Test script to validate production deployment setup
"""

import os
import sys
import subprocess
import time

def test_production_setup():
    """Test the production configuration"""
    print("Testing CardioPredict Production Setup...")
    
    # Set environment variables
    os.environ['FLASK_ENV'] = 'production'
    os.environ['SECRET_KEY'] = 'test-production-key'
    os.environ['PORT'] = '5003'
    
    # Test imports
    try:
        print("✓ Testing imports...")
        
        # Add web_platform to path
        web_platform_path = '/Users/rahulgupta/Developer/CardioPredict/web_platform'
        sys.path.insert(0, web_platform_path)
        
        from config import config
        print("✓ Configuration loaded successfully")
        
        # Test Flask app creation
        from app import app
        print("✓ Flask app created successfully")
        
        # Test database initialization
        with app.app_context():
            from models import db
            print("✓ Database models loaded successfully")
        
        print("All production setup tests passed!")
        return True
        
    except Exception as e:
        print(f"Error in production setup: {e}")
        return False

if __name__ == '__main__':
    success = test_production_setup()
    sys.exit(0 if success else 1)
