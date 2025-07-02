"""
Phase 2 Database initialization and enhanced sample data generation for CardioPredict
Includes advanced project management, collaboration, and analytics features
"""

import os
import random
from datetime import datetime, timedelta
from models import (db, User, Dataset, Prediction, Experiment, MLModel, Notification,
                   Project, Comment, AuditLog, Report, Workflow, project_members)

def init_phase2_database(app):
    """Initialize database with Phase 2 enhanced sample data"""
    with app.app_context():
        # Create all tables (including new Phase 2 tables)
        db.create_all()
        
        # Check if database is already populated
        if User.query.count() > 0:
            print("Database already populated, skipping initialization.")
            return
        
        print("Initializing Phase 2 database with enhanced sample data...")
        
        # Create enhanced users with additional fields
        create_enhanced_users()
        
        # Create sample projects
        create_sample_projects()
        
        # Assign users to projects
        create_project_memberships()
        
        # Create enhanced datasets with project relationships
        create_enhanced_datasets()
        
        # Create enhanced ML models
        create_enhanced_models()
        
        # Create enhanced predictions
        create_enhanced_predictions()
        
        # Create enhanced experiments with project relationships
        create_enhanced_experiments()
        
        # Create sample comments for collaboration
        create_sample_comments()
        
        # Create audit logs for tracking
        create_sample_audit_logs()
        
        # Create sample reports
        create_sample_reports()
        
        # Create sample workflows
        create_sample_workflows()
        
        # Create enhanced notifications
        create_enhanced_notifications()
        
        # Commit all changes
        db.session.commit()
        print("Phase 2 database initialized successfully!")

def create_enhanced_users():
    """Create enhanced team members with additional Phase 2 fields"""
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
            'joined': datetime(2024, 1, 15),
            'email': 'sarah.chen@cardiopredict.org',
            'phone': '+1-555-0101',
            'timezone': 'America/New_York',
            'last_login': datetime.now() - timedelta(hours=2)
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
            'joined': datetime(2024, 3, 20),
            'email': 'michael.rodriguez@cardiopredict.org',
            'phone': '+1-555-0102',
            'timezone': 'America/Los_Angeles',
            'last_login': datetime.now() - timedelta(minutes=30)
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
            'joined': datetime(2024, 2, 10),
            'email': 'emily.watson@cardiopredict.org',
            'phone': '+1-555-0103',
            'timezone': 'Europe/London',
            'last_login': datetime.now() - timedelta(hours=8)
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
            'joined': datetime(2024, 4, 5),
            'email': 'james.parker@cardiopredict.org',
            'phone': '+1-555-0104',
            'timezone': 'America/Chicago',
            'last_login': datetime.now() - timedelta(days=1)
        },
        {
            'user_id': 'lisa.kim',
            'name': 'Dr. Lisa Kim',
            'role': 'Space Medicine Researcher',
            'department': 'Space Medicine',
            'avatar': 'https://images.unsplash.com/photo-1438761681033-6461ffad8d80?w=64&h=64&fit=crop&crop=face',
            'status': 'online',
            'projects': 2,
            'experiments': 12,
            'joined': datetime(2024, 5, 15),
            'email': 'lisa.kim@cardiopredict.org',
            'phone': '+1-555-0105',
            'timezone': 'Asia/Seoul',
            'last_login': datetime.now() - timedelta(hours=1)
        }
    ]
    
    for user_data in users:
        # Set user preferences
        preferences = {
            'theme': random.choice(['light', 'dark']),
            'notifications_email': True,
            'notifications_desktop': random.choice([True, False]),
            'default_dashboard': random.choice(['overview', 'experiments', 'predictions']),
            'timezone_display': user_data['timezone']
        }
        
        user = User(**user_data)
        user.set_preferences(preferences)
        db.session.add(user)

def create_sample_projects():
    """Create sample research projects"""
    projects = [
        {
            'project_id': 'PROJ-MARS-2027',
            'name': 'Mars Mission 2027 Cardiovascular Study',
            'description': 'Comprehensive cardiovascular risk assessment for the upcoming Mars mission crew, including predictive modeling and countermeasure development.',
            'status': 'active',
            'priority': 'critical',
            'start_date': datetime(2024, 1, 1),
            'end_date': datetime(2027, 6, 30),
            'budget': 2500000.0,
            'funding_source': 'NASA Space Medicine Division',
            'progress': 35,
            'milestones_completed': 7,
            'milestones_total': 20
        },
        {
            'project_id': 'PROJ-ISS-CARDIO',
            'name': 'ISS Cardiovascular Monitoring Platform',
            'description': 'Real-time cardiovascular health monitoring system for International Space Station crew members using AI-powered biomarker analysis.',
            'status': 'active',
            'priority': 'high',
            'start_date': datetime(2024, 6, 1),
            'end_date': datetime(2025, 12, 31),
            'budget': 1200000.0,
            'funding_source': 'ESA Human Spaceflight Program',
            'progress': 65,
            'milestones_completed': 13,
            'milestones_total': 18
        },
        {
            'project_id': 'PROJ-BEDREST-AI',
            'name': 'AI-Enhanced Bedrest Study Analysis',
            'description': 'Machine learning analysis of cardiovascular changes during prolonged bedrest studies as Earth analogs for spaceflight.',
            'status': 'active',
            'priority': 'medium',
            'start_date': datetime(2024, 3, 15),
            'end_date': datetime(2025, 9, 15),
            'budget': 850000.0,
            'funding_source': 'DLR Space Medicine Research',
            'progress': 78,
            'milestones_completed': 14,
            'milestones_total': 16
        },
        {
            'project_id': 'PROJ-LUNAR-GATEWAY',
            'name': 'Lunar Gateway Health Protocols',
            'description': 'Development of cardiovascular health protocols and monitoring systems for Lunar Gateway operations.',
            'status': 'active',
            'priority': 'high',
            'start_date': datetime(2024, 9, 1),
            'end_date': datetime(2026, 8, 31),
            'budget': 1800000.0,
            'funding_source': 'NASA Artemis Program',
            'progress': 22,
            'milestones_completed': 4,
            'milestones_total': 22
        },
        {
            'project_id': 'PROJ-DEEP-SPACE',
            'name': 'Deep Space Mission Preparedness',
            'description': 'Cardiovascular risk prediction and mitigation strategies for extended deep space missions beyond Earth orbit.',
            'status': 'on_hold',
            'priority': 'medium',
            'start_date': datetime(2025, 1, 1),
            'end_date': datetime(2028, 12, 31),
            'budget': 3200000.0,
            'funding_source': 'NASA Deep Space Exploration',
            'progress': 5,
            'milestones_completed': 1,
            'milestones_total': 25
        }
    ]
    
    # Get users for project lead assignment
    users = User.query.all()
    
    for i, project_data in enumerate(projects):
        project = Project(**project_data)
        # Assign project lead (rotate among users)
        project.lead = users[i % len(users)]
        db.session.add(project)

def create_project_memberships():
    """Create project memberships with roles and permissions"""
    projects = Project.query.all()
    users = User.query.all()
    
    for project in projects:
        # Add 3-5 team members to each project
        team_size = random.randint(3, 5)
        selected_users = random.sample(users, team_size)
        
        for user in selected_users:
            if user not in project.team_members:
                project.team_members.append(user)

def create_enhanced_datasets():
    """Create enhanced datasets with project relationships"""
    projects = Project.query.all()
    users = User.query.all()
    
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
            'last_updated': datetime.now() - timedelta(days=5),
            'version': '2.1',
            'access_level': 'internal',
            'tags': ['spaceflight', 'rna-seq', 'cardiovascular', 'inspiration4']
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
            'last_updated': datetime.now() - timedelta(days=8),
            'version': '1.3',
            'access_level': 'public',
            'tags': ['microgravity', 'microarray', 'cardiac', 'gene-expression']
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
            'last_updated': datetime.now() - timedelta(days=2),
            'version': '1.0',
            'access_level': 'restricted',
            'tags': ['bedrest', 'biomarkers', 'analog', 'longitudinal']
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
            'last_updated': datetime.now() - timedelta(days=10),
            'version': '3.0',
            'access_level': 'internal',
            'tags': ['multi-omics', 'long-duration', 'spaceflight', 'comprehensive']
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
            'last_updated': datetime.now() - timedelta(days=15),
            'version': '2.2',
            'access_level': 'internal',
            'tags': ['iss', 'cardiovascular', 'monitoring', 'real-time']
        }
    ]
    
    for i, dataset_data in enumerate(datasets):
        # Extract tags and metadata separately
        tags = dataset_data.pop('tags', [])
        
        dataset = Dataset(**dataset_data)
        # Assign to projects and uploaders
        dataset.project = projects[i % len(projects)]
        dataset.uploader = users[i % len(users)]
        
        # Set tags using the setter method
        dataset.set_tags(tags)
        
        # Set enhanced metadata
        metadata = {
            'collection_date': (datetime.now() - timedelta(days=random.randint(30, 365))).isoformat(),
            'processing_pipeline': random.choice(['v1.2', 'v2.0', 'v2.1']),
            'quality_score': round(random.uniform(85, 98), 1),
            'data_format': random.choice(['HDF5', 'CSV', 'Parquet', 'JSON']),
            'compression': random.choice(['gzip', 'lz4', 'snappy']),
            'checksum': f"sha256:{random.randint(10**63, 10**64-1):064x}"
        }
        dataset.set_metadata(metadata)
        
        db.session.add(dataset)

def create_enhanced_models():
    """Create enhanced ML models with additional metadata"""
    projects = Project.query.all()
    
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

def create_enhanced_predictions():
    """Create enhanced cardiovascular predictions with project context"""
    environments = ['Space Station', 'Mars Mission', 'Lunar Gateway', 'Bedrest Study', 'Clinical']
    projects = Project.query.all()
    
    for i in range(40):  # Increased from 30 to 40
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
            created_at=datetime.now() - timedelta(days=random.randint(1, 60))
        )
        
        prediction.set_biomarkers(biomarkers)
        prediction.set_risk_factors(risk_factors)
        
        db.session.add(prediction)

def create_enhanced_experiments():
    """Create enhanced experiments with project relationships"""
    experiment_names = [
        'Microgravity Cardiac Adaptation ML Pipeline',
        'Bedrest Biomarker Deep Learning Analysis',
        'Deep Space Radiation Effects Prediction',
        'Lunar Surface Cardiovascular Risk Assessment',
        'Mars Mission Crew Health Forecasting',
        'ISS Real-time Health Monitoring',
        'Spaceflight Countermeasures Optimization',
        'Multi-omics Integration for Space Medicine',
        'Predictive Modeling for Long-duration Missions'
    ]
    
    statuses = ['Running', 'Completed', 'Failed', 'Paused']
    projects = Project.query.all()
    users = User.query.all()
    
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
            created_by=random.choice(['sarah.chen', 'michael.rodriguez', 'emily.watson']),
            cpu_hours=round(random.uniform(10, 500), 2),
            memory_peak=round(random.uniform(8, 64), 1),
            gpu_hours=round(random.uniform(5, 200), 2),
            code_version=f"v{random.randint(1, 3)}.{random.randint(0, 9)}.{random.randint(0, 9)}"
        )
        
        # Assign to project and creator
        experiment.project = projects[i % len(projects)]
        experiment.creator = users[i % len(users)]
        
        # Set experiment configuration
        config = {
            'model_type': random.choice(['RandomForest', 'XGBoost', 'NeuralNetwork', 'SVM']),
            'hyperparameters': {
                'learning_rate': round(random.uniform(0.001, 0.1), 4),
                'batch_size': random.choice([16, 32, 64, 128]),
                'regularization': round(random.uniform(0.0001, 0.01), 5)
            },
            'data_split': {'train': 0.7, 'validation': 0.15, 'test': 0.15},
            'cross_validation': random.choice(['5-fold', '10-fold', 'stratified'])
        }
        experiment.set_config(config)
        
        # Set environment details
        environment = {
            'python_version': random.choice(['3.8.10', '3.9.7', '3.10.4']),
            'cuda_version': random.choice(['11.2', '11.4', '11.6']),
            'frameworks': {
                'sklearn': '1.0.2',
                'tensorflow': '2.8.0',
                'pytorch': '1.11.0'
            },
            'compute_resources': {
                'cpu_cores': random.choice([8, 16, 32]),
                'gpu_type': random.choice(['V100', 'A100', 'RTX3090']),
                'memory_gb': random.choice([32, 64, 128])
            }
        }
        experiment.set_environment(environment)
        
        db.session.add(experiment)

def create_sample_comments():
    """Create sample comments for collaboration"""
    users = User.query.all()
    projects = Project.query.all()
    predictions = Prediction.query.limit(10).all()
    experiments = Experiment.query.limit(5).all()
    
    comments_data = [
        {
            'content': 'The biomarker patterns in this prediction are particularly interesting. The elevated TNF-alpha levels suggest significant inflammatory response.',
            'entity_type': 'prediction',
            'tags': ['biomarkers', 'inflammation']
        },
        {
            'content': 'This experiment configuration looks promising. Have we considered adjusting the learning rate for better convergence?',
            'entity_type': 'experiment',
            'tags': ['machine-learning', 'optimization']
        },
        {
            'content': 'Great progress on the Mars mission risk assessment. The risk stratification model is showing excellent performance.',
            'entity_type': 'project',
            'tags': ['mars-mission', 'risk-assessment']
        },
        {
            'content': 'We should schedule a team meeting to discuss the latest results from the ISS cardiovascular monitoring project.',
            'entity_type': 'project',
            'tags': ['team-meeting', 'iss-project']
        },
        {
            'content': 'The accuracy improvements in this latest model iteration are significant. Ready for deployment testing.',
            'entity_type': 'experiment',
            'tags': ['model-performance', 'deployment']
        }
    ]
    
    for i, comment_data in enumerate(comments_data):
        comment = Comment(
            comment_id=f'COMM-{1000 + i}',
            content=comment_data['content'],
            entity_type=comment_data['entity_type'],
            author=users[i % len(users)],
            project=projects[i % len(projects)]
        )
        
        # Set entity_id based on type
        if comment_data['entity_type'] == 'prediction' and predictions:
            comment.entity_id = predictions[i % len(predictions)].prediction_id
        elif comment_data['entity_type'] == 'experiment' and experiments:
            comment.entity_id = experiments[i % len(experiments)].experiment_id
        elif comment_data['entity_type'] == 'project':
            comment.entity_id = projects[i % len(projects)].project_id
        
        comment.set_tags(comment_data['tags'])
        db.session.add(comment)

def create_sample_audit_logs():
    """Create sample audit logs for system tracking"""
    users = User.query.all()
    actions = ['create', 'update', 'delete', 'view', 'export']
    entity_types = ['prediction', 'experiment', 'dataset', 'project', 'model']
    
    for i in range(20):
        audit_log = AuditLog(
            log_id=f'AUDIT-{10000 + i}',
            action=random.choice(actions),
            entity_type=random.choice(entity_types),
            entity_id=f'ENT-{random.randint(1000, 9999)}',
            user=users[i % len(users)],
            ip_address=f"192.168.1.{random.randint(100, 200)}",
            user_agent='Mozilla/5.0 (CardioPredict Platform)',
            session_id=f'sess_{random.randint(100000, 999999)}',
            changes_summary=f'Modified {random.choice(entity_types)} configuration',
            timestamp=datetime.now() - timedelta(hours=random.randint(1, 168))
        )
        
        # Set sample old/new values
        old_values = {'status': 'active', 'progress': random.randint(0, 50)}
        new_values = {'status': 'updated', 'progress': random.randint(51, 100)}
        
        audit_log.set_old_values(old_values)
        audit_log.set_new_values(new_values)
        
        db.session.add(audit_log)

def create_sample_reports():
    """Create sample reports for analytics"""
    projects = Project.query.all()
    users = User.query.all()
    
    reports_data = [
        {
            'report_id': 'RPT-MARS-001',
            'name': 'Mars Mission Quarterly Risk Assessment',
            'report_type': 'analytics',
            'is_scheduled': True,
            'schedule_expression': '0 0 1 */3 *'  # Quarterly
        },
        {
            'report_id': 'RPT-ISS-002',
            'name': 'ISS Health Monitoring Summary',
            'report_type': 'summary',
            'is_scheduled': True,
            'schedule_expression': '0 0 * * 1'  # Weekly
        },
        {
            'report_id': 'RPT-COMP-003',
            'name': 'Regulatory Compliance Report',
            'report_type': 'compliance',
            'is_scheduled': True,
            'schedule_expression': '0 0 1 * *'  # Monthly
        },
        {
            'report_id': 'RPT-CUSTOM-004',
            'name': 'Biomarker Trends Analysis',
            'report_type': 'custom',
            'is_scheduled': False,
            'schedule_expression': None
        }
    ]
    
    for i, report_data in enumerate(reports_data):
        report = Report(
            **report_data,
            status=random.choice(['completed', 'pending', 'generating']),
            project=projects[i % len(projects)],
            creator=users[i % len(users)],
            created_at=datetime.now() - timedelta(days=random.randint(1, 30)),
            next_run=datetime.now() + timedelta(days=random.randint(1, 30)) if report_data['is_scheduled'] else None
        )
        
        # Set sample parameters
        parameters = {
            'date_range': '30_days',
            'include_predictions': True,
            'include_experiments': True,
            'export_format': 'PDF'
        }
        report.set_parameters(parameters)
        
        db.session.add(report)

def create_sample_workflows():
    """Create sample workflows for automation"""
    projects = Project.query.all()
    users = User.query.all()
    
    workflows_data = [
        {
            'workflow_id': 'WF-AUTO-001',
            'name': 'Automated Model Training Pipeline',
            'description': 'Automated pipeline for training and evaluating cardiovascular prediction models',
            'trigger_type': 'scheduled',
            'status': 'active',
            'version': '2.1'
        },
        {
            'workflow_id': 'WF-DATA-002',
            'name': 'Data Processing and Validation',
            'description': 'Automated data ingestion, cleaning, and validation workflow',
            'trigger_type': 'event',
            'status': 'active',
            'version': '1.5'
        },
        {
            'workflow_id': 'WF-REPORT-003',
            'name': 'Weekly Report Generation',
            'description': 'Automated generation and distribution of weekly progress reports',
            'trigger_type': 'scheduled',
            'status': 'active',
            'version': '1.0'
        }
    ]
    
    for i, workflow_data in enumerate(workflows_data):
        workflow = Workflow(**workflow_data)
        workflow.project_id = projects[i % len(projects)].id
        workflow.created_by = users[i % len(users)].id
        workflow.last_run = datetime.now() - timedelta(days=random.randint(1, 7))
        workflow.next_run = datetime.now() + timedelta(days=random.randint(1, 7))
        workflow.run_count = random.randint(50, 200)
        workflow.success_count = random.randint(45, 190)
        
        # Set workflow definition
        definition = {
            'steps': [
                {'name': 'data_validation', 'type': 'validation', 'timeout': 300},
                {'name': 'model_training', 'type': 'ml_training', 'timeout': 3600},
                {'name': 'evaluation', 'type': 'evaluation', 'timeout': 600},
                {'name': 'deployment', 'type': 'deployment', 'timeout': 300}
            ],
            'error_handling': 'retry_with_backoff',
            'notifications': ['email', 'dashboard']
        }
        workflow.set_definition(definition)
        
        # Set trigger configuration
        trigger_config = {
            'schedule': '0 2 * * *' if workflow_data['trigger_type'] == 'scheduled' else None,
            'events': ['dataset_updated', 'model_ready'] if workflow_data['trigger_type'] == 'event' else None
        }
        workflow.set_trigger_config(trigger_config)
        
        db.session.add(workflow)

def create_enhanced_notifications():
    """Create enhanced notifications with project context"""
    users = User.query.all()
    projects = Project.query.all()
    
    notifications = [
        {
            'notification_id': 'notif-001',
            'notification_type': 'experiment_complete',
            'title': 'Mars Mission Model Training Completed',
            'message': 'The cardiovascular risk prediction model for Mars Mission 2027 has completed training with 94.2% accuracy',
            'priority': 'high',
            'read': False,
            'created_at': datetime.now() - timedelta(hours=2)
        },
        {
            'notification_id': 'notif-002',
            'notification_type': 'data_upload',
            'title': 'New ISS Dataset Available',
            'message': 'NASA OSD-789 dataset from ISS Expedition 68 has been successfully uploaded and processed',
            'priority': 'medium',
            'read': False,
            'created_at': datetime.now() - timedelta(hours=4)
        },
        {
            'notification_id': 'notif-003',
            'notification_type': 'team_update',
            'title': 'Project Team Assignment',
            'message': 'Dr. Lisa Kim has been assigned to the Lunar Gateway Health Protocols project team',
            'priority': 'low',
            'read': True,
            'created_at': datetime.now() - timedelta(days=1)
        },
        {
            'notification_id': 'notif-004',
            'notification_type': 'model_update',
            'title': 'Model Performance Alert',
            'message': 'CardioPredict v2.1 showing degraded performance. Model retraining recommended.',
            'priority': 'high',
            'read': False,
            'created_at': datetime.now() - timedelta(hours=6)
        },
        {
            'notification_id': 'notif-005',
            'notification_type': 'system_alert',
            'title': 'Scheduled Maintenance Complete',
            'message': 'Database optimization and backup completed successfully. All systems operational.',
            'priority': 'medium',
            'read': True,
            'created_at': datetime.now() - timedelta(hours=12)
        },
        {
            'notification_id': 'notif-006',
            'notification_type': 'workflow_success',
            'title': 'Automated Pipeline Success',
            'message': 'Weekly data processing workflow completed successfully. 15 new predictions generated.',
            'priority': 'medium',
            'read': False,
            'created_at': datetime.now() - timedelta(hours=18)
        },
        {
            'notification_id': 'notif-007',
            'notification_type': 'report_ready',
            'title': 'Quarterly Report Generated',
            'message': 'Q2 2025 Cardiovascular Risk Assessment Report is ready for review',
            'priority': 'medium',
            'read': False,
            'created_at': datetime.now() - timedelta(days=2)
        }
    ]
    
    for notif_data in notifications:
        notification = Notification(**notif_data)
        # Assign to random user (for user-specific notifications)
        if random.choice([True, False]):
            notification.user_id = random.choice(users).user_id
        db.session.add(notification)
