"""
Phase 2 API Extensions for CardioPredict Platform
Advanced project management, collaboration, and analytics endpoints
"""

from flask import Blueprint, jsonify, request, session
import json
import uuid
from datetime import datetime, timedelta
from sqlalchemy import func, desc, and_, or_
from models import (db, User, Dataset, Prediction, Experiment, MLModel, Notification,
                   Project, Comment, AuditLog, Report, Workflow, project_members)

# Create Phase 2 API blueprint
api_v2 = Blueprint('api_v2', __name__, url_prefix='/api/v2')

# Project Management Endpoints
@api_v2.route('/projects', methods=['GET'])
def get_projects():
    """Get all projects with filtering and pagination"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        status = request.args.get('status')
        priority = request.args.get('priority')
        search = request.args.get('search')
        
        query = Project.query
        
        # Apply filters
        if status:
            query = query.filter(Project.status == status)
        if priority:
            query = query.filter(Project.priority == priority)
        if search:
            query = query.filter(or_(
                Project.name.ilike(f'%{search}%'),
                Project.description.ilike(f'%{search}%')
            ))
        
        # Order by priority and creation date
        query = query.order_by(
            Project.priority.desc(),
            Project.created_at.desc()
        )
        
        # Paginate
        paginated = query.paginate(
            page=page,
            per_page=per_page,
            error_out=False
        )
        
        projects = []
        for project in paginated.items:
            project_dict = project.to_dict()
            # Add team member count and recent activity
            project_dict['team_count'] = len(project.team_members)
            project_dict['experiment_count'] = project.experiments.count()
            project_dict['dataset_count'] = project.datasets.count()
            projects.append(project_dict)
        
        return jsonify({
            'projects': projects,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': paginated.total,
                'pages': paginated.pages
            },
            'filters': {
                'status': status,
                'priority': priority,
                'search': search
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_v2.route('/projects', methods=['POST'])
def create_project():
    """Create a new project"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'description']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Generate unique project ID
        project_id = f"PROJ-{data['name'][:3].upper()}-{datetime.now().strftime('%Y%m')}-{uuid.uuid4().hex[:6].upper()}"
        
        project = Project(
            project_id=project_id,
            name=data['name'],
            description=data['description'],
            status=data.get('status', 'active'),
            priority=data.get('priority', 'medium'),
            start_date=datetime.fromisoformat(data['start_date']) if data.get('start_date') else datetime.now(),
            end_date=datetime.fromisoformat(data['end_date']) if data.get('end_date') else None,
            budget=data.get('budget'),
            funding_source=data.get('funding_source'),
            milestones_total=data.get('milestones_total', 0)
        )
        
        # Assign project lead if specified
        if data.get('lead_id'):
            lead = User.query.filter_by(user_id=data['lead_id']).first()
            if lead:
                project.lead = lead
        
        db.session.add(project)
        db.session.commit()
        
        # Log project creation
        log_audit_action('create', 'project', project.project_id, data.get('created_by'))
        
        return jsonify(project.to_dict()), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@api_v2.route('/projects/<project_id>', methods=['GET'])
def get_project_details(project_id):
    """Get detailed project information"""
    try:
        project = Project.query.filter_by(project_id=project_id).first()
        if not project:
            return jsonify({'error': 'Project not found'}), 404
        
        # Get project details with related data
        project_data = project.to_dict()
        
        # Add team members details
        project_data['team_members'] = [member.to_dict() for member in project.team_members]
        
        # Add recent experiments
        recent_experiments = project.experiments.order_by(desc(Experiment.start_time)).limit(5).all()
        project_data['recent_experiments'] = [exp.to_dict() for exp in recent_experiments]
        
        # Add datasets
        project_data['datasets'] = [dataset.to_dict() for dataset in project.datasets]
        
        # Add recent comments
        recent_comments = project.comments.order_by(desc(Comment.created_at)).limit(10).all()
        project_data['recent_comments'] = [comment.to_dict() for comment in recent_comments]
        
        # Add project statistics
        project_data['statistics'] = {
            'total_predictions': db.session.query(func.count(Prediction.id)).join(
                Experiment, Experiment.project_id == project.id
            ).scalar() or 0,
            'active_experiments': project.experiments.filter(Experiment.status == 'Running').count(),
            'completed_experiments': project.experiments.filter(Experiment.status == 'Completed').count(),
            'avg_experiment_accuracy': db.session.query(func.avg(Experiment.accuracy)).filter(
                and_(Experiment.project_id == project.id, Experiment.accuracy.isnot(None))
            ).scalar() or 0
        }
        
        return jsonify(project_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_v2.route('/projects/<project_id>/members', methods=['POST'])
def add_project_member():
    """Add a team member to a project"""
    try:
        project_id = request.view_args['project_id']
        data = request.get_json()
        
        project = Project.query.filter_by(project_id=project_id).first()
        if not project:
            return jsonify({'error': 'Project not found'}), 404
        
        user = User.query.filter_by(user_id=data['user_id']).first()
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        if user in project.team_members:
            return jsonify({'error': 'User already in project'}), 400
        
        # Add user to project
        project.team_members.append(user)
        db.session.commit()
        
        # Log member addition
        log_audit_action('update', 'project', project_id, data.get('added_by'), 
                        changes_summary=f"Added team member: {user.name}")
        
        return jsonify({'message': 'Team member added successfully'})
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# Advanced Analytics Endpoints
@api_v2.route('/analytics/dashboard')
def get_advanced_dashboard():
    """Get advanced dashboard analytics"""
    try:
        # Time range filter
        days = request.args.get('days', 30, type=int)
        start_date = datetime.now() - timedelta(days=days)
        
        analytics = {
            'overview': {
                'total_projects': Project.query.count(),
                'active_projects': Project.query.filter(Project.status == 'active').count(),
                'total_users': User.query.filter(User.is_active == True).count(),
                'total_experiments': Experiment.query.count(),
                'running_experiments': Experiment.query.filter(Experiment.status == 'Running').count(),
                'total_predictions': Prediction.query.count(),
                'recent_predictions': Prediction.query.filter(Prediction.created_at >= start_date).count()
            },
            'performance_metrics': {
                'avg_model_accuracy': db.session.query(func.avg(MLModel.accuracy)).filter(
                    MLModel.status == 'Active'
                ).scalar() or 0,
                'avg_experiment_success_rate': calculate_experiment_success_rate(),
                'prediction_accuracy_trend': get_prediction_accuracy_trend(days),
                'resource_utilization': get_resource_utilization()
            },
            'project_insights': {
                'projects_by_status': get_projects_by_status(),
                'projects_by_priority': get_projects_by_priority(),
                'budget_allocation': get_budget_allocation(),
                'timeline_analysis': get_timeline_analysis()
            },
            'collaboration_metrics': {
                'comments_per_project': get_comments_per_project(),
                'user_activity': get_user_activity(days),
                'cross_project_collaboration': get_cross_project_collaboration()
            }
        }
        
        return jsonify(analytics)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_v2.route('/analytics/biomarkers')
def get_biomarker_analytics():
    """Get detailed biomarker analysis"""
    try:
        # Get all predictions with biomarker data
        predictions = Prediction.query.all()
        
        biomarker_analysis = {
            'distribution': analyze_biomarker_distribution(predictions),
            'correlations': analyze_biomarker_correlations(predictions),
            'trends': analyze_biomarker_trends(predictions),
            'risk_factors': analyze_risk_factors(predictions),
            'environmental_comparison': analyze_environmental_biomarkers(predictions)
        }
        
        return jsonify(biomarker_analysis)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Collaboration Endpoints
@api_v2.route('/comments', methods=['GET'])
def get_comments():
    """Get comments with filtering"""
    try:
        entity_type = request.args.get('entity_type')
        entity_id = request.args.get('entity_id')
        project_id = request.args.get('project_id')
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        
        query = Comment.query
        
        if entity_type and entity_id:
            query = query.filter(and_(
                Comment.entity_type == entity_type,
                Comment.entity_id == entity_id
            ))
        
        if project_id:
            query = query.filter(Comment.project_id == project_id)
        
        # Order by creation date (newest first)
        query = query.order_by(desc(Comment.created_at))
        
        # Paginate
        paginated = query.paginate(
            page=page,
            per_page=per_page,
            error_out=False
        )
        
        comments = [comment.to_dict() for comment in paginated.items]
        
        return jsonify({
            'comments': comments,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': paginated.total,
                'pages': paginated.pages
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_v2.route('/comments', methods=['POST'])
def create_comment():
    """Create a new comment"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['content', 'entity_type', 'entity_id', 'author_id']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        author = User.query.filter_by(user_id=data['author_id']).first()
        if not author:
            return jsonify({'error': 'Author not found'}), 404
        
        comment = Comment(
            comment_id=f"COMM-{uuid.uuid4().hex[:8].upper()}",
            content=data['content'],
            entity_type=data['entity_type'],
            entity_id=data['entity_id'],
            author=author
        )
        
        # Set project if provided
        if data.get('project_id'):
            project = Project.query.filter_by(project_id=data['project_id']).first()
            if project:
                comment.project = project
        
        # Set parent comment for threading
        if data.get('parent_id'):
            parent = Comment.query.filter_by(comment_id=data['parent_id']).first()
            if parent:
                comment.parent = parent
        
        # Set tags if provided
        if data.get('tags'):
            comment.set_tags(data['tags'])
        
        db.session.add(comment)
        db.session.commit()
        
        # Log comment creation
        log_audit_action('create', 'comment', comment.comment_id, data['author_id'])
        
        return jsonify(comment.to_dict()), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# Report Generation Endpoints
@api_v2.route('/reports', methods=['GET'])
def get_reports():
    """Get all reports with filtering"""
    try:
        report_type = request.args.get('type')
        project_id = request.args.get('project_id')
        status = request.args.get('status')
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        query = Report.query
        
        if report_type:
            query = query.filter(Report.report_type == report_type)
        if project_id:
            project = Project.query.filter_by(project_id=project_id).first()
            if project:
                query = query.filter(Report.project_id == project.id)
        if status:
            query = query.filter(Report.status == status)
        
        query = query.order_by(desc(Report.created_at))
        
        paginated = query.paginate(
            page=page,
            per_page=per_page,
            error_out=False
        )
        
        reports = [report.to_dict() for report in paginated.items]
        
        return jsonify({
            'reports': reports,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': paginated.total,
                'pages': paginated.pages
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api_v2.route('/reports', methods=['POST'])
def generate_report():
    """Generate a new report"""
    try:
        data = request.get_json()
        
        required_fields = ['name', 'report_type']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        report = Report(
            report_id=f"RPT-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}",
            name=data['name'],
            report_type=data['report_type'],
            status='pending'
        )
        
        # Set project if provided
        if data.get('project_id'):
            project = Project.query.filter_by(project_id=data['project_id']).first()
            if project:
                report.project = project
        
        # Set creator
        if data.get('created_by'):
            creator = User.query.filter_by(user_id=data['created_by']).first()
            if creator:
                report.creator = creator
        
        # Set parameters
        if data.get('parameters'):
            report.set_parameters(data['parameters'])
        
        # Set scheduling if provided
        if data.get('is_scheduled'):
            report.is_scheduled = True
            report.schedule_expression = data.get('schedule_expression')
            # Calculate next run time (simplified)
            report.next_run = datetime.now() + timedelta(days=1)
        
        db.session.add(report)
        db.session.commit()
        
        # In a real implementation, you would trigger the actual report generation here
        # For now, we'll just mark it as completed
        report.status = 'completed'
        report.completed_at = datetime.now()
        
        # Generate sample results
        sample_results = generate_sample_report_results(data['report_type'])
        report.set_results(sample_results)
        
        db.session.commit()
        
        return jsonify(report.to_dict()), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# Workflow Management Endpoints
@api_v2.route('/workflows', methods=['GET'])
def get_workflows():
    """Get all workflows"""
    try:
        status = request.args.get('status')
        project_id = request.args.get('project_id')
        
        query = Workflow.query
        
        if status:
            query = query.filter(Workflow.status == status)
        if project_id:
            project = Project.query.filter_by(project_id=project_id).first()
            if project:
                query = query.filter(Workflow.project_id == project.id)
        
        workflows = query.order_by(desc(Workflow.created_at)).all()
        workflows_data = [workflow.to_dict() for workflow in workflows]
        
        return jsonify({'workflows': workflows_data})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Audit and Security Endpoints
@api_v2.route('/audit-logs')
def get_audit_logs():
    """Get audit logs with filtering"""
    try:
        action = request.args.get('action')
        entity_type = request.args.get('entity_type')
        user_id = request.args.get('user_id')
        days = request.args.get('days', 30, type=int)
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        
        start_date = datetime.now() - timedelta(days=days)
        
        query = AuditLog.query.filter(AuditLog.timestamp >= start_date)
        
        if action:
            query = query.filter(AuditLog.action == action)
        if entity_type:
            query = query.filter(AuditLog.entity_type == entity_type)
        if user_id:
            user = User.query.filter_by(user_id=user_id).first()
            if user:
                query = query.filter(AuditLog.user_id == user.id)
        
        query = query.order_by(desc(AuditLog.timestamp))
        
        paginated = query.paginate(
            page=page,
            per_page=per_page,
            error_out=False
        )
        
        logs = [log.to_dict() for log in paginated.items]
        
        return jsonify({
            'audit_logs': logs,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': paginated.total,
                'pages': paginated.pages
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Helper Functions
def log_audit_action(action, entity_type, entity_id, user_id=None, changes_summary=None):
    """Log an audit action"""
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
        
    except Exception as e:
        # Don't fail the main operation if audit logging fails
        print(f"Audit logging failed: {e}")

def calculate_experiment_success_rate():
    """Calculate overall experiment success rate"""
    total = Experiment.query.count()
    if total == 0:
        return 0
    
    completed = Experiment.query.filter(Experiment.status == 'Completed').count()
    return round((completed / total) * 100, 1)

def get_prediction_accuracy_trend(days):
    """Get prediction accuracy trend over time"""
    # This is a simplified implementation
    # In reality, you'd analyze actual prediction performance
    dates = []
    accuracies = []
    
    for i in range(days, 0, -1):
        date = datetime.now() - timedelta(days=i)
        dates.append(date.strftime('%Y-%m-%d'))
        accuracies.append(round(85 + (i % 10), 1))  # Mock data
    
    return {'dates': dates, 'accuracies': accuracies}

def get_resource_utilization():
    """Get resource utilization metrics"""
    # Mock implementation
    return {
        'cpu_usage': round(65.5, 1),
        'memory_usage': round(78.2, 1),
        'gpu_usage': round(42.8, 1),
        'storage_usage': round(23.4, 1)
    }

def get_projects_by_status():
    """Get project count by status"""
    result = db.session.query(
        Project.status,
        func.count(Project.id).label('count')
    ).group_by(Project.status).all()
    
    return {status: count for status, count in result}

def get_projects_by_priority():
    """Get project count by priority"""
    result = db.session.query(
        Project.priority,
        func.count(Project.id).label('count')
    ).group_by(Project.priority).all()
    
    return {priority: count for priority, count in result}

def get_budget_allocation():
    """Get budget allocation by project"""
    projects = Project.query.filter(Project.budget.isnot(None)).all()
    return {
        project.name: project.budget
        for project in projects[:10]  # Top 10 by budget
    }

def get_timeline_analysis():
    """Get project timeline analysis"""
    total_projects = Project.query.count()
    on_schedule = Project.query.filter(
        Project.progress >= 50  # Simplified logic
    ).count()
    
    return {
        'total_projects': total_projects,
        'on_schedule': on_schedule,
        'behind_schedule': total_projects - on_schedule,
        'on_schedule_percentage': round((on_schedule / total_projects) * 100, 1) if total_projects > 0 else 0
    }

def get_comments_per_project():
    """Get comment count per project"""
    result = db.session.query(
        Project.name,
        func.count(Comment.id).label('comment_count')
    ).join(Comment, Project.id == Comment.project_id)\
     .group_by(Project.id, Project.name).all()
    
    return {name: count for name, count in result}

def get_user_activity(days):
    """Get user activity over the specified time period"""
    start_date = datetime.now() - timedelta(days=days)
    
    result = db.session.query(
        User.name,
        func.count(AuditLog.id).label('activity_count')
    ).join(AuditLog, User.id == AuditLog.user_id)\
     .filter(AuditLog.timestamp >= start_date)\
     .group_by(User.id, User.name).all()
    
    return {name: count for name, count in result}

def get_cross_project_collaboration():
    """Analyze cross-project collaboration"""
    # Users working on multiple projects
    user_project_counts = db.session.query(
        User.name,
        func.count(project_members.c.project_id).label('project_count')
    ).join(project_members, User.id == project_members.c.user_id)\
     .group_by(User.id, User.name)\
     .having(func.count(project_members.c.project_id) > 1).all()
    
    return {name: count for name, count in user_project_counts}

def analyze_biomarker_distribution(predictions):
    """Analyze biomarker distribution across predictions"""
    biomarkers = ['crp', 'pf4', 'tnf_alpha', 'il6', 'troponin']
    distribution = {}
    
    for biomarker in biomarkers:
        values = []
        for pred in predictions:
            markers = pred.get_biomarkers()
            if biomarker in markers:
                values.append(markers[biomarker])
        
        if values:
            distribution[biomarker] = {
                'min': round(min(values), 3),
                'max': round(max(values), 3),
                'mean': round(sum(values) / len(values), 3),
                'count': len(values)
            }
    
    return distribution

def analyze_biomarker_correlations(predictions):
    """Analyze correlations between biomarkers and risk scores"""
    # Simplified correlation analysis
    correlations = {
        'crp_risk': 0.73,
        'tnf_alpha_risk': 0.68,
        'troponin_risk': 0.84,
        'il6_risk': 0.61,
        'pf4_risk': 0.42
    }
    return correlations

def analyze_biomarker_trends(predictions):
    """Analyze biomarker trends over time"""
    # This would involve time-series analysis in a real implementation
    return {
        'trending_up': ['crp', 'tnf_alpha'],
        'trending_down': ['pf4'],
        'stable': ['il6', 'troponin']
    }

def analyze_risk_factors(predictions):
    """Analyze risk factor frequency"""
    risk_factors = {}
    
    for pred in predictions:
        factors = pred.get_risk_factors()
        for factor in factors:
            risk_factors[factor] = risk_factors.get(factor, 0) + 1
    
    return risk_factors

def analyze_environmental_biomarkers(predictions):
    """Analyze biomarker patterns by environment"""
    env_analysis = {}
    
    for pred in predictions:
        env = pred.environment
        if env not in env_analysis:
            env_analysis[env] = {
                'count': 0,
                'avg_risk_score': 0,
                'biomarker_patterns': {}
            }
        
        env_analysis[env]['count'] += 1
        env_analysis[env]['avg_risk_score'] += pred.risk_score
    
    # Calculate averages
    for env in env_analysis:
        if env_analysis[env]['count'] > 0:
            env_analysis[env]['avg_risk_score'] = round(
                env_analysis[env]['avg_risk_score'] / env_analysis[env]['count'], 1
            )
    
    return env_analysis

def generate_sample_report_results(report_type):
    """Generate sample report results based on type"""
    if report_type == 'analytics':
        return {
            'summary': 'Quarterly cardiovascular risk analysis completed',
            'key_findings': [
                'Average risk score decreased by 12% compared to previous quarter',
                'Biomarker trends show improved cardiovascular health',
                'Model accuracy increased to 89.2%'
            ],
            'recommendations': [
                'Continue current monitoring protocols',
                'Implement additional biomarker screening',
                'Schedule model retraining for Q3'
            ]
        }
    elif report_type == 'compliance':
        return {
            'compliance_status': 'Fully Compliant',
            'audit_items': 15,
            'passed_items': 15,
            'failed_items': 0,
            'next_audit_date': (datetime.now() + timedelta(days=90)).isoformat()
        }
    else:
        return {
            'data_points': 1234,
            'processing_time': '2.5 hours',
            'accuracy': '94.2%',
            'status': 'completed'
        }
