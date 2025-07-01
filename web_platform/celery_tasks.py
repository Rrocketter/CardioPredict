"""
Phase 3 Background Task Processing for CardioPredict Platform
Celery tasks for ML training, data processing, and notifications
"""

from celery import Celery
from celery.schedules import crontab
from datetime import datetime
import json
import logging
import os
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Celery
def make_celery(app=None):
    """Create Celery instance"""
    celery = Celery(
        'cardiopredict',
        broker=os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
        backend=os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
    )
    
    # Configure Celery
    celery.conf.update(
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone='UTC',
        enable_utc=True,
        task_routes={
            'cardiopredict.ml.*': {'queue': 'ml_processing'},
            'cardiopredict.data.*': {'queue': 'data_processing'},
            'cardiopredict.notifications.*': {'queue': 'notifications'},
        },
        task_default_queue='default',
        worker_prefetch_multiplier=1,
        task_acks_late=True,
        worker_disable_rate_limits=False,
        task_default_rate_limit='100/m'
    )
    
    if app:
        # Update configuration with Flask app config
        celery.conf.update(app.config)
        
        class ContextTask(celery.Task):
            """Make celery tasks work with Flask app context"""
            def __call__(self, *args, **kwargs):
                with app.app_context():
                    return self.run(*args, **kwargs)
        
        celery.Task = ContextTask
    
    return celery

# Create Celery instance
celery = make_celery()

@celery.task(bind=True, name='cardiopredict.ml.train_model')
def train_model_async(self, model_config, dataset_id, user_id):
    """Asynchronously train a machine learning model"""
    try:
        # Update job status
        update_background_job(self.request.id, 'running', {'progress': 0})
        
        logger.info(f"Starting model training job {self.request.id}")
        
        # Simulate model training with progress updates
        for i in range(10):
            time.sleep(2)  # Simulate training time
            progress = (i + 1) * 10
            
            # Update progress
            update_background_job(
                self.request.id, 
                'running', 
                {
                    'progress': progress,
                    'step': f'Training epoch {i+1}/10',
                    'current_metrics': {
                        'accuracy': 0.7 + (i * 0.02),
                        'loss': 1.0 - (i * 0.08)
                    }
                }
            )
            
            # Send real-time update
            send_realtime_update('model_training_update', {
                'job_id': self.request.id,
                'progress': progress,
                'user_id': user_id
            })
        
        # Simulate model saving
        model_path = f"/models/trained_model_{self.request.id}.joblib"
        
        # Final results
        results = {
            'model_path': model_path,
            'final_metrics': {
                'accuracy': 0.89,
                'precision': 0.87,
                'recall': 0.91,
                'f1_score': 0.89
            },
            'training_time': 20,
            'dataset_id': dataset_id,
            'completed_at': datetime.now().isoformat()
        }
        
        # Update job status to completed
        update_background_job(self.request.id, 'completed', results)
        
        # Send completion notification
        send_notification(user_id, 'model_training_completed', {
            'job_id': self.request.id,
            'model_metrics': results['final_metrics']
        })
        
        logger.info(f"Model training job {self.request.id} completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Model training job {self.request.id} failed: {e}")
        
        # Update job status to failed
        update_background_job(self.request.id, 'failed', {'error': str(e)})
        
        # Send failure notification
        send_notification(user_id, 'model_training_failed', {
            'job_id': self.request.id,
            'error': str(e)
        })
        
        raise

@celery.task(bind=True, name='cardiopredict.data.process_dataset')
def process_dataset_async(self, dataset_path, processing_config, user_id):
    """Asynchronously process a dataset"""
    try:
        update_background_job(self.request.id, 'running', {'progress': 0})
        
        logger.info(f"Starting dataset processing job {self.request.id}")
        
        # Simulate data processing steps
        steps = [
            'Loading data',
            'Data validation',
            'Feature extraction',
            'Data cleaning',
            'Normalization',
            'Quality checks',
            'Saving processed data'
        ]
        
        for i, step in enumerate(steps):
            time.sleep(1)  # Simulate processing time
            progress = int((i + 1) / len(steps) * 100)
            
            update_background_job(
                self.request.id,
                'running',
                {
                    'progress': progress,
                    'step': step,
                    'records_processed': (i + 1) * 1000
                }
            )
        
        # Results
        results = {
            'processed_records': 7000,
            'output_path': f"/processed_data/dataset_{self.request.id}.csv",
            'data_quality_score': 0.94,
            'processing_time': len(steps),
            'completed_at': datetime.now().isoformat()
        }
        
        update_background_job(self.request.id, 'completed', results)
        
        send_notification(user_id, 'dataset_processing_completed', {
            'job_id': self.request.id,
            'records_processed': results['processed_records']
        })
        
        logger.info(f"Dataset processing job {self.request.id} completed")
        return results
        
    except Exception as e:
        logger.error(f"Dataset processing job {self.request.id} failed: {e}")
        update_background_job(self.request.id, 'failed', {'error': str(e)})
        send_notification(user_id, 'dataset_processing_failed', {
            'job_id': self.request.id,
            'error': str(e)
        })
        raise

@celery.task(bind=True, name='cardiopredict.ml.detect_model_drift')
def detect_model_drift_async(self, model_id, new_data_path, user_id):
    """Asynchronously detect model drift"""
    try:
        update_background_job(self.request.id, 'running', {'progress': 0})
        
        logger.info(f"Starting model drift detection job {self.request.id}")
        
        # Simulate drift detection
        time.sleep(5)
        
        # Simulate drift metrics
        drift_metrics = {
            'psi_score': 0.12,  # Population Stability Index
            'ks_statistic': 0.08,  # Kolmogorov-Smirnov
            'drift_detected': True,
            'affected_features': ['feature_1', 'feature_3', 'feature_7'],
            'severity': 'moderate'
        }
        
        results = {
            'model_id': model_id,
            'drift_metrics': drift_metrics,
            'recommendations': [
                'Consider retraining the model with recent data',
                'Monitor affected features more closely',
                'Update feature engineering pipeline'
            ],
            'completed_at': datetime.now().isoformat()
        }
        
        update_background_job(self.request.id, 'completed', results)
        
        # Send drift alert if detected
        if drift_metrics['drift_detected']:
            send_notification(user_id, 'model_drift_detected', {
                'model_id': model_id,
                'severity': drift_metrics['severity'],
                'psi_score': drift_metrics['psi_score']
            })
        
        logger.info(f"Model drift detection job {self.request.id} completed")
        return results
        
    except Exception as e:
        logger.error(f"Model drift detection job {self.request.id} failed: {e}")
        update_background_job(self.request.id, 'failed', {'error': str(e)})
        raise

@celery.task(bind=True, name='cardiopredict.notifications.send_batch_notifications')
def send_batch_notifications_async(self, notification_batch):
    """Send batch notifications"""
    try:
        logger.info(f"Processing batch of {len(notification_batch)} notifications")
        
        sent_count = 0
        failed_count = 0
        
        for notification in notification_batch:
            try:
                # Send individual notification
                send_notification(
                    notification['user_id'],
                    notification['type'],
                    notification['data']
                )
                sent_count += 1
                
            except Exception as e:
                logger.error(f"Failed to send notification: {e}")
                failed_count += 1
        
        results = {
            'total': len(notification_batch),
            'sent': sent_count,
            'failed': failed_count,
            'completed_at': datetime.now().isoformat()
        }
        
        logger.info(f"Batch notification job completed: {results}")
        return results
        
    except Exception as e:
        logger.error(f"Batch notification job failed: {e}")
        raise

@celery.task(name='cardiopredict.maintenance.cleanup_old_data')
def cleanup_old_data():
    """Periodic cleanup of old data"""
    try:
        logger.info("Starting periodic data cleanup")
        
        # Simulate cleanup operations
        cleanup_results = {
            'old_sessions_cleaned': 150,
            'old_events_cleaned': 500,
            'old_logs_cleaned': 1000,
            'disk_space_freed': '2.5 GB',
            'completed_at': datetime.now().isoformat()
        }
        
        logger.info(f"Data cleanup completed: {cleanup_results}")
        return cleanup_results
        
    except Exception as e:
        logger.error(f"Data cleanup failed: {e}")
        raise

# Helper functions
def update_background_job(job_id, status, result_data=None):
    """Update background job status"""
    try:
        # In a real implementation, this would update the database
        # For now, we'll just log the update
        logger.info(f"Job {job_id} status: {status}, data: {result_data}")
        
        # This would interact with the BackgroundJob model
        # from models_phase3 import BackgroundJob
        # job = BackgroundJob.query.filter_by(job_id=job_id).first()
        # if job:
        #     job.status = status
        #     job.result = result_data
        #     job.updated_at = datetime.now()
        #     db.session.commit()
        
    except Exception as e:
        logger.error(f"Error updating background job: {e}")

def send_realtime_update(event_type, data):
    """Send real-time update via WebSocket"""
    try:
        # In a real implementation, this would use the WebSocket server
        logger.info(f"Sending real-time update: {event_type}, data: {data}")
        
        # This would interact with the WebSocket server
        # from websocket_server import socketio
        # socketio.emit(event_type, data)
        
    except Exception as e:
        logger.error(f"Error sending real-time update: {e}")

def send_notification(user_id, notification_type, data):
    """Send notification to user"""
    try:
        # In a real implementation, this would create a notification record
        logger.info(f"Sending notification to user {user_id}: {notification_type}")
        
        # This would interact with the Notification model
        # from models import Notification
        # notification = Notification(
        #     user_id=user_id,
        #     type=notification_type,
        #     message=data.get('message', ''),
        #     data=data,
        #     created_at=datetime.now()
        # )
        # db.session.add(notification)
        # db.session.commit()
        
    except Exception as e:
        logger.error(f"Error sending notification: {e}")

# Periodic tasks setup
@celery.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """Setup periodic tasks"""
    try:
        # Run cleanup every day at 2 AM
        sender.add_periodic_task(
            crontab(hour=2, minute=0),
            cleanup_old_data.s(),
            name='daily cleanup'
        )
        
        # Health check every 5 minutes
        sender.add_periodic_task(
            300.0,  # 5 minutes
            health_check.s(),
            name='health check'
        )
        
        logger.info("Periodic tasks configured")
        
    except Exception as e:
        logger.error(f"Error setting up periodic tasks: {e}")

@celery.task(name='cardiopredict.maintenance.health_check')
def health_check():
    """System health check"""
    try:
        import psutil
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        health_status = {
            'cpu_usage': cpu_percent,
            'memory_usage': memory.percent,
            'disk_usage': disk.percent,
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy'
        }
        
        # Check for issues
        if cpu_percent > 90:
            health_status['status'] = 'warning'
            health_status['issues'] = ['High CPU usage']
        
        if memory.percent > 90:
            health_status['status'] = 'warning'
            health_status['issues'] = health_status.get('issues', []) + ['High memory usage']
        
        logger.info(f"Health check completed: {health_status}")
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

if __name__ == "__main__":
    # For testing
    print("Celery worker configuration loaded")
    print("Available tasks:")
    for task_name in celery.tasks.keys():
        print(f"  - {task_name}")
