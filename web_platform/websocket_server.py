"""
Phase 3 WebSocket Server for CardioPredict Platform
Real-time notifications, collaboration, and monitoring
"""

from flask import request, session
from flask_socketio import SocketIO, emit, join_room, leave_room, disconnect
from flask_jwt_extended import decode_token, verify_jwt_in_request
from datetime import datetime
import json
import logging
from models import db, User, Project
from models_phase3 import RealtimeEvent, UserSession

logger = logging.getLogger(__name__)

# Initialize SocketIO
socketio = SocketIO(cors_allowed_origins="*", async_mode='threading')

# Store active connections
active_connections = {}
user_rooms = {}

@socketio.on('connect')
def handle_connect(auth):
    """Handle client connection"""
    try:
        client_id = request.sid
        
        # Authentication
        user_id = None
        if auth and 'token' in auth:
            try:
                # Verify JWT token
                token_data = decode_token(auth['token'])
                user_id = token_data['sub']
            except Exception as e:
                logger.warning(f"Invalid token in WebSocket connection: {e}")
                disconnect()
                return False
        
        # Store connection info
        active_connections[client_id] = {
            'user_id': user_id,
            'connected_at': datetime.now(),
            'ip_address': request.environ.get('REMOTE_ADDR'),
            'user_agent': request.headers.get('User-Agent')
        }
        
        if user_id:
            # Join user-specific room
            join_room(f"user_{user_id}")
            user_rooms[client_id] = f"user_{user_id}"
            
            # Log connection event
            create_realtime_event(
                event_type='user_connected',
                user_id=user_id,
                data={'client_id': client_id}
            )
        
        logger.info(f"WebSocket connection established: {client_id} (user: {user_id})")
        
        # Send welcome message
        emit('connected', {
            'message': 'Connected to CardioPredict real-time server',
            'client_id': client_id,
            'authenticated': user_id is not None,
            'timestamp': datetime.now().isoformat()
        })
        
        return True
        
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
        disconnect()
        return False

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    try:
        client_id = request.sid
        
        if client_id in active_connections:
            connection_info = active_connections[client_id]
            user_id = connection_info.get('user_id')
            
            if user_id:
                # Leave user room
                if client_id in user_rooms:
                    leave_room(user_rooms[client_id])
                    del user_rooms[client_id]
                
                # Log disconnection event
                create_realtime_event(
                    event_type='user_disconnected',
                    user_id=user_id,
                    data={'client_id': client_id}
                )
            
            # Remove from active connections
            del active_connections[client_id]
            
            logger.info(f"WebSocket disconnection: {client_id} (user: {user_id})")
        
    except Exception as e:
        logger.error(f"WebSocket disconnection error: {e}")

@socketio.on('join_project')
def handle_join_project(data):
    """Join a project room for collaboration"""
    try:
        client_id = request.sid
        project_id = data.get('project_id')
        
        if not project_id:
            emit('error', {'message': 'Project ID required'})
            return
        
        # Get user info
        connection_info = active_connections.get(client_id, {})
        user_id = connection_info.get('user_id')
        
        if not user_id:
            emit('error', {'message': 'Authentication required'})
            return
        
        # Verify user has access to project
        project = Project.query.get(project_id)
        if not project:
            emit('error', {'message': 'Project not found'})
            return
        
        # Check if user is project member (simplified check)
        has_access = (project.owner_id == user_id or 
                     any(member.id == user_id for member in project.members))
        
        if not has_access:
            emit('error', {'message': 'Access denied to project'})
            return
        
        # Join project room
        room = f"project_{project_id}"
        join_room(room)
        
        # Notify others in the project
        emit('user_joined_project', {
            'user_id': user_id,
            'project_id': project_id,
            'timestamp': datetime.now().isoformat()
        }, room=room, include_self=False)
        
        # Confirm join to user
        emit('joined_project', {
            'project_id': project_id,
            'project_name': project.name,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"User {user_id} joined project room {project_id}")
        
    except Exception as e:
        logger.error(f"Error joining project room: {e}")
        emit('error', {'message': 'Failed to join project'})

@socketio.on('leave_project')
def handle_leave_project(data):
    """Leave a project room"""
    try:
        client_id = request.sid
        project_id = data.get('project_id')
        
        connection_info = active_connections.get(client_id, {})
        user_id = connection_info.get('user_id')
        
        if project_id:
            room = f"project_{project_id}"
            leave_room(room)
            
            # Notify others in the project
            emit('user_left_project', {
                'user_id': user_id,
                'project_id': project_id,
                'timestamp': datetime.now().isoformat()
            }, room=room)
            
            emit('left_project', {'project_id': project_id})
            logger.info(f"User {user_id} left project room {project_id}")
        
    except Exception as e:
        logger.error(f"Error leaving project room: {e}")

@socketio.on('prediction_update')
def handle_prediction_update(data):
    """Handle real-time prediction updates"""
    try:
        client_id = request.sid
        connection_info = active_connections.get(client_id, {})
        user_id = connection_info.get('user_id')
        
        if not user_id:
            emit('error', {'message': 'Authentication required'})
            return
        
        # Broadcast prediction update to user's room
        emit('prediction_result', {
            'prediction_id': data.get('prediction_id'),
            'status': data.get('status', 'completed'),
            'result': data.get('result'),
            'timestamp': datetime.now().isoformat()
        }, room=f"user_{user_id}")
        
        # Log event
        create_realtime_event(
            event_type='prediction_update',
            user_id=user_id,
            data=data
        )
        
    except Exception as e:
        logger.error(f"Error handling prediction update: {e}")

@socketio.on('model_training_update')
def handle_model_training_update(data):
    """Handle real-time model training updates"""
    try:
        # Broadcast to relevant users (admin/researchers)
        emit('training_progress', {
            'model_id': data.get('model_id'),
            'progress': data.get('progress', 0),
            'status': data.get('status', 'training'),
            'metrics': data.get('metrics', {}),
            'timestamp': datetime.now().isoformat()
        }, room='researchers')
        
        # Log event
        create_realtime_event(
            event_type='model_training_update',
            data=data
        )
        
    except Exception as e:
        logger.error(f"Error handling model training update: {e}")

@socketio.on('system_alert')
def handle_system_alert(data):
    """Handle system-wide alerts"""
    try:
        alert_level = data.get('level', 'info')
        message = data.get('message', '')
        
        # Broadcast to all connected users
        emit('alert', {
            'level': alert_level,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }, broadcast=True)
        
        # Log event
        create_realtime_event(
            event_type='system_alert',
            data=data
        )
        
        logger.info(f"System alert broadcast: {alert_level} - {message}")
        
    except Exception as e:
        logger.error(f"Error handling system alert: {e}")

def create_realtime_event(event_type, user_id=None, data=None):
    """Create a realtime event record"""
    try:
        event = RealtimeEvent(
            event_type=event_type,
            user_id=user_id,
            data=data or {},
            timestamp=datetime.now()
        )
        db.session.add(event)
        db.session.commit()
        
    except Exception as e:
        logger.error(f"Error creating realtime event: {e}")

def broadcast_to_users(user_ids, event_type, data):
    """Broadcast message to specific users"""
    try:
        for user_id in user_ids:
            socketio.emit(event_type, data, room=f"user_{user_id}")
        
        logger.info(f"Broadcast sent to {len(user_ids)} users: {event_type}")
        
    except Exception as e:
        logger.error(f"Error broadcasting to users: {e}")

def broadcast_to_project(project_id, event_type, data):
    """Broadcast message to project members"""
    try:
        socketio.emit(event_type, data, room=f"project_{project_id}")
        logger.info(f"Broadcast sent to project {project_id}: {event_type}")
        
    except Exception as e:
        logger.error(f"Error broadcasting to project: {e}")

def get_active_users():
    """Get list of active users"""
    try:
        active_users = []
        for client_id, info in active_connections.items():
            if info.get('user_id'):
                active_users.append({
                    'user_id': info['user_id'],
                    'connected_at': info['connected_at'].isoformat(),
                    'client_id': client_id
                })
        
        return active_users
        
    except Exception as e:
        logger.error(f"Error getting active users: {e}")
        return []

def get_connection_stats():
    """Get WebSocket connection statistics"""
    try:
        total_connections = len(active_connections)
        authenticated_connections = len([
            conn for conn in active_connections.values() 
            if conn.get('user_id')
        ])
        
        return {
            'total_connections': total_connections,
            'authenticated_connections': authenticated_connections,
            'anonymous_connections': total_connections - authenticated_connections,
            'active_rooms': len(set(user_rooms.values())),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting connection stats: {e}")
        return {}

# Background task for cleanup
def cleanup_old_events():
    """Clean up old realtime events (run periodically)"""
    try:
        from datetime import timedelta
        
        # Delete events older than 7 days
        cutoff_date = datetime.now() - timedelta(days=7)
        deleted_count = RealtimeEvent.query.filter(
            RealtimeEvent.timestamp < cutoff_date
        ).delete()
        
        db.session.commit()
        logger.info(f"Cleaned up {deleted_count} old realtime events")
        
    except Exception as e:
        logger.error(f"Error cleaning up realtime events: {e}")

if __name__ == "__main__":
    # For testing
    from flask import Flask
    from models import db
    
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'test-secret'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test_websocket.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    db.init_app(app)
    socketio.init_app(app)
    
    with app.app_context():
        db.create_all()
        print("WebSocket server ready for testing")
        socketio.run(app, debug=True, port=5001)
