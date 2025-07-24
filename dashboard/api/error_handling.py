"""
API error handling utilities for DeepEarth Dashboard.

Provides decorators and utilities for consistent error handling across API endpoints.
"""

import logging
import traceback
from functools import wraps
from flask import jsonify

logger = logging.getLogger(__name__)


def handle_api_error(func):
    """
    Comprehensive error handling decorator for API routes.
    
    Handles common exception types with appropriate HTTP status codes:
    - ValueError: 400 (Bad Request)
    - FileNotFoundError: 404 (Not Found) 
    - RuntimeError: 500 (Internal Server Error)
    - Exception: 500 (Internal Server Error)
    
    Automatically logs errors and returns consistent JSON error responses.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            logger.error(f"ValueError in {func.__name__}: {str(e)}")
            return jsonify({'error': str(e)}), 400
        except FileNotFoundError as e:
            logger.error(f"FileNotFoundError in {func.__name__}: {str(e)}")
            return jsonify({'error': str(e)}), 404
        except RuntimeError as e:
            logger.error(f"RuntimeError in {func.__name__}: {str(e)}")
            return jsonify({'error': str(e)}), 500
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': str(e)}), 500
    
    return wrapper


def handle_image_proxy_error(func):
    """
    Specialized error handling decorator for image proxy endpoints.
    
    Returns string responses instead of JSON for better compatibility
    with image requests and browser error handling.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            logger.error(f"FileNotFoundError in {func.__name__}: {str(e)}")
            return str(e), 404
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            return "Error serving image", 500
    
    return wrapper


def handle_vision_error(func):
    """
    Specialized error handling for vision embedding endpoints.
    
    Includes additional traceback logging for complex vision processing errors.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            logger.error(f"ValueError in {func.__name__}: {str(e)}")
            return jsonify({
                'count': 0,
                'observations': [],
                'error': str(e)
            })
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': str(e)}), 500
    
    return wrapper


def handle_health_check_error(func):
    """
    Specialized error handling for health check endpoints.
    
    Returns structured health status with error information.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            from datetime import datetime
            logger.error(f"Health check failed: {e}")
            return jsonify({
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }), 500
    
    return wrapper