"""
Global error handling and logging configuration
"""

import logging
import traceback
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from typing import Any
import os
from datetime import datetime

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/sar_system_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class SADroneException(Exception):
    """Base exception for SAR Drone System"""
    def __init__(self, message: str, status_code: int = 500, details: Any = None):
        self.message = message
        self.status_code = status_code
        self.details = details
        super().__init__(self.message)

async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for all unhandled exceptions
    """
    # Log the full exception
    logger.error(
        f"Unhandled exception: {request.method} {request.url.path}",
        exc_info=True,
        extra={
            "request_id": getattr(request.state, "request_id", "unknown"),
            "method": request.method,
            "path": request.url.path,
            "client": request.client.host if request.client else "unknown"
        }
    )
    
    # Return safe error response
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again.",
            "request_id": getattr(request.state, "request_id", "unknown"),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Handler for HTTPException
    """
    logger.warning(
        f"HTTP exception: {exc.status_code} - {exc.detail}",
        extra={
            "request_id": getattr(request.state, "request_id", "unknown"),
            "method": request.method,
            "path": request.url.path,
            "status_code": exc.status_code
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "request_id": getattr(request.state, "request_id", "unknown"),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

def log_mission_critical(message: str, **kwargs):
    """
    Log mission-critical events (potential life-safety impact)
    """
    logger.critical(f"MISSION CRITICAL: {message}", extra=kwargs)

def log_safety_event(message: str, **kwargs):
    """
    Log safety-related events
    """
    logger.warning(f"SAFETY EVENT: {message}", extra=kwargs)