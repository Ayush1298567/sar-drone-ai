"""
System monitoring and status endpoints
Provides health checks and system information
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import psutil
import os
import aiofiles
from datetime import datetime, timezone
import logging
from sqlalchemy import text

from core.database import get_db, SessionLocal
from modules.simulator import DroneSimulator

logger = logging.getLogger(__name__)
router = APIRouter()

# Add this variable and function like other API modules
drone_simulator: Optional[DroneSimulator] = None

def set_simulator(simulator: DroneSimulator):
    global drone_simulator
    drone_simulator = simulator

@router.get("/status", response_model=Dict[str, Any])
async def get_system_status():
    """
    Get comprehensive system status including:
    - System health
    - Database status
    - AI model status
    - Drone simulator status
    - Resource usage
    """
    try:
        status = {
            "status": "operational",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {}
        }
        
        # Check database
        try:
            db = SessionLocal()
            db.execute(text("SELECT 1"))
            db.close()
            status["components"]["database"] = {
                "status": "healthy",
                "type": "SQLite"
            }
        except Exception as e:
            logger.error(f"Database check failed: {e}")
            status["components"]["database"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            status["status"] = "degraded"
        
        # Check simulator
        try:
            if drone_simulator:
                drone_count = len(drone_simulator.drones)
                status["components"]["simulator"] = {
                    "status": "healthy",
                    "active_drones": drone_count,
                    "running": drone_simulator.running
                }
            else:
                status["components"]["simulator"] = {
                    "status": "not_initialized",
                    "active_drones": 0,
                    "running": False
                }
        except Exception as e:
            logger.error(f"Simulator check failed: {e}")
            status["components"]["simulator"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Check AI models
        status["components"]["ai_models"] = await check_ai_models()
        
        # System resources
        status["components"]["system_resources"] = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent
        }
        
        # Check data directories
        status["components"]["storage"] = check_storage_directories()
        
        return status
        
    except Exception as e:
        logger.error(f"System status check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system status: {str(e)}"
        )

async def check_ai_models() -> Dict[str, Any]:
    """Check if AI models are available"""
    ai_status = {
        "ollama_connection": False,
        "models_available": [],
        "models_required": ["mistral:7b-instruct", "phi:2.7b", "llama2:7b"]
    }
    
    try:
        import requests
        # Check if Ollama is running
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            ai_status["ollama_connection"] = True
            models = response.json().get("models", [])
            ai_status["models_available"] = [m["name"] for m in models]
    except:
        ai_status["status"] = "Ollama not running"
    
    return ai_status

def check_storage_directories() -> Dict[str, Any]:
    """Check if required directories exist"""
    required_dirs = ["data", "logs", "portable_ai_system"]
    storage_status = {}
    
    for dir_name in required_dirs:
        dir_path = os.path.join(os.getcwd(), dir_name)
        exists = os.path.exists(dir_path)
        storage_status[dir_name] = {
            "exists": exists,
            "path": dir_path
        }
        if not exists:
            try:
                os.makedirs(dir_path, exist_ok=True)
                storage_status[dir_name]["created"] = True
            except Exception as e:
                storage_status[dir_name]["error"] = str(e)
    
    return storage_status

@router.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}