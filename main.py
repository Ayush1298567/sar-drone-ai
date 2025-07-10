"""
SAR Drone System - Complete Backend with Simulator
Main entry point that ties all components together
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from api import system
import uuid
import asyncio
import logging
from datetime import datetime
from typing import List
import json
import numpy as np

# Import all our modules
from core.config import settings
from core.database import init_db
from modules.simulator import DroneSimulator
from api import drones, missions, detections, mapping
from modules.ai_planner import ai_planner
from api.mapping import map_system

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
active_websockets: List[WebSocket] = []
drone_simulator = None
start_time = datetime.now()

# Background tasks
background_tasks = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager - runs on startup and shutdown
    """
    global drone_simulator, background_tasks
    
    # Startup
    logger.info("Starting SAR Drone System Backend...")
    logger.info(f"Debug mode: {settings.DEBUG}")
    
    # Initialize database
    logger.info("Initializing database...")
    init_db()
    
    # Initialize drone simulator
    logger.info("Starting drone simulator...")
    drone_simulator = DroneSimulator()
    await drone_simulator.start()
    
    # Pass simulator to API modules
    drones.set_simulator(drone_simulator)
    missions.set_simulator(drone_simulator)
    detections.set_simulator(drone_simulator)
    
    # Start background tasks
    background_tasks.append(
        asyncio.create_task(telemetry_broadcaster())
    )
    background_tasks.append(
        asyncio.create_task(detection_simulator())
    )
    
    logger.info("SAR Drone System ready!")
    logger.info(f"Simulated drones: {len(drone_simulator.get_all_drones())}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down SAR Drone System...")
    
    # Cancel background tasks
    for task in background_tasks:
        task.cancel()
    
    # Stop simulator
    if drone_simulator:
        await drone_simulator.stop()
    
    logger.info("Shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="SAR Drone System",
    description="Search and Rescue Drone Control System with AI Mission Planning",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# Remove the exception handler registrations since the functions don't exist
# app.add_exception_handler(Exception, global_exception_handler)
# app.add_exception_handler(HTTPException, http_exception_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all API routers
app.include_router(drones.router, prefix="/api/drones", tags=["Drones"])
app.include_router(missions.router, prefix="/api/missions", tags=["Missions"])
app.include_router(detections.router, prefix="/api/detections", tags=["Detections"])
app.include_router(mapping.router, prefix="/api/mapping", tags=["Mapping"])
app.include_router(system.router, prefix="/api/system", tags=["System"])

# Root endpoint
@app.get("/", tags=["System"])
async def root():
    """
    System health check and status
    """
    if not drone_simulator:
        status = "initializing"
        drone_count = 0
    else:
        status = "operational"
        drone_count = len(drone_simulator.get_all_drones())
    
    return {
        "application": "SAR Drone System",
        "version": "1.0.0",
        "status": status,
        "time": datetime.now().isoformat(),
        "uptime": str(datetime.now() - start_time).split('.')[0],
        "mode": "SIMULATOR",
        "features": {
            "ai_mission_planning": True,
            "real_time_mapping": True,
            "human_detection": True,
            "emergency_controls": True,
            "natural_language_commands": True
        },
        "active_components": {
            "drones": drone_count,
            "websocket_clients": len(active_websockets),
            "simulator": drone_simulator is not None
        },
        "api_endpoints": {
            "documentation": "/docs",
            "drones": "/api/drones",
            "missions": "/api/missions",
            "detections": "/api/detections",
            "mapping": "/api/mapping",
            "websocket": "/ws"
        }
    }

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket connection for real-time updates
    Broadcasts: telemetry, detections, mission updates, map changes
    """
    await websocket.accept()
    active_websockets.append(websocket)
    logger.info(f"New WebSocket client connected. Total clients: {len(active_websockets)}")
    
    try:
        # Send initial connection message
        import json
        json_str = json.dumps({
            "type": "connection",
            "status": "connected",
            "message": "Connected to SAR Drone System",
            "time": datetime.now().isoformat(),
            "capabilities": [
                "real_time_telemetry",
                "detection_alerts",
                "mission_updates",
                "map_updates"
            ]
        }, cls=NumpyEncoder)
        await websocket.send_text(json_str)
        
        # Send initial state
        if drone_simulator:
            drones_data = []
            for drone in drone_simulator.get_all_drones():
                drones_data.append(drone.get_telemetry())
            
            json_str = json.dumps({
                "type": "initial_state",
                "drones": drones_data,
                "map_stats": await mapping.get_map_statistics()
            }, cls=NumpyEncoder)
            await websocket.send_text(json_str)
        
        # Keep connection alive
        while True:
            # Wait for client messages
            data = await websocket.receive_text()
            
            # Handle client commands
            if data == "ping":
                json_str = json.dumps({
                    "type": "pong",
                    "time": datetime.now().isoformat()
                }, cls=NumpyEncoder)
                await websocket.send_text(json_str)
            elif data.startswith("subscribe:"):
                # Handle subscription requests
                topic = data.split(":")[1]
                logger.info(f"Client subscribed to: {topic}")
            
    except WebSocketDisconnect:
        active_websockets.remove(websocket)
        logger.info(f"WebSocket client disconnected. Remaining clients: {len(active_websockets)}")

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# Broadcast function
async def broadcast_update(message: dict):
    """
    Send update to all connected WebSocket clients
    """
    if not active_websockets:
        return
    
    disconnected = []
    message["timestamp"] = datetime.now().isoformat()
    
    for websocket in active_websockets:
        try:
            # Use custom encoder to handle numpy types
            json_str = json.dumps(message, cls=NumpyEncoder)
            await websocket.send_text(json_str)
        except:
            disconnected.append(websocket)
    
    # Remove disconnected clients
    for websocket in disconnected:
        if websocket in active_websockets:
            active_websockets.remove(websocket)

# Background task: Broadcast telemetry
async def telemetry_broadcaster():
    """
    Broadcast drone telemetry to all connected clients
    """
    while True:
        try:
            if drone_simulator and active_websockets:
                telemetry_data = []
                
                for drone in drone_simulator.get_all_drones():
                    telemetry = drone.get_telemetry()
                    telemetry_data.append(telemetry)
                    
                    # Update map with drone position
                    await mapping.update_map(
                        drone_id=drone.id,
                        position=drone.position,
                        scan_data=[]  # Empty for now, would have sensor data
                    )
                
                # Broadcast telemetry
                await broadcast_update({
                    "type": "telemetry_update",
                    "drones": telemetry_data
                })
            
            await asyncio.sleep(0.5)  # 2Hz updates
            
        except Exception as e:
            logger.error(f"Error in telemetry broadcaster: {e}")
            await asyncio.sleep(1)

# Background task: Simulate detections
async def detection_simulator():
    """
    Simulate random detections for testing
    """
    while True:
        try:
            if drone_simulator and settings.SIMULATOR_ENABLED:
                for drone in drone_simulator.get_all_drones():
                    if drone.status == "flying":
                        # Check for simulated detection
                        detection = drone.simulate_detection()
                        
                        if detection:
                            # Report detection
                            from core.database import SessionLocal
                            db = SessionLocal()
                            try:
                                result = await detections.report_detection(
                                    drone_id=drone.id,
                                    detection_type=detection["type"],
                                    confidence=detection["confidence"],
                                    position=detection["position"],
                                    temperature=detection.get("temperature"),
                                    db=db
                                )
                            finally:
                                db.close()
                            
                            # Broadcast detection alert
                            await broadcast_update({
                                "type": "detection_alert",
                                "priority": "high",
                                "detection": detection,
                                "drone_id": drone.id,
                                "message": f"New {detection['type']} detection!"
                            })
                            
                            # Add marker to map
                            if detection["type"] == "human":
                                await mapping.add_marker(
                                    marker_type="human_detections",
                                    position=detection["position"],
                                    description=f"Possible human (confidence: {detection['confidence']})",
                                    priority="critical"
                                )
            
            await asyncio.sleep(2)  # Check every 2 seconds
            
        except Exception as e:
            logger.error(f"Error in detection simulator: {e}")
            await asyncio.sleep(5)

# System status endpoint
@app.get("/api/system/status", tags=["System"])
async def get_system_status():
    """
    Get detailed system status
    """
    status = {
        "operational": True,
        "mode": "SIMULATOR",
        "uptime": str(datetime.now() - start_time).split('.')[0],
        "start_time": start_time.isoformat()
    }
    # Get drone summary
    if drone_simulator:
        drones = drone_simulator.get_all_drones()
        status["drones"] = {
            "total": len(drones),
            "flying": len([d for d in drones if d.status == "flying"]),
            "armed": len([d for d in drones if d.armed]),
            "idle": len([d for d in drones if d.status == "idle"])
        }
    else:
        status["drones"] = {"total": 0, "flying": 0, "armed": 0, "idle": 0}
    # Simple mission count (without calling the API)
    status["missions"] = {
        "total": len(missions.active_missions) if hasattr(missions, 'active_missions') else 0,
        "active": len([m for m in missions.active_missions.values() if m.get("status") == "executing"]) if hasattr(missions, 'active_missions') else 0,
        "completed": len([m for m in missions.active_missions.values() if m.get("status") == "completed"]) if hasattr(missions, 'active_missions') else 0
    }
    # Simple detection count
    status["detections"] = {
        "total": len(detections.active_detections) if hasattr(detections, 'active_detections') else 0,
        "unverified": len([d for d in detections.active_detections.values() if d.get("status") == "unverified"]) if hasattr(detections, 'active_detections') else 0,
        "critical": len([d for d in detections.active_detections.values() if d.get("priority") == "critical"]) if hasattr(detections, 'active_detections') else 0
    }
    # Simple map stats
    status["mapping"] = {
        "coverage_percentage": map_system.get_exploration_percentage() if 'map_system' in globals() else 0,
        "area_explored_m2": 0
    }
    status["communications"] = {
        "websocket_clients": len(active_websockets),
        "simulator_running": drone_simulator is not None and drone_simulator.running
    }
    return status

# Quick start endpoint for testing
@app.post("/api/quick_start", tags=["System"])
async def quick_start_mission():
    """
    Quick start a test mission - useful for demos
    """
    # Create a simple search mission
    from core.database import SessionLocal
    db = SessionLocal()
    try:
        mission_response = await missions.create_mission(
            command="Search the building for survivors and create a complete map",
            db=db
        )
    finally:
        db.close()
    
    if mission_response["success"]:
        # Execute the mission
        execution_response = await missions.execute_mission(
            mission_id=mission_response["mission_id"]
        )
        
        return {
            "success": True,
            "message": "Test mission started!",
            "mission_id": mission_response["mission_id"],
            "assigned_drones": mission_response["assigned_drones"],
            "watch_progress": "Connect to WebSocket /ws for live updates"
        }
    
    return {"success": False, "message": "Failed to create mission"}

if __name__ == "__main__":
    import uvicorn
    
    logger.info("=" * 50)
    logger.info("SAR DRONE SYSTEM STARTING")
    logger.info("=" * 50)
    logger.info(f"Server: http://0.0.0.0:{settings.PORT}")
    logger.info(f"API Docs: http://0.0.0.0:{settings.PORT}/docs")
    logger.info(f"WebSocket: ws://0.0.0.0:{settings.PORT}/ws")
    logger.info("=" * 50)
    
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)