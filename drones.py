"""
Drone API endpoints - All drone control commands go through here
This is what the frontend will call to control drones
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from core.database import get_db, Drone as DroneModel
from core.config import settings

import logging

# Add this line after the imports
logger = logging.getLogger(__name__)

router = APIRouter()

# Store simulator reference (will be set from main.py)
drone_simulator = None

def set_simulator(simulator):
    """Set the simulator instance - called from main.py"""
    global drone_simulator
    drone_simulator = simulator

# Drone status endpoint
@router.get("/")
async def get_all_drones(
    db: Session = Depends(get_db),
    include_offline: bool = Query(True, description="Include offline drones")
):
    """
    Get status of all drones
    """
    if not drone_simulator:
        raise HTTPException(status_code=503, detail="Simulator not initialized")
    
    drones_data = []
    
    # Get all simulated drones
    for sim_drone in drone_simulator.get_all_drones():
        if not include_offline and sim_drone.status == "offline":
            continue
            
        drone_data = {
            "id": sim_drone.id,
            "name": sim_drone.name,
            "status": sim_drone.status,
            "battery": sim_drone.battery,
            "position": sim_drone.position,
            "velocity": sim_drone.velocity,
            "armed": sim_drone.armed,
            "heading": sim_drone.heading,
            "last_update": datetime.now().isoformat()
        }
        drones_data.append(drone_data)
    
    return {
        "count": len(drones_data),
        "drones": drones_data
    }

# Get single drone
@router.get("/{drone_id}")
async def get_drone(drone_id: int):
    """
    Get status of a specific drone
    """
    if not drone_simulator:
        raise HTTPException(status_code=503, detail="Simulator not initialized")
    
    drone = drone_simulator.get_drone(drone_id)
    if not drone:
        raise HTTPException(status_code=404, detail=f"Drone {drone_id} not found")
    
    return drone.get_telemetry()

# Arm drone
@router.post("/{drone_id}/arm")
async def arm_drone(drone_id: int):
    """
    Arm drone motors - required before takeoff
    """
    if not drone_simulator:
        raise HTTPException(status_code=503, detail="Simulator not initialized")
    
    drone = drone_simulator.get_drone(drone_id)
    if not drone:
        raise HTTPException(status_code=404, detail=f"Drone {drone_id} not found")
    
    success = await drone.arm()
    if not success:
        raise HTTPException(status_code=400, detail="Cannot arm drone - check battery and status")
    
    return {
        "success": True,
        "message": f"Drone {drone_id} armed",
        "drone_status": drone.status
    }

# Disarm drone
@router.post("/{drone_id}/disarm")
async def disarm_drone(drone_id: int):
    """
    Disarm drone motors
    """
    if not drone_simulator:
        raise HTTPException(status_code=503, detail="Simulator not initialized")
    
    drone = drone_simulator.get_drone(drone_id)
    if not drone:
        raise HTTPException(status_code=404, detail=f"Drone {drone_id} not found")
    
    success = await drone.disarm()
    if not success:
        raise HTTPException(status_code=400, detail="Cannot disarm drone while flying")
    
    return {
        "success": True,
        "message": f"Drone {drone_id} disarmed",
        "drone_status": drone.status
    }

# Takeoff
@router.post("/{drone_id}/takeoff")
async def takeoff_drone(
    drone_id: int,
    altitude: float = Query(10.0, description="Target altitude in meters", ge=1, le=50)
):
    """
    Command drone to takeoff
    """
    if not drone_simulator:
        raise HTTPException(status_code=503, detail="Simulator not initialized")
    
    drone = drone_simulator.get_drone(drone_id)
    if not drone:
        raise HTTPException(status_code=404, detail=f"Drone {drone_id} not found")
    
    success = await drone.takeoff(altitude)
    if not success:
        raise HTTPException(status_code=400, detail="Drone must be armed before takeoff")
    
    return {
        "success": True,
        "message": f"Drone {drone_id} taking off to {altitude}m",
        "target_altitude": altitude
    }

# Land
@router.post("/{drone_id}/land")
async def land_drone(drone_id: int):
    """
    Command drone to land at current position
    """
    if not drone_simulator:
        raise HTTPException(status_code=503, detail="Simulator not initialized")
    
    drone = drone_simulator.get_drone(drone_id)
    if not drone:
        raise HTTPException(status_code=404, detail=f"Drone {drone_id} not found")
    
    success = await drone.land()
    if not success:
        raise HTTPException(status_code=400, detail="Drone must be flying to land")
    
    return {
        "success": True,
        "message": f"Drone {drone_id} landing",
        "landing_position": drone.position
    }

# Go to position
@router.post("/{drone_id}/goto")
async def goto_position(
    drone_id: int,
    x: float = Query(..., description="Target X position in meters"),
    y: float = Query(..., description="Target Y position in meters"),
    z: float = Query(None, description="Target Z position in meters", ge=0, le=50)
):
    """
    Command drone to fly to specific position
    """
    if not drone_simulator:
        raise HTTPException(status_code=503, detail="Simulator not initialized")
    
    drone = drone_simulator.get_drone(drone_id)
    if not drone:
        raise HTTPException(status_code=404, detail=f"Drone {drone_id} not found")
    
    success = await drone.goto(x, y, z)
    if not success:
        raise HTTPException(status_code=400, detail="Drone must be flying")
    
    return {
        "success": True,
        "message": f"Drone {drone_id} flying to position",
        "target_position": {"x": x, "y": y, "z": z or drone.position["z"]}
    }

# Return home
@router.post("/{drone_id}/return_home")
async def return_home(drone_id: int):
    """
    Command drone to return to home position
    """
    if not drone_simulator:
        raise HTTPException(status_code=503, detail="Simulator not initialized")
    
    drone = drone_simulator.get_drone(drone_id)
    if not drone:
        raise HTTPException(status_code=404, detail=f"Drone {drone_id} not found")
    
    success = await drone.return_home()
    if not success:
        raise HTTPException(status_code=400, detail="Drone must be flying")
    
    return {
        "success": True,
        "message": f"Drone {drone_id} returning home",
        "home_position": drone.home_position
    }

# Emergency stop
@router.post("/{drone_id}/emergency_stop")
async def emergency_stop(drone_id: int):
    """
    Emergency stop - drone will land immediately
    """
    if not drone_simulator:
        raise HTTPException(status_code=503, detail="Simulator not initialized")
    
    drone = drone_simulator.get_drone(drone_id)
    if not drone:
        raise HTTPException(status_code=404, detail=f"Drone {drone_id} not found")
    
    # Force immediate landing
    drone.status = "emergency"
    await drone.land()
    
    return {
        "success": True,
        "message": f"EMERGENCY STOP - Drone {drone_id} landing immediately",
        "status": "emergency"
    }

# Set waypoints for autonomous flight
@router.post("/{drone_id}/waypoints")
async def set_waypoints(
    drone_id: int,
    waypoints: List[dict]
):
    """
    Set waypoints for autonomous mission
    Example waypoints: [{"x": 10, "y": 20, "z": 15}, {"x": 30, "y": 40, "z": 20}]
    """
    if not drone_simulator:
        raise HTTPException(status_code=503, detail="Simulator not initialized")
    
    drone = drone_simulator.get_drone(drone_id)
    if not drone:
        raise HTTPException(status_code=404, detail=f"Drone {drone_id} not found")
    
    if not waypoints:
        raise HTTPException(status_code=400, detail="Waypoints list cannot be empty")
    
    # Validate waypoints
    for i, wp in enumerate(waypoints):
        if "x" not in wp or "y" not in wp:
            raise HTTPException(status_code=400, detail=f"Waypoint {i} missing x or y coordinate")
    
    await drone.set_waypoints(waypoints)
    
    return {
        "success": True,
        "message": f"Set {len(waypoints)} waypoints for drone {drone_id}",
        "waypoints": waypoints
    }

# Get drone telemetry stream
@router.get("/{drone_id}/telemetry")
async def get_telemetry(drone_id: int):
    """
    Get current telemetry data from drone
    """
    if not drone_simulator:
        raise HTTPException(status_code=503, detail="Simulator not initialized")
    
    drone = drone_simulator.get_drone(drone_id)
    if not drone:
        raise HTTPException(status_code=404, detail=f"Drone {drone_id} not found")
    
    return drone.get_telemetry()

# Emergency kill switch - stops ALL drones immediately
@router.post("/emergency/kill_all")
async def emergency_kill_all_drones():
    """
    EMERGENCY KILL SWITCH - Immediately stops all drone motors
    WARNING: Drones will fall from current position!
    """
    if not drone_simulator:
        raise HTTPException(status_code=503, detail="Simulator not initialized")
    
    killed_drones = []
    
    # Kill all drones
    for drone in drone_simulator.get_all_drones():
        # Force immediate motor stop
        drone.status = "emergency_killed"
        drone.armed = False
        drone.velocity = {"x": 0, "y": 0, "z": 0}
        drone.target_position = None
        
        killed_drones.append({
            "id": drone.id,
            "name": drone.name,
            "last_position": drone.position
        })
    
    # Log emergency event
    logger.warning(f"EMERGENCY KILL SWITCH ACTIVATED - {len(killed_drones)} drones killed")
    
    return {
        "success": True,
        "severity": "CRITICAL",
        "message": "EMERGENCY KILL SWITCH ACTIVATED",
        "drones_affected": len(killed_drones),
        "killed_drones": killed_drones,
        "warning": "Drones have fallen from position - manual recovery required"
    }

# Return all drones to home
@router.post("/emergency/return_all_home")
async def return_all_drones_home():
    """
    Return all flying drones to home position safely
    This is the safe alternative to kill switch
    """
    if not drone_simulator:
        raise HTTPException(status_code=503, detail="Simulator not initialized")
    
    returning_drones = []
    already_home = []
    
    for drone in drone_simulator.get_all_drones():
        if drone.status == "flying":
            success = await drone.return_home()
            if success:
                returning_drones.append({
                    "id": drone.id,
                    "name": drone.name,
                    "battery": drone.battery,
                    "distance_to_home": _calculate_distance_to_home(drone)
                })
        else:
            already_home.append(drone.name)
    
    return {
        "success": True,
        "message": f"{len(returning_drones)} drones returning home",
        "returning_drones": returning_drones,
        "already_home": already_home,
        "estimated_return_time": "Based on distance and speed"
    }

# Individual drone emergency land
@router.post("/{drone_id}/emergency_land")
async def emergency_land_drone(drone_id: int):
    """
    Emergency landing for specific drone - lands at current position
    Safer than kill switch but still immediate
    """
    if not drone_simulator:
        raise HTTPException(status_code=503, detail="Simulator not initialized")
    
    drone = drone_simulator.get_drone(drone_id)
    if not drone:
        raise HTTPException(status_code=404, detail=f"Drone {drone_id} not found")
    
    # Force immediate landing
    drone.status = "emergency_landing"
    drone.target_position = {
        "x": drone.position["x"],
        "y": drone.position["y"],
        "z": 0  # Ground level
    }
    
    return {
        "success": True,
        "message": f"Drone {drone_id} emergency landing initiated",
        "landing_position": drone.position,
        "battery_remaining": drone.battery
    }

# System failsafe status
@router.get("/failsafe/status")
async def get_failsafe_status():
    """
    Get current failsafe system status
    """
    if not drone_simulator:
        raise HTTPException(status_code=503, detail="Simulator not initialized")
    
    drones = drone_simulator.get_all_drones()
    
    # Check for various failure conditions
    low_battery_drones = [d for d in drones if d.battery < 20]
    lost_drones = [d for d in drones if d.status == "lost"]
    emergency_drones = [d for d in drones if "emergency" in d.status]
    
    failsafe_triggered = len(low_battery_drones) > 0 or len(lost_drones) > 0
    
    return {
        "system_status": "FAILSAFE_ACTIVE" if failsafe_triggered else "NORMAL",
        "issues": {
            "low_battery_drones": len(low_battery_drones),
            "lost_drones": len(lost_drones),
            "emergency_status_drones": len(emergency_drones)
        },
        "recommendations": _get_failsafe_recommendations(low_battery_drones, lost_drones),
        "emergency_controls_available": {
            "kill_all": True,
            "return_all_home": True,
            "individual_emergency_land": True
        }
    }

# Helper function
def _calculate_distance_to_home(drone):
    """Calculate distance from drone to home position"""
    dx = drone.position["x"] - drone.home_position["x"]
    dy = drone.position["y"] - drone.home_position["y"]
    dz = drone.position["z"] - drone.home_position["z"]
    return round((dx**2 + dy**2 + dz**2) ** 0.5, 2)

def _get_failsafe_recommendations(low_battery_drones, lost_drones):
    """Get recommended actions based on current issues"""
    recommendations = []
    
    if len(low_battery_drones) > 0:
        recommendations.append("Return low battery drones to home immediately")
    
    if len(lost_drones) > 0:
        recommendations.append("Attempt to reestablish connection with lost drones")
    
    if len(low_battery_drones) > 2:
        recommendations.append("Consider aborting mission - multiple drones critical")
    
    return recommendations