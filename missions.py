"""
Mission API endpoints - Uses advanced AI planner for natural language commands
"""

from fastapi import APIRouter, HTTPException, Depends, Body
from sqlalchemy.orm import Session
from typing import List, Dict, Optional
from datetime import datetime
import asyncio

from core.database import get_db, Mission as MissionModel
from modules.ai_planner import ai_planner
from modules.simulator import DroneSimulator

router = APIRouter()

# Store active missions and simulator reference
active_missions: Dict[int, Dict] = {}
mission_counter = 0
drone_simulator: Optional[DroneSimulator] = None

def set_simulator(simulator: DroneSimulator):
    """Set the simulator instance - called from main.py"""
    global drone_simulator
    drone_simulator = simulator

# Create mission using AI
@router.post("/create")
async def create_mission(
    command: str = Body(..., description="Natural language command", 
                        example="Search the building for survivors"),
    db: Session = Depends(get_db)
):
    """
    Create a mission from natural language using AI
    
    The AI will:
    1. Understand what you want
    2. Assess available drones
    3. Create optimal strategy
    4. Assign drones to roles
    5. Generate flight plans
    
    Examples:
    - "Search the building for survivors"
    - "Check all rooms on the second floor"
    - "Find the safest exit route"
    - "Look for gas leaks or fire hazards"
    """
    global mission_counter
    
    if not drone_simulator:
        raise HTTPException(status_code=503, detail="Simulator not initialized")
    
    # Get available drones
    available_drones = []
    for drone in drone_simulator.get_all_drones():
        drone_info = {
            "id": drone.id,
            "name": drone.name,
            "battery": drone.battery,
            "status": drone.status,
            "position": drone.position,
            "thermal_camera": True,  # All simulated drones have thermal
            "camera": True,
            "range": 1000  # meters
        }
        # Only include drones that are ready
        if drone.status in ["idle", "armed"] and drone.battery > 30:
            available_drones.append(drone_info)
    
    if not available_drones:
        raise HTTPException(
            status_code=400, 
            detail="No drones available. All drones are busy or have low battery."
        )
    
    # Use AI to create mission plan
    try:
        mission_plan = ai_planner.analyze_mission(
            user_command=command,
            available_drones=available_drones,
            environment_data={
                "type": "unknown",
                "estimated_size": {"x": 100, "y": 100, "z": 30}
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI planning failed: {str(e)}"
        )
    
    # Create mission ID
    mission_counter += 1
    mission_id = mission_counter
    
    # Store mission
    mission_data = {
        "id": mission_id,
        "status": "planned",
        "created_at": datetime.now().isoformat(),
        **mission_plan
    }
    # Defensive access for ai_analysis and strategy
    ai_analysis = mission_plan.get('ai_analysis', {})
    strategy = mission_plan.get('strategy', {})
    
    active_missions[mission_id] = mission_data
    
    # Save to database
    db_mission = MissionModel(
        name=strategy.get('name', 'Unknown Mission'),
        type=ai_analysis.get('mission_type', 'unknown'),
        status="planned",
        parameters={
            "user_command": command,
            "ai_plan": mission_plan
        }
    )
    db.add(db_mission)
    db.commit()
    db.refresh(db_mission)
    mission_data["db_id"] = db_mission.id
    
    # Create simplified response
    return {
        "success": True,
        "mission_id": mission_id,
        "ai_understanding": {
            "interpreted_as": ai_analysis.get('mission_type', 'unknown'),
            "confidence": ai_analysis.get('confidence', 0),
            "priority": ai_analysis.get('priority', 'normal')
        },
        "strategy": strategy,
        "assigned_drones": [
            {
                "drone": a.get('drone_name', 'Unknown'),
                "role": a.get('role', 'Unknown'),
                "task": a.get('primary_task', 'Unknown')
            }
            for a in mission_plan.get('drone_assignments', [])
        ],
        "estimated_duration": f"{ai_analysis.get('estimated_duration', 0)} minutes",
        "message": f"Mission planned successfully. {len(available_drones)} drones assigned."
    }

# Execute mission
@router.post("/{mission_id}/execute")
async def execute_mission(mission_id: int):
    """
    Execute a planned mission - drones will start flying
    """
    if mission_id not in active_missions:
        raise HTTPException(status_code=404, detail=f"Mission {mission_id} not found")
    
    if not drone_simulator:
        raise HTTPException(status_code=503, detail="Simulator not initialized")
    
    mission = active_missions[mission_id]
    
    if mission["status"] != "planned":
        raise HTTPException(
            status_code=400, 
            detail=f"Mission is {mission['status']}, cannot execute"
        )
    
    # Update status
    mission["status"] = "executing"
    mission["started_at"] = datetime.now().isoformat()
    
    # Execute flight plans for each drone
    flight_plans = mission.get("flight_plans", {})
    execution_tasks = []
    
    for drone_id_str, flight_plan in flight_plans.items():
        drone_id = int(drone_id_str)
        drone = drone_simulator.get_drone(drone_id)
        
        if drone:
            # Arm and takeoff
            await drone.arm()
            await drone.takeoff(altitude=15.0)
            
            # Set waypoints
            waypoints = flight_plan.get("waypoints", [])
            if waypoints:
                await drone.set_waypoints(waypoints)
    
    return {
        "success": True,
        "message": f"Mission {mission_id} execution started",
        "status": "executing",
        "drones_deployed": len(flight_plans),
        "mission_type": mission['ai_analysis']['mission_type'],
        "real_time_updates": "Connect to WebSocket /ws for live updates"
    }

# Get mission status
@router.get("/{mission_id}")
async def get_mission_status(mission_id: int):
    """
    Get detailed status of a mission
    """
    if mission_id not in active_missions:
        raise HTTPException(status_code=404, detail=f"Mission {mission_id} not found")
    
    mission = active_missions[mission_id]
    
    # Get current drone positions if executing
    if mission["status"] == "executing" and drone_simulator:
        drone_updates = []
        for assignment in mission.get("drone_assignments", []):
            drone = drone_simulator.get_drone(assignment["drone_id"])
            if drone:
                drone_updates.append({
                    "drone_id": drone.id,
                    "name": drone.name,
                    "role": assignment["role"],
                    "position": drone.position,
                    "battery": drone.battery,
                    "status": drone.status
                })
        
        mission["current_drone_status"] = drone_updates
    
    return mission

# Abort mission
@router.post("/{mission_id}/abort")
async def abort_mission(mission_id: int):
    """
    Abort a running mission - all drones return home
    """
    if mission_id not in active_missions:
        raise HTTPException(status_code=404, detail=f"Mission {mission_id} not found")
    
    if not drone_simulator:
        raise HTTPException(status_code=503, detail="Simulator not initialized")
    
    mission = active_missions[mission_id]
    
    # Send return home command to all assigned drones
    for assignment in mission.get("drone_assignments", []):
        drone = drone_simulator.get_drone(assignment["drone_id"])
        if drone and drone.status == "flying":
            await drone.return_home()
    
    # Update mission status
    mission["status"] = "aborted"
    mission["ended_at"] = datetime.now().isoformat()
    
    return {
        "success": True,
        "message": f"Mission {mission_id} aborted",
        "drones_recalled": len(mission.get("drone_assignments", []))
    }

# List all missions
@router.get("/")
async def list_missions(
    status: Optional[str] = None,
    limit: int = 10
):
    """
    List all missions with optional status filter
    """
    missions = list(active_missions.values())
    
    # Filter by status if specified
    if status:
        missions = [m for m in missions if m["status"] == status]
    
    # Sort by creation time (newest first)
    missions.sort(key=lambda m: m["created_at"], reverse=True)
    
    # Limit results
    missions = missions[:limit]
    
    # Simplify response
    return {
        "count": len(missions),
        "missions": [
            {
                "id": m["id"],
                "command": m["user_command"],
                "status": m["status"],
                "type": m["ai_analysis"]["mission_type"],
                "created_at": m["created_at"],
                "drone_count": len(m.get("drone_assignments", [])),
                "strategy": m["strategy"]["name"]
            }
            for m in missions
        ]
    }

# Get AI explanation
@router.post("/explain")
async def explain_mission_planning(
    command: str = Body(..., description="Natural language command to explain")
):
    """
    Get AI explanation of how it would handle a mission without creating it
    """
    if not drone_simulator:
        raise HTTPException(status_code=503, detail="Simulator not initialized")
    
    # Get available drones for planning
    available_drones = []
    for drone in drone_simulator.get_all_drones():
        if drone.status in ["idle", "armed"] and drone.battery > 30:
            available_drones.append({
                "id": drone.id,
                "name": drone.name,
                "battery": drone.battery
            })
    
    # Get AI analysis
    mission_plan = ai_planner.analyze_mission(
        user_command=command,
        available_drones=available_drones,
        environment_data={"type": "unknown"}
    )
    
    return {
        "user_command": command,
        "ai_interpretation": {
            "understood_as": mission_plan['ai_analysis']['mission_type'],
            "confidence": mission_plan['ai_analysis']['confidence'],
            "priority_level": mission_plan['ai_analysis']['priority']
        },
        "proposed_strategy": {
            "name": mission_plan['strategy']['name'],
            "description": mission_plan['strategy']['description'],
            "approach": mission_plan['strategy']['approach']
        },
        "drone_allocation": [
            f"{a['drone_name']} as {a['role']}: {a['primary_task']}"
            for a in mission_plan['drone_assignments']
        ],
        "estimated_duration": f"{mission_plan['ai_analysis']['estimated_duration']} minutes",
        "success_criteria": mission_plan['success_criteria']
    }