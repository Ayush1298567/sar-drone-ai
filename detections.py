"""
Detection API endpoints - Handles detection events from drones
Manages human detection, hazard identification, and verification
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import random

from core.database import get_db, Detection as DetectionModel
from modules.simulator import DroneSimulator

router = APIRouter()

# Store detections in memory for real-time access
active_detections: Dict[int, Dict] = {}
detection_counter = 0
drone_simulator: Optional[DroneSimulator] = None

def set_simulator(simulator: DroneSimulator):
    """Set the simulator instance - called from main.py"""
    global drone_simulator
    drone_simulator = simulator

# Report new detection
@router.post("/report")
async def report_detection(
    drone_id: int,
    detection_type: str,
    confidence: float,
    position: Dict[str, float],
    temperature: Optional[float] = None,
    image_data: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Report a new detection from a drone
    
    Detection types:
    - human: Possible human detected
    - heat_signature: Unusual heat source
    - movement: Movement detected
    - hazard: Dangerous condition (fire, gas, structural)
    - obstacle: Path blocked
    """
    global detection_counter
    
    # Validate confidence
    if not 0 <= confidence <= 1:
        raise HTTPException(status_code=400, detail="Confidence must be between 0 and 1")
    
    # Create detection ID
    detection_counter += 1
    detection_id = detection_counter
    
    # Determine priority based on type and confidence
    priority = "low"
    if detection_type == "human" and confidence > 0.7:
        priority = "critical"
    elif detection_type == "hazard":
        priority = "high"
    elif confidence > 0.8:
        priority = "medium"
    
    # Store detection
    detection_data = {
        "id": detection_id,
        "drone_id": drone_id,
        "type": detection_type,
        "confidence": confidence,
        "position": position,
        "temperature": temperature,
        "priority": priority,
        "status": "unverified",
        "reported_at": datetime.now().isoformat(),
        "image_data": image_data,
        "verification_required": detection_type in ["human", "hazard"],
        "false_positive": False,
        "notes": []
    }
    
    active_detections[detection_id] = detection_data
    
    # Save to database
    db_detection = DetectionModel(
        drone_id=drone_id,
        type=detection_type,
        confidence=confidence,
        position_x=position.get("x", 0),
        position_y=position.get("y", 0),
        position_z=position.get("z", 0),
        temperature=temperature,
        verified=False
    )
    db.add(db_detection)
    db.commit()
    db.refresh(db_detection)
    
    return {
        "success": True,
        "detection_id": detection_id,
        "priority": priority,
        "message": f"{detection_type.capitalize()} detection reported",
        "requires_verification": detection_data["verification_required"],
        "action_required": _get_required_action(detection_type, priority)
    }

# Get all detections
@router.get("/")
async def get_detections(
    type: Optional[str] = Query(None, description="Filter by detection type"),
    priority: Optional[str] = Query(None, description="Filter by priority"),
    status: Optional[str] = Query(None, description="Filter by status"),
    verified: Optional[bool] = Query(None, description="Filter by verification status"),
    last_hours: Optional[int] = Query(24, description="Get detections from last N hours")
):
    """
    Get all detections with optional filters
    """
    detections = list(active_detections.values())
    
    # Apply filters
    if type:
        detections = [d for d in detections if d["type"] == type]
    if priority:
        detections = [d for d in detections if d["priority"] == priority]
    if status:
        detections = [d for d in detections if d["status"] == status]
    if verified is not None:
        detections = [d for d in detections if d.get("verified", False) == verified]
    
    # Time filter
    if last_hours:
        cutoff_time = datetime.now() - timedelta(hours=last_hours)
        detections = [
            d for d in detections 
            if datetime.fromisoformat(d["reported_at"]) > cutoff_time
        ]
    
    # Sort by priority and time
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    detections.sort(
        key=lambda d: (
            priority_order.get(d["priority"], 999),
            d["reported_at"]
        ),
        reverse=True
    )
    
    return {
        "count": len(detections),
        "detections": detections,
        "summary": _get_detection_summary(detections)
    }

# Get specific detection
@router.get("/{detection_id}")
async def get_detection(detection_id: int):
    """
    Get details of a specific detection
    """
    if detection_id not in active_detections:
        raise HTTPException(status_code=404, detail=f"Detection {detection_id} not found")
    
    detection = active_detections[detection_id]
    
    # Add additional context
    if drone_simulator:
        drone = drone_simulator.get_drone(detection["drone_id"])
        if drone:
            detection["current_drone_position"] = drone.position
            detection["drone_status"] = drone.status
    
    return detection

# Verify detection
@router.post("/{detection_id}/verify")
async def verify_detection(
    detection_id: int,
    verified: bool,
    notes: Optional[str] = None,
    verified_by: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Verify or reject a detection
    """
    if detection_id not in active_detections:
        raise HTTPException(status_code=404, detail=f"Detection {detection_id} not found")
    
    detection = active_detections[detection_id]
    
    # Update verification status
    detection["verified"] = verified
    detection["status"] = "verified" if verified else "rejected"
    detection["verified_at"] = datetime.now().isoformat()
    detection["verified_by"] = verified_by or "operator"
    
    if notes:
        detection["notes"].append({
            "time": datetime.now().isoformat(),
            "text": notes,
            "by": verified_by or "operator"
        })
    
    # Update false positive flag
    if not verified and detection["type"] == "human":
        detection["false_positive"] = True
    
    # Update database
    db_detection = db.query(DetectionModel).filter(
        DetectionModel.id == detection_id
    ).first()
    if db_detection:
        db_detection.verified = verified  # type: ignore[linter]
        db_detection.false_positive = not verified  # type: ignore[linter]
        if notes:
            import json
            db_detection.notes = json.dumps(notes)  # type: ignore[linter]
        db.commit()
    
    return {
        "success": True,
        "detection_id": detection_id,
        "status": detection["status"],
        "message": f"Detection {detection_id} {'verified' if verified else 'rejected'}",
        "follow_up_action": _get_follow_up_action(detection, verified)
    }

# Request drone investigation
@router.post("/{detection_id}/investigate")
async def investigate_detection(
    detection_id: int,
    drone_id: Optional[int] = None
):
    """
    Send a drone to investigate a detection more closely
    """
    if detection_id not in active_detections:
        raise HTTPException(status_code=404, detail=f"Detection {detection_id} not found")
    
    if not drone_simulator:
        raise HTTPException(status_code=503, detail="Simulator not initialized")
    
    detection = active_detections[detection_id]
    
    # Find available drone if not specified
    if drone_id is None:
        # Find closest available drone
        available_drones = [
            d for d in drone_simulator.get_all_drones()
            if d.status == "flying" and d.battery > 40
        ]
        
        if not available_drones:
            raise HTTPException(
                status_code=400,
                detail="No drones available for investigation"
            )
        
        # Find closest drone
        closest_drone = min(
            available_drones,
            key=lambda d: _calculate_distance(d.position, detection["position"])
        )
        drone_id = closest_drone.id
    
    # Send drone to investigate
    drone = drone_simulator.get_drone(drone_id)
    if not drone:
        raise HTTPException(status_code=404, detail=f"Drone {drone_id} not found")
    
    # Command drone to go to detection location
    investigate_position = {
        "x": detection["position"]["x"],
        "y": detection["position"]["y"],
        "z": 5.0  # Lower altitude for better investigation
    }
    
    await drone.goto(
        investigate_position["x"],
        investigate_position["y"],
        investigate_position["z"]
    )
    
    # Update detection status
    detection["status"] = "investigating"
    detection["investigating_drone_id"] = drone_id
    
    return {
        "success": True,
        "message": f"Drone {drone.name} dispatched to investigate",
        "drone_id": drone_id,
        "eta_seconds": _calculate_eta(drone.position, investigate_position, drone.max_speed),
        "investigation_plan": {
            "approach_altitude": 5.0,
            "scan_pattern": "circular",
            "scan_radius": 3.0,
            "thermal_priority": detection["type"] == "human"
        }
    }

# Get detection statistics
@router.get("/stats/summary")
async def get_detection_statistics():
    """
    Get summary statistics of all detections
    """
    detections = list(active_detections.values())
    
    if not detections:
        return {
            "total_detections": 0,
            "by_type": {},
            "by_priority": {},
            "verification_rate": 0,
            "false_positive_rate": 0
        }
    
    # Calculate statistics
    total = len(detections)
    verified = [d for d in detections if d.get("verified", False)]
    false_positives = [d for d in detections if d.get("false_positive", False)]
    
    # Group by type
    by_type = {}
    for d in detections:
        dtype = d["type"]
        if dtype not in by_type:
            by_type[dtype] = {"count": 0, "verified": 0}
        by_type[dtype]["count"] += 1
        if d.get("verified", False):
            by_type[dtype]["verified"] += 1
    
    # Group by priority
    by_priority = {}
    for d in detections:
        priority = d["priority"]
        if priority not in by_priority:
            by_priority[priority] = 0
        by_priority[priority] += 1
    
    return {
        "total_detections": total,
        "verified_detections": len(verified),
        "pending_verification": len([d for d in detections if d["status"] == "unverified"]),
        "by_type": by_type,
        "by_priority": by_priority,
        "verification_rate": len(verified) / total if total > 0 else 0,
        "false_positive_rate": len(false_positives) / total if total > 0 else 0,
        "critical_detections": len([d for d in detections if d["priority"] == "critical"]),
        "recent_activity": {
            "last_hour": len([
                d for d in detections
                if (datetime.now() - datetime.fromisoformat(d["reported_at"])).seconds < 3600
            ]),
            "last_15_min": len([
                d for d in detections
                if (datetime.now() - datetime.fromisoformat(d["reported_at"])).seconds < 900
            ])
        }
    }

# Simulate detection for testing
@router.post("/simulate")
async def simulate_detection(
    detection_type: str = "human",
    count: int = 1
):
    """
    Simulate detections for testing (only in debug mode)
    """
    if not drone_simulator:
        raise HTTPException(status_code=503, detail="Simulator not initialized")
    
    simulated = []
    
    for _ in range(count):
        # Get random drone
        drones = drone_simulator.get_all_drones()
        if not drones:
            break
            
        drone = random.choice(drones)
        
        # Generate random detection near drone
        detection_data = {
            "drone_id": drone.id,
            "type": detection_type,
            "confidence": round(random.uniform(0.6, 0.95), 2),
            "position": {
                "x": drone.position["x"] + random.uniform(-10, 10),
                "y": drone.position["y"] + random.uniform(-10, 10),
                "z": 0
            },
            "temperature": round(random.uniform(35, 38), 1) if detection_type == "human" else None
        }
        
        # Report detection
        from core.database import SessionLocal, get_db
        db = next(get_db(SessionLocal))
        result = await report_detection(
            drone_id=detection_data["drone_id"],
            detection_type=detection_data["type"],
            confidence=detection_data["confidence"],
            position=detection_data["position"],
            temperature=detection_data["temperature"],
            db=db
        )
        
        simulated.append(result)
    
    return {
        "success": True,
        "simulated_count": len(simulated),
        "detections": simulated
    }

# Helper functions
def _get_required_action(detection_type: str, priority: str) -> str:
    """Get required action based on detection"""
    if detection_type == "human" and priority == "critical":
        return "Immediate investigation required - possible survivor"
    elif detection_type == "hazard":
        return "Mark area as dangerous - avoid sending drones"
    elif detection_type == "movement":
        return "Monitor area for additional activity"
    else:
        return "Log for review"

def _get_follow_up_action(detection: Dict, verified: bool) -> str:
    """Get follow-up action after verification"""
    if verified and detection["type"] == "human":
        return "Dispatch rescue team to location"
    elif verified and detection["type"] == "hazard":
        return "Update map with hazard zone"
    elif not verified:
        return "Update AI model to reduce false positives"
    else:
        return "Continue monitoring"

def _calculate_distance(pos1: Dict, pos2: Dict) -> float:
    """Calculate 3D distance between positions"""
    dx = pos1["x"] - pos2["x"]
    dy = pos1["y"] - pos2["y"]
    dz = pos1.get("z", 0) - pos2.get("z", 0)
    return (dx**2 + dy**2 + dz**2) ** 0.5

def _calculate_eta(current_pos: Dict, target_pos: Dict, speed: float) -> int:
    """Calculate estimated time of arrival in seconds"""
    distance = _calculate_distance(current_pos, target_pos)
    return int(distance / speed) if speed > 0 else 0

def _get_detection_summary(detections: List[Dict]) -> Dict:
    """Generate summary of detections"""
    if not detections:
        return {"status": "No detections"}
    
    human_detections = [d for d in detections if d["type"] == "human" and d.get("verified", False)]
    hazard_detections = [d for d in detections if d["type"] == "hazard"]
    
    return {
        "humans_found": len(human_detections),
        "hazards_identified": len(hazard_detections),
        "areas_of_interest": len(set(
            f"{int(d['position']['x']//10)},{int(d['position']['y']//10)}"
            for d in detections
        )),
        "requires_immediate_attention": len([
            d for d in detections
            if d["priority"] == "critical" and d["status"] == "unverified"
        ])
    }