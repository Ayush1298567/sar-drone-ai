"""
Mapping API endpoints - Handles map building and visualization
Creates real-time 2D/3D maps from drone exploration data
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np

router = APIRouter()

# Global map storage
class MapSystem:
    def __init__(self, size: int = 1000, resolution: float = 0.5):
        """
        Initialize map system
        size: map size in meters (creates size x size grid)
        resolution: meters per grid cell
        """
        self.size = size
        self.resolution = resolution
        self.grid_size = int(size / resolution)
        
        # 2D occupancy grid (-1: unknown, 0: empty, 1: occupied)
        self.occupancy_grid = np.full((self.grid_size, self.grid_size), -1, dtype=np.int8)
        
        # Height map (stores ceiling height or obstacle height)
        self.height_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # Confidence map (0-1, how confident we are about each cell)
        self.confidence_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # Danger zones (0: safe, 1: dangerous)
        self.danger_map = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        
        # Exploration status (0: unexplored, 1: explored)
        self.explored_map = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        
        # Special markers
        self.markers = {
            "drone_positions": {},
            "human_detections": [],
            "hazards": [],
            "points_of_interest": []
        }
        
        # Map metadata
        self.metadata = {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "total_area_explored": 0,
            "origin": {"x": size / 2, "y": size / 2}  # Center of map
        }
    
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices"""
        grid_x = int((x + self.metadata["origin"]["x"]) / self.resolution)
        grid_y = int((y + self.metadata["origin"]["y"]) / self.resolution)
        
        # Clamp to grid bounds
        grid_x = max(0, min(self.grid_size - 1, grid_x))
        grid_y = max(0, min(self.grid_size - 1, grid_y))
        
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates"""
        x = grid_x * self.resolution - self.metadata["origin"]["x"]
        y = grid_y * self.resolution - self.metadata["origin"]["y"]
        return x, y
    
    def update_cell(self, x: float, y: float, cell_type: str, confidence: float = 1.0):
        """Update a single map cell"""
        grid_x, grid_y = self.world_to_grid(x, y)
        
        # Update occupancy
        if cell_type == "empty":
            self.occupancy_grid[grid_y, grid_x] = 0
        elif cell_type == "occupied":
            self.occupancy_grid[grid_y, grid_x] = 1
        
        # Update confidence
        self.confidence_map[grid_y, grid_x] = max(
            self.confidence_map[grid_y, grid_x],
            confidence
        )
        
        # Mark as explored
        self.explored_map[grid_y, grid_x] = 1
        
        self.metadata["last_updated"] = datetime.now().isoformat()
    
    def get_exploration_percentage(self) -> float:
        """Calculate percentage of map explored"""
        explored_cells = np.sum(self.explored_map)
        total_cells = self.grid_size * self.grid_size
        return (explored_cells / total_cells) * 100

# Create global map instance
map_system = MapSystem()

# Update map with drone data
@router.post("/update")
async def update_map(
    drone_id: int,
    position: Dict[str, float],
    scan_data: List[Dict[str, float]],
    scan_type: str = "lidar"
):
    """
    Update map with sensor data from drone
    
    scan_data format: [{"angle": 0, "distance": 5.2}, ...]
    """
    drone_x = position.get("x", 0)
    drone_y = position.get("y", 0)
    drone_z = position.get("z", 0)
    
    # Update drone position
    map_system.markers["drone_positions"][drone_id] = {
        "x": drone_x,
        "y": drone_y,
        "z": drone_z,
        "last_update": datetime.now().isoformat()
    }
    
    # Process scan data
    for scan_point in scan_data:
        angle = scan_point.get("angle", 0)
        distance = scan_point.get("distance", 0)
        
        if distance > 0 and distance < 100:  # Valid range
            # Calculate obstacle position
            obstacle_x = drone_x + distance * np.cos(np.radians(angle))
            obstacle_y = drone_y + distance * np.sin(np.radians(angle))
            
            # Mark obstacle
            map_system.update_cell(obstacle_x, obstacle_y, "occupied")
            
            # Mark empty space between drone and obstacle
            steps = int(distance / map_system.resolution)
            for i in range(1, steps):
                ratio = i / steps
                empty_x = drone_x + ratio * (obstacle_x - drone_x)
                empty_y = drone_y + ratio * (obstacle_y - drone_y)
                map_system.update_cell(empty_x, empty_y, "empty", confidence=0.9)
    
    # Update exploration area around drone
    exploration_radius = 5  # meters
    grid_x, grid_y = map_system.world_to_grid(drone_x, drone_y)
    radius_cells = int(exploration_radius / map_system.resolution)
    
    for dx in range(-radius_cells, radius_cells + 1):
        for dy in range(-radius_cells, radius_cells + 1):
            gx = grid_x + dx
            gy = grid_y + dy
            if 0 <= gx < map_system.grid_size and 0 <= gy < map_system.grid_size:
                map_system.explored_map[gy, gx] = 1
    
    return {
        "success": True,
        "cells_updated": len(scan_data),
        "exploration_percentage": round(map_system.get_exploration_percentage(), 2)
    }

# Get current map data
@router.get("/data")
async def get_map_data(
    min_x: Optional[float] = None,
    max_x: Optional[float] = None,
    min_y: Optional[float] = None,
    max_y: Optional[float] = None,
    layer: str = Query("occupancy", description="Map layer to retrieve")
):
    """
    Get map data for visualization
    
    Layers: occupancy, height, confidence, danger, explored
    """
    # Determine bounds
    if min_x is not None and max_x is not None and min_y is not None and max_y is not None:
        # Convert bounds to grid coordinates
        min_gx, min_gy = map_system.world_to_grid(min_x, min_y)
        max_gx, max_gy = map_system.world_to_grid(max_x, max_y)
    else:
        # Return full map
        min_gx, min_gy = 0, 0
        max_gx, max_gy = map_system.grid_size, map_system.grid_size
    
    # Get requested layer
    if layer == "occupancy":
        data = map_system.occupancy_grid[min_gy:max_gy, min_gx:max_gx]
    elif layer == "height":
        data = map_system.height_map[min_gy:max_gy, min_gx:max_gx]
    elif layer == "confidence":
        data = map_system.confidence_map[min_gy:max_gy, min_gx:max_gx]
    elif layer == "danger":
        data = map_system.danger_map[min_gy:max_gy, min_gx:max_gx]
    elif layer == "explored":
        data = map_system.explored_map[min_gy:max_gy, min_gx:max_gx]
    else:
        raise HTTPException(status_code=400, detail=f"Unknown layer: {layer}")
    
    return {
        "layer": layer,
        "bounds": {
            "min_x": min_x or -map_system.metadata["origin"]["x"],
            "max_x": max_x or map_system.metadata["origin"]["x"],
            "min_y": min_y or -map_system.metadata["origin"]["y"],
            "max_y": max_y or map_system.metadata["origin"]["y"]
        },
        "resolution": map_system.resolution,
        "data": data.tolist(),  # Convert numpy array to list
        "shape": data.shape
    }

# Get map markers
@router.get("/markers")
async def get_map_markers(
    marker_type: Optional[str] = None
):
    """
    Get special markers on the map
    
    Types: drone_positions, human_detections, hazards, points_of_interest
    """
    if marker_type:
        if marker_type not in map_system.markers:
            raise HTTPException(status_code=400, detail=f"Unknown marker type: {marker_type}")
        return {
            "type": marker_type,
            "markers": map_system.markers[marker_type]
        }
    
    return map_system.markers

# Add marker to map
@router.post("/markers/add")
async def add_marker(
    marker_type: str,
    position: Dict[str, float],
    description: Optional[str] = None,
    priority: str = "normal"
):
    """
    Add a special marker to the map
    """
    if marker_type not in ["human_detections", "hazards", "points_of_interest"]:
        raise HTTPException(status_code=400, detail=f"Invalid marker type: {marker_type}")
    
    marker = {
        "id": len(map_system.markers[marker_type]) + 1,
        "position": position,
        "description": description,
        "priority": priority,
        "created_at": datetime.now().isoformat()
    }
    
    map_system.markers[marker_type].append(marker)
    
    # Mark danger zone if hazard
    if marker_type == "hazards":
        x, y = position.get("x", 0), position.get("y", 0)
        grid_x, grid_y = map_system.world_to_grid(x, y)
        
        # Mark 5m radius as dangerous
        danger_radius = int(5 / map_system.resolution)
        for dx in range(-danger_radius, danger_radius + 1):
            for dy in range(-danger_radius, danger_radius + 1):
                gx = grid_x + dx
                gy = grid_y + dy
                if 0 <= gx < map_system.grid_size and 0 <= gy < map_system.grid_size:
                    if dx*dx + dy*dy <= danger_radius*danger_radius:
                        map_system.danger_map[gy, gx] = 1
    
    return {
        "success": True,
        "marker_id": marker["id"],
        "message": f"{marker_type} marker added at ({position['x']}, {position['y']})"
    }

# Get map statistics
@router.get("/stats")
async def get_map_statistics():
    """
    Get statistics about the current map
    """
    # Calculate statistics
    total_cells = map_system.grid_size * map_system.grid_size
    explored_cells = np.sum(map_system.explored_map)
    occupied_cells = np.sum(map_system.occupancy_grid == 1)
    empty_cells = np.sum(map_system.occupancy_grid == 0)
    unknown_cells = np.sum(map_system.occupancy_grid == -1)
    danger_cells = np.sum(map_system.danger_map)
    
    # Area calculations
    cell_area = map_system.resolution * map_system.resolution
    
    return {
        "map_size": {
            "meters": map_system.size,
            "cells": map_system.grid_size,
            "resolution": map_system.resolution
        },
        "exploration": {
            "percentage": round(map_system.get_exploration_percentage(), 2),
            "explored_area_m2": explored_cells * cell_area,
            "unexplored_area_m2": (total_cells - explored_cells) * cell_area
        },
        "occupancy": {
            "occupied_cells": occupied_cells,
            "empty_cells": empty_cells,
            "unknown_cells": unknown_cells,
            "occupied_area_m2": occupied_cells * cell_area
        },
        "safety": {
            "danger_zones": danger_cells,
            "danger_area_m2": danger_cells * cell_area,
            "safe_area_m2": (explored_cells - danger_cells) * cell_area
        },
        "markers": {
            "active_drones": len(map_system.markers["drone_positions"]),
            "human_detections": len(map_system.markers["human_detections"]),
            "hazards": len(map_system.markers["hazards"]),
            "points_of_interest": len(map_system.markers["points_of_interest"])
        },
        "metadata": map_system.metadata
    }

# Clear map
@router.post("/clear")
async def clear_map(
    clear_markers: bool = True,
    clear_occupancy: bool = True
):
    """
    Clear map data (for new mission)
    """
    global map_system
    
    if clear_occupancy:
        map_system.occupancy_grid.fill(-1)
        map_system.height_map.fill(0)
        map_system.confidence_map.fill(0)
        map_system.danger_map.fill(0)
        map_system.explored_map.fill(0)
    
    if clear_markers:
        map_system.markers = {
            "drone_positions": {},
            "human_detections": [],
            "hazards": [],
            "points_of_interest": []
        }
    
    map_system.metadata["last_updated"] = datetime.now().isoformat()
    
    return {
        "success": True,
        "message": "Map cleared",
        "cleared": {
            "occupancy": clear_occupancy,
            "markers": clear_markers
        }
    }

# Find path between points
@router.post("/pathfind")
async def find_path(
    start: Dict[str, float],
    end: Dict[str, float],
    avoid_danger: bool = True
):
    """
    Find safe path between two points using A* algorithm
    """
    start_x, start_y = start.get("x", 0), start.get("y", 0)
    end_x, end_y = end.get("x", 0), end.get("y", 0)
    
    # Convert to grid coordinates
    start_gx, start_gy = map_system.world_to_grid(start_x, start_y)
    end_gx, end_gy = map_system.world_to_grid(end_x, end_y)
    
    # Simple pathfinding (can be replaced with A* for better results)
    path = []
    
    # For now, return direct path with waypoints
    distance = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
    num_waypoints = int(distance / 5) + 1  # Waypoint every 5 meters
    
    for i in range(num_waypoints + 1):
        ratio = i / num_waypoints
        waypoint_x = start_x + ratio * (end_x - start_x)
        waypoint_y = start_y + ratio * (end_y - start_y)
        
        # Check if waypoint is safe
        gx, gy = map_system.world_to_grid(waypoint_x, waypoint_y)
        is_safe = (
            map_system.occupancy_grid[gy, gx] != 1 and
            (not avoid_danger or map_system.danger_map[gy, gx] == 0)
        )
        
        path.append({
            "x": round(waypoint_x, 2),
            "y": round(waypoint_y, 2),
            "safe": is_safe
        })
    
    return {
        "path": path,
        "distance": round(distance, 2),
        "waypoint_count": len(path),
        "path_safe": all(p["safe"] for p in path)
    }