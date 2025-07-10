"""
Drone Simulator - Simulates drone behavior for testing without real hardware
This creates virtual drones that behave like real ones
"""

import asyncio
import random
import math
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class SimulatedDrone:
    """
    Represents a single simulated drone
    """
    def __init__(self, drone_id: int, name: str):
        self.id = drone_id
        self.name = name
        
        # Status
        self.status = "idle"  # idle, armed, flying, returning, emergency
        self.battery = 100.0
        self.armed = False
        
        # Position (meters from origin)
        self.position = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.velocity = {"x": 0.0, "y": 0.0, "z": 0.0}
        self.heading = 0.0  # degrees
        
        # Target position for autonomous flight
        self.target_position = None
        self.home_position = {"x": 0.0, "y": 0.0, "z": 0.0}
        
        # Simulation parameters
        self.max_speed = 5.0  # m/s
        self.max_altitude = 50.0  # meters
        self.battery_drain_rate = 0.1  # % per second while flying
        
        # Mission data
        self.waypoints = []
        self.current_waypoint_index = 0
        
        # Sensor simulation
        self.thermal_camera_on = True
        self.detection_range = 10.0  # meters
        
    async def update(self, dt: float):
        """
        Update drone state - called every frame
        dt: time delta in seconds
        """
        # Update battery
        if self.status == "flying":
            self.battery -= self.battery_drain_rate * dt
            self.battery = max(0, self.battery)
            
            # Emergency landing if battery critical
            if self.battery < 10:
                self.status = "emergency"
                self.target_position = {"x": self.position["x"], "y": self.position["y"], "z": 0}
        
        # Update position if flying
        if self.status in ["flying", "returning", "emergency"]:
            await self._update_position(dt)
        
        # Check if reached target
        if self.target_position and self._distance_to_target() < 0.5:
            await self._reached_target()
    
    async def arm(self):
        """
        Arm the drone motors
        """
        if self.status == "idle" and self.battery > 20:
            self.armed = True
            self.status = "armed"
            logger.info(f"Drone {self.name} armed")
            return True
        return False
    
    async def disarm(self):
        """
        Disarm the drone motors
        """
        if self.status in ["armed", "landed"]:
            self.armed = False
            self.status = "idle"
            logger.info(f"Drone {self.name} disarmed")
            return True
        return False
    
    async def takeoff(self, altitude: float = 10.0):
        """
        Takeoff to specified altitude
        """
        if self.status == "armed":
            self.status = "flying"
            self.target_position = {
                "x": self.position["x"],
                "y": self.position["y"],
                "z": min(altitude, self.max_altitude)
            }
            logger.info(f"Drone {self.name} taking off to {altitude}m")
            return True
        return False
    
    async def land(self):
        """
        Land at current position
        """
        if self.status in ["flying", "returning"]:
            self.target_position = {
                "x": self.position["x"],
                "y": self.position["y"],
                "z": 0.0
            }
            self.status = "flying"  # Will change to landed when reaches ground
            logger.info(f"Drone {self.name} landing")
            return True
        return False
    
    async def goto(self, x: float, y: float, z: float = None):
        """
        Fly to specified position
        """
        if self.status == "flying":
            if z is None:
                z = self.position["z"]
            
            self.target_position = {
                "x": x,
                "y": y,
                "z": min(z, self.max_altitude)
            }
            logger.info(f"Drone {self.name} going to ({x}, {y}, {z})")
            return True
        return False
    
    async def return_home(self):
        """
        Return to home position
        """
        if self.status == "flying":
            self.status = "returning"
            self.target_position = self.home_position.copy()
            self.target_position["z"] = 10.0  # Fly at 10m while returning
            logger.info(f"Drone {self.name} returning home")
            return True
        return False
    
    async def set_waypoints(self, waypoints: List[Dict]):
        """
        Set mission waypoints
        """
        self.waypoints = waypoints
        self.current_waypoint_index = 0
        if waypoints and self.status == "flying":
            await self._go_to_next_waypoint()
    
    def get_telemetry(self) -> Dict:
        """
        Get current telemetry data
        """
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "armed": self.armed,
            "battery": round(self.battery, 1),
            "position": {
                "x": round(self.position["x"], 2),
                "y": round(self.position["y"], 2),
                "z": round(self.position["z"], 2)
            },
            "velocity": {
                "x": round(self.velocity["x"], 2),
                "y": round(self.velocity["y"], 2),
                "z": round(self.velocity["z"], 2)
            },
            "heading": round(self.heading, 1),
            "timestamp": datetime.now().isoformat()
        }
    
    def simulate_detection(self) -> Optional[Dict]:
        """
        Simulate random detection events
        """
        if self.status == "flying" and self.thermal_camera_on:
            # 5% chance of detection per update
            if random.random() < 0.05:
                # Generate random detection near drone
                detection = {
                    "type": random.choice(["human", "vehicle", "heat_signature"]),
                    "confidence": round(random.uniform(0.6, 0.95), 2),
                    "position": {
                        "x": self.position["x"] + random.uniform(-5, 5),
                        "y": self.position["y"] + random.uniform(-5, 5),
                        "z": 0  # Ground level
                    },
                    "temperature": round(random.uniform(35, 38), 1) if random.random() > 0.5 else None
                }
                logger.info(f"Drone {self.name} detected {detection['type']} at confidence {detection['confidence']}")
                return detection
        return None
    
    async def _update_position(self, dt: float):
        """
        Update position based on target
        """
        if not self.target_position:
            return
        
        # Calculate direction to target
        dx = self.target_position["x"] - self.position["x"]
        dy = self.target_position["y"] - self.position["y"]
        dz = self.target_position["z"] - self.position["z"]
        
        distance = math.sqrt(dx**2 + dy**2 + dz**2)
        
        if distance > 0.1:  # Not at target yet
            # Normalize direction
            dx /= distance
            dy /= distance
            dz /= distance
            
            # Set velocity
            speed = min(self.max_speed, distance / dt)  # Slow down when approaching
            self.velocity["x"] = dx * speed
            self.velocity["y"] = dy * speed
            self.velocity["z"] = dz * speed
            
            # Update position
            self.position["x"] += self.velocity["x"] * dt
            self.position["y"] += self.velocity["y"] * dt
            self.position["z"] += self.velocity["z"] * dt
            
            # Update heading
            if dx != 0 or dy != 0:
                self.heading = math.degrees(math.atan2(dy, dx))
    
    def _distance_to_target(self) -> float:
        """
        Calculate distance to target position
        """
        if not self.target_position:
            return 0
        
        dx = self.target_position["x"] - self.position["x"]
        dy = self.target_position["y"] - self.position["y"]
        dz = self.target_position["z"] - self.position["z"]
        
        return math.sqrt(dx**2 + dy**2 + dz**2)
    
    async def _reached_target(self):
        """
        Called when drone reaches its target
        """
        # Check if landed
        if self.position["z"] < 0.1:
            if self.status == "returning":
                self.status = "armed"
                logger.info(f"Drone {self.name} returned home")
            else:
                self.status = "armed"
                logger.info(f"Drone {self.name} landed")
        
        # Check for next waypoint
        elif self.waypoints and self.current_waypoint_index < len(self.waypoints):
            await self._go_to_next_waypoint()
    
    async def _go_to_next_waypoint(self):
        """
        Go to next waypoint in mission
        """
        if self.current_waypoint_index < len(self.waypoints):
            waypoint = self.waypoints[self.current_waypoint_index]
            await self.goto(waypoint["x"], waypoint["y"], waypoint.get("z", 10))
            self.current_waypoint_index += 1


class DroneSimulator:
    """
    Manages all simulated drones
    """
    def __init__(self):
        self.drones: Dict[int, SimulatedDrone] = {}
        self.running = False
        self.update_interval = 0.1  # 10Hz update rate
        self._task = None
        
    async def start(self):
        """
        Start the simulator
        """
        self.running = True
        self._task = asyncio.create_task(self._simulation_loop())
        logger.info("Drone simulator started")
        
        # Create default drones
        await self.create_drone(1, "Alpha")
        await self.create_drone(2, "Bravo")
        await self.create_drone(3, "Charlie")
    
    async def stop(self):
        """
        Stop the simulator
        """
        self.running = False
        if self._task:
            await self._task
        logger.info("Drone simulator stopped")
    
    async def create_drone(self, drone_id: int, name: str) -> SimulatedDrone:
        """
        Create a new simulated drone
        """
        drone = SimulatedDrone(drone_id, name)
        self.drones[drone_id] = drone
        logger.info(f"Created simulated drone: {name} (ID: {drone_id})")
        return drone
    
    def get_drone(self, drone_id: int) -> Optional[SimulatedDrone]:
        """
        Get drone by ID
        """
        return self.drones.get(drone_id)
    
    def get_all_drones(self) -> List[SimulatedDrone]:
        """
        Get all drones
        """
        return list(self.drones.values())
    
    async def _simulation_loop(self):
        """
        Main simulation loop
        """
        last_time = asyncio.get_event_loop().time()
        
        while self.running:
            current_time = asyncio.get_event_loop().time()
            dt = current_time - last_time
            last_time = current_time
            
            # Update all drones
            for drone in self.drones.values():
                await drone.update(dt)
            
            # Wait for next update
            await asyncio.sleep(self.update_interval)