"""
AI Coordinator - Pure AI-driven swarm coordination
The AI makes ALL coordination decisions based on context
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

from core.config import settings
from core.ai_manager import ai_manager
from core.errors import log_mission_critical, log_safety_event

logger = logging.getLogger(__name__)

@dataclass
class DroneState:
    """Complete state of a single drone"""
    id: int
    name: str
    position: Dict[str, float]  # x, y, z
    battery: float
    status: str  # idle, flying, investigating, returning
    current_task: Optional[str] = None
    last_update: Optional[datetime] = None
    sensor_data: Optional[Dict[str, Any]] = None
    
    def to_context_string(self) -> str:
        """Convert to string for AI context"""
        return f"Drone {self.name} (ID:{self.id}): Position({self.position['x']:.1f}, {self.position['y']:.1f}, {self.position['z']:.1f}m), Battery:{self.battery}%, Status:{self.status}, Task:{self.current_task or 'none'}"

@dataclass 
class MissionContext:
    """Complete mission context for AI decision making"""
    mission_type: str
    mission_description: str
    start_time: datetime
    priority: str
    environment: str  # building, outdoor, underground
    weather_conditions: Dict[str, Any]
    known_hazards: List[str]
    discovered_items: List[Dict[str, Any]]
    covered_area: List[Dict[str, float]]  # List of covered coordinates
    no_fly_zones: List[Dict[str, float]]
    time_elapsed: float
    
    def to_context_string(self) -> str:
        """Convert to comprehensive string for AI"""
        return f"""Mission Type: {self.mission_type}
Description: {self.mission_description}
Priority: {self.priority}
Environment: {self.environment}
Time Elapsed: {self.time_elapsed:.1f} seconds
Discovered Items: {len(self.discovered_items)} items found
Coverage: {len(self.covered_area)} points covered
Known Hazards: {', '.join(self.known_hazards) if self.known_hazards else 'none'}
Weather: {json.dumps(self.weather_conditions)}"""

class AICoordinator:
    """
    Pure AI-driven coordinator - AI makes ALL decisions
    No algorithms, just AI reasoning based on context
    """
    
    def __init__(self):
        self.drones: Dict[int, DroneState] = {}
        self.mission_context: Optional[MissionContext] = None
        self.coordination_history: List[Dict[str, Any]] = []
        self.active = False
        self._coordination_task = None
        self._decision_interval = 2.0  # Ask AI for decisions every 2 seconds
        
    async def initialize(self, mission_context: MissionContext, drones: List[DroneState]):
        """Initialize coordinator with mission context and available drones"""
        self.mission_context = mission_context
        self.drones = {drone.id: drone for drone in drones}
        
        logger.info(f"AI Coordinator initialized with {len(drones)} drones for {mission_context.mission_type}")
        
        # Get initial AI strategy
        initial_strategy = await self._get_ai_strategy()
        logger.info(f"AI Initial Strategy: {initial_strategy}")
        
    async def start_coordination(self):
        """Start AI coordination loop"""
        self.active = True
        self._coordination_task = asyncio.create_task(self._ai_coordination_loop())
        logger.info("AI Coordinator started - AI is now in control")
        
    async def stop_coordination(self):
        """Stop AI coordination"""
        self.active = False
        if self._coordination_task:
            self._coordination_task.cancel()
            await asyncio.gather(self._coordination_task, return_exceptions=True)
        logger.info("AI Coordinator stopped")
        
    async def _ai_coordination_loop(self):
        """Main loop - AI makes all decisions"""
        while self.active:
            try:
                # Update mission context time
                if self.mission_context:
                    self.mission_context.time_elapsed = (
                        datetime.now() - self.mission_context.start_time
                    ).total_seconds()
                
                # Get complete context
                full_context = self._build_complete_context()
                
                # Ask AI what to do
                ai_decision = await self._get_ai_coordination_decision(full_context)
                
                # Execute AI's decision
                await self._execute_ai_decision(ai_decision)
                
                # Record decision for learning
                self._record_decision(ai_decision)
                
                # Wait before next decision
                await asyncio.sleep(self._decision_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"AI coordination error: {e}", exc_info=True)
                await asyncio.sleep(self._decision_interval)
    
    def _build_complete_context(self) -> str:
        """Build comprehensive context string for AI"""
        context_parts = []
        
        # Mission context
        if self.mission_context:
            context_parts.append("=== MISSION CONTEXT ===")
            context_parts.append(self.mission_context.to_context_string())
        
        # Drone states
        context_parts.append("\n=== DRONE STATES ===")
        for drone in self.drones.values():
            context_parts.append(drone.to_context_string())
        
        # Recent discoveries
        if self.mission_context and self.mission_context.discovered_items:
            context_parts.append("\n=== RECENT DISCOVERIES ===")
            for item in self.mission_context.discovered_items[-5:]:  # Last 5
                context_parts.append(f"- {item['type']} at ({item['x']:.1f}, {item['y']:.1f}) with confidence {item['confidence']:.2f}")
        
        # Current challenges
        context_parts.append("\n=== CURRENT SITUATION ===")
        challenges = self._identify_current_challenges()
        for challenge in challenges:
            context_parts.append(f"- {challenge}")
            
        return "\n".join(context_parts)
    
    def _identify_current_challenges(self) -> List[str]:
        """Identify current challenges for AI context"""
        challenges = []
        
        # Check drone batteries
        low_battery_drones = [d for d in self.drones.values() if d.battery < 30]
        if low_battery_drones:
            challenges.append(f"{len(low_battery_drones)} drones have low battery")
        
        # Check coverage
        if self.mission_context:
            if len(self.mission_context.covered_area) < 10:
                challenges.append("Limited area coverage so far")
            
            # Check time pressure
            if self.mission_context.priority == "critical" and self.mission_context.time_elapsed > 300:
                challenges.append("Critical mission running for over 5 minutes")
        
        # Check idle drones
        idle_drones = [d for d in self.drones.values() if d.status == "idle"]
        if idle_drones:
            challenges.append(f"{len(idle_drones)} drones are idle")
            
        return challenges if challenges else ["No immediate challenges"]
    
    async def _get_ai_strategy(self) -> str:
        """Get initial strategy from AI"""
        prompt = f"""You are coordinating a search and rescue drone swarm.

{self._build_complete_context()}

What is your overall strategy for this mission? Be specific about:
1. Search pattern (grid, spiral, expanding squares, converging)
2. Drone spacing and altitude
3. Priority areas to search first
4. How to handle discoveries
5. Safety considerations

Provide a clear, actionable strategy."""

        response = await ai_manager.get_response(
            prompt=prompt,
            model_type="mission_planning",
            context={"coordination": "strategy"}
        )
        
        return response.content
    
    async def _get_ai_coordination_decision(self, context: str) -> Dict[str, Any]:
        """Get specific coordination decision from AI"""
        prompt = f"""You are the AI coordinator for a search and rescue drone swarm. Based on the current situation, decide what each drone should do RIGHT NOW.

{context}

Provide your decision in this EXACT JSON format:
{{
    "reasoning": "Brief explanation of your decision",
    "commands": [
        {{
            "drone_id": 1,
            "action": "move_to",
            "parameters": {{"x": 50, "y": 100, "z": 15}},
            "reason": "Search unexplored northeast area"
        }},
        {{
            "drone_id": 2,
            "action": "investigate",
            "parameters": {{"x": 30, "y": 40, "duration": 30}},
            "reason": "Verify potential survivor detection"
        }}
    ],
    "coordination_mode": "spread_search",  // or "converge", "relay_chain", "perimeter", etc.
    "next_focus": "Complete northeast quadrant search",
    "safety_notes": "Drone 3 battery at 25%, will need recall soon"
}}

Actions available: move_to, investigate, return_home, hover, relay_position, scan_area
Be decisive and specific. Every drone should have a task."""

        response = await ai_manager.get_response(
            prompt=prompt,
            model_type="fast_decisions",
            context={"coordination": "real_time"}
        )
        
        try:
            # Parse JSON from response
            json_start = response.content.find('{')
            json_end = response.content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                decision = json.loads(response.content[json_start:json_end])
                decision["confidence"] = response.confidence
                decision["timestamp"] = datetime.now().isoformat()
                return decision
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            logger.error(f"Failed to parse AI decision: {e}")
            # Return safe default
            return {
                "reasoning": "Failed to parse AI response, maintaining current operations",
                "commands": [],
                "coordination_mode": "maintain",
                "confidence": 0.0
            }
    
    async def _execute_ai_decision(self, decision: Dict[str, Any]):
        """Execute the AI's coordination decision"""
        logger.info(f"Executing AI decision: {decision.get('reasoning', 'No reasoning provided')}")
        
        # Execute each command
        for command in decision.get("commands", []):
            try:
                drone_id = command["drone_id"]
                action = command["action"]
                parameters = command.get("parameters", {})
                
                # Update drone state
                if drone_id in self.drones:
                    self.drones[drone_id].current_task = f"{action}: {command.get('reason', '')}"
                    self.drones[drone_id].last_update = datetime.now()
                    
                    # Log the command
                    logger.info(f"Drone {drone_id}: {action} with params {parameters}")
                    
                    # In real implementation, this would send actual commands to drones
                    # For now, we'll just update the state
                    if action == "move_to":
                        self.drones[drone_id].status = "flying"
                    elif action == "investigate":
                        self.drones[drone_id].status = "investigating"
                    elif action == "return_home":
                        self.drones[drone_id].status = "returning"
                        
            except Exception as e:
                logger.error(f"Failed to execute command for drone {command.get('drone_id')}: {e}")
        
        # Update coordination mode if changed
        if decision.get("coordination_mode"):
            logger.info(f"AI set coordination mode: {decision['coordination_mode']}")
            
        # Log safety notes
        if decision.get("safety_notes"):
            log_safety_event(f"AI safety note: {decision['safety_notes']}")
    
    def _record_decision(self, decision: Dict[str, Any]):
        """Record AI decision for learning and analysis"""
        self.coordination_history.append({
            "timestamp": datetime.now().isoformat(),
            "decision": decision,
            "drone_states": {
                drone_id: asdict(drone) 
                for drone_id, drone in self.drones.items()
            },
            "mission_elapsed": self.mission_context.time_elapsed if self.mission_context else 0
        })
        
        # Keep only last 100 decisions in memory
        if len(self.coordination_history) > 100:
            self.coordination_history = self.coordination_history[-100:]
    
    async def handle_new_detection(self, detection: Dict[str, Any]):
        """Handle new detection by asking AI what to do"""
        prompt = f"""New detection received during search and rescue mission:
Type: {detection['type']}
Location: ({detection['x']:.1f}, {detection['y']:.1f})
Confidence: {detection['confidence']:.2f}
Details: {detection.get('details', 'none')}

Current drone states:
{self._build_complete_context()}

How should the swarm respond to this detection? Should we:
1. Send one drone to investigate?
2. Send multiple drones to converge?
3. Continue current search pattern?
4. Change coordination strategy?

Provide your decision and reasoning."""

        response = await ai_manager.get_response(
            prompt=prompt,
            model_type="fast_decisions",
            context={"detection": detection}
        )
        
        logger.info(f"AI detection response: {response.content}")
        
        # Update mission context
        if self.mission_context:
            self.mission_context.discovered_items.append({
                **detection,
                "ai_response": response.content,
                "timestamp": datetime.now().isoformat()
            })
    
    async def handle_emergency(self, emergency_type: str, details: Dict[str, Any]):
        """Handle emergency by asking AI for immediate response"""
        log_mission_critical(f"Emergency: {emergency_type}", **details)
        
        prompt = f"""EMERGENCY SITUATION in search and rescue operation:
Type: {emergency_type}
Details: {json.dumps(details)}

Current situation:
{self._build_complete_context()}

Provide IMMEDIATE emergency response. Safety is the top priority.
What should each drone do RIGHT NOW?"""

        response = await ai_manager.get_response(
            prompt=prompt,
            model_type="safety_analysis",
            context={"emergency": True}
        )
        
        # Parse and execute emergency commands
        emergency_decision = {
            "reasoning": f"Emergency response: {emergency_type}",
            "commands": self._parse_emergency_response(response.content),
            "coordination_mode": "emergency",
            "safety_notes": response.content
        }
        
        await self._execute_ai_decision(emergency_decision)
    
    def _parse_emergency_response(self, ai_response: str) -> List[Dict[str, Any]]:
        """Parse emergency response into commands"""
        # In a real implementation, this would parse the AI's emergency instructions
        # For now, return all drones to home as safety measure
        commands = []
        for drone_id in self.drones:
            commands.append({
                "drone_id": drone_id,
                "action": "return_home",
                "parameters": {"emergency": True},
                "reason": "Emergency protocol activated"
            })
        return commands
    
    def get_status(self) -> Dict[str, Any]:
        """Get current coordination status"""
        return {
            "active": self.active,
            "mission_type": self.mission_context.mission_type if self.mission_context else "none",
            "time_elapsed": self.mission_context.time_elapsed if self.mission_context else 0,
            "active_drones": len([d for d in self.drones.values() if d.status != "idle"]),
            "total_drones": len(self.drones),
            "discoveries": len(self.mission_context.discovered_items) if self.mission_context else 0,
            "last_decision": self.coordination_history[-1] if self.coordination_history else None,
            "ai_confidence": self.coordination_history[-1]["decision"].get("confidence", 0) if self.coordination_history else 0
        }

# Global AI coordinator instance
ai_coordinator = AICoordinator()