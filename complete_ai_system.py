"""
Complete AI System for SAR Drones
Includes all AI components with user confirmation at every step
"""

import json
import asyncio
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class AIComponent(Enum):
    """Different AI components in the system"""
    MISSION_INTERPRETER = "mission_interpreter"
    LOCATION_ANALYZER = "location_analyzer"
    MISSION_PLANNER = "mission_planner"
    COORDINATOR = "coordinator"
    DATA_ANALYZER = "data_analyzer"
    SAFETY_MONITOR = "safety_monitor"
    REPORT_GENERATOR = "report_generator"

class CompleteSARAI:
    """
    Complete AI system with all components and user confirmations
    """
    
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.models = {
            AIComponent.MISSION_INTERPRETER: "mistral:7b-instruct",
            AIComponent.LOCATION_ANALYZER: "phi:2.7b",
            AIComponent.MISSION_PLANNER: "mistral:7b-instruct",
            AIComponent.COORDINATOR: "phi:2.7b",
            AIComponent.DATA_ANALYZER: "llama2:7b",
            AIComponent.SAFETY_MONITOR: "phi:2.7b",
            AIComponent.REPORT_GENERATOR: "llama2:7b"
        }
        
        # Contexts for each AI component
        self.contexts = self._initialize_contexts()
        
        # Confirmation tracking
        self.pending_confirmations = {}
        
    def _initialize_contexts(self) -> Dict[AIComponent, str]:
        """Initialize context for each AI component"""
        
        base_context = """You are part of a Search and Rescue drone system.
        Core principles:
        1. Human life is the absolute priority
        2. Safety of rescue teams is critical
        3. Time is often critical in rescue operations
        4. Clear communication saves lives
        """
        
        return {
            AIComponent.MISSION_INTERPRETER: base_context + """
            Your role: Interpret human commands into clear mission objectives.
            
            Extract:
            - WHAT: The primary objective (find survivors, assess damage, etc.)
            - WHERE: Specific location or area
            - PRIORITY: Urgency level (immediate, systematic, careful)
            - CONSTRAINTS: Any limitations or special requirements
            
            Always ask for clarification if the location is unclear.
            Output format: JSON with these fields.""",
            
            AIComponent.LOCATION_ANALYZER: base_context + """
            Your role: Convert location descriptions to precise coordinates/zones.
            
            Reference points:
            - Base station: (0, 0)
            - Building entrance: (0, 50)
            - Map boundaries: -500 to +500 meters
            
            Convert vague descriptions to specific areas.
            Output format: JSON with coordinates and search radius.""",
            
            AIComponent.MISSION_PLANNER: base_context + """
            Your role: Create detailed execution plans for drone missions.
            
            Consider:
            - Available drones and battery levels
            - Optimal search patterns
            - Safety zones and no-fly areas
            - Efficient area coverage
            
            Output format: JSON with drone assignments and waypoints.""",
            
            AIComponent.COORDINATOR: base_context + """
            Your role: Make real-time decisions during active missions.
            
            Handle:
            - Dynamic re-routing
            - Resource reallocation
            - Emergency responses
            - Priority changes
            
            Always maintain safety margins.
            Output format: JSON with immediate actions.""",
            
            AIComponent.DATA_ANALYZER: base_context + """
            Your role: Analyze sensor data for important findings.
            
            Identify:
            - Human presence (thermal 35-38°C + movement)
            - Hazards (fire, gas, structural damage)
            - Safe passages
            - Areas needing investigation
            
            Output format: JSON with findings and confidence levels.""",
            
            AIComponent.SAFETY_MONITOR: base_context + """
            Your role: Monitor all operations for safety issues.
            
            Watch for:
            - Low battery situations
            - Dangerous areas
            - Lost communications
            - Weather hazards
            - Structural instabilities
            
            Immediately flag any safety concerns.
            Output format: JSON with safety status and required actions.""",
            
            AIComponent.REPORT_GENERATOR: base_context + """
            Your role: Generate clear reports for rescue teams.
            
            Create:
            - Executive summaries
            - Detailed findings
            - Recommended actions
            - Resource status
            
            Use clear, non-technical language.
            Output format: Structured text report."""
        }
    
    async def process_user_command(self, command: str, available_resources: Dict) -> Dict:
        """
        Process user command through all AI components with confirmations
        """
        
        # Step 1: Interpret the mission
        interpretation = await self._interpret_mission(command)
        
        # Step 2: Get confirmation on interpretation
        interpretation_confirmed = await self._get_user_confirmation(
            "mission_interpretation",
            interpretation,
            "I understood your command as follows. Is this correct?"
        )
        
        if not interpretation_confirmed:
            return {"status": "cancelled", "reason": "User rejected interpretation"}
        
        # Step 3: Analyze location
        location_analysis = await self._analyze_location(interpretation)
        
        # Step 4: Confirm location understanding
        location_confirmed = await self._get_user_confirmation(
            "location_analysis", 
            location_analysis,
            "I've identified the search area. Is this correct?"
        )
        
        if not location_confirmed:
            return {"status": "cancelled", "reason": "User rejected location"}
        
        # Step 5: Create mission plan
        mission_plan = await self._create_mission_plan(
            interpretation, 
            location_analysis,
            available_resources
        )
        
        # Step 6: Final mission confirmation
        plan_confirmed = await self._get_user_confirmation(
            "mission_plan",
            mission_plan,
            "Here's the complete mission plan. Approve to start?"
        )
        
        if not plan_confirmed:
            return {"status": "cancelled", "reason": "User rejected plan"}
        
        # Step 7: Initialize safety monitoring
        safety_status = await self._initialize_safety_monitor(mission_plan)
        
        return {
            "status": "approved",
            "interpretation": interpretation,
            "location": location_analysis,
            "plan": mission_plan,
            "safety": safety_status,
            "confirmation_id": f"MISSION_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
    
    async def _interpret_mission(self, command: str) -> Dict:
        """Use AI to interpret mission command"""
        
        prompt = f"{self.contexts[AIComponent.MISSION_INTERPRETER]}\n\nUser command: {command}\n\nInterpret this command:"
        
        # Query AI (with fallback)
        try:
            response = await self._query_ai(AIComponent.MISSION_INTERPRETER, prompt)
            if response:
                return response
        except Exception as e:
            logger.warning(f"AI interpretation failed: {e}")
        
        # Fallback interpretation
        return {
            "objective": "search_and_rescue",
            "location": "unspecified - requires clarification",
            "priority": "high",
            "constraints": [],
            "original_command": command,
            "ai_confidence": 0.5
        }
    
    async def _analyze_location(self, interpretation: Dict) -> Dict:
        """Analyze and convert location to coordinates"""
        
        location_desc = interpretation.get("location", "unspecified")
        prompt = f"{self.contexts[AIComponent.LOCATION_ANALYZER]}\n\nLocation description: {location_desc}\n\nConvert to coordinates:"
        
        try:
            response = await self._query_ai(AIComponent.LOCATION_ANALYZER, prompt)
            if response:
                return response
        except Exception as e:
            logger.warning(f"Location analysis failed: {e}")
        
        # Fallback - request manual input
        return {
            "needs_manual_input": True,
            "description": location_desc,
            "suggested_area": {
                "center": {"x": 0, "y": 50},
                "radius": 100
            }
        }
    
    async def _create_mission_plan(self, interpretation: Dict, location: Dict, resources: Dict) -> Dict:
        """Create detailed mission plan"""
        
        prompt = f"""{self.contexts[AIComponent.MISSION_PLANNER]}
        
        Mission objective: {interpretation['objective']}
        Location: {json.dumps(location)}
        Available resources: {json.dumps(resources)}
        
        Create detailed mission plan:"""
        
        try:
            response = await self._query_ai(AIComponent.MISSION_PLANNER, prompt)
            if response:
                return self._enhance_plan_with_safety(response)
        except Exception as e:
            logger.warning(f"Mission planning failed: {e}")
        
        # Fallback plan
        return self._create_fallback_plan(interpretation, location, resources)
    
    async def _initialize_safety_monitor(self, mission_plan: Dict) -> Dict:
        """Initialize safety monitoring for mission"""
        
        prompt = f"""{self.contexts[AIComponent.SAFETY_MONITOR]}
        
        Mission plan: {json.dumps(mission_plan)}
        
        Identify safety considerations:"""
        
        try:
            response = await self._query_ai(AIComponent.SAFETY_MONITOR, prompt)
            if response:
                return response
        except Exception as e:
            logger.warning(f"Safety analysis failed: {e}")
        
        # Default safety parameters
        return {
            "battery_return_threshold": 25,
            "no_fly_zones": [],
            "max_altitude": 30,
            "emergency_procedures": "active",
            "communication_check_interval": 30
        }
    
    async def _get_user_confirmation(self, confirmation_type: str, data: Dict, message: str) -> bool:
        """
        Get user confirmation for a decision
        This would be connected to the frontend
        """
        
        confirmation_id = f"{confirmation_type}_{datetime.now().timestamp()}"
        
        self.pending_confirmations[confirmation_id] = {
            "type": confirmation_type,
            "data": data,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "status": "pending"
        }
        
        # In real implementation, this would wait for user response
        # For now, we'll simulate automatic approval
        # The frontend would call confirm_decision(confirmation_id, approved=True/False)
        
        await asyncio.sleep(0.5)  # Simulate user thinking
        return True  # Simulate approval
    
    async def confirm_decision(self, confirmation_id: str, approved: bool, modifications: Optional[Dict] = None):
        """
        Process user confirmation decision
        Called by frontend when user approves/rejects
        """
        
        if confirmation_id not in self.pending_confirmations:
            return {"error": "Invalid confirmation ID"}
        
        confirmation = self.pending_confirmations[confirmation_id]
        confirmation["status"] = "approved" if approved else "rejected"
        confirmation["decided_at"] = datetime.now().isoformat()
        
        if modifications:
            confirmation["modifications"] = modifications
        
        return {"status": "confirmed", "confirmation_id": confirmation_id}
    
    async def _query_ai(self, component: AIComponent, prompt: str) -> Optional[Dict]:
        """Query specific AI model"""
        
        # Check if offline mode
        if not self._check_ai_available():
            return None
        
        # This would query Ollama
        # For now, return mock response
        return {
            "response": "AI response would go here",
            "confidence": 0.85,
            "model": self.models[component]
        }
    
    def _check_ai_available(self) -> bool:
        """Check if AI models are available"""
        # This would check Ollama connection
        return True
    
    def _enhance_plan_with_safety(self, plan: Dict) -> Dict:
        """Add safety enhancements to plan"""
        plan["safety_features"] = {
            "auto_return_low_battery": True,
            "collision_avoidance": True,
            "emergency_land_zones": [],
            "communication_redundancy": True
        }
        return plan
    
    def _create_fallback_plan(self, interpretation: Dict, location: Dict, resources: Dict) -> Dict:
        """Create basic plan when AI unavailable"""
        
        num_drones = len(resources.get("drones", []))
        
        return {
            "type": "basic_search",
            "pattern": "expanding_spiral",
            "drone_assignments": [
                {
                    "drone_id": d["id"],
                    "role": "searcher",
                    "area": f"sector_{i}"
                }
                for i, d in enumerate(resources.get("drones", []))
            ],
            "estimated_duration": 20,
            "coverage_area": 1000,
            "ai_generated": False
        }

# Global AI system instance
complete_ai_system = CompleteSARAI()