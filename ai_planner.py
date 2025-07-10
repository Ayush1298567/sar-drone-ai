"""
Advanced AI Mission Planner - Understands natural language and coordinates drone swarms
This AI analyzes missions and creates optimal plans for all available drones
"""

import json
import math
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class AIPlanner:
    """
    Advanced AI that understands missions and coordinates drone swarms
    """
    
    def __init__(self):
        # Mission understanding patterns
        self.mission_types = {
            "search_survivors": ["survivor", "people", "person", "victim", "trapped", "help", "rescue"],
            "explore_building": ["building", "structure", "room", "floor", "inside", "interior"],
            "assess_damage": ["damage", "collapsed", "structural", "unsafe", "hazard", "danger"],
            "perimeter_check": ["perimeter", "outside", "around", "exterior", "boundary"],
            "find_exit": ["exit", "way out", "escape", "route", "path"],
            "locate_hazards": ["fire", "gas", "chemical", "smoke", "danger", "hazard"]
        }
        
        # Building knowledge
        self.building_types = {
            "residential": {"avg_room_size": 20, "floors": 2, "layout": "grid"},
            "office": {"avg_room_size": 50, "floors": 5, "layout": "open"},
            "warehouse": {"avg_room_size": 500, "floors": 1, "layout": "open"},
            "unknown": {"avg_room_size": 30, "floors": 3, "layout": "mixed"}
        }
    
    def analyze_mission(self, user_command: str, available_drones: List[Dict], 
                       environment_data: Optional[Dict] = None) -> Dict:
        """
        Main AI function - analyzes mission and creates optimal plan
        
        Args:
            user_command: Natural language command from user
            available_drones: List of available drones with their capabilities
            environment_data: Any known information about the environment
            
        Returns:
            Complete mission plan with drone assignments
        """
        
        # Step 1: Understand the mission
        mission_analysis = self._understand_mission(user_command)
        logger.info(f"Mission understood as: {mission_analysis['type']}")
        
        # Step 2: Assess available resources
        resource_assessment = self._assess_resources(available_drones)
        logger.info(f"Resources: {resource_assessment['summary']}")
        
        # Step 3: Analyze environment
        environment = self._analyze_environment(environment_data, mission_analysis)
        logger.info(f"Environment: {environment['type']}")
        
        # Step 4: Create optimal strategy
        strategy = self._create_strategy(mission_analysis, resource_assessment, environment)
        logger.info(f"Strategy: {strategy['name']}")
        
        # Step 5: Assign drones to tasks
        assignments = self._assign_drones(strategy, available_drones)
        
        # Step 6: Generate coordinated flight plans
        flight_plans = self._generate_flight_plans(assignments, strategy, environment)
        
        # Create complete mission plan
        mission_plan = {
            "mission_id": f"AI_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "user_command": user_command,
            "ai_analysis": {
                "mission_type": mission_analysis['type'],
                "priority": mission_analysis['priority'],
                "estimated_duration": strategy['estimated_duration'],
                "confidence": mission_analysis['confidence']
            },
            "strategy": {
                "name": strategy['name'],
                "description": strategy['description'],
                "approach": strategy['approach']
            },
            "drone_assignments": assignments,
            "coordination": {
                "formation": strategy.get('formation', 'distributed'),
                "communication": "mesh_network",
                "failover": "automatic_reassignment"
            },
            "flight_plans": flight_plans,
            "safety_parameters": {
                "min_battery_return": 25,
                "max_altitude": 30,
                "collision_avoidance": True,
                "emergency_protocols": True
            },
            "success_criteria": self._define_success_criteria(mission_analysis)
        }
        
        return mission_plan
    
    def _understand_mission(self, command: str) -> Dict:
        """
        Use AI to understand what the user wants
        """
        command_lower = command.lower()
        
        # Determine mission type
        mission_type = "general_search"
        confidence = 0.5
        priority = "normal"
        
        # Check for specific mission types
        for m_type, keywords in self.mission_types.items():
            if any(keyword in command_lower for keyword in keywords):
                mission_type = m_type
                confidence = 0.9
                break
        
        # Determine urgency
        if any(word in command_lower for word in ["urgent", "emergency", "quickly", "asap", "now"]):
            priority = "urgent"
        elif any(word in command_lower for word in ["careful", "slowly", "thoroughly"]):
            priority = "thorough"
            
        # Extract specific requirements
        requirements = []
        if "survivor" in command_lower or "people" in command_lower:
            requirements.append("thermal_detection")
            requirements.append("audio_detection")
        if "damage" in command_lower:
            requirements.append("structural_analysis")
        if "map" in command_lower or "layout" in command_lower:
            requirements.append("3d_mapping")
            
        return {
            "type": mission_type,
            "priority": priority,
            "confidence": confidence,
            "requirements": requirements,
            "original_command": command
        }
    
    def _assess_resources(self, drones: List[Dict]) -> Dict:
        """
        Assess available drone capabilities
        """
        total_drones = len(drones)
        
        # Categorize drones by capabilities
        capabilities = {
            "thermal_capable": 0,
            "high_battery": 0,
            "camera_equipped": 0,
            "long_range": 0
        }
        
        for drone in drones:
            if drone.get('battery', 0) > 80:
                capabilities['high_battery'] += 1
            if drone.get('thermal_camera', False):
                capabilities['thermal_capable'] += 1
            if drone.get('camera', False):
                capabilities['camera_equipped'] += 1
            if drone.get('range', 0) > 1000:
                capabilities['long_range'] += 1
                
        return {
            "total_drones": total_drones,
            "capabilities": capabilities,
            "summary": f"{total_drones} drones available, {capabilities['high_battery']} with high battery",
            "limitations": self._identify_limitations(capabilities, total_drones)
        }
    
    def _analyze_environment(self, env_data: Optional[Dict], mission: Dict) -> Dict:
        """
        Analyze the environment for mission planning
        """
        if not env_data:
            # Make intelligent assumptions based on mission
            if "building" in mission.get('original_command', '').lower():
                return {
                    "type": "building_interior",
                    "estimated_size": {"x": 50, "y": 50, "z": 15},
                    "complexity": "high",
                    "hazards": ["confined_spaces", "possible_obstacles"],
                    "building_type": "unknown"
                }
            else:
                return {
                    "type": "open_area",
                    "estimated_size": {"x": 100, "y": 100, "z": 50},
                    "complexity": "medium",
                    "hazards": ["none_identified"]
                }
                
        return env_data
    
    def _create_strategy(self, mission: Dict, resources: Dict, environment: Dict) -> Dict:
        """
        Create optimal strategy based on mission, resources, and environment
        """
        strategies = {
            "search_survivors": self._survivor_search_strategy,
            "explore_building": self._building_exploration_strategy,
            "assess_damage": self._damage_assessment_strategy,
            "perimeter_check": self._perimeter_check_strategy,
            "find_exit": self._exit_finding_strategy,
            "locate_hazards": self._hazard_location_strategy
        }
        
        # Get specific strategy or use default
        strategy_func = strategies.get(mission['type'], self._default_search_strategy)
        return strategy_func(mission, resources, environment)
    
    def _survivor_search_strategy(self, mission: Dict, resources: Dict, environment: Dict) -> Dict:
        """
        Strategy for searching for survivors
        """
        num_drones = resources['total_drones']
        
        if environment['type'] == 'building_interior':
            if num_drones >= 6:
                return {
                    "name": "Multi-Floor Parallel Search",
                    "description": "Divide drones across floors for faster coverage",
                    "approach": "parallel_floors",
                    "formation": "distributed",
                    "drone_roles": {
                        "scouts": 2,  # Quick overview
                        "searchers": 3,  # Detailed search
                        "relay": 1  # Communication relay
                    },
                    "estimated_duration": 15,  # minutes
                    "pattern": "room_by_room"
                }
            else:
                return {
                    "name": "Sequential Floor Search",
                    "description": "Search floor by floor with all drones",
                    "approach": "sequential",
                    "formation": "line",
                    "drone_roles": {
                        "lead": 1,
                        "searchers": num_drones - 1
                    },
                    "estimated_duration": 25,
                    "pattern": "systematic_sweep"
                }
        else:
            return {
                "name": "Grid Pattern Search",
                "description": "Systematic grid search of area",
                "approach": "grid",
                "formation": "line_abreast",
                "drone_roles": {
                    "searchers": num_drones
                },
                "estimated_duration": 20,
                "pattern": "parallel_lanes"
            }
    
    def _building_exploration_strategy(self, mission: Dict, resources: Dict, environment: Dict) -> Dict:
        """
        Strategy for exploring buildings
        """
        return {
            "name": "Systematic Building Exploration",
            "description": "Map building layout while searching",
            "approach": "explore_and_map",
            "formation": "adaptive",
            "drone_roles": {
                "mapper": 1,  # Creates 3D map
                "explorers": resources['total_drones'] - 1
            },
            "estimated_duration": 20,
            "pattern": "expanding_frontier"
        }

    def _damage_assessment_strategy(self, mission: Dict, resources: Dict, environment: Dict) -> Dict:
        """
        Strategy for assessing structural damage
        """
        return {
            "name": "Structural Damage Assessment",
            "description": "Systematically document damage and identify unsafe areas",
            "approach": "systematic_scan",
            "formation": "spread",
            "drone_roles": {
                "inspectors": resources['total_drones']
            },
            "estimated_duration": 20,
            "pattern": "grid_with_photos"
        }

    def _perimeter_check_strategy(self, mission: Dict, resources: Dict, environment: Dict) -> Dict:
        """
        Strategy for checking building perimeter
        """
        return {
            "name": "Perimeter Security Check",
            "description": "Circle the building exterior checking for hazards and entry points",
            "approach": "circular",
            "formation": "single_file",
            "drone_roles": {
                "perimeter": resources['total_drones']
            },
            "estimated_duration": 15,
            "pattern": "clockwise_circle"
        }

    def _exit_finding_strategy(self, mission: Dict, resources: Dict, environment: Dict) -> Dict:
        """
        Strategy for finding safe exit routes
        """
        return {
            "name": "Safe Exit Route Mapping",
            "description": "Find and verify all possible exit routes",
            "approach": "path_finding",
            "formation": "scout_and_follow",
            "drone_roles": {
                "pathfinder": 1,
                "verifiers": resources['total_drones'] - 1
            },
            "estimated_duration": 15,
            "pattern": "explore_paths"
        }

    def _hazard_location_strategy(self, mission: Dict, resources: Dict, environment: Dict) -> Dict:
        """
        Strategy for locating hazards
        """
        return {
            "name": "Hazard Detection Sweep",
            "description": "Identify fire, gas, structural hazards",
            "approach": "careful_sweep",
            "formation": "spread",
            "drone_roles": {
                "hazard_detectors": resources['total_drones']
            },
            "estimated_duration": 25,
            "pattern": "slow_grid"
        }
    
    def _default_search_strategy(self, mission: Dict, resources: Dict, environment: Dict) -> Dict:
        """
        Default strategy when specific type not recognized
        """
        return {
            "name": "Adaptive Search",
            "description": "Flexible search pattern based on discoveries",
            "approach": "adaptive",
            "formation": "dynamic",
            "drone_roles": {
                "adaptive": resources['total_drones']
            },
            "estimated_duration": 20,
            "pattern": "intelligent_coverage"
        }
    
    def _assign_drones(self, strategy: Dict, drones: List[Dict]) -> List[Dict]:
        """
        Assign specific drones to specific roles
        """
        assignments = []
        drone_roles = strategy.get('drone_roles', {})
        
        # Sort drones by battery for optimal assignment
        sorted_drones = sorted(drones, key=lambda d: d.get('battery', 100), reverse=True)
        
        drone_index = 0
        for role, count in drone_roles.items():
            for i in range(count):
                if drone_index < len(sorted_drones):
                    drone = sorted_drones[drone_index]
                    assignments.append({
                        "drone_id": drone['id'],
                        "drone_name": drone.get('name', f"Drone {drone['id']}"),
                        "role": role,
                        "primary_task": self._get_role_task(role),
                        "battery_level": drone.get('battery', 100),
                        "capabilities": self._get_drone_capabilities(drone)
                    })
                    drone_index += 1
                    
        return assignments
    
    def _generate_flight_plans(self, assignments: List[Dict], strategy: Dict, 
                              environment: Dict) -> Dict:
        """
        Generate coordinated flight plans for all drones
        """
        pattern = strategy.get('pattern', 'grid')
        approach = strategy.get('approach', 'systematic')
        
        flight_plans = {}
        
        for assignment in assignments:
            drone_id = assignment['drone_id']
            role = assignment['role']
            
            # Generate role-specific flight plan
            if role == 'scouts':
                plan = self._generate_scout_plan(environment)
            elif role == 'searchers':
                plan = self._generate_search_plan(pattern, environment, len(assignments))
            elif role == 'relay':
                plan = self._generate_relay_plan(environment)
            elif role == 'mapper':
                plan = self._generate_mapping_plan(environment)
            else:
                plan = self._generate_default_plan(pattern, environment)
                
            flight_plans[drone_id] = {
                "drone_id": drone_id,
                "role": role,
                "waypoints": plan['waypoints'],
                "behavior": plan['behavior'],
                "priority_actions": plan.get('priorities', []),
                "communication": {
                    "report_interval": 5,  # seconds
                    "priority_events": ["human_detection", "hazard_found", "path_blocked"]
                }
            }
            
        return flight_plans
    
    def _generate_scout_plan(self, environment: Dict) -> Dict:
        """
        Generate flight plan for scout drones
        """
        size = environment.get('estimated_size', {'x': 100, 'y': 100, 'z': 30})
        
        # Quick perimeter check first
        waypoints = [
            {"x": 0, "y": 0, "z": 20, "action": "scan_area"},
            {"x": size['x']/2, "y": size['y']/2, "z": 25, "action": "quick_scan"},
            {"x": -size['x']/2, "y": size['y']/2, "z": 25, "action": "quick_scan"},
            {"x": -size['x']/2, "y": -size['y']/2, "z": 25, "action": "quick_scan"},
            {"x": size['x']/2, "y": -size['y']/2, "z": 25, "action": "quick_scan"},
        ]
        
        return {
            "waypoints": waypoints,
            "behavior": "fast_reconnaissance",
            "priorities": ["identify_layout", "find_entry_points", "detect_hazards"]
        }
    
    def _generate_search_plan(self, pattern: str, environment: Dict, num_searchers: int) -> Dict:
        """
        Generate search pattern for searcher drones
        """
        size = environment.get('estimated_size', {'x': 100, 'y': 100, 'z': 30})
        waypoints = []
        
        if pattern == "room_by_room":
            # Systematic room search
            rooms_per_floor = 10  # Estimate
            for room in range(rooms_per_floor):
                x = (room % 5) * 10 - 20
                y = (room // 5) * 10 - 10
                waypoints.append({"x": x, "y": y, "z": 3, "action": "detailed_scan", "hover_time": 10})
                
        elif pattern == "parallel_lanes":
            # Parallel search lanes
            lane_spacing = size['x'] / (num_searchers + 1)
            for i in range(10):  # 10 passes
                y = -size['y']/2 + i * 10
                waypoints.append({"x": 0, "y": y, "z": 10, "action": "scan_lane"})
                
        else:
            # Default grid
            for x in range(-50, 51, 10):
                for y in range(-50, 51, 10):
                    waypoints.append({"x": x, "y": y, "z": 15, "action": "scan_point"})
                    
        return {
            "waypoints": waypoints,
            "behavior": "thorough_search",
            "priorities": ["detect_humans", "avoid_obstacles", "maintain_coverage"]
        }
    
    def _generate_relay_plan(self, environment: Dict) -> Dict:
        """
        Generate plan for communication relay drone
        """
        # Position for optimal communication
        return {
            "waypoints": [
                {"x": 0, "y": 0, "z": 30, "action": "maintain_position", "hover_time": -1}
            ],
            "behavior": "communication_relay",
            "priorities": ["maintain_link", "adjust_position", "monitor_battery"]
        }
    
    def _generate_mapping_plan(self, environment: Dict) -> Dict:
        """
        Generate mapping plan
        """
        waypoints = []
        size = environment.get('estimated_size', {'x': 100, 'y': 100, 'z': 30})
        
        # Systematic coverage for mapping
        for altitude in [5, 15, 25]:
            for x in range(int(-size['x']/2), int(size['x']/2) + 1, 20):
                for y in range(int(-size['y']/2), int(size['y']/2) + 1, 20):
                    waypoints.append({
                        "x": x, "y": y, "z": altitude, 
                        "action": "3d_scan",
                        "hover_time": 5
                    })
                    
        return {
            "waypoints": waypoints,
            "behavior": "systematic_mapping",
            "priorities": ["complete_coverage", "high_detail", "overlap_scans"]
        }
    
    def _generate_default_plan(self, pattern: str, environment: Dict) -> Dict:
        """
        Generate default flight plan
        """
        return {
            "waypoints": [
                {"x": 0, "y": 0, "z": 15, "action": "hover"},
                {"x": 10, "y": 10, "z": 15, "action": "scan"},
                {"x": -10, "y": 10, "z": 15, "action": "scan"},
                {"x": -10, "y": -10, "z": 15, "action": "scan"},
                {"x": 10, "y": -10, "z": 15, "action": "scan"},
            ],
            "behavior": "adaptive",
            "priorities": ["explore", "detect", "report"]
        }
    
    def _get_role_task(self, role: str) -> str:
        """
        Get primary task for role
        """
        role_tasks = {
            "scouts": "Quick reconnaissance and hazard identification",
            "searchers": "Detailed search for survivors",
            "relay": "Maintain communication link",
            "mapper": "Create 3D map of environment",
            "lead": "Coordinate team and make decisions",
            "explorers": "Explore unknown areas"
        }
        return role_tasks.get(role, "General search and exploration")
    
    def _get_drone_capabilities(self, drone: Dict) -> List[str]:
        """
        List drone capabilities
        """
        capabilities = []
        if drone.get('thermal_camera'):
            capabilities.append("thermal_imaging")
        if drone.get('camera'):
            capabilities.append("visual_imaging")
        if drone.get('lidar'):
            capabilities.append("3d_mapping")
        if drone.get('audio'):
            capabilities.append("audio_detection")
        return capabilities
    
    def _identify_limitations(self, capabilities: Dict, total_drones: int) -> List[str]:
        """
        Identify resource limitations
        """
        limitations = []
        if capabilities['thermal_capable'] < total_drones / 2:
            limitations.append("Limited thermal imaging capability")
        if capabilities['high_battery'] < total_drones / 2:
            limitations.append("Some drones have low battery")
        return limitations
    
    def _define_success_criteria(self, mission: Dict) -> List[str]:
        """
        Define what success looks like for this mission
        """
        criteria = []
        
        if mission['type'] == 'search_survivors':
            criteria.extend([
                "All accessible areas searched",
                "All heat signatures investigated",
                "No areas left unchecked"
            ])
        elif mission['type'] == 'explore_building':
            criteria.extend([
                "Complete map generated",
                "All rooms identified",
                "Safe paths marked"
            ])
        else:
            criteria.extend([
                "Mission area fully covered",
                "All objectives completed",
                "All drones returned safely"
            ])
            
        return criteria

# Global AI planner instance
ai_planner = AIPlanner()