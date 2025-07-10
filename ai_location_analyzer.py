"""
AI Location Analyzer V2 - Map-based with pin dropping and self-learning
Integrates with visual map interface for intuitive location selection
"""

import json
import logging
import math
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import sqlite3
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import requests

logger = logging.getLogger(__name__)

@dataclass
class MapPin:
    """Represents a user-placed pin on the map"""
    x: float
    y: float
    z: float = 0.0
    label: Optional[str] = None
    radius: float = 50.0  # Search radius in meters
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class MapBasedLocationAI:
    """
    AI system that works with map pins and learns area characteristics
    """
    
    def __init__(self, model_name: str = "mistral:7b-instruct"):
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434"
        
        # Initialize learning database
        self.learning_db_path = Path("data/map_location_learning.db")
        self.learning_db_path.parent.mkdir(exist_ok=True)
        self._init_learning_db()
        
        # Area knowledge that grows over time
        self.area_knowledge = self._load_area_knowledge()
        
        # Map configuration
        self.map_center = {"x": 0, "y": 0}  # Base station location
        self.map_scale = 1.0  # meters per pixel
        self.explored_areas = []
        
        # Learning parameters
        self.area_classification_threshold = 0.75
        self.environment_learning_rate = 0.2
        
    def _init_learning_db(self):
        """Initialize database for map-based learning"""
        
        conn = sqlite3.connect(self.learning_db_path)
        cursor = conn.cursor()
        
        # Pin history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pin_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                x REAL NOT NULL,
                y REAL NOT NULL,
                z REAL DEFAULT 0,
                user_description TEXT,
                ai_analysis TEXT,
                area_type TEXT,
                mission_type TEXT,
                mission_outcome TEXT,
                success_score REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Area characteristics table (learns what different areas are like)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS area_characteristics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                center_x REAL NOT NULL,
                center_y REAL NOT NULL,
                radius REAL NOT NULL,
                area_name TEXT,
                area_type TEXT,  -- building, open_area, debris_field, etc.
                characteristics TEXT,  -- JSON of learned features
                hazards TEXT,  -- JSON of known hazards
                accessibility_score REAL,
                last_explored DATETIME,
                exploration_count INTEGER DEFAULT 1,
                confidence REAL DEFAULT 0.5
            )
        ''')
        
        # Environmental patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS environmental_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_name TEXT NOT NULL,
                pattern_data TEXT NOT NULL,  -- JSON
                occurrence_locations TEXT,  -- JSON array of x,y coordinates
                reliability_score REAL,
                last_observed DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Mission success correlation table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mission_correlations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                area_type TEXT NOT NULL,
                mission_type TEXT NOT NULL,
                success_rate REAL,
                optimal_parameters TEXT,  -- JSON of best practices
                sample_size INTEGER,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_area_knowledge(self) -> Dict:
        """Load learned knowledge about different areas"""
        
        knowledge = {
            "known_areas": {},
            "area_types": {},
            "environmental_patterns": [],
            "mission_correlations": {}
        }
        
        try:
            conn = sqlite3.connect(self.learning_db_path)
            cursor = conn.cursor()
            
            # Load known areas
            cursor.execute('''
                SELECT center_x, center_y, radius, area_name, area_type, 
                       characteristics, hazards, accessibility_score, confidence
                FROM area_characteristics
                WHERE confidence > 0.6
                ORDER BY exploration_count DESC
            ''')
            
            for row in cursor.fetchall():
                x, y, radius, name, area_type, chars, hazards, access, conf = row
                
                area_key = f"{int(x)}_{int(y)}"
                knowledge["known_areas"][area_key] = {
                    "center": {"x": x, "y": y},
                    "radius": radius,
                    "name": name,
                    "type": area_type,
                    "characteristics": json.loads(chars) if chars else {},
                    "hazards": json.loads(hazards) if hazards else [],
                    "accessibility": access,
                    "confidence": conf
                }
            
            # Load environmental patterns
            cursor.execute('''
                SELECT pattern_name, pattern_data, occurrence_locations, reliability_score
                FROM environmental_patterns
                WHERE reliability_score > 0.7
                ORDER BY reliability_score DESC
            ''')
            
            for name, data, locations, reliability in cursor.fetchall():
                knowledge["environmental_patterns"].append({
                    "name": name,
                    "pattern": json.loads(data),
                    "locations": json.loads(locations) if locations else [],
                    "reliability": reliability
                })
            
            # Load mission correlations
            cursor.execute('''
                SELECT area_type, mission_type, success_rate, optimal_parameters
                FROM mission_correlations
                WHERE sample_size > 5
            ''')
            
            for area_type, mission_type, success_rate, params in cursor.fetchall():
                if area_type not in knowledge["mission_correlations"]:
                    knowledge["mission_correlations"][area_type] = {}
                
                knowledge["mission_correlations"][area_type][mission_type] = {
                    "success_rate": success_rate,
                    "optimal_parameters": json.loads(params) if params else {}
                }
            
            conn.close()
            
        except Exception as e:
            logger.warning(f"Could not load area knowledge: {e}")
        
        return knowledge
    
    def analyze_pin_location(self, pin: MapPin, user_description: str, 
                           map_context: Optional[Dict] = None) -> Dict:
        """
        Analyze a user-placed pin with their description
        
        Args:
            pin: MapPin object with coordinates
            user_description: What the user said about this location
            map_context: Additional map information (zoom level, visible features, etc.)
        
        Returns:
            Comprehensive analysis of the pinned location
        """
        
        # Record the pin
        pin_id = self._record_pin(pin, user_description)
        
        try:
            # Step 1: Check if we know this area from previous missions
            area_knowledge = self._get_area_knowledge(pin.x, pin.y, pin.radius)
            
            # Step 2: Use AI to understand user intent with area context
            ai_analysis = self._ai_analyze_pin_intent(pin, user_description, area_knowledge, map_context)
            
            # Step 3: Correlate with learned patterns
            enhanced_analysis = self._enhance_with_patterns(ai_analysis, pin, area_knowledge)
            
            # Step 4: Generate area-specific recommendations
            recommendations = self._generate_area_recommendations(enhanced_analysis, area_knowledge)
            
            # Step 5: Build complete analysis
            complete_analysis = {
                "pin_id": pin_id,
                "location": {
                    "coordinates": {"x": pin.x, "y": pin.y, "z": pin.z},
                    "radius": pin.radius,
                    "label": pin.label
                },
                "user_intent": ai_analysis.get("interpreted_intent", {}),
                "area_analysis": {
                    "type": area_knowledge.get("type", "unknown"),
                    "name": area_knowledge.get("name", "Unexplored area"),
                    "characteristics": area_knowledge.get("characteristics", {}),
                    "known_hazards": area_knowledge.get("hazards", []),
                    "accessibility": area_knowledge.get("accessibility", 0.5),
                    "exploration_history": area_knowledge.get("exploration_count", 0)
                },
                "recommendations": recommendations,
                "confidence": enhanced_analysis.get("confidence", 0.7),
                "learning_applied": enhanced_analysis.get("learning_notes", []),
                "clarifications_needed": ai_analysis.get("clarifications", [])
            }
            
            # Record for learning
            self._update_pin_analysis(pin_id, complete_analysis)
            
            return complete_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze pin location: {e}")
            return self._fallback_pin_analysis(pin, user_description)
    
    def _record_pin(self, pin: MapPin, description: str) -> int:
        """Record a new pin placement"""
        
        conn = sqlite3.connect(self.learning_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO pin_history (x, y, z, user_description)
            VALUES (?, ?, ?, ?)
        ''', (pin.x, pin.y, pin.z, description))
        
        pin_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return pin_id if pin_id is not None else 0
    
    def _update_pin_analysis(self, pin_id: int, analysis: dict):
        """Update the pin_history entry with the AI's analysis."""
        conn = sqlite3.connect(self.learning_db_path)
        cursor = conn.cursor()
        cursor.execute(
            '''
            UPDATE pin_history
            SET ai_analysis = ?, area_type = ?, mission_type = ?
            WHERE id = ?
            ''',
            (
                json.dumps(analysis),
                analysis.get("area_analysis", {}).get("type", "unknown"),
                analysis.get("user_intent", {}).get("mission_type", "unknown"),
                pin_id
            )
        )
        conn.commit()
        conn.close()
    
    def _get_area_knowledge(self, x: float, y: float, radius: float) -> Dict:
        """Get what we know about this area from learning"""
        
        # Check if this area overlaps with known areas
        for area_key, area_info in self.area_knowledge["known_areas"].items():
            center = area_info["center"]
            area_radius = area_info["radius"]
            
            # Calculate distance between centers
            distance = math.sqrt((x - center["x"])**2 + (y - center["y"])**2)
            
            # Check for overlap
            if distance < (radius + area_radius):
                logger.info(f"Found knowledge about area: {area_info.get('name', 'Unknown')}")
                return area_info
        
        # Check for nearby explored areas
        nearby_areas = self._find_nearby_areas(x, y, radius * 2)
        if nearby_areas:
            # Synthesize knowledge from nearby areas
            return self._synthesize_area_knowledge(nearby_areas, x, y)
        
        # No knowledge - return unknown
        return {
            "type": "unknown",
            "characteristics": {},
            "hazards": [],
            "accessibility": 0.5,
            "exploration_count": 0,
            "confidence": 0.1
        }
    
    def _ai_analyze_pin_intent(self, pin: MapPin, description: str, 
                               area_knowledge: Dict, map_context: Optional[Dict]) -> Dict:
        """Use Ollama to understand what the user wants at this location"""
        
        # Build context for the AI
        prompt = f"""You are an AI assistant for a Search and Rescue drone system analyzing a map location.

The user has placed a pin on the map at coordinates ({pin.x}, {pin.y}) with a search radius of {pin.radius} meters.

User's description: "{description}"

What we know about this area:
- Area type: {area_knowledge.get('type', 'unknown')}
- Previous explorations: {area_knowledge.get('exploration_count', 0)}
- Known characteristics: {json.dumps(area_knowledge.get('characteristics', {}))}
- Known hazards: {json.dumps(area_knowledge.get('hazards', []))}
- Accessibility score: {area_knowledge.get('accessibility', 'unknown')}

Your task is to understand:
1. What the user wants to do at this location
2. What type of mission this is
3. What specific concerns or objectives they have
4. What additional information would be helpful

Consider the map context and area knowledge when interpreting the user's intent.

Output as JSON:
{{
    "interpreted_intent": {{
        "primary_objective": "what they want to accomplish",
        "mission_type": "search_rescue/exploration/assessment/etc",
        "specific_targets": ["list of what to look for"],
        "constraints": ["any limitations mentioned"],
        "urgency": "critical/high/normal/low"
    }},
    "area_considerations": {{
        "relevant_characteristics": ["which area features matter for this mission"],
        "potential_challenges": ["based on area knowledge"],
        "recommended_approach": "how to best accomplish the objective here"
    }},
    "clarifications": ["list of things that need clarification"],
    "confidence": 0.0-1.0
}}"""
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "temperature": 0.3
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return json.loads(result['response'])
            else:
                raise Exception(f"AI analysis failed: {response.status_code}")
                
        except Exception as e:
            logger.warning(f"AI pin analysis failed: {e}")
            return self._rule_based_intent_analysis(description, area_knowledge)
    
    def _enhance_with_patterns(self, ai_analysis: Dict, pin: MapPin, area_knowledge: Dict) -> Dict:
        """Enhance analysis with learned environmental patterns"""
        
        # Check if any known patterns apply to this location
        applicable_patterns = []
        
        for pattern in self.area_knowledge["environmental_patterns"]:
            # Check if this pattern has been observed near this location
            for loc in pattern["locations"]:
                distance = math.sqrt((pin.x - loc["x"])**2 + (pin.y - loc["y"])**2)
                if distance < pin.radius * 1.5:
                    applicable_patterns.append(pattern)
                    break
        
        if applicable_patterns:
            ai_analysis["environmental_patterns"] = [
                {
                    "name": p["name"],
                    "relevance": p["reliability"],
                    "implications": self._get_pattern_implications(p, ai_analysis)
                }
                for p in applicable_patterns
            ]
            
            # Adjust confidence based on pattern reliability
            pattern_confidence = np.mean([p["reliability"] for p in applicable_patterns])
            ai_analysis["confidence"] = ai_analysis.get("confidence", 0.5) * 0.7 + pattern_confidence * 0.3
        
        # Add learning notes
        ai_analysis["learning_notes"] = ai_analysis.get("learning_notes", [])
        ai_analysis["learning_notes"].append(f"Applied {len(applicable_patterns)} environmental patterns")
        
        return ai_analysis
    
    def _get_pattern_implications(self, pattern: Dict, analysis: Dict) -> List[str]:
        """Get implications of an environmental pattern for the current analysis"""
        
        implications = []
        pattern_type = pattern.get("type", "")
        analysis_intent = analysis.get("interpreted_intent", {}).get("mission_type", "")
        
        if pattern_type == "structural_instability":
            implications.extend([
                "Increased risk of building collapse",
                "Requires cautious approach",
                "May need structural assessment first"
            ])
        elif pattern_type == "fire_damage":
            implications.extend([
                "Potential for re-ignition",
                "Structural integrity compromised",
                "Thermal imaging recommended"
            ])
        elif pattern_type == "water_damage":
            implications.extend([
                "Electrical hazards present",
                "Mold and contamination risks",
                "Equipment waterproofing required"
            ])
        elif pattern_type == "debris_pattern":
            implications.extend([
                "Systematic search pattern needed",
                "Multiple passes may be required",
                "Consider debris removal strategy"
            ])
        
        # Add mission-specific implications
        if analysis_intent == "rescue" and pattern_type in ["structural_instability", "fire_damage"]:
            implications.append("High priority - potential life safety issue")
        elif analysis_intent == "assessment" and pattern_type == "debris_pattern":
            implications.append("Detailed mapping required for clearance planning")
        
        return implications
    
    def _learn_environmental_patterns(self, x: float, y: float, patterns: List[Dict]):
        """Learn new environmental patterns from mission discoveries"""
        
        conn = sqlite3.connect(self.learning_db_path)
        cursor = conn.cursor()
        
        for pattern in patterns:
            cursor.execute('''
                INSERT OR REPLACE INTO environmental_patterns
                (pattern_type, center_x, center_y, radius, reliability, 
                 characteristics, locations, discovery_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, 
                    COALESCE((SELECT discovery_count FROM environmental_patterns 
                              WHERE pattern_type = ? AND ABS(center_x - ?) < ? AND ABS(center_y - ?) < ?) + 1, 1))
            ''', (pattern.get("type", "unknown"), x, y, pattern.get("radius", 20),
                  pattern.get("reliability", 0.7), json.dumps(pattern.get("characteristics", {})),
                  json.dumps([{"x": x, "y": y}]), pattern.get("type", "unknown"), x, 10, y, 10))
        
        conn.commit()
        conn.close()
    
    def _generate_area_recommendations(self, analysis: Dict, area_knowledge: Dict) -> Dict:
        """Generate specific recommendations based on area knowledge"""
        
        recommendations = {
            "search_pattern": "spiral",  # default
            "drone_allocation": {},
            "special_equipment": [],
            "safety_precautions": [],
            "estimated_duration": 20,  # minutes
            "priority_zones": []
        }
        
        area_type = area_knowledge.get("type", "unknown")
        mission_type = analysis.get("interpreted_intent", {}).get("mission_type", "exploration")
        
        # Check mission success correlations
        if area_type in self.area_knowledge["mission_correlations"]:
            if mission_type in self.area_knowledge["mission_correlations"][area_type]:
                correlation = self.area_knowledge["mission_correlations"][area_type][mission_type]
                
                # Use optimal parameters from past successes
                optimal = correlation.get("optimal_parameters", {})
                recommendations["search_pattern"] = optimal.get("pattern", "spiral")
                recommendations["estimated_duration"] = optimal.get("duration", 20)
                
                # Add confidence-based adjustments
                if correlation["success_rate"] < 0.5:
                    recommendations["safety_precautions"].append(
                        f"Low success rate ({correlation['success_rate']:.0%}) in this area type"
                    )
        
        # Area-specific recommendations
        if area_type == "building":
            recommendations["search_pattern"] = "floor_by_floor"
            recommendations["special_equipment"].append("structural_sensors")
            recommendations["priority_zones"] = area_knowledge.get("characteristics", {}).get("entry_points", [])
            
        elif area_type == "debris_field":
            recommendations["search_pattern"] = "grid"
            recommendations["safety_precautions"].append("unstable_terrain")
            recommendations["drone_allocation"]["scouts"] = 1
            recommendations["drone_allocation"]["searchers"] = 2
            
        elif area_type == "open_area":
            recommendations["search_pattern"] = "expanding_spiral"
            recommendations["estimated_duration"] = 15
        
        # Hazard-based precautions
        for hazard in area_knowledge.get("hazards", []):
            if hazard["type"] == "structural_instability":
                recommendations["safety_precautions"].append("maintain_safe_altitude")
            elif hazard["type"] == "fire":
                recommendations["special_equipment"].append("thermal_resistant")
            elif hazard["type"] == "water":
                recommendations["safety_precautions"].append("waterproof_equipment_required")
        
        # Accessibility adjustments
        accessibility = area_knowledge.get("accessibility", 0.5)
        if accessibility < 0.3:
            recommendations["search_pattern"] = "cautious_exploration"
            recommendations["estimated_duration"] *= 1.5
        
        return recommendations
    
    def learn_from_mission_outcome(self, pin_id: int, mission_outcome: Dict):
        """Learn from the results of a mission at this pin location"""
        
        try:
            conn = sqlite3.connect(self.learning_db_path)
            cursor = conn.cursor()
            
            # Get pin details
            cursor.execute('SELECT x, y, z, area_type FROM pin_history WHERE id = ?', (pin_id,))
            result = cursor.fetchone()
            
            if result:
                x, y, z, area_type = result
                
                # Update pin history with outcome
                cursor.execute('''
                    UPDATE pin_history
                    SET mission_outcome = ?, success_score = ?
                    WHERE id = ?
                ''', (json.dumps(mission_outcome), mission_outcome.get("success_score", 0.5), pin_id))
                
                # Learn area characteristics
                self._learn_area_characteristics(x, y, mission_outcome)
                
                # Update mission correlations
                self._update_mission_correlations(area_type, mission_outcome)
                
                # Learn new environmental patterns
                if "discovered_patterns" in mission_outcome:
                    self._learn_environmental_patterns(x, y, mission_outcome["discovered_patterns"])
            
            conn.commit()
            conn.close()
            
            # Reload knowledge
            self.area_knowledge = self._load_area_knowledge()
            
        except Exception as e:
            logger.error(f"Failed to learn from mission outcome: {e}")
    
    def _learn_area_characteristics(self, x: float, y: float, outcome: Dict):
        """Learn characteristics of an area from mission results"""
        
        conn = sqlite3.connect(self.learning_db_path)
        cursor = conn.cursor()
        
        # Check if we already have info about this area
        cursor.execute('''
            SELECT id, characteristics, exploration_count, confidence
            FROM area_characteristics
            WHERE ABS(center_x - ?) < ? AND ABS(center_y - ?) < ?
            LIMIT 1
        ''', (x, 50, y, 50))  # Within 50 meters
        
        existing = cursor.fetchone()
        
        if existing:
            # Update existing area knowledge
            area_id, old_chars, count, old_confidence = existing
            
            # Merge characteristics
            old_chars_dict = json.loads(old_chars) if old_chars else {}
            new_chars = outcome.get("area_characteristics", {})
            
            # Weight new information based on success
            weight = outcome.get("success_score", 0.5) * self.environment_learning_rate
            
            # Update characteristics
            for key, value in new_chars.items():
                if key in old_chars_dict:
                    # Weighted average for numeric values
                    if isinstance(value, (int, float)):
                        old_chars_dict[key] = old_chars_dict[key] * (1 - weight) + value * weight
                    else:
                        old_chars_dict[key] = value
                else:
                    old_chars_dict[key] = value
            
            # Update confidence
            new_confidence = min(0.95, old_confidence + weight * 0.1)
            
            cursor.execute('''
                UPDATE area_characteristics
                SET characteristics = ?, exploration_count = ?, confidence = ?,
                    last_explored = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (json.dumps(old_chars_dict), count + 1, new_confidence, area_id))
            
        else:
            # Create new area record
            cursor.execute('''
                INSERT INTO area_characteristics
                (center_x, center_y, radius, area_name, area_type, characteristics, 
                 hazards, accessibility_score, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (x, y, outcome.get("search_radius", 50),
                  outcome.get("area_name", f"Area_{int(x)}_{int(y)}"),
                  outcome.get("area_type", "unknown"),
                  json.dumps(outcome.get("area_characteristics", {})),
                  json.dumps(outcome.get("hazards_found", [])),
                  outcome.get("accessibility_score", 0.5),
                  outcome.get("success_score", 0.5)))
        
        conn.commit()
        conn.close()
    
    def _update_mission_correlations(self, area_type: str, outcome: Dict):
        """Update success correlations for mission types in area types"""
        
        if not area_type or area_type == "unknown":
            return
        
        mission_type = outcome.get("mission_type", "unknown")
        success = outcome.get("success_score", 0.5)
        
        conn = sqlite3.connect(self.learning_db_path)
        cursor = conn.cursor()
        
        # Check if correlation exists
        cursor.execute('''
            SELECT success_rate, sample_size, optimal_parameters
            FROM mission_correlations
            WHERE area_type = ? AND mission_type = ?
        ''', (area_type, mission_type))
        
        existing = cursor.fetchone()
        
        if existing:
            old_rate, samples, old_params = existing
            
            # Update success rate with moving average
            new_rate = (old_rate * samples + success) / (samples + 1)
            
            # Update optimal parameters if this was successful
            if success > 0.7:
                params = json.loads(old_params) if old_params else {}
                new_params = outcome.get("mission_parameters", {})
                
                # Merge successful parameters
                for key, value in new_params.items():
                    params[key] = value
                
                cursor.execute('''
                    UPDATE mission_correlations
                    SET success_rate = ?, sample_size = ?, optimal_parameters = ?,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE area_type = ? AND mission_type = ?
                ''', (new_rate, samples + 1, json.dumps(params), area_type, mission_type))
            else:
                cursor.execute('''
                    UPDATE mission_correlations
                    SET success_rate = ?, sample_size = ?, last_updated = CURRENT_TIMESTAMP
                    WHERE area_type = ? AND mission_type = ?
                ''', (new_rate, samples + 1, area_type, mission_type))
        else:
            # Create new correlation
            params = outcome.get("mission_parameters", {}) if success > 0.7 else {}
            
            cursor.execute('''
                INSERT INTO mission_correlations
                (area_type, mission_type, success_rate, optimal_parameters, sample_size)
                VALUES (?, ?, ?, ?, 1)
            ''', (area_type, mission_type, success, json.dumps(params)))
        
        conn.commit()
        conn.close()
    
    def get_map_overlay_data(self, bounds: Dict) -> Dict:
        """Get data to overlay on the map showing learned areas"""
        
        overlay_data = {
            "explored_areas": [],
            "hazard_zones": [],
            "high_success_areas": [],
            "labels": []
        }
        
        # Get areas within bounds
        min_x = bounds.get("min_x", -500)
        max_x = bounds.get("max_x", 500)
        min_y = bounds.get("min_y", -500)
        max_y = bounds.get("max_y", 500)
        
        for area_key, area_info in self.area_knowledge["known_areas"].items():
            center = area_info["center"]
            
            if min_x <= center["x"] <= max_x and min_y <= center["y"] <= max_y:
                # Add explored area
                overlay_data["explored_areas"].append({
                    "center": center,
                    "radius": area_info["radius"],
                    "confidence": area_info["confidence"],
                    "type": area_info["type"],
                    "exploration_count": area_info.get("exploration_count", 1)
                })
                
                # Add hazard zones
                for hazard in area_info.get("hazards", []):
                    overlay_data["hazard_zones"].append({
                        "center": center,
                        "radius": area_info["radius"] * 0.8,
                        "hazard_type": hazard.get("type", "unknown"),
                        "severity": hazard.get("severity", "medium")
                    })
                
                # Add labels for named areas
                if area_info.get("name"):
                    overlay_data["labels"].append({
                        "position": center,
                        "text": area_info["name"],
                        "type": area_info["type"]
                    })
        
        # Add high success areas from correlations
        for area_type, missions in self.area_knowledge["mission_correlations"].items():
            for mission_type, data in missions.items():
                if data["success_rate"] > 0.8:
                    # This is a high success combination
                    # Find areas of this type
                    for area_key, area_info in self.area_knowledge["known_areas"].items():
                        if area_info["type"] == area_type:
                            overlay_data["high_success_areas"].append({
                                "center": area_info["center"],
                                "radius": area_info["radius"],
                                "area_type": area_type,
                                "mission_type": mission_type,
                                "success_rate": data["success_rate"]
                            })
        
        return overlay_data
    
    def suggest_locations_for_mission(self, mission_type: str, constraints: Optional[Dict] = None) -> List[Dict]:
        """Suggest best locations for a specific mission type based on learning"""
        
        suggestions = []
        
        # Check mission correlations
        for area_type, missions in self.area_knowledge["mission_correlations"].items():
            if mission_type in missions:
                success_data = missions[mission_type]
                
                if success_data["success_rate"] > 0.6:
                    # Find areas of this type
                    for area_key, area_info in self.area_knowledge["known_areas"].items():
                        if area_info["type"] == area_type:
                            score = (success_data["success_rate"] * 0.5 + 
                                   area_info["confidence"] * 0.3 +
                                   area_info.get("accessibility", 0.5) * 0.2)
                            
                            suggestions.append({
                                "location": area_info["center"],
                                "radius": area_info["radius"],
                                "area_name": area_info.get("name", f"Area {area_key}"),
                                "area_type": area_type,
                                "predicted_success": success_data["success_rate"],
                                "confidence": area_info["confidence"],
                                "score": score,
                                "reasoning": f"High success rate for {mission_type} in {area_type} areas"
                            })
        
        # Sort by score
        suggestions.sort(key=lambda x: x["score"], reverse=True)
        
        return suggestions[:5]  # Top 5 suggestions
    
    def _fallback_pin_analysis(self, pin: MapPin, description: str) -> Dict:
        """Fallback analysis when AI fails"""
        
        return {
            "location": {
                "coordinates": {"x": pin.x, "y": pin.y, "z": pin.z},
                "radius": pin.radius
            },
            "user_intent": {
                "interpreted_intent": {
                    "primary_objective": "exploration",
                    "mission_type": "general_search",
                    "urgency": "normal"
                }
            },
            "area_analysis": {
                "type": "unknown",
                "characteristics": {},
                "known_hazards": [],
                "accessibility": 0.5
            },
            "recommendations": {
                "search_pattern": "spiral",
                "estimated_duration": 20
            },
            "confidence": 0.3,
            "clarifications_needed": ["Please provide more details about the mission objective"]
        }

    def _rule_based_intent_analysis(self, description: str, area_knowledge: Dict) -> Dict:
        """Fallback: Use simple rules to interpret intent if AI fails"""
        result = {
            "interpreted_intent": {
                "mission_type": "exploration",
                "priority": "normal"
            },
            "confidence": 0.3,
            "assumptions": ["Used rule-based fallback"],
            "needs_clarification": True
        }
        desc_lower = description.lower()
        if "rescue" in desc_lower:
            result["interpreted_intent"]["mission_type"] = "rescue"
            result["confidence"] += 0.2
        elif "assessment" in desc_lower or "survey" in desc_lower:
            result["interpreted_intent"]["mission_type"] = "assessment"
            result["confidence"] += 0.1
        if "urgent" in desc_lower or "immediate" in desc_lower:
            result["interpreted_intent"]["priority"] = "high"
            result["confidence"] += 0.1
        return result

    def _synthesize_area_knowledge(self, nearby_areas: list, x: float, y: float) -> dict:
        """Combine knowledge from nearby areas to synthesize area knowledge."""
        if not nearby_areas:
            return {}
        # Simple average of characteristics and hazards
        combined = {"characteristics": {}, "hazards": [], "type": "unknown"}
        char_counts = {}
        for area in nearby_areas:
            chars = area.get("characteristics", {})
            for k, v in chars.items():
                if isinstance(v, (int, float)):
                    combined["characteristics"][k] = combined["characteristics"].get(k, 0) + v
                    char_counts[k] = char_counts.get(k, 0) + 1
                else:
                    combined["characteristics"][k] = v
            for hazard in area.get("hazards", []):
                if hazard not in combined["hazards"]:
                    combined["hazards"].append(hazard)
            if area.get("type"):
                combined["type"] = area["type"]
        # Average numeric characteristics
        for k, v in combined["characteristics"].items():
            if isinstance(v, (int, float)):
                combined["characteristics"][k] = v / char_counts[k]
        return combined

    def _find_nearby_areas(self, x: float, y: float, radius: float) -> list:
        """Find areas within a given radius of (x, y)."""
        results = []
        for area in self.area_knowledge.get("known_areas", {}).values():
            center = area.get("center", {})
            dist = math.sqrt((x - center.get("x", 0))**2 + (y - center.get("y", 0))**2)
            if dist <= radius:
                results.append(area)
        return results

# Global instance
map_location_ai = MapBasedLocationAI()