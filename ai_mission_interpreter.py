"""
AI Mission Interpreter with Continuous Learning
Learns from user feedback and mission outcomes to improve over time
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import re
import sqlite3
from pathlib import Path
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)

class LearningMissionInterpreterAI:
    """
    AI component that interprets natural language commands and learns from feedback
    """
    
    def __init__(self, model_name: str = "mistral:7b-instruct"):
        self.model_name = model_name
        self.ollama_url = "http://localhost:11434"
        
        # Initialize learning database
        self.learning_db_path = Path("data/ai_learning.db")
        self.learning_db_path.parent.mkdir(exist_ok=True)
        self._init_learning_db()
        
        # Performance tracking
        self.performance_metrics = {
            "total_interpretations": 0,
            "successful_interpretations": 0,
            "user_corrections": 0,
            "confidence_improvements": []
        }
        
        # Learning parameters
        self.learning_rate = 0.1
        self.feedback_weight = 0.8
        self.min_confidence_threshold = 0.7
        
        # Mission patterns (will be updated through learning)
        self.mission_patterns = self._load_learned_patterns()
        
        # Context that evolves through learning
        self.system_context = self._build_dynamic_context()
        
    def _init_learning_db(self):
        """Initialize database for storing learning data"""
        
        conn = sqlite3.connect(self.learning_db_path)
        cursor = conn.cursor()
        
        # Interpretations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interpretations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                command TEXT NOT NULL,
                ai_interpretation TEXT NOT NULL,
                user_correction TEXT,
                final_interpretation TEXT,
                confidence_before REAL,
                confidence_after REAL,
                mission_outcome TEXT,
                success_score REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Learned patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learned_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT NOT NULL,
                pattern_text TEXT NOT NULL,
                success_rate REAL,
                usage_count INTEGER DEFAULT 1,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # User preferences table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                preference_type TEXT NOT NULL,
                preference_value TEXT NOT NULL,
                frequency INTEGER DEFAULT 1,
                last_used DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interpretation_id INTEGER,
                feedback_type TEXT,
                feedback_value TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(interpretation_id) REFERENCES interpretations(id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def _load_learned_patterns(self) -> Dict:
        """Load learned patterns from database"""
        
        base_patterns = {
            "search_rescue": [
                "survivor", "people", "person", "victim", "trapped", 
                "injured", "help", "rescue", "find someone", "casualties"
            ],
            "damage_assessment": [
                "damage", "assess", "inspect", "structural", "collapsed",
                "broken", "destroyed", "condition", "stability"
            ],
            "hazard_detection": [
                "fire", "gas", "chemical", "hazard", "danger", "toxic",
                "smoke", "leak", "explosion", "unsafe"
            ]
        }
        
        # Load learned patterns from database
        try:
            conn = sqlite3.connect(self.learning_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT pattern_type, pattern_text, success_rate
                FROM learned_patterns
                WHERE success_rate > 0.6
                ORDER BY success_rate DESC
            ''')
            
            for pattern_type, pattern_text, success_rate in cursor.fetchall():
                if pattern_type in base_patterns:
                    if pattern_text not in base_patterns[pattern_type]:
                        base_patterns[pattern_type].append(pattern_text)
                else:
                    base_patterns[pattern_type] = [pattern_text]
            
            conn.close()
            
        except Exception as e:
            logger.warning(f"Could not load learned patterns: {e}")
        
        return base_patterns
    
    def _build_dynamic_context(self) -> str:
        """Build context that includes learned information"""
        
        base_context = """You are an AI mission interpreter for a Search and Rescue drone system.
        
Your role is to understand natural language commands from rescue operators and convert them into structured mission parameters.

IMPORTANT: You are continuously learning from user feedback. Consider these learned insights:
"""
        
        # Add learned insights
        try:
            conn = sqlite3.connect(self.learning_db_path)
            cursor = conn.cursor()
            
            # Get top successful patterns
            cursor.execute('''
                SELECT preference_type, preference_value, frequency
                FROM user_preferences
                ORDER BY frequency DESC
                LIMIT 10
            ''')
            
            preferences = cursor.fetchall()
            if preferences:
                base_context += "\n\nLearned User Preferences:\n"
                for pref_type, pref_value, freq in preferences:
                    base_context += f"- {pref_type}: {pref_value} (used {freq} times)\n"
            
            # Get common corrections
            cursor.execute('''
                SELECT command, user_correction
                FROM interpretations
                WHERE user_correction IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT 5
            ''')
            
            corrections = cursor.fetchall()
            if corrections:
                base_context += "\n\nRecent User Corrections:\n"
                for cmd, correction in corrections:
                    base_context += f"- Command: '{cmd[:50]}...' was corrected\n"
            
            conn.close()
            
        except Exception as e:
            logger.warning(f"Could not load learning context: {e}")
        
        base_context += """
        
Key responsibilities:
1. Extract the PRIMARY OBJECTIVE (what needs to be done)
2. Identify the LOCATION (where to search/operate)
3. Determine PRIORITY LEVEL (critical/high/normal/low)
4. Learn from user corrections and improve interpretations
5. Track confidence based on past performance

You must ALWAYS output valid JSON with this structure:
{
    "objective": "primary mission type",
    "objective_details": "specific details about what to do",
    "location": "where to perform the mission",
    "location_details": "specific areas, floors, sections mentioned",
    "priority": "critical/high/normal/low",
    "time_constraint": "any time limits mentioned",
    "special_instructions": ["list of specific requirements"],
    "constraints": ["list of limitations"],
    "confidence": 0.0-1.0,
    "clarifications_needed": ["list of ambiguous items"],
    "learning_notes": "what patterns you recognized from past learning"
}"""
        
        return base_context
    
    def interpret_command(self, command: str, context: Optional[Dict] = None) -> Dict:
        """
        Interpret a natural language command with learning
        """
        interpretation_id = None
        
        try:
            # Record start of interpretation
            interpretation_id = self._record_interpretation_start(command)
            
            # Check if we've seen similar command before
            similar_past = self._find_similar_interpretations(command)
            
            # Adjust confidence based on past performance
            base_confidence = self._calculate_base_confidence(similar_past)
            
            # First try AI interpretation with learning context
            ai_result = self._ai_interpret_with_learning(command, context, similar_past)
            
            # Adjust confidence based on learning
            ai_result["confidence"] = self._adjust_confidence(ai_result.get("confidence", 0.5), base_confidence)
            
            # Validate and enhance with learned rules
            enhanced_result = self._enhance_with_learned_rules(command, ai_result)
            
            # Add metadata
            enhanced_result["interpreted_at"] = datetime.now().isoformat()
            enhanced_result["interpreter_version"] = "2.0-learning"
            enhanced_result["original_command"] = command
            enhanced_result["interpretation_id"] = interpretation_id
            
            # Update performance metrics
            self.performance_metrics["total_interpretations"] += 1
            
            logger.info(f"Successfully interpreted command with confidence {enhanced_result['confidence']}")
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Failed to interpret command: {e}")
            # Fallback to rule-based interpretation
            return self._rule_based_interpret(command)
    
    def _rule_based_interpret(self, command: str) -> Dict:
        """Fallback rule-based interpretation when AI fails"""
        return {
            "objective": "search_and_rescue",
            "objective_details": "Basic search operation",
            "location": "unspecified",
            "location_details": "",
            "priority": "normal",
            "time_constraint": None,
            "special_instructions": [],
            "constraints": [],
            "confidence": 0.3,
            "clarifications_needed": ["Please specify location", "Please clarify objective"],
            "learning_notes": "Used fallback interpretation due to AI failure",
            "interpreted_at": datetime.now().isoformat(),
            "interpreter_version": "2.0-learning",
            "original_command": command
        }
    
    def _record_interpretation_start(self, command: str) -> int:
        """Record the start of an interpretation for learning"""
        
        conn = sqlite3.connect(self.learning_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO interpretations (command, ai_interpretation, confidence_before)
            VALUES (?, '{}', 0.5)
        ''', (command,))
        
        interpretation_id = cursor.lastrowid or 0
        conn.commit()
        conn.close()
        
        return interpretation_id
    
    def _find_similar_interpretations(self, command: str) -> List[Dict]:
        """Find similar past interpretations"""
        
        similar = []
        
        try:
            conn = sqlite3.connect(self.learning_db_path)
            cursor = conn.cursor()
            
            # Simple similarity: commands with overlapping words
            words = set(command.lower().split())
            
            cursor.execute('''
                SELECT command, final_interpretation, success_score, confidence_after
                FROM interpretations
                WHERE success_score IS NOT NULL
                ORDER BY timestamp DESC
                LIMIT 100
            ''')
            
            for past_cmd, final_interp, success, confidence in cursor.fetchall():
                past_words = set(past_cmd.lower().split())
                overlap = len(words.intersection(past_words)) / max(len(words), len(past_words))
                
                if overlap > 0.5:  # 50% word overlap
                    similar.append({
                        "command": past_cmd,
                        "interpretation": json.loads(final_interp) if final_interp else {},
                        "success_score": success,
                        "confidence": confidence,
                        "similarity": overlap
                    })
            
            conn.close()
            
        except Exception as e:
            logger.warning(f"Could not find similar interpretations: {e}")
        
        return sorted(similar, key=lambda x: x["similarity"], reverse=True)[:5]
    
    def _calculate_base_confidence(self, similar_past: List[Dict]) -> float:
        """Calculate base confidence from past performance"""
        
        if not similar_past:
            return 0.5
        
        # Weighted average of past success
        total_weight = 0
        weighted_confidence = 0
        
        for past in similar_past:
            weight = past["similarity"] * past.get("success_score", 0.5)
            weighted_confidence += weight * past.get("confidence", 0.5)
            total_weight += weight
        
        if total_weight > 0:
            return weighted_confidence / total_weight
        
        return 0.5
    
    def _ai_interpret_with_learning(self, command: str, context: Optional[Dict], similar_past: List[Dict]) -> Dict:
        """AI interpretation enhanced with learning"""
        
        # Build prompt with learning context
        prompt = self.system_context + "\n\n"
        
        if similar_past:
            prompt += "Similar past interpretations:\n"
            for past in similar_past[:3]:
                prompt += f"- Command: '{past['command']}' → Success: {past.get('success_score', 'unknown')}\n"
            prompt += "\n"
        
        if context:
            prompt += f"Additional context:\n"
            for key, value in context.items():
                prompt += f"- {key}: {value}\n"
            prompt += "\n"
        
        prompt += f"Interpret this command: {command}\n\nOutput JSON:"
        
        try:
            # Query Ollama with learning context
            import requests
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
                raise Exception(f"AI query failed: {response.status_code}")
                
        except Exception as e:
            logger.warning(f"AI interpretation failed: {e}")
            raise
    
    def _adjust_confidence(self, ai_confidence: float, base_confidence: float) -> float:
        """Adjust confidence based on learning"""
        
        # Blend AI confidence with learned confidence
        adjusted = (ai_confidence * 0.6) + (base_confidence * 0.4)
        
        # Apply learning rate
        adjusted = adjusted + (self.learning_rate * (base_confidence - 0.5))
        
        # Ensure in valid range
        return max(0.1, min(1.0, adjusted))
    
    def _enhance_with_learned_rules(self, command: str, ai_result: Dict) -> Dict:
        """Enhance with rules learned from user feedback"""
        
        # Load user preferences
        try:
            conn = sqlite3.connect(self.learning_db_path)
            cursor = conn.cursor()
            
            # Check for learned location preferences
            if ai_result.get("location") == "unspecified":
                cursor.execute('''
                    SELECT preference_value
                    FROM user_preferences
                    WHERE preference_type = 'default_location'
                    ORDER BY frequency DESC
                    LIMIT 1
                ''')
                
                result = cursor.fetchone()
                if result:
                    ai_result["location"] = result[0]
                    ai_result["learning_notes"] = "Used learned default location"
            
            conn.close()
            
        except Exception as e:
            logger.warning(f"Could not apply learned rules: {e}")
        
        return ai_result
    
    def record_user_feedback(self, interpretation_id: int, feedback: Dict):
        """Record user feedback for learning"""
        
        try:
            conn = sqlite3.connect(self.learning_db_path)
            cursor = conn.cursor()
            
            # Update interpretation with feedback
            if "correction" in feedback:
                cursor.execute('''
                    UPDATE interpretations
                    SET user_correction = ?, final_interpretation = ?
                    WHERE id = ?
                ''', (json.dumps(feedback["correction"]), json.dumps(feedback["correction"]), interpretation_id))
                
                self.performance_metrics["user_corrections"] += 1
            
            # Record specific feedback
            for feedback_type, feedback_value in feedback.items():
                cursor.execute('''
                    INSERT INTO feedback (interpretation_id, feedback_type, feedback_value)
                    VALUES (?, ?, ?)
                ''', (interpretation_id, feedback_type, str(feedback_value)))
            
            # Learn from corrections
            if "correction" in feedback:
                self._learn_from_correction(interpretation_id, feedback["correction"])
            
            conn.commit()
            conn.close()
            
            # Rebuild context with new learning
            self.system_context = self._build_dynamic_context()
            
        except Exception as e:
            logger.error(f"Failed to record feedback: {e}")
    
    def _learn_from_correction(self, interpretation_id: int, correction: Dict):
        """Learn patterns from user corrections"""
        
        try:
            conn = sqlite3.connect(self.learning_db_path)
            cursor = conn.cursor()
            
            # Get original command
            cursor.execute('SELECT command FROM interpretations WHERE id = ?', (interpretation_id,))
            result = cursor.fetchone()
            
            if result:
                command = result[0]
                
                # Learn objective patterns
                if correction.get("objective"):
                    words = command.lower().split()
                    for word in words:
                        if len(word) > 3:  # Skip short words
                            cursor.execute('''
                                INSERT OR REPLACE INTO learned_patterns 
                                (pattern_type, pattern_text, success_rate, usage_count)
                                VALUES (?, ?, 
                                    COALESCE((SELECT success_rate FROM learned_patterns 
                                              WHERE pattern_type = ? AND pattern_text = ?), 0.7),
                                    COALESCE((SELECT usage_count FROM learned_patterns 
                                              WHERE pattern_type = ? AND pattern_text = ?) + 1, 1)
                                )
                            ''', (correction["objective"], word, correction["objective"], word, 
                                  correction["objective"], word))
                
                # Learn location preferences
                if correction.get("location") and correction["location"] != "unspecified":
                    cursor.execute('''
                        INSERT OR REPLACE INTO user_preferences
                        (preference_type, preference_value, frequency)
                        VALUES ('common_location', ?, 
                            COALESCE((SELECT frequency FROM user_preferences 
                                      WHERE preference_type = 'common_location' 
                                      AND preference_value = ?) + 1, 1)
                        )
                    ''', (correction["location"], correction["location"]))
            
            conn.commit()
            conn.close()
            
            # Reload patterns
            self.mission_patterns = self._load_learned_patterns()
            
        except Exception as e:
            logger.error(f"Failed to learn from correction: {e}")
    
    def record_mission_outcome(self, interpretation_id: int, outcome: Dict):
        """Record mission outcome for learning"""
        
        try:
            conn = sqlite3.connect(self.learning_db_path)
            cursor = conn.cursor()
            
            success_score = outcome.get("success_score", 0.5)
            
            cursor.execute('''
                UPDATE interpretations
                SET mission_outcome = ?, success_score = ?, confidence_after = ?
                WHERE id = ?
            ''', (json.dumps(outcome), success_score, outcome.get("final_confidence", 0.5), interpretation_id))
            
            # Update performance metrics
            if success_score > 0.7:
                self.performance_metrics["successful_interpretations"] += 1
            
            self.performance_metrics["confidence_improvements"].append(
                outcome.get("final_confidence", 0.5) - 0.5
            )
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to record mission outcome: {e}")
    
    def get_learning_stats(self) -> Dict:
        """Get learning statistics"""
        
        stats = {
            "performance_metrics": self.performance_metrics,
            "learning_progress": {}
        }
        
        try:
            conn = sqlite3.connect(self.learning_db_path)
            cursor = conn.cursor()
            
            # Get interpretation accuracy over time
            cursor.execute('''
                SELECT 
                    DATE(timestamp) as date,
                    AVG(success_score) as avg_success,
                    AVG(confidence_after) as avg_confidence,
                    COUNT(*) as count
                FROM interpretations
                WHERE success_score IS NOT NULL
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
                LIMIT 30
            ''')
            
            stats["learning_progress"]["daily_performance"] = [
                {
                    "date": row[0],
                    "avg_success": row[1],
                    "avg_confidence": row[2],
                    "interpretation_count": row[3]
                }
                for row in cursor.fetchall()
            ]
            
            # Get most learned patterns
            cursor.execute('''
                SELECT pattern_type, COUNT(*) as count, AVG(success_rate) as avg_success
                FROM learned_patterns
                GROUP BY pattern_type
                ORDER BY count DESC
            ''')
            
            stats["learning_progress"]["pattern_learning"] = [
                {
                    "type": row[0],
                    "patterns_learned": row[1],
                    "avg_success_rate": row[2]
                }
                for row in cursor.fetchall()
            ]
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to get learning stats: {e}")
        
        return stats
    
    def export_learning_data(self, filepath: str):
        """Export learning data for backup or analysis"""
        
        try:
            conn = sqlite3.connect(self.learning_db_path)
            
            # Export to JSON
            data = {
                "exported_at": datetime.now().isoformat(),
                "interpretations": [],
                "learned_patterns": [],
                "user_preferences": []
            }
            
            # Export interpretations
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM interpretations')
            columns = [description[0] for description in cursor.description]
            for row in cursor.fetchall():
                data["interpretations"].append(dict(zip(columns, row)))
            
            # Export patterns
            cursor.execute('SELECT * FROM learned_patterns')
            columns = [description[0] for description in cursor.description]
            for row in cursor.fetchall():
                data["learned_patterns"].append(dict(zip(columns, row)))
            
            # Export preferences
            cursor.execute('SELECT * FROM user_preferences')
            columns = [description[0] for description in cursor.description]
            for row in cursor.fetchall():
                data["user_preferences"].append(dict(zip(columns, row)))
            
            conn.close()
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Exported learning data to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export learning data: {e}")

# Global instance with learning capabilities
mission_interpreter = LearningMissionInterpreterAI()