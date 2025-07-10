"""
Comprehensive AI Manager for Ollama Integration
Handles all AI model operations with production-grade reliability
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import aiohttp
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
import hashlib
from collections import deque
import threading
import queue

from core.config import settings
from core.errors import SADroneException, log_mission_critical, log_safety_event

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """AI Model metadata"""
    name: str
    size_gb: float
    purpose: str
    max_tokens: int
    temperature: float
    timeout_seconds: int
    retry_attempts: int
    fallback_model: Optional[str] = None
    
@dataclass
class AIResponse:
    """Structured AI response with metadata"""
    content: str
    model_used: str
    confidence: float
    processing_time: float
    timestamp: datetime
    tokens_used: int
    cached: bool = False
    fallback_used: bool = False

class ResponseCache:
    """LRU cache for AI responses to improve performance"""
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache: Dict[str, Tuple[AIResponse, datetime]] = {}
        self.access_order = deque(maxlen=max_size)
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()
    
    def _generate_key(self, prompt: str, model: str) -> str:
        """Generate cache key from prompt and model"""
        content = f"{model}:{prompt}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, prompt: str, model: str) -> Optional[AIResponse]:
        """Get cached response if available and not expired"""
        with self._lock:
            key = self._generate_key(prompt, model)
            if key in self.cache:
                response, timestamp = self.cache[key]
                if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                    self.hits += 1
                    # Move to end (most recently used)
                    self.access_order.remove(key)
                    self.access_order.append(key)
                    response.cached = True
                    logger.debug(f"Cache hit for model {model}, ratio: {self.get_hit_ratio():.2%}")
                    return response
                else:
                    # Expired
                    del self.cache[key]
                    self.access_order.remove(key)
            
            self.misses += 1
            return None
    
    def put(self, prompt: str, model: str, response: AIResponse):
        """Cache a response"""
        with self._lock:
            key = self._generate_key(prompt, model)
            
            # Remove oldest if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                oldest_key = self.access_order.popleft()
                del self.cache[oldest_key]
            
            self.cache[key] = (response, datetime.now())
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
    
    def get_hit_ratio(self) -> float:
        """Get cache hit ratio"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

class OllamaConnectionPool:
    """Connection pool for Ollama API with health monitoring"""
    def __init__(self, base_url: str, pool_size: int = 5):
        self.base_url = base_url
        self.pool_size = pool_size
        self.health_check_interval = 30  # seconds
        self.last_health_check = None
        self.is_healthy = False
        self.available_models: List[str] = []
        self._session = None
        self._health_check_task = None
        
    async def initialize(self):
        """Initialize connection pool and start health monitoring"""
        timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout for AI ops
        connector = aiohttp.TCPConnector(limit=self.pool_size, force_close=True)
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout
        )
        self._health_check_task = asyncio.create_task(self._health_monitor())
        await self._check_health()
    
    async def close(self):
        """Close connection pool"""
        if self._health_check_task:
            self._health_check_task.cancel()
        if self._session:
            await self._session.close()
    
    async def _check_health(self) -> bool:
        """Check Ollama health and available models"""
        if not self._session:
            logger.error("Session not initialized")
            self.is_healthy = False
            return False
        
        try:
            async with self._session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    self.available_models = [m["name"] for m in data.get("models", [])]
                    self.is_healthy = True
                    self.last_health_check = datetime.now()
                    logger.info(f"Ollama healthy with models: {self.available_models}")
                    return True
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            self.is_healthy = False
            return False
        
        # Add default return for when response.status != 200
        self.is_healthy = False
        return False
    
    async def _health_monitor(self):
        """Continuous health monitoring"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._check_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool"""
        if not self.is_healthy:
            await self._check_health()
            if not self.is_healthy:
                raise SADroneException("Ollama service unavailable", status_code=503)
        yield self._session

class AIManager:
    """
    Production-grade AI Manager for the SAR Drone System
    Handles all AI operations with safety, reliability, and performance
    """
    
    # Model configurations for different purposes
    MODELS = {
        "mission_planning": ModelInfo(
            name="mistral:7b-instruct",
            size_gb=4.1,
            purpose="Complex mission understanding and planning",
            max_tokens=2048,
            temperature=0.3,
            timeout_seconds=30,
            retry_attempts=3,
            fallback_model="phi:2.7b"
        ),
        "fast_decisions": ModelInfo(
            name="phi:2.7b",
            size_gb=1.6,
            purpose="Quick tactical decisions and responses",
            max_tokens=1024,
            temperature=0.1,
            timeout_seconds=10,
            retry_attempts=2
        ),
        "safety_analysis": ModelInfo(
            name="llama2:7b",
            size_gb=3.8,
            purpose="Safety and risk assessment",
            max_tokens=1024,
            temperature=0.1,
            timeout_seconds=20,
            retry_attempts=3
        ),
        "report_generation": ModelInfo(
            name="mistral:7b-instruct",
            size_gb=4.1,
            purpose="Detailed report generation",
            max_tokens=4096,
            temperature=0.5,
            timeout_seconds=60,
            retry_attempts=2
        )
    }
    
    def __init__(self):
        self.ollama_url = settings.OLLAMA_HOST
        self.connection_pool: Optional[OllamaConnectionPool] = None
        self.response_cache = ResponseCache(max_size=500, ttl_seconds=1800)
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "fallback_used": 0,
            "total_response_time": 0.0,
            "model_usage": {model: 0 for model in self.MODELS}
        }
        self._initialized = False
        self._fallback_rules = self._initialize_fallback_rules()
        
    def _initialize_fallback_rules(self) -> Dict[str, Any]:
        """Initialize rule-based fallback system for when AI is unavailable"""
        return {
            "mission_planning": {
                "search_building": {
                    "pattern": "spiral",
                    "altitude": 15,
                    "spacing": 5,
                    "speed": 3
                },
                "search_area": {
                    "pattern": "grid",
                    "altitude": 25,
                    "spacing": 10,
                    "speed": 5
                }
            },
            "safety_analysis": {
                "low_battery": 25.0,
                "max_wind_speed": 15.0,
                "min_visibility": 100.0,
                "no_fly_zones": []
            },
            "emergency_responses": {
                "drone_malfunction": "immediate_landing",
                "lost_communication": "return_to_home",
                "low_battery": "return_to_home",
                "bad_weather": "seek_shelter"
            }
        }
    
    async def initialize(self) -> bool:
        """Initialize AI manager and verify models"""
        try:
            logger.info("Initializing AI Manager...")
            
            # Initialize connection pool
            self.connection_pool = OllamaConnectionPool(self.ollama_url)
            await self.connection_pool.initialize()
            
            # Verify required models
            missing_models = []
            for model_type, model_info in self.MODELS.items():
                if model_info.name not in self.connection_pool.available_models:
                    missing_models.append(model_info.name)
                    logger.warning(f"Model {model_info.name} not available for {model_type}")
            
            if missing_models:
                logger.warning(f"Missing AI models: {missing_models}. System will use fallbacks.")
                log_safety_event("AI models missing, using fallback systems", models=missing_models)
            
            self._initialized = True
            logger.info("AI Manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize AI Manager: {e}", exc_info=True)
            log_mission_critical("AI system initialization failed", error=str(e))
            self._initialized = False
            return False
    
    async def get_response(
        self,
        prompt: str,
        model_type: str,
        context: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> AIResponse:
        """
        Get AI response with automatic fallback and caching
        
        Args:
            prompt: The prompt to send to the AI
            model_type: Type of model to use (mission_planning, fast_decisions, etc.)
            context: Additional context for the prompt
            use_cache: Whether to use cached responses
            
        Returns:
            AIResponse with content and metadata
        """
        start_time = time.time()
        self.performance_metrics["total_requests"] += 1
        
        if model_type not in self.MODELS:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_info = self.MODELS[model_type]
        
        # Check cache first
        if use_cache:
            cached_response = self.response_cache.get(prompt, model_info.name)
            if cached_response:
                self.performance_metrics["successful_requests"] += 1
                return cached_response
        
        # Build full prompt with context
        full_prompt = self._build_prompt_with_context(prompt, context, model_type)
        
        try:
            # Try primary model
            response = await self._query_model(full_prompt, model_info)
            
            # Cache successful response
            if use_cache:
                self.response_cache.put(prompt, model_info.name, response)
            
            self.performance_metrics["successful_requests"] += 1
            self.performance_metrics["model_usage"][model_type] += 1
            self.performance_metrics["total_response_time"] += response.processing_time
            
            return response
            
        except Exception as primary_error:
            logger.error(f"Primary model {model_info.name} failed: {primary_error}")
            
            # Try fallback model if available
            if model_info.fallback_model:
                try:
                    fallback_info = ModelInfo(
                        name=model_info.fallback_model,
                        size_gb=0,
                        purpose="fallback",
                        max_tokens=model_info.max_tokens,
                        temperature=model_info.temperature,
                        timeout_seconds=model_info.timeout_seconds // 2,
                        retry_attempts=1
                    )
                    
                    response = await self._query_model(full_prompt, fallback_info)
                    response.fallback_used = True
                    
                    self.performance_metrics["successful_requests"] += 1
                    self.performance_metrics["fallback_used"] += 1
                    
                    log_safety_event(
                        "AI fallback model used",
                        primary_model=model_info.name,
                        fallback_model=model_info.fallback_model
                    )
                    
                    return response
                    
                except Exception as fallback_error:
                    logger.error(f"Fallback model also failed: {fallback_error}")
            
            # Use rule-based fallback as last resort
            self.performance_metrics["failed_requests"] += 1
            return self._get_rule_based_response(prompt, model_type)
    
    async def _query_model(self, prompt: str, model_info: ModelInfo) -> AIResponse:
        """Query a specific model with retries"""
        if not self._initialized or not self.connection_pool:
            raise SADroneException("AI Manager not initialized")
        
        last_error = None
        for attempt in range(model_info.retry_attempts):
            try:
                start_time = time.time()
                
                async with self.connection_pool.get_connection() as session:
                    if not session:
                        raise SADroneException("No session available from connection pool")
                    
                    payload = {
                        "model": model_info.name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": model_info.temperature,
                            "num_predict": model_info.max_tokens,
                            "stop": ["Human:", "User:", "\n\n\n"]
                        }
                    }
                    
                    async with session.post(
                        f"{self.ollama_url}/api/generate",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=model_info.timeout_seconds)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            processing_time = time.time() - start_time
                            
                            # Parse response and extract confidence
                            content = data.get("response", "").strip()
                            confidence = self._extract_confidence(content)
                            
                            return AIResponse(
                                content=content,
                                model_used=model_info.name,
                                confidence=confidence,
                                processing_time=processing_time,
                                timestamp=datetime.now(),
                                tokens_used=data.get("eval_count", 0)
                            )
                        else:
                            error_text = await response.text()
                            raise SADroneException(
                                f"Ollama API error: {response.status} - {error_text}",
                                status_code=response.status
                            )
                            
            except asyncio.TimeoutError:
                last_error = f"Timeout after {model_info.timeout_seconds}s (attempt {attempt + 1})"
                logger.warning(f"Model {model_info.name} timeout: {last_error}")
                await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
                
            except Exception as e:
                last_error = str(e)
                logger.error(f"Model {model_info.name} error (attempt {attempt + 1}): {e}")
                if attempt < model_info.retry_attempts - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))
        
        raise SADroneException(f"All attempts failed for {model_info.name}: {last_error}")
    
    def _build_prompt_with_context(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]],
        model_type: str
    ) -> str:
        """Build comprehensive prompt with system context"""
        system_context = f"""You are an AI assistant for a Search and Rescue Drone System.
Current Mission Context:
- System is controlling {context.get('active_drones', 0) if context else 0} drones
- Operating in {context.get('environment', 'unknown') if context else 'unknown'} environment
- Priority: {context.get('priority', 'normal') if context else 'normal'}
- Safety mode: {context.get('safety_mode', 'enabled') if context else 'enabled'}

Your role: {self.MODELS[model_type].purpose}

CRITICAL SAFETY RULES:
1. Human life is the absolute priority
2. Drone safety is secondary but important
3. Always err on the side of caution
4. If unsure, request human confirmation
5. Consider battery life in all decisions

Provide response with confidence level (0.0-1.0) at the end in format: [CONFIDENCE: X.X]
"""
        
        if context:
            context_str = "\nAdditional Context:\n"
            for key, value in context.items():
                if key not in ['active_drones', 'environment', 'priority', 'safety_mode']:
                    context_str += f"- {key}: {value}\n"
            system_context += context_str
        
        return f"{system_context}\n\nUser Request: {prompt}\n\nResponse:"
    
    def _extract_confidence(self, response: str) -> float:
        """Extract confidence score from response"""
        import re
        confidence_match = re.search(r'\[CONFIDENCE:\s*(\d+\.?\d*)\]', response)
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
                # Remove confidence tag from response
                response = response.replace(confidence_match.group(0), "").strip()
                return min(max(confidence, 0.0), 1.0)
            except:
                pass
        return 0.5  # Default confidence
    
    def _get_rule_based_response(self, prompt: str, model_type: str) -> AIResponse:
        """Generate rule-based response when AI is unavailable"""
        logger.warning(f"Using rule-based fallback for {model_type}")
        log_safety_event("Rule-based system activated", model_type=model_type)
        
        # Simple keyword-based responses
        prompt_lower = prompt.lower()
        
        if model_type == "mission_planning":
            if "search" in prompt_lower and "building" in prompt_lower:
                rules = self._fallback_rules["mission_planning"]["search_building"]
                content = f"""Based on standard procedures:
- Use {rules['pattern']} search pattern
- Maintain {rules['altitude']}m altitude
- Space drones {rules['spacing']}m apart
- Flight speed: {rules['speed']}m/s
- Start from entrance and work inward
[RULE-BASED RESPONSE - Requires human verification]"""
                
            elif "search" in prompt_lower and "area" in prompt_lower:
                rules = self._fallback_rules["mission_planning"]["search_area"]
                content = f"""Based on standard procedures:
- Use {rules['pattern']} search pattern
- Maintain {rules['altitude']}m altitude  
- Grid spacing: {rules['spacing']}m
- Flight speed: {rules['speed']}m/s
- Cover area systematically
[RULE-BASED RESPONSE - Requires human verification]"""
            else:
                content = "Unable to generate AI response. Please provide manual mission parameters."
                
        elif model_type == "safety_analysis":
            safety_rules = self._fallback_rules["safety_analysis"]
            content = f"""Safety check based on standard rules:
- Minimum battery for operations: {safety_rules['low_battery']}%
- Maximum safe wind speed: {safety_rules['max_wind_speed']}m/s
- Minimum visibility: {safety_rules['min_visibility']}m
- All drones must maintain safe distances
- Emergency RTH if conditions deteriorate
[RULE-BASED RESPONSE - Conservative parameters applied]"""
            
        else:
            content = "AI unavailable. Manual operation required for this request."
        
        return AIResponse(
            content=content,
            model_used="rule_based",
            confidence=0.3,  # Low confidence for rule-based
            processing_time=0.01,
            timestamp=datetime.now(),
            tokens_used=0,
            fallback_used=True
        )
    
    async def validate_response(
        self,
        response: AIResponse,
        validation_type: str = "safety"
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate AI response for safety and correctness
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        content = response.content.lower()
        
        # Safety validation
        if validation_type == "safety":
            unsafe_keywords = [
                "ignore safety", "bypass", "override safety",
                "disable protection", "skip check"
            ]
            for keyword in unsafe_keywords:
                if keyword in content:
                    return False, f"Response contains unsafe instruction: {keyword}"
            
            # Check for minimum safety requirements
            if response.confidence < 0.4 and "emergency" not in content:
                return False, "Response confidence too low for safety-critical operation"
        
        # Mission validation
        elif validation_type == "mission":
            required_elements = ["pattern", "altitude", "area", "drones"]
            missing = [elem for elem in required_elements if elem not in content]
            if missing:
                return False, f"Mission plan missing required elements: {missing}"
        
        # Technical validation
        elif validation_type == "technical":
            if len(response.content) < 10:
                return False, "Response too short"
            
            # Get model info from the models dictionary
            model_info = self.MODELS.get(response.model_used)
            if model_info and response.tokens_used > model_info.max_tokens * 0.95:
                return False, "Response may be truncated"
        
        return True, None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get AI system performance metrics"""
        metrics = self.performance_metrics.copy()
        
        # Calculate additional metrics
        if metrics["total_requests"] > 0:
            metrics["success_rate"] = metrics["successful_requests"] / metrics["total_requests"]
            metrics["fallback_rate"] = metrics["fallback_used"] / metrics["total_requests"]
            metrics["average_response_time"] = (
                metrics["total_response_time"] / metrics["successful_requests"]
                if metrics["successful_requests"] > 0 else 0
            )
        else:
            metrics["success_rate"] = 0
            metrics["fallback_rate"] = 0
            metrics["average_response_time"] = 0
        
        metrics["cache_hit_ratio"] = self.response_cache.get_hit_ratio()
        metrics["models_available"] = (
            len(self.connection_pool.available_models)
            if self.connection_pool else 0
        )
        
        return metrics
    
    async def shutdown(self):
        """Gracefully shutdown AI manager"""
        logger.info("Shutting down AI Manager...")
        if self.connection_pool:
            await self.connection_pool.close()
        
        # Save performance metrics
        metrics = self.get_performance_metrics()
        logger.info(f"AI Manager final metrics: {json.dumps(metrics, indent=2)}")

# Global AI manager instance
ai_manager = AIManager()