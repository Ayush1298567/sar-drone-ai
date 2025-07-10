"""
Configuration settings for the SAR Drone System
All system-wide settings are defined here
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    """
    System configuration using environment variables
    """
    # Application settings
    APP_NAME: str = "SAR Drone System"
    VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Database settings
    DATABASE_URL: str = "sqlite:///./sar_drone.db"
    
    # Drone settings
    MAX_DRONES: int = 10
    DEFAULT_ALTITUDE: float = 10.0  # meters
    MAX_ALTITUDE: float = 50.0  # meters
    MIN_BATTERY: float = 20.0  # percent to trigger return
    
    # Mission settings
    DEFAULT_SEARCH_PATTERN: str = "spiral"
    SEARCH_GRID_SIZE: float = 1.0  # meters between search points
    
    # Simulator settings (since no real drones yet)
    SIMULATOR_ENABLED: bool = True
    SIM_DRONES: int = 3  # Number of simulated drones
    SIM_BATTERY_DRAIN_RATE: float = 0.1  # percent per second
    SIM_SPEED: float = 5.0  # meters per second
    
    # Communication settings
    HEARTBEAT_INTERVAL: int = 1  # seconds
    COMMAND_TIMEOUT: int = 5  # seconds
    
    # Detection settings
    DETECTION_CONFIDENCE_THRESHOLD: float = 0.7
    THERMAL_HUMAN_MIN: float = 35.0  # Celsius
    THERMAL_HUMAN_MAX: float = 38.0  # Celsius
    
    # Map settings
    MAP_GRID_SIZE: int = 1000  # 1000x1000 grid
    MAP_RESOLUTION: float = 0.5  # meters per grid cell
    
    # AI/Ollama settings
    OLLAMA_HOST: str = "http://localhost:11434"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create global settings instance
settings = Settings()