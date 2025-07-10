"""
Database setup and models for SAR Drone System
Handles all data storage for drones, missions, and detections
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from datetime import datetime
import json

from core.config import settings

# Create database engine
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

# Database Models

class Drone(Base):
    """
    Drone database model - stores all drone information
    """
    __tablename__ = "drones"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    status = Column(String, default="offline")  # offline, idle, armed, flying, returning
    battery = Column(Float, default=100.0)
    
    # Position data
    position_x = Column(Float, default=0.0)
    position_y = Column(Float, default=0.0)
    position_z = Column(Float, default=0.0)
    heading = Column(Float, default=0.0)  # degrees
    
    # Velocity data
    velocity_x = Column(Float, default=0.0)
    velocity_y = Column(Float, default=0.0)
    velocity_z = Column(Float, default=0.0)
    
    # System data
    last_seen = Column(DateTime, default=datetime.utcnow)
    total_flight_time = Column(Integer, default=0)  # seconds
    total_distance = Column(Float, default=0.0)  # meters
    error_count = Column(Integer, default=0)
    
    # Mission assignment
    current_mission_id = Column(Integer, ForeignKey("missions.id"), nullable=True)
    
    # Relationships
    detections = relationship("Detection", back_populates="drone")
    telemetry_logs = relationship("TelemetryLog", back_populates="drone")

class Mission(Base):
    """
    Mission database model - stores search mission data
    """
    __tablename__ = "missions"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    type = Column(String)  # search, patrol, specific_location
    status = Column(String, default="planned")  # planned, active, paused, completed, aborted
    
    # Timing
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    ended_at = Column(DateTime, nullable=True)
    
    # Mission parameters
    search_area = Column(JSON)  # Polygon coordinates
    parameters = Column(JSON)  # Mission-specific settings
    priority_zones = Column(JSON)  # High-priority areas
    
    # Results
    area_covered = Column(Float, default=0.0)  # square meters
    detections_count = Column(Integer, default=0)
    
    # Relationships
    assigned_drones = relationship("Drone", backref="current_mission")
    detections = relationship("Detection", back_populates="mission")

class Detection(Base):
    """
    Detection database model - stores all detection events
    """
    __tablename__ = "detections"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Detection info
    type = Column(String)  # human, vehicle, heat_signature, movement
    confidence = Column(Float)  # 0.0 to 1.0
    
    # Location
    position_x = Column(Float)
    position_y = Column(Float)
    position_z = Column(Float)
    
    # Additional data
    temperature = Column(Float, nullable=True)  # For thermal detections
    size_estimate = Column(Float, nullable=True)  # Estimated size in meters
    image_path = Column(String, nullable=True)  # Path to saved image
    
    # Metadata
    timestamp = Column(DateTime, default=datetime.utcnow)
    verified = Column(Boolean, default=False)
    false_positive = Column(Boolean, default=False)
    notes = Column(String, nullable=True)
    
    # Foreign keys
    drone_id = Column(Integer, ForeignKey("drones.id"))
    mission_id = Column(Integer, ForeignKey("missions.id"))
    
    # Relationships
    drone = relationship("Drone", back_populates="detections")
    mission = relationship("Mission", back_populates="detections")

class MapCell(Base):
    """
    Map cell database model - stores explored area data
    """
    __tablename__ = "map_cells"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Grid position
    grid_x = Column(Integer, index=True)
    grid_y = Column(Integer, index=True)
    
    # Cell data
    cell_type = Column(String, default="unknown")  # unknown, empty, obstacle, danger, explored
    elevation = Column(Float, nullable=True)  # Height in meters
    last_updated = Column(DateTime, default=datetime.utcnow)
    confidence = Column(Float, default=1.0)
    
    # Which drone explored this
    explored_by_drone_id = Column(Integer, ForeignKey("drones.id"), nullable=True)
    mission_id = Column(Integer, ForeignKey("missions.id"), nullable=True)

class TelemetryLog(Base):
    """
    Telemetry log - stores drone telemetry for analysis
    """
    __tablename__ = "telemetry_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Drone data at this moment
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    drone_id = Column(Integer, ForeignKey("drones.id"))
    
    # Position and motion
    position_x = Column(Float)
    position_y = Column(Float)
    position_z = Column(Float)
    velocity_x = Column(Float)
    velocity_y = Column(Float)
    velocity_z = Column(Float)
    
    # System status
    battery = Column(Float)
    signal_strength = Column(Integer)  # 0-100
    cpu_usage = Column(Float)
    temperature = Column(Float)
    
    # Relationship
    drone = relationship("Drone", back_populates="telemetry_logs")

# Create tables
def init_db():
    """
    Initialize database - create all tables
    """
    Base.metadata.create_all(bind=engine)

# Dependency to get database session
def get_db(Session):
    """
    Get database session - used in API endpoints
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()