"""
Database Models
SQLAlchemy models for storing network metrics and agent information.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Boolean, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import json

Base = declarative_base()

def local_now():
    """Return current local time instead of UTC"""
    return datetime.now()

class Agent(Base):
    """Agent registration and status"""
    __tablename__ = "agents"
    
    id = Column(Integer, primary_key=True, index=True)
    hostname = Column(String(255), unique=True, index=True, nullable=False)
    ip_address = Column(String(45))  # Support IPv6
    mac_address = Column(String(17))
    agent_version = Column(String(50))
    first_seen = Column(DateTime, default=local_now)
    last_seen = Column(DateTime, default=local_now)
    status = Column(String(20), default="active")  # active, inactive, error
    agent_metadata = Column(JSON)
    
    # Relationships
    metrics = relationship("NetworkMetric", back_populates="agent", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="agent")

class NetworkMetric(Base):
    """Network metrics data"""
    __tablename__ = "network_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # System metrics
    cpu_percent = Column(Float)
    memory_percent = Column(Float)
    memory_total = Column(Integer)
    memory_used = Column(Integer)
    memory_available = Column(Integer)
    disk_percent = Column(Float)
    disk_total = Column(Integer)
    disk_used = Column(Integer)
    disk_free = Column(Integer)
    load_avg_1min = Column(Float)
    load_avg_5min = Column(Float)
    load_avg_15min = Column(Float)
    
    # Network interface data (JSON)
    interfaces = Column(JSON)
    
    # Connection statistics
    tcp_established = Column(Integer)
    tcp_listen = Column(Integer)
    tcp_time_wait = Column(Integer)
    tcp_close_wait = Column(Integer)
    udp_connections = Column(Integer)
    total_connections = Column(Integer)
    
    # Bandwidth statistics (JSON per interface)
    bandwidth_stats = Column(JSON)
    
    # Latency metrics
    google_dns_latency_ms = Column(Float)
    cloudflare_dns_latency_ms = Column(Float)
    local_gateway_latency_ms = Column(Float)
    
    # Packet statistics
    packet_stats = Column(JSON)
    
    # Raw data for AI analysis
    raw_data = Column(JSON)
    
    # Relationships
    agent = relationship("Agent", back_populates="metrics")

class Alert(Base):
    """Anomaly detection alerts"""
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False, index=True)
    timestamp = Column(DateTime, default=local_now, index=True)
    
    # Alert information
    alert_type = Column(String(50), nullable=False)  # anomaly, threshold, system_error
    severity = Column(String(20), default="medium")  # low, medium, high, critical
    title = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Alert data
    metric_name = Column(String(100))
    metric_value = Column(Float)
    threshold_value = Column(Float)
    anomaly_score = Column(Float)
    
    # Status
    status = Column(String(20), default="active")  # active, acknowledged, resolved
    acknowledged_at = Column(DateTime)
    resolved_at = Column(DateTime)
    
    # Additional data
    context = Column(JSON)  # Additional context from AI analysis
    
    # Relationships
    agent = relationship("Agent", back_populates="alerts")

class AIAnalysis(Base):
    """AI analysis results"""
    __tablename__ = "ai_analysis"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False, index=True)
    timestamp = Column(DateTime, default=local_now, index=True)
    
    # Analysis information
    analysis_type = Column(String(50))  # anomaly_detection, trend_analysis, prediction
    model_used = Column(String(100))
    confidence_score = Column(Float)
    
    # Results
    findings = Column(JSON)  # Structured findings
    recommendations = Column(JSON)  # AI recommendations
    risk_assessment = Column(JSON)  # Risk levels and categories
    
    # Processing info
    processing_time_ms = Column(Integer)
    data_points_analyzed = Column(Integer)
    
    # Status
    status = Column(String(20), default="completed")  # pending, completed, failed

class SystemConfig(Base):
    """System configuration and settings"""
    __tablename__ = "system_config"
    
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(100), unique=True, nullable=False, index=True)
    value = Column(JSON)
    description = Column(Text)
    created_at = Column(DateTime, default=local_now)
    updated_at = Column(DateTime, default=local_now, onupdate=local_now)

class ApiKey(Base):
    """API keys for agent authentication"""
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    key_hash = Column(String(255), nullable=False, unique=True, index=True)
    created_at = Column(DateTime, default=local_now)
    expires_at = Column(DateTime)
    is_active = Column(Boolean, default=True)
    permissions = Column(JSON)  # List of allowed operations
    agent_hostname = Column(String(255))  # Optional: restrict to specific agent
    
    # Usage tracking
    last_used_at = Column(DateTime)
    usage_count = Column(Integer, default=0)

class MetricsAggregation(Base):
    """Pre-aggregated metrics for dashboard performance"""
    __tablename__ = "metrics_aggregation"
    
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    aggregation_period = Column(String(20), nullable=False)  # minute, hour, day
    
    # Aggregated values
    avg_cpu_percent = Column(Float)
    max_cpu_percent = Column(Float)
    avg_memory_percent = Column(Float)
    max_memory_percent = Column(Float)
    avg_disk_percent = Column(Float)
    max_disk_percent = Column(Float)
    
    # Network aggregations
    total_bytes_sent = Column(Integer)
    total_bytes_recv = Column(Integer)
    avg_latency_ms = Column(Float)
    max_latency_ms = Column(Float)
    
    # Anomaly counts
    anomaly_count = Column(Integer, default=0)
    alert_count = Column(Integer, default=0)
    
    # Data points count
    data_points = Column(Integer)

class ChatCheckpoint(Base):
    """Stores chat checkpoints for the AI agent"""
    __tablename__ = "chat_checkpoints"

    thread_id = Column(String(255), primary_key=True, index=True)
    checkpoint = Column(JSON, nullable=False)
    updated_at = Column(DateTime, default=local_now, onupdate=local_now, index=True)
