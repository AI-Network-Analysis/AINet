"""
API Schemas
Pydantic models for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime

# Base schemas
class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str

# Agent schemas
class HeartbeatRequest(BaseModel):
    hostname: str
    timestamp: str
    agent_version: str = "1.0.0"
    status: str = "active"
    queue_size: int = 0

class HeartbeatResponse(BaseModel):
    status: str
    agent_id: int
    message: str

class AgentInfo(BaseModel):
    id: int
    hostname: str
    ip_address: Optional[str] = None
    mac_address: Optional[str] = None
    agent_version: Optional[str] = None
    first_seen: datetime
    last_seen: datetime
    status: str
    is_online: bool
    metrics_count: int

    class Config:
        from_attributes = True

# Metrics schemas
class MetricsRequest(BaseModel):
    timestamp: str
    hostname: str
    interfaces: Dict[str, Dict[str, Any]]
    connections: Dict[str, int]
    bandwidth: Dict[str, Dict[str, int]]
    system_metrics: Dict[str, float]
    latency_metrics: Dict[str, float]
    packet_stats: Dict[str, Dict[str, int]]

class MetricsResponse(BaseModel):
    status: str
    agent_id: int
    metric_id: int
    message: str

class BatchMetricsRequest(BaseModel):
    metrics: List[Dict[str, Any]]  # List of MetricsRequest dicts
    batch_size: int
    timestamp: str

class BatchMetricsResponse(BaseModel):
    status: str
    batch_size: int
    processed_count: int
    failed_count: int
    message: str

class MetricsSummary(BaseModel):
    id: int
    timestamp: datetime
    cpu_percent: Optional[float] = None
    memory_percent: Optional[float] = None
    disk_percent: Optional[float] = None
    total_connections: Optional[int] = None
    avg_latency_ms: Optional[float] = None

    class Config:
        from_attributes = True

class NetworkMetricsDetailed(BaseModel):
    id: int
    timestamp: datetime
    # System metrics
    cpu_percent: Optional[float] = None
    memory_percent: Optional[float] = None
    disk_percent: Optional[float] = None
    # Network metrics  
    interfaces: Optional[Dict[str, Any]] = None
    total_connections: Optional[int] = None
    tcp_established: Optional[int] = None
    tcp_listen: Optional[int] = None
    udp_connections: Optional[int] = None
    bandwidth_stats: Optional[Dict[str, Any]] = None
    packet_stats: Optional[Dict[str, Any]] = None
    # Latency metrics
    google_dns_latency_ms: Optional[float] = None
    cloudflare_dns_latency_ms: Optional[float] = None
    local_gateway_latency_ms: Optional[float] = None
    avg_latency_ms: Optional[float] = None

    class Config:
        from_attributes = True

# Alert schemas
class AlertInfo(BaseModel):
    id: int
    agent_hostname: str
    timestamp: datetime
    alert_type: str
    severity: str
    title: str
    description: Optional[str] = None
    status: str
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    anomaly_score: Optional[float] = None

    class Config:
        from_attributes = True

class CreateAlertRequest(BaseModel):
    agent_id: int
    alert_type: str
    severity: str = "medium"
    title: str
    description: Optional[str] = None
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    anomaly_score: Optional[float] = None
    context: Optional[Dict[str, Any]] = None

# Dashboard schemas
class DashboardOverview(BaseModel):
    total_agents: int
    active_agents: int
    total_alerts: int
    critical_alerts: int
    avg_cpu_usage: float
    avg_memory_usage: float
    avg_network_latency: float
    data_points_last_hour: int
    timestamp: datetime

class NetworkOverview(BaseModel):
    hostname: str
    status: str
    last_seen: datetime
    cpu_percent: Optional[float] = None
    memory_percent: Optional[float] = None
    disk_percent: Optional[float] = None
    total_connections: Optional[int] = None
    bandwidth_usage_mbps: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    active_alerts: int

# AI Analysis schemas
class AnomalyDetectionRequest(BaseModel):
    agent_id: int
    time_window_hours: int = 24
    metrics_to_analyze: List[str] = ["cpu_percent", "memory_percent", "network_latency"]
    sensitivity: float = 0.8

class AnomalyDetectionResponse(BaseModel):
    analysis_id: int
    agent_hostname: str
    anomalies_found: int
    confidence_score: float
    findings: List[Dict[str, Any]]
    recommendations: List[str]
    processing_time_ms: int
    status: str

# System configuration schemas
class ConfigUpdate(BaseModel):
    key: str
    value: Any
    description: Optional[str] = None

class ConfigResponse(BaseModel):
    key: str
    value: Any
    description: Optional[str] = None
    updated_at: datetime

# API Key management schemas
class CreateApiKeyRequest(BaseModel):
    name: str
    permissions: List[str] = ["metrics:write", "heartbeat:write"]
    agent_hostname: Optional[str] = None
    expires_days: Optional[int] = None

class ApiKeyResponse(BaseModel):
    id: int
    name: str
    key: str  # Only returned on creation
    created_at: datetime
    expires_at: Optional[datetime] = None
    permissions: List[str]
    agent_hostname: Optional[str] = None

class ApiKeyInfo(BaseModel):
    id: int
    name: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    is_active: bool
    permissions: List[str]
    agent_hostname: Optional[str] = None
    last_used_at: Optional[datetime] = None
    usage_count: int

    class Config:
        from_attributes = True
