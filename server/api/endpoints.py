"""
API Endpoints
FastAPI endpoints for receiving data from agents and serving dashboard.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Header
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import logging
import hashlib
import json

from ..database.connection import get_db
from ..database.models import Agent, NetworkMetric, Alert, ApiKey
from .schemas import *

logger = logging.getLogger(__name__)

# Create routers
api_router = APIRouter(prefix="/api/v1")
dashboard_router = APIRouter(prefix="/dashboard")

# Authentication dependency
async def get_current_agent(
    x_api_key: Optional[str] = Header(None),
    db: Session = Depends(get_db)
) -> Optional[Agent]:
    """Authenticate agent using API key"""
    if not x_api_key:
        # Allow access without API key for development/testing
        return None
    
    # Hash the provided key
    key_hash = hashlib.sha256(x_api_key.encode()).hexdigest()
    
    # Find API key in database
    api_key = db.query(ApiKey).filter(
        ApiKey.key_hash == key_hash,
        ApiKey.is_active == True
    ).first()
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    # Update usage statistics
    api_key.last_used_at = datetime.utcnow()
    api_key.usage_count += 1
    db.commit()
    
    # If API key is restricted to a specific agent, find that agent
    if api_key.agent_hostname:
        agent = db.query(Agent).filter(
            Agent.hostname == api_key.agent_hostname
        ).first()
        return agent
    
    return None

# Health check endpoint
@api_router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="1.0.0"
    )

# Agent heartbeat endpoint
@api_router.post("/heartbeat", response_model=HeartbeatResponse)
async def agent_heartbeat(
    heartbeat: HeartbeatRequest,
    db: Session = Depends(get_db),
    current_agent: Optional[Agent] = Depends(get_current_agent)
):
    """Receive agent heartbeat"""
    try:
        # Find or create agent
        agent = db.query(Agent).filter(Agent.hostname == heartbeat.hostname).first()
        
        if not agent:
            agent = Agent(
                hostname=heartbeat.hostname,
                agent_version=heartbeat.agent_version,
                first_seen=datetime.utcnow(),
                status="active"
            )
            db.add(agent)
            logger.info(f"New agent registered: {heartbeat.hostname}")
        
        # Update agent status
        agent.last_seen = datetime.utcnow()
        agent.agent_version = heartbeat.agent_version
        agent.status = heartbeat.status
        
        db.commit()
        
        return HeartbeatResponse(
            status="received",
            agent_id=agent.id,
            message="Heartbeat processed successfully"
        )
        
    except Exception as e:
        logger.error(f"Error processing heartbeat from {heartbeat.hostname}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process heartbeat"
        )

# Single metrics endpoint
@api_router.post("/metrics", response_model=MetricsResponse)
async def receive_metrics(
    metrics: MetricsRequest,
    db: Session = Depends(get_db),
    current_agent: Optional[Agent] = Depends(get_current_agent)
):
    """Receive network metrics from agent"""
    try:
        # Find or create agent
        agent = db.query(Agent).filter(Agent.hostname == metrics.hostname).first()
        
        if not agent:
            agent = Agent(
                hostname=metrics.hostname,
                first_seen=datetime.utcnow(),
                status="active"
            )
            db.add(agent)
            db.flush()  # Get agent ID
        
        # Update agent last seen
        agent.last_seen = datetime.utcnow()
        
        # Create network metric record
        network_metric = NetworkMetric(
            agent_id=agent.id,
            timestamp=datetime.fromisoformat(metrics.timestamp.replace('Z', '+00:00')),
            
            # System metrics
            cpu_percent=metrics.system_metrics.get('cpu_percent'),
            memory_percent=metrics.system_metrics.get('memory_percent'),
            memory_total=metrics.system_metrics.get('memory_total'),
            memory_used=metrics.system_metrics.get('memory_used'),
            memory_available=metrics.system_metrics.get('memory_available'),
            disk_percent=metrics.system_metrics.get('disk_percent'),
            disk_total=metrics.system_metrics.get('disk_total'),
            disk_used=metrics.system_metrics.get('disk_used'),
            disk_free=metrics.system_metrics.get('disk_free'),
            load_avg_1min=metrics.system_metrics.get('load_avg_1min'),
            load_avg_5min=metrics.system_metrics.get('load_avg_5min'),
            load_avg_15min=metrics.system_metrics.get('load_avg_15min'),
            
            # Network interface data
            interfaces=metrics.interfaces,
            
            # Connection statistics
            tcp_established=metrics.connections.get('tcp_established'),
            tcp_listen=metrics.connections.get('tcp_listen'),
            tcp_time_wait=metrics.connections.get('tcp_time_wait'),
            tcp_close_wait=metrics.connections.get('tcp_close_wait'),
            udp_connections=metrics.connections.get('udp_connections'),
            total_connections=metrics.connections.get('total_connections'),
            
            # Bandwidth statistics
            bandwidth_stats=metrics.bandwidth,
            
            # Latency metrics
            google_dns_latency_ms=metrics.latency_metrics.get('google_dns_latency_ms'),
            cloudflare_dns_latency_ms=metrics.latency_metrics.get('cloudflare_dns_latency_ms'),
            local_gateway_latency_ms=metrics.latency_metrics.get('local_gateway_latency_ms'),
            
            # Packet statistics
            packet_stats=metrics.packet_stats,
            
            # Store raw data for AI analysis
            raw_data=metrics.dict()
        )
        
        db.add(network_metric)
        db.commit()
        
        logger.debug(f"Stored metrics for agent {metrics.hostname}")
        
        return MetricsResponse(
            status="received",
            agent_id=agent.id,
            metric_id=network_metric.id,
            message="Metrics stored successfully"
        )
        
    except Exception as e:
        logger.error(f"Error storing metrics from {metrics.hostname}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store metrics"
        )

# Batch metrics endpoint
@api_router.post("/metrics/batch", response_model=BatchMetricsResponse)
async def receive_metrics_batch(
    batch: BatchMetricsRequest,
    db: Session = Depends(get_db),
    current_agent: Optional[Agent] = Depends(get_current_agent)
):
    """Receive batch of network metrics from agent"""
    try:
        processed_count = 0
        failed_count = 0
        
        for metrics_data in batch.metrics:
            try:
                # Convert dict to MetricsRequest if needed
                if isinstance(metrics_data, dict):
                    metrics = MetricsRequest(**metrics_data)
                else:
                    metrics = metrics_data
                
                # Process single metrics inline (can't call async function)
                # Find or create agent
                agent = db.query(Agent).filter(Agent.hostname == metrics.hostname).first()
                
                if not agent:
                    agent = Agent(
                        hostname=metrics.hostname,
                        first_seen=datetime.utcnow(),
                        status="active"
                    )
                    db.add(agent)
                    db.flush()  # Get agent ID
                
                # Update agent last seen
                agent.last_seen = datetime.utcnow()
                
                # Create network metric record
                network_metric = NetworkMetric(
                    agent_id=agent.id,
                    timestamp=datetime.fromisoformat(metrics.timestamp.replace('Z', '+00:00')),
                    
                    # System metrics
                    cpu_percent=metrics.system_metrics.get('cpu_percent'),
                    memory_percent=metrics.system_metrics.get('memory_percent'),
                    memory_total=metrics.system_metrics.get('memory_total'),
                    memory_used=metrics.system_metrics.get('memory_used'),
                    memory_available=metrics.system_metrics.get('memory_available'),
                    disk_percent=metrics.system_metrics.get('disk_percent'),
                    disk_total=metrics.system_metrics.get('disk_total'),
                    disk_used=metrics.system_metrics.get('disk_used'),
                    disk_free=metrics.system_metrics.get('disk_free'),
                    load_avg_1min=metrics.system_metrics.get('load_avg_1min'),
                    load_avg_5min=metrics.system_metrics.get('load_avg_5min'),
                    load_avg_15min=metrics.system_metrics.get('load_avg_15min'),
                    
                    # Network interface data
                    interfaces=metrics.interfaces,
                    
                    # Connection statistics
                    tcp_established=metrics.connections.get('tcp_established'),
                    tcp_listen=metrics.connections.get('tcp_listen'),
                    tcp_time_wait=metrics.connections.get('tcp_time_wait'),
                    tcp_close_wait=metrics.connections.get('tcp_close_wait'),
                    udp_connections=metrics.connections.get('udp_connections'),
                    total_connections=metrics.connections.get('total_connections'),
                    
                    # Bandwidth statistics
                    bandwidth_stats=metrics.bandwidth,
                    
                    # Latency metrics
                    google_dns_latency_ms=metrics.latency_metrics.get('google_dns_latency_ms'),
                    cloudflare_dns_latency_ms=metrics.latency_metrics.get('cloudflare_dns_latency_ms'),
                    local_gateway_latency_ms=metrics.latency_metrics.get('local_gateway_latency_ms'),
                    
                    # Packet statistics
                    packet_stats=metrics.packet_stats,
                    
                    # Store raw data for AI analysis
                    raw_data=metrics.dict()
                )
                
                db.add(network_metric)
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Failed to process metrics in batch: {e}")
                failed_count += 1
        
        # Commit all changes
        db.commit()
        
        return BatchMetricsResponse(
            status="processed",
            batch_size=batch.batch_size,
            processed_count=processed_count,
            failed_count=failed_count,
            message=f"Processed {processed_count}/{batch.batch_size} metrics successfully"
        )
        
    except Exception as e:
        logger.error(f"Error processing metrics batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process metrics batch"
        )

# Agent management endpoints
@api_router.get("/agents", response_model=List[AgentInfo])
async def get_agents(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get list of registered agents"""
    try:
        agents = db.query(Agent).offset(skip).limit(limit).all()
        
        agent_list = []
        for agent in agents:
            # Get latest metrics count
            metrics_count = db.query(NetworkMetric).filter(
                NetworkMetric.agent_id == agent.id
            ).count()
            
            # Check if agent is online (last seen within 5 minutes)
            is_online = (datetime.utcnow() - agent.last_seen).total_seconds() < 300
            
            agent_info = AgentInfo(
                id=agent.id,
                hostname=agent.hostname,
                ip_address=agent.ip_address,
                mac_address=agent.mac_address,
                agent_version=agent.agent_version,
                first_seen=agent.first_seen,
                last_seen=agent.last_seen,
                status=agent.status,
                is_online=is_online,
                metrics_count=metrics_count
            )
            agent_list.append(agent_info)
        
        return agent_list
        
    except Exception as e:
        logger.error(f"Error retrieving agents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve agents"
        )

@api_router.get("/agents/{agent_id}/metrics", response_model=List[MetricsSummary])
async def get_agent_metrics(
    agent_id: int,
    hours: int = 24,
    db: Session = Depends(get_db)
):
    """Get recent metrics for a specific agent"""
    try:
        # Check if agent exists
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent not found"
            )
        
        # Get metrics from the last N hours
        since = datetime.utcnow() - timedelta(hours=hours)
        metrics = db.query(NetworkMetric).filter(
            NetworkMetric.agent_id == agent_id,
            NetworkMetric.timestamp >= since
        ).order_by(NetworkMetric.timestamp.desc()).all()
        
        metrics_list = []
        for metric in metrics:
            summary = MetricsSummary(
                id=metric.id,
                timestamp=metric.timestamp,
                cpu_percent=metric.cpu_percent,
                memory_percent=metric.memory_percent,
                disk_percent=metric.disk_percent,
                total_connections=metric.total_connections,
                avg_latency_ms=(
                    (metric.google_dns_latency_ms or 0) +
                    (metric.cloudflare_dns_latency_ms or 0) +
                    (metric.local_gateway_latency_ms or 0)
                ) / 3 if any([
                    metric.google_dns_latency_ms,
                    metric.cloudflare_dns_latency_ms,
                    metric.local_gateway_latency_ms
                ]) else None
            )
            metrics_list.append(summary)
        
        return metrics_list
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving metrics for agent {agent_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve metrics"
        )

# Alerts endpoints
@api_router.get("/alerts", response_model=List[AlertInfo])
async def get_alerts(
    skip: int = 0,
    limit: int = 100,
    severity: Optional[str] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get list of alerts"""
    try:
        query = db.query(Alert)
        
        if severity:
            query = query.filter(Alert.severity == severity)
        if status:
            query = query.filter(Alert.status == status)
        
        alerts = query.order_by(Alert.timestamp.desc()).offset(skip).limit(limit).all()
        
        alert_list = []
        for alert in alerts:
            alert_info = AlertInfo(
                id=alert.id,
                agent_hostname=alert.agent.hostname,
                timestamp=alert.timestamp,
                alert_type=alert.alert_type,
                severity=alert.severity,
                title=alert.title,
                description=alert.description,
                status=alert.status,
                metric_name=alert.metric_name,
                metric_value=alert.metric_value,
                anomaly_score=alert.anomaly_score
            )
            alert_list.append(alert_info)
        
        return alert_list
        
    except Exception as e:
        logger.error(f"Error retrieving alerts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve alerts"
        )
