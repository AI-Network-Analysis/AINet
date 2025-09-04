"""
API Endpoints
FastAPI endpoints for receiving data from agents and serving dashboard.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Header, Request
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

# ---- Helpers ----
def _parse_timestamp(ts: Any) -> datetime:
    """Parse timestamp from various formats (ISO string, epoch str/float/int)."""
    try:
        if ts is None:
            return datetime.utcnow()
        # If already datetime, return as-is
        if isinstance(ts, datetime):
            return ts
        # Try numeric epoch (string or number)
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(float(ts))
        if isinstance(ts, str):
            s = ts.strip()
            # Numeric string epoch
            if s.replace(".", "", 1).isdigit():
                return datetime.fromtimestamp(float(s))
            # ISO 8601 (allow trailing Z)
            try:
                return datetime.fromisoformat(s.replace('Z', '+00:00'))
            except Exception:
                pass
        # Fallback to now if all else fails
        return datetime.utcnow()
    except Exception:
        return datetime.utcnow()

def _normalize_connections(conn: Optional[Dict[str, Any]]) -> Dict[str, int]:
    """Normalize various connection stats shapes to the expected keys.
    Supports keys from older/newer agents.
    """
    conn = conn or {}
    def _get(*keys, default=0):
        for k in keys:
            v = conn.get(k)
            if isinstance(v, (int, float)):
                return int(v)
        return default

    total = _get('total_connections', 'total', 'count', default=0)
    tcp_established = _get('tcp_established', 'established', 'tcp_established_count', default=0)
    tcp_listen = _get('tcp_listen', 'listen', 'listening', default=0)
    tcp_time_wait = _get('tcp_time_wait', 'time_wait', default=0)
    tcp_close_wait = _get('tcp_close_wait', 'close_wait', default=0)
    udp = _get('udp_connections', 'udp', default=0)

    # If only aggregate tcp is provided, apportion known subtypes and set others to 0
    tcp_total = _get('tcp', default=tcp_established + tcp_listen)
    if tcp_total and not (tcp_established or tcp_listen):
        # We don't know distribution; keep tcp_established as tcp_total, others 0
        tcp_established = tcp_total

    # Recompute total if not provided
    if not total:
        total = tcp_established + tcp_listen + udp

    return {
        'total_connections': total,
        'tcp_established': tcp_established,
        'tcp_listen': tcp_listen,
        'tcp_time_wait': tcp_time_wait,
        'tcp_close_wait': tcp_close_wait,
        'udp_connections': udp,
    }

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
                first_seen=datetime.now(),
                status="active"
            )
            db.add(agent)
            logger.info(f"New agent registered: {heartbeat.hostname}")
        
        # Update agent status
        agent.last_seen = datetime.now()
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
                first_seen=datetime.now(),
                status="active"
            )
            db.add(agent)
            db.flush()  # Get agent ID
        
        # Update agent last seen
        agent.last_seen = datetime.now()
        
        # Normalize timestamp and connections from various agent versions
        parsed_ts = _parse_timestamp(metrics.timestamp)
        conn = _normalize_connections(metrics.connections)

        # Create network metric record
        network_metric = NetworkMetric(
            agent_id=agent.id,
            timestamp=parsed_ts,
            
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
            tcp_established=conn.get('tcp_established'),
            tcp_listen=conn.get('tcp_listen'),
            tcp_time_wait=conn.get('tcp_time_wait'),
            tcp_close_wait=conn.get('tcp_close_wait'),
            udp_connections=conn.get('udp_connections'),
            total_connections=conn.get('total_connections'),
            
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

        # Simple DoS/MITM-lite alerting
        try:
            from ..ai_engine.chat_agent import local_now
            # Load config thresholds from monitoring config
            try:
                from config.monitoring_config import ConfigurationManager
                _cfg = ConfigurationManager().get_config()
                conn_warn = float(_cfg.network.conn_rate_warning_threshold)
                conn_crit = float(_cfg.network.conn_rate_critical_threshold)
                recv_spike_factor = float(_cfg.network.recv_rate_spike_factor)
                detect_arp_change = bool(_cfg.network.detect_arp_gateway_change)
            except Exception:
                # Safe fallbacks
                conn_warn = 50.0
                conn_crit = 200.0
                recv_spike_factor = 3.0
                detect_arp_change = True

            # 1) Connection rate alert
            new_cps = None
            try:
                new_cps = metrics.system_metrics.get('new_connections_per_s')
            except Exception:
                pass
            if isinstance(new_cps, (int, float)) and new_cps is not None:
                severity = None
                if new_cps >= conn_crit:
                    severity = 'critical'
                elif new_cps >= conn_warn:
                    severity = 'high'
                if severity:
                    alert = Alert(
                        agent_id=agent.id,
                        timestamp=local_now(),
                        alert_type='dos_suspected',
                        severity=severity,
                        title='High connection rate detected',
                        description=f'new_connections_per_s={new_cps:.2f} (warn>{conn_warn}, crit>{conn_crit})',
                        metric_name='new_connections_per_s',
                        metric_value=float(new_cps),
                        anomaly_score=None,
                        status='active',
                        context={'thresholds': {'warn': conn_warn, 'crit': conn_crit}}
                    )
                    db.add(alert)

            # 2) Receive rate spike vs baseline (simple moving avg of last 10)
            try:
                # Compute aggregate recv rate from interfaces if present
                agg_recv_rate = 0.0
                if isinstance(metrics.interfaces, dict):
                    for val in metrics.interfaces.values():
                        v = val.get('bytes_recv_rate')
                        if isinstance(v, (int, float)):
                            agg_recv_rate += float(v)

                # Get previous 10 metrics for baseline
                prev_metrics = (
                    db.query(NetworkMetric)
                    .filter(NetworkMetric.agent_id == agent.id)
                    .order_by(NetworkMetric.timestamp.desc())
                    .limit(10)
                    .all()
                )
                baseline = 0.0
                count = 0
                for m in prev_metrics:
                    if isinstance(m.interfaces, dict):
                        r = 0.0
                        for val in m.interfaces.values():
                            v = (val or {}).get('bytes_recv_rate')
                            if isinstance(v, (int, float)):
                                r += float(v)
                        baseline += r
                        count += 1
                baseline = (baseline / count) if count > 0 else 0.0
                if baseline > 0 and agg_recv_rate > baseline * float(recv_spike_factor):
                    alert = Alert(
                        agent_id=agent.id,
                        timestamp=local_now(),
                        alert_type='bandwidth_spike',
                        severity='high',
                        title='High receive bandwidth spike',
                        description=f'bytes_recv_rate agg={agg_recv_rate:.0f}B/s baseline~{baseline:.0f}B/s factor>{recv_spike_factor}',
                        metric_name='bytes_recv_rate',
                        metric_value=float(agg_recv_rate),
                        anomaly_score=None,
                        status='active',
                        context={'baseline': baseline, 'factor': recv_spike_factor}
                    )
                    db.add(alert)
            except Exception:
                pass

            # 3) ARP gateway MAC change: compare last two raw_data snapshots
            try:
                if detect_arp_change and isinstance(metrics.interfaces, dict):
                    # Get previous metric for this agent
                    prev = (
                        db.query(NetworkMetric)
                        .filter(NetworkMetric.agent_id == agent.id, NetworkMetric.id != network_metric.id)
                        .order_by(NetworkMetric.timestamp.desc())
                        .first()
                    )
                    def gateway_mac_from_interfaces(intf_map):
                        # Heuristic: list arp_neighbors for any iface; pick entry where IP equals default gateway from latency metrics if present
                        if not isinstance(intf_map, dict):
                            return None
                        # Flatten neighbors
                        for val in intf_map.values():
                            neigh = (val or {}).get('arp_neighbors')
                            if isinstance(neigh, list):
                                # If any entry has flags '0x2' it is a complete entry
                                for n in neigh:
                                    if isinstance(n, dict) and n.get('mac') and n.get('flags'):
                                        return n.get('mac')
                        return None
                    prev_mac = gateway_mac_from_interfaces(prev.interfaces) if prev else None
                    curr_mac = gateway_mac_from_interfaces(metrics.interfaces)
                    if prev_mac and curr_mac and prev_mac != curr_mac:
                        alert = Alert(
                            agent_id=agent.id,
                            timestamp=local_now(),
                            alert_type='arp_gateway_change',
                            severity='high',
                            title='Gateway MAC address changed',
                            description=f'Previous MAC {prev_mac} -> Current MAC {curr_mac}',
                            metric_name='arp_gateway_mac',
                            metric_value=None,
                            anomaly_score=None,
                            status='active',
                            context={'prev_mac': prev_mac, 'curr_mac': curr_mac}
                        )
                        db.add(alert)

            except Exception:
                pass

            # Commit alerts (if any)
            db.commit()
        except Exception:
            # Non-fatal; metrics are stored even if alerts fail
            pass
        
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
                        first_seen=datetime.now(),
                        status="active"
                    )
                    db.add(agent)
                    db.flush()  # Get agent ID
                
                # Update agent last seen
                agent.last_seen = datetime.now()
                
                # Normalize timestamp and connections
                parsed_ts = _parse_timestamp(metrics.timestamp)
                conn = _normalize_connections(metrics.connections)

                # Create network metric record
                network_metric = NetworkMetric(
                    agent_id=agent.id,
                    timestamp=parsed_ts,
                    
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
                    tcp_established=conn.get('tcp_established'),
                    tcp_listen=conn.get('tcp_listen'),
                    tcp_time_wait=conn.get('tcp_time_wait'),
                    tcp_close_wait=conn.get('tcp_close_wait'),
                    udp_connections=conn.get('udp_connections'),
                    total_connections=conn.get('total_connections'),
                    
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
            is_online = (datetime.now() - agent.last_seen).total_seconds() < 300
            
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

@api_router.get("/agents/{agent_id}/metrics/detailed", response_model=List[NetworkMetricsDetailed])
async def get_agent_metrics_detailed(
    agent_id: int,
    hours: int = 72,
    limit: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get detailed network metrics for a specific agent"""
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
        query = db.query(NetworkMetric).filter(
            NetworkMetric.agent_id == agent_id,
            NetworkMetric.timestamp >= since
        ).order_by(NetworkMetric.timestamp.desc())

        if limit is not None and isinstance(limit, int) and limit > 0:
            metrics = query.limit(limit).all()
        else:
            metrics = query.all()
        
        metrics_list = []
        for metric in metrics:
            detailed = NetworkMetricsDetailed(
                id=metric.id,
                timestamp=metric.timestamp,
                cpu_percent=metric.cpu_percent,
                memory_percent=metric.memory_percent,
                disk_percent=metric.disk_percent,
                interfaces=metric.interfaces,
                total_connections=metric.total_connections,
                tcp_established=metric.tcp_established,
                tcp_listen=metric.tcp_listen,
                udp_connections=metric.udp_connections,
                bandwidth_stats=metric.bandwidth_stats,
                packet_stats=metric.packet_stats,
                google_dns_latency_ms=metric.google_dns_latency_ms,
                cloudflare_dns_latency_ms=metric.cloudflare_dns_latency_ms,
                local_gateway_latency_ms=metric.local_gateway_latency_ms,
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
            metrics_list.append(detailed)
        
        return metrics_list
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving detailed metrics for agent {agent_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve detailed metrics"
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

@api_router.post("/analyze/agent/{agent_id}", response_model=Dict[str, Any])
async def analyze_single_agent(
    agent_id: int,
    request: Request,
    time_window_hours: int = 24,
    force: bool = False,
    wait_if_in_progress: bool = False,
    db: Session = Depends(get_db)
):
    """Trigger AI anomaly analysis for a specific agent"""
    try:
        ai_service = request.app.state.ai_service
        # Optionally wait for current analysis to complete
        if wait_if_in_progress and agent_id in ai_service.analysis_in_progress and not force:
            # simple wait loop up to 30s
            from datetime import datetime, timedelta
            import asyncio as _asyncio
            end_time = datetime.utcnow() + timedelta(seconds=30)
            while agent_id in ai_service.analysis_in_progress and datetime.utcnow() < end_time:
                await _asyncio.sleep(0.5)

        analysis = await ai_service.analyze_agent_anomalies(
            agent_id, time_window_hours, force_analysis=force
        )
        if not analysis:
            raise HTTPException(status_code=400, detail="Analysis skipped or failed")
        return {"status": "completed", "analysis_id": analysis.id}
    except Exception as e:
        logger.error(f"Error analyzing agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed")

@api_router.post("/analyze/all", response_model=Dict[str, Any])
async def analyze_all_agents_endpoint(
    request: Request,
    time_window_hours: int = 24,
    force: bool = False,
    wait_if_in_progress: bool = False
):
    """Trigger AI anomaly analysis for all active agents"""
    try:
        ai_service = request.app.state.ai_service
        results = await ai_service.analyze_all_agents(
            time_window_hours, force_analysis=force, wait_if_in_progress=wait_if_in_progress
        )
        return results
    except Exception as e:
        logger.error(f"Error analyzing all agents: {e}")
        raise HTTPException(status_code=500, detail="Batch analysis failed")

@api_router.get("/analysis", response_model=List[Dict[str, Any]])
async def get_analysis_results(
    limit: int = 10,
    agent_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get recent AI analysis results"""
    try:
        from ..database.models import AIAnalysis
        
        query = db.query(AIAnalysis)
        
        if agent_id:
            query = query.filter(AIAnalysis.agent_id == agent_id)
        
        analyses = query.order_by(AIAnalysis.timestamp.desc()).limit(limit).all()
        
        results = []
        for analysis in analyses:
            # Get agent hostname
            agent = db.query(Agent).filter(Agent.id == analysis.agent_id).first()
            agent_hostname = agent.hostname if agent else f"Agent {analysis.agent_id}"
            
            result = {
                "id": analysis.id,
                "agent_id": analysis.agent_id,
                "agent_hostname": agent_hostname,
                "timestamp": analysis.timestamp,
                "analysis_type": analysis.analysis_type,
                "model_used": analysis.model_used,
                "confidence_score": analysis.confidence_score,
                "findings": analysis.findings,
                "recommendations": analysis.recommendations,
                "risk_assessment": analysis.risk_assessment,
                "processing_time_ms": analysis.processing_time_ms,
                "data_points_analyzed": analysis.data_points_analyzed,
                "status": analysis.status
            }
            results.append(result)
        
        return results
    except Exception as e:
        logger.error(f"Error retrieving analysis results: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve analysis results"
        )

@api_router.get("/analysis/{analysis_id}", response_model=Dict[str, Any])
async def get_analysis_result(
    analysis_id: int,
    db: Session = Depends(get_db)
):
    """Get specific AI analysis result by ID"""
    try:
        from ..database.models import AIAnalysis
        
        analysis = db.query(AIAnalysis).filter(AIAnalysis.id == analysis_id).first()
        
        if not analysis:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis result not found"
            )
        
        # Get agent hostname
        agent = db.query(Agent).filter(Agent.id == analysis.agent_id).first()
        agent_hostname = agent.hostname if agent else f"Agent {analysis.agent_id}"
        
        return {
            "id": analysis.id,
            "agent_id": analysis.agent_id,
            "agent_hostname": agent_hostname,
            "timestamp": analysis.timestamp,
            "analysis_type": analysis.analysis_type,
            "model_used": analysis.model_used,
            "confidence_score": analysis.confidence_score,
            "findings": analysis.findings,
            "recommendations": analysis.recommendations,
            "risk_assessment": analysis.risk_assessment,
            "processing_time_ms": analysis.processing_time_ms,
            "data_points_analyzed": analysis.data_points_analyzed,
            "status": analysis.status
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving analysis result {analysis_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve analysis result"
        )

# ============== CHAT ENDPOINTS ==============

class ChatRequest(BaseModel):
    """Request model for chat interactions"""
    message: str = Field(..., description="User message to the AI agent")
    session_id: Optional[str] = Field(None, description="Optional session ID for conversation continuity")

class ChatResponse(BaseModel):
    """Response model for chat interactions"""
    response: str = Field(..., description="AI agent response")
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description="List of tool calls made by the agent")
    session_id: str = Field(..., description="Session ID for conversation continuity")

@api_router.post("/chat", response_model=ChatResponse)
async def chat_with_agent(
    request: ChatRequest,
    db: Session = Depends(get_db)
):
    """
    Chat with the AI agent that has access to system data and tools.
    
    The agent can:
    - Answer questions about system status
    - Retrieve agent information and metrics
    - Check recent alerts and analysis results
    - Trigger new analysis on demand
    """
    try:
        from ..ai_engine.chat_agent import get_chat_agent
        
        # Create a basic config for the chat agent
        config = {
            'ai': {
                'model_name': 'gemini-2.0-flash-exp',
                'temperature': 0.1,
                'max_tokens': 1000
            }
        }
        
        # Get chat agent
        chat_agent = get_chat_agent(config)
        
        # Process the chat request with session-based memory
        thread_id = request.session_id or "default"
        result = await chat_agent.chat(
            message=request.message,
            thread_id=thread_id
        )
        
        return ChatResponse(
            response=result["response"],
            tool_calls=result.get("tool_calls", []),
            session_id=request.session_id or "default"
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat processing failed: {str(e)}"
        )

@api_router.get("/chat/status")
async def get_chat_status():
    """Get the status of the chat agent system"""
    try:
        from ..ai_engine.chat_agent import AINetChatAgent
        from ..main import config as app_config

        # Test agent initialization
        chat_agent = AINetChatAgent(app_config)

        return {
            "status": "online",
            "available_tools": [tool.name for tool in chat_agent.tools],
            "model": chat_agent.llm.model,
            "temperature": chat_agent.llm.temperature,
        }
    except Exception as e:
        logger.error(f"Error checking chat status: {e}")
        return {"status": "error", "error": str(e)}

@api_router.get("/chat/conversations")
async def get_chat_conversations(db: Session = Depends(get_db)):
    """Get a list of all chat conversations (thread IDs)"""
    try:
        from ..database.models import ChatCheckpoint
        import json
        
        conversations = db.query(ChatCheckpoint).order_by(ChatCheckpoint.updated_at.desc()).all()
        
        result = []
        for conv in conversations:
            # Get the first user message as preview
            checkpoint_data = conv.checkpoint
            
            # Handle both JSON object and JSON string formats
            if isinstance(checkpoint_data, str):
                try:
                    checkpoint_data = json.loads(checkpoint_data)
                except json.JSONDecodeError:
                    checkpoint_data = {}
            elif checkpoint_data is None:
                checkpoint_data = {}
            
            messages = checkpoint_data.get("messages", [])
            
            preview = "New conversation"
            last_message_time = conv.updated_at
            
            if messages:
                # Find first user message for preview
                for msg in messages:
                    if isinstance(msg, dict) and msg.get("type") == "human":
                        preview = msg.get("content", "")[:100]
                        if len(msg.get("content", "")) > 100:
                            preview += "..."
                        break
                
                # Get timestamp from last message
                if messages:
                    last_msg = messages[-1]
                    if isinstance(last_msg, dict) and "timestamp" in last_msg:
                        try:
                            from datetime import datetime
                            last_message_time = datetime.fromisoformat(last_msg["timestamp"])
                        except:
                            pass
            
            result.append({
                "thread_id": conv.thread_id,
                "preview": preview,
                "message_count": len(messages),
                "last_updated": last_message_time.isoformat(),
                "updated_at": conv.updated_at.isoformat()
            })
        
        return {
            "conversations": result,
            "total": len(result)
        }
        
    except Exception as e:
        logger.error(f"Error getting chat conversations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get conversations: {str(e)}"
        )

@api_router.get("/chat/conversations/{thread_id}")
async def get_chat_conversation(thread_id: str):
    """Get conversation history for a specific thread"""
    try:
        from ..ai_engine.chat_agent import get_chat_agent
        
        # Create a basic config for the chat agent
        config = {
            'ai': {
                'model_name': 'gemini-2.0-flash-exp',
                'temperature': 0.1,
                'max_tokens': 1000
            }
        }
        
        # Get chat agent
        chat_agent = get_chat_agent(config)
        
        # Get conversation history
        history = chat_agent.get_conversation_history(thread_id)
        
        return {
            "thread_id": thread_id,
            "messages": history,
            "message_count": len(history)
        }
        
    except Exception as e:
        logger.error(f"Error getting conversation {thread_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get conversation: {str(e)}"
        )

@api_router.delete("/chat/conversations/{thread_id}")
async def delete_chat_conversation(thread_id: str):
    """Delete a specific conversation"""
    try:
        from ..ai_engine.chat_agent import get_chat_agent
        
        # Create a basic config for the chat agent
        config = {
            'ai': {
                'model_name': 'gemini-2.0-flash-exp',
                'temperature': 0.1,
                'max_tokens': 1000
            }
        }
        
        # Get chat agent
        chat_agent = get_chat_agent(config)
        
        # Clear conversation
        success = chat_agent.clear_conversation(thread_id)
        
        if success:
            return {"message": f"Conversation {thread_id} deleted successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete conversation"
            )
        
    except Exception as e:
        logger.error(f"Error deleting conversation {thread_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete conversation: {str(e)}"
        )

# Configuration endpoints
@api_router.get("/config/monitoring", response_model=Dict[str, Any])
async def get_monitoring_config():
    """Get current monitoring configuration"""
    try:
        from config.monitoring_config import ConfigurationManager
        
        config_manager = ConfigurationManager()
        config = config_manager.get_config()
        
        return {
            "mode": config.mode.value,
            "collection_interval": config.collection_interval,
            "retention_days": config.retention_days,
            "ai_analysis_enabled": config.ai_analysis_enabled,
            "ai_analysis_interval": config.ai_analysis_interval,
            "network": {
                "enabled": config.network.enabled,
                "anomaly_detection": config.network.anomaly_detection,
                "bandwidth_monitoring": config.network.bandwidth_monitoring,
                "connection_monitoring": config.network.connection_monitoring,
                "latency_monitoring": config.network.latency_monitoring,
                "packet_loss_monitoring": config.network.packet_loss_monitoring,
                "sensitivity": config.network.sensitivity,
                "baseline_samples": config.network.baseline_samples,
                "history_size": config.network.history_size
            },
            "system": {
                "enabled": config.system.enabled,
                "cpu_monitoring": config.system.cpu_monitoring,
                "memory_monitoring": config.system.memory_monitoring,
                "disk_monitoring": config.system.disk_monitoring,
                "process_monitoring": config.system.process_monitoring,
                "cpu_warning_threshold": config.system.cpu_warning_threshold,
                "cpu_critical_threshold": config.system.cpu_critical_threshold,
                "memory_warning_threshold": config.system.memory_warning_threshold,
                "memory_critical_threshold": config.system.memory_critical_threshold,
                "disk_warning_threshold": config.system.disk_warning_threshold,
                "disk_critical_threshold": config.system.disk_critical_threshold
            }
        }
    except Exception as e:
        logger.error(f"Error getting monitoring configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get configuration: {str(e)}"
        )

@api_router.post("/config/monitoring", response_model=Dict[str, str])
async def update_monitoring_config(config_data: Dict[str, Any]):
    """Update monitoring configuration"""
    try:
        from config.monitoring_config import ConfigurationManager, MonitoringMode
        
        config_manager = ConfigurationManager()
        config = config_manager.get_config()
        
        # Update mode
        if "mode" in config_data:
            mode_value = config_data["mode"]
            if mode_value in [m.value for m in MonitoringMode]:
                config.mode = MonitoringMode(mode_value)
        
        # Update basic settings
        if "collection_interval" in config_data:
            config.collection_interval = config_data["collection_interval"]
        if "ai_analysis_enabled" in config_data:
            config.ai_analysis_enabled = config_data["ai_analysis_enabled"]
        
        # Update network settings
        if "network" in config_data:
            network_data = config_data["network"]
            for key, value in network_data.items():
                if hasattr(config.network, key):
                    setattr(config.network, key, value)
        
        # Update system settings
        if "system" in config_data:
            system_data = config_data["system"]
            for key, value in system_data.items():
                if hasattr(config.system, key):
                    setattr(config.system, key, value)
        
        # Apply mode-specific settings
        config._apply_mode_settings()
        
        # Save configuration
        config_manager.save_config(config)
        
        return {"message": "Configuration updated successfully"}
        
    except Exception as e:
        logger.error(f"Error updating monitoring configuration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update configuration: {str(e)}"
        )

@api_router.get("/config/presets", response_model=Dict[str, Any])
async def get_monitoring_presets():
    """Get available monitoring configuration presets"""
    try:
        from config.monitoring_config import ConfigurationManager
        
        config_manager = ConfigurationManager()
        presets = config_manager.get_monitoring_presets()
        
        return presets
        
    except Exception as e:
        logger.error(f"Error getting monitoring presets: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get presets: {str(e)}"
        )

@api_router.post("/config/presets/{preset_name}", response_model=Dict[str, str])
async def apply_monitoring_preset(preset_name: str):
    """Apply a monitoring configuration preset"""
    try:
        from config.monitoring_config import ConfigurationManager
        
        config_manager = ConfigurationManager()
        config_manager.apply_preset(preset_name)
        
        return {"message": f"Preset '{preset_name}' applied successfully"}
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error applying monitoring preset {preset_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to apply preset: {str(e)}"
        )