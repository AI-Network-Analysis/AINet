"""
AI Chat Agent with LangGraph Integration
Provides conversational interface with tool calling capabilities for system monitoring.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, TypedDict, Annotated
import asyncio

def local_now():
    """Return current local time instead of UTC"""
    return datetime.now()

from server.ai_engine.llm_provider import build_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from sqlalchemy.orm import Session
from sqlalchemy import desc

# Import database models and connection
from server.ai_engine.chat_memory import SQLAlchemyCheckpointer
from server.database.models import Agent, NetworkMetric, Alert, AIAnalysis, ChatCheckpoint
from server.database.connection import get_db, get_db_manager
from server.ai_engine.service import AIEngineService
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# State definition for chat workflow
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_calls: List[Dict[str, Any]]

class AgentStatusResponse(BaseModel):
    """Response model for agent status queries"""
    total_agents: int
    online_agents: int
    offline_agents: int
    agents: List[Dict[str, Any]]

class MetricsResponse(BaseModel):
    """Response model for metrics queries"""
    agent_hostname: str
    latest_metrics: Dict[str, Any]
    metric_count: int
    time_range: str

class AlertsResponse(BaseModel):
    """Response model for alerts queries"""
    total_alerts: int
    alerts: List[Dict[str, Any]]
    severity_breakdown: Dict[str, int]

class AnalysisResponse(BaseModel):
    """Response model for analysis queries"""
    total_analyses: int
    latest_analysis: Optional[Dict[str, Any]]
    analyses: List[Dict[str, Any]]

# Define tools that the agent can use
@tool
def get_system_status() -> Dict[str, Any]:
    """Get overall system status including server, database, and AI engine status."""
    try:
        db_manager = get_db_manager()
        db_healthy = db_manager.health_check()
        
        return {
            "status": "healthy" if db_healthy else "unhealthy",
            "database": "connected" if db_healthy else "disconnected",
            "ai_engine": "running",
            "timestamp": local_now().isoformat(),
            "message": "System is operational" if db_healthy else "System has issues"
        }
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return {
            "status": "error",
            "message": f"Failed to get system status: {str(e)}"
        }

@tool
def get_agents_status() -> AgentStatusResponse:
    """Get status of all agents including online/offline counts and details."""
    try:
        db_manager = get_db_manager()
        with db_manager.get_session() as db:
            agents = db.query(Agent).all()
            
            online_agents = 0
            offline_agents = 0
            agent_details = []
            
            for agent in agents:
                # Consider agent online if last seen within 5 minutes
                is_online = (local_now() - agent.last_seen).total_seconds() < 300
                
                if is_online:
                    online_agents += 1
                else:
                    offline_agents += 1
                
                agent_details.append({
                    "id": agent.id,
                    "hostname": agent.hostname,
                    "ip_address": agent.ip_address,
                    "status": "online" if is_online else "offline",
                    "last_seen": agent.last_seen.isoformat(),
                    "agent_version": agent.agent_version
                })
            
            return AgentStatusResponse(
                total_agents=len(agents),
                online_agents=online_agents,
                offline_agents=offline_agents,
                agents=agent_details
            )
    except Exception as e:
        logger.error(f"Error getting agents status: {e}")
        return AgentStatusResponse(
            total_agents=0,
            online_agents=0,
            offline_agents=0,
            agents=[]
        )

@tool
def get_agent_metrics(agent_hostname: str, hours: int = 24) -> MetricsResponse:
    """Get recent metrics for a specific agent.
    
    Args:
        agent_hostname: The hostname of the agent
        hours: How many hours back to look for metrics (default: 24)
    """
    try:
        db_manager = get_db_manager()
        with db_manager.get_session() as db:
            # Find agent
            agent = db.query(Agent).filter(Agent.hostname == agent_hostname).first()
            if not agent:
                return MetricsResponse(
                    agent_hostname=agent_hostname,
                    latest_metrics={},
                    metric_count=0,
                    time_range=f"last {hours} hours",
                )
            
            # Get recent metrics
            since_time = local_now() - timedelta(hours=hours)
            metrics = db.query(NetworkMetric).filter(
                NetworkMetric.agent_id == agent.id,
                NetworkMetric.timestamp >= since_time
            ).order_by(NetworkMetric.timestamp.desc()).all()
            
            if not metrics:
                return MetricsResponse(
                    agent_hostname=agent_hostname,
                    latest_metrics={},
                    metric_count=0,
                    time_range=f"last {hours} hours",
                )
            
            # Get latest metrics
            latest = metrics[0]
            latest_metrics = {
                "timestamp": latest.timestamp.isoformat(),
                "cpu_percent": latest.cpu_percent,
                "memory_percent": latest.memory_percent,
                "disk_percent": latest.disk_percent,
                "total_connections": latest.total_connections,
                "google_dns_latency_ms": latest.google_dns_latency_ms,
                "cloudflare_dns_latency_ms": latest.cloudflare_dns_latency_ms,
                "local_gateway_latency_ms": latest.local_gateway_latency_ms
            }
            
            return MetricsResponse(
                agent_hostname=agent_hostname,
                latest_metrics=latest_metrics,
                metric_count=len(metrics),
                time_range=f"last {hours} hours"
            )
    except Exception as e:
        logger.error(f"Error getting metrics for {agent_hostname}: {e}")
        return MetricsResponse(
            agent_hostname=agent_hostname,
            latest_metrics={},
            metric_count=0,
            time_range=f"last {hours} hours",
        )

# New network-focused tools
@tool
def get_network_overview(agent_hostname: str, hours: int = 24) -> Dict[str, Any]:
    """Get a high-level network overview for an agent: connection breakdown and average latencies.

    Args:
        agent_hostname: The agent hostname
        hours: Time window for averaging latencies and scanning metrics (default: 24)
    """
    try:
        db_manager = get_db_manager()
        with db_manager.get_session() as db:
            agent = db.query(Agent).filter(Agent.hostname == agent_hostname).first()
            if not agent:
                return {
                    "success": False,
                    "message": f"Agent '{agent_hostname}' not found"
                }

            since_time = local_now() - timedelta(hours=hours)
            metrics = (
                db.query(NetworkMetric)
                .filter(NetworkMetric.agent_id == agent.id, NetworkMetric.timestamp >= since_time)
                .order_by(NetworkMetric.timestamp.desc())
                .all()
            )

            if not metrics:
                return {
                    "success": True,
                    "agent_hostname": agent_hostname,
                    "connection_breakdown": {},
                    "latest_total_connections": 0,
                    "avg_latencies_ms": {},
                    "message": "No metrics in the selected time window"
                }

            latest = metrics[0]
            breakdown = {
                "tcp_established": int(latest.tcp_established or 0),
                "tcp_listen": int(latest.tcp_listen or 0),
                "tcp_time_wait": int(latest.tcp_time_wait or 0),
                "tcp_close_wait": int(latest.tcp_close_wait or 0),
                "udp_connections": int(latest.udp_connections or 0)
            }

            # Average latencies over available points in window
            lat_g_vals = [m.google_dns_latency_ms for m in metrics if isinstance(m.google_dns_latency_ms, (int, float))]
            lat_cf_vals = [m.cloudflare_dns_latency_ms for m in metrics if isinstance(m.cloudflare_dns_latency_ms, (int, float))]
            lat_gw_vals = [m.local_gateway_latency_ms for m in metrics if isinstance(m.local_gateway_latency_ms, (int, float))]

            def avg(vals: List[float]) -> Optional[float]:
                return round(sum(vals) / len(vals), 2) if vals else None

            avg_latencies = {
                "google_dns": avg(lat_g_vals),
                "cloudflare_dns": avg(lat_cf_vals),
                "local_gateway": avg(lat_gw_vals),
            }

            return {
                "success": True,
                "agent_hostname": agent_hostname,
                "timestamp": latest.timestamp.isoformat() if latest.timestamp else None,
                "latest_total_connections": int(latest.total_connections or 0),
                "connection_breakdown": breakdown,
                "avg_latencies_ms": avg_latencies,
            }
    except Exception as e:
        logger.error(f"Error in get_network_overview for {agent_hostname}: {e}")
        return {"success": False, "message": f"Failed to get network overview: {str(e)}"}


@tool
def get_interface_stats(agent_hostname: str, hours: int = 6, top_n: int = 5) -> Dict[str, Any]:
    """Get per-interface traffic rates (MB/s) using the last two samples in the time window.

    Args:
        agent_hostname: Agent hostname
        hours: Time window to search for samples (default: 6)
        top_n: Number of top interfaces to return by combined rate (default: 5)
    """
    try:
        db_manager = get_db_manager()
        with db_manager.get_session() as db:
            agent = db.query(Agent).filter(Agent.hostname == agent_hostname).first()
            if not agent:
                return {"success": False, "message": f"Agent '{agent_hostname}' not found"}

            since_time = local_now() - timedelta(hours=hours)
            # Get just the last two points for efficiency
            metrics = (
                db.query(NetworkMetric)
                .filter(NetworkMetric.agent_id == agent.id, NetworkMetric.timestamp >= since_time)
                .order_by(NetworkMetric.timestamp.desc())
                .limit(2)
                .all()
            )

            if not metrics:
                return {
                    "success": True,
                    "agent_hostname": agent_hostname,
                    "interfaces": [],
                    "message": "No interface data in the selected window"
                }

            latest = metrics[0]
            prev = metrics[1] if len(metrics) > 1 else None

            def mb_per_s(delta_bytes: float, seconds: float) -> float:
                if seconds <= 0:
                    return 0.0
                return round(max(0.0, delta_bytes) / seconds / (1024 * 1024), 3)

            rates = []
            if isinstance(latest.interfaces, dict):
                for name, iface in latest.interfaces.items():
                    try:
                        latest_sent = float(iface.get("bytes_sent", 0) or 0)
                        latest_recv = float(iface.get("bytes_recv", 0) or 0)

                        prev_sent = 0.0
                        prev_recv = 0.0
                        seconds = 0.0
                        if prev and isinstance(prev.interfaces, dict) and name in prev.interfaces:
                            prev_if = prev.interfaces.get(name) or {}
                            prev_sent = float(prev_if.get("bytes_sent", 0) or 0)
                            prev_recv = float(prev_if.get("bytes_recv", 0) or 0)
                            seconds = max(0.0, (latest.timestamp - prev.timestamp).total_seconds()) if latest.timestamp and prev.timestamp else 0.0

                        rate_sent = mb_per_s(latest_sent - prev_sent, seconds)
                        rate_recv = mb_per_s(latest_recv - prev_recv, seconds)
                        combined = round(rate_sent + rate_recv, 3)

                        rates.append({
                            "interface": name,
                            "rate_sent_mb_s": rate_sent,
                            "rate_recv_mb_s": rate_recv,
                            "rate_total_mb_s": combined
                        })
                    except Exception:
                        # Skip malformed interface entries
                        continue

            # Sort and trim
            rates.sort(key=lambda x: x.get("rate_total_mb_s", 0), reverse=True)
            rates = rates[: max(1, int(top_n))]

            return {
                "success": True,
                "agent_hostname": agent_hostname,
                "timestamp": latest.timestamp.isoformat() if latest.timestamp else None,
                "interfaces": rates,
            }
    except Exception as e:
        logger.error(f"Error in get_interface_stats for {agent_hostname}: {e}")
        return {"success": False, "message": f"Failed to get interface stats: {str(e)}"}

@tool
def get_threat_summary(agent_hostname: str, hours: int = 1) -> Dict[str, Any]:
    """Summarize recent network threats/signals for an agent in the last N hours.

    Reports:
    - Peak new connections per second (approx from total_connections delta)
    - Receive bandwidth spikes (vs. simple baseline)
    - ARP gateway MAC change alerts
    """
    try:
        db_manager = get_db_manager()
        with db_manager.get_session() as db:
            agent = db.query(Agent).filter(Agent.hostname == agent_hostname).first()
            if not agent:
                return {"success": False, "message": f"Agent '{agent_hostname}' not found"}

            since_time = local_now() - timedelta(hours=hours)
            metrics = (
                db.query(NetworkMetric)
                .filter(NetworkMetric.agent_id == agent.id, NetworkMetric.timestamp >= since_time)
                .order_by(NetworkMetric.timestamp.asc())
                .all()
            )

            peak_new_cps = 0.0
            recv_rates = []
            # Compute per-sample agg recv rate and new connections per second
            for i in range(1, len(metrics)):
                prev = metrics[i-1]
                cur = metrics[i]
                dt = (cur.timestamp - prev.timestamp).total_seconds() if cur.timestamp and prev.timestamp else 0
                if dt <= 0:
                    continue
                # New connections per second via total_connections delta
                prev_tc = float(prev.total_connections or 0)
                cur_tc = float(cur.total_connections or 0)
                new_cps = max(0.0, (cur_tc - prev_tc) / dt)
                peak_new_cps = max(peak_new_cps, new_cps)

                # Aggregate recv byte rate from interfaces
                agg_prev = 0.0
                agg_cur = 0.0
                if isinstance(prev.interfaces, dict):
                    for v in prev.interfaces.values():
                        agg_prev += float((v or {}).get('bytes_recv', 0) or 0)
                if isinstance(cur.interfaces, dict):
                    for v in cur.interfaces.values():
                        agg_cur += float((v or {}).get('bytes_recv', 0) or 0)
                recv_rate = max(0.0, (agg_cur - agg_prev) / dt)
                recv_rates.append(recv_rate)

            # Baseline and spike detection
            baseline = sum(recv_rates[:-1]) / max(1, len(recv_rates[:-1])) if len(recv_rates) > 1 else (sum(recv_rates) / max(1, len(recv_rates)))
            latest_rate = recv_rates[-1] if recv_rates else 0.0
            spike_factor = (latest_rate / baseline) if baseline > 0 else 0.0

            # Recent ARP gateway change alerts
            alerts = (
                db.query(Alert)
                .filter(Alert.agent_id == agent.id, Alert.alert_type == 'arp_gateway_change', Alert.timestamp >= since_time)
                .order_by(Alert.timestamp.desc())
                .all()
            )

            return {
                "success": True,
                "agent_hostname": agent_hostname,
                "time_window_hours": hours,
                "peak_new_connections_per_s": round(peak_new_cps, 2),
                "recv_rate_latest_Bps": round(latest_rate, 1),
                "recv_baseline_Bps": round(baseline, 1),
                "recv_spike_factor": round(spike_factor, 2) if baseline > 0 else None,
                "arp_gateway_change_alerts": [
                    {
                        "timestamp": a.timestamp.isoformat(),
                        "severity": a.severity,
                        "title": a.title,
                        "description": a.description,
                    } for a in alerts
                ]
            }
    except Exception as e:
        logger.error(f"Error in get_threat_summary for {agent_hostname}: {e}")
        return {"success": False, "message": f"Failed to get threat summary: {str(e)}"}
@tool
def get_recent_alerts(hours: int = 24, severity: Optional[str] = None) -> AlertsResponse:
    """Get recent alerts from the system.
    
    Args:
        hours: How many hours back to look for alerts (default: 24)
        severity: Filter by severity (low, medium, high, critical) or None for all
    """
    try:
        db_manager = get_db_manager()
        with db_manager.get_session() as db:
            # Build query
            since_time = local_now() - timedelta(hours=hours)
            query = db.query(Alert).filter(Alert.timestamp >= since_time)
            
            if severity:
                query = query.filter(Alert.severity == severity)
            
            alerts = query.order_by(Alert.timestamp.desc()).all()
            
            # Build severity breakdown
            severity_breakdown = {"low": 0, "medium": 0, "high": 0, "critical": 0}
            alert_details = []
            
            for alert in alerts:
                severity_breakdown[alert.severity] = severity_breakdown.get(alert.severity, 0) + 1
                
                # Get agent hostname
                agent = db.query(Agent).filter(Agent.id == alert.agent_id).first()
                agent_hostname = agent.hostname if agent else f"Agent {alert.agent_id}"
                
                alert_details.append({
                    "id": alert.id,
                    "agent_hostname": agent_hostname,
                    "timestamp": alert.timestamp.isoformat(),
                    "alert_type": alert.alert_type,
                    "severity": alert.severity,
                    "title": alert.title,
                    "description": alert.description,
                    "metric_name": alert.metric_name,
                    "metric_value": alert.metric_value,
                    "anomaly_score": alert.anomaly_score
                })
            
            return AlertsResponse(
                total_alerts=len(alerts),
                alerts=alert_details,
                severity_breakdown=severity_breakdown
            )
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        return AlertsResponse(
            total_alerts=0,
            alerts=[],
            severity_breakdown={"low": 0, "medium": 0, "high": 0, "critical": 0}
        )

@tool
def get_analysis_results(limit: int = 5, agent_hostname: Optional[str] = None) -> AnalysisResponse:
    """Get recent AI analysis results.
    
    Args:
        limit: Maximum number of analyses to return (default: 5)
        agent_hostname: Filter by agent hostname or None for all agents
    """
    try:
        db_manager = get_db_manager()
        with db_manager.get_session() as db:
            # Build query
            query = db.query(AIAnalysis)
            
            if agent_hostname:
                agent = db.query(Agent).filter(Agent.hostname == agent_hostname).first()
                if agent:
                    query = query.filter(AIAnalysis.agent_id == agent.id)
            
            analyses = query.order_by(AIAnalysis.timestamp.desc()).limit(limit).all()
            
            analysis_details = []
            latest_analysis = None
            
            for i, analysis in enumerate(analyses):
                # Get agent hostname
                agent = db.query(Agent).filter(Agent.id == analysis.agent_id).first()
                agent_hostname_result = agent.hostname if agent else f"Agent {analysis.agent_id}"
                
                analysis_data = {
                    "id": analysis.id,
                    "agent_hostname": agent_hostname_result,
                    "timestamp": analysis.timestamp.isoformat(),
                    "analysis_type": analysis.analysis_type,
                    "model_used": analysis.model_used,
                    "confidence_score": analysis.confidence_score,
                    "status": analysis.status,
                    "processing_time_ms": analysis.processing_time_ms,
                    "data_points_analyzed": analysis.data_points_analyzed,
                    "findings_count": len(analysis.findings) if analysis.findings else 0,
                    "recommendations_count": len(analysis.recommendations) if analysis.recommendations else 0
                }
                
                analysis_details.append(analysis_data)
                
                if i == 0:  # Latest analysis
                    latest_analysis = analysis_data
            
            return AnalysisResponse(
                total_analyses=len(analyses),
                latest_analysis=latest_analysis,
                analyses=analysis_details
            )
    except Exception as e:
        logger.error(f"Error getting analysis results: {e}")
        return AnalysisResponse(
            total_analyses=0,
            latest_analysis=None,
            analyses=[]
        )

@tool
def get_analysis_details(analysis_id: int) -> Dict[str, Any]:
    """Get detailed information about a specific AI analysis by ID.
    
    Args:
        analysis_id: The ID of the analysis to retrieve details for
    """
    try:
        db_manager = get_db_manager()
        with db_manager.get_session() as db:
            # Find the analysis
            analysis = db.query(AIAnalysis).filter(AIAnalysis.id == analysis_id).first()
            if not analysis:
                return {
                    "success": False,
                    "message": f"Analysis with ID {analysis_id} not found"
                }
            
            # Get agent hostname
            agent = db.query(Agent).filter(Agent.id == analysis.agent_id).first()
            agent_hostname = agent.hostname if agent else f"Agent {analysis.agent_id}"
            
            # Format detailed analysis information
            details = {
                "id": analysis.id,
                "agent_hostname": agent_hostname,
                "agent_id": analysis.agent_id,
                "timestamp": analysis.timestamp.isoformat(),
                "analysis_type": analysis.analysis_type,
                "model_used": analysis.model_used,
                "confidence_score": analysis.confidence_score,
                "status": analysis.status,
                "processing_time_ms": analysis.processing_time_ms,
                "data_points_analyzed": analysis.data_points_analyzed,
                "findings": analysis.findings or [],
                "recommendations": analysis.recommendations or [],
                "risk_assessment": analysis.risk_assessment or {},
                "findings_count": len(analysis.findings) if analysis.findings else 0,
                "recommendations_count": len(analysis.recommendations) if analysis.recommendations else 0
            }
            
            return {
                "success": True,
                "analysis": details
            }
            
    except Exception as e:
        logger.error(f"Error getting analysis details for ID {analysis_id}: {e}")
        return {
            "success": False,
            "message": f"Failed to get analysis details: {str(e)}"
        }

@tool
def trigger_agent_analysis(agent_hostname: str, time_window_hours: int = 24) -> Dict[str, Any]:
    """Trigger AI analysis for a specific agent and wait for completion.
    
    Args:
        agent_hostname: The hostname of the agent to analyze
        time_window_hours: Time window in hours for the analysis (default: 24)
    """
    try:
        db_manager = get_db_manager()
        with db_manager.get_session() as db:
            # Find agent
            agent = db.query(Agent).filter(Agent.hostname == agent_hostname).first()
            if not agent:
                return {
                    "success": False,
                    "message": f"Agent '{agent_hostname}' not found"
                }
            
            # Get the AI service - we need to import and initialize it
            from server.ai_engine.service import AIEngineService
            from pathlib import Path
            
            # Load config from the chat agent's config if available
            # or use default config
            config = {
                'ai': {
                    'model_name': 'gemini-2.5-flash',
                    'temperature': 0.1
                },
                'anomaly_detection': {
                    'min_samples': 10,
                    'window_size': 24
                }
            }
            
            # Try to load actual config
            try:
                import yaml
                config_path = Path(__file__).parent.parent.parent / 'config' / 'config.yaml'
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        loaded_config = yaml.safe_load(f)
                        if loaded_config:
                            config.update(loaded_config)
            except Exception as e:
                logger.warning(f"Could not load config file, using defaults: {e}")
            
            # Initialize AI service
            ai_service = AIEngineService(config)
            
            # Trigger the analysis and wait for completion
            logger.info(f"Starting AI analysis for agent {agent_hostname} (ID: {agent.id})")
            
            # Run the async analysis in the current event loop
            import asyncio
            
            try:
                # Get the current event loop, or create a new one if none exists
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're already in an async context, we need to use a different approach
                    # Create a new task and use asyncio.create_task
                    analysis = loop.run_until_complete(
                        ai_service.analyze_agent_anomalies(
                            agent.id, 
                            time_window_hours, 
                            force_analysis=True
                        )
                    )
                else:
                    analysis = asyncio.run(
                        ai_service.analyze_agent_anomalies(
                            agent.id, 
                            time_window_hours, 
                            force_analysis=True
                        )
                    )
            except RuntimeError:
                # If there's already an event loop running, use asyncio.create_task approach
                # This is a workaround for nested async calls
                import concurrent.futures
                
                async def run_analysis():
                    return await ai_service.analyze_agent_anomalies(
                        agent.id, 
                        time_window_hours, 
                        force_analysis=True
                    )
                
                # Use ThreadPoolExecutor as a fallback
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, run_analysis())
                    analysis = future.result(timeout=300)  # 5 minute timeout
            
            if not analysis:
                return {
                    "success": False,
                    "message": f"Analysis could not be completed for agent '{agent_hostname}'. This may be due to insufficient data or the agent being offline."
                }
            
            # Return detailed results
            return {
                "success": True,
                "message": f"AI analysis completed for agent '{agent_hostname}'",
                "analysis_id": analysis.id,
                "agent_id": agent.id,
                "agent_hostname": agent_hostname,
                "time_window_hours": time_window_hours,
                "status": analysis.status,
                "confidence_score": analysis.confidence_score,
                "processing_time_ms": analysis.processing_time_ms,
                "data_points_analyzed": analysis.data_points_analyzed,
                "findings_count": len(analysis.findings) if analysis.findings else 0,
                "recommendations_count": len(analysis.recommendations) if analysis.recommendations else 0,
                "completion_time": analysis.timestamp.isoformat()
            }
    except Exception as e:
        logger.error(f"Error triggering analysis for {agent_hostname}: {e}")
        return {
            "success": False,
            "message": f"Failed to trigger analysis: {str(e)}"
        }

class AINetChatAgent:
    """AI Chat Agent with access to AINet system data and tools."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Initialize LLM via provider (supports Vertex tuned models via env)
        ai_config = self.config.get('ai', {})
        self.llm = build_chat_model(ai_config)

        # Define available tools
        self.tools = [
            get_system_status,
            get_agents_status,
            get_agent_metrics,
            get_recent_alerts,
            get_analysis_results,
            get_analysis_details,
            trigger_agent_analysis,
            get_network_overview,
            get_interface_stats,
            get_threat_summary,
        ]

        # Create tool node
        self.tool_node = ToolNode(self.tools)

        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Initialize memory for conversation history
        self.memory = MemorySaver()

        # Initialize database checkpointer for persistence
        self.db_checkpointer = SQLAlchemyCheckpointer()

        # Create the graph
        self.graph = self._create_graph()

        # Load existing conversation state if available
        self._conversation_cache = {}
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow."""
        workflow = StateGraph(ChatState)
        
        # Add nodes
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", self.tool_node)
        
        # Add edges
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END,
            },
        )
        workflow.add_edge("tools", "agent")
        
        return workflow.compile(checkpointer=self.memory)
    
    def _call_model(self, state: ChatState):
        """Call the model with the current state."""
        system_message = SystemMessage(content="""
You are an AI assistant for the AINet network monitoring system. You have access to tools that can query the system's data including:

- System status and health
- Network agent information and status
- Network metrics and performance data
- Security alerts and anomalies
- AI analysis results and insights
- Detailed analysis information by ID
- Ability to trigger new analyses

You can help users:
1. Check system status and agent health
2. Analyze network performance and metrics
3. Investigate alerts and anomalies
4. Review AI analysis results
5. Get detailed information about specific analyses by ID
6. Trigger new analyses when needed
7. Provide insights and recommendations

IMPORTANT GUIDELINES:
- When using tools, always explain what you're doing first (e.g., "Let me check the system status..." or "I'll trigger an analysis for that agent...")
- After receiving tool results, ALWAYS interpret and summarize the key findings for the user
- For the trigger_agent_analysis tool specifically, explain that the analysis is running and provide a summary of the results when complete
- When analysis is triggered successfully, inform the user about the analysis ID and key metrics like confidence score, findings count, etc.
- If a tool returns an error or no data, explain what this means and suggest alternative actions
- Always provide context and actionable insights based on the data you retrieve
- Be conversational but professional and informative

When a user asks about a specific analysis or wants more details about an analysis, use the get_analysis_details tool with the analysis ID to provide comprehensive information including findings, recommendations, and risk assessments.

If asked about capabilities outside of network monitoring, politely redirect to your core functions.
        """)
        
        messages = [system_message] + state["messages"]
        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def _should_continue(self, state: ChatState):
        """Decide whether to continue with tool calls or end."""
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "continue"
        return "end"
    
    async def chat(self, message: str, thread_id: str = "default") -> Dict[str, Any]:
        """Process a chat message and return the response with any tool calls."""
        try:
            # Load conversation history from database if not in memory
            if thread_id not in self._conversation_cache:
                saved_state = self.db_checkpointer.get_checkpoint(thread_id)
                if saved_state:
                    self._conversation_cache[thread_id] = saved_state.get("messages", [])
                else:
                    self._conversation_cache[thread_id] = []
            
            # Create a configuration with thread ID for memory persistence
            config = {"configurable": {"thread_id": thread_id}}
            
            # Initialize state with conversation history and current message
            conversation_history = self._conversation_cache[thread_id]
            messages_to_add = []
            
            # Add conversation history if not already in the langgraph memory
            for msg in conversation_history:
                if msg.get("type") == "human":
                    messages_to_add.append(HumanMessage(content=msg["content"]))
                elif msg.get("type") == "ai":
                    messages_to_add.append(AIMessage(content=msg["content"]))
            
            # Add current message
            messages_to_add.append(HumanMessage(content=message))
            
            initial_state = {
                "messages": messages_to_add,
                "tool_calls": []
            }
            
            # Run the graph with memory
            result = await self.graph.ainvoke(initial_state, config)
            
            # Extract response and tool calls
            final_message = result["messages"][-1]
            
            # Extract tool calls and their results that occurred during the conversation
            tool_calls = []
            for i, msg in enumerate(result["messages"]):
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        # Look for the corresponding tool result in the next messages
                        tool_result = None
                        for j in range(i + 1, len(result["messages"])):
                            next_msg = result["messages"][j]
                            if hasattr(next_msg, 'tool_call_id') and next_msg.tool_call_id == tool_call["id"]:
                                tool_result = next_msg.content
                                break
                            elif hasattr(next_msg, 'content') and isinstance(next_msg.content, str):
                                # For ToolMessage, check if it corresponds to this tool call
                                try:
                                    from langchain_core.messages import ToolMessage
                                    if isinstance(next_msg, ToolMessage) and hasattr(next_msg, 'tool_call_id') and next_msg.tool_call_id == tool_call["id"]:
                                        tool_result = next_msg.content
                                        break
                                except:
                                    pass
                        
                        tool_calls.append({
                            "name": tool_call["name"],
                            "args": tool_call["args"],
                            "id": tool_call["id"],
                            "result": tool_result
                        })
            
            # Update conversation cache
            self._conversation_cache[thread_id].append({
                "type": "human",
                "content": message,
                "timestamp": datetime.now().isoformat()
            })
            self._conversation_cache[thread_id].append({
                "type": "ai", 
                "content": final_message.content,
                "timestamp": datetime.now().isoformat()
            })
            
            # Save to database
            self.db_checkpointer.put_checkpoint(thread_id, {
                "messages": self._conversation_cache[thread_id],
                "last_updated": datetime.now().isoformat()
            })
            
            return {
                "response": final_message.content,
                "tool_calls": tool_calls,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error in chat processing: {e}")
            return {
                "response": f"I encountered an error while processing your request: {str(e)}",
                "tool_calls": [],
                "success": False
            }
    
    def get_conversation_history(self, thread_id: str = "default") -> List[Dict[str, Any]]:
        """Get conversation history for a thread."""
        try:
            if thread_id in self._conversation_cache:
                return self._conversation_cache[thread_id]
            
            saved_state = self.db_checkpointer.get_checkpoint(thread_id)
            if saved_state:
                return saved_state.get("messages", [])
            
            return []
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []
    
    def clear_conversation(self, thread_id: str = "default") -> bool:
        """Clear conversation history for a thread."""
        try:
            if thread_id in self._conversation_cache:
                del self._conversation_cache[thread_id]
            
            self.db_checkpointer.clear_checkpoint(thread_id)
            return True
        except Exception as e:
            logger.error(f"Error clearing conversation: {e}")
            return False

# Global instance
chat_agent = None

def get_chat_agent(config: Dict[str, Any]) -> AINetChatAgent:
    """Get or create the chat agent instance."""
    global chat_agent
    if chat_agent is None:
        chat_agent = AINetChatAgent(config)
    return chat_agent
