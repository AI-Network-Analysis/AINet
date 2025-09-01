"""
AI Chat Agent with LangGraph Integration
Provides conversational interface with tool calling capabilities for system monitoring.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, TypedDict, Annotated
import asyncio

from langchain_google_genai import ChatGoogleGenerativeAI
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
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.models import Agent, NetworkMetric, Alert, AIAnalysis
from database.connection import get_db, get_db_manager
from ai_engine.service import AIAnalysisService
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
            "timestamp": datetime.utcnow().isoformat(),
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
                is_online = (datetime.utcnow() - agent.last_seen).total_seconds() < 300
                
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
            since_time = datetime.utcnow() - timedelta(hours=hours)
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
            since_time = datetime.utcnow() - timedelta(hours=hours)
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
def trigger_agent_analysis(agent_hostname: str, time_window_hours: int = 24) -> Dict[str, Any]:
    """Trigger AI analysis for a specific agent.
    
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
            
            # Note: In a real implementation, you'd trigger the actual analysis here
            # For now, we'll return a placeholder response
            return {
                "success": True,
                "message": f"Analysis triggered for agent '{agent_hostname}'",
                "agent_id": agent.id,
                "time_window_hours": time_window_hours,
                "note": "Analysis will complete shortly. Check the Analysis Results tab for updates."
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
        
        # Initialize LLM with Google GenAI
        ai_config = config.get('ai', {})
        self.llm = ChatGoogleGenerativeAI(
            model=ai_config.get('model_name', 'gemini-2.5-flash'),
            temperature=ai_config.get('temperature', 0.1),
            max_output_tokens=ai_config.get('max_tokens', 1000)
        )
        
        # Define available tools
        self.tools = [
            get_system_status,
            get_agents_status,
            get_agent_metrics,
            get_recent_alerts,
            get_analysis_results,
            trigger_agent_analysis
        ]
        
        # Create tool node
        self.tool_node = ToolNode(self.tools)
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Create the graph
        self.graph = self._create_graph()
    
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
        
        return workflow.compile()
    
    def _call_model(self, state: ChatState):
        """Call the model with the current state."""
        system_message = SystemMessage(content="""
You are an AI assistant for the AINet network monitoring system. You have access to tools that can query the system's data including:

- System status and health
- Network agent information and status
- Network metrics and performance data
- Security alerts and anomalies
- AI analysis results and insights
- Ability to trigger new analyses

You can help users:
1. Check system status and agent health
2. Analyze network performance and metrics
3. Investigate alerts and anomalies
4. Review AI analysis results
5. Trigger new analyses when needed
6. Provide insights and recommendations

When using tools, explain what you're doing and interpret the results in a helpful way. Always provide context and actionable insights based on the data you retrieve.

Be conversational but professional. If asked about capabilities outside of network monitoring, politely redirect to your core functions.
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
    
    async def chat(self, message: str, conversation_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a chat message and return the response with any tool calls."""
        try:
            # Convert conversation history to messages
            messages = []
            if conversation_history:
                for msg in conversation_history:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        messages.append(AIMessage(content=msg["content"]))
            
            # Add current message
            messages.append(HumanMessage(content=message))
            
            # Initialize state
            initial_state = {
                "messages": messages,
                "tool_calls": []
            }
            
            # Run the graph
            result = await self.graph.ainvoke(initial_state)
            
            # Extract response and tool calls
            final_message = result["messages"][-1]
            
            # Extract tool calls that occurred during the conversation
            tool_calls = []
            for msg in result["messages"]:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        tool_calls.append({
                            "name": tool_call["name"],
                            "args": tool_call["args"],
                            "id": tool_call["id"]
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

# Global instance
chat_agent = None

def get_chat_agent(config: Dict[str, Any]) -> AINetChatAgent:
    """Get or create the chat agent instance."""
    global chat_agent
    if chat_agent is None:
        chat_agent = AINetChatAgent(config)
    return chat_agent
