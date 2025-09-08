"""
AI Anomaly Detection Engine
Uses LangGraph and LangChain to analyze network metrics and detect anomalies.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, TypedDict
import numpy as np
from dataclasses import dataclass, asdict

from server.ai_engine.llm_provider import build_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# Pydantic models for structured outputs
class AnomalyFinding(BaseModel):
    """Individual anomaly finding"""
    metric_name: str = Field(description="Name of the metric with anomaly")
    anomaly_type: str = Field(description="Type of anomaly: spike, drop, pattern_break, trend_change")
    severity: str = Field(description="Severity level: low, medium, high, critical")
    confidence: float = Field(description="Confidence score 0-1", ge=0, le=1)
    description: str = Field(description="Human-readable description of the anomaly")
    affected_timeframe: str = Field(description="Time period affected by the anomaly")
    baseline_value: Optional[float] = Field(description="Expected baseline value")
    anomalous_value: float = Field(description="The anomalous value observed")
    deviation_percentage: float = Field(description="Percentage deviation from baseline")

class NetworkRecommendation(BaseModel):
    """AI recommendation for network optimization"""
    category: str = Field(description="Category: performance, security, capacity, maintenance")
    priority: str = Field(description="Priority: low, medium, high, urgent")
    title: str = Field(description="Short title of recommendation")
    description: str = Field(description="Detailed recommendation description")
    expected_impact: str = Field(description="Expected impact of implementing recommendation")
    implementation_effort: str = Field(description="Estimated effort: low, medium, high")

class RiskAssessment(BaseModel):
    """Risk assessment for network health"""
    overall_risk_level: str = Field(description="Overall risk: low, medium, high, critical")
    risk_factors: List[str] = Field(description="List of identified risk factors")
    potential_impacts: List[str] = Field(description="Potential impacts if issues not addressed")
    urgency_score: float = Field(description="Urgency score 0-1", ge=0, le=1)

class AnomalyAnalysisResult(BaseModel):
    """Complete anomaly analysis result"""
    findings: List[AnomalyFinding] = Field(description="List of anomaly findings")
    recommendations: List[NetworkRecommendation] = Field(description="AI recommendations")
    risk_assessment: RiskAssessment = Field(description="Overall risk assessment")
    summary: str = Field(description="Executive summary of analysis")
    analysis_confidence: float = Field(description="Overall analysis confidence 0-1", ge=0, le=1)

# LangGraph State
class AnalysisState(TypedDict):
    """State for the analysis workflow"""
    agent_id: int
    hostname: str
    raw_metrics: List[Dict[str, Any]]
    processed_metrics: Dict[str, List[float]]
    statistical_analysis: Dict[str, Any]
    anomaly_candidates: List[Dict[str, Any]]
    ai_analysis: Optional[AnomalyAnalysisResult]
    messages: List[AnyMessage]
    error: Optional[str]

@dataclass
class MetricStatistics:
    """Statistical analysis of a metric"""
    mean: float
    std: float
    min: float
    max: float
    median: float
    q1: float
    q3: float
    trend: str  # increasing, decreasing, stable
    recent_change_pct: float

class AnomalyDetector:
    """AI-powered network anomaly detector"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize LLM via provider (supports Vertex tuned models via env)
        model_config = self.config.get('ai', {})
        try:
            self.llm = build_chat_model(model_config)
        except Exception as e:
            logger.warning(f"LLM unavailable, using heuristic fallback. Reason: {e}")
            self.llm = None
        
        # Configure analysis parameters
        self.window_size = model_config.get('anomaly_detection', {}).get('window_size', 100)
        self.sensitivity = model_config.get('anomaly_detection', {}).get('sensitivity', 0.8)
        self.min_samples = model_config.get('anomaly_detection', {}).get('min_samples', 20)
        
        # Create workflow
        self.workflow = self._create_workflow()
        
        # Metrics to analyze
        self.key_metrics = [
            'cpu_percent', 'memory_percent', 'disk_percent',
            'total_connections', 'google_dns_latency_ms',
            'cloudflare_dns_latency_ms', 'local_gateway_latency_ms'
        ]
        
    def _create_workflow(self):
        """Create LangGraph workflow for anomaly detection"""
        workflow = StateGraph(AnalysisState)

        # Add nodes
        workflow.add_node("preprocess_data", self._preprocess_data)
        workflow.add_node("statistical_analysis", self._statistical_analysis)
        workflow.add_node("detect_anomalies", self._detect_anomalies)
        workflow.add_node("ai_analysis", self._ai_analysis)
        workflow.add_node("generate_recommendations", self._generate_recommendations)

        # Add edges
        workflow.add_edge(START, "preprocess_data")
        workflow.add_edge("preprocess_data", "statistical_analysis")
        workflow.add_edge("statistical_analysis", "detect_anomalies")
        workflow.add_edge("detect_anomalies", "ai_analysis")
        workflow.add_edge("ai_analysis", "generate_recommendations")
        workflow.add_edge("generate_recommendations", END)

        # Compile without checkpointing to avoid serialization issues in runtime
        return workflow.compile()
    
    def _preprocess_data(self, state: AnalysisState) -> Dict[str, Any]:
        """Preprocess raw metrics data"""
        try:
            logger.info(f"Preprocessing data for agent {state['hostname']}")
            
            processed_metrics = {}
            raw_metrics = state['raw_metrics']
            
            if len(raw_metrics) < self.min_samples:
                return {
                    **state,
                    "error": f"Insufficient data points: {len(raw_metrics)} < {self.min_samples}"
                }
            
            # Extract time series for each metric
            for metric in self.key_metrics:
                values = []
                for data_point in raw_metrics:
                    value = self._extract_metric_value(data_point, metric)
                    if value is not None and not np.isnan(value):
                        values.append(value)
                
                if len(values) > 0:
                    processed_metrics[metric] = values
                    logger.debug(f"Processed {len(values)} values for metric {metric}")
                
            return {
                **state,
                "processed_metrics": processed_metrics,
                "messages": state.get("messages", []) + [
                    HumanMessage(content=f"Preprocessed {len(processed_metrics)} metrics for analysis")
                ]
            }
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            return {**state, "error": str(e)}
    
    def _statistical_analysis(self, state: AnalysisState) -> Dict[str, Any]:
        """Perform statistical analysis on metrics"""
        try:
            logger.info("Performing statistical analysis")
            
            statistical_analysis = {}
            processed_metrics = state['processed_metrics']
            
            for metric_name, values in processed_metrics.items():
                if len(values) < 5:  # Need minimum samples for statistics
                    continue
                    
                values_array = np.array(values)
                
                # Calculate statistics
                stats = MetricStatistics(
                    mean=float(np.mean(values_array)),
                    std=float(np.std(values_array)),
                    min=float(np.min(values_array)),
                    max=float(np.max(values_array)),
                    median=float(np.median(values_array)),
                    q1=float(np.percentile(values_array, 25)),
                    q3=float(np.percentile(values_array, 75)),
                    trend=self._analyze_trend(values_array),
                    recent_change_pct=self._calculate_recent_change(values_array)
                )
                
                statistical_analysis[metric_name] = asdict(stats)
                
            return {
                **state,
                "statistical_analysis": statistical_analysis,
                "messages": state.get("messages", []) + [
                    HumanMessage(content=f"Completed statistical analysis for {len(statistical_analysis)} metrics")
                ]
            }
            
        except Exception as e:
            logger.error(f"Error in statistical analysis: {e}")
            return {**state, "error": str(e)}
    
    def _detect_anomalies(self, state: AnalysisState) -> Dict[str, Any]:
        """Detect anomaly candidates using statistical methods"""
        try:
            logger.info("Detecting anomaly candidates")
            
            anomaly_candidates = []
            processed_metrics = state['processed_metrics']
            statistical_analysis = state['statistical_analysis']
            
            for metric_name, values in processed_metrics.items():
                if metric_name not in statistical_analysis:
                    continue
                    
                stats = statistical_analysis[metric_name]
                values_array = np.array(values)
                
                # Z-score based anomaly detection
                z_scores = np.abs((values_array - stats['mean']) / max(stats['std'], 0.001))
                anomaly_threshold = 2.5 * self.sensitivity  # Adjustable threshold
                
                anomalous_indices = np.where(z_scores > anomaly_threshold)[0]
                
                for idx in anomalous_indices:
                    anomaly_candidates.append({
                        'metric_name': metric_name,
                        'value': values[idx],
                        'z_score': float(z_scores[idx]),
                        'index': int(idx),
                        'baseline': stats['mean'],
                        'deviation_pct': ((values[idx] - stats['mean']) / max(abs(stats['mean']), 0.001)) * 100,
                        'severity': self._classify_anomaly_severity(float(z_scores[idx]))
                    })
                
                # Check for trend anomalies
                if abs(stats['recent_change_pct']) > 50:  # 50% change threshold
                    anomaly_candidates.append({
                        'metric_name': metric_name,
                        'value': values[-1] if values else 0,
                        'z_score': 0,
                        'index': len(values) - 1 if values else 0,
                        'baseline': stats['mean'],
                        'deviation_pct': stats['recent_change_pct'],
                        'severity': 'medium' if abs(stats['recent_change_pct']) < 100 else 'high',
                        'type': 'trend_change'
                    })
            
            logger.info(f"Detected {len(anomaly_candidates)} anomaly candidates")
            
            return {
                **state,
                "anomaly_candidates": anomaly_candidates,
                "messages": state.get("messages", []) + [
                    HumanMessage(content=f"Identified {len(anomaly_candidates)} potential anomalies")
                ]
            }
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return {**state, "error": str(e)}
    
    def _ai_analysis(self, state: AnalysisState) -> Dict[str, Any]:
        """Use AI to analyze anomalies and provide insights"""
        try:
            logger.info("Performing AI analysis of anomalies")
            
            if not state['anomaly_candidates']:
                # No anomalies found, still provide analysis
                result = AnomalyAnalysisResult(
                    findings=[],
                    recommendations=[],
                    risk_assessment=RiskAssessment(
                        overall_risk_level="low",
                        risk_factors=[],
                        potential_impacts=[],
                        urgency_score=0.1
                    ),
                    summary="No significant anomalies detected. Network metrics appear normal.",
                    analysis_confidence=0.9
                )
                
                return {
                    **state,
                    "ai_analysis": result,
                    "messages": state.get("messages", []) + [
                        HumanMessage(content="AI analysis completed - no anomalies found")
                    ]
                }
            
            # If LLM is unavailable, go straight to fallback
            if not getattr(self, 'llm', None):
                raise RuntimeError("LLM not initialized; using fallback analysis")

            # Prepare context for AI analysis
            context = self._prepare_ai_context(
                state['hostname'],
                state['statistical_analysis'],
                state['anomaly_candidates']
            )
            
            # Create AI analysis prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", self._get_analysis_system_prompt()),
                ("human", "Analyze the following network metrics data and anomalies:\n\n{context}")
            ])
            
            # Set up output parser
            parser = PydanticOutputParser(pydantic_object=AnomalyAnalysisResult)
            
            # Create chain
            chain = prompt | self.llm | parser
            
            # Run analysis
            result = chain.invoke({
                "context": context,
                "format_instructions": parser.get_format_instructions()
            })
            
            logger.info(f"AI analysis completed with {len(result.findings)} findings")
            
            return {
                **state,
                "ai_analysis": result,
                "messages": state.get("messages", []) + [
                    HumanMessage(content=f"AI analysis completed with {len(result.findings)} findings")
                ]
            }
            
        except Exception as e:
            # Fallback: build a basic analysis from anomaly candidates without LLM
            logger.error(f"Error in AI analysis, using fallback: {e}")
            try:
                findings = []
                for cand in state.get('anomaly_candidates', [])[:20]:  # cap to 20
                    findings.append(AnomalyFinding(
                        metric_name=cand.get('metric_name', 'unknown'),
                        anomaly_type=cand.get('type', 'z_score'),
                        severity=str(cand.get('severity', 'low')),
                        confidence=min(1.0, max(0.0, float(cand.get('z_score', 0)) / 5.0 + 0.2)),
                        description=f"Detected anomalous deviation of {cand.get('deviation_pct', 0):.1f}% from baseline",
                        affected_timeframe="recent",
                        baseline_value=float(cand.get('baseline', 0) or 0),
                        anomalous_value=float(cand.get('value', 0) or 0),
                        deviation_percentage=float(cand.get('deviation_pct', 0) or 0),
                    ))

                # Simple risk derivation
                sev_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
                max_sev = max([sev_order.get(f.severity, 0) for f in findings], default=0)
                overall = [k for k, v in sev_order.items() if v == max_sev][0] if findings else "low"
                urgency = min(1.0, 0.2 + 0.2 * max_sev + 0.02 * len(findings))

                # Basic recommendations
                recs = []
                if findings:
                    recs.append(NetworkRecommendation(
                        category="performance",
                        priority="high" if max_sev >= 2 else "medium",
                        title="Investigate recent metric anomalies",
                        description="Review system and network metrics for spikes or drops. Correlate with deployments or traffic changes.",
                        expected_impact="Stabilized performance and reduced incident risk",
                        implementation_effort="medium",
                    ))

                analysis = AnomalyAnalysisResult(
                    findings=findings,
                    recommendations=recs,
                    risk_assessment=RiskAssessment(
                        overall_risk_level=overall,
                        risk_factors=[f"Anomalies detected in {len(findings)} metrics"] if findings else [],
                        potential_impacts=["Performance degradation", "Service instability"] if findings else [],
                        urgency_score=urgency,
                    ),
                    summary=(
                        f"Identified {len(findings)} anomaly candidates across key metrics. "
                        "This is a heuristic summary generated without LLM."
                    ),
                    analysis_confidence=0.6 if findings else 0.9,
                )

                return {
                    **state,
                    "ai_analysis": analysis,
                    "messages": state.get("messages", []) + [
                        HumanMessage(content=f"Fallback analysis completed with {len(findings)} findings")
                    ]
                }
            except Exception as inner:
                logger.error(f"Fallback analysis also failed: {inner}")
                return {**state, "error": str(e)}
    
    def _generate_recommendations(self, state: AnalysisState) -> Dict[str, Any]:
        """Generate final recommendations and store results"""
        try:
            logger.info("Generating final recommendations")
            
            ai_analysis = state.get('ai_analysis')
            if not ai_analysis:
                return {**state, "error": "No AI analysis available"}
            
            # Additional processing could be done here
            # For now, the AI analysis already includes recommendations
            
            return {
                **state,
                "messages": state.get("messages", []) + [
                    HumanMessage(content="Analysis workflow completed successfully")
                ]
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return {**state, "error": str(e)}
    
    # Helper methods
    def _extract_metric_value(self, data_point: Dict[str, Any], metric_name: str) -> Optional[float]:
        """Extract metric value from raw data point"""
        try:
            # Handle system metrics
            if metric_name in ['cpu_percent', 'memory_percent', 'disk_percent']:
                return data_point.get('system_metrics', {}).get(metric_name)
            
            # Handle connection metrics
            if metric_name == 'total_connections':
                return data_point.get('connections', {}).get(metric_name)
            
            # Handle latency metrics
            if metric_name.endswith('_latency_ms'):
                return data_point.get('latency_metrics', {}).get(metric_name)
                
            return None
            
        except (KeyError, TypeError):
            return None
    
    def _analyze_trend(self, values: np.ndarray) -> str:
        """Analyze trend in time series data"""
        if len(values) < 3:
            return "stable"
            
        # Simple linear trend analysis
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        # Normalize by standard deviation to account for natural variation
        std = np.std(values)
        normalized_slope = slope / max(std, 0.001)
        
        if normalized_slope > 0.1:
            return "increasing"
        elif normalized_slope < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_recent_change(self, values: np.ndarray) -> float:
        """Calculate recent percentage change"""
        if len(values) < 2:
            return 0
            
        # Compare last 25% of data with previous 25%
        split_point = len(values) // 2
        if split_point < 2:
            return 0
            
        recent_mean = np.mean(values[split_point:])
        previous_mean = np.mean(values[:split_point])
        
        if abs(previous_mean) < 0.001:
            return 0
            
        return ((recent_mean - previous_mean) / abs(previous_mean)) * 100
    
    def _classify_anomaly_severity(self, z_score: float) -> str:
        """Classify anomaly severity based on z-score"""
        if z_score < 2:
            return "low"
        elif z_score < 3:
            return "medium"
        elif z_score < 4:
            return "high"
        else:
            return "critical"
    
    def _prepare_ai_context(self, hostname: str, stats: Dict, anomalies: List[Dict]) -> str:
        """Prepare context string for AI analysis"""
        context_parts = [
            f"Network Host: {hostname}",
            f"Analysis Window: {self.window_size} data points",
            f"Anomalies Detected: {len(anomalies)}",
            "",
            "STATISTICAL SUMMARY:",
        ]
        
        for metric, stat in stats.items():
            context_parts.append(
                f"- {metric}: mean={stat['mean']:.2f}, std={stat['std']:.2f}, "
                f"trend={stat['trend']}, recent_change={stat['recent_change_pct']:.1f}%"
            )
        
        if anomalies:
            context_parts.extend(["", "DETECTED ANOMALIES:"])
            for i, anomaly in enumerate(anomalies, 1):
                context_parts.append(
                    f"{i}. {anomaly['metric_name']}: value={anomaly['value']:.2f}, "
                    f"baseline={anomaly['baseline']:.2f}, "
                    f"deviation={anomaly['deviation_pct']:.1f}%, "
                    f"severity={anomaly['severity']}"
                )
        
        return "\n".join(context_parts)
    
    def _get_analysis_system_prompt(self) -> str:
        """Get system prompt for AI analysis"""
        return """You are an expert network administrator and cybersecurity analyst with deep knowledge of network performance monitoring and anomaly detection. 

Your task is to analyze network metrics data and provide comprehensive insights about potential issues, security concerns, and optimization opportunities.

When analyzing anomalies, consider:
1. Network performance implications
2. Security vulnerabilities or attack patterns
3. Capacity planning concerns
4. System stability risks
5. Operational best practices

Provide specific, actionable recommendations with clear priorities. Focus on practical solutions that network administrators can implement.

Your response must be valid JSON matching the specified schema."""

    # Public methods
    async def analyze_agent_metrics(
        self,
        agent_id: int,
        hostname: str,
        raw_metrics: List[Dict[str, Any]]
    ) -> AnomalyAnalysisResult:
        """Analyze metrics for a specific agent"""
        try:
            logger.info(f"Starting anomaly analysis for agent {hostname}")
            
            initial_state: AnalysisState = {
                "agent_id": agent_id,
                "hostname": hostname,
                "raw_metrics": raw_metrics,
                "processed_metrics": {},
                "statistical_analysis": {},
                "anomaly_candidates": [],
                "ai_analysis": None,
                "messages": [],
                "error": None
            }
            
            # Run the workflow
            config = {"configurable": {"thread_id": f"agent_{agent_id}_{datetime.utcnow().timestamp()}"}}
            result = await self.workflow.ainvoke(initial_state, config)
            
            if result.get("error"):
                raise Exception(result["error"])
                
            ai_analysis = result.get("ai_analysis")
            if not ai_analysis:
                raise Exception("AI analysis failed to produce results")
                
            logger.info(f"Anomaly analysis completed for agent {hostname}")
            return ai_analysis
            
        except Exception as e:
            logger.error(f"Error in anomaly analysis: {e}")
            raise
