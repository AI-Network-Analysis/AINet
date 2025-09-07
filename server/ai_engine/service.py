"""
AI Engine Service
Service layer for AI-powered network analysis and anomaly detection.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc

from .anomaly_detector import AnomalyDetector, AnomalyAnalysisResult
from ..database.models import Agent, NetworkMetric, Alert, AIAnalysis
from ..database.connection import get_db_manager

logger = logging.getLogger(__name__)

def local_now():
    """Return current local time instead of UTC"""
    return datetime.now()

class AIEngineService:
    """Service for AI-powered network analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.anomaly_detector = AnomalyDetector(config)
        self.analysis_in_progress = set()  # Track ongoing analyses
        
    async def analyze_agent_anomalies(
        self, 
        agent_id: int, 
        time_window_hours: int = 24,
        force_analysis: bool = False
    ) -> Optional[AIAnalysis]:
        """Analyze anomalies for a specific agent"""
        
        # Check if analysis is already in progress
        if agent_id in self.analysis_in_progress and not force_analysis:
            logger.info(f"Analysis already in progress for agent {agent_id}")
            return None
            
        try:
            self.analysis_in_progress.add(agent_id)
            
            db_manager = get_db_manager()
            with db_manager.get_session() as db:
                # Get agent info
                agent = db.query(Agent).filter(Agent.id == agent_id).first()
                if not agent:
                    logger.error(f"Agent {agent_id} not found")
                    return None
                
                # Get metrics for analysis
                # Get recent metrics for analysis
                since_time = local_now() - timedelta(hours=time_window_hours)
                metrics = db.query(NetworkMetric).filter(
                    NetworkMetric.agent_id == agent_id,
                    NetworkMetric.timestamp >= since_time
                ).order_by(NetworkMetric.timestamp.asc()).all()
                
                if len(metrics) < self.anomaly_detector.min_samples:
                    logger.warning(f"Insufficient metrics for agent {agent_id}: {len(metrics)} samples")
                    return None
                
                # Convert metrics to raw data format
                raw_metrics = [metric.raw_data for metric in metrics if metric.raw_data]
                
                logger.info(f"Starting AI analysis for agent {agent.hostname} with {len(raw_metrics)} data points")
                
                # Create AI analysis record
                from os import getenv
                model_used = getenv('VERTEX_TUNED_MODEL') or self.config.get('ai', {}).get('model_name', 'unknown')
                ai_analysis = AIAnalysis(
                    agent_id=agent_id,
                    timestamp=local_now(),
                    analysis_type="anomaly_detection",
                    model_used=model_used,
                    status="pending",
                    data_points_analyzed=len(raw_metrics)
                )
                db.add(ai_analysis)
                db.commit()
                
                start_time = local_now()
                
                try:
                    # Run anomaly detection
                    analysis_result = await self.anomaly_detector.analyze_agent_metrics(
                        agent_id=agent_id,
                        hostname=agent.hostname,
                        raw_metrics=raw_metrics
                    )
                    
                    # Calculate processing time
                    processing_time = (local_now() - start_time).total_seconds() * 1000
                    
                    # Update AI analysis record
                    ai_analysis.confidence_score = analysis_result.analysis_confidence
                    ai_analysis.findings = self._convert_findings_to_dict(analysis_result.findings)
                    ai_analysis.recommendations = self._convert_recommendations_to_dict(analysis_result.recommendations)
                    ai_analysis.risk_assessment = self._convert_risk_assessment_to_dict(analysis_result.risk_assessment)
                    ai_analysis.processing_time_ms = int(processing_time)
                    ai_analysis.status = "completed"
                    
                    db.commit()
                    
                    # Create alerts for significant findings
                    alerts_created = await self._create_alerts_from_findings(
                        db, agent_id, analysis_result.findings
                    )
                    
                    logger.info(f"AI analysis completed for agent {agent.hostname}. "
                              f"Created {alerts_created} alerts in {processing_time:.0f}ms")
                    
                    return ai_analysis
                    
                except Exception as e:
                    logger.error(f"Error during AI analysis for agent {agent_id}: {e}")
                    ai_analysis.status = "failed"
                    ai_analysis.processing_time_ms = int((local_now() - start_time).total_seconds() * 1000)
                    db.commit()
                    raise
                
        except Exception as e:
            logger.error(f"Error in analyze_agent_anomalies: {e}")
            raise
        finally:
            self.analysis_in_progress.discard(agent_id)
    
    async def analyze_all_agents(
        self,
        time_window_hours: int = 24,
        *,
        force_analysis: bool = False,
        wait_if_in_progress: bool = False,
    ) -> Dict[str, Any]:
        """Run anomaly analysis for all active agents.

        Args:
            time_window_hours: Lookback window for metrics.
            force_analysis: If True, bypass recent-analysis and in-progress checks.
            wait_if_in_progress: If True and an analysis is already running for an agent,
                wait until it finishes (up to a short timeout) instead of skipping.
        """
        results = {
            "analyzed_agents": 0,
            "failed_analyses": 0,
            "total_alerts": 0,
            "start_time": local_now(),
            "agent_results": []
        }
        
        try:
            db_manager = get_db_manager()
            with db_manager.get_session() as db:
                # Get active agents (last seen within 1 hour)
                cutoff_time = local_now() - timedelta(hours=1)
                agents = db.query(Agent).filter(
                    Agent.last_seen >= cutoff_time,
                    Agent.status == "active"
                ).all()
                
                logger.info(f"Starting batch analysis for {len(agents)} active agents")
                
                # Analyze agents in parallel (with limited concurrency)
                semaphore = asyncio.Semaphore(3)  # Limit concurrent analyses
                
                async def _wait_for_slot(agent_id: int, timeout_seconds: int = 30):
                    """Optionally wait for any ongoing analysis on this agent to finish."""
                    if not wait_if_in_progress or force_analysis:
                        return
                    end_time = local_now() + timedelta(seconds=timeout_seconds)
                    while agent_id in self.analysis_in_progress and local_now() < end_time:
                        await asyncio.sleep(0.5)

                async def analyze_single_agent(agent):
                    async with semaphore:
                        try:
                            # Optionally wait if another analysis is in progress
                            await _wait_for_slot(agent.id)

                            analysis = await self.analyze_agent_anomalies(
                                agent.id, time_window_hours, force_analysis=force_analysis
                            )
                            
                            if analysis:
                                results["analyzed_agents"] += 1
                                # Count alerts created for this agent
                                with db_manager.get_session() as alert_db:
                                    alert_count = alert_db.query(Alert).filter(
                                        Alert.agent_id == agent.id,
                                        Alert.timestamp >= results["start_time"]
                                    ).count()
                                    results["total_alerts"] += alert_count
                                
                                results["agent_results"].append({
                                    "agent_id": agent.id,
                                    "hostname": agent.hostname,
                                    "status": "completed",
                                    "alerts_created": alert_count,
                                    "confidence": analysis.confidence_score
                                })
                            else:
                                # Improve skip reason when analysis is in progress
                                reason = (
                                    "analysis_in_progress" if agent.id in self.analysis_in_progress
                                    else "insufficient_data_or_recent_analysis"
                                )
                                results["agent_results"].append({
                                    "agent_id": agent.id,
                                    "hostname": agent.hostname,
                                    "status": "skipped",
                                    "reason": reason
                                })
                        except Exception as e:
                            results["failed_analyses"] += 1
                            results["agent_results"].append({
                                "agent_id": agent.id,
                                "hostname": agent.hostname,
                                "status": "failed",
                                "error": str(e)
                            })
                            logger.error(f"Failed to analyze agent {agent.hostname}: {e}")
                
                # Run analyses
                await asyncio.gather(*[analyze_single_agent(agent) for agent in agents])
                
                results["end_time"] = local_now()
                results["duration_seconds"] = (results["end_time"] - results["start_time"]).total_seconds()
                
                logger.info(f"Batch analysis completed: {results['analyzed_agents']} analyzed, "
                          f"{results['failed_analyses']} failed, {results['total_alerts']} alerts created")
                
                return results
                
        except Exception as e:
            logger.error(f"Error in analyze_all_agents: {e}")
            results["error"] = str(e)
            return results
    
    async def get_analysis_history(
        self, 
        agent_id: int, 
        limit: int = 10
    ) -> List[AIAnalysis]:
        """Get analysis history for an agent"""
        try:
            db_manager = get_db_manager()
            with db_manager.get_session() as db:
                analyses = db.query(AIAnalysis).filter(
                    AIAnalysis.agent_id == agent_id
                ).order_by(desc(AIAnalysis.timestamp)).limit(limit).all()
                
                return analyses
                
        except Exception as e:
            logger.error(f"Error getting analysis history: {e}")
            return []
    
    async def get_system_health_overview(self) -> Dict[str, Any]:
        """Get overall system health overview"""
        try:
            db_manager = get_db_manager()
            with db_manager.get_session() as db:
                # Get recent alerts
                recent_alerts = db.query(Alert).filter(
                    Alert.timestamp >= local_now() - timedelta(hours=24),
                    Alert.status == "active"
                ).all()
                
                # Categorize alerts by severity
                alert_counts = {
                    "critical": 0,
                    "high": 0,
                    "medium": 0,
                    "low": 0
                }
                
                for alert in recent_alerts:
                    alert_counts[alert.severity] += 1
                
                # Get active agents
                active_agents = db.query(Agent).filter(
                    Agent.last_seen >= local_now() - timedelta(hours=1)
                ).count()
                
                # Get recent analyses
                recent_analyses = db.query(AIAnalysis).filter(
                    AIAnalysis.timestamp >= local_now() - timedelta(hours=24)
                ).count()
                
                # Calculate overall health score
                health_score = self._calculate_health_score(alert_counts, active_agents)
                
                return {
                    "timestamp": local_now().isoformat(),
                    "health_score": health_score,
                    "active_agents": active_agents,
                    "recent_analyses": recent_analyses,
                    "alert_summary": alert_counts,
                    "total_alerts": sum(alert_counts.values()),
                    "critical_issues": alert_counts["critical"] + alert_counts["high"]
                }
                
        except Exception as e:
            logger.error(f"Error getting system health overview: {e}")
            return {
                "error": str(e),
                "timestamp": local_now().isoformat()
            }
    
    # Helper methods
    def _convert_findings_to_dict(self, findings) -> List[Dict[str, Any]]:
        """Convert Pydantic findings to dictionary format"""
        return [finding.dict() for finding in findings]
    
    def _convert_recommendations_to_dict(self, recommendations) -> List[Dict[str, Any]]:
        """Convert Pydantic recommendations to dictionary format"""
        return [rec.dict() for rec in recommendations]
    
    def _convert_risk_assessment_to_dict(self, risk_assessment) -> Dict[str, Any]:
        """Convert Pydantic risk assessment to dictionary format"""
        return risk_assessment.dict()
    
    async def _create_alerts_from_findings(
        self, 
        db: Session, 
        agent_id: int, 
        findings
    ) -> int:
        """Create alerts from anomaly findings"""
        alerts_created = 0
        
        for finding in findings:
            # Only create alerts for medium+ severity findings
            if finding.severity in ["medium", "high", "critical"]:
                alert = Alert(
                    agent_id=agent_id,
                    timestamp=local_now(),
                    alert_type="anomaly",
                    severity=finding.severity,
                    title=f"Anomaly detected in {finding.metric_name}",
                    description=finding.description,
                    metric_name=finding.metric_name,
                    metric_value=finding.anomalous_value,
                    anomaly_score=finding.confidence,
                    context={
                        "anomaly_type": finding.anomaly_type,
                        "baseline_value": finding.baseline_value,
                        "deviation_percentage": finding.deviation_percentage,
                        "affected_timeframe": finding.affected_timeframe
                    }
                )
                
                db.add(alert)
                alerts_created += 1
        
        if alerts_created > 0:
            db.commit()
            
        return alerts_created
    
    def _calculate_health_score(self, alert_counts: Dict[str, int], active_agents: int) -> float:
        """Calculate overall system health score (0-100)"""
        base_score = 100.0
        
        # Deduct points for alerts
        base_score -= alert_counts["critical"] * 25
        base_score -= alert_counts["high"] * 15
        base_score -= alert_counts["medium"] * 5
        base_score -= alert_counts["low"] * 1
        
        # Bonus for having active agents
        if active_agents > 0:
            base_score += min(active_agents * 2, 10)  # Cap bonus at 10 points
        else:
            base_score -= 50  # Major penalty for no active agents
        
        # Ensure score is between 0 and 100
        return max(0.0, min(100.0, base_score))

# Background task runner
class AIEngineScheduler:
    """Scheduler for running periodic AI analyses"""
    
    def __init__(self, ai_service: AIEngineService, config: Dict[str, Any]):
        self.ai_service = ai_service
        self.config = config
        self.running = False
        
    async def start_periodic_analysis(self):
        """Start periodic analysis of all agents"""
        self.running = True
        analysis_interval = self.config.get('ai', {}).get('analysis_interval_minutes', 60)
        
        logger.info(f"Starting periodic AI analysis every {analysis_interval} minutes")
        
        while self.running:
            try:
                logger.info("Starting scheduled anomaly analysis for all agents")
                
                results = await self.ai_service.analyze_all_agents()
                
                logger.info(f"Scheduled analysis completed: {results.get('analyzed_agents', 0)} agents, "
                          f"{results.get('total_alerts', 0)} alerts")
                
            except Exception as e:
                logger.error(f"Error in periodic analysis: {e}")
            
            # Wait for next interval
            await asyncio.sleep(analysis_interval * 60)
    
    def stop(self):
        """Stop periodic analysis"""
        self.running = False
        logger.info("Stopping periodic AI analysis")
