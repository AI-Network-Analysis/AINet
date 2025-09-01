"""
Enhanced Network Anomaly Detection
Specialized monitoring for network activities and anomaly detection.
"""

import psutil
import time
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque, defaultdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class NetworkAnomaly:
    """Network anomaly detection result"""
    timestamp: str
    anomaly_type: str
    severity: str  # low, medium, high, critical
    description: str
    metric_name: str
    current_value: float
    expected_range: Tuple[float, float]
    confidence: float
    interface: Optional[str] = None

class NetworkAnomalyDetector:
    """Advanced network anomaly detection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.history_size = config.get('history_size', 100)
        self.sensitivity = config.get('sensitivity', 0.8)
        
        # Historical data storage (interface -> metric -> values)
        self.bandwidth_history = defaultdict(lambda: defaultdict(lambda: deque(maxlen=self.history_size)))
        self.connection_history = deque(maxlen=self.history_size)
        self.latency_history = defaultdict(lambda: deque(maxlen=self.history_size))
        self.packet_loss_history = defaultdict(lambda: deque(maxlen=self.history_size))
        
        # Baseline establishment
        self.baselines = {}
        self.baseline_established = False
        self.baseline_samples = config.get('baseline_samples', 50)
        
    def analyze_network_metrics(self, metrics: Dict[str, Any]) -> List[NetworkAnomaly]:
        """Analyze network metrics for anomalies"""
        anomalies = []
        timestamp = datetime.utcnow().isoformat()
        
        # Update historical data
        self._update_history(metrics)
        
        # Check if we have enough data for baseline
        if not self.baseline_established and self._has_sufficient_data():
            self._establish_baselines()
            logger.info("Network baselines established")
            
        if self.baseline_established:
            # Detect bandwidth anomalies
            anomalies.extend(self._detect_bandwidth_anomalies(metrics, timestamp))
            
            # Detect connection anomalies
            anomalies.extend(self._detect_connection_anomalies(metrics, timestamp))
            
            # Detect latency anomalies
            anomalies.extend(self._detect_latency_anomalies(metrics, timestamp))
            
            # Detect packet loss anomalies
            anomalies.extend(self._detect_packet_loss_anomalies(metrics, timestamp))
            
            # Detect suspicious patterns
            anomalies.extend(self._detect_suspicious_patterns(metrics, timestamp))
        
        return anomalies
    
    def _update_history(self, metrics: Dict[str, Any]):
        """Update historical data with current metrics"""
        # Update bandwidth history
        bandwidth_stats = metrics.get('bandwidth_stats', {})
        for interface, stats in bandwidth_stats.items():
            for metric, value in stats.items():
                if isinstance(value, (int, float)):
                    self.bandwidth_history[interface][metric].append(value)
        
        # Update connection history
        connections = metrics.get('connections', {})
        if connections:
            self.connection_history.append(connections)
        
        # Update latency history
        latency_metrics = metrics.get('latency_metrics', {})
        for metric, value in latency_metrics.items():
            if isinstance(value, (int, float)) and value >= 0:
                self.latency_history[metric].append(value)
        
        # Update packet loss history
        packet_stats = metrics.get('packet_stats', {})
        if 'overall' in packet_stats:
            overall_stats = packet_stats['overall']
            for metric in ['error_rate', 'drop_rate']:
                if metric in overall_stats:
                    self.packet_loss_history[metric].append(overall_stats[metric])
    
    def _has_sufficient_data(self) -> bool:
        """Check if we have enough data to establish baselines"""
        return len(self.connection_history) >= self.baseline_samples
    
    def _establish_baselines(self):
        """Establish baseline values for anomaly detection"""
        self.baselines = {}
        
        # Bandwidth baselines
        self.baselines['bandwidth'] = {}
        for interface, metrics in self.bandwidth_history.items():
            self.baselines['bandwidth'][interface] = {}
            for metric, values in metrics.items():
                if len(values) >= self.baseline_samples:
                    mean = statistics.mean(values)
                    stdev = statistics.stdev(values) if len(values) > 1 else 0
                    self.baselines['bandwidth'][interface][metric] = {
                        'mean': mean,
                        'stdev': stdev,
                        'min': min(values),
                        'max': max(values)
                    }
        
        # Connection baselines
        if len(self.connection_history) >= self.baseline_samples:
            self.baselines['connections'] = {}
            connection_metrics = defaultdict(list)
            
            for conn_data in self.connection_history:
                for metric, value in conn_data.items():
                    connection_metrics[metric].append(value)
            
            for metric, values in connection_metrics.items():
                mean = statistics.mean(values)
                stdev = statistics.stdev(values) if len(values) > 1 else 0
                self.baselines['connections'][metric] = {
                    'mean': mean,
                    'stdev': stdev,
                    'min': min(values),
                    'max': max(values)
                }
        
        # Latency baselines
        self.baselines['latency'] = {}
        for metric, values in self.latency_history.items():
            if len(values) >= self.baseline_samples:
                mean = statistics.mean(values)
                stdev = statistics.stdev(values) if len(values) > 1 else 0
                self.baselines['latency'][metric] = {
                    'mean': mean,
                    'stdev': stdev,
                    'min': min(values),
                    'max': max(values)
                }
        
        # Packet loss baselines
        self.baselines['packet_loss'] = {}
        for metric, values in self.packet_loss_history.items():
            if len(values) >= self.baseline_samples:
                mean = statistics.mean(values)
                stdev = statistics.stdev(values) if len(values) > 1 else 0
                self.baselines['packet_loss'][metric] = {
                    'mean': mean,
                    'stdev': stdev,
                    'min': min(values),
                    'max': max(values)
                }
        
        self.baseline_established = True
    
    def _detect_bandwidth_anomalies(self, metrics: Dict[str, Any], timestamp: str) -> List[NetworkAnomaly]:
        """Detect bandwidth anomalies"""
        anomalies = []
        bandwidth_stats = metrics.get('bandwidth_stats', {})
        
        for interface, stats in bandwidth_stats.items():
            if interface not in self.baselines.get('bandwidth', {}):
                continue
                
            interface_baselines = self.baselines['bandwidth'][interface]
            
            # Check bytes rates
            for rate_metric in ['bytes_sent_rate', 'bytes_recv_rate']:
                if rate_metric in stats and rate_metric in interface_baselines:
                    current_value = stats[rate_metric]
                    baseline = interface_baselines[rate_metric]
                    
                    # Calculate anomaly score using standard deviations
                    if baseline['stdev'] > 0:
                        z_score = abs(current_value - baseline['mean']) / baseline['stdev']
                        
                        if z_score > 3:  # 3 standard deviations
                            severity = 'critical'
                        elif z_score > 2:
                            severity = 'high'
                        elif z_score > 1.5:
                            severity = 'medium'
                        else:
                            continue
                        
                        direction = 'spike' if current_value > baseline['mean'] else 'drop'
                        
                        anomalies.append(NetworkAnomaly(
                            timestamp=timestamp,
                            anomaly_type='bandwidth_anomaly',
                            severity=severity,
                            description=f"Unusual {direction} in {rate_metric.replace('_', ' ')} on {interface}: {current_value:,.0f} bytes/s (normal: {baseline['mean']:,.0f} ± {baseline['stdev']:,.0f})",
                            metric_name=rate_metric,
                            current_value=current_value,
                            expected_range=(baseline['mean'] - 2*baseline['stdev'], baseline['mean'] + 2*baseline['stdev']),
                            confidence=min(z_score / 3, 1.0),
                            interface=interface
                        ))
        
        return anomalies
    
    def _detect_connection_anomalies(self, metrics: Dict[str, Any], timestamp: str) -> List[NetworkAnomaly]:
        """Detect connection-related anomalies"""
        anomalies = []
        connections = metrics.get('connections', {})
        
        if 'connections' not in self.baselines:
            return anomalies
        
        connection_baselines = self.baselines['connections']
        
        # Check for unusual connection patterns
        suspicious_metrics = ['tcp_established', 'tcp_time_wait', 'total_connections']
        
        for metric in suspicious_metrics:
            if metric in connections and metric in connection_baselines:
                current_value = connections[metric]
                baseline = connection_baselines[metric]
                
                if baseline['stdev'] > 0:
                    z_score = abs(current_value - baseline['mean']) / baseline['stdev']
                    
                    if z_score > 2.5:
                        severity = 'high' if z_score > 3 else 'medium'
                        direction = 'spike' if current_value > baseline['mean'] else 'drop'
                        
                        # High connection counts could indicate port scans or DDoS
                        if metric == 'total_connections' and current_value > baseline['mean']:
                            anomaly_type = 'potential_attack'
                            description = f"Suspicious connection spike: {current_value} connections (normal: {baseline['mean']:.0f}). Possible port scan or DDoS attempt."
                        else:
                            anomaly_type = 'connection_anomaly'
                            description = f"Unusual {direction} in {metric.replace('_', ' ')}: {current_value} (normal: {baseline['mean']:.0f} ± {baseline['stdev']:.0f})"
                        
                        anomalies.append(NetworkAnomaly(
                            timestamp=timestamp,
                            anomaly_type=anomaly_type,
                            severity=severity,
                            description=description,
                            metric_name=metric,
                            current_value=current_value,
                            expected_range=(baseline['mean'] - 2*baseline['stdev'], baseline['mean'] + 2*baseline['stdev']),
                            confidence=min(z_score / 3, 1.0)
                        ))
        
        return anomalies
    
    def _detect_latency_anomalies(self, metrics: Dict[str, Any], timestamp: str) -> List[NetworkAnomaly]:
        """Detect latency anomalies"""
        anomalies = []
        latency_metrics = metrics.get('latency_metrics', {})
        
        if 'latency' not in self.baselines:
            return anomalies
        
        latency_baselines = self.baselines['latency']
        
        for metric, current_value in latency_metrics.items():
            if current_value < 0:  # Skip failed pings
                continue
                
            if metric in latency_baselines:
                baseline = latency_baselines[metric]
                
                if baseline['stdev'] > 0:
                    z_score = (current_value - baseline['mean']) / baseline['stdev']
                    
                    if z_score > 2:  # Only care about increases in latency
                        severity = 'critical' if z_score > 4 else 'high' if z_score > 3 else 'medium'
                        
                        target = metric.replace('_latency_ms', '').replace('_', ' ').title()
                        
                        anomalies.append(NetworkAnomaly(
                            timestamp=timestamp,
                            anomaly_type='latency_anomaly',
                            severity=severity,
                            description=f"High latency to {target}: {current_value:.1f}ms (normal: {baseline['mean']:.1f} ± {baseline['stdev']:.1f}ms)",
                            metric_name=metric,
                            current_value=current_value,
                            expected_range=(baseline['mean'] - baseline['stdev'], baseline['mean'] + 2*baseline['stdev']),
                            confidence=min(z_score / 4, 1.0)
                        ))
        
        return anomalies
    
    def _detect_packet_loss_anomalies(self, metrics: Dict[str, Any], timestamp: str) -> List[NetworkAnomaly]:
        """Detect packet loss anomalies"""
        anomalies = []
        packet_stats = metrics.get('packet_stats', {})
        
        if 'overall' not in packet_stats or 'packet_loss' not in self.baselines:
            return anomalies
        
        overall_stats = packet_stats['overall']
        packet_loss_baselines = self.baselines['packet_loss']
        
        for metric in ['error_rate', 'drop_rate']:
            if metric in overall_stats and metric in packet_loss_baselines:
                current_value = overall_stats[metric]
                baseline = packet_loss_baselines[metric]
                
                # Packet loss is always concerning if above baseline
                if current_value > baseline['mean'] + 2 * baseline['stdev']:
                    if current_value > 5:  # >5% is critical
                        severity = 'critical'
                    elif current_value > 2:  # >2% is high
                        severity = 'high'
                    else:
                        severity = 'medium'
                    
                    metric_name = metric.replace('_', ' ').title()
                    
                    anomalies.append(NetworkAnomaly(
                        timestamp=timestamp,
                        anomaly_type='packet_loss_anomaly',
                        severity=severity,
                        description=f"High {metric_name}: {current_value:.2f}% (normal: {baseline['mean']:.2f}%)",
                        metric_name=metric,
                        current_value=current_value,
                        expected_range=(0, baseline['mean'] + baseline['stdev']),
                        confidence=0.9
                    ))
        
        return anomalies
    
    def _detect_suspicious_patterns(self, metrics: Dict[str, Any], timestamp: str) -> List[NetworkAnomaly]:
        """Detect suspicious network patterns"""
        anomalies = []
        
        # Look for potential security threats
        connections = metrics.get('connections', {})
        
        # Too many TIME_WAIT connections (potential SYN flood)
        if connections.get('tcp_time_wait', 0) > 1000:
            anomalies.append(NetworkAnomaly(
                timestamp=timestamp,
                anomaly_type='potential_attack',
                severity='high',
                description=f"Excessive TIME_WAIT connections: {connections['tcp_time_wait']}. Possible SYN flood attack.",
                metric_name='tcp_time_wait',
                current_value=connections['tcp_time_wait'],
                expected_range=(0, 100),
                confidence=0.8
            ))
        
        # Unusual ratio of established to listening connections
        tcp_established = connections.get('tcp_established', 0)
        tcp_listen = connections.get('tcp_listen', 0)
        
        if tcp_listen > 0 and tcp_established / tcp_listen > 50:
            anomalies.append(NetworkAnomaly(
                timestamp=timestamp,
                anomaly_type='potential_attack',
                severity='medium',
                description=f"High connection ratio: {tcp_established} established vs {tcp_listen} listening. Possible connection flood.",
                metric_name='connection_ratio',
                current_value=tcp_established / tcp_listen,
                expected_range=(0, 10),
                confidence=0.7
            ))
        
        return anomalies
    
    def get_network_health_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall network health score (0-100)"""
        if not self.baseline_established:
            return 50.0  # Neutral score if no baseline
        
        score = 100.0
        penalties = []
        
        # Check latency health
        latency_metrics = metrics.get('latency_metrics', {})
        for metric, value in latency_metrics.items():
            if value > 0 and metric in self.baselines.get('latency', {}):
                baseline = self.baselines['latency'][metric]
                if value > baseline['mean'] + 2 * baseline['stdev']:
                    penalty = min(20, (value - baseline['mean']) / baseline['mean'] * 10)
                    penalties.append(penalty)
        
        # Check packet loss health
        packet_stats = metrics.get('packet_stats', {})
        if 'overall' in packet_stats:
            for metric in ['error_rate', 'drop_rate']:
                value = packet_stats['overall'].get(metric, 0)
                if value > 1:  # >1% is concerning
                    penalty = min(30, value * 6)  # Scale penalty with loss rate
                    penalties.append(penalty)
        
        # Apply penalties
        total_penalty = sum(penalties)
        score = max(0, score - total_penalty)
        
        return score
