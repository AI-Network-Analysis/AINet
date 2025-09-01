"""
System Resource Monitor
Specialized monitoring for CPU, memory, disk, and system resources.
"""

import psutil
import time
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import deque
import logging

logger = logging.getLogger(__name__)

@dataclass
class SystemAnomaly:
    """System resource anomaly detection result"""
    timestamp: str
    anomaly_type: str
    severity: str  # low, medium, high, critical
    description: str
    metric_name: str
    current_value: float
    expected_range: Tuple[float, float]
    confidence: float

class SystemResourceMonitor:
    """Advanced system resource monitoring and anomaly detection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.history_size = config.get('history_size', 100)
        
        # Historical data storage
        self.cpu_history = deque(maxlen=self.history_size)
        self.memory_history = deque(maxlen=self.history_size)
        self.disk_history = deque(maxlen=self.history_size)
        self.load_history = deque(maxlen=self.history_size)
        
        # Thresholds for alerts
        self.cpu_warning_threshold = config.get('cpu_warning_threshold', 80)
        self.cpu_critical_threshold = config.get('cpu_critical_threshold', 95)
        self.memory_warning_threshold = config.get('memory_warning_threshold', 85)
        self.memory_critical_threshold = config.get('memory_critical_threshold', 95)
        self.disk_warning_threshold = config.get('disk_warning_threshold', 85)
        self.disk_critical_threshold = config.get('disk_critical_threshold', 95)
        
    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics"""
        metrics = {}
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_times = psutil.cpu_times()
            cpu_freq = psutil.cpu_freq()
            
            metrics['cpu'] = {
                'percent': cpu_percent,
                'count': psutil.cpu_count(),
                'count_logical': psutil.cpu_count(logical=True),
                'times': {
                    'user': cpu_times.user,
                    'system': cpu_times.system,
                    'idle': cpu_times.idle,
                    'iowait': getattr(cpu_times, 'iowait', 0),
                },
                'frequency': {
                    'current': cpu_freq.current if cpu_freq else 0,
                    'min': cpu_freq.min if cpu_freq else 0,
                    'max': cpu_freq.max if cpu_freq else 0,
                } if cpu_freq else None
            }
            
            # Per-CPU metrics
            cpu_percents = psutil.cpu_percent(percpu=True)
            metrics['cpu']['per_cpu'] = cpu_percents
            metrics['cpu']['max_cpu'] = max(cpu_percents) if cpu_percents else 0
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            metrics['memory'] = {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'free': memory.free,
                'percent': memory.percent,
                'buffers': getattr(memory, 'buffers', 0),
                'cached': getattr(memory, 'cached', 0),
                'swap': {
                    'total': swap.total,
                    'used': swap.used,
                    'free': swap.free,
                    'percent': swap.percent,
                    'sin': swap.sin,
                    'sout': swap.sout
                }
            }
            
            # Disk metrics
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            metrics['disk'] = {
                'usage': {
                    'total': disk_usage.total,
                    'used': disk_usage.used,
                    'free': disk_usage.free,
                    'percent': (disk_usage.used / disk_usage.total) * 100
                }
            }
            
            if disk_io:
                metrics['disk']['io'] = {
                    'read_count': disk_io.read_count,
                    'write_count': disk_io.write_count,
                    'read_bytes': disk_io.read_bytes,
                    'write_bytes': disk_io.write_bytes,
                    'read_time': disk_io.read_time,
                    'write_time': disk_io.write_time
                }
            
            # Load average (Linux/Unix only)
            if hasattr(psutil, 'getloadavg'):
                load_avg = psutil.getloadavg()
                metrics['load_average'] = {
                    '1min': load_avg[0],
                    '5min': load_avg[1],
                    '15min': load_avg[2]
                }
            
            # Process information
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    pinfo = proc.info
                    if pinfo['cpu_percent'] > 0:  # Only include active processes
                        processes.append(pinfo)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            # Top CPU and memory consumers
            processes_by_cpu = sorted(processes, key=lambda x: x['cpu_percent'], reverse=True)[:10]
            processes_by_memory = sorted(processes, key=lambda x: x['memory_percent'], reverse=True)[:10]
            
            metrics['processes'] = {
                'total_count': len(psutil.pids()),
                'top_cpu': processes_by_cpu,
                'top_memory': processes_by_memory
            }
            
            # Boot time and uptime
            boot_time = psutil.boot_time()
            uptime = time.time() - boot_time
            
            metrics['system'] = {
                'boot_time': boot_time,
                'uptime_seconds': uptime,
                'uptime_days': uptime / 86400
            }
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
        
        return metrics
    
    def analyze_system_metrics(self, metrics: Dict[str, Any]) -> List[SystemAnomaly]:
        """Analyze system metrics for anomalies"""
        anomalies = []
        timestamp = datetime.utcnow().isoformat()
        
        # Update historical data
        self._update_history(metrics)
        
        # Check CPU anomalies
        anomalies.extend(self._check_cpu_anomalies(metrics, timestamp))
        
        # Check memory anomalies
        anomalies.extend(self._check_memory_anomalies(metrics, timestamp))
        
        # Check disk anomalies
        anomalies.extend(self._check_disk_anomalies(metrics, timestamp))
        
        # Check load average anomalies
        anomalies.extend(self._check_load_anomalies(metrics, timestamp))
        
        # Check process anomalies
        anomalies.extend(self._check_process_anomalies(metrics, timestamp))
        
        return anomalies
    
    def _update_history(self, metrics: Dict[str, Any]):
        """Update historical data"""
        if 'cpu' in metrics:
            self.cpu_history.append({
                'timestamp': time.time(),
                'percent': metrics['cpu']['percent'],
                'max_cpu': metrics['cpu']['max_cpu']
            })
        
        if 'memory' in metrics:
            self.memory_history.append({
                'timestamp': time.time(),
                'percent': metrics['memory']['percent'],
                'swap_percent': metrics['memory']['swap']['percent']
            })
        
        if 'disk' in metrics and 'usage' in metrics['disk']:
            self.disk_history.append({
                'timestamp': time.time(),
                'percent': metrics['disk']['usage']['percent']
            })
        
        if 'load_average' in metrics:
            self.load_history.append({
                'timestamp': time.time(),
                'load_1min': metrics['load_average']['1min'],
                'load_5min': metrics['load_average']['5min'],
                'load_15min': metrics['load_average']['15min']
            })
    
    def _check_cpu_anomalies(self, metrics: Dict[str, Any], timestamp: str) -> List[SystemAnomaly]:
        """Check for CPU-related anomalies"""
        anomalies = []
        
        if 'cpu' not in metrics:
            return anomalies
        
        cpu_data = metrics['cpu']
        cpu_percent = cpu_data['percent']
        max_cpu = cpu_data['max_cpu']
        
        # High CPU usage
        if cpu_percent >= self.cpu_critical_threshold:
            anomalies.append(SystemAnomaly(
                timestamp=timestamp,
                anomaly_type='high_cpu_usage',
                severity='critical',
                description=f"Critical CPU usage: {cpu_percent:.1f}% (threshold: {self.cpu_critical_threshold}%)",
                metric_name='cpu_percent',
                current_value=cpu_percent,
                expected_range=(0, self.cpu_warning_threshold),
                confidence=0.95
            ))
        elif cpu_percent >= self.cpu_warning_threshold:
            anomalies.append(SystemAnomaly(
                timestamp=timestamp,
                anomaly_type='high_cpu_usage',
                severity='high',
                description=f"High CPU usage: {cpu_percent:.1f}% (threshold: {self.cpu_warning_threshold}%)",
                metric_name='cpu_percent',
                current_value=cpu_percent,
                expected_range=(0, self.cpu_warning_threshold),
                confidence=0.90
            ))
        
        # CPU core imbalance
        if max_cpu > 90 and len(cpu_data.get('per_cpu', [])) > 1:
            per_cpu = cpu_data['per_cpu']
            cpu_variance = statistics.variance(per_cpu) if len(per_cpu) > 1 else 0
            
            if cpu_variance > 500:  # High variance indicates imbalance
                anomalies.append(SystemAnomaly(
                    timestamp=timestamp,
                    anomaly_type='cpu_imbalance',
                    severity='medium',
                    description=f"CPU core imbalance detected. Max core: {max_cpu:.1f}%, variance: {cpu_variance:.1f}",
                    metric_name='cpu_variance',
                    current_value=cpu_variance,
                    expected_range=(0, 200),
                    confidence=0.80
                ))
        
        # Sustained high CPU (trend analysis)
        if len(self.cpu_history) >= 5:
            recent_cpu = [entry['percent'] for entry in list(self.cpu_history)[-5:]]
            avg_recent = statistics.mean(recent_cpu)
            
            if avg_recent > self.cpu_warning_threshold:
                anomalies.append(SystemAnomaly(
                    timestamp=timestamp,
                    anomaly_type='sustained_high_cpu',
                    severity='high',
                    description=f"Sustained high CPU usage: {avg_recent:.1f}% average over last 5 measurements",
                    metric_name='cpu_sustained',
                    current_value=avg_recent,
                    expected_range=(0, self.cpu_warning_threshold),
                    confidence=0.85
                ))
        
        return anomalies
    
    def _check_memory_anomalies(self, metrics: Dict[str, Any], timestamp: str) -> List[SystemAnomaly]:
        """Check for memory-related anomalies"""
        anomalies = []
        
        if 'memory' not in metrics:
            return anomalies
        
        memory_data = metrics['memory']
        memory_percent = memory_data['percent']
        swap_percent = memory_data['swap']['percent']
        
        # High memory usage
        if memory_percent >= self.memory_critical_threshold:
            anomalies.append(SystemAnomaly(
                timestamp=timestamp,
                anomaly_type='high_memory_usage',
                severity='critical',
                description=f"Critical memory usage: {memory_percent:.1f}% (threshold: {self.memory_critical_threshold}%)",
                metric_name='memory_percent',
                current_value=memory_percent,
                expected_range=(0, self.memory_warning_threshold),
                confidence=0.95
            ))
        elif memory_percent >= self.memory_warning_threshold:
            anomalies.append(SystemAnomaly(
                timestamp=timestamp,
                anomaly_type='high_memory_usage',
                severity='high',
                description=f"High memory usage: {memory_percent:.1f}% (threshold: {self.memory_warning_threshold}%)",
                metric_name='memory_percent',
                current_value=memory_percent,
                expected_range=(0, self.memory_warning_threshold),
                confidence=0.90
            ))
        
        # High swap usage (indicates memory pressure)
        if swap_percent > 50:
            severity = 'critical' if swap_percent > 80 else 'high'
            anomalies.append(SystemAnomaly(
                timestamp=timestamp,
                anomaly_type='high_swap_usage',
                severity=severity,
                description=f"High swap usage: {swap_percent:.1f}% - system may be under memory pressure",
                metric_name='swap_percent',
                current_value=swap_percent,
                expected_range=(0, 20),
                confidence=0.90
            ))
        
        return anomalies
    
    def _check_disk_anomalies(self, metrics: Dict[str, Any], timestamp: str) -> List[SystemAnomaly]:
        """Check for disk-related anomalies"""
        anomalies = []
        
        if 'disk' not in metrics or 'usage' not in metrics['disk']:
            return anomalies
        
        disk_percent = metrics['disk']['usage']['percent']
        
        # High disk usage
        if disk_percent >= self.disk_critical_threshold:
            anomalies.append(SystemAnomaly(
                timestamp=timestamp,
                anomaly_type='high_disk_usage',
                severity='critical',
                description=f"Critical disk usage: {disk_percent:.1f}% (threshold: {self.disk_critical_threshold}%)",
                metric_name='disk_percent',
                current_value=disk_percent,
                expected_range=(0, self.disk_warning_threshold),
                confidence=0.95
            ))
        elif disk_percent >= self.disk_warning_threshold:
            anomalies.append(SystemAnomaly(
                timestamp=timestamp,
                anomaly_type='high_disk_usage',
                severity='high',
                description=f"High disk usage: {disk_percent:.1f}% (threshold: {self.disk_warning_threshold}%)",
                metric_name='disk_percent',
                current_value=disk_percent,
                expected_range=(0, self.disk_warning_threshold),
                confidence=0.90
            ))
        
        return anomalies
    
    def _check_load_anomalies(self, metrics: Dict[str, Any], timestamp: str) -> List[SystemAnomaly]:
        """Check for load average anomalies"""
        anomalies = []
        
        if 'load_average' not in metrics:
            return anomalies
        
        load_data = metrics['load_average']
        cpu_count = metrics.get('cpu', {}).get('count', 1)
        
        # Load average relative to CPU count
        load_1min = load_data['1min']
        load_5min = load_data['5min']
        load_15min = load_data['15min']
        
        # High load (1-minute)
        if load_1min > cpu_count * 2:
            anomalies.append(SystemAnomaly(
                timestamp=timestamp,
                anomaly_type='high_load_average',
                severity='critical',
                description=f"Very high 1-minute load: {load_1min:.2f} (CPUs: {cpu_count})",
                metric_name='load_1min',
                current_value=load_1min,
                expected_range=(0, cpu_count),
                confidence=0.90
            ))
        elif load_1min > cpu_count * 1.5:
            anomalies.append(SystemAnomaly(
                timestamp=timestamp,
                anomaly_type='high_load_average',
                severity='high',
                description=f"High 1-minute load: {load_1min:.2f} (CPUs: {cpu_count})",
                metric_name='load_1min',
                current_value=load_1min,
                expected_range=(0, cpu_count),
                confidence=0.85
            ))
        
        # Increasing load trend
        if load_1min > load_5min > load_15min and load_1min > cpu_count:
            anomalies.append(SystemAnomaly(
                timestamp=timestamp,
                anomaly_type='increasing_load_trend',
                severity='medium',
                description=f"Increasing load trend: 1min({load_1min:.2f}) > 5min({load_5min:.2f}) > 15min({load_15min:.2f})",
                metric_name='load_trend',
                current_value=load_1min - load_15min,
                expected_range=(0, 0.5),
                confidence=0.75
            ))
        
        return anomalies
    
    def _check_process_anomalies(self, metrics: Dict[str, Any], timestamp: str) -> List[SystemAnomaly]:
        """Check for process-related anomalies"""
        anomalies = []
        
        if 'processes' not in metrics:
            return anomalies
        
        processes_data = metrics['processes']
        
        # Check for processes consuming excessive resources
        for proc in processes_data.get('top_cpu', [])[:3]:  # Top 3 CPU consumers
            if proc['cpu_percent'] > 50:
                anomalies.append(SystemAnomaly(
                    timestamp=timestamp,
                    anomaly_type='high_process_cpu',
                    severity='medium',
                    description=f"Process '{proc['name']}' (PID: {proc['pid']}) using high CPU: {proc['cpu_percent']:.1f}%",
                    metric_name='process_cpu',
                    current_value=proc['cpu_percent'],
                    expected_range=(0, 25),
                    confidence=0.80
                ))
        
        for proc in processes_data.get('top_memory', [])[:3]:  # Top 3 memory consumers
            if proc['memory_percent'] > 20:
                anomalies.append(SystemAnomaly(
                    timestamp=timestamp,
                    anomaly_type='high_process_memory',
                    severity='medium',
                    description=f"Process '{proc['name']}' (PID: {proc['pid']}) using high memory: {proc['memory_percent']:.1f}%",
                    metric_name='process_memory',
                    current_value=proc['memory_percent'],
                    expected_range=(0, 10),
                    confidence=0.80
                ))
        
        return anomalies
    
    def get_system_health_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall system health score (0-100)"""
        score = 100.0
        penalties = []
        
        # CPU penalty
        if 'cpu' in metrics:
            cpu_percent = metrics['cpu']['percent']
            if cpu_percent > self.cpu_warning_threshold:
                penalty = (cpu_percent - self.cpu_warning_threshold) * 2
                penalties.append(min(penalty, 40))
        
        # Memory penalty
        if 'memory' in metrics:
            memory_percent = metrics['memory']['percent']
            if memory_percent > self.memory_warning_threshold:
                penalty = (memory_percent - self.memory_warning_threshold) * 2
                penalties.append(min(penalty, 30))
        
        # Disk penalty
        if 'disk' in metrics and 'usage' in metrics['disk']:
            disk_percent = metrics['disk']['usage']['percent']
            if disk_percent > self.disk_warning_threshold:
                penalty = (disk_percent - self.disk_warning_threshold) * 1.5
                penalties.append(min(penalty, 20))
        
        # Load average penalty
        if 'load_average' in metrics and 'cpu' in metrics:
            load_1min = metrics['load_average']['1min']
            cpu_count = metrics['cpu']['count']
            if load_1min > cpu_count:
                penalty = (load_1min - cpu_count) * 10
                penalties.append(min(penalty, 30))
        
        # Apply penalties
        total_penalty = sum(penalties)
        score = max(0, score - total_penalty)
        
        return score
