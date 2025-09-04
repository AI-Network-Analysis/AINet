"""
Enhanced Network Metrics Collector
Collects various network and system metrics with configurable monitoring focus.
"""

import psutil
import netifaces
import time
import socket
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
import logging
import subprocess
from typing import Dict, List, Optional, Any
from datetime import datetime
import subprocess
import logging

# Import our new monitoring modules
try:
    from .network_anomaly_detector import NetworkAnomalyDetector, NetworkAnomaly
    from .system_monitor import SystemResourceMonitor, SystemAnomaly
except ImportError:
    # For standalone usage
    pass

# Try to import configuration manager
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
    from config.monitoring_config import ConfigurationManager, MonitoringMode
except ImportError:
    # Fallback to basic configuration
    ConfigurationManager = None
    MonitoringMode = None

logger = logging.getLogger(__name__)

@dataclass
class EnhancedMetrics:
    """Enhanced metrics data structure"""
    timestamp: str
    hostname: str
    monitoring_mode: str
    
    # Network metrics (if network monitoring enabled)
    interfaces: Optional[Dict[str, Dict[str, Any]]] = None
    connections: Optional[Dict[str, int]] = None
    bandwidth: Optional[Dict[str, Dict[str, int]]] = None
    latency_metrics: Optional[Dict[str, float]] = None
    packet_stats: Optional[Dict[str, Dict[str, int]]] = None
    network_health_score: Optional[float] = None
    network_anomalies: Optional[List[Dict[str, Any]]] = None
    
    # System metrics (if system monitoring enabled)
    system_metrics: Optional[Dict[str, Any]] = None
    system_health_score: Optional[float] = None
    system_anomalies: Optional[List[Dict[str, Any]]] = None

class EnhancedMetricsCollector:
    """Enhanced metrics collector with configurable monitoring focus"""
    
    def __init__(self, config_path: str = None):
        self.config_manager = ConfigurationManager(config_path) if config_path else ConfigurationManager()
        self.config = self.config_manager.get_config()
        self.hostname = socket.gethostname()
        
        # Initialize monitoring components based on configuration
        self.network_detector = None
        self.system_monitor = None
        
        if self.config.network.enabled:
            network_config = {
                'sensitivity': self.config.network.sensitivity,
                'baseline_samples': self.config.network.baseline_samples,
                'history_size': self.config.network.history_size
            }
            self.network_detector = NetworkAnomalyDetector(network_config)
        
        if self.config.system.enabled:
            system_config = {
                'cpu_warning_threshold': self.config.system.cpu_warning_threshold,
                'cpu_critical_threshold': self.config.system.cpu_critical_threshold,
                'memory_warning_threshold': self.config.system.memory_warning_threshold,
                'memory_critical_threshold': self.config.system.memory_critical_threshold,
                'disk_warning_threshold': self.config.system.disk_warning_threshold,
                'disk_critical_threshold': self.config.system.disk_critical_threshold
            }
            self.system_monitor = SystemResourceMonitor(system_config)
        
        # Store previous network stats for rate calculations (used by _get_interface_metrics)
        self.previous_net_stats = {}
        # Track previous connection count and timestamp for connection rate
        self._prev_conn_snapshot = {"ts": None, "total": 0, "established": 0}

        logger.info(f"Enhanced metrics collector initialized with mode: {self.config.mode.value}")
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all network metrics based on current mode."""
        try:
            metrics = {
                'timestamp': time.time(),
                'mode': self.config.mode.value,
                'interfaces': [],
                'connections': [],
                'system_info': {}
            }
            
            # Collect network data if in network mode or both
            if self.config.mode in [MonitoringMode.NETWORK_FOCUS, MonitoringMode.BALANCED] if MonitoringMode else True:
                interfaces = self._get_network_interfaces()
                metrics['connections'] = self._get_network_connections()
                # Compute connection rates for DoS signals
                try:
                    metrics['connection_rates'] = self._get_connection_rates(metrics['connections'])
                except Exception:
                    metrics['connection_rates'] = {"new_connections_per_s": 0.0, "established_delta_per_s": 0.0}
                
                for interface in interfaces:
                    metrics['interfaces'].append(self._get_interface_metrics(interface))
                
                # Collect network anomalies if detector is available
                if hasattr(self, 'network_detector'):
                    network_data = {
                        'interfaces': metrics['interfaces'],
                        'connections': metrics['connections']
                    }
                    anomalies = self.network_detector.analyze_network_metrics(network_data)
                    metrics['network_anomalies'] = [asdict(anomaly) for anomaly in anomalies]
            
            # Collect system data whenever system monitoring is enabled (independent of mode)
            try:
                system_enabled = getattr(self.config, 'system', None) and getattr(self.config.system, 'enabled', True)
            except Exception:
                system_enabled = True
            if system_enabled:
                metrics['system_info'] = self._get_system_info()
                
                # Collect system anomalies if monitor is available
                if hasattr(self, 'system_monitor'):
                    system_data = metrics['system_info']
                    anomalies = self.system_monitor.analyze_system_metrics(system_data)
                    metrics['system_anomalies'] = [asdict(anomaly) for anomaly in anomalies]
            
            return metrics
            
        except Exception as e:
            logging.error(f"Error collecting metrics: {e}")
            raise

    def to_dict(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Convert metrics to dictionary format (already in dict format)."""
        return metrics

    def _get_network_interfaces(self) -> List[str]:
        """Get list of network interfaces."""
        try:
            return list(netifaces.interfaces())
        except Exception as e:
            logger.warning(f"Error getting network interfaces: {e}")
            return []

    def _get_interface_metrics(self, interface: str) -> Dict[str, Any]:
        """Get metrics for a specific network interface."""
        try:
            # Get interface addresses
            addresses = netifaces.ifaddresses(interface)

            # Get network statistics and ARP neighbors
            net_stats = psutil.net_io_counters(pernic=True).get(interface, None)
            arp_neighbors = self._get_arp_table().get(interface, [])

            metrics: Dict[str, Any] = {
                'interface': interface,
                'addresses': addresses,
                'is_up': interface in netifaces.interfaces(),
                'statistics': {},
                'arp_neighbors': arp_neighbors
            }

            if net_stats:
                prev_stats = self.previous_net_stats.get(interface, {})
                current_time = time.time()

                stats = {
                    'bytes_sent': net_stats.bytes_sent,
                    'bytes_recv': net_stats.bytes_recv,
                    'packets_sent': net_stats.packets_sent,
                    'packets_recv': net_stats.packets_recv,
                    'errin': net_stats.errin,
                    'errout': net_stats.errout,
                    'dropin': net_stats.dropin,
                    'dropout': net_stats.dropout
                }

                # Rates
                if prev_stats and 'timestamp' in prev_stats:
                    time_diff = current_time - prev_stats['timestamp']
                    if time_diff > 0:
                        stats['bytes_sent_rate'] = (net_stats.bytes_sent - prev_stats.get('bytes_sent', 0)) / time_diff
                        stats['bytes_recv_rate'] = (net_stats.bytes_recv - prev_stats.get('bytes_recv', 0)) / time_diff
                        stats['packets_sent_rate'] = (net_stats.packets_sent - prev_stats.get('packets_sent', 0)) / time_diff
                        stats['packets_recv_rate'] = (net_stats.packets_recv - prev_stats.get('packets_recv', 0)) / time_diff

                metrics['statistics'] = stats

                # Update snapshot
                self.previous_net_stats[interface] = {
                    'timestamp': current_time,
                    'bytes_sent': net_stats.bytes_sent,
                    'bytes_recv': net_stats.bytes_recv,
                    'packets_sent': net_stats.packets_sent,
                    'packets_recv': net_stats.packets_recv
                }

            return metrics

        except Exception as e:
            logger.warning(f"Error getting metrics for interface {interface}: {e}")
            return {'interface': interface, 'error': str(e)}

    def _get_arp_table(self) -> Dict[str, List[Dict[str, Any]]]:
        """Parse ARP table and return neighbors per interface.

        Returns: { iface: [ { ip, mac, flags }, ... ] }
        """
        result: Dict[str, List[Dict[str, Any]]] = {}
        try:
            if os.path.exists('/proc/net/arp'):
                with open('/proc/net/arp', 'r') as f:
                    lines = f.read().strip().splitlines()
                # Skip header
                for line in lines[1:]:
                    parts = line.split()
                    if len(parts) >= 6:
                        ip, hw_type, flags, mac, mask, dev = parts[:6]
                        result.setdefault(dev, []).append({
                            'ip': ip,
                            'mac': mac,
                            'flags': flags
                        })
        except Exception:
            pass
        return result

    def _get_network_connections(self) -> List[Dict[str, Any]]:
        """Get current network connections."""
        try:
            connections = []
            for conn in psutil.net_connections(kind='inet'):
                connections.append({
                    'family': conn.family.name if hasattr(conn.family, 'name') else str(conn.family),
                    'type': conn.type.name if hasattr(conn.type, 'name') else str(conn.type),
                    'local_address': f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else None,
                    'remote_address': f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else None,
                    'status': conn.status,
                    'pid': conn.pid
                })
            return connections
        except Exception as e:
            logger.warning(f"Error getting network connections: {e}")
            return []

    def _get_connection_rates(self, connections_list: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute connection rates per second using previous snapshot.

        Returns keys: new_connections_per_s, established_delta_per_s
        """
        try:
            now = time.time()
            total = len(connections_list)
            established = sum(1 for c in connections_list if c.get('status') == 'ESTABLISHED')

            prev = self._prev_conn_snapshot
            rate = {"new_connections_per_s": 0.0, "established_delta_per_s": 0.0}
            if prev["ts"] is not None:
                dt = max(0.0, now - prev["ts"])
                if dt > 0:
                    rate["new_connections_per_s"] = max(0.0, (total - prev["total"]) / dt)
                    rate["established_delta_per_s"] = (established - prev["established"]) / dt

            # Update snapshot
            self._prev_conn_snapshot = {"ts": now, "total": total, "established": established}
            return rate
        except Exception as e:
            logger.debug(f"Failed to compute connection rates: {e}")
            return {"new_connections_per_s": 0.0, "established_delta_per_s": 0.0}
    
    def _collect_network_metrics(self) -> Dict[str, Any]:
        """Collect network-specific metrics"""
        network_data = {}
        
        # Collect interface info if enabled
        if self.config.network.bandwidth_monitoring:
            network_data['interfaces'] = self.collect_interface_info()
            network_data['bandwidth'] = self.collect_bandwidth_stats()
        
        # Collect connection stats if enabled
        if self.config.network.connection_monitoring:
            network_data['connections'] = self.collect_connection_stats()
            # Also include connection rate estimates for DoS detection
            try:
                # Use fresh list to compute delta safely
                conn_list = self._get_network_connections()
                network_data['connection_rates'] = self._get_connection_rates(conn_list)
            except Exception:
                network_data['connection_rates'] = {"new_connections_per_s": 0.0, "established_delta_per_s": 0.0}
        
        # Collect latency metrics if enabled
        if self.config.network.latency_monitoring:
            network_data['latency_metrics'] = self.collect_latency_metrics()
        
        # Collect packet stats if enabled
        if self.config.network.packet_loss_monitoring:
            network_data['packet_stats'] = self.collect_packet_stats()
        
        return network_data
        
    def collect_interface_info(self) -> Dict[str, Dict[str, Any]]:
        """Collect network interface information"""
        interfaces = {}
        
        for interface in netifaces.interfaces():
            try:
                addresses = netifaces.ifaddresses(interface)
                interface_info = {
                    'name': interface,
                    'ipv4_addresses': [],
                    'ipv6_addresses': [],
                    'mac_address': None,
                    'is_up': False,
                    'mtu': None
                }
                
                # Get IPv4 addresses
                if netifaces.AF_INET in addresses:
                    for addr in addresses[netifaces.AF_INET]:
                        interface_info['ipv4_addresses'].append({
                            'addr': addr.get('addr'),
                            'netmask': addr.get('netmask'),
                            'broadcast': addr.get('broadcast')
                        })
                
                # Get IPv6 addresses
                if netifaces.AF_INET6 in addresses:
                    for addr in addresses[netifaces.AF_INET6]:
                        interface_info['ipv6_addresses'].append({
                            'addr': addr.get('addr'),
                            'netmask': addr.get('netmask')
                        })
                
                # Get MAC address
                if netifaces.AF_LINK in addresses:
                    interface_info['mac_address'] = addresses[netifaces.AF_LINK][0].get('addr')
                
                # Get interface statistics using psutil
                if interface in psutil.net_if_stats():
                    stats = psutil.net_if_stats()[interface]
                    interface_info['is_up'] = stats.isup
                    interface_info['mtu'] = stats.mtu
                    interface_info['speed'] = stats.speed
                
                interfaces[interface] = interface_info
                
            except Exception as e:
                logger.warning(f"Error collecting info for interface {interface}: {e}")
                
        return interfaces
    
    def collect_connection_stats(self) -> Dict[str, int]:
        """Collect network connection statistics"""
        connections = {
            'tcp_established': 0,
            'tcp_listen': 0,
            'tcp_time_wait': 0,
            'tcp_close_wait': 0,
            'udp_connections': 0,
            'total_connections': 0
        }
        
        try:
            net_connections = psutil.net_connections()
            for conn in net_connections:
                connections['total_connections'] += 1
                
                if conn.type == socket.SOCK_STREAM:  # TCP
                    if conn.status == psutil.CONN_ESTABLISHED:
                        connections['tcp_established'] += 1
                    elif conn.status == psutil.CONN_LISTEN:
                        connections['tcp_listen'] += 1
                    elif conn.status == psutil.CONN_TIME_WAIT:
                        connections['tcp_time_wait'] += 1
                    elif conn.status == psutil.CONN_CLOSE_WAIT:
                        connections['tcp_close_wait'] += 1
                elif conn.type == socket.SOCK_DGRAM:  # UDP
                    connections['udp_connections'] += 1
                    
        except Exception as e:
            logger.error(f"Error collecting connection stats: {e}")
            
        return connections
    
    def collect_bandwidth_stats(self) -> Dict[str, Dict[str, int]]:
        """Collect bandwidth statistics for network interfaces"""
        bandwidth_stats = {}
        
        try:
            current_stats = psutil.net_io_counters(pernic=True)
            current_time = time.time()
            
            for interface, stats in current_stats.items():
                interface_stats = {
                    'bytes_sent': stats.bytes_sent,
                    'bytes_recv': stats.bytes_recv,
                    'packets_sent': stats.packets_sent,
                    'packets_recv': stats.packets_recv,
                    'errin': stats.errin,
                    'errout': stats.errout,
                    'dropin': stats.dropin,
                    'dropout': stats.dropout
                }
                
                # Calculate rates if we have previous stats
                if interface in self.previous_net_stats:
                    prev_stats, prev_time = self.previous_net_stats[interface]
                    time_delta = current_time - prev_time
                    
                    if time_delta > 0:
                        interface_stats['bytes_sent_rate'] = int(
                            (stats.bytes_sent - prev_stats.bytes_sent) / time_delta
                        )
                        interface_stats['bytes_recv_rate'] = int(
                            (stats.bytes_recv - prev_stats.bytes_recv) / time_delta
                        )
                        interface_stats['packets_sent_rate'] = int(
                            (stats.packets_sent - prev_stats.packets_sent) / time_delta
                        )
                        interface_stats['packets_recv_rate'] = int(
                            (stats.packets_recv - prev_stats.packets_recv) / time_delta
                        )
                else:
                    # First time, set rates to 0
                    interface_stats['bytes_sent_rate'] = 0
                    interface_stats['bytes_recv_rate'] = 0
                    interface_stats['packets_sent_rate'] = 0
                    interface_stats['packets_recv_rate'] = 0
                
                bandwidth_stats[interface] = interface_stats
                # Store current stats for next calculation
                self.previous_net_stats[interface] = (stats, current_time)
                
        except Exception as e:
            logger.error(f"Error collecting bandwidth stats: {e}")
            
        return bandwidth_stats
    
    def collect_system_metrics(self) -> Dict[str, float]:
        """Collect system-level metrics"""
        metrics = {}
        
        try:
            # CPU usage
            metrics['cpu_percent'] = psutil.cpu_percent(interval=1)
            metrics['cpu_count'] = psutil.cpu_count()
            
            # Memory usage
            memory = psutil.virtual_memory()
            metrics['memory_percent'] = memory.percent
            metrics['memory_total'] = memory.total
            metrics['memory_available'] = memory.available
            metrics['memory_used'] = memory.used
            
            # Disk usage
            disk = psutil.disk_usage('/')
            metrics['disk_percent'] = (disk.used / disk.total) * 100
            metrics['disk_total'] = disk.total
            metrics['disk_used'] = disk.used
            metrics['disk_free'] = disk.free
            
            # Load average (Linux/Unix only)
            if hasattr(psutil, 'getloadavg'):
                load_avg = psutil.getloadavg()
                metrics['load_avg_1min'] = load_avg[0]
                metrics['load_avg_5min'] = load_avg[1]
                metrics['load_avg_15min'] = load_avg[2]
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            
        return metrics
    
    def collect_latency_metrics(self) -> Dict[str, float]:
        """Collect network latency metrics"""
        latency_metrics = {}
        
        # Test connectivity to common services
        test_hosts = [
            ('google_dns', '8.8.8.8'),
            ('cloudflare_dns', '1.1.1.1'),
            ('local_gateway', self._get_default_gateway())
        ]
        
        for name, host in test_hosts:
            if host:
                try:
                    latency = self._ping_host(host)
                    latency_metrics[f'{name}_latency_ms'] = latency
                except Exception as e:
                    logger.warning(f"Could not ping {name} ({host}): {e}")
                    latency_metrics[f'{name}_latency_ms'] = -1  # Indicates failure
            
        return latency_metrics
    
    def collect_packet_stats(self) -> Dict[str, Dict[str, int]]:
        """Collect packet-level statistics"""
        packet_stats = {}
        
        try:
            # Get overall network stats
            net_stats = psutil.net_io_counters()
            packet_stats['overall'] = {
                'packets_sent': net_stats.packets_sent,
                'packets_recv': net_stats.packets_recv,
                'errors_in': net_stats.errin,
                'errors_out': net_stats.errout,
                'drops_in': net_stats.dropin,
                'drops_out': net_stats.dropout
            }
            
            # Calculate error/drop rates and packet rates (per second) using previous snapshot
            total_packets = net_stats.packets_sent + net_stats.packets_recv
            if total_packets > 0:
                packet_stats['overall']['error_rate'] = (
                    (net_stats.errin + net_stats.errout) / total_packets
                ) * 100
                packet_stats['overall']['drop_rate'] = (
                    (net_stats.dropin + net_stats.dropout) / total_packets
                ) * 100
            else:
                packet_stats['overall']['error_rate'] = 0
                packet_stats['overall']['drop_rate'] = 0

            # Interface-level packet rates piggyback on collect_bandwidth_stats
            try:
                pernic = psutil.net_io_counters(pernic=True)
                now = time.time()
                if not hasattr(self, '_prev_packet_pernic'):
                    self._prev_packet_pernic = {}
                for nic, s in pernic.items():
                    prev = self._prev_packet_pernic.get(nic)
                    if prev:
                        dt = max(0.0, now - prev['ts'])
                        if dt > 0:
                            packet_stats[nic] = {
                                'packets_sent_rate': int((s.packets_sent - prev['packets_sent']) / dt),
                                'packets_recv_rate': int((s.packets_recv - prev['packets_recv']) / dt),
                                'errors_in_rate': int((s.errin - prev['errin']) / dt),
                                'errors_out_rate': int((s.errout - prev['errout']) / dt),
                                'drops_in_rate': int((s.dropin - prev['dropin']) / dt),
                                'drops_out_rate': int((s.dropout - prev['dropout']) / dt),
                            }
                    self._prev_packet_pernic[nic] = {
                        'ts': now,
                        'packets_sent': s.packets_sent,
                        'packets_recv': s.packets_recv,
                        'errin': s.errin,
                        'errout': s.errout,
                        'dropin': s.dropin,
                        'dropout': s.dropout,
                    }
            except Exception:
                pass
                
        except Exception as e:
            logger.error(f"Error collecting packet stats: {e}")
            
        return packet_stats
    
    def _get_default_gateway(self) -> Optional[str]:
        """Get the default gateway IP address"""
        try:
            gateways = netifaces.gateways()
            default_gateway = gateways.get('default')
            if default_gateway and netifaces.AF_INET in default_gateway:
                return default_gateway[netifaces.AF_INET][0]
        except Exception as e:
            logger.warning(f"Could not get default gateway: {e}")
        return None
    
    def _ping_host(self, host: str, timeout: int = 5) -> float:
        """Ping a host and return latency in milliseconds"""
        try:
            result = subprocess.run(
                ['ping', '-c', '1', '-W', str(timeout * 1000), host],
                capture_output=True,
                text=True,
                timeout=timeout + 1
            )
            
            if result.returncode == 0:
                # Parse ping output to extract latency
                output_lines = result.stdout.split('\n')
                for line in output_lines:
                    if 'time=' in line:
                        time_part = line.split('time=')[1].split()[0]
                        return float(time_part)
            
            return -1  # Ping failed
            
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError) as e:
            logger.warning(f"Ping failed for {host}: {e}")
            return -1

    def _get_system_info(self) -> Dict[str, Any]:
        """Get basic system information."""
        try:
            import psutil
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)
            vm = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': vm.percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'load_average': {
                    '1min': load_avg[0],
                    '5min': load_avg[1],
                    '15min': load_avg[2]
                } if load_avg else {'1min': 0, '5min': 0, '15min': 0},
                'uptime': time.time() - psutil.boot_time(),
                'memory': {
                    'total': vm.total,
                    'available': vm.available,
                    'used': vm.used,
                    'percent': vm.percent,
                    'swap': {
                        'total': swap.total,
                        'used': swap.used,
                        'free': swap.free,
                        'percent': swap.percent
                    }
                },
                'cpu_count': psutil.cpu_count(),
                'boot_time': psutil.boot_time(),
                'disk': {
                    'total': psutil.disk_usage('/').total,
                    'used': psutil.disk_usage('/').used,
                    'free': psutil.disk_usage('/').free,
                    'percent': psutil.disk_usage('/').percent
                }
            }
        except ImportError:
            return {
                'cpu_percent': None,
                'memory_percent': None,
                'disk_usage': None,
                'load_average': {'1min': 0, '5min': 0, '15min': 0},
                'uptime': None,
                'memory': {
                    'total': 0,
                    'available': 0,
                    'used': 0,
                    'percent': 0,
                    'swap': {'total': 0, 'used': 0, 'free': 0, 'percent': 0}
                },
                'cpu_count': None,
                'boot_time': None,
                'disk': {'total': 0, 'used': 0, 'free': 0, 'percent': 0}
            }
