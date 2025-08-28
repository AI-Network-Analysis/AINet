"""
Network Metrics Collector
Collects various network and system metrics from the local machine.
"""

import psutil
import netifaces
import time
import socket
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
import subprocess
import logging

logger = logging.getLogger(__name__)

@dataclass
class NetworkMetrics:
    """Network metrics data structure"""
    timestamp: str
    hostname: str
    interfaces: Dict[str, Dict[str, Any]]
    connections: Dict[str, int]
    bandwidth: Dict[str, Dict[str, int]]
    system_metrics: Dict[str, float]
    latency_metrics: Dict[str, float]
    packet_stats: Dict[str, Dict[str, int]]

class MetricsCollector:
    """Collects various network and system metrics"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.hostname = socket.gethostname()
        self.previous_net_stats = {}
        
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
            
            # Calculate error rates
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
    
    def collect_all_metrics(self) -> NetworkMetrics:
        """Collect all network and system metrics"""
        timestamp = datetime.utcnow().isoformat()
        
        logger.info("Collecting network metrics...")
        
        interfaces = self.collect_interface_info()
        connections = self.collect_connection_stats()
        bandwidth = self.collect_bandwidth_stats()
        system_metrics = self.collect_system_metrics()
        latency_metrics = self.collect_latency_metrics()
        packet_stats = self.collect_packet_stats()
        
        return NetworkMetrics(
            timestamp=timestamp,
            hostname=self.hostname,
            interfaces=interfaces,
            connections=connections,
            bandwidth=bandwidth,
            system_metrics=system_metrics,
            latency_metrics=latency_metrics,
            packet_stats=packet_stats
        )
    
    def to_dict(self, metrics: NetworkMetrics) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return asdict(metrics)
