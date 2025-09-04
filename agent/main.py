#!/usr/bin/env python3
"""
AINet Network Monitoring Agent
Main application that runs on client machines to collect and send network metrics.
"""

import asyncio
import signal
import sys
import os
import yaml
import logging
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from network_monitor.metrics_collector import EnhancedMetricsCollector
from data_collector.sender import DataSender

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NetworkAgent:
    """Main agent application"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.running = False
        self.metrics_collector = None
        self.data_sender = None
        
        # Agent information
        self.agent_info = {
            'hostname': socket.gethostname(),
            'version': '1.0.0',
            'start_time': datetime.utcnow().isoformat()
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
                return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            sys.exit(1)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def _setup_logging(self):
        """Setup logging based on configuration"""
        log_config = self.config.get('logging', {})
        level = getattr(logging, log_config.get('level', 'INFO').upper())
        
        # Create log directory if needed
        log_file = log_config.get('file')
        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        root_logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            try:
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(console_format)
                root_logger.addHandler(file_handler)
                logger.info(f"Logging to file: {log_file}")
            except Exception as e:
                logger.warning(f"Failed to setup file logging: {e}")
    
    async def initialize(self):
        """Initialize agent components"""
        logger.info("Initializing AINet Network Agent...")
        
        # Setup logging
        self._setup_logging()
        
        # Initialize metrics collector
        self.metrics_collector = EnhancedMetricsCollector()
        logger.info("Metrics collector initialized")
        
        # Initialize data sender
        sender_config = self.config.get('sender', {})
        self.data_sender = DataSender(sender_config)
        logger.info("Data sender initialized")
        
        # Test connection to server
        logger.info("Testing connection to central server...")
        if await self.data_sender.test_connection():
            logger.info("Successfully connected to central server")
        else:
            logger.warning("Could not connect to central server - will retry during operation")
        
        logger.info("Agent initialization complete")
    
    async def run_collection_cycle(self):
        """Run one complete metrics collection cycle"""
        try:
            logger.debug("Starting metrics collection cycle...")
            
            # Collect metrics using enhanced collector
            raw_metrics = self.metrics_collector.collect_all_metrics()
            
            # Transform to server-compatible format
            metrics_dict = self.transform_metrics_for_server(raw_metrics)
            
            # Send metrics
            success = await self.data_sender.send_metrics(metrics_dict)
            
            if success:
                logger.debug("Metrics collection cycle completed successfully")
            else:
                logger.warning("Failed to send metrics - queued for later retry")
                
        except Exception as e:
            logger.error(f"Error in collection cycle: {e}")

    def transform_metrics_for_server(self, raw_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Transform enhanced metrics format to server-compatible format."""
        try:
            # Convert timestamp to string
            timestamp = str(raw_metrics.get('timestamp', time.time()))
            
            # Transform interfaces from list to dict format
            interfaces_dict = {}
            for interface_data in raw_metrics.get('interfaces', []):
                if isinstance(interface_data, dict) and 'interface' in interface_data:
                    iface_name = interface_data['interface']
                    stats = interface_data.get('statistics', {})
                    interfaces_dict[iface_name] = {
                        'bytes_sent': stats.get('bytes_sent', 0),
                        'bytes_recv': stats.get('bytes_recv', 0),
                        # Optional rate-based metrics (per second)
                        'bytes_sent_rate': stats.get('bytes_sent_rate', 0),
                        'bytes_recv_rate': stats.get('bytes_recv_rate', 0),
                        'packets_sent_rate': stats.get('packets_sent_rate', 0),
                        'packets_recv_rate': stats.get('packets_recv_rate', 0),
                        'packets_sent': stats.get('packets_sent', 0),
                        'packets_recv': stats.get('packets_recv', 0),
                        'errors_in': stats.get('errin', 0),
                        'errors_out': stats.get('errout', 0),
                        'drops_in': stats.get('dropin', 0),
                        'drops_out': stats.get('dropout', 0)
                    }
            
            # Transform connections from list to server-expected counts
            connections_list = raw_metrics.get('connections', [])
            tcp_total = len([c for c in connections_list if c.get('type') == 'SOCK_STREAM'])
            connections_dict = {
                'total_connections': len(connections_list),
                'tcp_established': len([c for c in connections_list if c.get('status') == 'ESTABLISHED']),
                'tcp_listen': len([c for c in connections_list if c.get('status') == 'LISTEN']),
                'tcp_time_wait': len([c for c in connections_list if c.get('status') == 'TIME_WAIT']),
                'tcp_close_wait': len([c for c in connections_list if c.get('status') == 'CLOSE_WAIT']),
                'udp_connections': len([c for c in connections_list if c.get('type') == 'SOCK_DGRAM']),
                'tcp': tcp_total,  # keep aggregate for backward compatibility
                'total': len(connections_list),  # keep aggregate for backward compatibility
            }
            # Create bandwidth metrics from interface data
            bandwidth_dict = {}
            for iface_name, stats in interfaces_dict.items():
                bandwidth_dict[iface_name] = {
                    'bytes_sent': stats.get('bytes_sent', 0),
                    'bytes_recv': stats.get('bytes_recv', 0)
                }
            
            # Transform system info to system metrics
            sys_info = raw_metrics.get('system_info', {})
            system_metrics = {
                'cpu_percent': sys_info.get('cpu_percent', 0.0),
                'memory_percent': sys_info.get('memory_percent', 0.0),
                'disk_percent': sys_info.get('disk_usage', 0.0),
                'uptime': sys_info.get('uptime', 0.0),
                'load_avg_1min': sys_info.get('load_average', {}).get('1min', 0.0),
                'load_avg_5min': sys_info.get('load_average', {}).get('5min', 0.0),
                'load_avg_15min': sys_info.get('load_average', {}).get('15min', 0.0)
            }

            # Pass through connection rate hints as part of system metrics (floats allowed)
            if 'connection_rates' in raw_metrics:
                system_conn_rates = raw_metrics['connection_rates'] or {}
                system_metrics['new_connections_per_s'] = system_conn_rates.get('new_connections_per_s', 0.0)
                system_metrics['established_delta_per_s'] = system_conn_rates.get('established_delta_per_s', 0.0)
            
            # Use collector to get real latency metrics (keys match server schema)
            try:
                latency_metrics = self.metrics_collector.collect_latency_metrics()
            except Exception:
                latency_metrics = {
                    'google_dns_latency_ms': -1.0,
                    'cloudflare_dns_latency_ms': -1.0,
                    'local_gateway_latency_ms': -1.0,
                }
            
            # Create packet stats from interface data
            packet_stats = {}
            for iface_name, stats in interfaces_dict.items():
                packet_stats[iface_name] = {
                    'packets_sent': stats.get('packets_sent', 0),
                    'packets_recv': stats.get('packets_recv', 0),
                    'errors_in': stats.get('errors_in', 0),
                    'errors_out': stats.get('errors_out', 0),
                    'drops_in': stats.get('drops_in', 0),
                    'drops_out': stats.get('drops_out', 0)
                }
            
            # Build server-compatible metrics
            server_metrics = {
                'timestamp': timestamp,
                'hostname': self.agent_info['hostname'],
                'interfaces': interfaces_dict,
                'connections': connections_dict,
                'bandwidth': bandwidth_dict,
                'system_metrics': system_metrics,
                'latency_metrics': latency_metrics,
                'packet_stats': packet_stats
            }
            
            return server_metrics
            
        except Exception as e:
            logger.error(f"Error transforming metrics: {e}")
            # Return minimal valid structure
            return {
                'timestamp': str(time.time()),
                'hostname': self.agent_info['hostname'],
                'interfaces': {},
                'connections': {
                    'total_connections': 0,
                    'tcp_established': 0,
                    'tcp_listen': 0,
                    'tcp_time_wait': 0,
                    'tcp_close_wait': 0,
                    'udp_connections': 0,
                },
                'bandwidth': {},
                'system_metrics': {
                    'cpu_percent': 0.0,
                    'memory_percent': 0.0,
                    'disk_percent': 0.0,
                    'load_avg_1min': 0.0,
                    'load_avg_5min': 0.0,
                    'load_avg_15min': 0.0
                },
                'latency_metrics': {
                    'google_dns_latency_ms': -1.0,
                    'cloudflare_dns_latency_ms': -1.0,
                    'local_gateway_latency_ms': -1.0
                },
                'packet_stats': {}
            }
    
    async def run_heartbeat_cycle(self):
        """Send heartbeat to server"""
        try:
            await self.data_sender.send_heartbeat(self.agent_info)
        except Exception as e:
            logger.debug(f"Heartbeat failed: {e}")
    
    async def run(self):
        """Main agent loop"""
        logger.info("Starting AINet Network Agent...")
        
        # Initialize components
        await self.initialize()
        
        self.running = True
        collection_interval = self.config.get('agent', {}).get('collection_interval', 30)
        heartbeat_interval = 60  # Send heartbeat every minute
        
        last_heartbeat = 0
        
        logger.info(f"Agent started - collecting metrics every {collection_interval}s")
        
        try:
            while self.running:
                cycle_start = asyncio.get_event_loop().time()
                
                # Run collection cycle
                await self.run_collection_cycle()
                
                # Send heartbeat if needed
                current_time = asyncio.get_event_loop().time()
                if current_time - last_heartbeat >= heartbeat_interval:
                    await self.run_heartbeat_cycle()
                    last_heartbeat = current_time
                
                # Log queue status periodically
                queue_status = self.data_sender.get_queue_status()
                if queue_status['queue_size'] > 0:
                    logger.info(f"Queue status: {queue_status['queue_size']} metrics queued "
                              f"({queue_status['queue_usage_percent']:.1f}% capacity)")
                
                # Wait for next cycle
                cycle_duration = asyncio.get_event_loop().time() - cycle_start
                sleep_time = max(0, collection_interval - cycle_duration)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
        
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down agent...")
        
        if self.data_sender:
            # Try to flush any remaining metrics
            logger.info("Flushing remaining metrics...")
            try:
                flushed = await asyncio.wait_for(
                    self.data_sender.flush_queue(), 
                    timeout=30
                )
                if flushed > 0:
                    logger.info(f"Flushed {flushed} metrics before shutdown")
            except asyncio.TimeoutError:
                logger.warning("Timeout while flushing metrics")
            except Exception as e:
                logger.error(f"Error flushing metrics: {e}")
        
        logger.info("Agent shutdown complete")

def main():
    """Entry point"""
    # Determine config path
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        # Default to config in project root
        config_path = Path(__file__).parent.parent / 'config' / 'agent.yaml'
    
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        print("Usage: python agent/main.py [config_path]")
        sys.exit(1)
    
    # Create and run agent
    agent = NetworkAgent(config_path)
    
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
