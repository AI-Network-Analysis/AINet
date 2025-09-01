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
            
            # Collect metrics
            metrics = self.metrics_collector.collect_all_metrics()
            metrics_dict = self.metrics_collector.to_dict(metrics)
            
            # Send metrics
            success = await self.data_sender.send_metrics(metrics_dict)
            
            if success:
                logger.debug("Metrics collection cycle completed successfully")
            else:
                logger.warning("Failed to send metrics - queued for later retry")
                
        except Exception as e:
            logger.error(f"Error in collection cycle: {e}")
    
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
