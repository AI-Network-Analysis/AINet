"""
Data Sender
Sends collected network metrics to the central server.
"""

import aiohttp
import asyncio
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import asdict
import time
from datetime import datetime
import ssl
import os

logger = logging.getLogger(__name__)

class DataSender:
    """Sends network metrics to central server"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.server_url = config.get('server_url', 'http://localhost:8000')
        self.api_key = config.get('api_key')
        self.timeout = config.get('timeout', 30)
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 5)
        self.batch_size = config.get('batch_size', 100)
        
        # SSL configuration
        self.ssl_context = self._create_ssl_context()
        
        # Queue for storing metrics when server is unreachable
        self.metrics_queue = []
        self.max_queue_size = config.get('max_queue_size', 1000)
        
    def _create_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Create SSL context for HTTPS connections"""
        if self.server_url.startswith('https://'):
            context = ssl.create_default_context()
            
            # If using self-signed certificates in development
            if self.config.get('ssl_verify', True) is False:
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
                logger.warning("SSL verification disabled - not recommended for production!")
                
            return context
        return None
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests"""
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': f'AINet-Agent/1.0'
        }
        
        if self.api_key:
            headers[self.config.get('api_key_header', 'X-API-Key')] = self.api_key
            
        return headers
    
    async def send_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Send metrics to the central server"""
        endpoint = f"{self.server_url}/api/v1/metrics"
        
        # Add to queue first
        self.metrics_queue.append(metrics)
        
        # Process queue
        return await self._process_queue()
    
    async def _process_queue(self) -> bool:
        """Process queued metrics"""
        if not self.metrics_queue:
            return True
            
        # Limit queue size
        if len(self.metrics_queue) > self.max_queue_size:
            # Remove oldest entries
            excess = len(self.metrics_queue) - self.max_queue_size
            self.metrics_queue = self.metrics_queue[excess:]
            logger.warning(f"Queue overflow: removed {excess} oldest metrics")
        
        success = False
        batch = []
        
        # Create batch
        while self.metrics_queue and len(batch) < self.batch_size:
            batch.append(self.metrics_queue.pop(0))
        
        if batch:
            success = await self._send_batch(batch)
            
            # If failed, put batch back at the front of queue
            if not success:
                self.metrics_queue = batch + self.metrics_queue
        
        return success
    
    async def _send_batch(self, batch: list) -> bool:
        """Send a batch of metrics to the server"""
        endpoint = f"{self.server_url}/api/v1/metrics/batch"
        headers = self._get_headers()
        
        payload = {
            'metrics': batch,
            'batch_size': len(batch),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        for attempt in range(self.max_retries):
            try:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                
                async with aiohttp.ClientSession(
                    timeout=timeout,
                    connector=aiohttp.TCPConnector(ssl=self.ssl_context)
                ) as session:
                    
                    async with session.post(
                        endpoint,
                        json=payload,
                        headers=headers
                    ) as response:
                        
                        if response.status == 200:
                            result = await response.json()
                            logger.info(f"Successfully sent batch of {len(batch)} metrics")
                            return True
                        elif response.status == 401:
                            logger.error("Authentication failed - check API key")
                            return False
                        elif response.status == 429:
                            # Rate limited
                            retry_after = int(response.headers.get('Retry-After', self.retry_delay))
                            logger.warning(f"Rate limited, waiting {retry_after}s before retry")
                            await asyncio.sleep(retry_after)
                        else:
                            error_text = await response.text()
                            logger.error(f"Server error {response.status}: {error_text}")
                            
            except aiohttp.ClientError as e:
                logger.error(f"Network error (attempt {attempt + 1}/{self.max_retries}): {e}")
            except asyncio.TimeoutError:
                logger.error(f"Timeout error (attempt {attempt + 1}/{self.max_retries})")
            except Exception as e:
                logger.error(f"Unexpected error (attempt {attempt + 1}/{self.max_retries}): {e}")
            
            if attempt < self.max_retries - 1:
                wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                logger.info(f"Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
        
        logger.error(f"Failed to send batch after {self.max_retries} attempts")
        return False
    
    async def send_heartbeat(self, agent_info: Dict[str, Any]) -> bool:
        """Send heartbeat to indicate agent is alive"""
        endpoint = f"{self.server_url}/api/v1/heartbeat"
        headers = self._get_headers()
        
        payload = {
            'hostname': agent_info.get('hostname'),
            'timestamp': datetime.utcnow().isoformat(),
            'agent_version': agent_info.get('version', '1.0.0'),
            'status': 'active',
            'queue_size': len(self.metrics_queue)
        }
        
        try:
            timeout = aiohttp.ClientTimeout(total=10)  # Shorter timeout for heartbeat
            
            async with aiohttp.ClientSession(
                timeout=timeout,
                connector=aiohttp.TCPConnector(ssl=self.ssl_context)
            ) as session:
                
                async with session.post(
                    endpoint,
                    json=payload,
                    headers=headers
                ) as response:
                    
                    if response.status == 200:
                        logger.debug("Heartbeat sent successfully")
                        return True
                    else:
                        logger.warning(f"Heartbeat failed with status {response.status}")
                        
        except Exception as e:
            logger.debug(f"Heartbeat failed: {e}")
            
        return False
    
    async def test_connection(self) -> bool:
        """Test connection to the central server"""
        endpoint = f"{self.server_url}/api/v1/health"
        headers = self._get_headers()
        
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            
            async with aiohttp.ClientSession(
                timeout=timeout,
                connector=aiohttp.TCPConnector(ssl=self.ssl_context)
            ) as session:
                
                async with session.get(endpoint, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"Connection test successful: {result}")
                        return True
                    else:
                        logger.error(f"Connection test failed with status {response.status}")
                        
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            
        return False
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get status of the metrics queue"""
        return {
            'queue_size': len(self.metrics_queue),
            'max_queue_size': self.max_queue_size,
            'queue_usage_percent': (len(self.metrics_queue) / self.max_queue_size) * 100
        }
    
    async def flush_queue(self) -> int:
        """Flush all queued metrics to server"""
        initial_size = len(self.metrics_queue)
        
        while self.metrics_queue:
            success = await self._process_queue()
            if not success:
                break
        
        sent = initial_size - len(self.metrics_queue)
        logger.info(f"Flushed {sent} metrics from queue")
        return sent
