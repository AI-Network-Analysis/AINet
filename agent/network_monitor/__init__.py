"""
Network Monitor Module
Contains network metrics collection functionality.
"""

from .metrics_collector import EnhancedMetricsCollector

# For backward compatibility
MetricsCollector = EnhancedMetricsCollector

__all__ = ['EnhancedMetricsCollector', 'MetricsCollector']
