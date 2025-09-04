"""
Monitoring Configuration Management
Allows users to configure what aspects of the system to monitor.
"""

import yaml
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class MonitoringMode(Enum):
    """Monitoring mode options"""
    NETWORK_FOCUS = "network_focus"  # Primary focus on network anomalies
    SYSTEM_FOCUS = "system_focus"    # Primary focus on CPU/memory/disk
    BALANCED = "balanced"            # Monitor both equally
    MINIMAL = "minimal"              # Basic monitoring only

@dataclass
class NetworkMonitoringConfig:
    """Network monitoring configuration"""
    enabled: bool = True
    anomaly_detection: bool = True
    bandwidth_monitoring: bool = True
    connection_monitoring: bool = True
    latency_monitoring: bool = True
    packet_loss_monitoring: bool = True
    
    # Anomaly detection settings
    sensitivity: float = 0.8  # 0.0 = low sensitivity, 1.0 = high sensitivity
    baseline_samples: int = 50
    history_size: int = 100
    
    # Alert thresholds
    latency_warning_threshold: float = 100.0  # ms
    latency_critical_threshold: float = 500.0  # ms
    packet_loss_warning_threshold: float = 1.0  # %
    packet_loss_critical_threshold: float = 5.0  # %
    bandwidth_spike_threshold: float = 200.0  # % increase from baseline
    # Security thresholds
    conn_rate_warning_threshold: float = 50.0  # new connections per second
    conn_rate_critical_threshold: float = 200.0  # new connections per second
    recv_rate_spike_factor: float = 3.0  # X times baseline
    detect_arp_gateway_change: bool = True

@dataclass
class SystemMonitoringConfig:
    """System resource monitoring configuration"""
    enabled: bool = True
    cpu_monitoring: bool = True
    memory_monitoring: bool = True
    disk_monitoring: bool = True
    process_monitoring: bool = True
    
    # Alert thresholds
    cpu_warning_threshold: float = 80.0  # %
    cpu_critical_threshold: float = 95.0  # %
    memory_warning_threshold: float = 85.0  # %
    memory_critical_threshold: float = 95.0  # %
    disk_warning_threshold: float = 85.0  # %
    disk_critical_threshold: float = 95.0  # %
    
    # Process monitoring
    monitor_top_processes: bool = True
    process_cpu_threshold: float = 50.0  # %
    process_memory_threshold: float = 20.0  # %

@dataclass
class MonitoringConfig:
    """Complete monitoring configuration"""
    mode: MonitoringMode = MonitoringMode.BALANCED
    collection_interval: int = 30  # seconds
    retention_days: int = 30
    
    network: NetworkMonitoringConfig = None
    system: SystemMonitoringConfig = None
    
    # AI analysis settings
    ai_analysis_enabled: bool = True
    ai_analysis_interval: int = 300  # seconds (5 minutes)
    
    def __post_init__(self):
        if self.network is None:
            self.network = NetworkMonitoringConfig()
        if self.system is None:
            self.system = SystemMonitoringConfig()
        
        # Adjust settings based on monitoring mode
        self._apply_mode_settings()
    
    def _apply_mode_settings(self):
        """Apply settings based on monitoring mode"""
        if self.mode == MonitoringMode.NETWORK_FOCUS:
            # Prioritize network monitoring
            self.network.enabled = True
            self.network.anomaly_detection = True
            self.network.sensitivity = 0.9  # High sensitivity
            self.network.history_size = 200  # More history for better baselines
            
            self.system.enabled = True
            self.system.cpu_monitoring = True
            self.system.memory_monitoring = True
            self.system.disk_monitoring = False  # Disable disk monitoring
            self.system.process_monitoring = False  # Disable process monitoring
            
            self.collection_interval = 15  # More frequent collection
            
        elif self.mode == MonitoringMode.SYSTEM_FOCUS:
            # Prioritize system resource monitoring
            self.system.enabled = True
            self.system.cpu_monitoring = True
            self.system.memory_monitoring = True
            self.system.disk_monitoring = True
            self.system.process_monitoring = True
            
            self.network.enabled = True
            self.network.anomaly_detection = False  # Disable advanced network anomaly detection
            self.network.bandwidth_monitoring = False
            self.network.connection_monitoring = False
            self.network.latency_monitoring = True  # Keep basic connectivity check
            self.network.packet_loss_monitoring = False
            
        elif self.mode == MonitoringMode.MINIMAL:
            # Minimal monitoring - basic health checks only
            self.network.enabled = True
            self.network.anomaly_detection = False
            self.network.bandwidth_monitoring = False
            self.network.connection_monitoring = False
            self.network.latency_monitoring = True
            self.network.packet_loss_monitoring = False
            
            self.system.enabled = True
            self.system.cpu_monitoring = True
            self.system.memory_monitoring = True
            self.system.disk_monitoring = True
            self.system.process_monitoring = False
            
            self.collection_interval = 60  # Less frequent collection
            self.ai_analysis_enabled = False
            
        # BALANCED mode keeps default settings

class ConfigurationManager:
    """Manages monitoring configuration"""
    
    def __init__(self, config_path: str = "config/monitoring.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> MonitoringConfig:
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                return self._dict_to_config(config_data)
        except FileNotFoundError:
            # Create default config
            default_config = MonitoringConfig()
            self.save_config(default_config)
            return default_config
        except Exception as e:
            print(f"Error loading config: {e}")
            return MonitoringConfig()
    
    def _dict_to_config(self, config_data: Dict[str, Any]) -> MonitoringConfig:
        """Convert dictionary to MonitoringConfig"""
        # Handle mode
        mode_str = config_data.get('mode', 'balanced')
        mode = MonitoringMode(mode_str) if mode_str in [m.value for m in MonitoringMode] else MonitoringMode.BALANCED
        
        # Create network config
        network_data = config_data.get('network', {})
        network_config = NetworkMonitoringConfig(**network_data)
        
        # Create system config
        system_data = config_data.get('system', {})
        system_config = SystemMonitoringConfig(**system_data)
        
        # Create main config
        main_config = MonitoringConfig(
            mode=mode,
            collection_interval=config_data.get('collection_interval', 30),
            retention_days=config_data.get('retention_days', 30),
            network=network_config,
            system=system_config,
            ai_analysis_enabled=config_data.get('ai_analysis_enabled', True),
            ai_analysis_interval=config_data.get('ai_analysis_interval', 300)
        )
        
        return main_config
    
    def save_config(self, config: MonitoringConfig):
        """Save configuration to file"""
        try:
            config_dict = asdict(config)
            # Convert enum to string
            config_dict['mode'] = config.mode.value
            
            import os
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get_config(self) -> MonitoringConfig:
        """Get current configuration"""
        return self.config
    
    def update_config(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self.save_config(self.config)
    
    def set_monitoring_mode(self, mode: MonitoringMode):
        """Set monitoring mode and apply related settings"""
        self.config.mode = mode
        self.config._apply_mode_settings()
        self.save_config(self.config)
    
    def get_monitoring_presets(self) -> Dict[str, Dict[str, Any]]:
        """Get predefined monitoring configuration presets"""
        return {
            "network_security": {
                "name": "Network Security Focus",
                "description": "Optimized for detecting network intrusions, DDoS attacks, and suspicious traffic patterns",
                "mode": MonitoringMode.NETWORK_FOCUS,
                "settings": {
                    "collection_interval": 10,
                    "network": {
                        "sensitivity": 0.95,
                        "anomaly_detection": True,
                        "bandwidth_monitoring": True,
                        "connection_monitoring": True,
                        "latency_monitoring": True,
                        "packet_loss_monitoring": True,
                        "baseline_samples": 100,
                        "history_size": 500
                    }
                }
            },
            "server_performance": {
                "name": "Server Performance Monitoring",
                "description": "Optimized for monitoring server resource usage and performance bottlenecks",
                "mode": MonitoringMode.SYSTEM_FOCUS,
                "settings": {
                    "collection_interval": 15,
                    "system": {
                        "cpu_warning_threshold": 70.0,
                        "memory_warning_threshold": 80.0,
                        "process_monitoring": True,
                        "monitor_top_processes": True
                    }
                }
            },
            "balanced_monitoring": {
                "name": "Balanced Monitoring",
                "description": "Balanced monitoring of both network and system resources",
                "mode": MonitoringMode.BALANCED,
                "settings": {
                    "collection_interval": 30
                }
            },
            "lightweight": {
                "name": "Lightweight Monitoring",
                "description": "Minimal resource usage with basic health monitoring",
                "mode": MonitoringMode.MINIMAL,
                "settings": {
                    "collection_interval": 60,
                    "ai_analysis_enabled": False,
                    "network": {
                        "anomaly_detection": False,
                        "bandwidth_monitoring": False,
                        "connection_monitoring": False
                    },
                    "system": {
                        "process_monitoring": False
                    }
                }
            }
        }
    
    def apply_preset(self, preset_name: str):
        """Apply a predefined configuration preset"""
        presets = self.get_monitoring_presets()
        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}")
        
        preset = presets[preset_name]
        
        # Set mode
        self.config.mode = preset["mode"]
        
        # Apply settings
        settings = preset.get("settings", {})
        for key, value in settings.items():
            if key == "network" and isinstance(value, dict):
                for net_key, net_value in value.items():
                    if hasattr(self.config.network, net_key):
                        setattr(self.config.network, net_key, net_value)
            elif key == "system" and isinstance(value, dict):
                for sys_key, sys_value in value.items():
                    if hasattr(self.config.system, sys_key):
                        setattr(self.config.system, sys_key, sys_value)
            elif hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Apply mode-specific settings
        self.config._apply_mode_settings()
        
        # Save configuration
        self.save_config(self.config)

def get_default_config() -> MonitoringConfig:
    """Get default monitoring configuration"""
    return MonitoringConfig()

def create_network_focused_config() -> MonitoringConfig:
    """Create a network-focused monitoring configuration"""
    config = MonitoringConfig(mode=MonitoringMode.NETWORK_FOCUS)
    return config

def create_system_focused_config() -> MonitoringConfig:
    """Create a system-focused monitoring configuration"""
    config = MonitoringConfig(mode=MonitoringMode.SYSTEM_FOCUS)
    return config
