# AINet - AI-Powered Network Anomaly Detection System

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-teal.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-orange.svg)](https://langchain.com/)

AINet is an intelligent network monitoring and anomaly detection system that uses AI to analyze network metrics and detect unusual patterns across your infrastructure. Built with LangGraph and LangChain, it provides real-time monitoring, automated analysis, and actionable insights.

## 🚀 Features

### Core Capabilities
- **🔍 Real-time Network Monitoring**: Continuous collection of network metrics from multiple Linux machines
- **🤖 AI-Powered Anomaly Detection**: Uses LangGraph workflows and LangChain for intelligent analysis
- **📊 Comprehensive Metrics Collection**: 
  - System metrics (CPU, memory, disk usage)
  - Network statistics (bandwidth, connections, packet loss)
  - Latency measurements (DNS, gateway)
  - Interface details and configuration
- **🚨 Smart Alerting**: Automated alert generation with severity classification
- **📈 Historical Analysis**: Time-series data storage and trend analysis
- **🎯 Risk Assessment**: AI-generated risk evaluations and recommendations

### Technical Features
- **Distributed Architecture**: Lightweight agents + centralized analysis server
- **Scalable Design**: Handles hundreds of monitored machines
- **RESTful API**: Full API access for integration and custom dashboards
- **Flexible Configuration**: YAML-based configuration for easy customization
- **Secure Communication**: API key authentication and SSL support
- **Production Ready**: Systemd service integration, log rotation, monitoring

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Linux Agent   │    │   Linux Agent   │    │   Linux Agent   │
│                 │    │                 │    │                 │
│ • Metrics       │    │ • Metrics       │    │ • Metrics       │
│ • Collection    │    │ • Collection    │    │ • Collection    │
│ • Heartbeat     │    │ • Heartbeat     │    │ • Heartbeat     │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │    Central Server       │
                    │                         │
                    │ • FastAPI Server        │
                    │ • SQLite/PostgreSQL     │
                    │ • AI Analysis Engine    │
                    │ • LangGraph Workflows   │
                    │ • Alert Management      │
                    │ • Web Dashboard         │
                    └─────────────────────────┘
```

## 📋 Prerequisites

- **Python**: 3.11 or higher
- **Operating System**: Linux (Ubuntu, CentOS, RHEL, Debian, Amazon Linux)
- **Memory**: Minimum 512MB RAM per agent, 2GB+ for server
- **Network**: HTTP/HTTPS connectivity between agents and server
- **Permissions**: Root access for agent installation

### Optional Requirements
- **OpenAI API Key**: For advanced AI analysis (or use local models)
- **Redis**: For enhanced caching (falls back to in-memory)
- **PostgreSQL**: For production database (defaults to SQLite)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/ainet.git
cd ainet
```

### 2. Set Up the Central Server

```bash
# Create conda environment
conda create -n ainet python=3.11 -y
conda activate ainet

# Install dependencies
pip install -r requirements.txt

# Configure the server (optional - uses defaults)
cp config/config.yaml config/config.local.yaml
# Edit config/config.local.yaml as needed

# Start the server
python server/main.py
```

The server will be available at `http://localhost:8000`

### 3. Install Agents on Linux Machines

On each machine you want to monitor:

```bash
# Download and run the installation script
curl -sSL https://raw.githubusercontent.com/your-org/ainet/main/scripts/install_agent.sh | sudo bash

# Configure the agent
sudo nano /etc/ainet/agent.yaml
# Update server_url and api_key

# Start the agent
sudo ainet-start

# Check status
sudo ainet-status
```

### 4. Access the Dashboard

Open your browser and go to:
- **Dashboard**: `http://localhost:8000/dashboard`
- **API Documentation**: `http://localhost:8000/docs`
- **System Status**: `http://localhost:8000/status`

## 📁 Project Structure

```
AINet/
├── agent/                      # Network monitoring agent
│   ├── network_monitor/        # Metrics collection modules
│   ├── data_collector/         # Data transmission modules
│   └── main.py                 # Agent main application
├── server/                     # Central analysis server
│   ├── api/                    # FastAPI endpoints and schemas
│   ├── database/               # Database models and connections
│   ├── ai_engine/              # AI analysis and LangGraph workflows
│   └── main.py                 # Server main application
├── dashboard/                  # Web dashboard (future implementation)
├── config/                     # Configuration files
│   ├── config.yaml             # Server configuration
│   └── agent.yaml              # Agent configuration template
├── scripts/                    # Installation and management scripts
│   ├── install_agent.sh        # Agent installation script
│   └── uninstall_agent.sh      # Agent removal script
├── tests/                      # Test suites
└── docs/                       # Documentation
```

## ⚙️ Configuration

### Server Configuration

Edit `config/config.yaml`:

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  debug: false

database:
  url: "sqlite:///./ainet.db"  # or postgresql://...

ai:
  model_provider: "openai"
  model_name: "gpt-4"
  api_key: "your-openai-api-key"
  
monitoring:
  analysis_interval_minutes: 60
  anomaly_sensitivity: 0.8
```

### Agent Configuration

Edit `/etc/ainet/agent.yaml` on each monitored machine:

```yaml
agent:
  collection_interval: 30

sender:
  server_url: "http://your-server:8000"
  api_key: "your-api-key"

monitoring:
  interfaces: "all"
  ping_targets:
    - name: "google_dns"
      host: "8.8.8.8"
```

## 🤖 AI Analysis Workflow

AINet uses LangGraph to create sophisticated analysis workflows:

1. **Data Preprocessing**: Cleans and normalizes metrics data
2. **Statistical Analysis**: Calculates trends, baselines, and variations
3. **Anomaly Detection**: Identifies outliers using statistical methods
4. **AI Analysis**: LangChain analyzes patterns and generates insights
5. **Risk Assessment**: Evaluates potential impacts and urgency
6. **Recommendations**: Provides actionable suggestions for optimization

### Example Analysis Output

```json
{
  "findings": [
    {
      "metric_name": "cpu_percent",
      "anomaly_type": "spike",
      "severity": "high",
      "confidence": 0.92,
      "description": "CPU usage spiked to 95% at 14:30, significantly above the 45% baseline",
      "baseline_value": 45.2,
      "anomalous_value": 95.1,
      "deviation_percentage": 110.4
    }
  ],
  "recommendations": [
    {
      "category": "performance",
      "priority": "high",
      "title": "Investigate CPU Usage Spike",
      "description": "Check for runaway processes or resource-intensive applications",
      "expected_impact": "Prevent system slowdown and potential service disruption"
    }
  ],
  "risk_assessment": {
    "overall_risk_level": "medium",
    "urgency_score": 0.7
  }
}
```

## 🛠️ API Reference

### Key Endpoints

#### Agent Management
- `GET /api/v1/agents` - List all registered agents
- `GET /api/v1/agents/{id}/metrics` - Get metrics for specific agent
- `POST /api/v1/heartbeat` - Agent heartbeat endpoint

#### Metrics Collection
- `POST /api/v1/metrics` - Submit single metric data point
- `POST /api/v1/metrics/batch` - Submit batch of metrics

#### Analysis & Alerts
- `GET /api/v1/alerts` - Get alerts with filtering
- `POST /api/v1/analysis/agent/{id}` - Trigger AI analysis
- `GET /api/v1/analysis/history/{id}` - Get analysis history

#### System Health
- `GET /api/v1/health` - System health check
- `GET /status` - Detailed system status

### Authentication

All API requests require an API key in the header:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/api/v1/agents
```

## 🔧 Management

### Agent Management

```bash
# Start agent
sudo ainet-start

# Stop agent
sudo ainet-stop

# Restart agent
sudo ainet-restart

# Check status
sudo ainet-status

# View logs
sudo journalctl -u ainet-agent -f
```

### Server Management

```bash
# Start server (development)
python server/main.py

# Start with custom config
python server/main.py /path/to/config.yaml

# Production deployment (use systemd, docker, or process manager)
gunicorn -w 4 -k uvicorn.workers.UvicornWorker server.main:app
```

### Database Management

```python
# Access database directly
from server.database.connection import get_db_manager

db_manager = get_db_manager()
with db_manager.get_session() as db:
    agents = db.query(Agent).all()
```

## 🏭 Production Deployment

### Server Deployment

1. **Use Production Database**:
   ```yaml
   database:
     url: "postgresql://user:pass@localhost/ainet"
   ```

2. **Enable Security**:
   ```yaml
   security:
     api_keys_required: true
     ssl_enabled: true
   ```

3. **Deploy with Docker** (example):
   ```dockerfile
   FROM python:3.11-slim
   COPY . /app
   WORKDIR /app
   RUN pip install -r requirements.txt
   CMD ["python", "server/main.py"]
   ```

### Monitoring & Maintenance

- **Log Rotation**: Automatic via logrotate configuration
- **Health Checks**: Built-in endpoints for load balancer integration
- **Metrics Export**: Prometheus-compatible metrics available
- **Backup**: Regular database backups recommended

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=server --cov=agent tests/

# Run specific test category
pytest tests/test_agent.py -v
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run the test suite: `pytest`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Development Setup

```bash
# Clone and setup
git clone https://github.com/your-org/ainet.git
cd ainet

# Setup development environment
conda create -n ainet-dev python=3.11 -y
conda activate ainet-dev
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run in development mode
python server/main.py --config config/config.yaml --debug
```

## 📚 Documentation

- **API Documentation**: Available at `/docs` when server is running
- **Architecture Guide**: `docs/architecture.md`
- **Deployment Guide**: `docs/deployment.md`
- **Troubleshooting**: `docs/troubleshooting.md`

## 🔒 Security Considerations

- **API Keys**: Use strong, unique keys for each deployment
- **Network Security**: Deploy server behind firewall/VPN when possible
- **SSL/TLS**: Enable HTTPS in production environments
- **Least Privilege**: Agents run with minimal required permissions
- **Data Privacy**: Metrics are aggregated, no personal data collected

## 📈 Performance & Scalability

- **Agent Performance**: Minimal CPU/memory footprint (~10MB RAM)
- **Server Capacity**: Handles 100+ agents on modest hardware
- **Database**: SQLite for development, PostgreSQL recommended for production
- **Caching**: Redis integration for improved performance
- **Horizontal Scaling**: Stateless design supports load balancing

## 🐛 Troubleshooting

### Common Issues

1. **Agent can't connect to server**:
   ```bash
   # Check network connectivity
   curl -I http://your-server:8000/api/v1/health
   
   # Verify configuration
   sudo cat /etc/ainet/agent.yaml
   ```

2. **High memory usage**:
   - Adjust `collection_interval` in agent config
   - Configure log rotation
   - Monitor queue sizes

3. **AI analysis failures**:
   - Check OpenAI API key and quota
   - Verify internet connectivity from server
   - Review server logs for detailed errors

### Debug Mode

Enable debug logging:

```yaml
logging:
  level: "DEBUG"
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain Team** for the excellent AI framework
- **FastAPI** for the high-performance web framework
- **SQLAlchemy** for robust database ORM
- **Pydantic** for data validation
- Open source community for various dependencies

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-org/ainet/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/ainet/discussions)
- **Documentation**: [Wiki](https://github.com/your-org/ainet/wiki)
- **Email**: support@yourcompany.com

---

**Made with ❤️ by the AINet Team**

*AINet - Intelligent Network Monitoring for the Modern Infrastructure*
