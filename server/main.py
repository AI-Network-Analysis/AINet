#!/usr/bin/env python3
"""
AINet Central Server
FastAPI application for receiving metrics from agents and providing dashboard.
"""

import os
import sys
import yaml
import logging
import uvicorn
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from server.ai_engine import AIEngineService, AIEngineScheduler
import asyncio

load_dotenv()
# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.database.connection import initialize_database
from server.api.endpoints import api_router, dashboard_router
from server.api.finetune import finetune_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file"""
    if config_path is None:
        config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        # Return default configuration
        return {
            'server': {
                'host': '0.0.0.0',
                'port': 8000,
                'debug': True,
                'cors_origins': ['*']
            },
            'database': {
                'url': 'sqlite:///./ainet.db'
            }
        }

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting AINet Central Server...")
    
    # Initialize database
    db_url = app.state.config['database']['url']
    try:
        initialize_database(db_url)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
    
    # Initialize AI engine
    try:
        app.state.ai_service = AIEngineService(app.state.config)
        app.state.ai_scheduler = AIEngineScheduler(app.state.ai_service, app.state.config)
        app.state.ai_task = asyncio.create_task(app.state.ai_scheduler.start_periodic_analysis())
        logger.info("AI engine initialized and periodic analysis started")
    except Exception as e:
        logger.error(f"Failed to initialize AI engine: {e}")
        raise
    
    yield
    
    # Shutdown
    if hasattr(app.state, 'ai_scheduler'):
        app.state.ai_scheduler.stop()
        await app.state.ai_task
        logger.info("AI engine shut down")
    
    logger.info("Shutting down AINet Central Server...")

def create_app(config: dict) -> FastAPI:
    """Create FastAPI application"""
    app = FastAPI(
        title="AINet Central Server",
        description="Network monitoring and anomaly detection system",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )
    
    # Store config in app state
    app.state.config = config
    
    # Configure CORS
    cors_origins = config.get('server', {}).get('cors_origins', ['*'])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routers
    app.include_router(api_router, tags=["API"])
    app.include_router(dashboard_router, tags=["Dashboard"])
    app.include_router(finetune_router)
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "name": "AINet Central Server",
            "version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "status": "running",
            "docs": "/docs",
            "dashboard": "/dashboard",
            "finetune_dashboard": "/finetune"
        }
    
    # System status endpoint
    @app.get("/status")
    async def system_status():
        """System status endpoint"""
        from server.database.connection import get_db_manager
        
        try:
            db_manager = get_db_manager()
            db_healthy = db_manager.health_check()
        except:
            db_healthy = False
        
        ai_healthy = hasattr(app.state, 'ai_service') and hasattr(app.state, 'ai_task') and not app.state.ai_task.done()
        
        return {
            "status": "healthy" if db_healthy and ai_healthy else "unhealthy",
            "database": "connected" if db_healthy else "disconnected",
            "ai_engine": "running" if ai_healthy else "not running",
            "timestamp": datetime.utcnow().isoformat(),
            "uptime": "N/A",  # TODO: Calculate uptime
            "version": "1.0.0"
        }
    
    # Mount static files for dashboard
    static_dir = Path(__file__).parent.parent / "dashboard" / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    # Dashboard route
    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard():
        """Enhanced AI-powered dashboard"""
        templates_dir = Path(__file__).parent.parent / "dashboard" / "templates"
        
        # If template exists, serve it
        if (templates_dir / "dashboard.html").exists():
            try:
                with open(templates_dir / "dashboard.html", "r") as f:
                    html_content = f.read()
                return HTMLResponse(content=html_content)
            except Exception as e:
                logger.error(f"Error loading dashboard template: {e}")
        
        # Fallback to simple dashboard
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AINet Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }
                .card { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 5px; padding: 15px; margin: 10px 0; }
                .status { display: inline-block; padding: 5px 10px; border-radius: 3px; color: white; }
                .status.online { background: #28a745; }
                .status.offline { background: #dc3545; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AINet Network Monitoring Dashboard</h1>
                <p>AI-powered network anomaly detection system</p>
            </div>
            
            <div class="card">
                <h3>System Status</h3>
                <p><strong>Server:</strong> <span class="status online">Online</span></p>
                <p><strong>Database:</strong> <span class="status online">Connected</span></p>
                <p><strong>AI Engine:</strong> <span class="status online">Online</span></p>
            </div>
            
            <div class="card">
                <h3>Quick Links</h3>
                <ul>
                    <li><a href="/docs">API Documentation</a></li>
                    <li><a href="/api/v1/agents">View Agents</a></li>
                    <li><a href="/api/v1/alerts">View Alerts</a></li>
                    <li><a href="/status">System Status</a></li>
                </ul>
            </div>
            
            <div class="card">
                <h3>Features</h3>
                <ul>
                    <li>✅ Agent Registration & Heartbeat</li>
                    <li>✅ Network Metrics Collection</li>
                    <li>✅ Data Storage & Retrieval</li>
                    <li>✅ AI Anomaly Detection</li>
                    <li>✅ Real-time Dashboard</li>
                    <li>✅ Interactive AI Analysis</li>
                </ul>
            </div>
            
            <div class="card">
                <h3>Getting Started</h3>
                <p>1. Deploy agents on your Linux machines</p>
                <p>2. Configure agents to connect to this server</p>
                <p>3. Monitor network metrics and anomalies</p>
                <p>4. Use the dashboard to analyze data with AI</p>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    
    return app

# Global app instance for uvicorn reload mode
config = load_config(os.environ.get('AINET_CONFIG'))
app = create_app(config)

def main():
    """Main entry point"""
    # Load configuration
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    config = load_config(config_path)
    
    # Create application
    app = create_app(config)
    
    # Get server configuration
    server_config = config.get('server', {})
    host = server_config.get('host', '0.0.0.0')
    port = server_config.get('port', 8000)
    debug = server_config.get('debug', False)
    
    logger.info(f"Starting server on {host}:{port} (debug={debug})")
    
    # Run server
    if debug:
        # Use import string for reload mode
        os.environ['AINET_CONFIG'] = str(config_path) if config_path else ''
        uvicorn.run(
            "server.main:app",
            host=host,
            port=port,
            reload=True,
            log_level="debug"
        )
    else:
        # Use app instance for production
        app = create_app(config)
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info"
        )

if __name__ == "__main__":
    main()
