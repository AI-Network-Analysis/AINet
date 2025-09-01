"""
Database Connection
Handles database connection, session management, and initialization.
"""

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import logging
from typing import Generator
import os
from .models import Base

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database connections and sessions"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = None
        self.SessionLocal = None
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize the database engine"""
        connect_args = {}
        
        # SQLite specific configuration
        if self.database_url.startswith('sqlite'):
            connect_args = {
                "check_same_thread": False,  # Allow multiple threads
                "timeout": 20  # Connection timeout
            }
            
            # For in-memory SQLite, use StaticPool
            if ':memory:' in self.database_url:
                self.engine = create_engine(
                    self.database_url,
                    connect_args=connect_args,
                    poolclass=StaticPool,
                    echo=False  # Set to True for SQL debugging
                )
            else:
                # Ensure directory exists for file-based SQLite
                db_path = self.database_url.replace('sqlite:///', '')
                if db_path != ':memory:' and '/' in db_path:
                    db_dir = os.path.dirname(db_path)
                    if db_dir:  # Only create if there's actually a directory
                        os.makedirs(db_dir, exist_ok=True)
                
                self.engine = create_engine(
                    self.database_url,
                    connect_args=connect_args,
                    echo=False  # Set to True for SQL debugging
                )
        else:
            # PostgreSQL, MySQL, etc.
            self.engine = create_engine(
                self.database_url,
                pool_pre_ping=True,  # Verify connections before use
                pool_recycle=3600,   # Recycle connections after 1 hour
                echo=False  # Set to True for SQL debugging
            )
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,  # keep attributes accessible after commit
            bind=self.engine
        )
        
        logger.info(f"Database engine initialized: {self.database_url}")
    
    def create_tables(self):
        """Create all database tables"""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create database tables: {e}")
            raise
    
    def drop_tables(self):
        """Drop all database tables (use with caution!)"""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.warning("All database tables dropped")
        except Exception as e:
            logger.error(f"Failed to drop database tables: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get a new database session"""
        return self.SessionLocal()
    
    def get_session_generator(self) -> Generator[Session, None, None]:
        """Generator for database sessions (for dependency injection)"""
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    def health_check(self) -> bool:
        """Check if database connection is healthy"""
        try:
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def close(self):
        """Close database engine"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database engine closed")

# Global database manager instance
db_manager = None

def initialize_database(database_url: str) -> DatabaseManager:
    """Initialize the global database manager"""
    global db_manager
    db_manager = DatabaseManager(database_url)
    db_manager.create_tables()
    return db_manager

def get_db_manager() -> DatabaseManager:
    """Get the global database manager"""
    if db_manager is None:
        raise RuntimeError("Database not initialized. Call initialize_database() first.")
    return db_manager

def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for getting database sessions"""
    manager = get_db_manager()
    db = manager.get_session()
    try:
        yield db
    finally:
        db.close()

# Event listeners for SQLite optimization
# Note: Event listeners will be set up per engine instance in the DatabaseManager class
