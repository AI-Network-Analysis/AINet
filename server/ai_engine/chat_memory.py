"""
Custom SQLAlchemy Checkpointer for LangGraph
"""

import json
import pickle
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

from sqlalchemy.orm import Session
from server.database.connection import get_db_manager
from server.database.models import ChatCheckpoint

class SQLAlchemyCheckpointer:
    """
    A simple checkpointer that stores conversation state in a SQLAlchemy database.
    """

    def __init__(self):
        self.db_manager = get_db_manager()

    @contextmanager
    def _get_session(self) -> Generator[Session, None, None]:
        """Get a new database session."""
        db = self.db_manager.get_session()
        try:
            yield db
        finally:
            db.close()

    def get_checkpoint(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get a checkpoint from the database."""
        with self._get_session() as db:
            checkpoint_row = db.query(ChatCheckpoint).filter(ChatCheckpoint.thread_id == thread_id).first()
            if checkpoint_row:
                try:
                    return json.loads(checkpoint_row.checkpoint) if isinstance(checkpoint_row.checkpoint, str) else checkpoint_row.checkpoint
                except (json.JSONDecodeError, TypeError):
                    # Fallback for pickled data
                    try:
                        return pickle.loads(checkpoint_row.checkpoint)
                    except:
                        return None
        return None

    def put_checkpoint(self, thread_id: str, checkpoint: Dict[str, Any]) -> None:
        """Save a checkpoint to the database."""
        with self._get_session() as db:
            checkpoint_row = db.query(ChatCheckpoint).filter(ChatCheckpoint.thread_id == thread_id).first()
            checkpoint_data = json.dumps(checkpoint, default=str)
            
            if checkpoint_row:
                checkpoint_row.checkpoint = checkpoint_data
            else:
                db.add(ChatCheckpoint(thread_id=thread_id, checkpoint=checkpoint_data))
            db.commit()

    def clear_checkpoint(self, thread_id: str) -> None:
        """Clear a checkpoint from the database."""
        with self._get_session() as db:
            checkpoint_row = db.query(ChatCheckpoint).filter(ChatCheckpoint.thread_id == thread_id).first()
            if checkpoint_row:
                db.delete(checkpoint_row)
                db.commit()
