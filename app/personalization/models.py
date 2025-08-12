import os
import datetime
from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, JSON, Text, Float
)
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    user_id = Column(String, nullable=False)
    session_id = Column(String, nullable=True)
    query = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    rating = Column(Integer, nullable=False)  # 1–5
    comments = Column(Text, nullable=True)
    aspects = Column(JSON, nullable=True)  # e.g., {"retrieval":"low","factuality":"ok"}
    context_embedding = Column(JSON, nullable=True)  # list[float]
    modules_used = Column(JSON, nullable=False)      # {"chunker":"adaptive", ...}
    latency_ms = Column(Float, nullable=True)


class ModuleEvent(Base):
    __tablename__ = "module_events"
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    user_id = Column(String, nullable=False)
    session_id = Column(String, nullable=True)
    stage = Column(String, nullable=False)  # "chunking","retrieval","generation"
    module = Column(String, nullable=False) # e.g., "adaptive","advanced","ollama"
    config = Column(JSON, nullable=True)
    reward = Column(Float, nullable=True)   # backfilled from feedback


def create_session(db_url: str | None = None):
    """Create a SQLAlchemy session factory.

    Priority: env var RAG_FEEDBACK_DB_URL -> provided db_url -> sqlite fallback.
    """
    effective_url = os.environ.get("RAG_FEEDBACK_DB_URL", db_url or "sqlite:///rag_feedback.db")
    engine = create_engine(effective_url)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)

