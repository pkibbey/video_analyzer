"""Job management for background video analysis tasks."""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Status of an analysis job."""

    ANALYZING = "analyzing"
    ANALYZED = "analyzed"
    UNANALYZED = "unanalyzed"
    CANCELLED = "analysis-cancelled"


@dataclass
class AnalysisJob:
    """Represents a video analysis job."""

    job_id: str
    video_path: str
    status: JobStatus = JobStatus.ANALYZING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[dict] = None
    error: Optional[str] = None
    parameters: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, handling datetime and enum serialization."""
        data = asdict(self)
        data["status"] = self.status.value
        data["created_at"] = self.created_at.isoformat()
        if self.started_at:
            data["started_at"] = self.started_at.isoformat()
        if self.completed_at:
            data["completed_at"] = self.completed_at.isoformat()
        return data


class JobManager:
    """Manages job queue and persistence."""

    def __init__(self, db_path: str | Path = ".jobs.db"):
        """Initialize job manager with SQLite database.
        
        Args:
            db_path: Path to SQLite database file for job storage
        """
        self.db_path = Path(db_path)
        self._lock = threading.RLock()
        self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    video_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    result TEXT,
                    error TEXT,
                    parameters TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def create_job(
        self, video_path: str, parameters: dict | None = None
    ) -> AnalysisJob:
        """Create a new analysis job.
        
        Args:
            video_path: Path to video file
            parameters: Optional analysis parameters
            
        Returns:
            AnalysisJob instance
        """
        job_id = str(uuid.uuid4())
        job = AnalysisJob(
            job_id=job_id,
            video_path=video_path,
            parameters=parameters or {},
        )

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO jobs
                    (job_id, video_path, status, created_at, parameters)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        job.job_id,
                        job.video_path,
                        job.status.value,
                        job.created_at.isoformat(),
                        json.dumps(job.parameters),
                    ),
                )
                conn.commit()

        logger.info(f"Created job {job_id} for video: {video_path}")
        return job

    def get_job(self, job_id: str) -> AnalysisJob | None:
        """Get job by ID.
        
        Args:
            job_id: Job ID to retrieve
            
        Returns:
            AnalysisJob or None if not found
        """
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
                )
                row = cursor.fetchone()

        if not row:
            return None

        return self._row_to_job(row)

    def get_all_jobs(self) -> list[AnalysisJob]:
        """Get all jobs."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC")
                rows = cursor.fetchall()

        return [self._row_to_job(row) for row in rows]

    def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        result: dict | None = None,
        error: str | None = None,
    ) -> AnalysisJob | None:
        """Update job status and optionally set result or error.
        
        Args:
            job_id: Job ID to update
            status: New job status
            result: Optional result data (for analyzed jobs)
            error: Optional error message (for unanalyzed jobs)
            
        Returns:
            Updated AnalysisJob or None if not found
        """
        job = self.get_job(job_id)
        if not job:
            return None

        now = datetime.utcnow()
        if status == JobStatus.ANALYZING and not job.started_at:
            job.started_at = now
        elif status in (JobStatus.ANALYZED, JobStatus.UNANALYZED, JobStatus.CANCELLED):
            job.completed_at = now

        job.status = status
        if result is not None:
            job.result = result
        if error is not None:
            job.error = error

        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE jobs
                    SET status = ?, started_at = ?, completed_at = ?, result = ?, error = ?
                    WHERE job_id = ?
                    """,
                    (
                        status.value,
                        job.started_at.isoformat() if job.started_at else None,
                        job.completed_at.isoformat() if job.completed_at else None,
                        json.dumps(job.result) if job.result else None,
                        error,
                        job_id,
                    ),
                )
                conn.commit()

        logger.info(f"Updated job {job_id} status to {status.value}")
        return job

    def delete_job(self, job_id: str) -> bool:
        """Delete a job.
        
        Args:
            job_id: Job ID to delete
            
        Returns:
            True if job was deleted, False if not found
        """
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
                conn.commit()
                return cursor.rowcount > 0

    @staticmethod
    def _row_to_job(row: tuple) -> AnalysisJob:
        """Convert database row to AnalysisJob object."""
        (
            job_id,
            video_path,
            status,
            created_at,
            started_at,
            completed_at,
            result,
            error,
            parameters,
        ) = row

        normalized_status = status
        if status == "pending":
            normalized_status = JobStatus.ANALYZING.value
        elif status == "processing":
            normalized_status = JobStatus.ANALYZING.value
        elif status == "completed":
            normalized_status = JobStatus.ANALYZED.value
        elif status == "failed":
            normalized_status = JobStatus.UNANALYZED.value
        elif status == "cancelled":
            normalized_status = JobStatus.CANCELLED.value

        return AnalysisJob(
            job_id=job_id,
            video_path=video_path,
            status=JobStatus(normalized_status),
            created_at=datetime.fromisoformat(created_at),
            started_at=datetime.fromisoformat(started_at) if started_at else None,
            completed_at=datetime.fromisoformat(completed_at)
            if completed_at
            else None,
            result=json.loads(result) if result else None,
            error=error,
            parameters=json.loads(parameters),
        )
