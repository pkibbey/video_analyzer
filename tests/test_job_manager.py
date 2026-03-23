"""Tests for job management system."""

import json
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from video_analyzer.job_manager import AnalysisJob, JobManager, JobStatus


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def job_manager(temp_db):
    """Create a JobManager instance with temporary database."""
    return JobManager(db_path=temp_db)


class TestJobCreation:
    def test_create_job_basic(self, job_manager):
        """Test creating a basic job."""
        job = job_manager.create_job("test_video.mp4")
        
        assert job.job_id is not None
        assert job.video_path == "test_video.mp4"
        assert job.status == JobStatus.ANALYZING
        assert job.created_at is not None
        assert job.started_at is None
        assert job.completed_at is None

    def test_create_job_with_parameters(self, job_manager):
        """Test creating a job with parameters."""
        params = {"frame_model": "custom-model", "max_frames": 64}
        job = job_manager.create_job("test_video.mp4", parameters=params)
        
        assert job.parameters == params

    def test_create_multiple_jobs(self, job_manager):
        """Test creating multiple jobs generates unique IDs."""
        job1 = job_manager.create_job("video1.mp4")
        job2 = job_manager.create_job("video2.mp4")
        
        assert job1.job_id != job2.job_id
        assert job1.video_path == "video1.mp4"
        assert job2.video_path == "video2.mp4"


class TestJobRetrieval:
    def test_get_job_by_id(self, job_manager):
        """Test retrieving a job by ID."""
        created_job = job_manager.create_job("test_video.mp4")
        retrieved_job = job_manager.get_job(created_job.job_id)
        
        assert retrieved_job is not None
        assert retrieved_job.job_id == created_job.job_id
        assert retrieved_job.video_path == created_job.video_path

    def test_get_nonexistent_job(self, job_manager):
        """Test retrieving a nonexistent job returns None."""
        job = job_manager.get_job("nonexistent_id")
        assert job is None

    def test_get_all_jobs(self, job_manager):
        """Test retrieving all jobs."""
        job1 = job_manager.create_job("video1.mp4")
        job2 = job_manager.create_job("video2.mp4")
        job3 = job_manager.create_job("video3.mp4")
        
        all_jobs = job_manager.get_all_jobs()
        
        assert len(all_jobs) == 3
        job_ids = {job.job_id for job in all_jobs}
        assert job1.job_id in job_ids
        assert job2.job_id in job_ids
        assert job3.job_id in job_ids

    def test_get_all_jobs_empty(self, job_manager):
        """Test get_all_jobs on empty database."""
        all_jobs = job_manager.get_all_jobs()
        assert all_jobs == []

    def test_get_all_jobs_ordered_by_date(self, job_manager):
        """Test all jobs are ordered by creation date (descending)."""
        jobs_created = []
        for i in range(3):
            job = job_manager.create_job(f"video{i}.mp4")
            jobs_created.append(job)
        
        all_jobs = job_manager.get_all_jobs()
        
        # Should be in reverse order of creation (most recent first)
        assert all_jobs[0].job_id == jobs_created[-1].job_id


class TestJobStatusUpdate:
    def test_update_job_to_processing(self, job_manager):
        """Test updating job status to ANALYZING."""
        job = job_manager.create_job("test_video.mp4")
        updated_job = job_manager.update_job_status(job.job_id, JobStatus.ANALYZING)
        
        assert updated_job is not None
        assert updated_job.status == JobStatus.ANALYZING
        assert updated_job.started_at is not None

    def test_update_job_to_completed(self, job_manager):
        """Test updating job status to COMPLETED."""
        job = job_manager.create_job("test_video.mp4")
        result = {"summary": "Test summary", "frames": []}
        updated_job = job_manager.update_job_status(
            job.job_id, JobStatus.ANALYZED, result=result
        )
        
        assert updated_job is not None
        assert updated_job.status == JobStatus.ANALYZED
        assert updated_job.result == result
        assert updated_job.completed_at is not None

    def test_update_job_to_failed(self, job_manager):
        """Test updating job status to FAILED."""
        job = job_manager.create_job("test_video.mp4")
        error_msg = "Video file not found"
        updated_job = job_manager.update_job_status(
            job.job_id, JobStatus.FAILED, error=error_msg
        )
        
        assert updated_job is not None
        assert updated_job.status == JobStatus.FAILED
        assert updated_job.error == error_msg
        assert updated_job.completed_at is not None

    def test_update_job_to_cancelled(self, job_manager):
        """Test updating job status to CANCELLED."""
        job = job_manager.create_job("test_video.mp4")
        updated_job = job_manager.update_job_status(job.job_id, JobStatus.CANCELLED)
        
        assert updated_job is not None
        assert updated_job.status == JobStatus.CANCELLED
        assert updated_job.completed_at is not None

    def test_update_nonexistent_job(self, job_manager):
        """Test updating a nonexistent job returns None."""
        updated_job = job_manager.update_job_status(
            "nonexistent_id", JobStatus.ANALYZING
        )
        assert updated_job is None

    def test_started_at_not_reset(self, job_manager):
        """Test started_at is only set once."""
        job = job_manager.create_job("test_video.mp4")
        job_processing = job_manager.update_job_status(job.job_id, JobStatus.ANALYZING)
        started_at_first = job_processing.started_at
        
        # Update to another status
        job_completed = job_manager.update_job_status(
            job.job_id, JobStatus.ANALYZED
        )
        
        assert job_completed.started_at == started_at_first


class TestAnalysisJobModel:
    def test_job_to_dict_serialization(self):
        """Test AnalysisJob converts to dict correctly."""
        job = AnalysisJob(
            job_id="test_id",
            video_path="test.mp4",
            status=JobStatus.ANALYZED,
            result={"data": "test"},
        )
        job_dict = job.to_dict()
        
        assert job_dict["job_id"] == "test_id"
        assert job_dict["video_path"] == "test.mp4"
        assert job_dict["status"] == "analyzed"
        assert job_dict["result"] == {"data": "test"}
        assert isinstance(job_dict["created_at"], str)

    def test_job_to_dict_with_timestamps(self):
        """Test to_dict includes optional timestamps."""
        now = datetime.utcnow()
        job = AnalysisJob(
            job_id="test_id",
            video_path="test.mp4",
            status=JobStatus.ANALYZED,
            started_at=now,
            completed_at=now,
        )
        job_dict = job.to_dict()
        
        assert isinstance(job_dict["started_at"], str)
        assert isinstance(job_dict["completed_at"], str)

    def test_job_to_dict_without_optional_fields(self):
        """Test to_dict handles missing optional fields."""
        job = AnalysisJob(
            job_id="test_id",
            video_path="test.mp4",
            started_at=None,
            completed_at=None,
        )
        job_dict = job.to_dict()
        
        # Optional fields should not be present or be None
        assert job_dict["started_at"] is None
        assert job_dict["completed_at"] is None


class TestConcurrency:
    def test_concurrent_job_creation(self, job_manager):
        """Test thread safety of job creation."""
        import threading
        
        jobs_created = []
        
        def create_job(video_num):
            job = job_manager.create_job(f"video{video_num}.mp4")
            jobs_created.append(job)
        
        threads = [
            threading.Thread(target=create_job, args=(i,))
            for i in range(5)
        ]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All jobs should be created
        assert len(jobs_created) == 5
        # All job IDs should be unique
        job_ids = [job.job_id for job in jobs_created]
        assert len(set(job_ids)) == 5

    def test_concurrent_job_updates(self, job_manager):
        """Test thread safety of job updates."""
        import threading
        
        job = job_manager.create_job("test_video.mp4")
        update_count = [0]
        
        def update_job(status):
            job_manager.update_job_status(job.job_id, status)
            update_count[0] += 1
        
        threads = [
            threading.Thread(target=update_job, args=(JobStatus.ANALYZING,))
            for _ in range(3)
        ]
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All updates should complete
        assert update_count[0] == 3
        # Job should exist and have final status
        final_job = job_manager.get_job(job.job_id)
        assert final_job is not None


class TestJobStatusTransitions:
    def test_full_job_lifecycle(self, job_manager):
        """Test a job through its complete lifecycle."""
        # Create
        job = job_manager.create_job("test_video.mp4")
        assert job.status == JobStatus.ANALYZING
        
        # Start analyzing
        job = job_manager.update_job_status(job.job_id, JobStatus.ANALYZING)
        assert job.status == JobStatus.ANALYZING
        assert job.started_at is not None
        
        # Complete
        result = {"summary": "Analysis complete"}
        job = job_manager.update_job_status(
            job.job_id, JobStatus.ANALYZED, result=result
        )
        assert job.status == JobStatus.ANALYZED
        assert job.result == result
        assert job.completed_at is not None

    def test_failed_job_lifecycle(self, job_manager):
        """Test a job that fails during analyzing."""
        # Create
        job = job_manager.create_job("test_video.mp4")
        
        # Start analyzing
        job = job_manager.update_job_status(job.job_id, JobStatus.ANALYZING)
        assert job.started_at is not None
        
        # Fail
        error = "Processing unanalyzed: Model error"
        job = job_manager.update_job_status(
            job.job_id, JobStatus.FAILED, error=error
        )
        assert job.status == JobStatus.FAILED
        assert job.error == error
        assert job.completed_at is not None

    def test_cancelled_job_lifecycle(self, job_manager):
        """Test a job that is analysis-cancelled."""
        # Create
        job = job_manager.create_job("test_video.mp4")
        
        # Start analyzing
        job = job_manager.update_job_status(job.job_id, JobStatus.ANALYZING)
        
        # Cancel
        job = job_manager.update_job_status(job.job_id, JobStatus.CANCELLED)
        assert job.status == JobStatus.CANCELLED
        assert job.completed_at is not None
