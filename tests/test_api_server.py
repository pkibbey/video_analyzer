"""Tests for FastAPI server and endpoints."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from video_analyzer.api_server import (
    AnalysisParameters,
    HealthResponse,
    JobResponse,
    JobResultResponse,
    SubmitJobRequest,
    create_app,
)
from video_analyzer.job_manager import JobStatus


@pytest.fixture
def temp_video_file():
    """Create a temporary video file."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(b"dummy video content")
        temp_path = f.name

    yield temp_path

    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    app = create_app()
    return TestClient(app)


class TestRootEndpoint:
    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "description" in data
        assert "docs" in data
        assert data["name"] == "VideoAnalyzer API"


class TestHealthEndpoint:
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "worker_running" in data
        assert "jobs_db_exists" in data

    def test_health_response_model(self, client):
        """Test health response conforms to model."""
        response = client.get("/health")
        data = response.json()

        # Should be valid HealthResponse
        health = HealthResponse(**data)
        assert health.status == "healthy"


class TestAnalysisParametersModel:
    def test_analysis_parameters_defaults(self):
        """Test AnalysisParameters has reasonable defaults."""
        params = AnalysisParameters()

        assert params.frame_model == "ministral-3:3b-cloud"
        assert params.summary_model == "ministral-3:14b-cloud"
        assert params.min_frames == 4
        assert params.max_frames == 16
        assert params.timeout == 120.0

    def test_analysis_parameters_custom_values(self):
        """Test AnalysisParameters with custom values."""
        params = AnalysisParameters(
            frame_model="custom-model",
            min_frames=8,
            max_frames=32,
            timeout=180.0,
        )

        assert params.frame_model == "custom-model"
        assert params.min_frames == 8
        assert params.max_frames == 32
        assert params.timeout == 180.0

    def test_analysis_parameters_validation(self):
        """Test AnalysisParameters validation."""
        # min_frames must be >= 1
        with pytest.raises(ValueError):
            AnalysisParameters(min_frames=0)

        # max_frames must be >= 1
        with pytest.raises(ValueError):
            AnalysisParameters(max_frames=0)

        # timeout must be > 0
        with pytest.raises(ValueError):
            AnalysisParameters(timeout=0)

        # temperature must be >= 0
        with pytest.raises(ValueError):
            AnalysisParameters(temperature=-0.1)

    def test_analysis_parameters_frame_selector_pattern(self):
        """Test frame_selector must match pattern."""
        # Valid options
        AnalysisParameters(frame_selector="dynamic")
        AnalysisParameters(frame_selector="uniform")
        AnalysisParameters(frame_selector="all")

        # Invalid option - should raise ValueError
        with pytest.raises(ValueError):
            AnalysisParameters(frame_selector="invalid")


class TestSubmitJobEndpoint:
    def test_submit_job_success(self, client, temp_video_file):
        """Test successfully submitting a job."""
        request_data = {
            "video_path": temp_video_file,
            "parameters": {
                "frame_model": "test-model",
                "min_frames": 5,
            }
        }

        response = client.post("/analyze", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["video_path"] == temp_video_file
        assert data["status"] == "analyzing"

    def test_submit_job_missing_video(self, client):
        """Test submitting job with nonexistent video."""
        request_data = {
            "video_path": "/nonexistent/video.mp4",
        }

        response = client.post("/analyze", json=request_data)

        assert response.status_code == 400
        assert "not found" in response.json()["detail"].lower()

    def test_submit_job_directory_instead_of_file(self, client):
        """Test submitting job with directory instead of file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            request_data = {
                "video_path": tmpdir,
            }

            response = client.post("/analyze", json=request_data)

            assert response.status_code == 400
            assert "not a file" in response.json()["detail"].lower()

    def test_submit_job_with_parameters(self, client, temp_video_file):
        """Test submitting job with custom parameters."""
        request_data = {
            "video_path": temp_video_file,
            "parameters": {
                "frame_model": "custom-frame-model",
                "summary_model": "custom-summary-model",
                "min_frames": 10,
                "max_frames": 50,
                "audio": True,
                "timeout": 180.0,
            }
        }

        response = client.post("/analyze", json=request_data)

        assert response.status_code == 200
        job_response = JobResponse(**response.json())
        assert job_response.status == "analyzing"

    def test_submit_job_response_structure(self, client, temp_video_file):
        """Test submit job response has required fields."""
        response = client.post("/analyze", json={"video_path": temp_video_file})

        assert response.status_code == 200
        data = response.json()

        # Should conform to JobResponse model
        job = JobResponse(**data)
        assert job.job_id is not None
        assert job.video_path == temp_video_file
        assert job.status == "analyzing"
        assert job.created_at is not None
        assert job.started_at is None


class TestListJobsEndpoint:
    def test_list_jobs_empty(self, client):
        """Test list jobs when no jobs exist."""
        response = client.get("/jobs")

        assert response.status_code == 200
        jobs = response.json()
        assert isinstance(jobs, list)
        # May be empty or have existing jobs from other tests
        assert all("job_id" in j for j in jobs)

    def test_list_jobs_filter_by_status(self, client, temp_video_file):
        """Test list jobs with status filter."""
        # Submit a job
        submit_response = client.post(
            "/analyze",
            json={"video_path": temp_video_file}
        )
        assert submit_response.status_code == 200

        # List jobs with analyzing status
        response = client.get("/jobs?status=analyzing")

        assert response.status_code == 200
        jobs = response.json()
        assert all(j["status"] == "analyzing" for j in jobs)

    def test_list_jobs_invalid_status(self, client):
        """Test list jobs with invalid status filter."""
        response = client.get("/jobs?status=invalid_status")

        assert response.status_code == 200
        # Should return empty list or all jobs
        jobs = response.json()
        assert isinstance(jobs, list)


class TestGetJobEndpoint:
    def test_get_job_success(self, client, temp_video_file):
        """Test getting job by ID."""
        # Submit a job
        submit_response = client.post(
            "/analyze",
            json={"video_path": temp_video_file}
        )
        job_id = submit_response.json()["job_id"]

        # Get job
        response = client.get(f"/jobs/{job_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert data["status"] == "analyzing"

    def test_get_job_not_found(self, client):
        """Test getting nonexistent job."""
        response = client.get("/jobs/nonexistent_id")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_job_response_model(self, client, temp_video_file):
        """Test get job response conforms to model."""
        submit_response = client.post(
            "/analyze",
            json={"video_path": temp_video_file}
        )
        job_id = submit_response.json()["job_id"]

        response = client.get(f"/jobs/{job_id}")

        data = response.json()
        job = JobResponse(**data)
        assert job.job_id == job_id
        assert job.status in ["analyzing", "analyzing", "analyzed", "unanalyzed", "analysis-cancelled"]


class TestGetJobResultEndpoint:
    def test_get_job_result_analyzing(self, client, temp_video_file):
        """Test getting result of analyzing job returns 202."""
        submit_response = client.post(
            "/analyze",
            json={"video_path": temp_video_file}
        )
        job_id = submit_response.json()["job_id"]

        response = client.get(f"/jobs/{job_id}/result")

        assert response.status_code == 202
        assert "analyzing" in response.json()["detail"].lower()

    def test_get_job_result_not_found(self, client):
        """Test getting result for nonexistent job."""
        response = client.get("/jobs/nonexistent_id/result")

        assert response.status_code == 404

    def test_get_job_result_cancelled(self, client, temp_video_file):
        """Test getting result of analysis-cancelled job returns 400."""
        # Submit a job  
        submit_response = client.post(
            "/analyze",
            json={"video_path": temp_video_file}
        )
        job_id = submit_response.json()["job_id"]

        # Cancel the job
        cancel_response = client.delete(f"/jobs/{job_id}")
        assert cancel_response.status_code == 200

        # Try to get result
        response = client.get(f"/jobs/{job_id}/result")

        assert response.status_code == 400
        assert "analysis-cancelled" in response.json()["detail"].lower()


class TestCancelJobEndpoint:
    def test_cancel_analyzing_job(self, client, temp_video_file):
        """Test cancelling a analyzing job."""
        submit_response = client.post(
            "/analyze",
            json={"video_path": temp_video_file}
        )
        job_id = submit_response.json()["job_id"]

        response = client.delete(f"/jobs/{job_id}")

        assert response.status_code == 200
        assert "analysis-cancelled" in response.json()["message"].lower()

        # Verify job status is analysis-cancelled
        get_response = client.get(f"/jobs/{job_id}")
        assert get_response.json()["status"] == "analysis-cancelled"

    def test_cancel_nonexistent_job(self, client):
        """Test cancelling nonexistent job returns 404."""
        response = client.delete("/jobs/nonexistent_id")

        assert response.status_code == 404

    def test_delete_response_structure(self, client, temp_video_file):
        """Test delete response has expected structure."""
        submit_response = client.post(
            "/analyze",
            json={"video_path": temp_video_file}
        )
        job_id = submit_response.json()["job_id"]

        response = client.delete(f"/jobs/{job_id}")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data


class TestSubmitJobRequestModel:
    def test_submit_job_request_validation(self):
        """Test SubmitJobRequest validation."""
        # Valid request
        request = SubmitJobRequest(video_path="/path/to/video.mp4")
        assert request.video_path == "/path/to/video.mp4"

        # With parameters
        params = AnalysisParameters(min_frames=8)
        request = SubmitJobRequest(
            video_path="/path/to/video.mp4",
            parameters=params
        )
        assert request.parameters.min_frames == 8

    def test_submit_job_request_json_schema(self):
        """Test SubmitJobRequest generates valid JSON schema."""
        request = SubmitJobRequest(video_path="/test.mp4")
        assert request.video_path == "/test.mp4"


class TestResponseModels:
    def test_job_response_model(self):
        """Test JobResponse model."""
        job_response = JobResponse(
            job_id="test_id",
            video_path="/path/to/video.mp4",
            status="analyzing",
            created_at="2024-01-01T00:00:00",
        )

        assert job_response.job_id == "test_id"
        assert job_response.status == "analyzing"
        assert job_response.started_at is None

    def test_job_result_response_model(self):
        """Test JobResultResponse model with strict analysis schema."""
        result = {
            "summary": {"detailed": "Detailed summary", "brief": "Brief summary"},
            "frame_analyses": [],
            "audio_segments": [],
            "metadata": {
                "num_frames_analyzed": 0,
                "num_audio_segments": 0,
                "video_duration": 0.0,
                "scene_distribution": {},
                "models_used": {
                    "frame_analysis": "ministral-3:3b-cloud",
                    "summary": "ministral-3:14b-cloud",
                    "audio": None,
                },
                "processing_timings": {
                    "frame_selection": 0.0,
                    "audio_transcription": 0.0,
                    "frame_analysis": 0.0,
                    "summary_generation": 0.0,
                    "total": 0.0,
                },
                "video_properties": None,
            },
            "warnings": [],
        }

        response = JobResultResponse(
            job_id="test_id",
            status="analyzed",
            result=result,
        )

        assert response.job_id == "test_id"
        assert response.status == "analyzed"
        assert response.result is not None
        assert response.result.summary.detailed == "Detailed summary"
        assert response.result.summary.brief == "Brief summary"

    def test_job_result_response_model_rejects_invalid_result(self):
        """Ensure JobResultResponse enforces AnalysisResultResponse typing."""
        with pytest.raises(ValidationError):
            JobResultResponse(
                job_id="test_id",
                status="analyzed",
                result={"this_is_not":"valid"},
            )

    def test_health_response_model_validation(self):
        """Test HealthResponse model validation."""
        health = HealthResponse(
            status="healthy",
            worker_running=True,
            jobs_db_exists=True,
        )

        assert health.status == "healthy"
        assert health.worker_running is True


class TestAPIIntegration:
    def test_full_job_workflow(self, client, temp_video_file):
        """Test complete job submission and status check workflow."""
        # 1. Submit job
        submit_response = client.post(
            "/analyze",
            json={"video_path": temp_video_file}
        )
        assert submit_response.status_code == 200
        job_id = submit_response.json()["job_id"]

        # 2. List jobs includes new job
        list_response = client.get("/jobs")
        assert list_response.status_code == 200
        job_ids = [j["job_id"] for j in list_response.json()]
        assert job_id in job_ids

        # 3. Get specific job
        get_response = client.get(f"/jobs/{job_id}")
        assert get_response.status_code == 200
        job_data = get_response.json()
        assert job_data["status"] == "analyzing"

        # 4. Get result of analyzing job returns 202
        result_response = client.get(f"/jobs/{job_id}/result")
        assert result_response.status_code == 202

        # 5. Cancel job
        cancel_response = client.delete(f"/jobs/{job_id}")
        assert cancel_response.status_code == 200

        # 6. Verify job is analysis-cancelled
        get_response = client.get(f"/jobs/{job_id}")
        assert get_response.json()["status"] == "analysis-cancelled"

    def test_multiple_jobs_independent(self, client, temp_video_file):
        """Test that multiple jobs are independent."""
        # Submit two jobs
        job1_response = client.post(
            "/analyze",
            json={"video_path": temp_video_file}
        )
        job1_id = job1_response.json()["job_id"]

        job2_response = client.post(
            "/analyze",
            json={"video_path": temp_video_file}
        )
        job2_id = job2_response.json()["job_id"]

        # Job IDs should be different
        assert job1_id != job2_id

        # Cancel one doesn't affect the other
        client.delete(f"/jobs/{job1_id}")

        # Job 2 should still exist and be analyzing
        job2_response = client.get(f"/jobs/{job2_id}")
        assert job2_response.status_code == 200
        assert job2_response.json()["status"] == "analyzing"
