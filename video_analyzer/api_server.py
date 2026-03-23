"""FastAPI service for video analysis."""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from video_analyzer.analyzer import OllamaVideoAnalyzer
from video_analyzer.frame_selectors import (
    AllFrameSelector,
    DynamicFrameSelector,
    UniformFrameSelector,
)
from video_analyzer.job_manager import JobManager, JobStatus
from video_analyzer.models import AnalysisPrompts
from video_analyzer.transcriber import WhisperTranscriber

# Load environment variables
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global job manager and worker thread
job_manager: Optional[JobManager] = None
worker_thread: Optional[threading.Thread] = None
worker_running = False


# ============================================================================
# Pydantic Models
# ============================================================================


class AnalysisParameters(BaseModel):
    """Parameters for video analysis."""

    frame_model: str = Field(
        default="ministral-3:3b-cloud", description="Model for frame analysis"
    )
    summary_model: str = Field(
        default="ministral-3:14b-cloud", description="Model for summary generation"
    )
    whisper_model: str = Field(
        default="openai/whisper-small", description="Whisper model for audio"
    )
    host: str = Field(
        default="http://localhost:11434", description="Ollama server URL"
    )
    min_frames: int = Field(default=4, ge=1, description="Minimum frames to analyze")
    max_frames: int = Field(default=16, ge=1, description="Maximum frames to analyze")
    frames_per_minute: float = Field(
        default=4.0, gt=0, description="Frame sampling rate"
    )
    frame_selector: str = Field(
        default="dynamic",
        description="Frame selector method",
        pattern="^(dynamic|uniform|all)$",
    )
    dynamic_threshold: float = Field(
        default=20.0, ge=0, description="Threshold for dynamic frame selection"
    )
    audio: bool = Field(default=True, description="Enable audio transcription")
    device: Optional[str] = Field(
        default=None, description="Device for Whisper (cuda, cpu, etc.)"
    )
    no_collapse_repetitions: bool = Field(
        default=False, description="Keep repeated phrases in transcript"
    )
    segment_duration: int = Field(
        default=15, gt=0, description="Audio segment duration in seconds"
    )
    beam_size: int = Field(default=5, ge=1, description="Beam size for decoding")
    temperature: float = Field(
        default=0.0, ge=0, description="Temperature for decoding"
    )
    condition_on_prev_tokens: bool = Field(
        default=False, description="Use previous tokens when decoding"
    )
    local_files_only: bool = Field(
        default=True, description="Only load Whisper model from local cache"
    )
    timeout: float = Field(default=120.0, gt=0, description="Request timeout in seconds")
    retries: int = Field(default=3, ge=0, description="Number of retries")
    retry_backoff: float = Field(default=1.0, ge=0, description="Backoff multiplier")
    max_detailed_chars: int = Field(
        default=0, description="Max characters for detailed summary (0=unlimited)"
    )
    max_brief_chars: int = Field(
        default=0, description="Max characters for brief summary (0=unlimited)"
    )
    analyze_quality: bool = Field(
        default=True, description="Analyze frame quality metrics"
    )


class SubmitJobRequest(BaseModel):
    """Request to submit a video analysis job."""

    video_path: str = Field(description="Path to video file on server")
    parameters: Optional[AnalysisParameters] = Field(
        default=None, description="Analysis parameters"
    )


class SummaryResultResponse(BaseModel):
    """Summary section of analysis result."""

    detailed: str
    brief: str
    timeline: Optional[str] = None
    transcript: Optional[str] = None

    class Config:
        extra = "forbid"


class FrameAnalysisResponse(BaseModel):
    timestamp: float
    description: str
    scene_type: str
    error: Optional[str] = None
    quality_scores: Optional[str] = None
    quality_analysis: Optional[str] = None

    class Config:
        extra = "forbid"


class AudioSegmentResponse(BaseModel):
    text: Optional[str]
    start_time: float
    end_time: float
    confidence: float

    class Config:
        extra = "forbid"


class ModelsUsedResponse(BaseModel):
    frame_analysis: str
    summary: str
    audio: Optional[str] = None

    class Config:
        extra = "forbid"


class ProcessingTimingsResponse(BaseModel):
    frame_selection: float
    audio_transcription: float
    frame_analysis: float
    summary_generation: float
    total: float

    class Config:
        extra = "forbid"


class VideoPropertiesResponse(BaseModel):
    fps: Optional[float] = None
    total_frames: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    codec: Optional[str] = None
    bitrate: Optional[int] = None
    format: Optional[str] = None
    duration: Optional[float] = None
    data_rate: Optional[str] = None
    audio_codec: Optional[str] = None
    audio_sample_rate: Optional[int] = None
    file_size: Optional[int] = None
    file_modified_date: Optional[str] = None
    file_created_date: Optional[str] = None

    class Config:
        extra = "forbid"


class AnalysisMetadataResponse(BaseModel):
    num_frames_analyzed: int
    num_audio_segments: int
    video_duration: float
    scene_distribution: Dict[str, int]
    models_used: ModelsUsedResponse
    processing_timings: Optional[ProcessingTimingsResponse] = None
    video_properties: Optional[VideoPropertiesResponse] = None

    class Config:
        extra = "forbid"


class AnalysisResultResponse(BaseModel):
    summary: SummaryResultResponse
    frame_analyses: List[FrameAnalysisResponse]
    audio_segments: List[AudioSegmentResponse]
    metadata: AnalysisMetadataResponse
    warnings: List[str] = []

    class Config:
        extra = "forbid"


class JobResponse(BaseModel):
    """Response containing job information."""

    job_id: str
    video_path: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


class JobResultResponse(BaseModel):
    """Response containing job result."""

    job_id: str
    status: str
    result: Optional[AnalysisResultResponse] = None
    error: Optional[str] = None

    class Config:
        extra = "forbid"


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    worker_running: bool
    jobs_db_exists: bool


# ============================================================================
# FastAPI Application
# ============================================================================


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="VideoAnalyzer API",
        description="Video analysis service with job queue",
        version="1.0.0",
    )

    global job_manager

    # Initialize job manager
    if job_manager is None:
        job_manager = JobManager()

    @app.on_event("startup")
    async def startup_event() -> None:
        """Start background worker on app startup."""
        logger.info("Starting VideoAnalyzer API")
        start_worker()

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        """Stop background worker on app shutdown."""
        logger.info("Shutting down VideoAnalyzer API")
        stop_worker()

    @app.get("/", tags=["Info"])
    async def root() -> dict[str, str]:
        """API root endpoint with documentation."""
        logger.info("GET / called")
        return {
            "name": "VideoAnalyzer API",
            "description": "Video analysis service",
            "docs": "/docs",
            "openapi_schema": "/openapi.json",
        }

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health() -> HealthResponse:
        """Check API health."""
        logger.info("GET /health called")
        health_resp = HealthResponse(
            status="healthy",
            worker_running=worker_running,
            jobs_db_exists=job_manager.db_path.exists(),
        )
        logger.debug("Health response: %s", health_resp.dict())
        return health_resp

    @app.post("/analyze", response_model=JobResponse, tags=["Analysis"])
    async def submit_analysis(request: SubmitJobRequest) -> JobResponse:
        """Submit a new video analysis job.
        
        The API will process the video in the background. Use the returned job_id
        to check status and retrieve results.
        """
        logger.info("POST /analyze called: video_path=%s params=%s", request.video_path, request.parameters)

        # Validate video path
        video_path = Path(request.video_path)
        if not video_path.exists():
            logger.warning("Invalid video path: %s does not exist", request.video_path)
            raise HTTPException(
                status_code=400,
                detail=f"Video file not found: {request.video_path}",
            )

        if not video_path.is_file():
            logger.warning("Invalid video path: %s is not a file", request.video_path)
            raise HTTPException(
                status_code=400,
                detail=f"Path is not a file: {request.video_path}",
            )

        # Create job
        params = request.parameters.dict() if request.parameters else {}
        job = job_manager.create_job(str(video_path), params)
        response = JobResponse(
            job_id=job.job_id,
            video_path=job.video_path,
            status=job.status.value,
            created_at=job.created_at.isoformat(),
        )
        logger.info("Job created: %s", response.job_id)
        return response

    @app.get("/jobs", response_model=list[JobResponse], tags=["Jobs"])
    async def list_jobs(
        status: Optional[str] = Query(
            None, description="Filter by status (analyzing, analyzed, unanalyzed, analysis-cancelled)"
        ),
    ) -> list[JobResponse]:
        """List all jobs, optionally filtered by status."""
        logger.info("GET /jobs called status=%s", status)
        all_jobs = job_manager.get_all_jobs()

        if status:
            all_jobs = [j for j in all_jobs if j.status.value == status]

        response = [
            JobResponse(
                job_id=j.job_id,
                video_path=j.video_path,
                status=j.status.value,
                created_at=j.created_at.isoformat(),
                started_at=j.started_at.isoformat() if j.started_at else None,
                completed_at=j.completed_at.isoformat() if j.completed_at else None,
                error=j.error,
            )
            for j in all_jobs
        ]
        logger.debug("/jobs response count=%d", len(response))
        return response

    @app.get("/jobs/{job_id}", response_model=JobResponse, tags=["Jobs"])
    async def get_job(job_id: str) -> JobResponse:
        """Get job status by ID."""
        logger.info("GET /jobs/%s called", job_id)
        job = job_manager.get_job(job_id)
        if not job:
            logger.warning("Job not found: %s", job_id)
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        response = JobResponse(
            job_id=job.job_id,
            video_path=job.video_path,
            status=job.status.value,
            created_at=job.created_at.isoformat(),
            started_at=job.started_at.isoformat() if job.started_at else None,
            completed_at=job.completed_at.isoformat() if job.completed_at else None,
            error=job.error,
        )
        logger.debug("/jobs/%s response: %s", job_id, response.dict())
        return response

    @app.get("/jobs/{job_id}/result", response_model=JobResultResponse, tags=["Jobs"])
    async def get_job_result(job_id: str) -> JobResultResponse:
        """Get analysis result for analyzed job."""
        logger.info("GET /jobs/%s/result called", job_id)
        job = job_manager.get_job(job_id)
        if not job:
            logger.warning("Job result not found: %s", job_id)
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        if job.status == JobStatus.ANALYZING:
            logger.debug("Job %s analyzing", job_id)
            raise HTTPException(
                status_code=202,
                detail="Job is still analyzing",
            )

        if job.status == JobStatus.UNANALYZED:
            logger.error("Job %s unanalyzed: %s", job_id, job.error)
            raise HTTPException(
                status_code=500,
                detail=f"Job unanalyzed: {job.error}",
            )

        if job.status == JobStatus.CANCELLED:
            logger.warning("Job %s analysis-cancelled", job_id)
            raise HTTPException(
                status_code=400,
                detail="Job was analysis-cancelled",
            )

        response = JobResultResponse(
            job_id=job.job_id,
            status=job.status.value,
            result=AnalysisResultResponse.parse_obj(job.result) if job.result is not None else None,
            error=job.error,
        )
        logger.debug("Job %s result returned", job_id)
        return response

    @app.delete("/jobs/{job_id}", tags=["Jobs"])
    async def cancel_job(job_id: str) -> dict[str, str]:
        """Cancel or delete a job."""
        logger.info("DELETE /jobs/%s called", job_id)
        job = job_manager.get_job(job_id)
        if not job:
            logger.warning("Job not found for delete/cancel: %s", job_id)
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        if job.status in (JobStatus.ANALYZED, JobStatus.UNANALYZED):
            # Delete analyzed/unanalyzed jobs
            job_manager.delete_job(job_id)
            logger.info("Job %s deleted", job_id)
            return {"message": f"Job {job_id} deleted"}

        # Cancel analyzing jobs
        job_manager.update_job_status(job_id, JobStatus.CANCELLED)
        logger.info("Job %s analysis-cancelled", job_id)
        return {"message": f"Job {job_id} analysis-cancelled"}

    return app


# ============================================================================
# Worker Thread
# ============================================================================


def process_jobs() -> None:
    """Worker thread that processes jobs in analyzing state."""
    logger.info("Job worker started")

    while worker_running:
        job = None
        try:
            # Get next job that is analyzing but not yet started
            all_jobs = job_manager.get_all_jobs()
            analyzing_jobs = [j for j in all_jobs if j.status == JobStatus.ANALYZING and j.started_at is None]

            if not analyzing_jobs:
                time.sleep(1)
                continue

            job = analyzing_jobs[0]
            logger.info(f"Processing job {job.job_id}: {job.video_path}")

            # Mark started (still analyzing status)
            job_manager.update_job_status(job.job_id, JobStatus.ANALYZING)

            # Build analyzer with job parameters
            params = job.parameters or {}
            frame_model = params.get("frame_model", "ministral-3:3b-cloud")
            summary_model = params.get("summary_model", "ministral-3:14b-cloud")
            whisper_model = params.get("whisper_model", "openai/whisper-small")
            host = params.get("host", "http://localhost:11434")
            min_frames = params.get("min_frames", 4)
            max_frames = params.get("max_frames", 16)
            frames_per_minute = params.get("frames_per_minute", 4.0)
            frame_selector_name = params.get("frame_selector", "dynamic")
            dynamic_threshold = params.get("dynamic_threshold", 20.0)
            enable_audio = params.get("audio", True)
            device = params.get("device")
            timeout = params.get("timeout", 120.0)
            retries = params.get("retries", 3)
            retry_backoff = params.get("retry_backoff", 1.0)
            analyze_quality = params.get("analyze_quality", True)
            max_detailed_chars = params.get("max_detailed_chars", 0)
            max_brief_chars = params.get("max_brief_chars", 0)

            # Build frame selector
            selector_map = {
                "dynamic": DynamicFrameSelector(threshold=dynamic_threshold),
                "uniform": UniformFrameSelector(),
                "all": AllFrameSelector(),
            }
            frame_selector = selector_map.get(
                frame_selector_name, DynamicFrameSelector(threshold=dynamic_threshold)
            )

            # Build audio transcriber
            audio_transcriber = None
            if enable_audio:
                audio_transcriber = WhisperTranscriber(
                    model_name=whisper_model,
                    device=device,
                    collapse_repetitions=not params.get(
                        "no_collapse_repetitions", False
                    ),
                    segment_duration=params.get("segment_duration", 15),
                    beam_size=params.get("beam_size", 5),
                    temperature=params.get("temperature", 0.0),
                    condition_on_prev_tokens=params.get("condition_on_prev_tokens", False),
                    local_files_only=params.get("local_files_only", True),
                )

            # Create analyzer
            analyzer = OllamaVideoAnalyzer(
                frame_analysis_model=frame_model,
                summary_model=summary_model,
                host=host,
                min_frames=min_frames,
                max_frames=max_frames,
                frames_per_minute=frames_per_minute,
                frame_selector=frame_selector,
                audio_transcriber=audio_transcriber,
                request_timeout=timeout,
                request_retries=retries,
                request_backoff=retry_backoff,
                max_detailed_summary_chars=max_detailed_chars,
                max_brief_summary_chars=max_brief_chars,
                analyze_quality=analyze_quality,
            )

            # Analyze video
            result_obj = analyzer.analyze_video_structured(job.video_path)
            result_dict = result_obj.to_dict()

            # Mark as analyzed
            job_manager.update_job_status(
                job.job_id, JobStatus.ANALYZED, result=result_dict
            )
            logger.info(f"Completed job {job.job_id}")

        except Exception as e:
            job_id = job.job_id if job is not None else "<unknown>"
            logger.error(
                "Error analyzing job %s: %s",
                job_id,
                str(e),
                exc_info=True,
            )
            if job is not None:
                job_manager.update_job_status(
                    job.job_id, JobStatus.UNANALYZED, error=str(e)
                )
            time.sleep(1)


def start_worker() -> None:
    """Start background worker thread."""
    global worker_running, worker_thread

    if worker_running:
        return

    worker_running = True
    worker_thread = threading.Thread(target=process_jobs, daemon=True)
    worker_thread.start()
    logger.info("Worker thread started")


def stop_worker() -> None:
    """Stop background worker thread."""
    global worker_running, worker_thread

    worker_running = False
    if worker_thread:
        worker_thread.join(timeout=5)
    logger.info("Worker thread stopped")


# ============================================================================
# CLI Entry Point
# ============================================================================


def run() -> None:
    """Run FastAPI server via uvicorn."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(
        description="Run VideoAnalyzer API server"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=3002, help="Server port (default: 3002)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)",
    )
    parser.add_argument(
        "--db", default=".jobs.db", help="Job database path (default: .jobs.db)"
    )

    args = parser.parse_args()

    # Set database path globally before creating app
    global job_manager
    job_manager = JobManager(args.db)

    # Create app
    app = create_app()

    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
    )


if __name__ == "__main__":
    run()
