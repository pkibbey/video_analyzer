# VideoAnalyzer API - Quick Reference

## Installation

```bash
pip install -e .
```

This installs the package with all dependencies including FastAPI and Uvicorn.

## Start the API Server

```bash
# Default (localhost:8000)
video-analyzer-api

# Custom host and port
video-analyzer-api --host 0.0.0.0 --port 8000

# Development with auto-reload
video-analyzer-api --reload

# Production with multiple workers
video-analyzer-api --workers 4

# Custom database location
video-analyzer-api --db /var/lib/jobs.db
```

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Check API health |
| POST | `/analyze` | Submit video for analysis |
| GET | `/jobs` | List all jobs |
| GET | `/jobs/{job_id}` | Get job status |
| GET | `/jobs/{job_id}/result` | Get analysis result |
| DELETE | `/jobs/{job_id}` | Cancel/delete job |

## Using the Python Client

```python
from api_client import APIClient

client = APIClient("http://localhost:8000")

# Submit analysis
job = client.submit_analysis("/path/to/video.mp4", {
    "frame_model": "ministral-3:3b-cloud",
    "min_frames": 4
})

# Wait for completion
result = client.wait_for_completion(job['job_id'])
print(result['result'])

# Or check status manually
status = client.get_job_status(job['job_id'])
print(f"Status: {status['status']}")
```

## CLI Commands (api_client.py)

```bash
# Health check
python api_client.py health

# Submit and wait
python api_client.py submit /path/to/video.mp4 --wait

# Check status
python api_client.py status <job_id>

# Get result
python api_client.py result <job_id>

# List jobs
python api_client.py list --status analyzed

# Cancel job
python api_client.py cancel <job_id>
```

## cURL Examples

```bash
# Submit video
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"video_path": "/path/to/video.mp4"}'

# Check job status
curl http://localhost:8000/jobs/<job_id>

# Get result (when complete)
curl http://localhost:8000/jobs/<job_id>/result

# List all jobs
curl http://localhost:8000/jobs

# List analyzed jobs
curl "http://localhost:8000/jobs?status=analyzed"

# Cancel job
curl -X DELETE http://localhost:8000/jobs/<job_id>
```

## Job Status Values

- `analysis-pending` - Job submitted, waiting to be processed
- `analyzing` - Job is currently being analyzed
- `analyzed` - Analysis finished successfully
- `unanalyzed` - Analysis encountered an error
- `analysis-cancelled` - Job was analysis-cancelled by user

## Architecture

The API uses:
- **FastAPI** - Modern async web framework
- **Uvicorn** - ASGI application server
- **SQLite** - Job storage and persistence
- **Threading** - Background job worker
- **Pydantic** - Data validation

Key features:
- Async/non-blocking job submission
- Background analyzing thread
- Persistent job storage
- Automatic worker lifecycle management
- Interactive Swagger documentation at `/docs`

## Configuration Parameters

When submitting jobs, these parameters can be customized:

```python
parameters = {
    "frame_model": "ministral-3:3b-cloud",
    "summary_model": "ministral-3:14b-cloud",
    "whisper_model": "openai/whisper-small",
    "host": "http://localhost:11434",
    "min_frames": 4,
    "max_frames": 16,
    "frames_per_minute": 4.0,
    "frame_selector": "dynamic",
    "dynamic_threshold": 20.0,
    "audio": True,
    "timeout": 120.0,
    "retries": 3,
    "retry_backoff": 1.0,
    "analyze_quality": True
}
```

See [Docs/api.md](../Docs/api.md) for complete documentation.

## Example Workflow

```bash
# 1. Start server
video-analyzer-api &

# 2. Submit video
JOB_ID=$(curl -s -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"video_path": "/path/to/video.mp4"}' | jq -r '.job_id')

echo "Job ID: $JOB_ID"

# 3. Poll status
while true; do
  STATUS=$(curl -s http://localhost:8000/jobs/$JOB_ID | jq -r '.status')
  echo "Status: $STATUS"
  
  if [ "$STATUS" = "analyzed" ]; then
    break
  elif [ "$STATUS" = "unanalyzed" ]; then
    echo "Job unanalyzed!"
    exit 1
  fi
  
  sleep 2
done

# 4. Get result
curl http://localhost:8000/jobs/$JOB_ID/result | jq .
```

## Files Overview

- `video_analyzer/api_server.py` - FastAPI application and worker
- `video_analyzer/job_manager.py` - Job storage and management
- `api_client.py` - Python client library and CLI
- `Docs/api.md` - Full API documentation

## Troubleshooting

**API won't start**
- Check port isn't in use: `lsof -i :8000`
- Ensure dependencies installed: `pip install -e .`
- Verify Ollama running: `curl http://localhost:11434/api/tags`

**Jobs stuck analyzing**
- Check worker running: `curl http://localhost:8000/health`
- Review API server logs
- Verify video file is readable

**Database issues**
- Remove database: `rm .jobs.db`
- Restart API server

**Large video timeouts**
- Increase timeout parameter: `"timeout": 300`
- Reduce max_frames: `"max_frames": 8`
- Use smaller model: `"frame_model": "ministral-3b"`
