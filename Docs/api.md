# VideoAnalyzer API Documentation

The VideoAnalyzer service includes a REST API that allows you to submit video analysis jobs to a queue and retrieve results asynchronously.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Running the API Server](#running-the-api-server)
3. [API Endpoints](#api-endpoints)
4. [Request/Response Examples](#requestresponse-examples)
5. [Python Client](#python-client)
6. [cURL Examples](#curl-examples)
7. [Error Handling](#error-handling)

## Quick Start

### 1. Install dependencies

```bash
pip install -e .
```

### 2. Start the API server

```bash
openscenesense-ollama-api --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

### 3. Submit a video

```bash
python api_client.py submit /path/to/video.mp4 --wait
```

## Running the API Server

### Basic Usage

```bash
video-analyzer-api
```

Starts the API on `0.0.0.0:8000` by default.

### Options

```bash
video-analyzer-api --help

Options:
  --host TEXT        Server host (default: 0.0.0.0)
  --port INTEGER     Server port (default: 8000)
  --reload           Enable auto-reload on code changes
  --workers INTEGER  Number of worker processes (default: 1)
  --db TEXT         Job database path (default: .jobs.db)
```

### Examples

**Development with auto-reload:**
```bash
video-analyzer-api --reload
```

**Production with multiple workers:**
```bash
video-analyzer-api --host 0.0.0.0 --port 8000 --workers 4
```

**Custom database location:**
```bash
video-analyzer-api --db /var/lib/jobs.db
```

## API Endpoints

### Health Check

**GET /health**

Check if the API is running and the worker is active.

```json
{
  "status": "healthy",
  "worker_running": true,
  "jobs_db_exists": true
}
```

### Submit Job

**POST /analyze**

Submit a video for analysis.

**Request:**
```json
{
  "video_path": "/path/to/video.mp4",
  "parameters": {
    "frame_model": "ministral-3:3b-cloud",
    "summary_model": "ministral-3:14b-cloud",
    "min_frames": 4,
    "max_frames": 16,
    "audio": true
  }
}
```

**Response (202 Accepted):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "video_path": "/path/to/video.mp4",
  "status": "pending",
  "created_at": "2024-03-21T10:30:00"
}
```

### Get Job Status

**GET /jobs/{job_id}**

Get the current status of a job.

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "video_path": "/path/to/video.mp4",
  "status": "processing",
  "created_at": "2024-03-21T10:30:00",
  "started_at": "2024-03-21T10:31:00"
}
```

### Get Job Result

**GET /jobs/{job_id}/result**

Get analysis results for a completed job.

**Response (200 OK if completed):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "result": {
    "summary": {
      "brief": "A person walking in the park",
      "detailed": {...},
      "timeline": "00:00 - Person appears, walking..."
    },
    "metadata": {
      "video_duration": 45.3,
      "num_frames_analyzed": 12,
      "models_used": {...}
    }
  }
}
```

**Response (202 if still processing):**
```json
{
  "detail": "Job is still processing"
}
```

### List Jobs

**GET /jobs?status=pending**

List all jobs, optionally filtered by status.

**Query Parameters:**
- `status` (optional): Filter by status - `pending`, `processing`, `completed`, or `failed`

**Response:**
```json
[
  {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "video_path": "/path/to/video.mp4",
    "status": "completed",
    "created_at": "2024-03-21T10:30:00"
  }
]
```

### Cancel/Delete Job

**DELETE /jobs/{job_id}**

Cancel a pending/processing job or delete a completed/failed job.

**Response:**
```json
{
  "message": "Job 550e8400-e29b-41d4-a716-446655440000 cancelled"
}
```

## Request/Response Examples

### Complete Example: Submit and Wait

1. **Submit video:**
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/videos/sample.mp4",
    "parameters": {
      "min_frames": 4,
      "max_frames": 16
    }
  }'
```

Response:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "created_at": "2024-03-21T10:30:00"
}
```

2. **Poll job status:**
```bash
curl http://localhost:8000/jobs/550e8400-e29b-41d4-a716-446655440000
```

3. **Get result when complete:**
```bash
curl http://localhost:8000/jobs/550e8400-e29b-41d4-a716-446655440000/result
```

## Python Client

A Python client is provided in `api_client.py` for easy interaction.

### Installation

```bash
pip install requests
```

### Usage

```python
from api_client import APIClient

client = APIClient("http://localhost:8000")

# Submit a job
job = client.submit_analysis("/path/to/video.mp4")
print(f"Job ID: {job['job_id']}")

# Wait for completion
result = client.wait_for_completion(job['job_id'])
print(f"Analysis complete: {result['result']}")
```

### CLI Commands

```bash
# Check health
python api_client.py health

# Submit and wait
python api_client.py submit /path/to/video.mp4 --wait

# Check job status
python api_client.py status <job_id>

# Get result
python api_client.py result <job_id>

# List jobs
python api_client.py list --status completed

# Cancel job
python api_client.py cancel <job_id>
```

## cURL Examples

### Check health
```bash
curl http://localhost:8000/health
```

### Submit video
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "video_path": "/path/to/video.mp4",
    "parameters": {
      "frame_model": "ministral-3:3b-cloud",
      "summary_model": "ministral-3:14b-cloud",
      "min_frames": 4,
      "max_frames": 16,
      "audio": true
    }
  }'
```

### Get job status
```bash
curl http://localhost:8000/jobs/<job_id>
```

### Get job result
```bash
curl http://localhost:8000/jobs/<job_id>/result
```

### List all jobs
```bash
curl http://localhost:8000/jobs
```

### List completed jobs
```bash
curl "http://localhost:8000/jobs?status=completed"
```

### Cancel job
```bash
curl -X DELETE http://localhost:8000/jobs/<job_id>
```

## Error Handling

### Status Codes

- **200 OK**: Request successful
- **202 Accepted**: Job submitted or still processing
- **400 Bad Request**: Invalid request (e.g., video file not found)
- **404 Not Found**: Job or resource not found
- **500 Internal Server Error**: Job failed or server error

### Error Responses

**Video not found:**
```json
{
  "detail": "Video file not found: /nonexistent/video.mp4"
}
```

**Job not found:**
```json
{
  "detail": "Job 550e8400-e29b-41d4-a716-446655440000 not found"
}
```

**Job still processing:**
```json
{
  "detail": "Job is still processing"
}
```

**Job failed:**
```json
{
  "detail": "Job failed: Connection refused to Ollama server"
}
```

## Analysis Parameters

All parameters from the CLI are available in the API:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `frame_model` | string | `ministral-3:3b-cloud` | Model for frame analysis |
| `summary_model` | string | `ministral-3:14b-cloud` | Model for summary generation |
| `whisper_model` | string | `openai/whisper-small` | Whisper model for audio |
| `host` | string | `http://localhost:11434` | Ollama server URL |
| `min_frames` | integer | `4` | Minimum frames to analyze |
| `max_frames` | integer | `16` | Maximum frames to analyze |
| `frames_per_minute` | float | `4.0` | Frame sampling rate |
| `frame_selector` | string | `dynamic` | Frame selector method (`dynamic`, `uniform`, `all`) |
| `dynamic_threshold` | float | `20.0` | Threshold for dynamic frame selection |
| `audio` | boolean | `true` | Enable audio transcription |
| `device` | string | `null` | Device for Whisper (cuda, cpu, etc.) |
| `timeout` | float | `120.0` | Request timeout in seconds |
| `retries` | integer | `3` | Number of retries |
| `retry_backoff` | float | `1.0` | Backoff multiplier |
| `analyze_quality` | boolean | `true` | Analyze frame quality metrics |

## Image Processing Architecture

The API uses a background worker thread to process jobs asynchronously:

```
Client
  ↓
POST /analyze (submit job)
  ↓
SQLite DB (job stored as PENDING)
  ↓
Worker Thread (polls DB)
  ↓
OllamaVideoAnalyzer (processes video)
  ↓
SQLite DB (job status = COMPLETED, result stored)
  ↓
Client polls GET /jobs/{job_id}/result
  ↓
Returns analysis result
```

## Scaling Considerations

- **Single Worker**: Default configuration with one worker thread processing jobs sequentially
- **Multiple Processes**: Use `--workers N` to run multiple API server processes
- **Database**: SQLite is suitable for development; consider PostgreSQL for production
- **Storage**: Job results are stored in the database; consider implementing result cleanup for long-running servers

## Common Usage Patterns

### Pattern 1: Submit and Poll

```python
client = APIClient()
job = client.submit_analysis("/path/to/video.mp4")

# Poll until ready
for i in range(600):  # 10 minutes
    try:
        result = client.get_job_result(job['job_id'])
        print("Analysis complete!")
        break
    except Exception as e:
        if "still processing" in str(e):
            time.sleep(1)
        else:
            raise
```

### Pattern 2: Wait for Result

```python
client = APIClient()
job = client.submit_analysis("/path/to/video.mp4")
result = client.wait_for_completion(job['job_id'], timeout=3600)
print(result)
```

### Pattern 3: Batch Processing

```python
client = APIClient()
videos = ["/video1.mp4", "/video2.mp4", "/video3.mp4"]

# Submit all jobs
jobs = [client.submit_analysis(v) for v in videos]

# Wait for all to complete
results = {}
for job in jobs:
    results[job['job_id']] = client.wait_for_completion(job['job_id'])
```

## Troubleshooting

### API won't start
- Ensure port 8000 is not in use
- Check that dependencies are installed: `pip install -e .`
- Verify Ollama server is running at the configured host

### Jobs stuck in processing
- Check worker thread is running: `GET /health`
- Review logs for errors
- Verify video files are accessible to the worker process

### Database corruption
- Stop the API server
- Remove `.jobs.db` file
- Restart the API server (new database will be created)

### Memory issues with large videos
- Reduce `max_frames` parameter
- Reduce `frame_model` size
- Process videos one at a time instead of in parallel
