#!/usr/bin/env python3
"""Example client for VideoAnalyzer API."""

from __future__ import annotations

import json
import time
from typing import Any, Optional

import requests

BASE_URL = "http://localhost:8000"


class APIClient:
    """Client for VideoAnalyzer API."""

    def __init__(self, base_url: str = BASE_URL):
        """Initialize API client.
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip("/")

    def submit_analysis(
        self, video_path: str, parameters: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """Submit a video for analysis.
        
        Args:
            video_path: Path to video file on server
            parameters: Optional analysis parameters
            
        Returns:
            Job information dict with job_id
        """
        payload = {"video_path": video_path}
        if parameters:
            payload["parameters"] = parameters

        response = requests.post(
            f"{self.base_url}/analyze",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Get job status.
        
        Args:
            job_id: Job ID
            
        Returns:
            Job status dict
        """
        response = requests.get(
            f"{self.base_url}/jobs/{job_id}",
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

    def get_job_result(self, job_id: str) -> dict[str, Any]:
        """Get analysis result for completed job.
        
        Args:
            job_id: Job ID
            
        Returns:
            Analysis result dict
            
        Raises:
            requests.HTTPError: If job not completed
        """
        response = requests.get(
            f"{self.base_url}/jobs/{job_id}/result",
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

    def list_jobs(self, status: Optional[str] = None) -> list[dict[str, Any]]:
        """List all jobs, optionally filtered by status.
        
        Args:
            status: Filter by status (pending, processing, completed, failed)
            
        Returns:
            List of job dicts
        """
        params = {}
        if status:
            params["status"] = status

        response = requests.get(
            f"{self.base_url}/jobs",
            params=params,
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        """Cancel or delete a job.
        
        Args:
            job_id: Job ID
            
        Returns:
            Response dict with message
        """
        response = requests.delete(
            f"{self.base_url}/jobs/{job_id}",
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

    def health_check(self) -> dict[str, Any]:
        """Check API health.
        
        Returns:
            Health status dict
        """
        response = requests.get(
            f"{self.base_url}/health",
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

    def wait_for_completion(
        self,
        job_id: str,
        timeout: float = 3600,
        poll_interval: float = 2,
    ) -> dict[str, Any]:
        """Wait for job to complete and return result.
        
        Args:
            job_id: Job ID
            timeout: Maximum time to wait in seconds
            poll_interval: Polling interval in seconds
            
        Returns:
            Analysis result dict
            
        Raises:
            TimeoutError: If job doesn't complete within timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            job = self.get_job_status(job_id)
            status = job["status"]

            if status == "completed":
                return self.get_job_result(job_id)

            if status == "failed":
                raise RuntimeError(f"Job failed: {job.get('error', 'Unknown error')}")

            if status == "cancelled":
                raise RuntimeError("Job was cancelled")

            print(f"Job {job_id} status: {status}... (elapsed: {time.time() - start_time:.0f}s)")
            time.sleep(poll_interval)

        raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")


def main() -> None:
    """Example usage of the API client."""
    import argparse

    parser = argparse.ArgumentParser(description="VideoAnalyzer API client")
    parser.add_argument("--url", default=BASE_URL, help="API base URL")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Health command
    health_parser = subparsers.add_parser("health", help="Check API health")
    
    # Submit command
    submit_parser = subparsers.add_parser("submit", help="Submit video for analysis")
    submit_parser.add_argument("video_path", help="Path to video file on server")
    submit_parser.add_argument("--frame-model", help="Frame analysis model")
    submit_parser.add_argument("--summary-model", help="Summary model")
    submit_parser.add_argument("--wait", action="store_true", help="Wait for completion")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check job status")
    status_parser.add_argument("job_id", help="Job ID")
    
    # Result command
    result_parser = subparsers.add_parser("result", help="Get job result")
    result_parser.add_argument("job_id", help="Job ID")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List jobs")
    list_parser.add_argument(
        "--status", 
        choices=["pending", "processing", "completed", "failed"],
        help="Filter by status"
    )
    
    # Cancel command
    cancel_parser = subparsers.add_parser("cancel", help="Cancel job")
    cancel_parser.add_argument("job_id", help="Job ID")

    args = parser.parse_args()

    client = APIClient(args.url)

    if args.command == "health":
        health = client.health_check()
        print(json.dumps(health, indent=2))

    elif args.command == "submit":
        params = {}
        if args.frame_model:
            params["frame_model"] = args.frame_model
        if args.summary_model:
            params["summary_model"] = args.summary_model

        job = client.submit_analysis(args.video_path, params or None)
        print(f"Submitted job: {job['job_id']}")
        print(json.dumps(job, indent=2))

        if args.wait:
            print("\nWaiting for completion...")
            result = client.wait_for_completion(job["job_id"])
            print("\nAnalysis complete!")
            print(json.dumps(result, indent=2))

    elif args.command == "status":
        job = client.get_job_status(args.job_id)
        print(json.dumps(job, indent=2))

    elif args.command == "result":
        result = client.get_job_result(args.job_id)
        print(json.dumps(result, indent=2))

    elif args.command == "list":
        jobs = client.list_jobs(args.status)
        print(json.dumps(jobs, indent=2))

    elif args.command == "cancel":
        response = client.cancel_job(args.job_id)
        print(json.dumps(response, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
