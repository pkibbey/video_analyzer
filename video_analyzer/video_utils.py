from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any

import ffmpeg
import cv2

logger = logging.getLogger(__name__)


def get_video_properties(video_path: str) -> Dict[str, Any]:
    """Extract comprehensive video properties including fps, dimensions, codec, file dates, audio format, etc."""
    properties = {
        "fps": None,
        "total_frames": None,
        "width": None,
        "height": None,
        "codec": None,
        "duration": None,
        "bitrate": None,
        "format": None,
        "data_rate": None,
        "audio_codec": None,
        "audio_sample_rate": None,
        "file_modified_date": None,
        "file_created_date": None,
    }
    
    # Extract file date information
    try:
        stat = os.stat(video_path)
        # File modification date
        mtime = datetime.fromtimestamp(stat.st_mtime)
        properties["file_modified_date"] = mtime.isoformat()
        # File creation date (ctime may differ across OS)
        ctime = datetime.fromtimestamp(stat.st_ctime)
        properties["file_created_date"] = ctime.isoformat()
    except Exception as exc:
        logger.debug("Failed to extract file dates: %s", exc)
    
    # Try to get properties using ffprobe first (more reliable)
    try:
        info = ffmpeg.probe(video_path)
        
        # Get format info
        format_info = info.get("format", {})
        properties["duration"] = float(format_info.get("duration", 0)) if format_info.get("duration") else None
        properties["bitrate"] = int(format_info.get("bit_rate", 0)) if format_info.get("bit_rate") else None
        properties["format"] = format_info.get("format_name", "").split(",")[0] if format_info.get("format_name") else None
        
        # Format bitrate as human-readable string (e.g., "1.5 Mbps")
        if properties["bitrate"] and properties["bitrate"] > 0:
            bitrate_mbps = properties["bitrate"] / 1_000_000
            properties["data_rate"] = f"{bitrate_mbps:.2f} Mbps"
        
        # Get video and audio stream info
        streams = info.get("streams", [])
        video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
        
        if video_stream:
            properties["codec"] = video_stream.get("codec_name")
            properties["width"] = video_stream.get("width")
            properties["height"] = video_stream.get("height")
            properties["fps"] = eval(video_stream.get("r_frame_rate", "0/1")) if video_stream.get("r_frame_rate") else None
            
            if properties["fps"] and properties["duration"]:
                properties["total_frames"] = int(properties["fps"] * properties["duration"])
            elif video_stream.get("nb_frames"):
                properties["total_frames"] = int(video_stream.get("nb_frames"))
        
        # Extract audio stream information
        if audio_stream:
            properties["audio_codec"] = audio_stream.get("codec_name")
            properties["audio_sample_rate"] = int(audio_stream.get("sample_rate", 0)) if audio_stream.get("sample_rate") else None
    except Exception as exc:
        logger.debug("ffprobe lookup failed: %s", exc)
    
    # Fill in missing properties using cv2
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if properties["fps"] is None and fps > 0:
                properties["fps"] = fps
            if properties["total_frames"] is None and frame_count > 0:
                properties["total_frames"] = frame_count
            if properties["width"] is None and width > 0:
                properties["width"] = width
            if properties["height"] is None and height > 0:
                properties["height"] = height
            
            # Calculate duration from fps and frame count if missing
            if properties["duration"] is None and fps > 0 and frame_count > 0:
                properties["duration"] = frame_count / fps
        finally:
            cap.release()
    
    return properties


def get_video_duration(video_path: str, fallback_duration: float = 0.0) -> float:
    """Return video duration in seconds using ffprobe or cv2 as fallback."""
    duration = _probe_duration_ffmpeg(video_path)
    if duration and duration > 0:
        return duration

    duration = _probe_duration_cv2(video_path)
    if duration and duration > 0:
        return duration

    return fallback_duration


def _probe_duration_ffmpeg(video_path: str) -> Optional[float]:
    try:
        info = ffmpeg.probe(video_path)
        value = float(info["format"]["duration"])
        return value if value > 0 else None
    except Exception as exc:
        logger.debug("ffprobe duration lookup failed: %s", exc)
        return None


def _probe_duration_cv2(video_path: str) -> Optional[float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = 0.0

        if fps and fps > 0 and frame_count > 0:
            duration = frame_count / fps

        if duration <= 0 and frame_count > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count - 1))
            ret, _ = cap.read()
            if ret:
                pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if pos_ms and pos_ms > 0:
                    duration = pos_ms / 1000.0

        return duration if duration > 0 else None
    finally:
        cap.release()
