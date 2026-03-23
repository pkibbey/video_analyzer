
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np
from typing import List, Dict, Optional, Any, Union
import copy
class SceneType(Enum):
    STATIC = "static"
    ACTION = "action"
    TRANSITION = "transition"


@dataclass
class Frame:
    image: np.ndarray
    timestamp: float
    scene_type: SceneType
    difference_score: float = 0.0


@dataclass
class AudioSegment:
    """Represents a transcribed segment of audio with timing information"""
    text: Optional[str]
    start_time: float
    end_time: float
    confidence: float = 0.0

@dataclass
class AnalysisPrompts:
    """Customizable prompts for video and audio analysis"""
    frame_analysis: str = "Describe what's happening in this moment of the video, focusing on important actions, objects, or changes."
    frame_analysis_system: str = (
        "You are a visual analyst. Describe only what is visible in the frame. "
        "Do not mention prior responses, do not apologize, and do not speculate. "
        "If context is provided, use it only for continuity without referencing it directly."
    )

    detailed_summary: str = """You are an expert video and audio analyst. Based on the following chronological descriptions 
    of key moments from a {duration:.1f}-second video, along with its audio transcript, provide a factual, explicit summary.

    Video Timeline:
    {timeline}

    Audio Transcript:
    {transcript}

    Please provide a detailed summary that:
    1. Uses only information present in the timeline and transcript (no invention)
    2. Emphasizes concrete events, speakers, timing, and actions
    3. Avoids creative storytelling, metaphors, or speculation
    4. Notes key changes, repeated themes, and important audio cues
    5. Keeps sentences direct and to the point

    Focus on factual accuracy and clear structure."""

    brief_summary: str = """You are an expert video and audio analyst. Based on the following information from a {duration:.1f}-second video:

    Video Timeline:
    {timeline}

    Audio Transcript:
    {transcript}

    Provide a concise 2-3 line summary with:
    - only factual observable points
    - no inventing details or presuming motives
    - no narrative embellishment
    - a clear statement of what happened and what was heard."""
    
    quality_analysis: str = "Analyze this frame from a video editing perspective. Provide quality scores (1-10) for each metric and note any issues."
    quality_analysis_system: str = "You are a professional video editor evaluating footage for usability. Provide objective quality assessments based on technical criteria."


@dataclass
class FrameAnalysis:
    timestamp: float
    description: str
    scene_type: str
    error: Optional[str] = None
    quality_scores: Optional[str] = None
    quality_analysis: Optional[str] = None


@dataclass
class SummaryResult:
    detailed: str
    brief: str
    timeline: Optional[str] = None
    transcript: Optional[str] = None


@dataclass
class ModelsUsed:
    frame_analysis: str
    summary: str
    audio: Optional[str]


@dataclass
class ProcessingTimings:
    """Timing information for different stages of video analysis"""
    frame_selection: float  # Time to select frames
    audio_transcription: float = 0.0  # Time to transcribe audio
    frame_analysis: float = 0.0  # Time to analyze all frames
    summary_generation: float = 0.0  # Time to generate summaries
    total: float = 0.0  # Total analysis time


@dataclass
class VideoProperties:
    """Technical properties of the video file"""
    fps: Optional[float] = None  # Frames per second
    total_frames: Optional[int] = None  # Total number of frames
    width: Optional[int] = None  # Video width in pixels
    height: Optional[int] = None  # Video height in pixels
    codec: Optional[str] = None  # Video codec (e.g., h264, hevc)
    bitrate: Optional[int] = None  # Bitrate in bits per second
    format: Optional[str] = None  # Container format (e.g., mp4, mov)
    duration: Optional[float] = None  # Video duration in seconds
    data_rate: Optional[str] = None  # Human-readable bitrate (e.g., "1.5 Mbps")
    audio_codec: Optional[str] = None  # Audio codec (e.g., aac, mp3)
    audio_sample_rate: Optional[int] = None  # Audio sample rate in Hz (e.g., 48000)
    file_size: Optional[int] = None  # File size in bytes
    file_modified_date: Optional[str] = None  # File modification date (ISO format)
    file_created_date: Optional[str] = None  # File creation date (ISO format)


@dataclass
class AnalysisMetadata:
    num_frames_analyzed: int
    num_audio_segments: int
    video_duration: float
    scene_distribution: Dict[str, int]
    models_used: ModelsUsed
    processing_timings: Optional['ProcessingTimings'] = None
    video_properties: Optional['VideoProperties'] = None


@dataclass
class AnalysisResult:
    summary: SummaryResult
    frame_analyses: List[FrameAnalysis]
    audio_segments: List[AudioSegment]
    metadata: AnalysisMetadata
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_legacy_dict(self) -> Dict[str, Any]:
        frame_analyses = []
        for analysis in self.frame_analyses:
            item = {
                "timestamp": analysis.timestamp,
                "description": analysis.description,
                "scene_type": analysis.scene_type,
            }
            if analysis.error:
                item["error"] = analysis.error
            if analysis.quality_scores is not None:
                item["quality_scores"] = analysis.quality_scores
            if analysis.quality_analysis is not None:
                item["quality_analysis"] = analysis.quality_analysis
            frame_analyses.append(item)

        return {
            "summary": self.summary.detailed,
            "brief_summary": self.summary.brief,
            "frame_analyses": frame_analyses,
            "audio_segments": [asdict(segment) for segment in self.audio_segments],
            "metadata": asdict(self.metadata),
            "warnings": list(self.warnings),
        }

    @staticmethod
    def schema() -> Dict[str, Any]:
        return analysis_result_schema()


def analysis_result_schema() -> Dict[str, Any]:
    return copy.deepcopy(ANALYSIS_RESULT_SCHEMA)


ANALYSIS_RESULT_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "AnalysisResult",
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "frame_analyses", "audio_segments", "metadata", "warnings"],
    "properties": {
        "summary": {
            "type": "object",
            "additionalProperties": False,
            "required": ["detailed", "brief"],
            "properties": {
                "detailed": {"type": ["string", "object"]},
                "brief": {"type": "string"},
                "timeline": {"type": ["string", "null"]},
                "transcript": {"type": ["string", "null"]},
            },
        },
        "frame_analyses": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["timestamp", "description", "scene_type"],
                "properties": {
                    "timestamp": {"type": "number"},
                    "description": {"type": "string"},
                    "scene_type": {"type": "string"},
                    "error": {"type": ["string", "null"]},
                    "quality_scores": {"type": ["string", "null"]},
                    "quality_analysis": {"type": ["string", "null"]},
                },
            },
        },
        "audio_segments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "start_time", "end_time", "confidence"],
                "properties": {
                    "text": {"type": ["string", "null"]},
                    "start_time": {"type": "number"},
                    "end_time": {"type": "number"},
                    "confidence": {"type": "number"},
                },
            },
        },
        "metadata": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "num_frames_analyzed",
                "num_audio_segments",
                "video_duration",
                "scene_distribution",
                "models_used",
            ],
            "properties": {
                "num_frames_analyzed": {"type": "integer"},
                "num_audio_segments": {"type": "integer"},
                "video_duration": {"type": "number"},
                "scene_distribution": {
                    "type": "object",
                    "additionalProperties": {"type": "integer"},
                },
                "models_used": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["frame_analysis", "summary", "audio"],
                    "properties": {
                        "frame_analysis": {"type": "string"},
                        "summary": {"type": "string"},
                        "audio": {"type": ["string", "null"]},
                    },
                },
                "processing_timings": {
                    "type": ["object", "null"],
                    "additionalProperties": False,
                    "required": ["frame_selection", "audio_transcription", "frame_analysis", "summary_generation", "total"],
                    "properties": {
                        "frame_selection": {"type": "number"},
                        "audio_transcription": {"type": "number"},
                        "frame_analysis": {"type": "number"},
                        "summary_generation": {"type": "number"},
                        "total": {"type": "number"},
                    },
                },
                "video_properties": {
                    "type": ["object", "null"],
                    "additionalProperties": False,
                    "properties": {
                        "fps": {"type": ["number", "null"]},
                        "total_frames": {"type": ["integer", "null"]},
                        "width": {"type": ["integer", "null"]},
                        "height": {"type": ["integer", "null"]},
                        "codec": {"type": ["string", "null"]},
                        "bitrate": {"type": ["integer", "null"]},
                        "format": {"type": ["string", "null"]},
                        "duration": {"type": ["number", "null"]},
                        "data_rate": {"type": ["string", "null"]},
                        "audio_codec": {"type": ["string", "null"]},
                        "audio_sample_rate": {"type": ["integer", "null"]},
                        "file_size": {"type": ["integer", "null"]},
                        "file_modified_date": {"type": ["string", "null"]},
                        "file_created_date": {"type": ["string", "null"]},
                    },
                },
            },
        },
        "warnings": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}
