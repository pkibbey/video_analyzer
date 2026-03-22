
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

    detailed_summary: str = """You are an expert video and audio analyst and storyteller. Based on the following chronological descriptions 
    of key moments from a {duration:.1f}-second video, along with its audio transcript, create a comprehensive narrative.

    Video Timeline:
    {timeline}

    Audio Transcript:
    {transcript}

    Please provide a detailed summary that:
    1. Tells a cohesive story integrating both visual and audio elements
    2. Captures the progression and flow of events
    3. Highlights significant moments, changes, and patterns
    4. Notes any important dialogue or audio cues
    5. Identifies relationships between what is seen and heard
    6. Maintains a natural, engaging narrative style

    Focus on creating a flowing narrative that combines visual and audio elements."""

    brief_summary: str = """You are an expert video and audio analyst. Based on the following information from a {duration:.1f}-second video:

    Video Timeline:
    {timeline}

    Audio Transcript:
    {transcript}

    Provide a concise 2-3 line summary that captures the essence of both the visual and audio content."""
    
    quality_analysis: str = "Analyze this frame from a video editing perspective. Provide quality scores (1-10) for each metric and note any issues."
    quality_analysis_system: str = "You are a professional video editor evaluating footage for usability. Provide objective quality assessments based on technical criteria."


@dataclass
class FrameAnalysis:
    timestamp: float
    description: str
    scene_type: str
    error: Optional[str] = None
    quality_scores: Optional[Dict[str, Any]] = None
    quality_analysis: Optional[Union[str, Dict[str, Any]]] = None


@dataclass
class SummaryResult:
    detailed: str
    brief: str


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
                "detailed": {"type": "string"},
                "brief": {"type": "string"},
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
                    "quality_scores": {
                        "type": ["object", "null"],
                        "additionalProperties": True
                    },
                    "quality_analysis": {
                        "type": ["object", "string", "null"],
                        "additionalProperties": True
                    },
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
