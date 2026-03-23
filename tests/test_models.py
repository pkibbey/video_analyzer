"""Tests for data models and enums."""

import json
from dataclasses import asdict

import numpy as np
import pytest

from video_analyzer.models import (
    AnalysisMetadata,
    AnalysisPrompts,
    AnalysisResult,
    AudioSegment,
    Frame,
    FrameAnalysis,
    ModelsUsed,
    ProcessingTimings,
    SceneType,
    SummaryResult,
    VideoProperties,
    analysis_result_schema,
)


class TestSceneType:
    def test_scene_type_values(self):
        """Test SceneType enum has expected values."""
        assert SceneType.STATIC.value == "static"
        assert SceneType.ACTION.value == "action"
        assert SceneType.TRANSITION.value == "transition"

    def test_scene_type_all_members(self):
        """Test all SceneType members are defined."""
        members = [st for st in SceneType]
        assert len(members) == 3
        assert SceneType.STATIC in members
        assert SceneType.ACTION in members
        assert SceneType.TRANSITION in members


class TestFrame:
    def test_frame_creation(self):
        """Test creating a Frame with required and optional fields."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(
            image=image,
            timestamp=1.5,
            scene_type=SceneType.ACTION,
            difference_score=25.5,
        )

        assert frame.timestamp == 1.5
        assert frame.scene_type == SceneType.ACTION
        assert frame.difference_score == 25.5
        np.testing.assert_array_equal(frame.image, image)

    def test_frame_default_difference_score(self):
        """Test Frame has default difference_score of 0.0."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = Frame(
            image=image, timestamp=1.0, scene_type=SceneType.STATIC
        )

        assert frame.difference_score == 0.0

    def test_frame_image_types(self):
        """Test Frame can handle different image types."""
        # Test with uint8
        image_uint8 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame1 = Frame(image=image_uint8, timestamp=1.0, scene_type=SceneType.ACTION)
        assert frame1.image.dtype == np.uint8

        # Test with float32
        image_float32 = np.zeros((480, 640, 3), dtype=np.float32)
        frame2 = Frame(image=image_float32, timestamp=1.0, scene_type=SceneType.ACTION)
        assert frame2.image.dtype == np.float32


class TestAudioSegment:
    def test_audio_segment_creation(self):
        """Test creating an AudioSegment."""
        segment = AudioSegment(
            text="Hello world",
            start_time=0.5,
            end_time=2.5,
            confidence=0.95,
        )

        assert segment.text == "Hello world"
        assert segment.start_time == 0.5
        assert segment.end_time == 2.5
        assert segment.confidence == 0.95

    def test_audio_segment_none_text(self):
        """Test AudioSegment can have None text."""
        segment = AudioSegment(
            text=None,
            start_time=0.5,
            end_time=2.5,
        )

        assert segment.text is None
        assert segment.confidence == 0.0

    def test_audio_segment_default_confidence(self):
        """Test AudioSegment has default confidence of 0.0."""
        segment = AudioSegment(
            text="Test",
            start_time=0.0,
            end_time=1.0,
        )

        assert segment.confidence == 0.0

    def test_audio_segment_timing_validation(self):
        """Test AudioSegment with various timing values."""
        # End time greater than start time
        segment1 = AudioSegment(
            text="Test", start_time=1.0, end_time=2.0, confidence=0.9
        )
        assert segment1.end_time > segment1.start_time

        # Same start and end time
        segment2 = AudioSegment(
            text="Test", start_time=1.0, end_time=1.0, confidence=0.9
        )
        assert segment2.end_time == segment2.start_time


class TestAnalysisPrompts:
    def test_analysis_prompts_defaults(self):
        """Test AnalysisPrompts has all default values."""
        prompts = AnalysisPrompts()

        assert prompts.frame_analysis is not None
        assert len(prompts.frame_analysis) > 0
        assert prompts.frame_analysis_system is not None
        assert prompts.detailed_summary is not None
        assert prompts.brief_summary is not None
        assert prompts.quality_analysis is not None
        assert prompts.quality_analysis_system is not None

    def test_analysis_prompts_custom_values(self):
        """Test AnalysisPrompts with custom values."""
        custom_frame = "Analyze the frame"
        custom_summary = "Summarize the video"

        prompts = AnalysisPrompts(
            frame_analysis=custom_frame,
            detailed_summary=custom_summary,
        )

        assert prompts.frame_analysis == custom_frame
        assert prompts.detailed_summary == custom_summary
        # Other fields should still have defaults
        assert prompts.frame_analysis_system is not None

    def test_analysis_prompts_contain_format_placeholders(self):
        """Test summary prompts contain expected placeholders."""
        prompts = AnalysisPrompts()

        assert "{duration:" in prompts.detailed_summary
        assert "{timeline}" in prompts.detailed_summary
        assert "{transcript}" in prompts.detailed_summary

        assert "{duration:" in prompts.brief_summary
        assert "{timeline}" in prompts.brief_summary
        assert "{transcript}" in prompts.brief_summary


class TestFrameAnalysis:
    def test_frame_analysis_required_fields(self):
        """Test FrameAnalysis with only required fields."""
        analysis = FrameAnalysis(
            timestamp=5.0,
            description="A person is walking",
            scene_type="action",
        )

        assert analysis.timestamp == 5.0
        assert analysis.description == "A person is walking"
        assert analysis.scene_type == "action"
        assert analysis.error is None
        assert analysis.quality_scores is None
        assert analysis.quality_analysis is None

    def test_frame_analysis_with_optional_fields(self):
        """Test FrameAnalysis with all optional fields."""
        quality_scores = "Focus:8, Exposure:7, Composition:9"
        quality_text = "Frame is well-lit and composed"

        analysis = FrameAnalysis(
            timestamp=5.0,
            description="A person is walking",
            scene_type="action",
            error=None,
            quality_scores=quality_scores,
            quality_analysis=quality_text,
        )

        assert analysis.quality_scores == quality_scores
        assert analysis.quality_analysis == quality_text

    def test_frame_analysis_quality_analysis_string_only(self):
        """Ensure quality_analysis only accepts string-like output for strict typing."""
        quality_text = "Overall quality is 7.5/10"

        analysis = FrameAnalysis(
            timestamp=5.0,
            description="Test",
            scene_type="action",
            quality_analysis=quality_text,
        )

        assert isinstance(analysis.quality_analysis, str)
        assert analysis.quality_analysis == quality_text


class TestSummaryResult:
    def test_summary_result_creation(self):
        """Test creating a SummaryResult."""
        result = SummaryResult(
            detailed="This is a detailed summary of the video content.",
            brief="Brief summary here.",
        )

        assert result.detailed == "This is a detailed summary of the video content."
        assert result.brief == "Brief summary here."

    def test_summary_result_empty_strings(self):
        """Test SummaryResult with empty strings."""
        result = SummaryResult(detailed="", brief="")

        assert result.detailed == ""
        assert result.brief == ""


class TestModelsUsed:
    def test_models_used_creation(self):
        """Test creating a ModelsUsed record."""
        models = ModelsUsed(
            frame_analysis="ministral-3:3b",
            summary="ministral-3:14b",
            audio="openai/whisper-small",
        )

        assert models.frame_analysis == "ministral-3:3b"
        assert models.summary == "ministral-3:14b"
        assert models.audio == "openai/whisper-small"

    def test_models_used_none_audio(self):
        """Test ModelsUsed with None audio model."""
        models = ModelsUsed(
            frame_analysis="ministral-3:3b",
            summary="ministral-3:14b",
            audio=None,
        )

        assert models.audio is None


class TestProcessingTimings:
    def test_processing_timings_creation(self):
        """Test creating ProcessingTimings."""
        timings = ProcessingTimings(
            frame_selection=1.5,
            audio_transcription=2.0,
            frame_analysis=5.5,
            summary_generation=3.0,
            total=12.0,
        )

        assert timings.frame_selection == 1.5
        assert timings.audio_transcription == 2.0
        assert timings.frame_analysis == 5.5
        assert timings.summary_generation == 3.0
        assert timings.total == 12.0

    def test_processing_timings_defaults(self):
        """Test ProcessingTimings has correct defaults."""
        timings = ProcessingTimings(frame_selection=1.0)

        assert timings.frame_selection == 1.0
        assert timings.audio_transcription == 0.0
        assert timings.frame_analysis == 0.0
        assert timings.summary_generation == 0.0
        assert timings.total == 0.0


class TestVideoProperties:
    def test_video_properties_all_fields(self):
        """Test VideoProperties with all fields set."""
        props = VideoProperties(
            fps=29.97,
            total_frames=1800,
            width=1920,
            height=1080,
            codec="h264",
            bitrate=5000000,
            format="mp4",
            duration=60.0,
            data_rate="5.0 Mbps",
            audio_codec="aac",
            audio_sample_rate=48000,
            file_modified_date="2024-01-01T12:00:00",
            file_created_date="2024-01-01T10:00:00",
        )

        assert props.fps == 29.97
        assert props.total_frames == 1800
        assert props.width == 1920
        assert props.height == 1080
        assert props.codec == "h264"

    def test_video_properties_partial_fields(self):
        """Test VideoProperties with only some fields set."""
        props = VideoProperties(
            fps=30.0,
            width=1920,
            height=1080,
        )

        assert props.fps == 30.0
        assert props.width == 1920
        assert props.total_frames is None
        assert props.codec is None


class TestAnalysisMetadata:
    def test_analysis_metadata_creation(self):
        """Test creating AnalysisMetadata."""
        models = ModelsUsed(
            frame_analysis="model1",
            summary="model2",
            audio="model3",
        )
        timings = ProcessingTimings(frame_selection=1.0, total=5.0)

        metadata = AnalysisMetadata(
            num_frames_analyzed=10,
            num_audio_segments=5,
            video_duration=60.0,
            scene_distribution={"static": 5, "action": 4, "transition": 1},
            models_used=models,
            processing_timings=timings,
        )

        assert metadata.num_frames_analyzed == 10
        assert metadata.num_audio_segments == 5
        assert metadata.video_duration == 60.0
        assert metadata.scene_distribution["static"] == 5
        assert metadata.models_used.frame_analysis == "model1"
        assert metadata.processing_timings.total == 5.0

    def test_analysis_metadata_without_optional_fields(self):
        """Test AnalysisMetadata without optional fields."""
        models = ModelsUsed(
            frame_analysis="model1",
            summary="model2",
            audio=None,
        )

        metadata = AnalysisMetadata(
            num_frames_analyzed=10,
            num_audio_segments=0,
            video_duration=30.0,
            scene_distribution={"static": 10},
            models_used=models,
        )

        assert metadata.processing_timings is None
        assert metadata.video_properties is None


class TestAnalysisResult:
    def test_analysis_result_creation(self):
        """Test creating a complete AnalysisResult."""
        summary = SummaryResult(
            detailed="Detailed summary",
            brief="Brief summary",
        )
        frame_analyses = [
            FrameAnalysis(
                timestamp=0.0,
                description="Frame description",
                scene_type="action",
            ),
        ]
        audio_segments = [
            AudioSegment(
                text="Hello",
                start_time=0.0,
                end_time=1.0,
                confidence=0.95,
            ),
        ]
        models = ModelsUsed(
            frame_analysis="model1",
            summary="model2",
            audio="model3",
        )
        metadata = AnalysisMetadata(
            num_frames_analyzed=1,
            num_audio_segments=1,
            video_duration=10.0,
            scene_distribution={"action": 1},
            models_used=models,
        )

        result = AnalysisResult(
            summary=summary,
            frame_analyses=frame_analyses,
            audio_segments=audio_segments,
            metadata=metadata,
        )

        assert result.summary.detailed == "Detailed summary"
        assert len(result.frame_analyses) == 1
        assert len(result.audio_segments) == 1
        assert result.metadata.num_frames_analyzed == 1

    def test_analysis_result_to_dict(self):
        """Test AnalysisResult.to_dict() conversion."""
        summary = SummaryResult(detailed="Detail", brief="Brief")
        models = ModelsUsed(
            frame_analysis="m1",
            summary="m2",
            audio="m3",
        )
        metadata = AnalysisMetadata(
            num_frames_analyzed=1,
            num_audio_segments=0,
            video_duration=10.0,
            scene_distribution={},
            models_used=models,
        )

        result = AnalysisResult(
            summary=summary,
            frame_analyses=[],
            audio_segments=[],
            metadata=metadata,
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "summary" in result_dict
        assert "frame_analyses" in result_dict
        assert "audio_segments" in result_dict
        assert "metadata" in result_dict

    def test_analysis_result_to_legacy_dict(self):
        """Test AnalysisResult.to_legacy_dict() maintains backward compatibility."""
        summary = SummaryResult(
            detailed="Detailed summary",
            brief="Brief summary",
        )
        frame_analyses = [
            FrameAnalysis(
                timestamp=1.0,
                description="Test frame",
                scene_type="action",
                quality_scores={"brightness": 8},
            ),
        ]
        models = ModelsUsed(
            frame_analysis="m1",
            summary="m2",
            audio="m3",
        )
        metadata = AnalysisMetadata(
            num_frames_analyzed=1,
            num_audio_segments=0,
            video_duration=10.0,
            scene_distribution={"action": 1},
            models_used=models,
        )

        result = AnalysisResult(
            summary=summary,
            frame_analyses=frame_analyses,
            audio_segments=[],
            metadata=metadata,
        )

        legacy_dict = result.to_legacy_dict()

        # Check legacy format has expected keys
        assert legacy_dict["summary"] == "Detailed summary"
        assert legacy_dict["brief_summary"] == "Brief summary"
        assert "frame_analyses" in legacy_dict
        assert "audio_segments" in legacy_dict
        assert "metadata" in legacy_dict

    def test_analysis_result_with_warnings(self):
        """Test AnalysisResult with warnings."""
        summary = SummaryResult(detailed="", brief="")
        models = ModelsUsed(frame_analysis="m1", summary="m2", audio=None)
        metadata = AnalysisMetadata(
            num_frames_analyzed=1,
            num_audio_segments=0,
            video_duration=10.0,
            scene_distribution={},
            models_used=models,
        )

        result = AnalysisResult(
            summary=summary,
            frame_analyses=[],
            audio_segments=[],
            metadata=metadata,
            warnings=["Low audio quality", "Few frames selected"],
        )

        assert len(result.warnings) == 2
        assert "Low audio quality" in result.warnings

    def test_analysis_result_json_serializable(self):
        """Test AnalysisResult can be serialized to JSON."""
        summary = SummaryResult(detailed="Detail", brief="Brief")
        models = ModelsUsed(frame_analysis="m1", summary="m2", audio=None)
        metadata = AnalysisMetadata(
            num_frames_analyzed=1,
            num_audio_segments=0,
            video_duration=10.0,
            scene_distribution={},
            models_used=models,
        )

        result = AnalysisResult(
            summary=summary,
            frame_analyses=[],
            audio_segments=[],
            metadata=metadata,
        )

        result_dict = result.to_dict()
        json_str = json.dumps(result_dict)
        parsed = json.loads(json_str)

        assert parsed["summary"]["detailed"] == "Detail"
        assert parsed["metadata"]["num_frames_analyzed"] == 1


class TestAnalysisSchema:
    def test_analysis_result_schema_exists(self):
        """Test that analysis result schema is defined."""
        schema = analysis_result_schema()

        assert isinstance(schema, dict)
        assert "$schema" in schema
        assert "title" in schema
        assert "properties" in schema

    def test_analysis_result_schema_has_required_properties(self):
        """Test schema includes all required properties."""
        schema = analysis_result_schema()
        required = schema.get("required", [])

        assert "summary" in required
        assert "frame_analyses" in required
        assert "audio_segments" in required
        assert "metadata" in required
        assert "warnings" in required

    def test_analysis_result_schema_is_independent_copy(self):
        """Test that analysis_result_schema() returns independent copies."""
        schema1 = analysis_result_schema()
        schema2 = analysis_result_schema()

        # Modify one schema
        schema1["title"] = "Modified"

        # Verify the other is unchanged
        assert schema2["title"] == "AnalysisResult"

    def test_analysis_result_schema_structure(self):
        """Test schema structure is valid JSON schema."""
        schema = analysis_result_schema()

        # Check basic structure
        assert schema["type"] == "object"
        assert isinstance(schema["properties"], dict)
        assert isinstance(schema["required"], list)

        # Check summary property
        assert "summary" in schema["properties"]
        summary_schema = schema["properties"]["summary"]
        assert summary_schema["type"] == "object"
        assert "detailed" in summary_schema["required"]
        assert "brief" in summary_schema["required"]

        # Check frame_analyses property
        assert "frame_analyses" in schema["properties"]
        assert schema["properties"]["frame_analyses"]["type"] == "array"
