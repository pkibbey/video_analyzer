"""Extended tests for video analyzer."""

import io
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests

from video_analyzer.analyzer import OllamaVideoAnalyzer
from video_analyzer.models import (
    AnalysisPrompts,
    AudioSegment,
    Frame,
    SceneType,
)


@pytest.fixture
def analyzer():
    """Create a test analyzer instance."""
    return OllamaVideoAnalyzer(
        frame_analysis_model="test-model",
        summary_model="test-summary-model",
        host="http://localhost:11434",
        min_frames=4,
        max_frames=16,
        frames_per_minute=4.0,
        request_retries=1,
        request_backoff=0.0,
    )


@pytest.fixture
def sample_frame():
    """Create a sample frame for testing."""
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    return Frame(
        image=image,
        timestamp=1.5,
        scene_type=SceneType.ACTION,
        difference_score=10.0,
    )


@pytest.fixture
def sample_audio_segments():
    """Create sample audio segments."""
    return [
        AudioSegment(
            text="Hello world",
            start_time=0.0,
            end_time=2.0,
            confidence=0.95,
        ),
        AudioSegment(
            text="This is a test",
            start_time=2.0,
            end_time=4.5,
            confidence=0.92,
        ),
    ]


class TestFrameToBase64:
    def test_frame_to_base64_conversion(self, analyzer, sample_frame):
        """Test converting frame to base64 string."""
        base64_str = analyzer._frame_to_base64(sample_frame.image)

        assert isinstance(base64_str, str)
        assert len(base64_str) > 0
        # Base64 string should only contain valid characters
        assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=" for c in base64_str)

    def test_frame_to_base64_different_sizes(self, analyzer):
        """Test base64 conversion with different frame sizes."""
        # Small frame
        small_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        base64_small = analyzer._frame_to_base64(small_frame)

        # Large frame
        large_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        base64_large = analyzer._frame_to_base64(large_frame)

        # Different sizes should produce different base64 strings
        assert base64_small != base64_large
        assert len(base64_large) > len(base64_small)

    def test_frame_to_base64_consistency(self, analyzer):
        """Test base64 conversion is consistent."""
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128

        base64_1 = analyzer._frame_to_base64(frame.copy())
        base64_2 = analyzer._frame_to_base64(frame.copy())

        # Same frame should produce same base64 (approximately)
        assert base64_1 == base64_2


class TestFormatTranscript:
    def test_format_transcript_with_segments(self, analyzer, sample_audio_segments):
        """Test formatting audio transcript."""
        formatted = analyzer._format_transcript(sample_audio_segments)

        assert "Hello world" in formatted
        assert "This is a test" in formatted
        assert "[0.0s - 2.0s]" in formatted
        assert "[2.0s - 4.5s]" in formatted

    def test_format_transcript_empty_segments(self, analyzer):
        """Test formatting empty audio segments."""
        formatted = analyzer._format_transcript([])

        assert formatted == "No speech detected."

    def test_format_transcript_none_text(self, analyzer):
        """Test formatting segments with None text."""
        segments = [
            AudioSegment(text=None, start_time=0.0, end_time=1.0),
            AudioSegment(text="Speech", start_time=1.0, end_time=2.0),
        ]

        formatted = analyzer._format_transcript(segments)

        assert "Speech" in formatted
        assert "None" not in formatted

    def test_format_transcript_timestamp_precision(self, analyzer):
        """Test transcript formatting has correct timestamp precision."""
        segments = [
            AudioSegment(
                text="Test",
                start_time=0.123456,
                end_time=1.987654,
            ),
        ]

        formatted = analyzer._format_transcript(segments)

        # Should have one decimal place precision
        assert "[0.1s - 2.0s]" in formatted


class TestFormatFrameDescriptions:
    def test_format_frame_descriptions(self, analyzer):
        """Test formatting frame descriptions."""
        from video_analyzer.models import FrameAnalysis

        analyses = [
            FrameAnalysis(timestamp=0.0, description="Frame 1 desc", scene_type="static"),
            FrameAnalysis(timestamp=5.0, description="Frame 2 desc", scene_type="action"),
        ]

        formatted = analyzer._format_frame_descriptions(analyses)

        assert "Time 0.00s (static): Frame 1 desc" in formatted
        assert "Time 5.00s (action): Frame 2 desc" in formatted

    def test_format_frame_descriptions_empty(self, analyzer):
        """Test formatting empty frame descriptions."""
        formatted = analyzer._format_frame_descriptions([])

        assert formatted == ""

    def test_format_frame_descriptions_precision(self, analyzer):
        """Test timestamp precision in formatted descriptions."""
        from video_analyzer.models import FrameAnalysis

        analyses = [
            FrameAnalysis(
                timestamp=1.23456,
                description="Test",
                scene_type="action"
            ),
        ]

        formatted = analyzer._format_frame_descriptions(analyses)

        # Should have 2 decimal places
        assert "Time 1.23s" in formatted


class TestCalculateFrameCount:
    def test_calculate_uniform_frame_count(self, analyzer):
        """Test calculating uniform frame count."""
        # 30 second video -> 2 frames with default 4.0 fps_pm
        frame_count = analyzer._calculate_uniform_frame_count(30.0)
        assert analyzer.min_frames <= frame_count <= analyzer.max_frames

    def test_calculate_uniform_frame_count_respects_min_max(self, analyzer):
        """Test frame count respects min/max bounds."""
        # Very short video
        short_count = analyzer._calculate_uniform_frame_count(1.0)
        assert short_count >= analyzer.min_frames

        # Very long video
        long_count = analyzer._calculate_uniform_frame_count(3600.0)
        assert long_count <= analyzer.max_frames

    def test_calculate_uniform_frame_count_scales_with_duration(self, analyzer):
        """Test frame count scales with video duration."""
        count_30s = analyzer._calculate_uniform_frame_count(30.0)
        count_60s = analyzer._calculate_uniform_frame_count(60.0)

        # Longer video should have more frames (or same if hitting max)
        assert count_60s >= count_30s

    def test_calculate_dynamic_frame_count(self, analyzer):
        """Test calculating dynamic frame count based on scene changes."""
        # Few scene changes
        few_changes = analyzer._calculate_dynamic_frame_count(30.0, [])
        # Many scene changes
        many_changes = analyzer._calculate_dynamic_frame_count(30.0, [5, 10, 15, 20, 25])

        assert analyzer.min_frames <= few_changes <= analyzer.max_frames
        assert analyzer.min_frames <= many_changes <= analyzer.max_frames


class TestTruncateText:
    def test_truncate_text_no_truncation_needed(self, analyzer):
        """Test truncating text that doesn't need truncation."""
        text = "Short text"
        truncated = analyzer._truncate_text(text, 100)

        assert truncated == text

    def test_truncate_text_truncation(self, analyzer):
        """Test truncating long text."""
        text = "This is a long text that needs to be truncated because it exceeds the maximum character limit"
        truncated = analyzer._truncate_text(text, 30)

        assert len(truncated) <= 32  # 30 + "..."
        assert truncated.endswith("...")

    def test_truncate_text_zero_max_chars(self, analyzer):
        """Test truncate with zero max chars means unlimited."""
        text = "This is some text"
        truncated = analyzer._truncate_text(text, 0)

        assert truncated == text

    def test_truncate_text_respects_word_boundaries(self, analyzer):
        """Test truncation respects word boundaries."""
        text = "The quick brown fox jumps over the lazy dog"
        truncated = analyzer._truncate_text(text, 20)

        # Should not cut off in the middle of a word
        assert not truncated.rstrip("...").endswith(" ")

    def test_truncate_text_whitespace_normalization(self, analyzer):
        """Test text is normalized (whitespace collapsed)."""
        text = "Text  with   multiple    spaces"
        truncated = analyzer._truncate_text(text, 100)

        assert "  " not in truncated


class TestIsLowSignalAudio:
    def test_is_low_signal_audio_short_text(self, analyzer):
        """Test detection of low signal audio (very short)."""
        text = "um um um"
        assert analyzer._is_low_signal_audio(text) is True

    def test_is_low_signal_audio_normal_text(self, analyzer):
        """Test normal audio is not flagged as low signal."""
        text = "This is a normal sentence with proper words and content"
        assert analyzer._is_low_signal_audio(text) is False

    def test_is_low_signal_audio_repetitive(self, analyzer):
        """Test detection of repetitive audio."""
        text = "test test test test test test test test test"
        assert analyzer._is_low_signal_audio(text) is True

    def test_is_low_signal_audio_case_insensitive(self, analyzer):
        """Test low signal detection is case insensitive."""
        text_lower = "um um um"
        text_upper = "UM UM UM"
        
        assert analyzer._is_low_signal_audio(text_lower) == analyzer._is_low_signal_audio(text_upper)


class TestBuildContextNote:
    def test_build_context_note_both_contexts(self, analyzer):
        """Test building context note with both visual and audio."""
        visual = "Person walking on street"
        audio = "Hello, how are you?"

        context = analyzer._build_context_note(visual, audio)

        assert context is not None
        assert "Visual context" in context
        assert "Audio context" in context
        assert visual in context
        assert audio in context

    def test_build_context_note_only_visual(self, analyzer):
        """Test context note with only visual context."""
        visual = "Person walking on street"

        context = analyzer._build_context_note(visual, None)

        assert context is not None
        assert "Visual context" in context
        assert visual in context

    def test_build_context_note_only_audio(self, analyzer):
        """Test context note with only audio context."""
        audio = "Hello world"

        context = analyzer._build_context_note(None, audio)

        assert context is not None
        assert "Audio context" in context
        assert audio in context

    def test_build_context_note_none(self, analyzer):
        """Test context note with no contexts."""
        context = analyzer._build_context_note(None, None)

        assert context is None

    def test_build_context_note_truncation(self, analyzer):
        """Test context note truncates long text."""
        analyzer.context_max_chars = 20
        visual = "This is a very long visual context that should be truncated"

        context = analyzer._build_context_note(visual, None)

        assert context is not None
        # Should be truncated
        assert "..." in context or len(context) < len(visual)


class TestDetectHallucinations:
    def test_detect_hallucinations_no_hallucinations(self, analyzer):
        """Test detection when no hallucinations present."""
        text = "This is a normal sentence. This is another sentence. Here is a third one."

        result = analyzer._detect_hallucinations(text)

        assert result["text"] == text
        assert len(result["hallucinations"]) == 0

    def test_detect_hallucinations_repeated_sentence(self, analyzer):
        """Test detection of repeated sentences."""
        text = "This is repeated. This is repeated. This is repeated. Different sentence."

        result = analyzer._detect_hallucinations(text, min_repeat_count=3)

        assert len(result["hallucinations"]) > 0
        found = any(h["text"] == "This is repeated" for h in result["hallucinations"])
        assert found

    def test_detect_hallucinations_custom_threshold(self, analyzer):
        """Test hallucination detection with custom threshold."""
        text = "Item. Item. Item."

        result_2 = analyzer._detect_hallucinations(text, min_repeat_count=2)
        result_4 = analyzer._detect_hallucinations(text, min_repeat_count=4)

        assert len(result_2["hallucinations"]) > 0
        assert len(result_4["hallucinations"]) == 0


class TestSleepWithBackoff:
    def test_sleep_with_backoff_zero_backoff(self, analyzer):
        """Test sleep with zero backoff doesn't delay."""
        analyzer.request_backoff = 0.0

        start = time.time()
        analyzer._sleep_with_backoff(1)
        elapsed = time.time() - start

        assert elapsed < 0.1

    def test_sleep_with_backoff_exponential(self, analyzer):
        """Test backoff increases exponentially."""
        analyzer.request_backoff = 0.1

        start_1 = time.time()
        analyzer._sleep_with_backoff(1)
        elapsed_1 = time.time() - start_1

        start_2 = time.time()
        analyzer._sleep_with_backoff(2)
        elapsed_2 = time.time() - start_2

        # Second sleep should be roughly twice as long
        assert elapsed_2 > elapsed_1


class TestPostWithRetries:
    def test_post_with_retries_success_first_try(self, analyzer, monkeypatch):
        """Test successful POST request on first try."""
        class MockResponse:
            status_code = 200
            def raise_for_status(self):
                pass
            def json(self):
                return {"message": {"content": "ok"}}

        calls = [0]
        def mock_post(*args, **kwargs):
            calls[0] += 1
            return MockResponse()

        monkeypatch.setattr(analyzer.session, "post", mock_post)

        response = analyzer._post_with_retries({"test": True}, "test")

        assert calls[0] == 1
        assert response.json()["message"]["content"] == "ok"

    def test_post_with_retries_timeout_then_success(self, analyzer, monkeypatch):
        """Test retry on timeout."""
        class MockResponse:
            status_code = 200
            def raise_for_status(self):
                pass
            def json(self):
                return {"success": True}

        calls = [0]
        def mock_post(*args, **kwargs):
            calls[0] += 1
            if calls[0] == 1:
                raise requests.exceptions.Timeout("Timeout")
            return MockResponse()

        monkeypatch.setattr(analyzer.session, "post", mock_post)

        response = analyzer._post_with_retries({"test": True}, "test")

        assert calls[0] == 2
        assert response.json()["success"] is True

    def test_post_with_retries_max_retries_exceeded(self, analyzer, monkeypatch):
        """Test exception after max retries exceeded."""
        def mock_post(*args, **kwargs):
            raise requests.exceptions.Timeout("Timeout")

        monkeypatch.setattr(analyzer.session, "post", mock_post)

        with pytest.raises(requests.exceptions.Timeout):
            analyzer._post_with_retries({"test": True}, "test")


class TestAnalyzerConfiguration:
    def test_analyzer_min_max_frames_validation(self):
        """Test analyzer respects min/max frames."""
        analyzer = OllamaVideoAnalyzer(
            min_frames=5,
            max_frames=20,
        )

        assert analyzer.min_frames == 5
        assert analyzer.max_frames == 20

    def test_analyzer_timeout_configuration(self):
        """Test analyzer timeout configuration."""
        analyzer = OllamaVideoAnalyzer(
            request_timeout=180.0,
        )

        assert analyzer.request_timeout == 180.0

    def test_analyzer_retry_backoff_validation(self):
        """Test retry and backoff validation."""
        analyzer = OllamaVideoAnalyzer(
            request_retries=-1,  # Should be set to 0
            request_backoff=-0.5,  # Should be set to 0.0
        )

        assert analyzer.request_retries == 0
        assert analyzer.request_backoff == 0.0

    def test_analyzer_custom_prompts(self):
        """Test analyzer with custom prompts."""
        custom_prompts = AnalysisPrompts(
            frame_analysis="Custom frame prompt",
            detailed_summary="Custom detailed",
        )

        analyzer = OllamaVideoAnalyzer(
            prompts=custom_prompts,
        )

        assert analyzer.prompts.frame_analysis == "Custom frame prompt"
        assert analyzer.prompts.detailed_summary == "Custom detailed"

    def test_analyzer_analyze_quality_flag(self):
        """Test analyze_quality configuration."""
        analyzer_with_quality = OllamaVideoAnalyzer(
            analyze_quality=True,
        )
        analyzer_without_quality = OllamaVideoAnalyzer(
            analyze_quality=False,
        )

        assert analyzer_with_quality.analyze_quality is True
        assert analyzer_without_quality.analyze_quality is False
