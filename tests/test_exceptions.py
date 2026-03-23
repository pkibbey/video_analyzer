"""Tests for custom exception classes."""

import pytest

from video_analyzer.exceptions import (
    APIError,
    FrameExtractionError,
    ModelInferenceError,
    TranscriptionError,
    VideoAnalysisError,
    VideoLoadError,
)


class TestVideoAnalysisError:
    def test_exception_is_exception(self):
        """Test VideoAnalysisError is a proper Exception."""
        assert issubclass(VideoAnalysisError, Exception)

    def test_exception_message(self):
        """Test exception can be raised with a message."""
        with pytest.raises(VideoAnalysisError) as exc_info:
            raise VideoAnalysisError("Test error message")

        assert str(exc_info.value) == "Test error message"

    def test_exception_no_args(self):
        """Test exception can be raised without args."""
        with pytest.raises(VideoAnalysisError):
            raise VideoAnalysisError()


class TestVideoLoadError:
    def test_inherits_from_video_analysis_error(self):
        """Test VideoLoadError inherits from VideoAnalysisError."""
        assert issubclass(VideoLoadError, VideoAnalysisError)

    def test_video_load_error_with_message(self):
        """Test VideoLoadError with message."""
        with pytest.raises(VideoLoadError) as exc_info:
            raise VideoLoadError("File not found: /path/to/video.mp4")

        assert "File not found" in str(exc_info.value)

    def test_video_load_error_caught_as_parent(self):
        """Test VideoLoadError can be caught as VideoAnalysisError."""
        with pytest.raises(VideoAnalysisError):
            raise VideoLoadError("File error")


class TestTranscriptionError:
    def test_inherits_from_video_analysis_error(self):
        """Test TranscriptionError inherits from VideoAnalysisError."""
        assert issubclass(TranscriptionError, VideoAnalysisError)

    def test_transcription_error_message(self):
        """Test TranscriptionError with descriptive message."""
        error_msg = "Whisper unanalyzed: CUDA out of memory"
        with pytest.raises(TranscriptionError) as exc_info:
            raise TranscriptionError(error_msg)

        assert error_msg in str(exc_info.value)

    def test_transcription_error_caught_as_parent(self):
        """Test TranscriptionError can be caught as VideoAnalysisError."""
        with pytest.raises(VideoAnalysisError):
            raise TranscriptionError("Audio transcription unanalyzed")


class TestFrameExtractionError:
    def test_inherits_from_video_analysis_error(self):
        """Test FrameExtractionError inherits from VideoAnalysisError."""
        assert issubclass(FrameExtractionError, VideoAnalysisError)

    def test_frame_extraction_error_message(self):
        """Test FrameExtractionError with message."""
        with pytest.raises(FrameExtractionError) as exc_info:
            raise FrameExtractionError("OpenCV unanalyzed to extract frame at timestamp 5.0")

        assert "timestamp" in str(exc_info.value)

    def test_frame_extraction_error_caught_as_parent(self):
        """Test FrameExtractionError can be caught as VideoAnalysisError."""
        with pytest.raises(VideoAnalysisError):
            raise FrameExtractionError("Frame extraction unanalyzed")


class TestModelInferenceError:
    def test_inherits_from_video_analysis_error(self):
        """Test ModelInferenceError inherits from VideoAnalysisError."""
        assert issubclass(ModelInferenceError, VideoAnalysisError)

    def test_model_inference_error_message(self):
        """Test ModelInferenceError with descriptive message."""
        error_msg = "Ollama API timeout after 120 seconds"
        with pytest.raises(ModelInferenceError) as exc_info:
            raise ModelInferenceError(error_msg)

        assert error_msg in str(exc_info.value)

    def test_model_inference_error_caught_as_parent(self):
        """Test ModelInferenceError can be caught as VideoAnalysisError."""
        with pytest.raises(VideoAnalysisError):
            raise ModelInferenceError("Model inference unanalyzed")


class TestAPIError:
    def test_inherits_from_video_analysis_error(self):
        """Test APIError inherits from VideoAnalysisError."""
        assert issubclass(APIError, VideoAnalysisError)

    def test_api_error_message(self):
        """Test APIError with message."""
        with pytest.raises(APIError) as exc_info:
            raise APIError("HTTP 503: Service Unavailable")

        assert "503" in str(exc_info.value)

    def test_api_error_caught_as_parent(self):
        """Test APIError can be caught as VideoAnalysisError."""
        with pytest.raises(VideoAnalysisError):
            raise APIError("API call unanalyzed")


class TestExceptionHierarchy:
    def test_all_custom_exceptions_inherit_from_base(self):
        """Test all custom exceptions inherit from VideoAnalysisError."""
        custom_exceptions = [
            VideoLoadError,
            TranscriptionError,
            FrameExtractionError,
            ModelInferenceError,
            APIError,
        ]

        for exc_class in custom_exceptions:
            assert issubclass(exc_class, VideoAnalysisError)

    def test_exception_catching_specificity(self):
        """Test catching specific exception types."""
        # Test catching specific types
        with pytest.raises(VideoLoadError):
            raise VideoLoadError("Load error")

        with pytest.raises(TranscriptionError):
            raise TranscriptionError("Transcription error")

        with pytest.raises(ModelInferenceError):
            raise ModelInferenceError("Inference error")

    def test_exception_chain(self):
        """Test exception chaining with cause."""
        original_error = ValueError("Original error")
        with pytest.raises(VideoAnalysisError) as exc_info:
            try:
                raise original_error
            except ValueError as e:
                raise VideoLoadError("Loading unanalyzed") from e

        assert exc_info.value.__cause__ == original_error


class TestExceptionContextManagement:
    def test_exception_context_preservation(self):
        """Test exception context is preserved through raising."""
        def failing_function():
            raise VideoLoadError("File not found")

        with pytest.raises(VideoLoadError) as exc_info:
            failing_function()

        assert isinstance(exc_info.value, VideoLoadError)
        assert isinstance(exc_info.value, VideoAnalysisError)

    def test_multiple_exception_types(self):
        """Test handling multiple exception types."""
        def function_that_may_fail(error_type):
            if error_type == "load":
                raise VideoLoadError("Load error")
            elif error_type == "transcribe":
                raise TranscriptionError("Transcribe error")
            elif error_type == "frame":
                raise FrameExtractionError("Frame error")
            elif error_type == "inference":
                raise ModelInferenceError("Inference error")
            elif error_type == "api":
                raise APIError("API error")

        # Test each error type
        for error_type in ["load", "transcribe", "frame", "inference", "api"]:
            with pytest.raises(VideoAnalysisError):
                function_that_may_fail(error_type)


class TestExceptionStringRepresentation:
    def test_exception_string_representation(self):
        """Test exception string representation."""
        error = VideoLoadError("Test error message")
        assert str(error) == "Test error message"

    def test_exception_repr(self):
        """Test exception repr."""
        error = ModelInferenceError("Inference unanalyzed")
        # repr should include the class name and message
        assert "ModelInferenceError" in repr(error) or "Inference unanalyzed" in repr(error)

    def test_empty_exception_message(self):
        """Test exception with empty message."""
        error = APIError()
        assert isinstance(error, APIError)
        assert isinstance(error, Exception)


class TestExceptionPropagation:
    def test_exception_propagates_through_functions(self):
        """Test exception propagates through function calls."""
        def level_3():
            raise FrameExtractionError("Frame error at level 3")

        def level_2():
            return level_3()

        def level_1():
            return level_2()

        with pytest.raises(FrameExtractionError) as exc_info:
            level_1()

        assert "level 3" in str(exc_info.value)

    def test_exception_context_in_nested_calls(self):
        """Test exception context in nested try-except blocks."""
        def inner():
            raise TranscriptionError("Inner error")

        def outer():
            try:
                return inner()
            except TranscriptionError:
                raise VideoAnalysisError("Outer error") from None

        with pytest.raises(VideoAnalysisError) as exc_info:
            outer()

        assert str(exc_info.value) == "Outer error"
