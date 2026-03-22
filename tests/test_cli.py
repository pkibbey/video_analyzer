"""Tests for command-line interface."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from video_analyzer.cli import _build_cache_key, _load_prompts, _write_output
from video_analyzer.models import AnalysisPrompts


class TestLoadPrompts:
    def test_load_empty_prompts_file(self):
        """Test loading empty prompts file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({}, f)
            f.flush()
            temp_path = f.name

        try:
            prompts = _load_prompts(temp_path)
            assert prompts == {}
        finally:
            Path(temp_path).unlink()

    def test_load_prompts_with_values(self):
        """Test loading prompts file with values."""
        prompts_data = {
            "frame_analysis": "Custom frame prompt",
            "detailed_summary": "Custom detailed prompt",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(prompts_data, f)
            f.flush()
            temp_path = f.name

        try:
            loaded = _load_prompts(temp_path)
            assert loaded["frame_analysis"] == "Custom frame prompt"
            assert loaded["detailed_summary"] == "Custom detailed prompt"
        finally:
            Path(temp_path).unlink()

    def test_load_prompts_filters_empty_values(self):
        """Test loading prompts filters out empty values."""
        prompts_data = {
            "frame_analysis": "Custom prompt",
            "detailed_summary": "",  # Empty - should be filtered
            "brief_summary": None,  # None - should be filtered
            "other_key": "value",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(prompts_data, f)
            f.flush()
            temp_path = f.name

        try:
            loaded = _load_prompts(temp_path)
            assert "frame_analysis" in loaded
            assert "detailed_summary" not in loaded
            assert "other_key" in loaded
        finally:
            Path(temp_path).unlink()

    def test_load_prompts_none_file(self):
        """Test loading prompts with None file returns empty dict."""
        prompts = _load_prompts(None)
        assert prompts == {}

    def test_load_prompts_invalid_json(self):
        """Test loading invalid JSON file raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json")
            f.flush()
            temp_path = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                _load_prompts(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_load_prompts_not_dict_raises_error(self):
        """Test loading non-dict JSON raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(["list", "not", "dict"], f)
            f.flush()
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="must contain a JSON object"):
                _load_prompts(temp_path)
        finally:
            Path(temp_path).unlink()


class TestBuildCacheKey:
    def test_cache_key_consistency(self):
        """Test cache key is consistent for same inputs."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            test_file = f.name
            f.write(b"test content")

        try:
            params = {"model": "test", "frames": 10}
            key1 = _build_cache_key(test_file, params)
            key2 = _build_cache_key(test_file, params)

            assert key1 == key2
        finally:
            Path(test_file).unlink()

    def test_cache_key_different_params(self):
        """Test cache key differs for different parameters."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            test_file = f.name
            f.write(b"test content")

        try:
            params1 = {"model": "model1"}
            params2 = {"model": "model2"}

            key1 = _build_cache_key(test_file, params1)
            key2 = _build_cache_key(test_file, params2)

            assert key1 != key2
        finally:
            Path(test_file).unlink()

    def test_cache_key_different_files(self):
        """Test cache key differs for different files."""
        with tempfile.NamedTemporaryFile(delete=False) as f1:
            f1.write(b"content1")
            file1 = f1.name

        with tempfile.NamedTemporaryFile(delete=False) as f2:
            f2.write(b"content2")
            file2 = f2.name

        try:
            params = {"model": "test"}
            key1 = _build_cache_key(file1, params)
            key2 = _build_cache_key(file2, params)

            assert key1 != key2
        finally:
            Path(file1).unlink()
            Path(file2).unlink()

    def test_cache_key_is_hex_string(self):
        """Test cache key is a valid hex string."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            test_file = f.name
            f.write(b"test")

        try:
            key = _build_cache_key(test_file, {})

            # Should be a valid hex string (SHA256 produces 64 hex chars)
            assert isinstance(key, str)
            assert len(key) == 64
            assert all(c in "0123456789abcdef" for c in key)
        finally:
            Path(test_file).unlink()

    def test_cache_key_includes_file_metadata(self):
        """Test cache key is affected by file modification time and size."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            test_file = f.name
            f.write(b"initial content")

        try:
            key1 = _build_cache_key(test_file, {"model": "test"})

            # Modify file
            with open(test_file, "w") as f:
                f.write("modified content")

            key2 = _build_cache_key(test_file, {"model": "test"})

            # Keys should differ due to file modification
            assert key1 != key2
        finally:
            Path(test_file).unlink()

    def test_cache_key_parameter_order_independence(self):
        """Test that parameter order doesn't affect cache key."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            test_file = f.name
            f.write(b"test")

        try:
            params1 = {"a": 1, "b": 2, "c": 3}
            params2 = {"c": 3, "a": 1, "b": 2}

            key1 = _build_cache_key(test_file, params1)
            key2 = _build_cache_key(test_file, params2)

            assert key1 == key2
        finally:
            Path(test_file).unlink()


class TestWriteOutput:
    def test_write_output_to_file(self):
        """Test writing output to a file."""
        data = {"summary": "Test summary", "frames": []}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            _write_output(data, output_path)

            # Verify file was written
            assert Path(output_path).exists()

            # Verify content
            with open(output_path, "r") as f:
                loaded = json.load(f)

            assert loaded["summary"] == "Test summary"
        finally:
            Path(output_path).unlink()

    def test_write_output_to_stdout(self, capsys):
        """Test writing output to stdout when no file specified."""
        data = {"key": "value"}

        _write_output(data, None)

        captured = capsys.readouterr()
        assert "key" in captured.out
        assert "value" in captured.out

    def test_write_output_creates_parent_directories(self):
        """Test output file creation with nested directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "path" / "output.json"
            data = {"test": "data"}

            _write_output(data, str(output_path))

            assert output_path.exists()

    def test_write_output_json_formatting(self):
        """Test output is properly formatted JSON."""
        data = {
            "summary": "Test",
            "nested": {"key": "value"},
            "list": [1, 2, 3],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            _write_output(data, output_path)

            with open(output_path, "r") as f:
                content = f.read()

            # Should have indentation (pretty-printed)
            assert "\n" in content
            assert "    " in content or "  " in content
        finally:
            Path(output_path).unlink()

    def test_write_output_complex_data(self):
        """Test writing complex nested data structures."""
        data = {
            "summary": "Complex test",
            "frame_analyses": [
                {
                    "timestamp": 1.5,
                    "description": "Frame description",
                    "quality_scores": {"brightness": 8, "focus": 9},
                }
            ],
            "metadata": {
                "num_frames": 10,
                "models": {"frame": "model1", "summary": "model2"},
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            _write_output(data, output_path)

            with open(output_path, "r") as f:
                loaded = json.load(f)

            assert loaded["frame_analyses"][0]["timestamp"] == 1.5
            assert loaded["metadata"]["num_frames"] == 10
        finally:
            Path(output_path).unlink()


class TestAnalysisPromptsIntegration:
    def test_prompts_file_integration(self):
        """Test integration of prompts file with AnalysisPrompts."""
        prompts_data = {
            "frame_analysis": "Custom frame analysis",
            "detailed_summary": "Custom detailed",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(prompts_data, f)
            f.flush()
            temp_path = f.name

        try:
            loaded = _load_prompts(temp_path)
            prompts = AnalysisPrompts(**loaded)

            assert prompts.frame_analysis == "Custom frame analysis"
            assert prompts.detailed_summary == "Custom detailed"
            # Should have defaults for other fields
            assert prompts.frame_analysis_system is not None
        finally:
            Path(temp_path).unlink()


class TestCacheKeyIntegration:
    def test_cache_key_with_complex_params(self):
        """Test cache key generation with complex parameters."""
        with tempfile.NamedTemporaryFile(delete=False) as f:
            test_file = f.name
            f.write(b"test")

        try:
            params = {
                "frame_model": "model1",
                "summary_model": "model2",
                "host": "http://localhost:11434",
                "min_frames": 4,
                "max_frames": 16,
                "frames_per_minute": 4.0,
                "frame_selector": "dynamic",
                "dynamic_threshold": 20.0,
                "audio": True,
                "whisper_model": "openai/whisper-small",
                "structured_output": True,
            }

            key = _build_cache_key(test_file, params)

            # Should produce valid cache key
            assert isinstance(key, str)
            assert len(key) == 64
        finally:
            Path(test_file).unlink()


class TestOutputFileHandling:
    def test_overwrite_existing_file(self):
        """Test that output overwrites existing files."""
        data1 = {"version": 1}
        data2 = {"version": 2}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            # Write first version
            _write_output(data1, output_path)
            with open(output_path, "r") as f:
                result1 = json.load(f)
            assert result1["version"] == 1

            # Overwrite with second version
            _write_output(data2, output_path)
            with open(output_path, "r") as f:
                result2 = json.load(f)
            assert result2["version"] == 2
        finally:
            Path(output_path).unlink()

    def test_empty_output_path(self):
        """Test write_output with empty string path writes to stdout."""
        data = {"test": "data"}
        # Empty string should be treated as stdout
        _write_output(data, "")

    def test_output_with_special_characters(self):
        """Test output can handle data with special characters."""
        data = {
            "description": "Contains special chars: ñ, é, 中文, emoji 🎬",
            "unicode": "Unicode test: \u0041\u0300",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            _write_output(data, output_path)

            with open(output_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            assert "特" in loaded["description"] or "🎬" in loaded["description"]
        finally:
            Path(output_path).unlink()
