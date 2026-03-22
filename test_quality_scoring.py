#!/usr/bin/env python3
"""Quick test to verify quality scoring integration"""

from openscenesense_ollama.models import (
    AnalysisPrompts,
    FrameAnalysis,
    SummaryResult,
)
from openscenesense_ollama.analyzer import OllamaVideoAnalyzer

# Test 1: Verify prompts load correctly
print("Test 1: Loading prompts...")
prompts = AnalysisPrompts()
assert hasattr(prompts, 'quality_analysis'), "quality_analysis prompt missing"
assert hasattr(prompts, 'quality_analysis_system'), "quality_analysis_system prompt missing"
print("✓ Quality analysis prompts loaded")

# Test 2: Verify FrameAnalysis supports quality_scores
print("\nTest 2: Testing FrameAnalysis with quality scores...")
analysis = FrameAnalysis(
    timestamp=1.0,
    description="Test frame",
    scene_type="static",
    quality_scores={"focus": 8, "exposure": 7, "overall_quality": 7.5}
)
assert analysis.quality_scores is not None, "quality_scores not stored"
print(f"✓ Quality scores stored: {analysis.quality_scores}")

# Test 3: Verify SummaryResult creation works
print("\nTest 3: Testing SummaryResult construction...")
summary = SummaryResult(
    detailed="Test summary",
    brief="Brief"
)
assert summary is not None, "SummaryResult not constructable"
print("✓ SummaryResult constructed")

# Test 4: Verify analyzer has quality analysis flag
print("\nTest 4: Testing OllamaVideoAnalyzer with analyze_quality flag...")
analyzer = OllamaVideoAnalyzer(analyze_quality=True)
assert analyzer.analyze_quality is True, "analyze_quality flag not set"
assert hasattr(analyzer, '_analyze_frame_quality'), "_analyze_frame_quality method missing"
assert hasattr(analyzer, '_calculate_trim_recommendations'), "_calculate_trim_recommendations method missing"
assert hasattr(analyzer, '_extract_quality_scores_from_text'), "_extract_quality_scores_from_text method missing"

# Test 5: robust quality JSON extraction from text output
noisy = 'Quality check: {"focus": 8, "exposure": 7, "issues": ["none"], "overall_quality": 7.5} \n Some extra text.'
parsed = analyzer._extract_quality_scores_from_text(noisy)
assert parsed is not None and parsed.get('focus') == 8, "JSON extraction failed"
print("✓ Extracted quality scores from wrapped text")

# Test 6: normalize model text with code fences
raw_quality = "```json\n{\"focus\":8,\"exposure\":7}\n```"
normalized = analyzer._normalize_model_text(raw_quality)
assert normalized == '{"focus":8,"exposure":7}', f"normalize_model_text failed: {normalized}"
print("✓ Normalized code-fenced quality text")

# Test 7: trim recommendation handles numeric strings and decimal formats
frames = [
    FrameAnalysis(timestamp=0.0, description="A", scene_type="static", quality_scores={"overall_quality": "4.5", "issues": []}),
    FrameAnalysis(timestamp=1.0, description="B", scene_type="static", quality_scores={"overall_quality": "3", "issues": ["low_light"]}),
    FrameAnalysis(timestamp=2.0, description="C", scene_type="static", quality_scores={"overall_quality": "5", "issues": []}),
]
trim = analyzer._calculate_trim_recommendations(frames, 3.0)
assert trim is not None and "recommended_trim" in trim, "trim recommendation missing"
assert trim["recommended_trim"]["start"] == 0.0, "unexpected trim start"
print("✓ _calculate_trim_recommendations handles numeric string quality")

print("✓ Quality analysis methods available")

print("\n✅ All tests passed!")
