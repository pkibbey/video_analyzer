from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN', '')

import argparse
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional



from video_analyzer.analyzer import OllamaVideoAnalyzer
from video_analyzer.frame_selectors import DynamicFrameSelector, UniformFrameSelector, AllFrameSelector
from video_analyzer.models import AnalysisPrompts
from video_analyzer.transcriber import WhisperTranscriber


def _setup_colored_logging(level: str) -> None:
    """Configure logging."""
    logging.basicConfig(level=getattr(logging, level))


def _load_prompts(prompts_file: Optional[str]) -> Dict[str, str]:
    if not prompts_file:
        return {}

    with open(prompts_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError("Prompts file must contain a JSON object.")

    return {key: value for key, value in data.items() if value}


def _build_cache_key(video_path: str, params: Dict[str, Any]) -> str:
    stat = os.stat(video_path)
    payload = {
        "path": os.path.abspath(video_path),
        "mtime": stat.st_mtime,
        "size": stat.st_size,
        "params": params,
    }
    digest = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(digest).hexdigest()


def _write_json(data: Dict[str, Any], output_path: Optional[str]) -> None:
    if output_path:
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=True)
        return

    print(json.dumps(data, indent=2, ensure_ascii=True))


def _print_summary(data: Dict[str, Any], structured_output: bool) -> None:
    if structured_output:
        summary = data["summary"]
        metadata = data["metadata"]
        warnings = data.get("warnings", [])
        brief = summary["brief"]
        detailed = summary["detailed"]
        timeline = summary["timeline"]
        transcript = summary["transcript"]
    else:
        metadata = data["metadata"]
        warnings = data.get("warnings", [])
        brief = data["brief_summary"]
        detailed = data["summary"]
        timeline = data["timeline"]
        transcript = data["transcript"]

    print("\nBrief Summary:")
    print("-" * 50)
    print(brief)

    print("\nDetailed Summary:")
    print("-" * 50)
    # Handle both dict (structured) and string (legacy) formats
    if isinstance(detailed, dict):
        if detailed.get("objective_summary"):
            print(f"Objective: {detailed['objective_summary']}\n")
        if detailed.get("visual_observations"):
            print(f"Visual Observations:\n{detailed['visual_observations']}\n")
        if detailed.get("sequence_of_events"):
            print(f"Sequence of Events:\n{detailed['sequence_of_events']}\n")
        if detailed.get("audio_transcript"):
            print(f"Audio Notes:\n{detailed['audio_transcript']}")
    else:
        print(detailed)

    print("\nVideo Timeline with Audio:")
    print("-" * 50)
    print(timeline)

    print("\nAudio Transcript:")
    print("-" * 50)
    print(transcript)

    print("\nMetadata:")
    print("-" * 50)
    print(f"Video Duration: {metadata.get('video_duration', 'N/A'):.2f}s")
    print(f"Frames Analyzed: {metadata.get('num_frames_analyzed', 'N/A')}")
    print(f"Audio Segments: {metadata.get('num_audio_segments', 'N/A')}")
    
    # Print models used
    models_used = metadata.get("models_used", {})
    print(f"\nModels Used:")
    print(f"  Frame Analysis: {models_used.get('frame_analysis', 'N/A')}")
    print(f"  Summary: {models_used.get('summary', 'N/A')}")
    if models_used.get("audio"):
        print(f"  Audio Transcription: {models_used.get('audio', 'N/A')}")
    
    # Print processing timings
    timings = metadata.get("processing_timings")
    if timings:
        print(f"\nProcessing Times:")
        print(f"  Frame Selection: {timings.get('frame_selection', 0):.2f}s")
        if timings.get("audio_transcription", 0) > 0:
            print(f"  Audio Transcription: {timings.get('audio_transcription', 0):.2f}s")
        print(f"  Frame Analysis: {timings.get('frame_analysis', 0):.2f}s")
        print(f"  Summary Generation: {timings.get('summary_generation', 0):.2f}s")
        print(f"  Total: {timings.get('total', 0):.2f}s")
    
    # Print scene distribution
    scene_dist = metadata.get("scene_distribution", {})
    if scene_dist:
        print(f"\nScene Distribution:")
        for scene_type, count in sorted(scene_dist.items()):
            if count > 0:
                print(f"  {scene_type.title()}: {count}")
    
    # Print video properties
    video_props = metadata.get("video_properties")
    if video_props:
        print(f"\nVideo Properties:")
        if video_props.get("width") and video_props.get("height"):
            print(f"  Resolution: {video_props.get('width')}x{video_props.get('height')}")
        if video_props.get("fps"):
            print(f"  Frame Rate: {video_props.get('fps'):.2f} fps")
        if video_props.get("total_frames"):
            print(f"  Total Frames: {video_props.get('total_frames')}")
        if video_props.get("codec"):
            print(f"  Video Codec: {video_props.get('codec')}")
        if video_props.get("format"):
            print(f"  Container Format: {video_props.get('format')}")
        if video_props.get("bitrate"):
            bitrate_mbps = video_props.get('bitrate', 0) / 1_000_000
            print(f"  Bitrate: {bitrate_mbps:.2f} Mbps")
        if video_props.get("data_rate"):
            print(f"  Data Rate: {video_props.get('data_rate')}")
        if video_props.get("audio_codec"):
            print(f"  Audio Codec: {video_props.get('audio_codec')}")
        if video_props.get("audio_sample_rate"):
            sample_rate = video_props.get('audio_sample_rate', 0)
            sample_rate_khz = sample_rate / 1000
            print(f"  Audio Sample Rate: {sample_rate_khz:.1f} kHz ({sample_rate} Hz)")
        if video_props.get("file_modified_date"):
            print(f"  File Modified: {video_props.get('file_modified_date')}")
        if video_props.get("file_created_date"):
            print(f"  File Created: {video_props.get('file_created_date')}")

    if warnings:
        print("\nWarnings:")
        print("-" * 50)
        for warning in warnings:
            print(warning)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Playground demo for experimenting with VideoAnalyzer.",
    )
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--frame-model", default="ministral-3:3b-cloud")
    parser.add_argument("--summary-model", default="ministral-3:14b-cloud")
    parser.add_argument("--whisper-model", default="openai/whisper-small")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--min-frames", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--frames-per-minute", type=float, default=4.0)
    parser.add_argument("--frame-selector", choices=["dynamic", "uniform", "all"], default="dynamic")
    parser.add_argument("--dynamic-threshold", type=float, default=20.0)
    parser.add_argument("--audio", action="store_true", default=True, help="Enable Whisper audio transcription")
    parser.add_argument("--no-audio", action="store_false", dest="audio", help="Disable Whisper audio transcription")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--no-collapse-repetitions",
        action="store_true",
        help="Keep repeated short phrases in transcripts",
    )
    parser.add_argument("--segment-duration", type=int, default=15)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--condition-on-prev-tokens",
        action="store_true",
        help="Use previous tokens when decoding (may increase repetition)",
    )
    parser.add_argument(
        "--local-files-only",
        dest="local_files_only",
        action="store_true",
        default=True,
        help="Load Whisper model only from local HF cache (no network fetch, default)",
    )
    parser.add_argument(
        "--no-local-files-only",
        dest="local_files_only",
        action="store_false",
        help="Allow network fetch for Whisper model if not available locally",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-backoff", type=float, default=1.0)
    parser.add_argument("--prompts-file", default="constrained_prompts.json", help="JSON file with prompt templates")
    parser.add_argument("--frame-prompt")
    parser.add_argument("--detailed-prompt")
    parser.add_argument("--brief-prompt")
    parser.add_argument("--max-detailed-chars", type=int, default=0, help="Max characters for detailed summary (0 = unlimited)")
    parser.add_argument("--max-brief-chars", type=int, default=0, help="Max characters for brief summary (0 = unlimited)")
    parser.add_argument("--analyze-quality", dest="analyze_quality", action="store_true", default=True, help="Analyze frame quality metrics for video editing")
    parser.add_argument("--no-analyze-quality", dest="analyze_quality", action="store_false", help="Disable frame quality analysis")
    parser.add_argument("--output", default=None, help="Write results to this JSON file (defaults to stdout)")
    parser.add_argument("--cache-dir", help="Directory to cache analysis results")
    parser.add_argument("--force", action="store_true", help="Ignore cached results")
    parser.add_argument("--structured-output", action="store_true", default=True, help="Always output JSON (to file if --output specified, else to stdout)")
    parser.add_argument("--print-json", action="store_true", help="Print full JSON output")
    parser.add_argument("--legacy-output", action="store_true", help="Use legacy output format")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    _setup_colored_logging(args.log_level)

    prompts_data = _load_prompts(args.prompts_file)
    if args.frame_prompt:
        prompts_data["frame_analysis"] = args.frame_prompt
    if args.detailed_prompt:
        prompts_data["detailed_summary"] = args.detailed_prompt
    if args.brief_prompt:
        prompts_data["brief_summary"] = args.brief_prompt

    prompts = AnalysisPrompts(**prompts_data) if prompts_data else None

    selector_map = {
        "dynamic": DynamicFrameSelector(threshold=args.dynamic_threshold),
        "uniform": UniformFrameSelector(),
        "all": AllFrameSelector(),
    }

    audio_transcriber = None
    if args.audio:
        audio_transcriber = WhisperTranscriber(
            model_name=args.whisper_model,
            device=args.device,
            collapse_repetitions=not args.no_collapse_repetitions,
            segment_duration=args.segment_duration,
            beam_size=args.beam_size,
            temperature=args.temperature,
            condition_on_prev_tokens=args.condition_on_prev_tokens,
            local_files_only=args.local_files_only,
        )

    structured_output = not args.legacy_output

    # --structured-output forces JSON output
    force_json_output = args.structured_output or args.print_json

    params_for_cache = {
        "frame_model": args.frame_model,
        "summary_model": args.summary_model,
        "host": args.host,
        "min_frames": args.min_frames,
        "max_frames": args.max_frames,
        "frames_per_minute": args.frames_per_minute,
        "frame_selector": args.frame_selector,
        "dynamic_threshold": args.dynamic_threshold,
        "audio": args.audio,
        "whisper_model": args.whisper_model,
        "device": args.device,
        "timeout": args.timeout,
        "retries": args.retries,
        "retry_backoff": args.retry_backoff,
        "prompts": prompts_data,
        "structured_output": structured_output,
        "analyze_quality": args.analyze_quality,
    }

    cache_path = None
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = _build_cache_key(args.video_path, params_for_cache)
        cache_path = cache_dir / f"{cache_key}.json"

        if cache_path.exists() and not args.force:
            with open(cache_path, "r", encoding="utf-8") as handle:
                cached = json.load(handle)
            if force_json_output:
                _write_json(cached, args.output)
            else:
                _print_summary(cached, structured_output)
            return 0

    analyzer = OllamaVideoAnalyzer(
        frame_analysis_model=args.frame_model,
        summary_model=args.summary_model,
        host=args.host,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        frames_per_minute=args.frames_per_minute,
        frame_selector=selector_map[args.frame_selector],
        audio_transcriber=audio_transcriber,
        prompts=prompts,
        log_level=getattr(logging, args.log_level),
        request_timeout=args.timeout,
        request_retries=args.retries,
        request_backoff=args.retry_backoff,
        max_detailed_summary_chars=args.max_detailed_chars,
        max_brief_summary_chars=args.max_brief_chars,
        analyze_quality=args.analyze_quality,
    )

    analysis_start = time.time()
    if structured_output:
        result = analyzer.analyze_video_structured(args.video_path)
        data = result.to_dict()
    else:
        data = analyzer.analyze_video(args.video_path)
    analysis_time = time.time() - analysis_start
    
    logging.getLogger(__name__).info(f"Total analysis time: {analysis_time:.2f}s")

    if cache_path:
        with open(cache_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=True)

    if force_json_output:
        _write_json(data, args.output)
    else:
        _print_summary(data, structured_output)

    # Log the output file location
    if force_json_output and args.output:
        print(f"\n✓ Analysis saved to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
