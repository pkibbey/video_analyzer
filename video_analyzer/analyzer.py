from typing import List, Dict, Tuple, Optional, Callable, Union, Any
import logging
import json
import time
from .models import (
    AnalysisPrompts,
    Frame,
    AudioSegment,
    SceneType,
    FrameAnalysis,
    SummaryResult,
    AnalysisMetadata,
    ModelsUsed,
    ProcessingTimings,
    VideoProperties,
    AnalysisResult,
)
from .frame_selectors import FrameSelector,DynamicFrameSelector
from .transcriber import AudioTranscriber
import numpy as np
from PIL import Image
import io
import base64
import requests
import time
import re
from .video_utils import get_video_duration, get_video_properties

class OllamaVideoAnalyzer:
    def __init__(
            self,
            frame_analysis_model: str = "ministral-3:3b-cloud",
            summary_model: str = "ministral-3:3b-cloud",
            host: str = "http://localhost:11434",
            min_frames: int = 8,
            max_frames: int = 64,
            frames_per_minute: float = 4.0,
            frame_selector: Optional[FrameSelector] = None,
            audio_transcriber: Optional[AudioTranscriber] = None,
            prompts: Optional[AnalysisPrompts] = None,
            custom_frame_processor: Optional[Callable[[Frame], Dict]] = None,
            log_level: int = logging.INFO,
            request_timeout: float = 120.0,
            request_retries: int = 3,
            request_backoff: float = 1.0,
            context_max_chars: int = 0,
            audio_context_max_chars: int = 0,
            analyze_quality: bool = False,
            max_detailed_summary_chars: int = 0,
            max_brief_summary_chars: int = 0,
    ):
        self.frame_analysis_model = frame_analysis_model
        self.summary_model = summary_model
        self.host = host
        self.api_endpoint = f"{host}/api/chat"
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.frames_per_minute = frames_per_minute
        self.frame_selector = frame_selector or DynamicFrameSelector()
        self.audio_transcriber = audio_transcriber
        self.prompts = prompts or AnalysisPrompts()
        self.custom_frame_processor = custom_frame_processor
        self.analyze_quality = analyze_quality
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        self.request_timeout = request_timeout
        self.request_retries = max(0, request_retries)
        self.request_backoff = max(0.0, request_backoff)
        self.retry_status_codes = {408, 429, 500, 502, 503, 504}
        self.context_max_chars = max(0, context_max_chars)
        self.audio_context_max_chars = max(0, audio_context_max_chars)
        self.max_detailed_summary_chars = max(0, max_detailed_summary_chars)
        self.max_brief_summary_chars = max(0, max_brief_summary_chars)
        logging.basicConfig(level=log_level)

    def _frame_to_base64(self, frame: np.ndarray) -> str:
        """Convert a frame to base64 string"""
        image = Image.fromarray(frame)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def _format_transcript(self, segments: List[AudioSegment]) -> str:
        """Format audio transcript with timestamps"""
        formatted = []
        for segment in segments:
            if not segment.text:
                continue
            formatted.append(
                f"[{segment.start_time:.1f}s - {segment.end_time:.1f}s]: {segment.text}"
            )

        if not formatted:
            return "No speech detected."

        return "\n".join(formatted)

    def _format_frame_descriptions(self, descriptions: List[FrameAnalysis]) -> str:
        """Format frame descriptions for the summary prompt"""
        formatted = []
        for desc in descriptions:
            formatted.append(f"Time {desc.timestamp:.2f}s ({desc.scene_type}): {desc.description}")
        return "\n".join(formatted)

    def _calculate_dynamic_frame_count(self, video_duration: float, scene_changes: List[float]) -> int:
        """Calculate optimal number of frames to analyze"""
        base_frames = int(video_duration / 60 * self.frames_per_minute)
        scene_density = len(scene_changes) / video_duration if video_duration > 0 else 0
        scene_multiplier = min(2.0, max(0.5, scene_density * 30))
        optimal_frames = int(base_frames * scene_multiplier)
        return min(self.max_frames, max(self.min_frames, optimal_frames))

    def _calculate_uniform_frame_count(self, video_duration: float) -> int:
        """Calculate the number of frames to select uniformly based on video duration."""
        base_frames = int((video_duration / 60) * self.frames_per_minute)
        optimal_frames = min(self.max_frames, max(self.min_frames, base_frames))
        self.logger.debug(
            f"Calculated uniform frame count: {optimal_frames} (Base: {base_frames}, Min: {self.min_frames}, Max: {self.max_frames})")
        return optimal_frames

    def _truncate_text(self, text: str, max_chars: int) -> str:
        compact = " ".join(text.split())
        if max_chars <= 0 or len(compact) <= max_chars:
            return compact
        clipped = compact[:max_chars].rsplit(" ", 1)[0]
        return f"{clipped}..." if clipped else compact[:max_chars]

    def _is_low_signal_audio(self, text: str) -> bool:
        tokens = re.findall(r"[a-zA-Z0-9']+", text.lower())
        if len(tokens) < 8:
            return True
        unique_ratio = len(set(tokens)) / len(tokens)
        return unique_ratio < 0.2

    def _build_context_note(self, context: Optional[str], audio_text: Optional[str]) -> Optional[str]:
        parts = []
        if context:
            parts.append(f"Visual context: {self._truncate_text(context, self.context_max_chars)}")
        if audio_text:
            parts.append(f"Audio context: {self._truncate_text(audio_text, self.audio_context_max_chars)}")
        if not parts:
            return None

        header = (
            "Use the following context for continuity only. "
            "Do not reference it explicitly unless it is directly visible in the frame."
        )
        return f"{header}\n" + "\n".join(parts)

    def _detect_hallucinations(self, text: str, min_repeat_count: int = 3) -> Dict[str, any]:
        """
        Detect repeated chunks of text that appear more than min_repeat_count times in a row.
        Returns a dict with 'text' and 'hallucinations' keys.
        """
        hallucinations = []
        
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if not sentences:
            return {"text": text, "hallucinations": []}
        
        # Check for repeated sentence patterns
        i = 0
        while i < len(sentences):
            current_sentence = sentences[i].strip()
            if not current_sentence:
                i += 1
                continue
            
            # Count consecutive repeats of this sentence
            repeat_count = 1
            j = i + 1
            while j < len(sentences) and sentences[j].strip() == current_sentence:
                repeat_count += 1
                j += 1
            
            # If sentence repeats more than threshold, flag it
            if repeat_count >= min_repeat_count:
                hallucinations.append({
                    "text": current_sentence,
                    "count": repeat_count,
                    "indices": list(range(i, i + repeat_count))
                })
                i = j
            else:
                i += 1
        
        return {
            "text": text,
            "hallucinations": hallucinations
        }

    def _sleep_with_backoff(self, attempt: int) -> None:
        delay = self.request_backoff * (2 ** (attempt - 1))
        if delay > 0:
            time.sleep(delay)

    def _post_with_retries(self, payload: Dict, request_label: str) -> requests.Response:
        attempts = self.request_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                response = self.session.post(
                    self.api_endpoint,
                    json=payload,
                    timeout=self.request_timeout,
                )
                if response.status_code in self.retry_status_codes:
                    raise requests.exceptions.HTTPError(
                        f"Retryable HTTP status {response.status_code}",
                        response=response,
                    )
                response.raise_for_status()
                return response
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                if attempt >= attempts:
                    raise
                self.logger.warning(
                    "Ollama %s request failed (attempt %s/%s): %s",
                    request_label,
                    attempt,
                    attempts,
                    exc,
                )
                self._sleep_with_backoff(attempt)
            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response else None
                if status in self.retry_status_codes and attempt < attempts:
                    self.logger.warning(
                        "Ollama %s request returned %s (attempt %s/%s)",
                        request_label,
                        status,
                        attempt,
                        attempts,
                    )
                    self._sleep_with_backoff(attempt)
                    continue
                raise

    def _analyze_frame(self, frame: Frame, prompt_override: Optional[str] = None) -> FrameAnalysis:
        """Analyze a single frame using the frame analysis model"""
        if self.custom_frame_processor:
            return self.custom_frame_processor(frame)

        base64_image = self._frame_to_base64(frame.image)
        prompt = prompt_override or self.prompts.frame_analysis

        messages = []
        if self.prompts.frame_analysis_system:
            messages.append({"role": "system", "content": self.prompts.frame_analysis_system})
        messages.append({"role": "user", "content": prompt, "images": [base64_image]})

        payload = {
            "model": self.frame_analysis_model,
            "messages": messages,
            "stream": False
        }

        try:
            response = self._post_with_retries(payload, "frame analysis")
            result = response.json()
            if "message" in result:
                desc = self._normalize_model_text(result["message"]["content"])
                return FrameAnalysis(
                    timestamp=frame.timestamp,
                    description=desc,
                    scene_type=frame.scene_type.value,
                )
            else:
                raise Exception("Invalid response format")
        except Exception as e:
            self.logger.error(f"Error analyzing frame: {str(e)}")
            return FrameAnalysis(
                timestamp=frame.timestamp,
                description="Error analyzing frame",
                scene_type=frame.scene_type.value,
                error=str(e),
            )

    def _analyze_frame_quality(self, frame: Frame) -> Tuple[Optional[Union[Dict[str, Any], str]], Optional[Dict[str, Any]]]:
        """Analyze frame quality metrics for editing usability.
        
        Returns:
            Tuple of (quality_analysis_text, quality_scores_dict)
        """
        base64_image = self._frame_to_base64(frame.image)
        
        messages = []
        if self.prompts.quality_analysis_system:
            messages.append({"role": "system", "content": self.prompts.quality_analysis_system})
        messages.append({"role": "user", "content": self.prompts.quality_analysis, "images": [base64_image]})

        payload = {
            "model": self.frame_analysis_model,
            "messages": messages,
            "stream": False
        }

        try:
            response = self._post_with_retries(payload, "quality analysis")
            result = response.json()
            if "message" in result:
                raw_quality_text = result["message"]["content"]
                quality_text = self._normalize_model_text(raw_quality_text)
                quality_scores = self._extract_quality_scores_from_text(quality_text)

                if quality_scores is not None:
                    # Store parsed JSON object for quality_analysis + quality_scores
                    return quality_scores, quality_scores

                self.logger.warning(
                    f"Could not parse quality scores at {frame.timestamp}s; storing quality analysis object with raw text."
                )
                return {"raw_quality_analysis": quality_text}, None
            else:
                raise Exception("Invalid response format")
        except Exception as e:
            self.logger.error(f"Error analyzing frame quality: {str(e)}")
            return None, None

    def _extract_quality_scores_from_text(self, text: str) -> Optional[Dict]:
        """Try to extract a JSON object from model text output and parse to dict."""
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Attempt to extract JSON object even if wrapped by text
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            candidate = text[start:end + 1].strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # Fallback: try row-by-row key:value map (not strict JSON)
        # Example: focus: 8\nexposure: 7
        metrics = {}
        for line in text.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().strip('"')
                value = value.strip().strip('"')
                if not key:
                    continue
                try:
                    metrics[key] = float(value) if "." in value else int(value)
                    continue
                except ValueError:
                    pass
                # List detection
                if value.startswith("[") and value.endswith("]"):
                    try:
                        metrics[key] = json.loads(value)
                        continue
                    except json.JSONDecodeError:
                        pass
                metrics[key] = value
        return metrics or None

    def _normalize_model_text(self, text: Optional[str]) -> str:
        """Normalize model text outputs: strip code fences and whitespace."""
        if not text:
            return ""
        normalized = text.strip()

        # Remove triple backticks and optional language label
        code_fences = ["```json", "```", "~~~"]
        for fence in code_fences:
            if normalized.startswith(fence):
                # Remove leading fence
                normalized = normalized[len(fence):].strip()
                # Remove trailing fence if exists
                if normalized.endswith(fence):
                    normalized = normalized[:-len(fence)].strip()
                break

        # Remove standalone code block close markers
        if normalized.endswith("```"):
            normalized = normalized[:-3].strip()
        if normalized.endswith("~~~"):
            normalized = normalized[:-3].strip()

        return normalized

    def _calculate_trim_recommendations(self, frame_analyses: List[FrameAnalysis], video_duration: float) -> Dict[str, any]:
        """Calculate trim point recommendations based on frame quality scores"""
        def _to_float_quality(value: Any) -> float:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    return 0.0
                # Support values like "7", "7.5", "7/10"
                try:
                    return float(value)
                except ValueError:
                    if "/" in value:
                        parts = value.split("/")
                        try:
                            num = float(parts[0].strip())
                            den = float(parts[1].strip()) if len(parts) > 1 and parts[1].strip() else 1.0
                            return (num / den) * 10.0 if den != 0 else num
                        except ValueError:
                            pass
            return 0.0

        quality_timeline = []
        for analysis in frame_analyses:
            if analysis.quality_scores:
                quality_timeline.append({
                    "timestamp": analysis.timestamp,
                    "overall_quality": _to_float_quality(analysis.quality_scores.get("overall_quality", 5)),
                    "issues": analysis.quality_scores.get("issues", [])
                })
        
        if not quality_timeline:
            return {}
        
        # Sort by timestamp
        quality_timeline.sort(key=lambda x: x["timestamp"])
        
        # Find trim points (start and end of usable content)
        # Quality threshold for "usable" content
        quality_threshold = 4  # out of 10
        
        usable_regions = []
        current_start = None
        
        for idx, item in enumerate(quality_timeline):
            if item["overall_quality"] >= quality_threshold:
                if current_start is None:
                    current_start = item["timestamp"]
            else:
                if current_start is not None:
                    end_ts = quality_timeline[idx - 1]["timestamp"] if idx > 0 else item["timestamp"]
                    usable_regions.append({
                        "start": current_start,
                        "end": end_ts
                    })
                    current_start = None
        
        # Close final region if still open
        if current_start is not None:
            usable_regions.append({
                "start": current_start,
                "end": quality_timeline[-1]["timestamp"]
            })
        
        # Find best region (longest high-quality section)
        best_region = max(usable_regions, key=lambda x: x["end"] - x["start"]) if usable_regions else None
        
        # Identify problem zones
        problem_zones = [item for item in quality_timeline if item["issues"]]
        
        return {
            "quality_timeline": quality_timeline,
            "recommended_trim": {
                "start": best_region["start"] if best_region else 0,
                "end": best_region["end"] if best_region else video_duration
            } if best_region else None,
            "usable_regions": usable_regions,
            "problem_zones": problem_zones,
            "analysis_note": "Trim recommendations based on visual quality assessment across frames"
        }

    def _generate_summary(
        self,
        frame_descriptions: List[FrameAnalysis],
        audio_segments: List[AudioSegment],
        video_duration: float,
    ) -> Tuple[SummaryResult, Optional[str]]:
        """Generate summaries using both video and audio information"""
        timeline = self._format_frame_descriptions(frame_descriptions)
        transcript = self._format_transcript(audio_segments) if audio_segments else "No audio transcript available."

        try:
            # Get detailed summary
            detailed_prompt = self.prompts.detailed_summary.format(
                duration=video_duration,
                timeline=timeline,
                transcript=transcript
            )
            detailed_payload = {
                "model": self.summary_model,
                "messages": [{"role": "user", "content": detailed_prompt}],
                "stream": False
            }
            detailed_response = self._post_with_retries(detailed_payload, "summary (detailed)")
            detailed_result = detailed_response.json()

            # Get brief summary
            brief_prompt = self.prompts.brief_summary.format(
                duration=video_duration,
                timeline=timeline,
                transcript=transcript
            )
            brief_payload = {
                "model": self.summary_model,
                "messages": [{"role": "user", "content": brief_prompt}],
                "stream": False
            }
            brief_response = self._post_with_retries(brief_payload, "summary (brief)")
            brief_result = brief_response.json()

            # Extract detailed summary as plain string
            detailed_text = (
                self._normalize_model_text(detailed_result["message"]["content"])
                if "message" in detailed_result
                else "Unable to generate detailed summary"
            )
            
            # Apply text length limits to detailed summary
            if self.max_detailed_summary_chars > 0:
                detailed_text = self._truncate_text(detailed_text, self.max_detailed_summary_chars)
            
            # Extract brief summary
            brief_text = (
                self._normalize_model_text(brief_result["message"]["content"])
                if "message" in brief_result
                else "Unable to generate brief summary"
            )
            
            # Apply text length limits to brief summary
            if self.max_brief_summary_chars > 0:
                brief_text = self._truncate_text(brief_text, self.max_brief_summary_chars)

            return SummaryResult(
                detailed=detailed_text,
                brief=brief_text,
            ), None
        except Exception as e:
            self.logger.error(f"Error generating summaries: {str(e)}")
            return SummaryResult(
                detailed="Error generating detailed summary",
                brief="Error generating brief summary",
            ), f"Summary generation failed: {str(e)}"

    def analyze_video_structured(self, video_path: str) -> AnalysisResult:
        """Provide a comprehensive video analysis using both visual and audio content"""
        overall_start = time.time()
        self.logger.info(f"Starting video analysis for: {video_path}")

        try:
            # Get video properties
            video_props_dict = get_video_properties(video_path)
            video_properties = VideoProperties(**video_props_dict)
            
            # Extract key frames
            frame_start = time.time()
            self.logger.info("Selecting key frames")
            frames = self.frame_selector.select_frames(video_path, self)
            frame_time = time.time() - frame_start
            self.logger.info(f"Selected {len(frames)} frames for analysis in {frame_time:.2f}s")

            # Transcribe audio if available
            audio_segments: List[AudioSegment] = []
            warnings: List[str] = []
            audio_time = 0.0
            if self.audio_transcriber:
                audio_start = time.time()
                self.logger.info("Starting audio transcription")
                try:
                    audio_segments = self.audio_transcriber.transcribe(video_path)
                    audio_time = time.time() - audio_start
                    self.logger.info(f"Transcribed {len(audio_segments)} audio segments in {audio_time:.2f}s")
                except Exception as e:
                    audio_time = time.time() - audio_start
                    warning = f"Audio transcription failed after {audio_time:.2f}s: {str(e)}"
                    warnings.append(warning)
                    self.logger.error(warning)

            # Analyze frames with context
            frame_descriptions: List[FrameAnalysis] = []
            context = None
            frame_analysis_start = time.time()

            for i, frame in enumerate(frames):
                self.logger.info(f"Analyzing frame {i + 1}/{len(frames)} at {frame.timestamp:.2f}s")

                # Find relevant audio segments for this frame
                relevant_audio = [
                    segment for segment in audio_segments
                    if segment.start_time <= frame.timestamp <= segment.end_time
                ]

                frame_context = context
                audio_context = None
                if relevant_audio:
                    audio_text = relevant_audio[0].text
                    if audio_text and not self._is_low_signal_audio(audio_text):
                        audio_context = audio_text

                context_note = self._build_context_note(frame_context, audio_context)
                frame_prompt = self.prompts.frame_analysis
                if context_note:
                    frame_prompt = f"{frame_prompt}\n\n{context_note}"

                
                analysis = self._analyze_frame(frame, frame_prompt)
                
                # Analyze quality if enabled
                if self.analyze_quality:
                    quality_analysis_text, quality_scores = self._analyze_frame_quality(frame)
                    analysis.quality_analysis = quality_analysis_text
                    analysis.quality_scores = quality_scores
                
                frame_descriptions.append(analysis)
                context = analysis.description
                time.sleep(0.1)  # Rate limiting

            frame_analysis_time = time.time() - frame_analysis_start
            self.logger.info(f"Frame analysis completed in {frame_analysis_time:.2f}s "
                           f"({frame_analysis_time/len(frames):.2f}s per frame)")

            # Generate summaries with both video and audio
            # Calculate duration from video properties when available
            if video_properties.fps and video_properties.total_frames and video_properties.total_frames > 0:
                video_duration = video_properties.total_frames / video_properties.fps
            else:
                # Fallback to frame timestamp or probe if properties unavailable
                fallback_duration = frames[-1].timestamp if frames else 0.0
                video_duration = get_video_duration(video_path, fallback_duration=fallback_duration)
            self.logger.info("Generating video and audio summaries")
            summary_start = time.time()
            summaries, summary_warning = self._generate_summary(
                frame_descriptions,
                audio_segments,
                video_duration,
            )
            summary_time = time.time() - summary_start
            self.logger.info(f"Summary generation completed in {summary_time:.2f}s")
            
            # Calculate trim recommendations if quality analysis was performed
            if self.analyze_quality:
                self.logger.info("Calculating trim recommendations")
                trim_recs = self._calculate_trim_recommendations(frame_descriptions, video_duration)
                summaries.trim_recommendations = trim_recs
            
            if summary_warning:
                warnings.append(summary_warning)

            # Collect metadata
            scene_distribution = {
                scene_type.value: len([f for f in frames if f.scene_type == scene_type])
                for scene_type in SceneType
            }

            frame_errors = [analysis for analysis in frame_descriptions if analysis.error]
            if frame_errors:
                warnings.append(f"{len(frame_errors)} frame analyses failed.")

            metadata = AnalysisMetadata(
                num_frames_analyzed=len(frames),
                num_audio_segments=len(audio_segments),
                video_duration=video_duration,
                scene_distribution=scene_distribution,
                models_used=ModelsUsed(
                    frame_analysis=self.frame_analysis_model,
                    summary=self.summary_model,
                    audio=self.audio_transcriber.model_name if self.audio_transcriber else None,
                ),
                processing_timings=ProcessingTimings(
                    frame_selection=frame_time,
                    audio_transcription=audio_time,
                    frame_analysis=frame_analysis_time,
                    summary_generation=summary_time,
                    total=time.time() - overall_start,
                ),
                video_properties=video_properties,
            )
            result = AnalysisResult(
                summary=summaries,
                frame_analyses=frame_descriptions,
                audio_segments=audio_segments,
                metadata=metadata,
                warnings=warnings,
            )

            overall_time = time.time() - overall_start
            self.logger.info(f"Video and audio analysis completed in {overall_time:.2f}s total")
            self.logger.info(f"  Frame selection: {frame_time:.2f}s")
            if audio_time > 0:
                self.logger.info(f"  Audio transcription: {audio_time:.2f}s")
            self.logger.info(f"  Frame analysis: {frame_analysis_time:.2f}s")
            self.logger.info(f"  Summary generation: {summary_time:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f"Error during video analysis: {str(e)}", exc_info=True)
            raise

    def analyze_video(self, video_path: str) -> Dict:
        """Provide a comprehensive video analysis as a legacy dictionary."""
        return self.analyze_video_structured(video_path).to_legacy_dict()
