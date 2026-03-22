
# Video Analysis Prompt Examples for Personal and Semi-Professional Content

**Note:** These prompts are optimized for analyzing personal video content created by semi-professional users—including travel vlogs, pet videos, sports footage, festival coverage, fireworks displays, and videos of friends and family gatherings.

Harness the full potential of **VideoAnalyzer** with these innovative video analysis prompt examples. These prompts are categorized into **Frame-Based Analysis**, **Detailed Summaries**, and **Brief Summaries** to align with VideoAnalyzer's `AnalysisPrompts` structure. Each example includes a detailed description and potential use cases tailored for semi-professional personal content creators.

---

## Table of Contents

1. [Frame-Based Analysis Prompts](#frame-based-analysis-prompts)
2. [Detailed Summary Prompts](#detailed-summary-prompts)
3. [Brief Summary Prompts](#brief-summary-prompts)
4. [Leveraging These Prompts with VideoAnalyzer](#leveraging-these-prompts-with-videoanalyzer)
5. [Best Practices](#best-practices)
6. [Conclusion](#conclusion)

---

## 1. Frame-Based Analysis Prompts

Frame-based analysis focuses on extracting detailed information from individual frames within a video. Below are innovative prompts designed to enhance the granularity and depth of frame analysis.

### 1.1. **Object and Action Detection**

```plaintext
"Analyze this frame by identifying all visible objects and their current states. Describe the actions being performed and the relationships between different objects."
```

**Description:**  
This prompt directs the model to meticulously identify objects within a frame and understand their interactions and actions, providing a detailed snapshot of the scene.

**Use Cases:**
- **Travel Vlogging:** Identify landmarks, activities, and interactions at travel destinations.
- **Pet Videos:** Detect pet behaviors, tricks, and interactions with people and environment.
- **Event Documentation:** Analyze crowd interactions and key activities at festivals, sports events, and celebrations.

---

### 1.2. **Emotion and Expression Recognition**

```plaintext
"Examine this frame to identify the emotions expressed by individuals. Describe facial expressions, body language, and any contextual cues that indicate their emotional states."
```

**Description:**  
Focuses on interpreting the emotional states of individuals within a frame by analyzing visual cues such as facial expressions and body language.

**Use Cases:**
- **Social Content Creation:** Capture genuine emotional moments in personal videos for storytelling.
- **Memory Preservation:** Document emotional highlights from family gatherings, parties, and celebrations.
- **Sports Highlights:** Identify emotional peaks and reactions during sports events and competitions.

---

### 1.3. **Environmental and Contextual Analysis**

```plaintext
"Assess the environmental elements present in this frame, including lighting, weather conditions, and background settings. Explain how these factors contribute to the overall atmosphere of the scene."
```

**Description:**  
Encourages the model to evaluate the broader environmental context of a frame, understanding how elements like lighting and weather influence the scene's mood.

**Use Cases:**
- **Travel Content:** Assess lighting and weather conditions in various destinations for visual quality insight.
- **Festival and Event Filming:** Understand how environmental factors contribute to the atmosphere and mood.
- **Outdoor Sports Coverage:** Evaluate weather, lighting, and terrain impacts on video quality and storytelling.

---

### 1.4. **Action Sequence Identification**

```plaintext
"Identify and describe the sequence of actions taking place in this frame. Highlight any significant movements or interactions that are pivotal to the ongoing narrative."
```

**Description:**  
Aims to dissect the flow of actions within a frame, pinpointing key movements that drive the story forward.

**Use Cases:**
- **Sports Analysis:** Break down plays, tricks, and skill demonstrations in sports footage.
- **Pet Training Content:** Document behavioral sequences and tricks for instructional clips.
- **Event Highlights:** Identify and sequence pivotal moments like fireworks displays or performance routines.

---

### 1.5. **Color and Composition Evaluation**

```plaintext
"Evaluate the color palette and composition of this frame. Discuss how the use of color and spatial arrangement enhances the visual appeal and storytelling of the video."
```

**Description:**  
Focuses on the artistic aspects of a frame, assessing how color and composition contribute to the overall quality and narrative.

**Use Cases:**
- **Content Enhancement:** Analyze composition and color to improve semi-professional video aesthetics.
- **Instagram/TikTok Optimization:** Evaluate visual appeal for social media sharing and engagement.
- **Seasonal Content:** Assess how color palettes enhance the mood of travel, festival, and celebration videos.

---

## 2. Detailed Summary Prompts

Detailed summaries provide an in-depth overview of the video's content, integrating both visual and audio elements. These prompts are designed to generate comprehensive narratives that capture the essence of the video.

### 2.1. **Comprehensive Narrative Integration**

```plaintext
"Create a detailed narrative that integrates both visual and audio elements of this video. Include key events, character interactions, and contextual information to provide a full understanding of the content. Duration: {duration:.1f} seconds\nTimeline:\n{timeline}\nAudio Transcript:\n{transcript}"
```

**Description:**  
This prompt encourages the generation of a thorough and cohesive storyline by combining visual actions with audio transcripts, ensuring a complete depiction of the video.

**Use Cases:**
- **Vlog Scripting:** Create narrative-driven scripts from personal footage for YouTube or TikTok.
- **Travel Guides:** Generate comprehensive travelogue content from raw destination footage.
- **Memory Documentation:** Develop detailed narratives of family events, celebrations, and adventures for archival or sharing.

---

### 2.2. **Thematic Analysis and Insights**

```plaintext
"Analyze the underlying themes and messages conveyed in this video. Discuss how visual elements and audio contribute to these themes, providing examples from specific moments in the video. Duration: {duration:.1f} seconds\nTimeline:\n{timeline}\nAudio Transcript:\n{transcript}"
```

**Description:**  
Focuses on identifying and explaining the core themes of the video, illustrating how different elements work together to communicate deeper messages.

**Use Cases:**
- **Personal Storytelling:** Identify underlying themes and narratives in travel, pet, and family footage.
- **Social Content Strategy:** Understand thematic elements for YouTube, Instagram, and TikTok content creation.
- **Event Recaps:** Extract key themes and moments from festivals, celebrations, and sports events.

---

### 2.3. **Character Development and Interaction**

```plaintext
"Provide a detailed summary of character development and interactions throughout this video. Highlight key moments that showcase character growth, relationships, and dynamics. Duration: {duration:.1f} seconds\nTimeline:\n{timeline}\nAudio Transcript:\n{transcript}"
```

**Description:**  
Aims to capture the evolution of characters and their relationships, offering insights into their development and interactions within the video.

**Use Cases:**
- **Pet Content:** Document pet personality development and interactions across multiple videos.
- **Family Documentation:** Track relationship dynamics and interactions during family gatherings and milestones.
- **Social Content Narratives:** Develop consistent character/persona storytelling across personal video series.

---

### 2.4. **Contextual Event Sequencing**

```plaintext
"Generate a detailed summary outlining the sequence of significant events in this video. Explain how each event leads to the next, providing context and connections between different parts of the video. Duration: {duration:.1f} seconds\nTimeline:\n{timeline}\nAudio Transcript:\n{transcript}"
```

**Description:**  
Encourages the model to map out the progression of events, ensuring a logical flow and contextual connections throughout the video's timeline.

**Use Cases:**
- **Travel Vlogs:** Document the sequence of destinations and activities for narrative flow.
- **Festival and Event Coverage:** Outline the progression of performances, activities, and highlights.
- **Pet Milestones:** Sequence significant behavioral moments and life events from pet footage.

---

### 2.5. **Multimodal Contextual Synthesis**

```plaintext
"Combine visual and audio data to synthesize a detailed summary that captures the full context of this video. Include descriptions of scenes, dialogues, and their interrelations to present a unified overview. Duration: {duration:.1f} seconds\nTimeline:\n{timeline}\nAudio Transcript:\n{transcript}"
```

**Description:**  
Focuses on merging both modalities—visual and audio—to create a unified and comprehensive summary that reflects the complete context of the video.

**Use Cases:**
- **Personal Video Archives:** Create detailed records of family events, travel, and celebrations for long-term preservation.
- **Social Media Captions:** Generate comprehensive descriptions for YouTube, Instagram, and other platforms.
- **Shareable Content:** Synthesize multi-modal summaries for sharing with friends and family members.

---

## 3. Brief Summary Prompts

Brief summaries offer concise overviews of the video's content, highlighting the main points without extensive detail. These prompts are ideal for quick insights and easy-to-digest information.

### 3.1. **Concise Content Overview**

```plaintext
"Provide a concise summary that highlights the main visual and audio elements of this video. Focus on key points and essential information to deliver a clear and brief understanding. Duration: {duration:.1f} seconds\nTimeline:\n{timeline}\nTranscript:\n{transcript}"
```

**Description:**  
Generates a short and clear summary that captures the essential aspects of the video, making it easy to grasp the primary content quickly.

**Use Cases:**
- **Social Media Posts:** Create quick captions for Instagram, TikTok, and YouTube videos.
- **Content Previews:** Generate brief descriptions for video collections and multi-video series.
- **Quick Sharing:** Provide concise summaries for quick messaging and social sharing with friends.

---

### 3.2. **Highlight Reel Summary**

```plaintext
"Summarize the key highlights of this video, focusing on the most impactful moments and main messages. Ensure the summary is easy to read and provides the complete context in a succinct manner. Duration: {duration:.1f} seconds\nTimeline:\n{timeline}\nTranscript:\n{transcript}"
```

**Description:**  
Emphasizes the most significant and impactful parts of the video, delivering a summary that highlights the core messages and memorable moments.

**Use Cases:**
- **Event Highlights:** Create reels of key moments from festivals, sports, and celebrations.
- **Pet Trick Compilation:** Highlight the best moments from pet videos for social sharing.
- **Travel Montages:** Summarize destination highlights for quick-view content series.

---

### 3.3. **Quick Insights Summary**

```plaintext
"Generate a quick summary that captures the primary insights and takeaways from this video. Focus on delivering information that provides immediate understanding without delving into extensive details. Duration: {duration:.1f} seconds\nTimeline:\n{timeline}\nTranscript:\n{transcript}"
```

**Description:**  
Aims to present the main insights and lessons from the video, ensuring that viewers can quickly grasp the fundamental points.

**Use Cases:**
- **Travel Insights:** Extract main discoveries and experiences from travel footage.
- **Pet Behavior Tips:** Summarize key behaviors and learnings from pet videos.
- **Content Curation:** Provide brief insights for personal video collections and themed playlists.

---

### 3.4. **Executive Brief Summary**

```plaintext
"Create an executive-level brief summary of this video, highlighting strategic points and critical information. Ensure the summary is succinct and tailored for decision-makers who need a high-level overview. Duration: {duration:.1f} seconds\nTimeline:\n{timeline}\nTranscript:\n{transcript}"
```

**Description:**  
Tailored for executives and decision-makers, this prompt focuses on delivering strategic and critical information in a brief format.

**Use Cases:**
- **Content Planning:** Summarize key moments from events for future video planning.
- **Memory Snapshots:** Create brief summaries of important family and celebration moments.
- **Social Media Strategy:** Identify key content angles from personal footage for posting strategy.

---

### 3.5. **Snapshot Summary**

```plaintext
"Provide a snapshot summary of this video, capturing the essence and main points in a few sentences. Ensure the summary is clear, direct, and easily understandable at a glance. Duration: {duration:.1f} seconds\nTimeline:\n{timeline}\nTranscript:\n{transcript}"
```

**Description:**  
Delivers a quick and clear snapshot of the video's content, ideal for situations where brevity and clarity are paramount.

**Use Cases:**
- **Video Tagging:** Create brief tags and descriptions for organizing personal video libraries.
- **Social Media Stories:** Generate quick summaries for Instagram Stories, Reels, and TikTok clips.
- **Message Sharing:** Provide concise one-liner summaries for sending video clips to friends and family.

---

## 4. Leveraging These Prompts with VideoAnalyzer

Integrate these innovative prompts into your **VideoAnalyzer** workflows to unlock deeper insights and build more sophisticated video analysis applications. Customize and combine these prompts to suit your specific project needs, whether you're developing interactive applications, enhancing content creation, or conducting advanced research.

### **Getting Started**

1. **Choose Appropriate Prompts:** Select prompts from the categories that best align with your analysis goals—Frame-Based Analysis, Detailed Summaries, or Brief Summaries.
2. **Customize if Needed:** Modify the prompts to better fit your specific use case or to include additional context.
3. **Integrate with VideoAnalyzer:** Utilize the prompts within your `AnalysisPrompts` configuration.
4. **Analyze and Iterate:** Run the analysis, review the results, and refine your prompts for optimal outcomes.

### **Example Integration**

Here's how you can integrate these prompts into your **VideoAnalyzer** setup:

```python
from video_analyzer.models import AnalysisPrompts
from video_analyzer.analyzer import OllamaVideoAnalyzer
from video_analyzer.frame_selectors import DynamicFrameSelector
import logging

# Define custom prompts
custom_prompts = AnalysisPrompts(
    frame_analysis="""
    "Analyze this frame by identifying all visible objects and their current states. Describe the actions being performed and the relationships between different objects."
    """,
    detailed_summary="""
    "Create a detailed narrative that integrates both visual and audio elements of this video. Include key events, character interactions, and contextual information to provide a full understanding of the content. Duration: {duration:.1f} seconds\nTimeline:\n{timeline}\nAudio Transcript:\n{transcript}"
    """,
    brief_summary="""
    "Provide a concise summary that highlights the main visual and audio elements of this video. Focus on key points and essential information to deliver a clear and brief understanding. Duration: {duration:.1f} seconds\nTimeline:\n{timeline}\nTranscript:\n{transcript}"
    """
)

# Initialize the video analyzer
analyzer = OllamaVideoAnalyzer(
    frame_analysis_model="ministral-3:3b",
    summary_model="ministral-3:14b",
    prompts=custom_prompts,
    min_frames=8,
    max_frames=32,
    frame_selector=DynamicFrameSelector(),
    frames_per_minute=8.0,
    log_level=logging.INFO
)

# Analyze the video
video_path = "path/to/your/video.mp4"
results = analyzer.analyze_video(video_path)

# Print the results
print("\nBrief Summary:")
print(results['brief_summary'])

print("\nDetailed Summary:")
print(results['summary'])

print("\nVideo Timeline:")
print(results['timeline'])

print("\nMetadata:")
for key, value in results['metadata'].items():
    print(f"{key}: {value}")
```

---

## 5. Best Practices

To maximize the effectiveness of your prompts within **VideoAnalyzer**, consider the following best practices:

### **5.1. Be Specific and Clear**

- **Clarity:** Ensure your prompts are unambiguous and clearly state the analysis objectives.
- **Focus:** Direct the model’s attention to specific elements or aspects of the video to obtain targeted insights.

### **5.2. Use Descriptive Language**

- **Detail:** Incorporate descriptive terms to enhance the model’s understanding and analysis depth.
- **Contextual Cues:** Provide context within prompts to guide the model in generating relevant and accurate outputs.

### **5.3. Integrate Tags Effectively**

- **Utilize `{timeline}`, `{duration}`, and `{transcript}`:** These tags provide essential context, enabling the model to reference specific parts of the video and audio content.
- **Consistency:** Maintain consistent usage of tags across prompts to ensure cohesive and contextually aware analyses.

### **5.4. Iterate and Refine**

- **Testing:** Continuously test and refine your prompts based on the analysis results to improve accuracy and relevance.
- **Feedback Loop:** Use the outputs to inform prompt adjustments, enhancing the quality of future analyses.

### **5.5. Combine Prompts for Comprehensive Insights**

- **Multiple Angles:** Use a combination of frame-based and summary prompts to gather multi-faceted insights.
- **Layered Analysis:** Start with frame analysis to extract detailed information, then use summary prompts to compile and contextualize the findings.

---

## 6. Conclusion

These innovative prompt examples are tailored to fit **VideoAnalyzer**'s `AnalysisPrompts` structure, providing specialized prompts for **Frame-Based Analysis**, **Detailed Summaries**, and **Brief Summaries**. By integrating these prompts into your workflows, you can unlock deeper insights and build more sophisticated video analysis applications. Customize and expand upon these examples to explore new possibilities and enhance your projects with intelligent video-centric solutions.
