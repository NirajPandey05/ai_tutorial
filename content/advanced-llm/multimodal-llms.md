# Multimodal LLMs

Multimodal LLMs can understand and generate content across multiple modalities - text, images, audio, and video. This lesson explores how these models work and how to use them.

## Learning Objectives

By the end of this lesson, you will:
- Understand how multimodal models process different input types
- Know the architectures used for vision-language models
- Be able to use multimodal APIs effectively
- Recognize use cases and limitations of multimodal LLMs

---

## What Are Multimodal LLMs?

### Beyond Text

Traditional LLMs work with text only:

```
Text Input → LLM → Text Output
```

Multimodal LLMs can handle multiple modalities:

```
┌─────────────────────────────────────────────────────────────┐
│                  Multimodal LLM                             │
│                                                             │
│   Image ─────────┐                                          │
│                  │                                          │
│   Text ──────────┼──→ Multimodal LLM ──→ Text/Image Output  │
│                  │                                          │
│   Audio ─────────┘                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Modality Types

| Modality | Input | Output | Examples |
|----------|-------|--------|----------|
| **Text** | ✓ | ✓ | Questions, responses |
| **Image** | ✓ | ✓* | Photos, diagrams, generated images |
| **Audio** | ✓ | ✓ | Speech, music |
| **Video** | ✓ | ✓* | Clips, animations |

*Output capability varies by model

---

## Major Multimodal Models

### GPT-4V / GPT-4 Vision (OpenAI)

```
┌─────────────────────────────────────────────────────────────┐
│                    GPT-4V Capabilities                      │
├─────────────────────────────────────────────────────────────┤
│  Input:   Text + Images (up to 20 images)                   │
│  Output:  Text only                                         │
│                                                             │
│  Strengths:                                                 │
│    • Detailed image understanding                           │
│    • OCR (text extraction from images)                      │
│    • Document analysis                                      │
│    • Code from screenshots                                  │
│    • Diagram interpretation                                 │
│                                                             │
│  Limitations:                                               │
│    • Cannot generate images                                 │
│    • Limited spatial reasoning                              │
│    • May miscount objects                                   │
└─────────────────────────────────────────────────────────────┘
```

### Claude 3.5 / Claude 4 Vision (Anthropic)

```
┌─────────────────────────────────────────────────────────────┐
│                 Claude Vision Capabilities                  │
├─────────────────────────────────────────────────────────────┤
│  Input:   Text + Images (up to 20 images)                   │
│  Output:  Text only                                         │
│                                                             │
│  Strengths:                                                 │
│    • Excellent chart/graph analysis                         │
│    • Strong document understanding                          │
│    • Good at technical diagrams                             │
│    • Careful, nuanced descriptions                          │
│                                                             │
│  Unique Feature:                                            │
│    • Computer Use capability (screenshots + actions)        │
└─────────────────────────────────────────────────────────────┘
```

### Gemini 2.0 / Gemini Pro Vision (Google)

```
┌─────────────────────────────────────────────────────────────┐
│                  Gemini Capabilities                        │
├─────────────────────────────────────────────────────────────┤
│  Input:   Text + Images + Video + Audio                     │
│  Output:  Text + Images (Gemini 2.0)                        │
│                                                             │
│  Strengths:                                                 │
│    • Native multimodal (not separate encoders)              │
│    • Video understanding (up to 1 hour)                     │
│    • Audio transcription + understanding                    │
│    • Longest context window (2M tokens)                     │
│                                                             │
│  Unique Feature:                                            │
│    • Truly multimodal architecture from training            │
└─────────────────────────────────────────────────────────────┘
```

### Comparison Table

| Model | Text In | Image In | Audio In | Video In | Image Out |
|-------|---------|----------|----------|----------|-----------|
| **GPT-4V** | ✓ | ✓ | ✓* | ✓* | ✗ |
| **GPT-4o** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Claude 3.5** | ✓ | ✓ | ✗ | ✗ | ✗ |
| **Gemini 2.0** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **LLaVA** | ✓ | ✓ | ✗ | ✗ | ✗ |

*Via Whisper API or preprocessing

---

## Architecture Patterns

### Pattern 1: Vision Encoder + LLM

Most common approach - add a vision encoder to an existing LLM:

```
┌─────────────────────────────────────────────────────────────┐
│              Vision Encoder + LLM                           │
│                                                             │
│   Image                                                     │
│     │                                                       │
│     ↓                                                       │
│   ┌───────────────┐                                         │
│   │ Vision Encoder│  (CLIP, ViT, SigLIP)                   │
│   │  (Frozen)     │                                         │
│   └───────┬───────┘                                         │
│           │ Image embeddings                                │
│           ↓                                                 │
│   ┌───────────────┐                                         │
│   │  Projection   │  (MLP, cross-attention)                │
│   │    Layer      │                                         │
│   └───────┬───────┘                                         │
│           │ Projected tokens                                │
│           ↓                                                 │
│   ┌───────────────────────────────────────────────┐        │
│   │              LLM Backbone                      │        │
│   │ [Image tokens] [User: describe] [Assistant:]   │        │
│   └───────────────────────────────────────────────┘        │
│                       │                                     │
│                       ↓                                     │
│                  Text Output                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Used by**: LLaVA, InternVL, Qwen-VL

### Pattern 2: Native Multimodal

Train from scratch with interleaved modalities:

```
┌─────────────────────────────────────────────────────────────┐
│              Native Multimodal                              │
│                                                             │
│   Mixed Input Sequence:                                     │
│   [Text] [Image] [Text] [Audio] [Text]                     │
│      │      │       │      │       │                       │
│      ↓      ↓       ↓      ↓       ↓                       │
│   ┌─────────────────────────────────────┐                  │
│   │   Unified Multimodal Transformer    │                  │
│   │   (Handles all modalities natively) │                  │
│   └─────────────────────────────────────┘                  │
│                       │                                     │
│                       ↓                                     │
│              Unified Output Space                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Used by**: Gemini (partially), GPT-4 (speculated)

### Pattern 3: Separate Encoders with Fusion

```
┌─────────────────────────────────────────────────────────────┐
│              Multi-Encoder Fusion                           │
│                                                             │
│   Image    Text     Audio                                   │
│     │        │        │                                     │
│     ↓        ↓        ↓                                     │
│  ┌─────┐  ┌─────┐  ┌─────┐                                 │
│  │ViT  │  │Text │  │Audio│                                 │
│  │Enc. │  │Enc. │  │Enc. │                                 │
│  └──┬──┘  └──┬──┘  └──┬──┘                                 │
│     │        │        │                                     │
│     └────────┼────────┘                                     │
│              ↓                                              │
│        ┌──────────┐                                         │
│        │  Fusion  │  (Cross-attention / Concat)            │
│        │  Module  │                                         │
│        └────┬─────┘                                         │
│             ↓                                               │
│     ┌──────────────┐                                        │
│     │     LLM      │                                        │
│     └──────────────┘                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Using Vision APIs

### OpenAI GPT-4V

```python
from openai import OpenAI
import base64

client = OpenAI()

# Method 1: URL
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.jpg",
                        "detail": "high"  # or "low", "auto"
                    }
                }
            ]
        }
    ],
    max_tokens=500
)

# Method 2: Base64
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")

image_data = encode_image("local_image.jpg")
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                }
            ]
        }
    ]
)
```

### Anthropic Claude Vision

```python
import anthropic
import base64

client = anthropic.Anthropic()

# Load and encode image
with open("image.jpg", "rb") as f:
    image_data = base64.standard_b64encode(f.read()).decode("utf-8")

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data,
                    },
                },
                {
                    "type": "text",
                    "text": "Describe what you see in this image."
                }
            ],
        }
    ],
)
```

### Google Gemini

```python
import google.generativeai as genai
from PIL import Image

genai.configure(api_key="your-api-key")
model = genai.GenerativeModel('gemini-2.0-flash')

# From file
image = Image.open('image.jpg')
response = model.generate_content([
    "What's in this image?",
    image
])

# From URL
import urllib.request
urllib.request.urlretrieve("https://example.com/image.jpg", "temp.jpg")
image = Image.open("temp.jpg")
response = model.generate_content(["Describe this:", image])

# Multiple images
images = [Image.open(f"image{i}.jpg") for i in range(3)]
response = model.generate_content([
    "Compare these images:",
    *images
])
```

---

## Common Vision Tasks

### 1. Image Description

```python
prompt = "Describe this image in detail, including colors, objects, and any text visible."
```

### 2. OCR / Text Extraction

```python
prompt = """Extract all text from this image. 
Format it preserving the original structure as much as possible."""
```

### 3. Document Analysis

```python
prompt = """Analyze this document image:
1. What type of document is this?
2. Extract key information (dates, names, amounts)
3. Summarize the main content"""
```

### 4. Chart/Graph Interpretation

```python
prompt = """Analyze this chart:
1. What type of chart is this?
2. What are the axes/labels?
3. What are the key trends or insights?
4. Provide the approximate data values"""
```

### 5. Code from Screenshot

```python
prompt = """Extract the code from this screenshot.
Return only the code, properly formatted."""
```

### 6. UI Analysis

```python
prompt = """Analyze this UI screenshot:
1. What application/website is this?
2. What elements are visible (buttons, forms, navigation)?
3. What actions can a user take?
4. Any UX observations?"""
```

---

## Multi-Image Reasoning

### Comparing Images

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Compare these two product images. Which one looks more professional and why?"},
            {"type": "image_url", "image_url": {"url": image1_url}},
            {"type": "image_url", "image_url": {"url": image2_url}},
        ]
    }
]
```

### Sequential Analysis

```python
# Analyze a process or timeline
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "These images show a before/after transformation. Describe what changed:"},
            {"type": "image_url", "image_url": {"url": before_url}},
            {"type": "image_url", "image_url": {"url": after_url}},
        ]
    }
]
```

### Finding Differences

```python
prompt = """These two images are nearly identical but have some differences.
Find and list all the differences between them."""
```

---

## Audio Integration

### Speech-to-Text with Context

```python
# OpenAI Whisper + GPT-4
from openai import OpenAI

client = OpenAI()

# Transcribe audio
with open("audio.mp3", "rb") as audio_file:
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="text"
    )

# Analyze with LLM
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are analyzing a transcribed conversation."},
        {"role": "user", "content": f"Summarize this transcript:\n\n{transcript}"}
    ]
)
```

### Native Audio (Gemini)

```python
import google.generativeai as genai

model = genai.GenerativeModel('gemini-2.0-flash')

# Upload audio file
audio_file = genai.upload_file("audio.mp3")

response = model.generate_content([
    "Transcribe this audio and summarize the main points:",
    audio_file
])
```

---

## Video Understanding

### Gemini Video Analysis

```python
import google.generativeai as genai
import time

model = genai.GenerativeModel('gemini-2.0-flash')

# Upload video (up to 1 hour)
video_file = genai.upload_file("video.mp4")

# Wait for processing
while video_file.state.name == "PROCESSING":
    time.sleep(10)
    video_file = genai.get_file(video_file.name)

# Analyze
response = model.generate_content([
    "Analyze this video:",
    "1. Provide a summary",
    "2. List key moments with timestamps",
    "3. Identify any text or speech",
    video_file
])
```

### Frame Extraction Approach

```python
import cv2
from openai import OpenAI

client = OpenAI()

def extract_frames(video_path, interval_seconds=5):
    """Extract frames at regular intervals"""
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_seconds)
    
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            _, buffer = cv2.imencode('.jpg', frame)
            frames.append(base64.b64encode(buffer).decode())
        count += 1
    
    cap.release()
    return frames

# Extract and analyze
frames = extract_frames("video.mp4", interval_seconds=10)

content = [{"type": "text", "text": "These frames are from a video. Describe what's happening:"}]
for frame in frames[:10]:  # Limit frames
    content.append({
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{frame}"}
    })

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": content}]
)
```

---

## Best Practices

### Image Quality

```
┌─────────────────────────────────────────────────────────────┐
│                 Image Quality Guidelines                    │
├─────────────────────────────────────────────────────────────┤
│  Resolution:    At least 512x512 for best results          │
│  Format:        JPEG, PNG, WebP, GIF supported             │
│  Size:          Usually under 20MB                          │
│  Clarity:       Avoid blurry or low-contrast images        │
│  Compression:   Light compression OK, heavy reduces quality│
└─────────────────────────────────────────────────────────────┘
```

### Prompting for Vision

```python
# Be specific about what to look for
good_prompt = """Analyze this product photo:
1. Identify the product category
2. Describe the main features visible
3. Note any text, labels, or branding
4. Rate the photo quality (1-10) with justification"""

# Avoid vague prompts
bad_prompt = "What do you see?"  # Too generic
```

### Handling Limitations

```python
# Vision models can struggle with:
limitations = [
    "Precise counting (especially crowds)",
    "Spatial relationships (left/right sometimes confused)",
    "Reading handwriting",
    "Small text in large images",
    "Understanding abstract art",
    "Medical/technical image diagnosis",
]

# Mitigation strategies
strategies = {
    "counting": "Ask to estimate ranges instead of exact counts",
    "small_text": "Crop to relevant region, increase resolution",
    "spatial": "Ask about specific objects, not relationships",
}
```

---

## Practical Use Cases

| Use Case | Best Model | Notes |
|----------|------------|-------|
| **Document OCR** | GPT-4V, Claude | High accuracy |
| **Photo Description** | Any | All perform well |
| **Chart Analysis** | Claude, Gemini | Claude excels at charts |
| **Code Extraction** | GPT-4V | Best code understanding |
| **Video Summary** | Gemini | Native video support |
| **Real-time Vision** | GPT-4o | Low latency |
| **Medical Images** | Specialized | General LLMs not reliable |

---

## Key Takeaways

1. **Multimodal is mainstream** - All major models now support vision

2. **Architecture varies** - Encoder + LLM vs native multimodal

3. **API patterns are similar** - Base64 or URL, mixed content arrays

4. **Use case matters** - Different models excel at different tasks

5. **Know the limitations** - Counting, spatial reasoning, specialized domains

6. **Quality in = Quality out** - Better images give better results

---

## What's Next?

You've learned about advanced LLM concepts! Next, explore **Fine-tuning LLMs** to customize models for your specific needs → [Fine-tuning Fundamentals](../fine-tuning/why-finetune.md)

---

## Quick Reference

| Task | Provider | API Pattern |
|------|----------|-------------|
| **Single Image** | All | image_url or base64 in content array |
| **Multiple Images** | All | Multiple image objects in content |
| **Video** | Gemini | Upload file, then reference |
| **Audio** | Gemini, OpenAI | Whisper for speech-to-text |
| **Detail Level** | OpenAI | `detail: "high"/"low"/"auto"` |
