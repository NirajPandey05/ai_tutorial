# Lab: Working with Multimodal Models

In this lab, you'll experiment with vision capabilities of LLMs, learning to analyze images, extract text, and combine visual and textual understanding.

## Lab Overview

| Aspect | Details |
|--------|---------|
| **Duration** | 30 minutes |
| **Difficulty** | Intermediate |
| **Prerequisites** | LLM Fundamentals, API Key configured |
| **Providers** | OpenAI (GPT-4o), Anthropic (Claude), Google (Gemini) |

---

## Learning Objectives

By completing this lab, you will:
- Send images to LLM APIs for analysis
- Compare vision capabilities across providers
- Build practical applications using image understanding
- Handle multiple images in a single request

---

## Setup

Make sure you have configured at least one API key that supports vision:
- **OpenAI**: GPT-4o, GPT-4-turbo (with vision)
- **Anthropic**: Claude 3.5 Sonnet, Claude 3 Opus
- **Google**: Gemini 2.0 Flash, Gemini 1.5 Pro

---

## Exercise 1: Basic Image Description

Let's start by sending an image to the model and getting a description.

### Try It: Describe an Image

```python
# Select your provider and paste an image URL
provider = "openai"  # or "anthropic" or "google"

# Sample image URLs for testing
sample_images = {
    "cityscape": "https://images.unsplash.com/photo-1480714378408-67cf0d13bc1b",
    "food": "https://images.unsplash.com/photo-1546069901-ba9599a7e63c",
    "chart": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/24hr_Climate_spiral_HadCRUT4.gif/220px-24hr_Climate_spiral_HadCRUT4.gif",
    "document": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Handwritten_draft_of_the_Gettysburg_Address.jpg/220px-Handwritten_draft_of_the_Gettysburg_Address.jpg"
}

image_url = sample_images["cityscape"]

prompt = """Describe this image in detail. Include:
1. Main subject/scene
2. Colors and lighting
3. Mood or atmosphere
4. Any text visible
5. Notable details"""
```

### Expected Output

The model should provide a comprehensive description covering all requested aspects.

---

## Exercise 2: OCR - Text Extraction

Vision models excel at extracting text from images.

### Try It: Extract Text

```python
# Use a document or screenshot image
image_url = "YOUR_DOCUMENT_IMAGE_URL"

prompt = """Extract all text visible in this image.
Preserve the original formatting and structure as much as possible.
If there are multiple sections, use headers to organize them."""
```

### Challenge: Receipt Parsing

```python
# Receipt parsing prompt
prompt = """This is a receipt image. Extract the following information:
- Store name
- Date and time
- List of items with prices
- Subtotal
- Tax
- Total
- Payment method

Format as JSON."""
```

---

## Exercise 3: Chart Analysis

Models can interpret charts and graphs, extracting data and insights.

### Try It: Analyze a Chart

```python
prompt = """Analyze this chart/graph:

1. **Chart Type**: What kind of visualization is this?
2. **Title/Labels**: What are the axes, legend, and title?
3. **Data Interpretation**: What story does the data tell?
4. **Key Insights**: What are the main takeaways?
5. **Data Extraction**: Can you estimate the main data points?

Be specific about any trends, patterns, or anomalies you observe."""
```

### Expected Capabilities

| Chart Type | Model Capability |
|------------|------------------|
| Bar charts | Excellent |
| Line graphs | Good |
| Pie charts | Good |
| Scatter plots | Moderate |
| Complex dashboards | Varies |

---

## Exercise 4: Multi-Image Comparison

Compare two or more images in a single request.

### Try It: Compare Images

```python
# Two images for comparison
image1_url = "https://example.com/product_a.jpg"
image2_url = "https://example.com/product_b.jpg"

prompt = """Compare these two images:

1. What are the similarities?
2. What are the key differences?
3. If these are products, which appears higher quality and why?
4. Rate each image on a scale of 1-10 for visual appeal."""
```

### Find the Differences Game

```python
prompt = """These two images are nearly identical but contain differences.
List every difference you can find between them.
Be specific about location and nature of each difference."""
```

---

## Exercise 5: Code from Screenshot

Extract and reconstruct code from images.

### Try It: Code Extraction

```python
prompt = """Extract the code from this screenshot.

1. Identify the programming language
2. Extract the complete code
3. Format it properly with correct indentation
4. Add comments if the code's purpose isn't clear
5. Note any parts that are unclear or cut off"""
```

### Code Review from Image

```python
prompt = """Review the code in this screenshot:

1. Extract the code
2. Identify any bugs or issues
3. Suggest improvements
4. Rate code quality (1-10)"""
```

---

## Exercise 6: Real-World Application - Product Analysis

Build a practical product analysis tool.

### Try It: Product Photo Analyzer

```python
prompt = """Analyze this product image for an e-commerce listing:

## Product Details
- **Category**: (identify the product type)
- **Key Features**: (list visible features)
- **Material/Quality**: (assess from visual cues)
- **Condition**: (new, used, damaged?)

## Marketing Analysis
- **Target Audience**: (who would buy this?)
- **Suggested Price Range**: (based on perceived quality)
- **Selling Points**: (what to highlight in listing)

## Photo Quality Assessment
- **Lighting**: (good/poor/adequate)
- **Background**: (professional/amateur)
- **Composition**: (suggestions for improvement)
- **Overall Score**: (1-10)

## Suggested Tags
(list relevant keywords for searchability)"""
```

---

## Exercise 7: Provider Comparison

Compare how different providers analyze the same image.

### Comparison Experiment

```python
# Use the same image and prompt across providers
test_image = "https://images.unsplash.com/photo-1551782450-a2132b4ba21d"  # Food image

comparison_prompt = """Analyze this food image:
1. Identify the dish
2. List visible ingredients  
3. Estimate calorie range
4. Rate presentation (1-10)
5. Suggest improvements"""

# Compare results from:
# - OpenAI GPT-4o
# - Anthropic Claude 3.5
# - Google Gemini 2.0

# Note differences in:
# - Detail level
# - Accuracy
# - Response style
# - Specific observations
```

### Comparison Table Template

| Aspect | OpenAI | Anthropic | Google |
|--------|--------|-----------|--------|
| Detail Level | | | |
| Accuracy | | | |
| Response Time | | | |
| Unique Insights | | | |

---

## Exercise 8: Building a Vision Pipeline

Create a multi-step vision analysis pipeline.

### Document Processing Pipeline

```python
# Step 1: Classify document type
classification_prompt = """What type of document is this?
- Invoice
- Receipt  
- Contract
- Letter
- Form
- Other

Just respond with the type."""

# Step 2: Based on type, extract relevant information
extraction_prompts = {
    "Invoice": """Extract from this invoice:
        - Invoice number
        - Date
        - Vendor name
        - Line items
        - Total amount
        - Due date""",
    
    "Receipt": """Extract from this receipt:
        - Store name
        - Date/time
        - Items purchased
        - Payment method
        - Total""",
    
    "Form": """Extract from this form:
        - Form type
        - All filled fields
        - Any signatures
        - Date"""
}

# Step 3: Validate and structure output
validation_prompt = """Validate this extracted data:
{extracted_data}

Check for:
- Missing required fields
- Data format issues
- Potential OCR errors

Return corrected JSON."""
```

---

## Challenge: Build a Visual Assistant

Create a conversational visual assistant that can:
1. Remember previous images discussed
2. Answer follow-up questions about images
3. Compare new images to previously discussed ones

### Starter Code

```python
class VisualAssistant:
    def __init__(self):
        self.conversation_history = []
        self.image_context = []
    
    def add_image(self, image_url, description_request):
        """Add an image to context and get description"""
        pass
    
    def ask_about_images(self, question):
        """Ask a question about previously shared images"""
        pass
    
    def compare_images(self, image1_idx, image2_idx, comparison_aspect):
        """Compare two images from the conversation"""
        pass
```

---

## Solutions & Tips

### Common Issues

| Issue | Solution |
|-------|----------|
| "Cannot access image URL" | Use base64 encoding instead |
| "Image too large" | Resize or compress before sending |
| "Inaccurate text extraction" | Crop to relevant region, increase contrast |
| "Wrong object count" | Ask for ranges instead of exact counts |

### Performance Tips

```python
# Tip 1: Use appropriate detail level (OpenAI)
"detail": "low"   # Faster, cheaper, for simple images
"detail": "high"  # Better accuracy, more expensive
"detail": "auto"  # Let the model decide

# Tip 2: Crop images for focus
# Instead of sending a full screenshot, crop to the relevant area

# Tip 3: Guide the model
# Be specific about what you want analyzed
bad_prompt = "What's this?"
good_prompt = "This is a product photo. Identify the brand, model, and key features."
```

---

## Key Takeaways

1. **Vision capabilities vary** - Test across providers for your use case

2. **Specificity matters** - Detailed prompts get detailed responses

3. **Images + text = power** - Combine context for best results

4. **Know limitations** - Counting, spatial reasoning, and specialized domains can be challenging

5. **Build pipelines** - Multi-step analysis often beats single prompts

---

## Next Steps

Now that you've mastered multimodal basics, explore:
- [Fine-tuning LLMs](../fine-tuning/why-finetune.md) - Customize models for specific tasks
- [RAG with Images](../rag/multimodal-rag.md) - Combine retrieval with vision
- [Building Vision Agents](../agents/vision-agents.md) - Autonomous visual understanding
