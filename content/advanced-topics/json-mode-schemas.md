# JSON Mode and Structured Outputs

Master the art of getting consistent, parseable outputs from LLMs using JSON mode, schemas, and structured generation.

---

## The Problem with Unstructured Outputs

LLMs naturally produce free-form text, but applications need structured data:

```python
# What we often get
response = "The sentiment is positive with a confidence of about 85%"

# What we need
{"sentiment": "positive", "confidence": 0.85}
```

**Common issues with unstructured outputs:**
- Inconsistent formatting
- Missing fields
- Wrong data types
- Extra explanation text
- Parsing failures

---

## JSON Mode Basics

### OpenAI JSON Mode

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant. Always respond in valid JSON."
        },
        {
            "role": "user", 
            "content": "Analyze the sentiment of: 'I love this product!'"
        }
    ],
    response_format={"type": "json_object"}  # Enable JSON mode
)

import json
result = json.loads(response.choices[0].message.content)
print(result)
# {"sentiment": "positive", "confidence": 0.95, "emotion": "joy"}
```

**Important**: When using JSON mode, you MUST mention "JSON" in your prompt. The model won't know what structure to use otherwise.

### Anthropic JSON Output

```python
from anthropic import Anthropic

client = Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": """Analyze this text and respond ONLY with a JSON object:
            
Text: "The service was terrible and the food was cold."

Expected format:
{
    "sentiment": "positive" | "negative" | "neutral",
    "aspects": [{"aspect": string, "sentiment": string}],
    "overall_score": number between 0 and 1
}"""
        }
    ]
)

import json
# Claude may include markdown code blocks
content = response.content[0].text
if content.startswith("```"):
    content = content.split("```")[1]
    if content.startswith("json"):
        content = content[4:]
result = json.loads(content.strip())
```

### Google Gemini JSON Mode

```python
import google.generativeai as genai

genai.configure(api_key="your-key")

model = genai.GenerativeModel(
    'gemini-1.5-pro',
    generation_config={"response_mime_type": "application/json"}
)

response = model.generate_content("""
Extract entities from this text as JSON:
"Apple CEO Tim Cook announced the new iPhone in Cupertino."

Format: {"entities": [{"text": string, "type": string}]}
""")

import json
result = json.loads(response.text)
```

---

## Structured Outputs with Schemas

### OpenAI Structured Outputs (Recommended)

OpenAI's structured outputs guarantee schema compliance:

```python
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Optional
from enum import Enum

client = OpenAI()

# Define your schema with Pydantic
class Sentiment(str, Enum):
    positive = "positive"
    negative = "negative"
    neutral = "neutral"

class AspectSentiment(BaseModel):
    aspect: str
    sentiment: Sentiment
    confidence: float

class SentimentAnalysis(BaseModel):
    overall_sentiment: Sentiment
    confidence: float
    aspects: List[AspectSentiment]
    summary: str

# Use structured outputs
response = client.beta.chat.completions.parse(
    model="gpt-4o-2024-08-06",  # Requires recent model
    messages=[
        {
            "role": "system",
            "content": "Analyze the sentiment of the given text."
        },
        {
            "role": "user",
            "content": "The food was amazing but the service was slow."
        }
    ],
    response_format=SentimentAnalysis
)

# Guaranteed to match schema
result = response.choices[0].message.parsed
print(f"Overall: {result.overall_sentiment}")
print(f"Confidence: {result.confidence}")
for aspect in result.aspects:
    print(f"  - {aspect.aspect}: {aspect.sentiment} ({aspect.confidence})")
```

### JSON Schema Definition

For providers that don't support Pydantic directly:

```python
# Define JSON Schema manually
sentiment_schema = {
    "type": "object",
    "properties": {
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral"]
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1
        },
        "aspects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "aspect": {"type": "string"},
                    "sentiment": {"type": "string"}
                },
                "required": ["aspect", "sentiment"]
            }
        }
    },
    "required": ["sentiment", "confidence"]
}

# Include schema in prompt
prompt = f"""Analyze this review and respond with JSON matching this schema:
{json.dumps(sentiment_schema, indent=2)}

Review: "Great product, fast shipping, but packaging was damaged."
"""
```

---

## Advanced Schema Patterns

### Recursive Schemas

For nested or tree-like structures:

```python
from pydantic import BaseModel
from typing import List, Optional

class TreeNode(BaseModel):
    name: str
    value: Optional[str] = None
    children: List["TreeNode"] = []

# Enable forward references
TreeNode.model_rebuild()

class DocumentOutline(BaseModel):
    title: str
    sections: List[TreeNode]

# Use with OpenAI
response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "Create an outline for an article about AI safety"}
    ],
    response_format=DocumentOutline
)
```

### Union Types

Handle multiple possible output types:

```python
from pydantic import BaseModel
from typing import Union, Literal

class SuccessResult(BaseModel):
    status: Literal["success"]
    data: dict
    message: str

class ErrorResult(BaseModel):
    status: Literal["error"]
    error_code: str
    error_message: str

class APIResponse(BaseModel):
    result: Union[SuccessResult, ErrorResult]
    timestamp: str
    request_id: str
```

### Constrained Fields

Add validation constraints:

```python
from pydantic import BaseModel, Field, field_validator
from typing import List

class ProductReview(BaseModel):
    rating: int = Field(ge=1, le=5, description="Rating from 1 to 5")
    title: str = Field(max_length=100)
    pros: List[str] = Field(min_length=1, max_length=5)
    cons: List[str] = Field(max_length=5)
    recommendation: bool
    
    @field_validator('pros', 'cons')
    @classmethod
    def validate_items(cls, v):
        return [item.strip() for item in v if item.strip()]
```

---

## Multi-Step Structured Extraction

### Chain of Thought with Structure

Get reasoning AND structured output:

```python
from pydantic import BaseModel
from typing import List

class ReasoningStep(BaseModel):
    step_number: int
    description: str
    conclusion: str

class AnalysisResult(BaseModel):
    reasoning: List[ReasoningStep]
    final_answer: str
    confidence: float

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": """Analyze the problem step by step. 
            Show your reasoning in the 'reasoning' field before giving the final answer."""
        },
        {
            "role": "user",
            "content": "Should a startup use RAG or fine-tuning for their customer support chatbot?"
        }
    ],
    response_format=AnalysisResult
)

result = response.choices[0].message.parsed
print("Reasoning:")
for step in result.reasoning:
    print(f"  {step.step_number}. {step.description}")
    print(f"     → {step.conclusion}")
print(f"\nAnswer: {result.final_answer}")
print(f"Confidence: {result.confidence:.0%}")
```

### Extraction Pipeline

Multi-stage extraction for complex documents:

```python
from pydantic import BaseModel
from typing import List, Optional

# Stage 1: Extract raw entities
class RawEntity(BaseModel):
    text: str
    start_index: int
    end_index: int
    possible_types: List[str]

class EntityExtraction(BaseModel):
    entities: List[RawEntity]

# Stage 2: Classify and resolve entities
class ResolvedEntity(BaseModel):
    canonical_name: str
    entity_type: str
    mentions: List[str]
    confidence: float

class EntityResolution(BaseModel):
    resolved_entities: List[ResolvedEntity]
    relationships: List[dict]

# Stage 3: Extract structured facts
class Fact(BaseModel):
    subject: str
    predicate: str
    object: str
    confidence: float
    source_text: str

class KnowledgeGraph(BaseModel):
    facts: List[Fact]
    summary: str

async def extract_knowledge(document: str) -> KnowledgeGraph:
    # Stage 1
    entities = await client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"Extract entities from: {document}"}],
        response_format=EntityExtraction
    )
    
    # Stage 2
    resolved = await client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": f"Resolve these entities: {entities.choices[0].message.parsed.model_dump_json()}"}
        ],
        response_format=EntityResolution
    )
    
    # Stage 3
    knowledge = await client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": f"Extract facts from document using entities: {resolved.choices[0].message.parsed.model_dump_json()}\n\nDocument: {document}"}
        ],
        response_format=KnowledgeGraph
    )
    
    return knowledge.choices[0].message.parsed
```

---

## Handling Schema Errors

### Validation and Retry

```python
from pydantic import BaseModel, ValidationError
import json

class ExpectedOutput(BaseModel):
    name: str
    age: int
    email: str

def get_structured_output(prompt: str, max_retries: int = 3) -> ExpectedOutput:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            
            data = json.loads(response.choices[0].message.content)
            return ExpectedOutput(**data)
            
        except json.JSONDecodeError as e:
            print(f"Attempt {attempt + 1}: Invalid JSON - {e}")
            prompt += "\n\nIMPORTANT: Respond ONLY with valid JSON, no other text."
            
        except ValidationError as e:
            print(f"Attempt {attempt + 1}: Schema validation failed - {e}")
            prompt += f"\n\nThe response must match this schema exactly: {ExpectedOutput.model_json_schema()}"
    
    raise ValueError("Failed to get valid structured output after retries")
```

### Graceful Degradation

```python
from pydantic import BaseModel
from typing import Optional

class PartialExtraction(BaseModel):
    """Schema that allows partial results"""
    name: Optional[str] = None
    age: Optional[int] = None
    email: Optional[str] = None
    extraction_confidence: float = 0.0
    missing_fields: list[str] = []

def extract_with_fallback(text: str) -> PartialExtraction:
    try:
        # Try strict extraction first
        response = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "Extract person information. Set missing_fields for any data not found."
                },
                {"role": "user", "content": text}
            ],
            response_format=PartialExtraction
        )
        return response.choices[0].message.parsed
        
    except Exception as e:
        # Return partial result on failure
        return PartialExtraction(
            extraction_confidence=0.0,
            missing_fields=["name", "age", "email"]
        )
```

---

## Best Practices

### 1. Schema Design

```python
# ✅ Good: Clear, constrained schema
class GoodSchema(BaseModel):
    category: Literal["A", "B", "C"]  # Constrained choices
    score: float = Field(ge=0, le=1)  # Bounded range
    tags: List[str] = Field(max_length=10)  # Limited list

# ❌ Bad: Vague, unconstrained schema  
class BadSchema(BaseModel):
    category: str  # Any string
    score: float  # Any number
    tags: List[str]  # Unlimited list
```

### 2. Prompt Engineering for Schemas

```python
# ✅ Good: Clear instructions with examples
prompt = """Extract product information.

Example input: "iPhone 15 Pro - $999 - 256GB storage"
Example output: {"name": "iPhone 15 Pro", "price": 999, "storage_gb": 256}

Now extract from: "MacBook Air M3 - $1099 - 512GB SSD"
"""

# ❌ Bad: Vague instructions
prompt = "Get the product info from: MacBook Air M3 - $1099 - 512GB SSD"
```

### 3. Error Messages in Schema

```python
class RobustSchema(BaseModel):
    """Include helpful field descriptions"""
    
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="The overall emotional tone of the text"
    )
    confidence: float = Field(
        ge=0, le=1,
        description="How confident the model is (0=uncertain, 1=certain)"
    )
    key_phrases: List[str] = Field(
        description="Important phrases that influenced the sentiment classification",
        max_length=5
    )
```

---

## Summary

| Approach | Provider Support | Guarantee Level | Best For |
|----------|-----------------|-----------------|----------|
| JSON Mode | OpenAI, Gemini | Valid JSON only | Simple structures |
| Schema in Prompt | All providers | No guarantee | Compatibility |
| Structured Outputs | OpenAI | Schema compliance | Production apps |
| Pydantic + Validation | All (with retry) | Application-level | Complex validation |

**Key Takeaways:**
1. Use OpenAI's structured outputs for guaranteed schema compliance
2. Always include the schema or examples in your prompt
3. Implement retry logic for providers without native support
4. Design schemas with constraints to guide the model
5. Use Pydantic for validation and type safety

---

## Next Steps

- [Function Calling Best Practices](/learn/advanced-topics/structured-outputs/function-calling-practices) - When to use functions vs schemas
- [Pydantic Integration](/learn/advanced-topics/structured-outputs/pydantic-integration) - Deep dive into Pydantic with LLMs
- [Lab: Type-Safe Outputs](/learn/advanced-topics/structured-outputs/type-safe-lab) - Build a production extraction system
