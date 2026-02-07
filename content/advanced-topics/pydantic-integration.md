# Pydantic Integration for LLM Outputs

Build type-safe, validated LLM applications using Pydantic models for input validation, output parsing, and seamless integration with modern Python type hints.

---

## Why Pydantic for LLMs?

Pydantic provides:

1. **Type Safety** - Catch errors at parse time, not runtime
2. **Automatic Validation** - Ensure LLM outputs match expectations
3. **IDE Support** - Full autocomplete and type checking
4. **Serialization** - Easy JSON conversion for API calls
5. **Documentation** - Self-documenting schemas

```python
from pydantic import BaseModel, Field
from typing import List, Optional

# Define expected output structure
class SentimentResult(BaseModel):
    sentiment: str
    confidence: float
    keywords: List[str]

# Parse and validate LLM output
result = SentimentResult.model_validate_json(llm_output)

# Now you have type-safe access
print(result.sentiment)  # IDE knows this is a string
print(result.confidence)  # IDE knows this is a float
```

---

## Basic Pydantic Models for LLMs

### Simple Extraction Model

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date

class PersonInfo(BaseModel):
    """Extract person information from text"""
    
    name: str = Field(description="Full name of the person")
    age: Optional[int] = Field(default=None, description="Age if mentioned")
    occupation: Optional[str] = Field(default=None, description="Job or profession")
    location: Optional[str] = Field(default=None, description="City or country")

class ContactInfo(BaseModel):
    """Extract contact details"""
    
    email: Optional[str] = Field(default=None, pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    phone: Optional[str] = Field(default=None)
    website: Optional[str] = Field(default=None)

class ExtractedPerson(BaseModel):
    """Complete person extraction result"""
    
    person: PersonInfo
    contact: ContactInfo
    raw_text: str
    extraction_confidence: float = Field(ge=0, le=1)
```

### Classification Model

```python
from pydantic import BaseModel, Field
from typing import Literal, List
from enum import Enum

class Priority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"

class Category(str, Enum):
    bug = "bug"
    feature = "feature"
    question = "question"
    documentation = "documentation"

class TicketClassification(BaseModel):
    """Classify a support ticket"""
    
    category: Category
    priority: Priority
    estimated_effort_hours: float = Field(ge=0, le=100)
    tags: List[str] = Field(max_length=5)
    requires_escalation: bool
    suggested_assignee: Optional[str] = None
    summary: str = Field(max_length=200)
```

---

## Advanced Validation

### Custom Validators

```python
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List

class ProductReview(BaseModel):
    rating: int
    title: str
    pros: List[str]
    cons: List[str]
    recommendation: bool
    
    @field_validator('rating')
    @classmethod
    def validate_rating(cls, v):
        if not 1 <= v <= 5:
            raise ValueError('Rating must be between 1 and 5')
        return v
    
    @field_validator('title')
    @classmethod
    def validate_title(cls, v):
        if len(v) < 5:
            raise ValueError('Title must be at least 5 characters')
        return v.strip().title()
    
    @field_validator('pros', 'cons')
    @classmethod
    def validate_points(cls, v):
        # Clean up each point
        return [point.strip() for point in v if point.strip()]
    
    @model_validator(mode='after')
    def validate_consistency(self):
        # Rating should match recommendation
        if self.rating >= 4 and not self.recommendation:
            # Allow but flag it
            pass
        if self.rating <= 2 and self.recommendation:
            # This is suspicious
            pass
        return self

# Usage with LLM
response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": f"Analyze this review: {review_text}"}],
    response_format=ProductReview
)
```

### Computed Fields

```python
from pydantic import BaseModel, computed_field
from typing import List

class AnalysisResult(BaseModel):
    text: str
    word_count: int
    sentiment_scores: List[float]
    
    @computed_field
    @property
    def average_sentiment(self) -> float:
        if not self.sentiment_scores:
            return 0.0
        return sum(self.sentiment_scores) / len(self.sentiment_scores)
    
    @computed_field
    @property
    def sentiment_label(self) -> str:
        avg = self.average_sentiment
        if avg > 0.6:
            return "positive"
        elif avg < 0.4:
            return "negative"
        return "neutral"
    
    @computed_field
    @property
    def words_per_sentence(self) -> float:
        sentences = self.text.count('.') + self.text.count('!') + self.text.count('?')
        if sentences == 0:
            return float(self.word_count)
        return self.word_count / sentences
```

---

## Nested and Complex Models

### Recursive Structures

```python
from pydantic import BaseModel
from typing import List, Optional

class MenuItem(BaseModel):
    """Recursive menu structure"""
    name: str
    description: Optional[str] = None
    url: Optional[str] = None
    children: List["MenuItem"] = []

# Required for forward references
MenuItem.model_rebuild()

class WebsiteStructure(BaseModel):
    title: str
    navigation: List[MenuItem]
    footer_links: List[MenuItem]
```

### Polymorphic Types

```python
from pydantic import BaseModel, Field
from typing import Union, Literal, List

class TextContent(BaseModel):
    type: Literal["text"] = "text"
    content: str
    format: Literal["plain", "markdown", "html"] = "plain"

class ImageContent(BaseModel):
    type: Literal["image"] = "image"
    url: str
    alt_text: str
    width: Optional[int] = None
    height: Optional[int] = None

class CodeContent(BaseModel):
    type: Literal["code"] = "code"
    language: str
    code: str
    filename: Optional[str] = None

class TableContent(BaseModel):
    type: Literal["table"] = "table"
    headers: List[str]
    rows: List[List[str]]

# Union type for any content
ContentBlock = Union[TextContent, ImageContent, CodeContent, TableContent]

class Document(BaseModel):
    title: str
    author: str
    blocks: List[ContentBlock]

# The LLM can return any mix of content types
example = Document(
    title="My Doc",
    author="AI",
    blocks=[
        TextContent(content="Introduction"),
        CodeContent(language="python", code="print('hello')"),
        TableContent(headers=["A", "B"], rows=[["1", "2"]])
    ]
)
```

---

## Integration Patterns

### OpenAI Structured Outputs

```python
from openai import OpenAI
from pydantic import BaseModel
from typing import List

client = OpenAI()

class Task(BaseModel):
    title: str
    priority: int
    due_date: Optional[str]
    subtasks: List[str]

class TaskList(BaseModel):
    tasks: List[Task]
    summary: str

def extract_tasks(text: str) -> TaskList:
    response = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": "Extract tasks from the given text."
            },
            {"role": "user", "content": text}
        ],
        response_format=TaskList
    )
    
    return response.choices[0].message.parsed
```

### LangChain Integration

```python
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List

class MovieReview(BaseModel):
    title: str = Field(description="Movie title")
    rating: float = Field(description="Rating out of 10")
    genres: List[str] = Field(description="Movie genres")
    summary: str = Field(description="Brief review summary")

# Create parser
parser = PydanticOutputParser(pydantic_object=MovieReview)

# Create prompt with format instructions
prompt = ChatPromptTemplate.from_messages([
    ("system", "Analyze the movie review and extract information.\n{format_instructions}"),
    ("user", "{review}")
])

# Create chain
llm = ChatOpenAI(model="gpt-4o")
chain = prompt | llm | parser

# Run
result = chain.invoke({
    "review": "Inception is a mind-bending thriller...",
    "format_instructions": parser.get_format_instructions()
})

print(result.title)  # Type-safe access
```

### Instructor Library

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel
from typing import List

# Patch the client
client = instructor.from_openai(OpenAI())

class User(BaseModel):
    name: str
    age: int
    email: str

# Direct Pydantic extraction
user = client.chat.completions.create(
    model="gpt-4o",
    response_model=User,
    messages=[
        {"role": "user", "content": "John Doe is 30 years old, email: john@example.com"}
    ]
)

print(user.name)  # "John Doe"
print(user.age)   # 30
```

### Instructor with Validation Retries

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel, field_validator
from typing import List

client = instructor.from_openai(OpenAI())

class ValidatedUser(BaseModel):
    name: str
    age: int
    email: str
    
    @field_validator('age')
    @classmethod
    def validate_age(cls, v):
        if v < 0 or v > 150:
            raise ValueError('Age must be between 0 and 150')
        return v
    
    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v.lower()

# Instructor will automatically retry on validation failure
user = client.chat.completions.create(
    model="gpt-4o",
    response_model=ValidatedUser,
    max_retries=3,  # Retry up to 3 times on validation failure
    messages=[
        {"role": "user", "content": "Extract: John Doe, age thirty, john.doe@example.com"}
    ]
)
```

---

## Schema Generation

### Generating JSON Schema

```python
from pydantic import BaseModel
from typing import List, Optional
import json

class Product(BaseModel):
    name: str
    price: float
    category: str
    tags: List[str]
    description: Optional[str] = None

# Get JSON schema
schema = Product.model_json_schema()
print(json.dumps(schema, indent=2))

# Output:
# {
#   "type": "object",
#   "properties": {
#     "name": {"type": "string"},
#     "price": {"type": "number"},
#     "category": {"type": "string"},
#     "tags": {"type": "array", "items": {"type": "string"}},
#     "description": {"anyOf": [{"type": "string"}, {"type": "null"}]}
#   },
#   "required": ["name", "price", "category", "tags"]
# }
```

### Dynamic Schema from Config

```python
from pydantic import create_model
from typing import Any, Dict, List

def create_extraction_model(fields: Dict[str, Dict[str, Any]]) -> type:
    """Dynamically create a Pydantic model from field definitions"""
    
    field_definitions = {}
    
    for field_name, config in fields.items():
        field_type = config.get('type', str)
        default = config.get('default', ...)
        description = config.get('description', '')
        
        field_definitions[field_name] = (
            field_type,
            Field(default=default, description=description)
        )
    
    return create_model('DynamicExtraction', **field_definitions)

# Usage
fields = {
    "company_name": {"type": str, "description": "Name of the company"},
    "revenue": {"type": float, "default": None, "description": "Annual revenue"},
    "employees": {"type": int, "default": None, "description": "Number of employees"}
}

CompanyExtraction = create_extraction_model(fields)

# Use with LLM
response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Apple Inc reported $383B revenue with 164,000 employees"}],
    response_format=CompanyExtraction
)
```

---

## Error Handling

### Parsing Failures

```python
from pydantic import BaseModel, ValidationError
import json

class StrictOutput(BaseModel):
    name: str
    value: int

def safe_parse(llm_output: str) -> StrictOutput | dict:
    """Safely parse LLM output with fallback"""
    
    try:
        # Try to parse JSON
        data = json.loads(llm_output)
    except json.JSONDecodeError as e:
        return {
            "error": "invalid_json",
            "message": str(e),
            "raw_output": llm_output
        }
    
    try:
        # Try to validate against schema
        return StrictOutput.model_validate(data)
    except ValidationError as e:
        return {
            "error": "validation_failed",
            "message": str(e),
            "parsed_json": data
        }
```

### Partial Parsing with Defaults

```python
from pydantic import BaseModel, Field
from typing import Optional, Any
import json

class FlexibleExtraction(BaseModel):
    """Model that allows partial extraction"""
    
    name: Optional[str] = None
    age: Optional[int] = None
    email: Optional[str] = None
    
    # Track extraction quality
    extraction_complete: bool = False
    missing_fields: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    
    @classmethod
    def from_llm_output(cls, output: str) -> "FlexibleExtraction":
        """Parse with graceful handling of missing fields"""
        
        try:
            data = json.loads(output)
        except json.JSONDecodeError:
            return cls(
                extraction_complete=False,
                missing_fields=["name", "age", "email"],
                confidence=0.0
            )
        
        # Check which fields are present
        missing = []
        for field in ["name", "age", "email"]:
            if field not in data or data[field] is None:
                missing.append(field)
        
        return cls(
            name=data.get("name"),
            age=data.get("age"),
            email=data.get("email"),
            extraction_complete=len(missing) == 0,
            missing_fields=missing,
            confidence=1.0 - (len(missing) * 0.33)
        )
```

---

## Best Practices

### 1. Use Descriptive Field Names and Descriptions

```python
# ✅ Good
class GoodModel(BaseModel):
    customer_full_name: str = Field(
        description="The customer's complete legal name"
    )
    order_total_usd: float = Field(
        ge=0,
        description="Total order amount in US dollars"
    )

# ❌ Bad
class BadModel(BaseModel):
    n: str
    t: float
```

### 2. Add Examples

```python
from pydantic import BaseModel, Field
from typing import List

class ProductSearch(BaseModel):
    """Search query for products"""
    
    query: str = Field(
        description="Search terms",
        examples=["wireless headphones", "laptop under $1000"]
    )
    categories: List[str] = Field(
        description="Category filters",
        examples=[["electronics", "audio"], ["computers"]]
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "bluetooth speaker",
                    "categories": ["electronics", "audio"]
                }
            ]
        }
    }
```

### 3. Constrain Values Appropriately

```python
from pydantic import BaseModel, Field
from typing import Literal

class ConstrainedModel(BaseModel):
    # Use Literal for fixed choices
    status: Literal["pending", "approved", "rejected"]
    
    # Use Field constraints for ranges
    confidence: float = Field(ge=0, le=1)
    count: int = Field(ge=0, le=1000)
    
    # Use max_length for strings
    summary: str = Field(max_length=500)
    
    # Use pattern for format validation
    product_code: str = Field(pattern=r'^[A-Z]{3}-\d{4}$')
```

### 4. Document Edge Cases

```python
class RobustExtraction(BaseModel):
    """
    Extract entity information from text.
    
    Edge cases:
    - If no entity is found, return empty string for name
    - If confidence cannot be determined, default to 0.5
    - Multiple entities should be the most prominent one
    """
    
    name: str = Field(
        default="",
        description="Entity name, empty string if not found"
    )
    confidence: float = Field(
        default=0.5,
        ge=0, le=1,
        description="Extraction confidence, defaults to 0.5 if uncertain"
    )
```

---

## Summary

| Feature | Use Case | Example |
|---------|----------|---------|
| `Field()` | Validation constraints | `Field(ge=0, le=100)` |
| `@field_validator` | Custom validation | Normalize strings, check ranges |
| `@model_validator` | Cross-field validation | Ensure consistency |
| `Optional[]` | Nullable fields | `Optional[str] = None` |
| `Literal[]` | Fixed choices | `Literal["a", "b", "c"]` |
| `model_json_schema()` | Schema generation | For prompt engineering |
| `model_validate_json()` | Safe parsing | Parse LLM output |

**Key Takeaways:**

1. Define clear schemas before calling LLMs
2. Use constraints to guide model outputs
3. Add descriptions for better model understanding
4. Implement validation for production reliability
5. Handle parsing failures gracefully

---

## Next Steps

- [Lab: Type-Safe Outputs](/learn/advanced-topics/structured-outputs/type-safe-lab) - Build a complete extraction pipeline
- [Guardrails & Safety](/learn/advanced-topics/guardrails/prompt-injection) - Validate outputs for safety
