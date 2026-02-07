# Lab: Building Type-Safe LLM Outputs

In this hands-on lab, you'll build a production-ready extraction system that uses Pydantic models to ensure type-safe, validated outputs from LLMs.

---

## Learning Objectives

By the end of this lab, you will:
- Create robust Pydantic models for LLM outputs
- Implement extraction with automatic validation
- Build retry logic for handling failures
- Create a multi-stage extraction pipeline
- Handle edge cases gracefully

---

## Prerequisites

- Python 3.10+
- OpenAI API key
- Basic understanding of Pydantic

```bash
pip install openai pydantic instructor
```

---

## Part 1: Basic Type-Safe Extraction

### Step 1: Define Your Models

```python
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from enum import Enum
from datetime import date

class Priority(str, Enum):
    """Task priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class TaskStatus(str, Enum):
    """Task status values"""
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    DONE = "done"

class ExtractedTask(BaseModel):
    """A single task extracted from text"""
    
    title: str = Field(
        min_length=3,
        max_length=100,
        description="Clear, actionable task title"
    )
    description: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Additional context or details"
    )
    priority: Priority = Field(
        default=Priority.MEDIUM,
        description="Task priority level"
    )
    due_date: Optional[str] = Field(
        default=None,
        description="Due date in YYYY-MM-DD format"
    )
    assignee: Optional[str] = Field(
        default=None,
        description="Person responsible for the task"
    )
    tags: List[str] = Field(
        default_factory=list,
        max_length=5,
        description="Relevant tags or categories"
    )
    
    @field_validator('title')
    @classmethod
    def clean_title(cls, v: str) -> str:
        """Ensure title is properly formatted"""
        return v.strip().capitalize()
    
    @field_validator('due_date')
    @classmethod
    def validate_date_format(cls, v: Optional[str]) -> Optional[str]:
        """Validate date format if provided"""
        if v is None:
            return None
        try:
            date.fromisoformat(v)
            return v
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Use YYYY-MM-DD")
    
    @field_validator('tags')
    @classmethod
    def clean_tags(cls, v: List[str]) -> List[str]:
        """Clean and deduplicate tags"""
        cleaned = [tag.lower().strip() for tag in v if tag.strip()]
        return list(dict.fromkeys(cleaned))  # Preserve order, remove duplicates

class TaskExtractionResult(BaseModel):
    """Complete extraction result with metadata"""
    
    tasks: List[ExtractedTask]
    source_text: str
    extraction_confidence: float = Field(ge=0, le=1)
    warnings: List[str] = Field(default_factory=list)
```

### Step 2: Create the Extractor

```python
from openai import OpenAI
import json

client = OpenAI()

def extract_tasks(text: str) -> TaskExtractionResult:
    """Extract tasks from text with type-safe output"""
    
    response = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {
                "role": "system",
                "content": """You are a task extraction assistant. Extract actionable tasks from the given text.

Guidelines:
- Each task should be clear and actionable
- Infer priority from urgency words (ASAP, urgent, critical = urgent/high)
- Parse dates mentioned (tomorrow, next week, March 15th, etc.)
- Identify assignees if mentioned by name
- Add relevant tags based on task content
- Set extraction_confidence based on how clear the tasks were

If no clear tasks are found, return an empty tasks list with appropriate confidence."""
            },
            {"role": "user", "content": text}
        ],
        response_format=TaskExtractionResult
    )
    
    result = response.choices[0].message.parsed
    result.source_text = text  # Add source text
    
    return result

# Test it
text = """
Meeting notes from today:
- John needs to finish the API documentation by Friday (high priority)
- Sarah should review the pull request for the auth module ASAP
- We need to schedule a design review next week
- Don't forget to update the README with the new installation steps
"""

result = extract_tasks(text)
print(f"Found {len(result.tasks)} tasks:")
for task in result.tasks:
    print(f"  - [{task.priority.value}] {task.title}")
    if task.assignee:
        print(f"    Assignee: {task.assignee}")
    if task.due_date:
        print(f"    Due: {task.due_date}")
```

---

## Part 2: Multi-Stage Extraction Pipeline

### Step 1: Define Pipeline Models

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

# Stage 1: Entity Extraction
class EntityType(str, Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    PRODUCT = "product"
    MONEY = "money"

class RawEntity(BaseModel):
    """Raw entity extracted from text"""
    text: str
    entity_type: EntityType
    start_position: int
    end_position: int
    confidence: float = Field(ge=0, le=1)

class EntityExtractionResult(BaseModel):
    """Stage 1 result"""
    entities: List[RawEntity]
    
# Stage 2: Entity Resolution
class ResolvedEntity(BaseModel):
    """Entity with resolved references"""
    canonical_name: str
    entity_type: EntityType
    aliases: List[str] = Field(default_factory=list)
    attributes: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0, le=1)

class EntityResolutionResult(BaseModel):
    """Stage 2 result"""
    resolved_entities: List[ResolvedEntity]
    coreference_chains: Dict[str, List[str]] = Field(default_factory=dict)

# Stage 3: Relationship Extraction
class RelationshipType(str, Enum):
    WORKS_FOR = "works_for"
    LOCATED_IN = "located_in"
    OWNS = "owns"
    BOUGHT = "bought"
    SOLD = "sold"
    PARTNERED_WITH = "partnered_with"
    FOUNDED = "founded"
    INVESTED_IN = "invested_in"

class Relationship(BaseModel):
    """Extracted relationship between entities"""
    subject: str
    relationship: RelationshipType
    object: str
    confidence: float = Field(ge=0, le=1)
    evidence: str = Field(description="Quote from text supporting this relationship")

class RelationshipExtractionResult(BaseModel):
    """Stage 3 result"""
    relationships: List[Relationship]

# Final Combined Result
class KnowledgeExtractionResult(BaseModel):
    """Complete knowledge extraction"""
    entities: List[ResolvedEntity]
    relationships: List[Relationship]
    summary: str
    extraction_quality: float = Field(ge=0, le=1)
```

### Step 2: Build the Pipeline

```python
from openai import OpenAI
from typing import Tuple
import asyncio

client = OpenAI()

class KnowledgeExtractionPipeline:
    """Multi-stage extraction pipeline"""
    
    def __init__(self, model: str = "gpt-4o-2024-08-06"):
        self.model = model
        self.client = OpenAI()
    
    async def extract_entities(self, text: str) -> EntityExtractionResult:
        """Stage 1: Extract raw entities"""
        
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": """Extract all named entities from the text.
                    Mark the character positions where each entity appears.
                    Include confidence scores for each extraction."""
                },
                {"role": "user", "content": text}
            ],
            response_format=EntityExtractionResult
        )
        
        return response.choices[0].message.parsed
    
    async def resolve_entities(
        self, 
        text: str, 
        raw_entities: EntityExtractionResult
    ) -> EntityResolutionResult:
        """Stage 2: Resolve entity references"""
        
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": """Resolve the extracted entities:
                    1. Merge duplicate entities (e.g., "Tim Cook", "Cook", "Apple's CEO")
                    2. Determine canonical names
                    3. Track coreference chains (which mentions refer to the same entity)
                    4. Extract any additional attributes mentioned"""
                },
                {
                    "role": "user", 
                    "content": f"Text: {text}\n\nRaw entities: {raw_entities.model_dump_json()}"
                }
            ],
            response_format=EntityResolutionResult
        )
        
        return response.choices[0].message.parsed
    
    async def extract_relationships(
        self,
        text: str,
        resolved_entities: EntityResolutionResult
    ) -> RelationshipExtractionResult:
        """Stage 3: Extract relationships between entities"""
        
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": """Extract relationships between the resolved entities.
                    Only extract relationships that are explicitly stated or strongly implied.
                    Include the evidence (quote) for each relationship."""
                },
                {
                    "role": "user",
                    "content": f"Text: {text}\n\nEntities: {resolved_entities.model_dump_json()}"
                }
            ],
            response_format=RelationshipExtractionResult
        )
        
        return response.choices[0].message.parsed
    
    async def run(self, text: str) -> KnowledgeExtractionResult:
        """Run the complete pipeline"""
        
        # Stage 1
        raw_entities = await self.extract_entities(text)
        print(f"Stage 1: Found {len(raw_entities.entities)} raw entities")
        
        # Stage 2
        resolved = await self.resolve_entities(text, raw_entities)
        print(f"Stage 2: Resolved to {len(resolved.resolved_entities)} entities")
        
        # Stage 3
        relationships = await self.extract_relationships(text, resolved)
        print(f"Stage 3: Found {len(relationships.relationships)} relationships")
        
        # Combine results
        return KnowledgeExtractionResult(
            entities=resolved.resolved_entities,
            relationships=relationships.relationships,
            summary=f"Extracted {len(resolved.resolved_entities)} entities and {len(relationships.relationships)} relationships",
            extraction_quality=sum(e.confidence for e in resolved.resolved_entities) / max(len(resolved.resolved_entities), 1)
        )

# Run the pipeline
async def main():
    pipeline = KnowledgeExtractionPipeline()
    
    text = """
    Apple CEO Tim Cook announced today that the company is investing $1 billion 
    in a new research facility in Austin, Texas. The facility will focus on 
    artificial intelligence and will employ over 5,000 people. Cook mentioned 
    that this investment is part of Apple's commitment to the United States, 
    following their previous $430 billion investment pledge. The Austin campus 
    will complement Apple's existing headquarters in Cupertino, California.
    """
    
    result = await pipeline.run(text)
    
    print("\n=== Extraction Results ===")
    print(f"\nEntities ({len(result.entities)}):")
    for entity in result.entities:
        print(f"  - {entity.canonical_name} ({entity.entity_type.value})")
        if entity.aliases:
            print(f"    Aliases: {', '.join(entity.aliases)}")
    
    print(f"\nRelationships ({len(result.relationships)}):")
    for rel in result.relationships:
        print(f"  - {rel.subject} --[{rel.relationship.value}]--> {rel.object}")
        print(f"    Evidence: \"{rel.evidence[:50]}...\"")

# Run
asyncio.run(main())
```

---

## Part 3: Robust Extraction with Retries

### Step 1: Create Retry Logic with Instructor

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential

# Patch client with instructor
client = instructor.from_openai(OpenAI())

class EmailExtraction(BaseModel):
    """Extract email components"""
    
    sender: str = Field(description="Email sender name")
    sender_email: str = Field(description="Sender's email address")
    recipients: List[str] = Field(description="List of recipient names")
    subject: str = Field(description="Email subject line")
    summary: str = Field(max_length=200, description="Brief summary")
    action_items: List[str] = Field(default_factory=list)
    priority: str = Field(description="low, medium, or high")
    
    @field_validator('sender_email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        if '@' not in v or '.' not in v.split('@')[-1]:
            raise ValueError(f"Invalid email format: {v}")
        return v.lower()
    
    @field_validator('priority')
    @classmethod
    def validate_priority(cls, v: str) -> str:
        v = v.lower()
        if v not in ['low', 'medium', 'high']:
            raise ValueError(f"Priority must be low, medium, or high, got: {v}")
        return v

def extract_email_with_retry(email_text: str) -> EmailExtraction:
    """Extract email info with automatic retry on validation failure"""
    
    # Instructor handles retries automatically
    return client.chat.completions.create(
        model="gpt-4o",
        response_model=EmailExtraction,
        max_retries=3,  # Retry up to 3 times on validation errors
        messages=[
            {
                "role": "system",
                "content": "Extract structured information from the email."
            },
            {"role": "user", "content": email_text}
        ]
    )

# Test with a tricky email
email = """
From: John Smith <JOHN.SMITH@company.COM>
To: team@company.com, stakeholders@company.com
Subject: URGENT: Q4 Planning Meeting

Hi everyone,

We need to finalize the Q4 roadmap by end of week. Please review the 
attached document and come prepared with your team's priorities.

Action items:
1. Review roadmap document
2. Submit team priorities by Thursday
3. Prepare 5-minute presentation for Friday's meeting

This is high priority - we have board presentation next week.

Thanks,
John
"""

result = extract_email_with_retry(email)
print(f"From: {result.sender} <{result.sender_email}>")
print(f"Subject: {result.subject}")
print(f"Priority: {result.priority}")
print(f"Action items: {result.action_items}")
```

### Step 2: Manual Retry with Feedback

```python
from openai import OpenAI
from pydantic import BaseModel, ValidationError
import json
from typing import List, Tuple

client = OpenAI()

class StrictProduct(BaseModel):
    """Product with strict validation"""
    name: str = Field(min_length=2, max_length=100)
    price: float = Field(gt=0, description="Price must be positive")
    currency: str = Field(pattern=r'^[A-Z]{3}$', description="ISO currency code")
    category: str
    in_stock: bool

def extract_with_feedback(
    text: str,
    max_attempts: int = 3
) -> Tuple[StrictProduct | None, List[str]]:
    """Extract with detailed feedback on failures"""
    
    messages = [
        {
            "role": "system",
            "content": f"""Extract product information. 
            
Schema requirements:
{json.dumps(StrictProduct.model_json_schema(), indent=2)}

Be precise with:
- currency: Must be 3 uppercase letters (USD, EUR, GBP, etc.)
- price: Must be a positive number
- name: Between 2-100 characters"""
        },
        {"role": "user", "content": f"Extract product info from: {text}"}
    ]
    
    errors = []
    
    for attempt in range(max_attempts):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            data = json.loads(content)
            
            # Try to validate
            product = StrictProduct.model_validate(data)
            return product, errors
            
        except json.JSONDecodeError as e:
            error_msg = f"Attempt {attempt + 1}: Invalid JSON - {e}"
            errors.append(error_msg)
            
            messages.append({
                "role": "assistant",
                "content": content
            })
            messages.append({
                "role": "user",
                "content": f"Error: Invalid JSON. Please respond with valid JSON only."
            })
            
        except ValidationError as e:
            error_msg = f"Attempt {attempt + 1}: Validation failed - {e.errors()}"
            errors.append(error_msg)
            
            # Provide specific feedback
            error_details = "\n".join([
                f"- {err['loc']}: {err['msg']}" 
                for err in e.errors()
            ])
            
            messages.append({
                "role": "assistant", 
                "content": content
            })
            messages.append({
                "role": "user",
                "content": f"Validation errors:\n{error_details}\n\nPlease fix these issues and try again."
            })
    
    return None, errors

# Test with ambiguous input
text = "selling my old laptop for about two hundred bucks, still works great"
product, errors = extract_with_feedback(text)

if product:
    print(f"Extracted: {product.name} - {product.price} {product.currency}")
else:
    print("Failed to extract. Errors:")
    for error in errors:
        print(f"  {error}")
```

---

## Part 4: Handling Edge Cases

### Step 1: Graceful Degradation

```python
from pydantic import BaseModel, Field
from typing import Optional, List, Any
from enum import Enum

class ExtractionQuality(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    FAILED = "failed"

class PartialExtraction(BaseModel):
    """Extraction result that handles missing data gracefully"""
    
    # Core fields - always attempt
    title: Optional[str] = None
    
    # Optional enrichment fields
    author: Optional[str] = None
    date: Optional[str] = None
    summary: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    
    # Metadata about extraction
    quality: ExtractionQuality = ExtractionQuality.MEDIUM
    missing_fields: List[str] = Field(default_factory=list)
    extraction_notes: List[str] = Field(default_factory=list)
    raw_data: Optional[dict] = None
    
    def get_completeness_score(self) -> float:
        """Calculate how complete the extraction is"""
        fields = ['title', 'author', 'date', 'summary']
        present = sum(1 for f in fields if getattr(self, f) is not None)
        return present / len(fields)

class GracefulExtractor:
    """Extractor that degrades gracefully on failures"""
    
    def __init__(self):
        self.client = OpenAI()
    
    def extract(self, text: str) -> PartialExtraction:
        """Extract with graceful handling of failures"""
        
        result = PartialExtraction()
        notes = []
        
        # Try structured extraction first
        try:
            response = self.client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Extract article metadata."},
                    {"role": "user", "content": text}
                ],
                response_format=PartialExtraction
            )
            result = response.choices[0].message.parsed
            result.quality = ExtractionQuality.HIGH
            
        except Exception as e:
            notes.append(f"Structured extraction failed: {e}")
            
            # Fallback to simple extraction
            result = self._fallback_extraction(text)
            result.quality = ExtractionQuality.MEDIUM
        
        # Check what's missing
        result.missing_fields = self._find_missing_fields(result)
        result.extraction_notes = notes
        
        # Adjust quality based on completeness
        completeness = result.get_completeness_score()
        if completeness < 0.25:
            result.quality = ExtractionQuality.LOW
        elif completeness >= 0.75:
            result.quality = ExtractionQuality.HIGH
        
        return result
    
    def _fallback_extraction(self, text: str) -> PartialExtraction:
        """Simple fallback extraction"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "Extract what you can. Return JSON with: title, author, date, summary, tags"
                    },
                    {"role": "user", "content": text}
                ],
                response_format={"type": "json_object"}
            )
            
            data = json.loads(response.choices[0].message.content)
            return PartialExtraction(
                title=data.get("title"),
                author=data.get("author"),
                date=data.get("date"),
                summary=data.get("summary"),
                tags=data.get("tags", []),
                raw_data=data
            )
        except Exception:
            return PartialExtraction(quality=ExtractionQuality.FAILED)
    
    def _find_missing_fields(self, result: PartialExtraction) -> List[str]:
        """Identify which fields couldn't be extracted"""
        missing = []
        for field in ['title', 'author', 'date', 'summary']:
            if getattr(result, field) is None:
                missing.append(field)
        return missing

# Usage
extractor = GracefulExtractor()
result = extractor.extract("Just some random text without clear structure...")
print(f"Quality: {result.quality}")
print(f"Completeness: {result.get_completeness_score():.0%}")
print(f"Missing: {result.missing_fields}")
```

---

## Challenge Exercises

### Challenge 1: Invoice Extractor
Build an invoice extraction system that handles:
- Multiple line items
- Tax calculations
- Different currencies
- Various date formats

### Challenge 2: Resume Parser
Create a resume parser that extracts:
- Contact information
- Work history with dates
- Education
- Skills (categorized)
- Handles different resume formats

### Challenge 3: Contract Analyzer
Build a contract analyzer that:
- Extracts key terms and conditions
- Identifies parties involved
- Finds important dates and deadlines
- Flags potential issues

---

## Key Takeaways

1. **Define schemas before extraction** - Clear schemas guide the model
2. **Use Pydantic validators** - Catch errors early
3. **Implement retry logic** - LLMs aren't perfect
4. **Handle partial results** - Graceful degradation is key
5. **Pipeline complex extractions** - Break into stages
6. **Track confidence scores** - Know when to trust results

---

## Next Steps

- [Guardrails & Safety](/learn/advanced-topics/guardrails/prompt-injection) - Protect your extraction systems
- [Observability](/learn/advanced-topics/observability/langsmith) - Monitor extraction quality
