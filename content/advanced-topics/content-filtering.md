# Content Filtering for LLM Applications

Implement robust content filtering to prevent harmful, inappropriate, or policy-violating content from being generated or processed by your LLM applications.

---

## Why Content Filtering Matters

Content filtering protects your application from:
- **Harmful content**: Violence, self-harm, illegal activities
- **Inappropriate content**: Adult content, profanity, harassment
- **Policy violations**: PII exposure, copyright infringement
- **Brand safety issues**: Off-topic or reputation-damaging responses

---

## Content Filtering Approaches

### 1. Input Filtering

Filter user inputs before they reach the LLM:

```python
from typing import Tuple, List
import re

class InputFilter:
    """Filter inappropriate user inputs"""
    
    # Blocklist patterns
    BLOCKED_TOPICS = [
        r"how\s+to\s+(make|build|create)\s+(bomb|weapon|explosive)",
        r"how\s+to\s+(hack|break\s+into)",
        r"generate\s+(illegal|harmful)",
    ]
    
    # Profanity list (abbreviated example)
    PROFANITY = {"badword1", "badword2", "badword3"}
    
    def __init__(self):
        self.topic_patterns = [
            re.compile(p, re.IGNORECASE) 
            for p in self.BLOCKED_TOPICS
        ]
    
    def filter_input(self, text: str) -> Tuple[bool, str, List[str]]:
        """
        Filter input text.
        Returns: (is_allowed, filtered_text, violations)
        """
        violations = []
        
        # Check blocked topics
        for pattern in self.topic_patterns:
            if pattern.search(text):
                violations.append("blocked_topic")
        
        # Check profanity
        words = text.lower().split()
        found_profanity = [w for w in words if w in self.PROFANITY]
        if found_profanity:
            violations.append("profanity")
        
        # Replace profanity in filtered text
        filtered = text
        for word in found_profanity:
            filtered = re.sub(
                rf'\b{word}\b', 
                '*' * len(word), 
                filtered, 
                flags=re.IGNORECASE
            )
        
        is_allowed = len(violations) == 0
        return is_allowed, filtered, violations

# Usage
filter = InputFilter()
allowed, filtered, violations = filter.filter_input("User message here")

if not allowed:
    print(f"Input blocked: {violations}")
```

### 2. Output Filtering

Filter LLM responses before showing to users:

```python
from dataclasses import dataclass
from typing import List, Optional
import re

@dataclass
class FilterResult:
    """Result of content filtering"""
    original: str
    filtered: str
    was_modified: bool
    blocked_categories: List[str]
    confidence_scores: dict

class OutputFilter:
    """Comprehensive output filtering"""
    
    def __init__(self):
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        }
        
        self.sensitive_patterns = {
            'api_key': r'\b(api[_-]?key|apikey)["\s:=]+[A-Za-z0-9_-]{20,}\b',
            'password': r'\b(password|passwd|pwd)["\s:=]+\S+\b',
            'token': r'\b(bearer|token)["\s:=]+[A-Za-z0-9_-]{20,}\b',
        }
    
    def filter(self, text: str) -> FilterResult:
        """Apply all filters to text"""
        
        filtered = text
        blocked = []
        
        # Filter PII
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, filtered, re.IGNORECASE)
            if matches:
                blocked.append(f"pii_{pii_type}")
                filtered = re.sub(
                    pattern, 
                    f"[{pii_type.upper()}_REDACTED]", 
                    filtered, 
                    flags=re.IGNORECASE
                )
        
        # Filter sensitive data
        for sens_type, pattern in self.sensitive_patterns.items():
            if re.search(pattern, filtered, re.IGNORECASE):
                blocked.append(f"sensitive_{sens_type}")
                filtered = re.sub(
                    pattern,
                    f"[{sens_type.upper()}_REDACTED]",
                    filtered,
                    flags=re.IGNORECASE
                )
        
        return FilterResult(
            original=text,
            filtered=filtered,
            was_modified=text != filtered,
            blocked_categories=blocked,
            confidence_scores={}
        )

# Usage
output_filter = OutputFilter()
result = output_filter.filter("Contact john@email.com or call 555-123-4567")
print(result.filtered)  # Contact [EMAIL_REDACTED] or call [PHONE_REDACTED]
```

### 3. LLM-Based Content Moderation

Use a dedicated model for content classification:

```python
from openai import OpenAI
from pydantic import BaseModel
from typing import List
from enum import Enum

client = OpenAI()

class ContentCategory(str, Enum):
    SAFE = "safe"
    VIOLENCE = "violence"
    SEXUAL = "sexual"
    HATE = "hate"
    SELF_HARM = "self_harm"
    ILLEGAL = "illegal"
    HARASSMENT = "harassment"
    MISINFORMATION = "misinformation"

class ModerationResult(BaseModel):
    """Content moderation result"""
    is_safe: bool
    categories: List[ContentCategory]
    severity: str  # low, medium, high
    explanation: str
    confidence: float

def moderate_content(text: str) -> ModerationResult:
    """Use LLM for content moderation"""
    
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",  # Cheaper model for moderation
        messages=[
            {
                "role": "system",
                "content": """You are a content moderation system. Analyze the text for:
                
1. Violence or threats
2. Sexual or adult content
3. Hate speech or discrimination
4. Self-harm or suicide content
5. Illegal activities
6. Harassment or bullying
7. Misinformation or dangerous advice

Rate severity as: low (borderline), medium (clearly inappropriate), high (dangerous)
Be conservative - when in doubt, flag it."""
            },
            {
                "role": "user",
                "content": f"Moderate this content:\n\n{text}"
            }
        ],
        response_format=ModerationResult
    )
    
    return response.choices[0].message.parsed

# Usage
result = moderate_content("Some user generated content...")
if not result.is_safe:
    print(f"Content flagged: {result.categories}")
    print(f"Severity: {result.severity}")
```

### 4. OpenAI Moderation API

Use OpenAI's built-in moderation endpoint:

```python
from openai import OpenAI

client = OpenAI()

def check_moderation(text: str) -> dict:
    """Use OpenAI's moderation endpoint"""
    
    response = client.moderations.create(input=text)
    
    result = response.results[0]
    
    # Get flagged categories
    flagged_categories = [
        category for category, flagged 
        in result.categories.model_dump().items() 
        if flagged
    ]
    
    # Get category scores
    scores = result.category_scores.model_dump()
    
    return {
        "flagged": result.flagged,
        "categories": flagged_categories,
        "scores": scores,
        "highest_risk": max(scores, key=scores.get) if scores else None,
        "highest_score": max(scores.values()) if scores else 0
    }

# Usage
text = "Some potentially problematic content..."
result = check_moderation(text)

if result["flagged"]:
    print(f"Content flagged for: {result['categories']}")
    print(f"Highest risk: {result['highest_risk']} ({result['highest_score']:.2%})")
```

---

## Multi-Layer Filtering Pipeline

Combine multiple filtering approaches:

```python
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class FilterDecision(str, Enum):
    ALLOW = "allow"
    MODIFY = "modify"
    BLOCK = "block"
    REVIEW = "review"

@dataclass
class FilterStageResult:
    stage: str
    decision: FilterDecision
    modified_text: Optional[str] = None
    reason: Optional[str] = None
    metadata: dict = field(default_factory=dict)

@dataclass
class PipelineResult:
    final_decision: FilterDecision
    final_text: str
    stage_results: List[FilterStageResult]
    blocked_at_stage: Optional[str] = None

class ContentFilterPipeline:
    """Multi-stage content filtering pipeline"""
    
    def __init__(self):
        self.stages: List[tuple[str, Callable]] = []
        self.input_filter = InputFilter()
        self.output_filter = OutputFilter()
    
    def add_stage(self, name: str, filter_func: Callable):
        """Add a filtering stage"""
        self.stages.append((name, filter_func))
    
    def filter(self, text: str, direction: str = "input") -> PipelineResult:
        """
        Run content through all filter stages.
        direction: "input" or "output"
        """
        results = []
        current_text = text
        final_decision = FilterDecision.ALLOW
        blocked_at = None
        
        for stage_name, filter_func in self.stages:
            try:
                stage_result = filter_func(current_text, direction)
                results.append(stage_result)
                
                if stage_result.decision == FilterDecision.BLOCK:
                    final_decision = FilterDecision.BLOCK
                    blocked_at = stage_name
                    logger.warning(f"Content blocked at {stage_name}: {stage_result.reason}")
                    break
                
                elif stage_result.decision == FilterDecision.MODIFY:
                    if stage_result.modified_text:
                        current_text = stage_result.modified_text
                    if final_decision != FilterDecision.REVIEW:
                        final_decision = FilterDecision.MODIFY
                
                elif stage_result.decision == FilterDecision.REVIEW:
                    final_decision = FilterDecision.REVIEW
                    
            except Exception as e:
                logger.error(f"Error in filter stage {stage_name}: {e}")
                results.append(FilterStageResult(
                    stage=stage_name,
                    decision=FilterDecision.REVIEW,
                    reason=f"Filter error: {str(e)}"
                ))
        
        return PipelineResult(
            final_decision=final_decision,
            final_text=current_text,
            stage_results=results,
            blocked_at_stage=blocked_at
        )

# Define filter stages
def profanity_filter(text: str, direction: str) -> FilterStageResult:
    """Simple profanity filter"""
    # Implementation here
    return FilterStageResult(
        stage="profanity",
        decision=FilterDecision.ALLOW
    )

def pii_filter(text: str, direction: str) -> FilterStageResult:
    """PII detection and redaction"""
    output_filter = OutputFilter()
    result = output_filter.filter(text)
    
    if result.was_modified:
        return FilterStageResult(
            stage="pii",
            decision=FilterDecision.MODIFY,
            modified_text=result.filtered,
            reason=f"PII detected: {result.blocked_categories}"
        )
    
    return FilterStageResult(stage="pii", decision=FilterDecision.ALLOW)

def harmful_content_filter(text: str, direction: str) -> FilterStageResult:
    """Check for harmful content using moderation API"""
    result = check_moderation(text)
    
    if result["flagged"]:
        severity = result["highest_score"]
        
        if severity > 0.9:
            return FilterStageResult(
                stage="harmful_content",
                decision=FilterDecision.BLOCK,
                reason=f"High-severity content: {result['highest_risk']}"
            )
        elif severity > 0.7:
            return FilterStageResult(
                stage="harmful_content",
                decision=FilterDecision.REVIEW,
                reason=f"Medium-severity content: {result['highest_risk']}",
                metadata={"scores": result["scores"]}
            )
    
    return FilterStageResult(stage="harmful_content", decision=FilterDecision.ALLOW)

# Build pipeline
pipeline = ContentFilterPipeline()
pipeline.add_stage("profanity", profanity_filter)
pipeline.add_stage("pii", pii_filter)
pipeline.add_stage("harmful_content", harmful_content_filter)

# Usage
result = pipeline.filter("User input here", direction="input")
print(f"Decision: {result.final_decision}")
if result.blocked_at_stage:
    print(f"Blocked at: {result.blocked_at_stage}")
```

---

## Topic and Brand Safety Filtering

### Topic Restriction

```python
from openai import OpenAI
from pydantic import BaseModel
from typing import List

client = OpenAI()

class TopicAnalysis(BaseModel):
    """Analysis of content topics"""
    detected_topics: List[str]
    on_topic: bool
    off_topic_reason: Optional[str] = None
    confidence: float

def enforce_topic_boundaries(
    text: str,
    allowed_topics: List[str],
    blocked_topics: List[str]
) -> tuple[bool, str]:
    """Ensure content stays within topic boundaries"""
    
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"""Analyze if the content is on-topic.

Allowed topics: {', '.join(allowed_topics)}
Blocked topics: {', '.join(blocked_topics)}

Content is on-topic if it relates to allowed topics and doesn't discuss blocked topics."""
            },
            {"role": "user", "content": text}
        ],
        response_format=TopicAnalysis
    )
    
    result = response.choices[0].message.parsed
    return result.on_topic, result.off_topic_reason or ""

# Usage for a customer support bot
allowed = ["product support", "billing", "account management", "technical help"]
blocked = ["politics", "religion", "competitors", "legal advice"]

is_allowed, reason = enforce_topic_boundaries(
    "Can you help me reset my password?",
    allowed,
    blocked
)
```

### Brand Voice Compliance

```python
from pydantic import BaseModel
from typing import List

class BrandCompliance(BaseModel):
    """Brand voice compliance check"""
    is_compliant: bool
    issues: List[str]
    suggestions: List[str]
    tone_score: float  # 0-1, how well it matches brand voice

def check_brand_compliance(
    text: str,
    brand_guidelines: str
) -> BrandCompliance:
    """Ensure content matches brand voice"""
    
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"""You are a brand compliance checker.

Brand Guidelines:
{brand_guidelines}

Check if the content follows these guidelines and maintains the brand voice."""
            },
            {"role": "user", "content": f"Check this content:\n\n{text}"}
        ],
        response_format=BrandCompliance
    )
    
    return response.choices[0].message.parsed

# Usage
brand_guidelines = """
- Friendly but professional tone
- Use "we" instead of "the company"
- Avoid jargon - explain technical terms
- Be empathetic to customer frustrations
- Never make promises about features not yet released
- Don't discuss competitors by name
"""

result = check_brand_compliance(
    "Our competitor's product is terrible. Buy ours instead!",
    brand_guidelines
)

if not result.is_compliant:
    print(f"Brand issues: {result.issues}")
```

---

## Real-Time Streaming Filter

Filter content as it streams:

```python
from openai import OpenAI
import re

client = OpenAI()

class StreamingFilter:
    """Filter content in real-time during streaming"""
    
    def __init__(self):
        self.buffer = ""
        self.pii_patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
        }
    
    def process_chunk(self, chunk: str) -> str:
        """Process a streaming chunk"""
        self.buffer += chunk
        
        # Only process complete words
        if not chunk.endswith(' ') and not chunk.endswith('\n'):
            return ""
        
        # Check buffer for patterns
        filtered = self.buffer
        for pii_type, pattern in self.pii_patterns.items():
            filtered = pattern.sub(f"[{pii_type.upper()}]", filtered)
        
        # Return filtered content and reset buffer
        output = filtered
        self.buffer = ""
        return output
    
    def flush(self) -> str:
        """Flush remaining buffer"""
        if self.buffer:
            filtered = self.buffer
            for pii_type, pattern in self.pii_patterns.items():
                filtered = pattern.sub(f"[{pii_type.upper()}]", filtered)
            self.buffer = ""
            return filtered
        return ""

def stream_with_filter(prompt: str):
    """Stream LLM response with real-time filtering"""
    
    filter = StreamingFilter()
    
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            filtered = filter.process_chunk(content)
            if filtered:
                yield filtered
    
    # Flush any remaining content
    final = filter.flush()
    if final:
        yield final

# Usage
for text in stream_with_filter("Generate some content..."):
    print(text, end="", flush=True)
```

---

## Best Practices

### 1. Defense in Depth

```python
# Layer multiple filters
def comprehensive_filter(text: str) -> tuple[str, bool]:
    # Layer 1: Quick blocklist check
    if quick_blocklist_check(text):
        return "", False
    
    # Layer 2: Pattern-based filtering
    text = pattern_filter(text)
    
    # Layer 3: PII redaction
    text = redact_pii(text)
    
    # Layer 4: ML-based moderation
    if not ml_moderation_check(text):
        return "", False
    
    # Layer 5: Topic boundaries
    if not check_topic_boundaries(text):
        return "", False
    
    return text, True
```

### 2. Configurable Strictness

```python
from enum import Enum

class StrictnessLevel(Enum):
    PERMISSIVE = "permissive"  # Only block clearly harmful
    MODERATE = "moderate"      # Default - balanced filtering
    STRICT = "strict"          # Aggressive filtering
    PARANOID = "paranoid"      # Block anything questionable

def get_threshold(level: StrictnessLevel, category: str) -> float:
    """Get moderation threshold based on strictness"""
    thresholds = {
        StrictnessLevel.PERMISSIVE: {"violence": 0.9, "sexual": 0.9, "hate": 0.8},
        StrictnessLevel.MODERATE: {"violence": 0.7, "sexual": 0.7, "hate": 0.6},
        StrictnessLevel.STRICT: {"violence": 0.5, "sexual": 0.5, "hate": 0.4},
        StrictnessLevel.PARANOID: {"violence": 0.3, "sexual": 0.3, "hate": 0.2},
    }
    return thresholds[level].get(category, 0.5)
```

### 3. Audit Logging

```python
import logging
from datetime import datetime

def log_filter_decision(
    input_text: str,
    output_text: str,
    decision: str,
    reasons: List[str],
    user_id: Optional[str] = None
):
    """Log filtering decisions for audit"""
    
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_id": user_id,
        "input_hash": hashlib.sha256(input_text.encode()).hexdigest()[:16],
        "decision": decision,
        "reasons": reasons,
        "was_modified": input_text != output_text
    }
    
    logger.info(f"Content filter: {log_entry}")
```

---

## Summary

| Filter Type | Best For | Latency | Cost |
|-------------|----------|---------|------|
| Blocklist/Pattern | Known bad content | Very Low | Free |
| PII Detection | Data privacy | Low | Free |
| OpenAI Moderation | Harmful content | Low | Free |
| LLM Classifier | Complex decisions | Medium | $ |
| Custom ML Model | Domain-specific | Medium | $$ |

**Key Takeaways:**
1. Use multiple filtering layers for defense in depth
2. Filter both inputs and outputs
3. Make strictness configurable by use case
4. Log decisions for audit and improvement
5. Handle edge cases gracefully

---

## Next Steps

- [Output Validation](/learn/advanced-topics/guardrails/output-validation) - Ensure response quality
- [Lab: Safety Guardrails](/learn/advanced-topics/guardrails/safety-lab) - Build complete protection system
