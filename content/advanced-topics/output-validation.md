# Output Validation for LLM Applications

Ensure LLM outputs meet quality, accuracy, and safety standards before presenting them to users or using them in downstream processes.

---

## Why Validate Outputs?

LLM outputs can be:
- **Incorrect**: Hallucinated facts, wrong calculations
- **Inconsistent**: Contradicting previous responses
- **Incomplete**: Missing required information
- **Malformed**: Invalid format, broken structure
- **Harmful**: Bypassing input filters through generation

---

## Validation Strategies

### 1. Schema Validation

Ensure outputs match expected structure:

```python
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
import json

class ProductRecommendation(BaseModel):
    """Expected recommendation format"""
    product_name: str = Field(min_length=1, max_length=100)
    product_id: str = Field(pattern=r'^[A-Z]{2}\d{6}$')
    price: float = Field(gt=0, le=100000)
    reason: str = Field(min_length=20, max_length=500)
    confidence: float = Field(ge=0, le=1)

class RecommendationResponse(BaseModel):
    recommendations: List[ProductRecommendation] = Field(min_length=1, max_length=10)
    query_understood: bool
    total_results: int

def validate_recommendation_output(llm_output: str) -> tuple[RecommendationResponse | None, list[str]]:
    """Validate LLM output against schema"""
    errors = []
    
    # Try to parse JSON
    try:
        data = json.loads(llm_output)
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON: {e}")
        return None, errors
    
    # Validate against schema
    try:
        result = RecommendationResponse.model_validate(data)
        return result, []
    except ValidationError as e:
        for error in e.errors():
            field = " -> ".join(str(x) for x in error["loc"])
            errors.append(f"{field}: {error['msg']}")
        return None, errors

# Usage
llm_response = '{"recommendations": [...], "query_understood": true, "total_results": 5}'
result, errors = validate_recommendation_output(llm_response)

if errors:
    print(f"Validation failed: {errors}")
    # Request regeneration or return error to user
```

### 2. Factual Validation

Verify claims against known data:

```python
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Optional

client = OpenAI()

class FactCheck(BaseModel):
    """Fact checking result"""
    claim: str
    is_verifiable: bool
    is_accurate: Optional[bool] = None
    confidence: float
    source: Optional[str] = None
    correction: Optional[str] = None

class FactValidationResult(BaseModel):
    facts_checked: List[FactCheck]
    overall_accuracy: float
    has_hallucinations: bool
    recommended_action: str  # "accept", "flag", "reject"

def validate_facts(
    llm_output: str,
    context: str,
    known_facts: dict = None
) -> FactValidationResult:
    """Validate factual claims in LLM output"""
    
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """You are a fact-checker. Analyze the response for factual claims.

For each claim:
1. Determine if it's verifiable
2. Check against provided context and known facts
3. Flag any unsupported or contradicted claims

Known facts to verify against:
""" + (json.dumps(known_facts) if known_facts else "None provided")
            },
            {
                "role": "user",
                "content": f"""Context provided to LLM:
{context}

LLM Response to validate:
{llm_output}

Check all factual claims for accuracy."""
            }
        ],
        response_format=FactValidationResult
    )
    
    return response.choices[0].message.parsed

# Usage with RAG context
context = "Our company was founded in 2015. We have 500 employees."
llm_output = "The company was founded in 2010 and has over 1000 employees."

result = validate_facts(llm_output, context)
if result.has_hallucinations:
    print("Warning: Response contains inaccurate information")
    for fact in result.facts_checked:
        if fact.is_accurate == False:
            print(f"  - {fact.claim} -> {fact.correction}")
```

### 3. Consistency Validation

Ensure outputs are consistent with conversation history:

```python
from typing import List, Dict
from pydantic import BaseModel

class ConsistencyCheck(BaseModel):
    """Consistency validation result"""
    is_consistent: bool
    contradictions: List[str]
    context_alignment: float  # 0-1
    tone_consistent: bool

def check_consistency(
    new_response: str,
    conversation_history: List[Dict],
    system_prompt: str
) -> ConsistencyCheck:
    """Check if response is consistent with history"""
    
    # Build history summary
    history_text = "\n".join([
        f"{msg['role']}: {msg['content'][:200]}..."
        for msg in conversation_history[-10:]  # Last 10 messages
    ])
    
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """Analyze if the new response is consistent with:
1. Previous statements made by the assistant
2. The system prompt guidelines
3. The overall conversation context

Flag any contradictions or inconsistencies."""
            },
            {
                "role": "user",
                "content": f"""System Prompt: {system_prompt}

Conversation History:
{history_text}

New Response to Check:
{new_response}"""
            }
        ],
        response_format=ConsistencyCheck
    )
    
    return response.choices[0].message.parsed
```

### 4. Completeness Validation

Verify all required elements are present:

```python
from pydantic import BaseModel
from typing import List, Optional

class CompletenessCheck(BaseModel):
    """Check response completeness"""
    addresses_question: bool
    missing_elements: List[str]
    completeness_score: float  # 0-1
    needs_followup: bool
    suggested_additions: List[str]

def validate_completeness(
    user_query: str,
    llm_response: str,
    required_elements: List[str] = None
) -> CompletenessCheck:
    """Validate response completeness"""
    
    requirements = ""
    if required_elements:
        requirements = f"Required elements: {', '.join(required_elements)}"
    
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"""Evaluate if the response fully addresses the user's query.
                
{requirements}

Check:
1. Does it answer the main question?
2. Are all required elements present?
3. Is any important information missing?"""
            },
            {
                "role": "user",
                "content": f"""User Query: {user_query}

Response to Evaluate:
{llm_response}"""
            }
        ],
        response_format=CompletenessCheck
    )
    
    return response.choices[0].message.parsed

# Usage
query = "What are the pricing plans and how do I upgrade?"
response = "Our basic plan is $10/month. Contact support to upgrade."
required = ["pricing details", "plan comparison", "upgrade process"]

result = validate_completeness(query, response, required)
if result.completeness_score < 0.8:
    print(f"Missing: {result.missing_elements}")
```

---

## Validation Pipeline

Combine validations into a comprehensive pipeline:

```python
from dataclasses import dataclass, field
from typing import List, Optional, Callable
from enum import Enum

class ValidationDecision(str, Enum):
    ACCEPT = "accept"
    MODIFY = "modify"
    REGENERATE = "regenerate"
    REJECT = "reject"

@dataclass
class ValidationResult:
    """Result of a single validation check"""
    validator_name: str
    passed: bool
    decision: ValidationDecision
    issues: List[str] = field(default_factory=list)
    modified_output: Optional[str] = None
    metadata: dict = field(default_factory=dict)

@dataclass
class PipelineResult:
    """Result of full validation pipeline"""
    final_decision: ValidationDecision
    final_output: str
    validations: List[ValidationResult]
    overall_score: float
    needs_human_review: bool

class OutputValidationPipeline:
    """Comprehensive output validation pipeline"""
    
    def __init__(self):
        self.validators: List[tuple[str, Callable, float]] = []
    
    def add_validator(
        self, 
        name: str, 
        validator: Callable, 
        weight: float = 1.0
    ):
        """Add a validator with optional weight"""
        self.validators.append((name, validator, weight))
    
    def validate(
        self,
        output: str,
        context: dict
    ) -> PipelineResult:
        """Run all validators on output"""
        
        results = []
        current_output = output
        weighted_score = 0
        total_weight = 0
        
        for name, validator, weight in self.validators:
            try:
                result = validator(current_output, context)
                results.append(result)
                
                # Update score
                total_weight += weight
                if result.passed:
                    weighted_score += weight
                
                # Handle modifications
                if result.modified_output:
                    current_output = result.modified_output
                
                # Stop on rejection
                if result.decision == ValidationDecision.REJECT:
                    break
                    
            except Exception as e:
                results.append(ValidationResult(
                    validator_name=name,
                    passed=False,
                    decision=ValidationDecision.REGENERATE,
                    issues=[f"Validator error: {str(e)}"]
                ))
        
        # Determine final decision
        overall_score = weighted_score / total_weight if total_weight > 0 else 0
        
        if any(r.decision == ValidationDecision.REJECT for r in results):
            final_decision = ValidationDecision.REJECT
        elif any(r.decision == ValidationDecision.REGENERATE for r in results):
            final_decision = ValidationDecision.REGENERATE
        elif any(r.decision == ValidationDecision.MODIFY for r in results):
            final_decision = ValidationDecision.MODIFY
        else:
            final_decision = ValidationDecision.ACCEPT
        
        needs_review = overall_score < 0.7 or final_decision in [
            ValidationDecision.REJECT, 
            ValidationDecision.REGENERATE
        ]
        
        return PipelineResult(
            final_decision=final_decision,
            final_output=current_output,
            validations=results,
            overall_score=overall_score,
            needs_human_review=needs_review
        )

# Define validators
def schema_validator(output: str, context: dict) -> ValidationResult:
    """Validate output schema"""
    try:
        data = json.loads(output)
        # Add specific schema validation
        return ValidationResult(
            validator_name="schema",
            passed=True,
            decision=ValidationDecision.ACCEPT
        )
    except:
        return ValidationResult(
            validator_name="schema",
            passed=False,
            decision=ValidationDecision.REGENERATE,
            issues=["Invalid JSON format"]
        )

def length_validator(output: str, context: dict) -> ValidationResult:
    """Validate output length"""
    min_len = context.get("min_length", 10)
    max_len = context.get("max_length", 5000)
    
    if len(output) < min_len:
        return ValidationResult(
            validator_name="length",
            passed=False,
            decision=ValidationDecision.REGENERATE,
            issues=[f"Response too short ({len(output)} < {min_len})"]
        )
    
    if len(output) > max_len:
        # Truncate rather than reject
        return ValidationResult(
            validator_name="length",
            passed=True,
            decision=ValidationDecision.MODIFY,
            issues=[f"Response truncated ({len(output)} > {max_len})"],
            modified_output=output[:max_len] + "..."
        )
    
    return ValidationResult(
        validator_name="length",
        passed=True,
        decision=ValidationDecision.ACCEPT
    )

def safety_validator(output: str, context: dict) -> ValidationResult:
    """Validate output safety"""
    result = check_moderation(output)  # From earlier example
    
    if result["flagged"]:
        return ValidationResult(
            validator_name="safety",
            passed=False,
            decision=ValidationDecision.REJECT,
            issues=[f"Flagged for: {result['categories']}"]
        )
    
    return ValidationResult(
        validator_name="safety",
        passed=True,
        decision=ValidationDecision.ACCEPT
    )

# Build pipeline
pipeline = OutputValidationPipeline()
pipeline.add_validator("schema", schema_validator, weight=1.0)
pipeline.add_validator("length", length_validator, weight=0.5)
pipeline.add_validator("safety", safety_validator, weight=2.0)  # Higher weight

# Usage
result = pipeline.validate(
    output='{"answer": "Response here"}',
    context={"min_length": 50, "max_length": 1000}
)

print(f"Decision: {result.final_decision}")
print(f"Score: {result.overall_score:.0%}")
```

---

## Domain-Specific Validation

### Code Validation

```python
import ast
import subprocess
import tempfile
from typing import Optional

def validate_python_code(code: str) -> tuple[bool, list[str]]:
    """Validate Python code syntax and basic quality"""
    issues = []
    
    # Syntax check
    try:
        ast.parse(code)
    except SyntaxError as e:
        issues.append(f"Syntax error: {e.msg} at line {e.lineno}")
        return False, issues
    
    # Basic quality checks
    tree = ast.parse(code)
    
    # Check for common issues
    for node in ast.walk(tree):
        # Detect eval/exec usage
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ['eval', 'exec']:
                    issues.append(f"Dangerous function: {node.func.id}")
        
        # Detect empty except blocks
        if isinstance(node, ast.ExceptHandler):
            if node.type is None and len(node.body) == 1:
                if isinstance(node.body[0], ast.Pass):
                    issues.append("Empty except block detected")
    
    return len(issues) == 0, issues

def validate_code_execution(code: str, timeout: int = 5) -> tuple[bool, str]:
    """Actually execute code in sandbox and validate output"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        f.flush()
        
        try:
            result = subprocess.run(
                ['python', f.name],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, result.stderr
        except subprocess.TimeoutExpired:
            return False, "Execution timed out"
        except Exception as e:
            return False, str(e)
```

### SQL Validation

```python
import sqlparse
from typing import List, Tuple

def validate_sql_query(query: str, allowed_tables: List[str] = None) -> Tuple[bool, List[str]]:
    """Validate SQL query for safety and correctness"""
    issues = []
    
    # Parse SQL
    try:
        parsed = sqlparse.parse(query)
    except:
        issues.append("Failed to parse SQL")
        return False, issues
    
    if not parsed:
        issues.append("Empty query")
        return False, issues
    
    statement = parsed[0]
    
    # Check statement type
    stmt_type = statement.get_type()
    
    # Block dangerous operations
    dangerous_types = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE']
    if stmt_type in dangerous_types:
        issues.append(f"Dangerous operation: {stmt_type}")
    
    # Check for SQL injection patterns
    query_lower = query.lower()
    injection_patterns = [
        'union select',
        'or 1=1',
        'drop table',
        '; --',
        "' or '",
    ]
    
    for pattern in injection_patterns:
        if pattern in query_lower:
            issues.append(f"Potential SQL injection: {pattern}")
    
    # Validate table names if whitelist provided
    if allowed_tables:
        # Extract table names from query
        tokens = query_lower.split()
        for i, token in enumerate(tokens):
            if token in ['from', 'join', 'into', 'update']:
                if i + 1 < len(tokens):
                    table = tokens[i + 1].strip('(),')
                    if table not in [t.lower() for t in allowed_tables]:
                        issues.append(f"Unauthorized table: {table}")
    
    return len(issues) == 0, issues
```

### Math/Calculation Validation

```python
import re
from typing import Optional

def validate_calculation(
    llm_response: str,
    expected_result: Optional[float] = None,
    tolerance: float = 0.01
) -> tuple[bool, str]:
    """Validate mathematical calculations in response"""
    
    # Extract numbers from response
    numbers = re.findall(r'-?\d+\.?\d*', llm_response)
    
    if expected_result is not None and numbers:
        # Check if expected result appears in response
        for num_str in numbers:
            try:
                num = float(num_str)
                if abs(num - expected_result) / max(abs(expected_result), 1) < tolerance:
                    return True, f"Found expected result: {num}"
            except:
                continue
        
        return False, f"Expected {expected_result}, found: {numbers}"
    
    # If no expected result, just validate format
    return True, "No specific result to validate"

# For complex calculations, use symbolic math
def verify_math_with_sympy(expression: str, expected: float) -> tuple[bool, float]:
    """Use SymPy to verify mathematical expressions"""
    try:
        import sympy
        result = float(sympy.sympify(expression))
        matches = abs(result - expected) < 0.0001
        return matches, result
    except Exception as e:
        return False, 0.0
```

---

## Automatic Regeneration

When validation fails, automatically request a better response:

```python
from openai import OpenAI
from typing import Callable, Any

client = OpenAI()

class RegeneratingValidator:
    """Validator that automatically regenerates on failure"""
    
    def __init__(
        self,
        model: str = "gpt-4o",
        max_retries: int = 3
    ):
        self.client = OpenAI()
        self.model = model
        self.max_retries = max_retries
    
    def generate_with_validation(
        self,
        messages: list,
        validator: Callable[[str], tuple[bool, list[str]]],
        response_format: Any = None
    ) -> tuple[str, int, list[str]]:
        """
        Generate response with automatic regeneration on validation failure.
        Returns: (response, attempts, all_issues)
        """
        all_issues = []
        
        for attempt in range(self.max_retries):
            # Generate response
            kwargs = {
                "model": self.model,
                "messages": messages
            }
            if response_format:
                kwargs["response_format"] = response_format
            
            response = self.client.chat.completions.create(**kwargs)
            output = response.choices[0].message.content
            
            # Validate
            is_valid, issues = validator(output)
            
            if is_valid:
                return output, attempt + 1, all_issues
            
            all_issues.extend(issues)
            
            # Add feedback for retry
            messages = messages + [
                {"role": "assistant", "content": output},
                {
                    "role": "user",
                    "content": f"""Your response had the following issues:
{chr(10).join(f'- {issue}' for issue in issues)}

Please try again and fix these issues."""
                }
            ]
        
        # Return last attempt even if still invalid
        return output, self.max_retries, all_issues

# Usage
validator = RegeneratingValidator(max_retries=3)

def my_validator(output: str) -> tuple[bool, list[str]]:
    issues = []
    if len(output) < 100:
        issues.append("Response too short")
    if "error" in output.lower():
        issues.append("Contains error mention")
    return len(issues) == 0, issues

response, attempts, issues = validator.generate_with_validation(
    messages=[{"role": "user", "content": "Write a detailed product description"}],
    validator=my_validator
)

print(f"Final response after {attempts} attempt(s)")
```

---

## Best Practices

### 1. Validate Early and Often

```python
# Validate at multiple points
class ChatPipeline:
    def process(self, user_input: str) -> str:
        # Validate input
        validated_input = self.validate_input(user_input)
        
        # Generate response
        raw_response = self.generate(validated_input)
        
        # Validate output
        validated_output = self.validate_output(raw_response)
        
        # Post-process
        final_output = self.post_process(validated_output)
        
        return final_output
```

### 2. Log Validation Results

```python
import logging

def log_validation(
    validator_name: str,
    passed: bool,
    issues: list,
    response_hash: str
):
    """Log validation results for monitoring"""
    logging.info({
        "event": "validation",
        "validator": validator_name,
        "passed": passed,
        "issue_count": len(issues),
        "issues": issues[:3],  # First 3 issues
        "response_hash": response_hash
    })
```

### 3. Graceful Degradation

```python
def validate_with_fallback(output: str) -> str:
    """Validate with fallback responses"""
    
    result = pipeline.validate(output, {})
    
    if result.final_decision == ValidationDecision.ACCEPT:
        return result.final_output
    
    elif result.final_decision == ValidationDecision.MODIFY:
        return result.final_output
    
    elif result.final_decision == ValidationDecision.REGENERATE:
        # Try regeneration
        new_output = regenerate(...)
        return new_output
    
    else:  # REJECT
        # Return safe fallback
        return "I'm sorry, I couldn't generate a valid response. Please try rephrasing your question."
```

---

## Summary

| Validation Type | Purpose | When to Use |
|----------------|---------|-------------|
| Schema | Structure correctness | Structured outputs |
| Factual | Accuracy | Knowledge-based responses |
| Consistency | Coherence | Multi-turn conversations |
| Completeness | Coverage | Q&A, support |
| Safety | Harm prevention | All outputs |
| Domain-specific | Technical accuracy | Code, SQL, math |

**Key Takeaways:**
1. Never trust LLM outputs blindly
2. Layer multiple validation types
3. Implement automatic regeneration
4. Log validation results
5. Have fallback responses ready

---

## Next Steps

- [Lab: Safety Guardrails](/learn/advanced-topics/guardrails/safety-lab) - Build complete validation system
- [Observability](/learn/advanced-topics/observability/langsmith) - Monitor validation metrics
