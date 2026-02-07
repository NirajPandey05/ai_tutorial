# Hybrid Architectures: Combining Local and Cloud LLMs

The most effective production deployments often combine self-hosted models with cloud APIs. This guide covers hybrid architecture patterns and their implementation.

## Why Hybrid?

```yaml
hybrid_benefits:
  best_of_both:
    - "Privacy for sensitive data (local)"
    - "Cutting-edge capabilities (cloud)"
    - "Cost optimization through routing"
    - "Redundancy and fallback"
    
  use_cases:
    - "PII processing locally, general queries to cloud"
    - "Simple tasks locally, complex reasoning to cloud"
    - "High volume batch to local, premium features to cloud"
    - "Primary local with cloud backup"
```

## Architecture Patterns

### Pattern 1: Privacy-Based Routing

Route based on data sensitivity:

```
┌─────────────────────────────────────────────────────────────┐
│                    Incoming Request                         │
└───────────────────────────┬─────────────────────────────────┘
                            │
                    ┌───────▼───────┐
                    │  PII Detector │
                    └───────┬───────┘
                            │
            ┌───────────────┴───────────────┐
            │ Contains PII?                 │
            │                               │
      ┌─────▼─────┐                 ┌───────▼───────┐
      │    YES    │                 │      NO       │
      │           │                 │               │
      │  Local    │                 │    Cloud      │
      │  Model    │                 │    Model      │
      │ (Ollama)  │                 │   (OpenAI)    │
      └───────────┘                 └───────────────┘
```

```python
import re
from typing import Literal
from openai import OpenAI
import requests

class PrivacyRouter:
    """Route requests based on privacy requirements."""
    
    def __init__(
        self,
        local_base_url: str = "http://localhost:11434/v1",
        cloud_api_key: str = None
    ):
        self.local_client = OpenAI(
            base_url=local_base_url,
            api_key="local"
        )
        self.cloud_client = OpenAI(api_key=cloud_api_key)
        
        # PII patterns
        self.pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{16}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone
            r'\b\d{5}(?:-\d{4})?\b',  # ZIP code
        ]
    
    def contains_pii(self, text: str) -> bool:
        """Check if text contains PII."""
        for pattern in self.pii_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def route(
        self,
        messages: list[dict],
        force_local: bool = False
    ) -> dict:
        """Route request to appropriate backend."""
        
        # Check all messages for PII
        has_pii = any(
            self.contains_pii(msg.get("content", ""))
            for msg in messages
        )
        
        if has_pii or force_local:
            # Use local model for privacy
            response = self.local_client.chat.completions.create(
                model="llama3.2:3b",
                messages=messages
            )
            return {
                "response": response.choices[0].message.content,
                "routed_to": "local",
                "reason": "PII detected" if has_pii else "forced local"
            }
        else:
            # Use cloud for better quality
            response = self.cloud_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            return {
                "response": response.choices[0].message.content,
                "routed_to": "cloud",
                "reason": "no PII detected"
            }

# Usage
router = PrivacyRouter(cloud_api_key="sk-...")

# This goes to local
result = router.route([
    {"role": "user", "content": "Summarize this: SSN 123-45-6789, John Doe"}
])
print(f"Routed to: {result['routed_to']}")

# This goes to cloud
result = router.route([
    {"role": "user", "content": "Explain quantum computing"}
])
print(f"Routed to: {result['routed_to']}")
```

### Pattern 2: Complexity-Based Routing

Route based on task complexity:

```python
from enum import Enum
from openai import OpenAI

class TaskComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"

class ComplexityRouter:
    """Route based on task complexity."""
    
    def __init__(self):
        self.local_client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="local"
        )
        self.cloud_client = OpenAI()
        
        # Complexity indicators
        self.complex_keywords = [
            "analyze", "compare", "evaluate", "synthesize",
            "multi-step", "reasoning", "complex", "advanced",
            "code review", "debug", "optimize"
        ]
        
        self.simple_keywords = [
            "summarize", "translate", "format", "list",
            "define", "explain simply", "basic"
        ]
    
    def classify_complexity(self, prompt: str) -> TaskComplexity:
        """Classify task complexity from prompt."""
        prompt_lower = prompt.lower()
        
        # Check for complex indicators
        complex_score = sum(
            1 for kw in self.complex_keywords 
            if kw in prompt_lower
        )
        
        simple_score = sum(
            1 for kw in self.simple_keywords 
            if kw in prompt_lower
        )
        
        # Length also indicates complexity
        if len(prompt) > 1000:
            complex_score += 1
        
        if complex_score > simple_score:
            return TaskComplexity.COMPLEX
        elif simple_score > complex_score:
            return TaskComplexity.SIMPLE
        else:
            return TaskComplexity.MODERATE
    
    def route(self, messages: list[dict]) -> dict:
        """Route based on complexity."""
        
        # Get the user's main prompt
        user_content = " ".join(
            msg["content"] for msg in messages 
            if msg["role"] == "user"
        )
        
        complexity = self.classify_complexity(user_content)
        
        model_config = {
            TaskComplexity.SIMPLE: {
                "client": self.local_client,
                "model": "llama3.2:3b",
                "backend": "local"
            },
            TaskComplexity.MODERATE: {
                "client": self.local_client,
                "model": "llama3.1:8b",
                "backend": "local"
            },
            TaskComplexity.COMPLEX: {
                "client": self.cloud_client,
                "model": "gpt-4o",
                "backend": "cloud"
            }
        }
        
        config = model_config[complexity]
        
        response = config["client"].chat.completions.create(
            model=config["model"],
            messages=messages
        )
        
        return {
            "response": response.choices[0].message.content,
            "complexity": complexity.value,
            "model": config["model"],
            "backend": config["backend"]
        }
```

### Pattern 3: Cost-Optimized Routing

Route to minimize costs while meeting quality needs:

```python
import time
from dataclasses import dataclass
from openai import OpenAI

@dataclass
class ModelConfig:
    name: str
    cost_per_1k_tokens: float
    quality_score: float  # 0-1
    client: OpenAI
    is_local: bool

class CostOptimizedRouter:
    """Route to minimize cost while meeting quality threshold."""
    
    def __init__(self, local_url: str, cloud_api_key: str):
        self.local_client = OpenAI(base_url=local_url, api_key="local")
        self.cloud_client = OpenAI(api_key=cloud_api_key)
        
        self.models = [
            ModelConfig("llama3.2:3b", 0.0, 0.7, self.local_client, True),
            ModelConfig("llama3.1:8b", 0.0, 0.8, self.local_client, True),
            ModelConfig("gpt-4o-mini", 0.15, 0.85, self.cloud_client, False),
            ModelConfig("gpt-4o", 2.50, 0.95, self.cloud_client, False),
        ]
        
        # Usage tracking
        self.daily_cloud_spend = 0.0
        self.daily_budget = 10.0  # $10/day cloud budget
    
    def estimate_tokens(self, messages: list[dict]) -> int:
        """Rough token estimation."""
        text = " ".join(msg.get("content", "") for msg in messages)
        return len(text) // 4  # ~4 chars per token
    
    def route(
        self,
        messages: list[dict],
        min_quality: float = 0.75
    ) -> dict:
        """Route based on cost optimization."""
        
        estimated_tokens = self.estimate_tokens(messages) + 500  # + response
        
        # Filter models meeting quality threshold
        suitable_models = [
            m for m in self.models 
            if m.quality_score >= min_quality
        ]
        
        # Sort by cost (local = 0, prefer local)
        suitable_models.sort(key=lambda m: (
            m.cost_per_1k_tokens,
            -m.quality_score  # Higher quality as tiebreaker
        ))
        
        for model in suitable_models:
            # Check budget for cloud models
            if not model.is_local:
                estimated_cost = (estimated_tokens / 1000) * model.cost_per_1k_tokens
                if self.daily_cloud_spend + estimated_cost > self.daily_budget:
                    continue  # Skip if over budget
            
            try:
                start = time.time()
                response = model.client.chat.completions.create(
                    model=model.name,
                    messages=messages
                )
                elapsed = time.time() - start
                
                # Track spending
                actual_tokens = response.usage.total_tokens
                if not model.is_local:
                    cost = (actual_tokens / 1000) * model.cost_per_1k_tokens
                    self.daily_cloud_spend += cost
                else:
                    cost = 0
                
                return {
                    "response": response.choices[0].message.content,
                    "model": model.name,
                    "is_local": model.is_local,
                    "tokens": actual_tokens,
                    "cost": cost,
                    "latency": elapsed
                }
                
            except Exception as e:
                continue  # Try next model on failure
        
        raise RuntimeError("No suitable model available")
```

### Pattern 4: Fallback Chain

Local primary with cloud fallback:

```python
from openai import OpenAI
import time

class FallbackRouter:
    """Local primary with cloud fallback."""
    
    def __init__(
        self,
        local_url: str = "http://localhost:11434/v1",
        cloud_api_key: str = None
    ):
        self.local_client = OpenAI(base_url=local_url, api_key="local")
        self.cloud_client = OpenAI(api_key=cloud_api_key) if cloud_api_key else None
        
        self.local_timeout = 30  # seconds
        self.local_failures = 0
        self.max_failures = 3
    
    def route(self, messages: list[dict]) -> dict:
        """Try local first, fallback to cloud."""
        
        # Check if local should be tried
        if self.local_failures < self.max_failures:
            try:
                start = time.time()
                response = self.local_client.chat.completions.create(
                    model="llama3.2:3b",
                    messages=messages,
                    timeout=self.local_timeout
                )
                
                # Reset failure count on success
                self.local_failures = 0
                
                return {
                    "response": response.choices[0].message.content,
                    "backend": "local",
                    "latency": time.time() - start
                }
                
            except Exception as e:
                self.local_failures += 1
                print(f"Local failed ({self.local_failures}/{self.max_failures}): {e}")
        
        # Fallback to cloud
        if self.cloud_client:
            start = time.time()
            response = self.cloud_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            
            return {
                "response": response.choices[0].message.content,
                "backend": "cloud",
                "latency": time.time() - start,
                "fallback": True
            }
        
        raise RuntimeError("All backends unavailable")
```

## Complete Hybrid Service

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal
import os

app = FastAPI()

# Initialize router
router = PrivacyRouter(
    local_base_url=os.getenv("LOCAL_URL", "http://localhost:11434/v1"),
    cloud_api_key=os.getenv("OPENAI_API_KEY")
)

class ChatRequest(BaseModel):
    messages: list[dict]
    routing_preference: Optional[Literal["auto", "local", "cloud"]] = "auto"
    min_quality: Optional[float] = 0.75

class ChatResponse(BaseModel):
    response: str
    routed_to: str
    reason: str
    model: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Hybrid chat endpoint with intelligent routing."""
    
    try:
        if request.routing_preference == "local":
            result = router.route(request.messages, force_local=True)
        elif request.routing_preference == "cloud":
            result = router.route_to_cloud(request.messages)
        else:
            result = router.route(request.messages)
        
        return ChatResponse(
            response=result["response"],
            routed_to=result["routed_to"],
            reason=result["reason"],
            model=result.get("model", "unknown")
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Check health of both backends."""
    
    local_ok = router.check_local_health()
    cloud_ok = router.check_cloud_health()
    
    return {
        "local": "healthy" if local_ok else "unhealthy",
        "cloud": "healthy" if cloud_ok else "unhealthy",
        "overall": "healthy" if (local_ok or cloud_ok) else "unhealthy"
    }
```

## Docker Compose for Hybrid

```yaml
# docker-compose.hybrid.yml
version: '3.8'

services:
  # Local LLM (Ollama)
  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_data:/root/.ollama
    ports:
      - "11434:11434"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Hybrid API Service
  hybrid-api:
    build: ./hybrid-service
    ports:
      - "8000:8000"
    environment:
      - LOCAL_URL=http://ollama:11434/v1
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    depends_on:
      - ollama

  # Nginx Load Balancer
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - hybrid-api

volumes:
  ollama_data:
```

## Best Practices

```yaml
hybrid_best_practices:
  routing:
    - "Always have fallback options"
    - "Log routing decisions for analysis"
    - "Monitor cost and quality metrics"
    - "Implement circuit breakers"
    
  security:
    - "Never send PII to cloud without explicit consent"
    - "Audit routing decisions"
    - "Encrypt data in transit"
    - "Use API keys securely"
    
  performance:
    - "Cache frequent queries locally"
    - "Pre-warm local models"
    - "Use async for cloud calls"
    - "Set appropriate timeouts"
    
  cost:
    - "Set daily/monthly budgets"
    - "Track per-request costs"
    - "Optimize prompt lengths"
    - "Use smaller models when possible"
```

## Summary

Hybrid architectures let you:
- ✅ Keep sensitive data local
- ✅ Access cutting-edge cloud models
- ✅ Optimize costs intelligently
- ✅ Ensure high availability
- ✅ Scale as needed

## Next Steps

1. **Monitoring** - Set up observability
2. **Caching** - Add semantic caching
3. **Fine-tuning** - Customize local models
