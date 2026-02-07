# Model Routing Strategies

Learn to intelligently route requests between cheap and expensive models to optimize cost while maintaining quality.

---

## Why Model Routing?

Not all queries require GPT-4. Many can be handled by cheaper models:

| Query Type | Complexity | Recommended Model | Cost/1K tokens |
|------------|------------|-------------------|----------------|
| Simple Q&A | Low | GPT-4o-mini | $0.00015 |
| Summarization | Medium | GPT-4o-mini | $0.00015 |
| Code Generation | High | GPT-4o | $0.005 |
| Complex Reasoning | Very High | Claude 3 Opus | $0.015 |

**Potential savings**: 60-80% by routing appropriately.

---

## Basic Model Router

```python
from openai import OpenAI
from anthropic import Anthropic
from enum import Enum
from typing import Optional
import re

class ModelTier(str, Enum):
    CHEAP = "cheap"      # Simple tasks
    STANDARD = "standard" # Most tasks
    PREMIUM = "premium"   # Complex tasks

class ModelConfig:
    """Model configurations by tier"""
    
    MODELS = {
        ModelTier.CHEAP: {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "cost_per_1k_input": 0.00015,
            "cost_per_1k_output": 0.0006,
            "max_tokens": 4096
        },
        ModelTier.STANDARD: {
            "provider": "openai",
            "model": "gpt-4o",
            "cost_per_1k_input": 0.005,
            "cost_per_1k_output": 0.015,
            "max_tokens": 4096
        },
        ModelTier.PREMIUM: {
            "provider": "anthropic",
            "model": "claude-3-opus-20240229",
            "cost_per_1k_input": 0.015,
            "cost_per_1k_output": 0.075,
            "max_tokens": 4096
        }
    }

class SimpleModelRouter:
    """Route queries to appropriate model tier"""
    
    def __init__(self):
        self.openai = OpenAI()
        self.anthropic = Anthropic()
    
    def classify_query(self, query: str) -> ModelTier:
        """Classify query complexity"""
        query_lower = query.lower()
        
        # Premium indicators
        premium_patterns = [
            r"analyze.*complex",
            r"multi.?step.*reason",
            r"compare.*contrast.*multiple",
            r"write.*essay",
            r"legal.*analysis",
            r"medical.*diagnosis",
        ]
        
        for pattern in premium_patterns:
            if re.search(pattern, query_lower):
                return ModelTier.PREMIUM
        
        # Standard indicators
        standard_patterns = [
            r"explain.*detail",
            r"write.*code",
            r"debug",
            r"refactor",
            r"create.*function",
            r"implement",
        ]
        
        for pattern in standard_patterns:
            if re.search(pattern, query_lower):
                return ModelTier.STANDARD
        
        # Default to cheap for simple queries
        return ModelTier.CHEAP
    
    def route(
        self,
        messages: list,
        tier_override: ModelTier = None,
        **kwargs
    ) -> dict:
        """Route request to appropriate model"""
        
        # Extract user query for classification
        user_query = next(
            (m["content"] for m in messages if m["role"] == "user"),
            ""
        )
        
        # Determine tier
        tier = tier_override or self.classify_query(user_query)
        config = ModelConfig.MODELS[tier]
        
        # Call appropriate provider
        if config["provider"] == "openai":
            response = self._call_openai(messages, config, **kwargs)
        else:
            response = self._call_anthropic(messages, config, **kwargs)
        
        return {
            "content": response["content"],
            "model": config["model"],
            "tier": tier.value,
            "usage": response["usage"]
        }
    
    def _call_openai(self, messages: list, config: dict, **kwargs) -> dict:
        response = self.openai.chat.completions.create(
            model=config["model"],
            messages=messages,
            max_tokens=kwargs.get("max_tokens", config["max_tokens"])
        )
        
        return {
            "content": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            }
        }
    
    def _call_anthropic(self, messages: list, config: dict, **kwargs) -> dict:
        # Convert to Anthropic format
        system = next(
            (m["content"] for m in messages if m["role"] == "system"),
            None
        )
        
        anthropic_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in messages if m["role"] != "system"
        ]
        
        response = self.anthropic.messages.create(
            model=config["model"],
            messages=anthropic_messages,
            system=system or "",
            max_tokens=kwargs.get("max_tokens", config["max_tokens"])
        )
        
        return {
            "content": response.content[0].text,
            "usage": {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens
            }
        }

# Usage
router = SimpleModelRouter()

# Simple query -> cheap model
result = router.route([
    {"role": "user", "content": "What is the capital of France?"}
])
print(f"Tier: {result['tier']}, Model: {result['model']}")
# Output: Tier: cheap, Model: gpt-4o-mini

# Complex query -> premium model
result = router.route([
    {"role": "user", "content": "Analyze this complex legal contract and identify potential issues"}
])
print(f"Tier: {result['tier']}, Model: {result['model']}")
# Output: Tier: premium, Model: claude-3-opus-20240229
```

---

## ML-Based Query Classification

### Train a Classifier

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from typing import Tuple
import pickle

class MLQueryClassifier:
    """ML-based query complexity classifier"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2)
        )
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        self.is_trained = False
    
    def train(self, queries: list, labels: list):
        """Train classifier on labeled data"""
        X = self.vectorizer.fit_transform(queries)
        self.classifier.fit(X, labels)
        self.is_trained = True
    
    def predict(self, query: str) -> Tuple[ModelTier, float]:
        """Predict model tier with confidence"""
        if not self.is_trained:
            raise ValueError("Classifier not trained")
        
        X = self.vectorizer.transform([query])
        
        # Get prediction and probability
        prediction = self.classifier.predict(X)[0]
        probabilities = self.classifier.predict_proba(X)[0]
        confidence = max(probabilities)
        
        tier = ModelTier(prediction)
        return tier, confidence
    
    def save(self, path: str):
        """Save trained model"""
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'classifier': self.classifier
            }, f)
    
    def load(self, path: str):
        """Load trained model"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.vectorizer = data['vectorizer']
            self.classifier = data['classifier']
            self.is_trained = True

# Training data example
training_data = [
    # Cheap tier
    ("What is 2+2?", "cheap"),
    ("Define photosynthesis", "cheap"),
    ("What time is it in Tokyo?", "cheap"),
    ("Translate hello to Spanish", "cheap"),
    
    # Standard tier
    ("Write a Python function to sort a list", "standard"),
    ("Explain how neural networks work", "standard"),
    ("Debug this code snippet", "standard"),
    ("Summarize this article", "standard"),
    
    # Premium tier
    ("Analyze the philosophical implications of AI consciousness", "premium"),
    ("Compare and contrast three different economic theories", "premium"),
    ("Write a detailed legal analysis of this contract", "premium"),
    ("Create a comprehensive business strategy", "premium"),
]

queries = [t[0] for t in training_data]
labels = [t[1] for t in training_data]

classifier = MLQueryClassifier()
classifier.train(queries, labels)

# Predict
tier, confidence = classifier.predict("Write a simple hello world program")
print(f"Tier: {tier}, Confidence: {confidence:.2f}")
```

### LLM-Based Classification

```python
from openai import OpenAI

class LLMQueryClassifier:
    """Use LLM to classify query complexity"""
    
    CLASSIFICATION_PROMPT = """Classify the following query into one of three complexity tiers:

CHEAP: Simple factual questions, translations, basic math, definitions
STANDARD: Code writing, explanations, summaries, moderate analysis
PREMIUM: Complex reasoning, multi-step analysis, creative writing, specialized domains

Query: {query}

Respond with exactly one word: CHEAP, STANDARD, or PREMIUM"""
    
    def __init__(self):
        self.client = OpenAI()
        # Use cheap model for classification itself
        self.classifier_model = "gpt-4o-mini"
    
    def classify(self, query: str) -> ModelTier:
        """Classify query using LLM"""
        response = self.client.chat.completions.create(
            model=self.classifier_model,
            messages=[{
                "role": "user",
                "content": self.CLASSIFICATION_PROMPT.format(query=query)
            }],
            max_tokens=10,
            temperature=0
        )
        
        result = response.choices[0].message.content.strip().upper()
        
        tier_map = {
            "CHEAP": ModelTier.CHEAP,
            "STANDARD": ModelTier.STANDARD,
            "PREMIUM": ModelTier.PREMIUM
        }
        
        return tier_map.get(result, ModelTier.STANDARD)
```

---

## Cascading Model Strategy

Try cheaper model first, escalate if needed:

```python
from openai import OpenAI
from typing import Optional, Callable

class CascadingRouter:
    """Try cheap model first, escalate if quality check fails"""
    
    def __init__(
        self,
        quality_checker: Callable[[str, str], bool] = None,
        escalation_threshold: float = 0.7
    ):
        self.client = OpenAI()
        self.quality_checker = quality_checker or self._default_quality_check
        self.threshold = escalation_threshold
        
        # Track escalation rate
        self.total_requests = 0
        self.escalations = 0
    
    def route(
        self,
        messages: list,
        tiers: list = None,
        **kwargs
    ) -> dict:
        """Try models in order, escalating if quality is poor"""
        
        tiers = tiers or [ModelTier.CHEAP, ModelTier.STANDARD, ModelTier.PREMIUM]
        
        self.total_requests += 1
        user_query = next(
            (m["content"] for m in messages if m["role"] == "user"),
            ""
        )
        
        for i, tier in enumerate(tiers):
            config = ModelConfig.MODELS[tier]
            
            response = self.client.chat.completions.create(
                model=config["model"],
                messages=messages,
                **kwargs
            )
            
            content = response.choices[0].message.content
            
            # Last tier - must accept
            if i == len(tiers) - 1:
                return self._format_response(content, tier, response.usage)
            
            # Check quality
            is_good = self.quality_checker(user_query, content)
            
            if is_good:
                return self._format_response(content, tier, response.usage)
            
            # Escalate
            self.escalations += 1
        
        # Should never reach here
        return self._format_response(content, tiers[-1], response.usage)
    
    def _default_quality_check(self, query: str, response: str) -> bool:
        """Default quality check using LLM"""
        check_response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Rate if this response adequately answers the query (0-1).

Query: {query}
Response: {response}

Respond with just a number between 0 and 1."""
            }],
            max_tokens=10,
            temperature=0
        )
        
        try:
            score = float(check_response.choices[0].message.content.strip())
            return score >= self.threshold
        except:
            return True  # Accept on parse error
    
    def _format_response(self, content: str, tier: ModelTier, usage) -> dict:
        return {
            "content": content,
            "tier": tier.value,
            "model": ModelConfig.MODELS[tier]["model"],
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens
            }
        }
    
    @property
    def escalation_rate(self) -> float:
        if self.total_requests == 0:
            return 0
        return self.escalations / self.total_requests

# Usage
router = CascadingRouter(escalation_threshold=0.7)

result = router.route([
    {"role": "user", "content": "Explain quantum computing in simple terms"}
])

print(f"Used tier: {result['tier']}")
print(f"Escalation rate: {router.escalation_rate:.1%}")
```

---

## Context-Aware Routing

### Route Based on Conversation Context

```python
class ContextAwareRouter:
    """Route based on conversation context and history"""
    
    def __init__(self):
        self.client = OpenAI()
        self.conversation_tier = ModelTier.CHEAP
        self.turn_count = 0
    
    def route(self, messages: list, **kwargs) -> dict:
        """Route considering full conversation context"""
        
        self.turn_count += 1
        
        # Analyze conversation
        tier = self._determine_tier(messages)
        
        # Remember highest tier used (sticky escalation)
        if self._tier_value(tier) > self._tier_value(self.conversation_tier):
            self.conversation_tier = tier
        
        config = ModelConfig.MODELS[self.conversation_tier]
        
        response = self.client.chat.completions.create(
            model=config["model"],
            messages=messages,
            **kwargs
        )
        
        return {
            "content": response.choices[0].message.content,
            "tier": self.conversation_tier.value,
            "model": config["model"]
        }
    
    def _determine_tier(self, messages: list) -> ModelTier:
        """Determine tier based on conversation analysis"""
        
        # Count indicators
        code_blocks = sum(
            1 for m in messages 
            if "```" in m.get("content", "")
        )
        
        message_length = sum(
            len(m.get("content", "").split()) 
            for m in messages
        )
        
        # Long technical conversations need better models
        if code_blocks > 3 or message_length > 1000:
            return ModelTier.PREMIUM
        
        if code_blocks > 0 or message_length > 300:
            return ModelTier.STANDARD
        
        return ModelTier.CHEAP
    
    def _tier_value(self, tier: ModelTier) -> int:
        """Numeric value for tier comparison"""
        return {
            ModelTier.CHEAP: 1,
            ModelTier.STANDARD: 2,
            ModelTier.PREMIUM: 3
        }[tier]
    
    def reset_conversation(self):
        """Reset for new conversation"""
        self.conversation_tier = ModelTier.CHEAP
        self.turn_count = 0
```

### Domain-Specific Routing

```python
class DomainRouter:
    """Route based on detected domain"""
    
    DOMAIN_MODELS = {
        "code": {
            "model": "gpt-4o",
            "tier": ModelTier.STANDARD
        },
        "creative": {
            "model": "claude-3-opus-20240229",
            "tier": ModelTier.PREMIUM
        },
        "math": {
            "model": "gpt-4o",
            "tier": ModelTier.STANDARD
        },
        "general": {
            "model": "gpt-4o-mini",
            "tier": ModelTier.CHEAP
        }
    }
    
    DOMAIN_PATTERNS = {
        "code": [
            r"python|javascript|java|code|function|class|debug|api",
            r"implement|refactor|syntax|compile|runtime"
        ],
        "creative": [
            r"write.*story|poem|creative|imagine|fiction",
            r"compose|narrative|character|plot"
        ],
        "math": [
            r"calculate|equation|formula|solve|math|algebra",
            r"calculus|statistics|probability"
        ]
    }
    
    def __init__(self):
        self.openai = OpenAI()
        self.anthropic = Anthropic()
    
    def detect_domain(self, query: str) -> str:
        """Detect query domain"""
        query_lower = query.lower()
        
        for domain, patterns in self.DOMAIN_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return domain
        
        return "general"
    
    def route(self, messages: list, **kwargs) -> dict:
        """Route to domain-specific model"""
        
        user_query = next(
            (m["content"] for m in messages if m["role"] == "user"),
            ""
        )
        
        domain = self.detect_domain(user_query)
        config = self.DOMAIN_MODELS[domain]
        
        # Call appropriate model
        if "gpt" in config["model"]:
            response = self.openai.chat.completions.create(
                model=config["model"],
                messages=messages,
                **kwargs
            )
            content = response.choices[0].message.content
        else:
            # Anthropic
            system = next(
                (m["content"] for m in messages if m["role"] == "system"),
                None
            )
            anthropic_messages = [
                m for m in messages if m["role"] != "system"
            ]
            
            response = self.anthropic.messages.create(
                model=config["model"],
                messages=anthropic_messages,
                system=system or "",
                max_tokens=kwargs.get("max_tokens", 4096)
            )
            content = response.content[0].text
        
        return {
            "content": content,
            "domain": domain,
            "model": config["model"],
            "tier": config["tier"].value
        }
```

---

## Cost Tracking and Optimization

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List

@dataclass
class RoutingMetrics:
    """Track routing decisions and costs"""
    
    requests_by_tier: Dict[str, int] = field(default_factory=lambda: {
        "cheap": 0, "standard": 0, "premium": 0
    })
    tokens_by_tier: Dict[str, int] = field(default_factory=lambda: {
        "cheap": 0, "standard": 0, "premium": 0
    })
    costs_by_tier: Dict[str, float] = field(default_factory=lambda: {
        "cheap": 0, "standard": 0, "premium": 0
    })
    
    def record(
        self,
        tier: str,
        prompt_tokens: int,
        completion_tokens: int
    ):
        """Record a routing decision"""
        self.requests_by_tier[tier] += 1
        
        total_tokens = prompt_tokens + completion_tokens
        self.tokens_by_tier[tier] += total_tokens
        
        # Calculate cost
        config = ModelConfig.MODELS[ModelTier(tier)]
        cost = (
            (prompt_tokens / 1000) * config["cost_per_1k_input"] +
            (completion_tokens / 1000) * config["cost_per_1k_output"]
        )
        self.costs_by_tier[tier] += cost
    
    @property
    def total_cost(self) -> float:
        return sum(self.costs_by_tier.values())
    
    @property
    def cost_without_routing(self) -> float:
        """Estimate cost if all requests used premium tier"""
        premium_config = ModelConfig.MODELS[ModelTier.PREMIUM]
        total_tokens = sum(self.tokens_by_tier.values())
        # Rough estimate assuming 50/50 input/output split
        return (total_tokens / 1000) * (
            premium_config["cost_per_1k_input"] + 
            premium_config["cost_per_1k_output"]
        ) / 2
    
    @property
    def savings(self) -> float:
        return self.cost_without_routing - self.total_cost
    
    @property
    def savings_percentage(self) -> float:
        if self.cost_without_routing == 0:
            return 0
        return self.savings / self.cost_without_routing
    
    def summary(self) -> dict:
        return {
            "total_requests": sum(self.requests_by_tier.values()),
            "requests_by_tier": self.requests_by_tier,
            "total_cost": f"${self.total_cost:.4f}",
            "cost_without_routing": f"${self.cost_without_routing:.4f}",
            "savings": f"${self.savings:.4f}",
            "savings_percentage": f"{self.savings_percentage:.1%}"
        }

# Usage with router
class MetricsRouter:
    """Router with built-in metrics tracking"""
    
    def __init__(self):
        self.router = SimpleModelRouter()
        self.metrics = RoutingMetrics()
    
    def route(self, messages: list, **kwargs) -> dict:
        result = self.router.route(messages, **kwargs)
        
        # Record metrics
        self.metrics.record(
            tier=result["tier"],
            prompt_tokens=result["usage"]["prompt_tokens"],
            completion_tokens=result["usage"]["completion_tokens"]
        )
        
        return result
    
    def get_metrics(self) -> dict:
        return self.metrics.summary()

# Example
router = MetricsRouter()

# Simulate various queries
queries = [
    "What is 2+2?",  # cheap
    "Write a Python sorting function",  # standard
    "What day is today?",  # cheap
    "Explain quantum entanglement",  # standard
    "Analyze this legal contract deeply",  # premium
]

for query in queries:
    router.route([{"role": "user", "content": query}])

print(router.get_metrics())
# {
#     "total_requests": 5,
#     "requests_by_tier": {"cheap": 2, "standard": 2, "premium": 1},
#     "total_cost": "$0.0234",
#     "cost_without_routing": "$0.1250",
#     "savings": "$0.1016",
#     "savings_percentage": "81.3%"
# }
```

---

## Best Practices

### 1. Start Conservative

```python
# Start with cheap, escalate based on data
DEFAULT_TIER = ModelTier.CHEAP

# Collect data on escalation patterns
# Adjust routing rules based on actual quality metrics
```

### 2. Monitor Quality

```python
def monitor_quality(tier: str, query: str, response: str, user_rating: int):
    """Track quality by tier for optimization"""
    # Store in database
    # Analyze patterns
    # Adjust routing thresholds
    pass
```

### 3. Use Feature Flags

```python
class FeatureFlagRouter:
    """A/B test routing strategies"""
    
    def __init__(self, experiment_percentage: float = 0.1):
        self.experiment_pct = experiment_percentage
    
    def route(self, messages: list, user_id: str, **kwargs):
        import random
        
        # Deterministic assignment based on user_id
        random.seed(hash(user_id))
        
        if random.random() < self.experiment_pct:
            # Experiment: always use premium
            tier = ModelTier.PREMIUM
        else:
            # Control: use smart routing
            tier = self._classify(messages)
        
        return tier
```

---

## Key Takeaways

1. **60-80% savings** possible with intelligent routing
2. **Rule-based** routing works for clear patterns
3. **ML classification** improves with data
4. **Cascading** ensures quality while minimizing cost
5. **Monitor continuously** - patterns change over time

---

## Next Steps

- [Semantic Caching](/learn/advanced-topics/optimization/semantic-caching) - Cache responses
- [Prompt Compression](/learn/advanced-topics/optimization/prompt-compression) - Reduce token usage
- [Cost Optimization Lab](/learn/advanced-topics/optimization/cost-optimization-lab) - Hands-on implementation
