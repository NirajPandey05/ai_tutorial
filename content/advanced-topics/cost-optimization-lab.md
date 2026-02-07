# Lab: Building a Cost-Optimized LLM System

Build a complete cost-optimized LLM application combining semantic caching, prompt compression, and model routing.

---

## Learning Objectives

By the end of this lab, you will:
- Implement semantic caching to avoid redundant API calls
- Apply prompt compression to reduce token usage
- Build intelligent model routing based on query complexity
- Track and visualize cost savings
- Create a production-ready cost-optimized system

---

## Prerequisites

- Python 3.10+
- OpenAI API key
- Redis (for semantic cache)

```bash
pip install openai redis numpy scikit-learn tiktoken streamlit plotly
```

---

## Part 1: Building the Foundation

### Step 1: Cost Tracking System

```python
# cost_optimizer/tracker.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum
import json
import sqlite3

class ModelTier(str, Enum):
    CHEAP = "cheap"
    STANDARD = "standard"  
    PREMIUM = "premium"

# Pricing per 1K tokens (input, output)
MODEL_PRICING = {
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.005, 0.015),
    "gpt-4-turbo": (0.01, 0.03),
    "claude-3-haiku": (0.00025, 0.00125),
    "claude-3-sonnet": (0.003, 0.015),
    "claude-3-opus": (0.015, 0.075),
}

@dataclass
class CostEvent:
    """Single cost event"""
    timestamp: float
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    was_cached: bool = False
    was_compressed: bool = False
    original_tokens: Optional[int] = None
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": self.cost_usd,
            "was_cached": self.was_cached,
            "was_compressed": self.was_compressed,
            "original_tokens": self.original_tokens
        }

class CostTracker:
    """Track all costs and savings"""
    
    def __init__(self, db_path: str = "cost_tracking.db"):
        self.db_path = db_path
        self._init_db()
        
        # In-memory counters for quick access
        self.total_cost = 0.0
        self.total_saved = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        self.tokens_compressed = 0
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cost_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                model TEXT,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                cost_usd REAL,
                was_cached INTEGER,
                was_compressed INTEGER,
                original_tokens INTEGER,
                savings_usd REAL
            )
        """)
        conn.commit()
        conn.close()
    
    def record_llm_call(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        was_cached: bool = False,
        was_compressed: bool = False,
        original_tokens: int = None
    ) -> CostEvent:
        """Record an LLM call and calculate costs"""
        
        # Calculate actual cost
        input_price, output_price = MODEL_PRICING.get(model, (0.01, 0.03))
        actual_cost = (
            (prompt_tokens / 1000) * input_price +
            (completion_tokens / 1000) * output_price
        )
        
        # Calculate savings
        savings = 0.0
        
        if was_cached:
            # Saved full LLM call cost
            savings = actual_cost
            actual_cost = 0.0  # Cached response is free
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        if was_compressed and original_tokens:
            # Calculate what we would have paid
            original_cost = (original_tokens / 1000) * input_price
            compressed_cost = (prompt_tokens / 1000) * input_price
            savings += original_cost - compressed_cost
            self.tokens_compressed += (original_tokens - prompt_tokens)
        
        # Update totals
        self.total_cost += actual_cost
        self.total_saved += savings
        
        # Store in database
        event = CostEvent(
            timestamp=datetime.utcnow().timestamp(),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=actual_cost,
            was_cached=was_cached,
            was_compressed=was_compressed,
            original_tokens=original_tokens
        )
        
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO cost_events 
            (timestamp, model, prompt_tokens, completion_tokens, cost_usd,
             was_cached, was_compressed, original_tokens, savings_usd)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event.timestamp, event.model, event.prompt_tokens,
            event.completion_tokens, event.cost_usd, int(event.was_cached),
            int(event.was_compressed), event.original_tokens, savings
        ))
        conn.commit()
        conn.close()
        
        return event
    
    def get_summary(self) -> dict:
        """Get cost summary"""
        cache_total = self.cache_hits + self.cache_misses
        
        return {
            "total_cost": round(self.total_cost, 6),
            "total_saved": round(self.total_saved, 6),
            "savings_percentage": (
                self.total_saved / (self.total_cost + self.total_saved) * 100
                if (self.total_cost + self.total_saved) > 0 else 0
            ),
            "cache_hit_rate": (
                self.cache_hits / cache_total * 100
                if cache_total > 0 else 0
            ),
            "tokens_compressed": self.tokens_compressed
        }
    
    def get_hourly_costs(self, hours: int = 24) -> List[dict]:
        """Get hourly cost breakdown"""
        import time
        cutoff = time.time() - (hours * 3600)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            SELECT 
                CAST((timestamp / 3600) AS INTEGER) * 3600 as hour,
                SUM(cost_usd) as cost,
                SUM(savings_usd) as savings,
                COUNT(*) as requests
            FROM cost_events
            WHERE timestamp >= ?
            GROUP BY hour
            ORDER BY hour
        """, (cutoff,))
        
        results = [
            {"hour": row[0], "cost": row[1], "savings": row[2], "requests": row[3]}
            for row in cursor.fetchall()
        ]
        conn.close()
        
        return results
```

### Step 2: Semantic Cache

```python
# cost_optimizer/cache.py
import hashlib
import numpy as np
from typing import Optional, Tuple
from openai import OpenAI
import json

class SimpleSemanticCache:
    """In-memory semantic cache (use Redis for production)"""
    
    def __init__(
        self,
        similarity_threshold: float = 0.92,
        max_entries: int = 10000
    ):
        self.threshold = similarity_threshold
        self.max_entries = max_entries
        self.client = OpenAI()
        
        # Storage
        self.embeddings = []  # List of numpy arrays
        self.queries = []     # Original queries
        self.responses = []   # Cached responses
        self.models = []      # Models used
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return np.array(response.data[0].embedding)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def get(self, query: str, model: str = None) -> Optional[Tuple[str, float]]:
        """
        Get cached response if similar query exists.
        Returns (response, similarity) or None.
        """
        if not self.embeddings:
            return None
        
        query_embedding = self._get_embedding(query)
        
        best_similarity = 0.0
        best_idx = -1
        
        for idx, emb in enumerate(self.embeddings):
            # Optional: filter by model
            if model and self.models[idx] != model:
                continue
            
            similarity = self._cosine_similarity(query_embedding, emb)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_idx = idx
        
        if best_similarity >= self.threshold:
            return self.responses[best_idx], best_similarity
        
        return None
    
    def set(self, query: str, response: str, model: str):
        """Cache a query-response pair"""
        # Evict oldest if at capacity
        if len(self.embeddings) >= self.max_entries:
            self.embeddings.pop(0)
            self.queries.pop(0)
            self.responses.pop(0)
            self.models.pop(0)
        
        embedding = self._get_embedding(query)
        
        self.embeddings.append(embedding)
        self.queries.append(query)
        self.responses.append(response)
        self.models.append(model)
    
    @property
    def size(self) -> int:
        return len(self.embeddings)
```

### Step 3: Prompt Compressor

```python
# cost_optimizer/compressor.py
import tiktoken
from typing import Tuple

class SimpleCompressor:
    """Simple but effective prompt compression"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.encoding = tiktoken.encoding_for_model(model)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def compress_context(
        self,
        context: str,
        target_tokens: int = 1000,
        preserve_start: int = 200,
        preserve_end: int = 200
    ) -> Tuple[str, int]:
        """
        Compress context to target token count.
        Returns (compressed_text, original_tokens).
        """
        original_tokens = self.count_tokens(context)
        
        if original_tokens <= target_tokens:
            return context, original_tokens
        
        # Strategy: Keep start and end, summarize middle
        words = context.split()
        total_words = len(words)
        
        # Rough conversion: 1 token ≈ 0.75 words
        start_words = int(preserve_start * 0.75)
        end_words = int(preserve_end * 0.75)
        
        if total_words <= start_words + end_words:
            return context, original_tokens
        
        # Extract sections
        start = " ".join(words[:start_words])
        end = " ".join(words[-end_words:])
        middle = " ".join(words[start_words:-end_words])
        
        # Compress middle by extracting key sentences
        middle_compressed = self._extract_key_sentences(
            middle, 
            max_sentences=3
        )
        
        compressed = f"{start}\n[...key points: {middle_compressed}...]\n{end}"
        
        return compressed, original_tokens
    
    def _extract_key_sentences(self, text: str, max_sentences: int = 3) -> str:
        """Extract most important sentences"""
        import re
        sentences = re.split(r'[.!?]+', text)
        
        # Score sentences
        scored = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10:
                continue
            
            score = 0
            # Numbers are important
            score += len(re.findall(r'\d+', sent)) * 2
            # Proper nouns
            score += len(re.findall(r'[A-Z][a-z]+', sent))
            # Length bonus (not too short, not too long)
            words = len(sent.split())
            if 10 <= words <= 30:
                score += 1
            
            scored.append((sent, score))
        
        # Sort and take top
        scored.sort(key=lambda x: x[1], reverse=True)
        top = [s[0] for s in scored[:max_sentences]]
        
        return "; ".join(top)
    
    def compress_chat_history(
        self,
        messages: list,
        max_tokens: int = 2000
    ) -> Tuple[list, int]:
        """Compress chat history to fit token budget"""
        original_tokens = sum(
            self.count_tokens(m.get("content", ""))
            for m in messages
        )
        
        if original_tokens <= max_tokens:
            return messages, original_tokens
        
        # Strategy: Keep system and recent, summarize old
        system_msg = next(
            (m for m in messages if m["role"] == "system"),
            None
        )
        
        non_system = [m for m in messages if m["role"] != "system"]
        
        # Always keep last 4 messages
        recent = non_system[-4:] if len(non_system) > 4 else non_system
        older = non_system[:-4] if len(non_system) > 4 else []
        
        # Summarize older messages
        if older:
            summary = self._summarize_messages(older)
            summary_msg = {
                "role": "system",
                "content": f"Previous conversation summary: {summary}"
            }
            
            compressed = []
            if system_msg:
                compressed.append(system_msg)
            compressed.append(summary_msg)
            compressed.extend(recent)
            
            return compressed, original_tokens
        
        return messages, original_tokens
    
    def _summarize_messages(self, messages: list) -> str:
        """Create summary of messages"""
        # Simple extraction of key points
        points = []
        
        for msg in messages:
            content = msg.get("content", "")[:100]
            if msg["role"] == "user":
                points.append(f"User asked about: {content[:50]}...")
            else:
                points.append(f"Assistant: {content[:50]}...")
        
        return " | ".join(points[:5])
```

### Step 4: Model Router

```python
# cost_optimizer/router.py
from enum import Enum
import re
from typing import Tuple

class ModelTier(str, Enum):
    CHEAP = "cheap"
    STANDARD = "standard"
    PREMIUM = "premium"

MODEL_CONFIG = {
    ModelTier.CHEAP: "gpt-4o-mini",
    ModelTier.STANDARD: "gpt-4o",
    ModelTier.PREMIUM: "gpt-4-turbo"
}

class QueryRouter:
    """Route queries to appropriate model tier"""
    
    # Complexity patterns
    PREMIUM_PATTERNS = [
        r"analyze.*complex",
        r"compare.*multiple.*detail",
        r"legal|medical|financial.*analysis",
        r"write.*essay|article",
        r"research.*comprehensive",
    ]
    
    STANDARD_PATTERNS = [
        r"explain.*detail|depth",
        r"write.*code|function|class",
        r"debug|fix|refactor",
        r"summarize.*long",
        r"create.*plan|strategy",
    ]
    
    def classify(self, query: str) -> Tuple[ModelTier, str]:
        """
        Classify query complexity.
        Returns (tier, reason).
        """
        query_lower = query.lower()
        
        # Check premium patterns
        for pattern in self.PREMIUM_PATTERNS:
            if re.search(pattern, query_lower):
                return ModelTier.PREMIUM, f"Matched premium pattern: {pattern}"
        
        # Check standard patterns
        for pattern in self.STANDARD_PATTERNS:
            if re.search(pattern, query_lower):
                return ModelTier.STANDARD, f"Matched standard pattern: {pattern}"
        
        # Length-based heuristic
        if len(query.split()) > 100:
            return ModelTier.STANDARD, "Long query"
        
        return ModelTier.CHEAP, "Default to cheap"
    
    def get_model(self, tier: ModelTier) -> str:
        """Get model name for tier"""
        return MODEL_CONFIG[tier]
```

---

## Part 2: The Unified Cost Optimizer

### Step 5: Combine All Components

```python
# cost_optimizer/optimizer.py
from openai import OpenAI
from typing import Optional, Dict, Any
from .tracker import CostTracker, MODEL_PRICING
from .cache import SimpleSemanticCache
from .compressor import SimpleCompressor
from .router import QueryRouter, ModelTier

class CostOptimizedLLM:
    """LLM client with full cost optimization"""
    
    def __init__(
        self,
        enable_cache: bool = True,
        enable_compression: bool = True,
        enable_routing: bool = True,
        cache_threshold: float = 0.92,
        compression_target: int = 1000
    ):
        self.client = OpenAI()
        self.tracker = CostTracker()
        
        # Optional components
        self.cache = SimpleSemanticCache(cache_threshold) if enable_cache else None
        self.compressor = SimpleCompressor() if enable_compression else None
        self.router = QueryRouter() if enable_routing else None
        
        self.compression_target = compression_target
        
        # Feature flags
        self.enable_cache = enable_cache
        self.enable_compression = enable_compression
        self.enable_routing = enable_routing
    
    def chat(
        self,
        messages: list,
        model: str = None,
        context: str = None,
        use_cache: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send chat request with cost optimization.
        
        Args:
            messages: Chat messages
            model: Override model selection
            context: Additional context (will be compressed)
            use_cache: Whether to check cache
            **kwargs: Additional OpenAI parameters
        """
        
        # Extract user query for cache and routing
        user_query = self._get_user_query(messages)
        
        # Step 1: Check cache
        was_cached = False
        if self.enable_cache and use_cache and user_query:
            cached = self.cache.get(user_query, model)
            if cached:
                response, similarity = cached
                
                # Record as cache hit
                self.tracker.record_llm_call(
                    model=model or "gpt-4o-mini",
                    prompt_tokens=self.compressor.count_tokens(user_query) if self.compressor else 100,
                    completion_tokens=self.compressor.count_tokens(response) if self.compressor else 100,
                    was_cached=True
                )
                
                return {
                    "content": response,
                    "cached": True,
                    "cache_similarity": similarity,
                    "model": "cached",
                    "cost": 0.0
                }
        
        # Step 2: Route to appropriate model
        selected_model = model
        routing_reason = "User specified"
        
        if not model and self.enable_routing and user_query:
            tier, routing_reason = self.router.classify(user_query)
            selected_model = self.router.get_model(tier)
        elif not model:
            selected_model = "gpt-4o-mini"
        
        # Step 3: Compress context if provided
        original_tokens = None
        was_compressed = False
        
        if context and self.enable_compression:
            compressed_context, original_tokens = self.compressor.compress_context(
                context,
                target_tokens=self.compression_target
            )
            
            if original_tokens > self.compression_target:
                was_compressed = True
                # Add compressed context to messages
                messages = self._add_context(messages, compressed_context)
        elif context:
            messages = self._add_context(messages, context)
        
        # Step 4: Compress chat history if needed
        if self.enable_compression:
            messages, _ = self.compressor.compress_chat_history(
                messages,
                max_tokens=2000
            )
        
        # Step 5: Make API call
        response = self.client.chat.completions.create(
            model=selected_model,
            messages=messages,
            **kwargs
        )
        
        content = response.choices[0].message.content
        usage = response.usage
        
        # Step 6: Calculate and record cost
        input_price, output_price = MODEL_PRICING.get(selected_model, (0.01, 0.03))
        cost = (
            (usage.prompt_tokens / 1000) * input_price +
            (usage.completion_tokens / 1000) * output_price
        )
        
        self.tracker.record_llm_call(
            model=selected_model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            was_cached=False,
            was_compressed=was_compressed,
            original_tokens=original_tokens
        )
        
        # Step 7: Cache response
        if self.enable_cache and use_cache and user_query:
            self.cache.set(user_query, content, selected_model)
        
        return {
            "content": content,
            "cached": False,
            "model": selected_model,
            "routing_reason": routing_reason,
            "was_compressed": was_compressed,
            "tokens_saved": (original_tokens - usage.prompt_tokens) if was_compressed else 0,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "cost": cost
        }
    
    def _get_user_query(self, messages: list) -> Optional[str]:
        """Extract user query from messages"""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content")
        return None
    
    def _add_context(self, messages: list, context: str) -> list:
        """Add context to system message"""
        messages = list(messages)  # Copy
        
        # Find or create system message
        system_idx = next(
            (i for i, m in enumerate(messages) if m["role"] == "system"),
            None
        )
        
        if system_idx is not None:
            messages[system_idx] = {
                "role": "system",
                "content": f"{messages[system_idx]['content']}\n\nContext:\n{context}"
            }
        else:
            messages.insert(0, {
                "role": "system",
                "content": f"Context:\n{context}"
            })
        
        return messages
    
    def get_stats(self) -> dict:
        """Get optimization statistics"""
        tracker_summary = self.tracker.get_summary()
        
        return {
            **tracker_summary,
            "cache_size": self.cache.size if self.cache else 0,
            "features_enabled": {
                "cache": self.enable_cache,
                "compression": self.enable_compression,
                "routing": self.enable_routing
            }
        }
```

---

## Part 3: Testing the System

### Step 6: Demo Application

```python
# demo.py
from cost_optimizer.optimizer import CostOptimizedLLM
import time

def run_demo():
    """Demonstrate cost optimization"""
    
    print("=" * 60)
    print("Cost-Optimized LLM Demo")
    print("=" * 60)
    
    # Initialize optimizer
    llm = CostOptimizedLLM(
        enable_cache=True,
        enable_compression=True,
        enable_routing=True
    )
    
    # Test queries
    test_cases = [
        {
            "name": "Simple query (should use cheap model)",
            "messages": [{"role": "user", "content": "What is 2 + 2?"}]
        },
        {
            "name": "Same query (should hit cache)",
            "messages": [{"role": "user", "content": "What is 2 plus 2?"}]
        },
        {
            "name": "Code query (should use standard model)",
            "messages": [{"role": "user", "content": "Write a Python function to sort a list"}]
        },
        {
            "name": "Complex query (should use premium model)",
            "messages": [{"role": "user", "content": "Analyze the complex legal implications of AI liability"}]
        },
        {
            "name": "Query with long context (should compress)",
            "messages": [{"role": "user", "content": "Summarize this document"}],
            "context": """
            This is a very long document that contains lots of information about various topics.
            The quarterly financial report shows significant growth across all business segments.
            Revenue increased by 15% compared to the previous quarter, reaching $2.3 billion.
            The technology division led with 22% growth while services showed 8% improvement.
            Operating margins improved by 2 points to 18% driven by cost optimization.
            Cash position remains strong at $5.2 billion for strategic investments.
            Employee headcount grew 5% to support expansion in Asia-Pacific and Latin America.
            The company launched three new products in the enterprise software category.
            Customer satisfaction scores improved to 4.7 out of 5 stars.
            The partnership with major cloud providers expanded to include new services.
            """ * 10  # Repeat to make it long
        }
    ]
    
    for i, test in enumerate(test_cases):
        print(f"\n--- Test {i+1}: {test['name']} ---")
        
        result = llm.chat(
            messages=test["messages"],
            context=test.get("context")
        )
        
        print(f"Model: {result['model']}")
        print(f"Cached: {result['cached']}")
        
        if result.get('cache_similarity'):
            print(f"Cache similarity: {result['cache_similarity']:.2f}")
        
        if result.get('routing_reason'):
            print(f"Routing: {result['routing_reason']}")
        
        if result.get('was_compressed'):
            print(f"Compressed: Saved {result['tokens_saved']} tokens")
        
        print(f"Cost: ${result['cost']:.6f}")
        print(f"Response: {result['content'][:100]}...")
        
        time.sleep(1)  # Rate limiting
    
    # Print final stats
    print("\n" + "=" * 60)
    print("Final Statistics")
    print("=" * 60)
    
    stats = llm.get_stats()
    print(f"Total Cost: ${stats['total_cost']:.6f}")
    print(f"Total Saved: ${stats['total_saved']:.6f}")
    print(f"Savings: {stats['savings_percentage']:.1f}%")
    print(f"Cache Hit Rate: {stats['cache_hit_rate']:.1f}%")
    print(f"Tokens Compressed: {stats['tokens_compressed']}")

if __name__ == "__main__":
    run_demo()
```

---

## Part 4: Monitoring Dashboard

### Step 7: Streamlit Dashboard

```python
# dashboard.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
from cost_optimizer.tracker import CostTracker

st.set_page_config(
    page_title="LLM Cost Optimizer Dashboard",
    page_icon="💰",
    layout="wide"
)

st.title("💰 LLM Cost Optimizer Dashboard")

# Initialize tracker
tracker = CostTracker()

# Key Metrics
st.header("📊 Key Metrics")
stats = tracker.get_summary()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Cost",
        f"${stats['total_cost']:.4f}",
        help="Total API costs"
    )

with col2:
    st.metric(
        "Total Saved",
        f"${stats['total_saved']:.4f}",
        help="Savings from optimization"
    )

with col3:
    st.metric(
        "Savings %",
        f"{stats['savings_percentage']:.1f}%",
        help="Percentage saved"
    )

with col4:
    st.metric(
        "Cache Hit Rate",
        f"{stats['cache_hit_rate']:.1f}%",
        help="Cache effectiveness"
    )

# Charts
st.header("📈 Cost Over Time")

hourly_data = tracker.get_hourly_costs(hours=24)

if hourly_data:
    df = pd.DataFrame(hourly_data)
    df['hour'] = pd.to_datetime(df['hour'], unit='s')
    
    # Stacked bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['hour'],
        y=df['cost'],
        name='Actual Cost',
        marker_color='#FF6B6B'
    ))
    
    fig.add_trace(go.Bar(
        x=df['hour'],
        y=df['savings'],
        name='Savings',
        marker_color='#4ECDC4'
    ))
    
    fig.update_layout(
        barmode='stack',
        title='Hourly Cost Breakdown',
        xaxis_title='Time',
        yaxis_title='Cost (USD)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No cost data available yet")

# Optimization Tips
st.header("💡 Optimization Tips")

tips = [
    "**Increase cache threshold** if hit rate is low but similar queries exist",
    "**Adjust compression target** based on context importance",
    "**Review routing patterns** to catch misclassified queries",
    "**Monitor cache size** and implement eviction policies"
]

for tip in tips:
    st.markdown(f"- {tip}")

# Real-time testing
st.header("🧪 Test Query")

test_query = st.text_area("Enter a query to test:")

if st.button("Analyze Query"):
    if test_query:
        from cost_optimizer.router import QueryRouter
        router = QueryRouter()
        tier, reason = router.classify(test_query)
        
        st.write(f"**Tier:** {tier.value}")
        st.write(f"**Reason:** {reason}")
        st.write(f"**Estimated cost:** ${0.0001 if tier.value == 'cheap' else 0.005 if tier.value == 'standard' else 0.02:.4f}")

st.divider()
st.caption("LLM Cost Optimizer | Built with ❤️ and Python")
```

---

## Running the Lab

```bash
# Terminal 1: Run demo
python demo.py

# Terminal 2: Run dashboard
streamlit run dashboard.py
```

---

## Challenge Exercises

### Challenge 1: Add Redis Cache
Replace the in-memory cache with Redis for production scalability.

### Challenge 2: Implement Cascading
Add fallback to more expensive models when cheap models fail quality checks.

### Challenge 3: A/B Testing
Implement feature flags to A/B test different optimization strategies.

### Challenge 4: Export to Prometheus
Add Prometheus metrics export for monitoring integration.

---

## Key Takeaways

1. **Semantic caching** alone can save 70-90% for FAQ-style applications
2. **Prompt compression** typically saves 30-50% on context-heavy workloads
3. **Model routing** provides 50-80% savings by using appropriate tiers
4. **Combined optimization** can achieve 80-95% cost reduction
5. **Always monitor** - optimization effectiveness varies by use case

---

## Production Considerations

1. **Use Redis** for distributed caching
2. **Implement TTLs** for cache freshness
3. **Monitor quality** - ensure optimizations don't degrade output
4. **Set up alerts** for cost anomalies
5. **Regular audits** of routing decisions

---

## Next Steps

- Deploy optimized system to production
- Set up monitoring dashboards
- Implement cost alerts
- Review and tune parameters monthly
