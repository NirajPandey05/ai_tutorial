# Custom Logging and Tracing for LLM Applications

Build a comprehensive custom logging system tailored to your LLM application's specific needs.

---

## Why Custom Logging?

While tools like LangSmith and Phoenix are powerful, you might need custom logging for:

- **Privacy**: Keep all data on-premises
- **Cost**: Avoid third-party service fees
- **Customization**: Log exactly what you need
- **Integration**: Connect with existing monitoring tools
- **Compliance**: Meet specific regulatory requirements

---

## Structured Logging Foundation

### Basic Structure

```python
import logging
import json
from datetime import datetime
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict
import uuid

@dataclass
class LLMLogEntry:
    """Structured log entry for LLM operations"""
    timestamp: str
    trace_id: str
    span_id: str
    operation: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0
    status: str = "success"
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())

class LLMLogger:
    """Structured logger for LLM applications"""
    
    def __init__(self, service_name: str, log_level: int = logging.INFO):
        self.service_name = service_name
        self.logger = logging.getLogger(f"llm.{service_name}")
        self.logger.setLevel(log_level)
        
        # JSON formatter
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        self.logger.addHandler(handler)
    
    def log_llm_call(
        self,
        operation: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        trace_id: str = None,
        span_id: str = None,
        metadata: dict = None
    ):
        """Log an LLM API call"""
        entry = LLMLogEntry(
            timestamp=datetime.utcnow().isoformat(),
            trace_id=trace_id or str(uuid.uuid4()),
            span_id=span_id or str(uuid.uuid4()),
            operation=operation,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            metadata=metadata
        )
        
        self.logger.info(entry.to_json())
    
    def log_error(
        self,
        operation: str,
        error: str,
        trace_id: str = None,
        metadata: dict = None
    ):
        """Log an error"""
        entry = LLMLogEntry(
            timestamp=datetime.utcnow().isoformat(),
            trace_id=trace_id or str(uuid.uuid4()),
            span_id=str(uuid.uuid4()),
            operation=operation,
            model="",
            status="error",
            error=error,
            metadata=metadata
        )
        
        self.logger.error(entry.to_json())

class JsonFormatter(logging.Formatter):
    """Format logs as JSON"""
    
    def format(self, record: logging.LogRecord) -> str:
        try:
            # If message is already JSON, use it directly
            return record.getMessage()
        except:
            # Fallback to standard formatting
            return json.dumps({
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "message": record.getMessage()
            })
```

---

## Comprehensive Tracing System

### Trace Context Management

```python
from contextvars import ContextVar
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import time
import uuid

# Context variables for trace propagation
current_trace: ContextVar[Optional['TraceContext']] = ContextVar('current_trace', default=None)
current_span: ContextVar[Optional['SpanContext']] = ContextVar('current_span', default=None)

@dataclass
class SpanContext:
    """Span within a trace"""
    span_id: str
    parent_id: Optional[str]
    name: str
    start_time: float
    end_time: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict] = field(default_factory=list)
    status: str = "UNSET"
    error_message: Optional[str] = None
    
    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0
        return (self.end_time - self.start_time) * 1000

@dataclass
class TraceContext:
    """Complete trace context"""
    trace_id: str
    root_span_id: Optional[str] = None
    spans: List[SpanContext] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_ms(self) -> float:
        if not self.spans:
            return 0
        start = min(s.start_time for s in self.spans)
        end = max(s.end_time or time.time() for s in self.spans)
        return (end - start) * 1000

class TracingManager:
    """Manage distributed tracing"""
    
    def __init__(self, storage: 'TraceStorage' = None):
        self.storage = storage or InMemoryTraceStorage()
    
    def start_trace(self, metadata: Dict = None) -> TraceContext:
        """Start a new trace"""
        trace = TraceContext(
            trace_id=str(uuid.uuid4()),
            metadata=metadata or {}
        )
        current_trace.set(trace)
        return trace
    
    def start_span(
        self,
        name: str,
        attributes: Dict = None
    ) -> SpanContext:
        """Start a new span"""
        trace = current_trace.get()
        parent = current_span.get()
        
        span = SpanContext(
            span_id=str(uuid.uuid4()),
            parent_id=parent.span_id if parent else None,
            name=name,
            start_time=time.time(),
            attributes=attributes or {}
        )
        
        if trace:
            trace.spans.append(span)
            if not trace.root_span_id:
                trace.root_span_id = span.span_id
        
        current_span.set(span)
        return span
    
    def end_span(self, span: SpanContext, status: str = "OK", error: str = None):
        """End a span"""
        span.end_time = time.time()
        span.status = status
        span.error_message = error
        
        # Restore parent span
        trace = current_trace.get()
        if trace:
            parent_id = span.parent_id
            if parent_id:
                parent = next((s for s in trace.spans if s.span_id == parent_id), None)
                current_span.set(parent)
            else:
                current_span.set(None)
    
    def end_trace(self) -> TraceContext:
        """End current trace and store it"""
        trace = current_trace.get()
        if trace:
            self.storage.store(trace)
            current_trace.set(None)
            current_span.set(None)
        return trace
    
    def add_event(self, name: str, attributes: Dict = None):
        """Add event to current span"""
        span = current_span.get()
        if span:
            span.events.append({
                "name": name,
                "timestamp": time.time(),
                "attributes": attributes or {}
            })
    
    def set_attribute(self, key: str, value: Any):
        """Set attribute on current span"""
        span = current_span.get()
        if span:
            span.attributes[key] = value
```

### Context Manager for Spans

```python
from contextlib import contextmanager

class Tracer:
    """High-level tracing API"""
    
    def __init__(self, service_name: str, storage: 'TraceStorage' = None):
        self.service_name = service_name
        self.manager = TracingManager(storage)
    
    @contextmanager
    def trace(self, name: str = None, metadata: Dict = None):
        """Context manager for traces"""
        trace = self.manager.start_trace(metadata)
        root_span = self.manager.start_span(name or f"{self.service_name}.root")
        
        try:
            yield trace
            self.manager.end_span(root_span, status="OK")
        except Exception as e:
            self.manager.end_span(root_span, status="ERROR", error=str(e))
            raise
        finally:
            self.manager.end_trace()
    
    @contextmanager
    def span(self, name: str, attributes: Dict = None):
        """Context manager for spans"""
        span = self.manager.start_span(name, attributes)
        
        try:
            yield span
            self.manager.end_span(span, status="OK")
        except Exception as e:
            self.manager.end_span(span, status="ERROR", error=str(e))
            raise
    
    def event(self, name: str, attributes: Dict = None):
        """Add event to current span"""
        self.manager.add_event(name, attributes)
    
    def attribute(self, key: str, value: Any):
        """Set attribute on current span"""
        self.manager.set_attribute(key, value)

# Usage
tracer = Tracer("chatbot")

def process_message(user_message: str) -> str:
    with tracer.trace("process_message", {"user_id": "123"}):
        
        with tracer.span("preprocess") as span:
            tracer.attribute("input_length", len(user_message))
            processed = preprocess(user_message)
        
        with tracer.span("retrieve_context"):
            tracer.event("starting_retrieval")
            context = retrieve(processed)
            tracer.attribute("num_documents", len(context))
            tracer.event("retrieval_complete")
        
        with tracer.span("generate", {"model": "gpt-4o"}):
            response = generate(processed, context)
            tracer.attribute("output_length", len(response))
        
        return response
```

---

## Storage Backends

### In-Memory Storage

```python
from collections import defaultdict
from typing import List, Optional
import threading

class InMemoryTraceStorage:
    """Simple in-memory trace storage"""
    
    def __init__(self, max_traces: int = 10000):
        self.traces: List[TraceContext] = []
        self.max_traces = max_traces
        self._lock = threading.Lock()
    
    def store(self, trace: TraceContext):
        with self._lock:
            self.traces.append(trace)
            # Evict old traces
            if len(self.traces) > self.max_traces:
                self.traces = self.traces[-self.max_traces:]
    
    def get(self, trace_id: str) -> Optional[TraceContext]:
        for trace in self.traces:
            if trace.trace_id == trace_id:
                return trace
        return None
    
    def query(
        self,
        start_time: float = None,
        end_time: float = None,
        status: str = None,
        limit: int = 100
    ) -> List[TraceContext]:
        results = []
        
        for trace in reversed(self.traces):
            if limit and len(results) >= limit:
                break
            
            # Time filter
            if start_time and trace.spans:
                if trace.spans[0].start_time < start_time:
                    continue
            
            if end_time and trace.spans:
                if trace.spans[0].start_time > end_time:
                    continue
            
            # Status filter
            if status:
                root_status = trace.spans[0].status if trace.spans else None
                if root_status != status:
                    continue
            
            results.append(trace)
        
        return results
```

### File-Based Storage

```python
import json
import os
from datetime import datetime
from pathlib import Path

class FileTraceStorage:
    """File-based trace storage with rotation"""
    
    def __init__(
        self,
        base_path: str = "./traces",
        max_file_size_mb: int = 100,
        max_files: int = 10
    ):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.max_file_size = max_file_size_mb * 1024 * 1024
        self.max_files = max_files
        self.current_file = None
        self._lock = threading.Lock()
    
    def _get_current_file(self) -> Path:
        """Get current log file, rotate if needed"""
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        file_path = self.base_path / f"traces-{date_str}.jsonl"
        
        # Check if rotation needed
        if file_path.exists() and file_path.stat().st_size > self.max_file_size:
            # Rotate
            timestamp = datetime.utcnow().strftime("%H%M%S")
            rotated = self.base_path / f"traces-{date_str}-{timestamp}.jsonl"
            file_path.rename(rotated)
            self._cleanup_old_files()
        
        return file_path
    
    def _cleanup_old_files(self):
        """Remove old trace files"""
        files = sorted(self.base_path.glob("traces-*.jsonl"))
        while len(files) > self.max_files:
            files[0].unlink()
            files = files[1:]
    
    def store(self, trace: TraceContext):
        """Store trace to file"""
        with self._lock:
            file_path = self._get_current_file()
            
            trace_dict = {
                "trace_id": trace.trace_id,
                "metadata": trace.metadata,
                "spans": [
                    {
                        "span_id": s.span_id,
                        "parent_id": s.parent_id,
                        "name": s.name,
                        "start_time": s.start_time,
                        "end_time": s.end_time,
                        "duration_ms": s.duration_ms,
                        "attributes": s.attributes,
                        "events": s.events,
                        "status": s.status,
                        "error": s.error_message
                    }
                    for s in trace.spans
                ],
                "duration_ms": trace.duration_ms
            }
            
            with open(file_path, "a") as f:
                f.write(json.dumps(trace_dict) + "\n")
```

### Database Storage

```python
import sqlite3
from typing import List, Optional
import json

class SQLiteTraceStorage:
    """SQLite-based trace storage"""
    
    def __init__(self, db_path: str = "traces.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    created_at REAL,
                    duration_ms REAL,
                    status TEXT,
                    metadata TEXT
                );
                
                CREATE TABLE IF NOT EXISTS spans (
                    span_id TEXT PRIMARY KEY,
                    trace_id TEXT,
                    parent_id TEXT,
                    name TEXT,
                    start_time REAL,
                    end_time REAL,
                    duration_ms REAL,
                    status TEXT,
                    error TEXT,
                    attributes TEXT,
                    events TEXT,
                    FOREIGN KEY (trace_id) REFERENCES traces(trace_id)
                );
                
                CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id);
                CREATE INDEX IF NOT EXISTS idx_traces_created ON traces(created_at);
            """)
    
    def store(self, trace: TraceContext):
        """Store trace to SQLite"""
        with sqlite3.connect(self.db_path) as conn:
            # Get root status
            root_status = trace.spans[0].status if trace.spans else "UNKNOWN"
            
            conn.execute("""
                INSERT OR REPLACE INTO traces (trace_id, created_at, duration_ms, status, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                trace.trace_id,
                time.time(),
                trace.duration_ms,
                root_status,
                json.dumps(trace.metadata)
            ))
            
            for span in trace.spans:
                conn.execute("""
                    INSERT OR REPLACE INTO spans 
                    (span_id, trace_id, parent_id, name, start_time, end_time, 
                     duration_ms, status, error, attributes, events)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    span.span_id,
                    trace.trace_id,
                    span.parent_id,
                    span.name,
                    span.start_time,
                    span.end_time,
                    span.duration_ms,
                    span.status,
                    span.error_message,
                    json.dumps(span.attributes),
                    json.dumps(span.events)
                ))
    
    def query(
        self,
        status: str = None,
        min_duration_ms: float = None,
        limit: int = 100
    ) -> List[dict]:
        """Query traces"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            query = "SELECT * FROM traces WHERE 1=1"
            params = []
            
            if status:
                query += " AND status = ?"
                params.append(status)
            
            if min_duration_ms:
                query += " AND duration_ms >= ?"
                params.append(min_duration_ms)
            
            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
```

---

## LLM-Specific Logging

### Comprehensive LLM Call Logger

```python
from openai import OpenAI
from functools import wraps
import time

class LLMCallLogger:
    """Log all LLM API calls with comprehensive metrics"""
    
    def __init__(self, tracer: Tracer, logger: LLMLogger):
        self.tracer = tracer
        self.logger = logger
    
    def wrap_openai_client(self, client: OpenAI) -> OpenAI:
        """Wrap OpenAI client to add logging"""
        original_create = client.chat.completions.create
        
        @wraps(original_create)
        def logged_create(*args, **kwargs):
            return self._logged_completion(original_create, *args, **kwargs)
        
        client.chat.completions.create = logged_create
        return client
    
    def _logged_completion(self, func, *args, **kwargs):
        """Log a completion call"""
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        
        with self.tracer.span("llm_completion", {"model": model}) as span:
            start_time = time.time()
            
            try:
                # Make the actual call
                response = func(*args, **kwargs)
                
                # Extract metrics
                usage = response.usage
                latency_ms = (time.time() - start_time) * 1000
                
                # Set span attributes
                self.tracer.attribute("prompt_tokens", usage.prompt_tokens)
                self.tracer.attribute("completion_tokens", usage.completion_tokens)
                self.tracer.attribute("total_tokens", usage.total_tokens)
                self.tracer.attribute("latency_ms", latency_ms)
                self.tracer.attribute("finish_reason", response.choices[0].finish_reason)
                
                # Calculate cost (approximate)
                cost = self._estimate_cost(
                    model,
                    usage.prompt_tokens,
                    usage.completion_tokens
                )
                self.tracer.attribute("estimated_cost_usd", cost)
                
                # Log structured entry
                self.logger.log_llm_call(
                    operation="chat_completion",
                    model=model,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    latency_ms=latency_ms,
                    metadata={
                        "finish_reason": response.choices[0].finish_reason,
                        "cost_usd": cost,
                        "num_messages": len(messages)
                    }
                )
                
                return response
                
            except Exception as e:
                self.logger.log_error(
                    operation="chat_completion",
                    error=str(e),
                    metadata={"model": model}
                )
                raise
    
    def _estimate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> float:
        """Estimate cost based on model pricing"""
        # Approximate pricing (update as needed)
        pricing = {
            "gpt-4o": (0.005, 0.015),  # (input, output) per 1K tokens
            "gpt-4o-mini": (0.00015, 0.0006),
            "gpt-4-turbo": (0.01, 0.03),
            "gpt-3.5-turbo": (0.0005, 0.0015),
        }
        
        input_price, output_price = pricing.get(model, (0.01, 0.03))
        
        cost = (prompt_tokens / 1000 * input_price) + (completion_tokens / 1000 * output_price)
        return round(cost, 6)

# Usage
tracer = Tracer("my-app", storage=SQLiteTraceStorage())
logger = LLMLogger("my-app")
llm_logger = LLMCallLogger(tracer, logger)

# Wrap the client
client = OpenAI()
client = llm_logger.wrap_openai_client(client)

# All calls are now logged
with tracer.trace("user_query"):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello!"}]
    )
```

---

## Metrics and Analytics

### Metrics Collector

```python
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List
import statistics

@dataclass
class MetricsBucket:
    """Bucket for collecting metrics over time"""
    count: int = 0
    sum_value: float = 0
    min_value: float = float('inf')
    max_value: float = float('-inf')
    values: List[float] = None
    
    def __post_init__(self):
        if self.values is None:
            self.values = []
    
    def add(self, value: float):
        self.count += 1
        self.sum_value += value
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)
        self.values.append(value)
    
    @property
    def mean(self) -> float:
        return self.sum_value / self.count if self.count else 0
    
    @property
    def p50(self) -> float:
        return statistics.median(self.values) if self.values else 0
    
    @property
    def p95(self) -> float:
        if not self.values:
            return 0
        sorted_values = sorted(self.values)
        idx = int(len(sorted_values) * 0.95)
        return sorted_values[idx]
    
    @property
    def p99(self) -> float:
        if not self.values:
            return 0
        sorted_values = sorted(self.values)
        idx = int(len(sorted_values) * 0.99)
        return sorted_values[idx]

class MetricsCollector:
    """Collect and aggregate metrics"""
    
    def __init__(self):
        self.metrics: Dict[str, MetricsBucket] = defaultdict(MetricsBucket)
        self.counters: Dict[str, int] = defaultdict(int)
    
    def record(self, name: str, value: float, tags: Dict = None):
        """Record a metric value"""
        key = self._make_key(name, tags)
        self.metrics[key].add(value)
    
    def increment(self, name: str, value: int = 1, tags: Dict = None):
        """Increment a counter"""
        key = self._make_key(name, tags)
        self.counters[key] += value
    
    def _make_key(self, name: str, tags: Dict = None) -> str:
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}|{tag_str}"
    
    def get_summary(self) -> Dict:
        """Get summary of all metrics"""
        summary = {
            "metrics": {},
            "counters": dict(self.counters)
        }
        
        for key, bucket in self.metrics.items():
            summary["metrics"][key] = {
                "count": bucket.count,
                "mean": bucket.mean,
                "min": bucket.min_value,
                "max": bucket.max_value,
                "p50": bucket.p50,
                "p95": bucket.p95,
                "p99": bucket.p99
            }
        
        return summary
    
    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.counters.clear()

# Usage
metrics = MetricsCollector()

# Record latency
metrics.record("llm.latency_ms", 150, {"model": "gpt-4o"})
metrics.record("llm.latency_ms", 200, {"model": "gpt-4o"})
metrics.record("llm.latency_ms", 50, {"model": "gpt-4o-mini"})

# Increment counters
metrics.increment("llm.requests", tags={"model": "gpt-4o"})
metrics.increment("llm.errors", tags={"error_type": "rate_limit"})

# Get summary
print(json.dumps(metrics.get_summary(), indent=2))
```

---

## Best Practices

### 1. Use Correlation IDs

```python
# Pass trace_id through your entire request lifecycle
def handle_request(request):
    trace_id = request.headers.get("X-Trace-ID") or str(uuid.uuid4())
    
    with tracer.trace("request", {"trace_id": trace_id}):
        # Process request
        pass
```

### 2. Log Sensitive Data Carefully

```python
def sanitize_for_logging(data: dict) -> dict:
    """Remove sensitive data before logging"""
    sensitive_keys = {"password", "api_key", "token", "secret"}
    
    sanitized = {}
    for key, value in data.items():
        if key.lower() in sensitive_keys:
            sanitized[key] = "[REDACTED]"
        elif isinstance(value, dict):
            sanitized[key] = sanitize_for_logging(value)
        else:
            sanitized[key] = value
    
    return sanitized
```

### 3. Sample High-Volume Logs

```python
import random

SAMPLE_RATE = 0.1  # Log 10% of requests

def should_log_detailed() -> bool:
    return random.random() < SAMPLE_RATE
```

### 4. Use Log Levels Appropriately

```python
# INFO: Normal operations
logger.info("Processing request")

# WARNING: Potential issues
logger.warning("Retrying after rate limit")

# ERROR: Failures
logger.error("Failed to process request")

# DEBUG: Detailed debugging (production: off)
logger.debug(f"Full prompt: {prompt}")
```

---

## Summary

| Component | Purpose |
|-----------|---------|
| Structured Logging | Consistent, parseable log format |
| Tracing | Track request flow across components |
| Storage Backends | Persist traces for analysis |
| Metrics | Aggregate performance data |
| Cost Tracking | Monitor API expenses |

**Key Takeaways:**
1. Structure your logs for easy parsing
2. Use context managers for clean tracing
3. Choose storage based on your scale
4. Calculate and track costs
5. Sample in high-volume scenarios

---

## Next Steps

- [Lab: Observability](/learn/advanced-topics/observability/observability-lab) - Build complete system
- [Cost Optimization](/learn/advanced-topics/optimization/semantic-caching) - Use logs to optimize
