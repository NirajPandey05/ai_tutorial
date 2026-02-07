# Lab: Building LLM Observability

Build a complete observability system for LLM applications, combining tracing, logging, metrics, and visualization.

---

## Learning Objectives

By the end of this lab, you will:
- Implement a full-featured tracing system
- Create structured logging for LLM calls
- Build a metrics dashboard
- Set up alerting for issues
- Analyze performance and costs

---

## Prerequisites

- Python 3.10+
- OpenAI API key
- Basic understanding of observability concepts

```bash
pip install openai streamlit pandas plotly sqlite-utils
```

---

## Part 1: Building the Core Observability System

### Step 1: Create the Data Models

```python
# observability/models.py
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import uuid
import json

class SpanStatus(str, Enum):
    UNSET = "UNSET"
    OK = "OK"
    ERROR = "ERROR"

class SpanKind(str, Enum):
    INTERNAL = "INTERNAL"
    LLM = "LLM"
    RETRIEVER = "RETRIEVER"
    TOOL = "TOOL"
    CHAIN = "CHAIN"

@dataclass
class Event:
    """Event within a span"""
    name: str
    timestamp: float
    attributes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Span:
    """A single span in a trace"""
    span_id: str
    trace_id: str
    parent_id: Optional[str]
    name: str
    kind: SpanKind
    start_time: float
    end_time: Optional[float] = None
    status: SpanStatus = SpanStatus.UNSET
    error_message: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Event] = field(default_factory=list)
    
    @property
    def duration_ms(self) -> float:
        if self.end_time is None:
            return 0
        return (self.end_time - self.start_time) * 1000
    
    def to_dict(self) -> dict:
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "kind": self.kind.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "error_message": self.error_message,
            "attributes": self.attributes,
            "events": [asdict(e) for e in self.events]
        }

@dataclass
class Trace:
    """Complete trace with multiple spans"""
    trace_id: str
    spans: List[Span] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def root_span(self) -> Optional[Span]:
        for span in self.spans:
            if span.parent_id is None:
                return span
        return self.spans[0] if self.spans else None
    
    @property
    def duration_ms(self) -> float:
        root = self.root_span
        return root.duration_ms if root else 0
    
    @property
    def status(self) -> SpanStatus:
        root = self.root_span
        return root.status if root else SpanStatus.UNSET
    
    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "metadata": self.metadata,
            "spans": [s.to_dict() for s in self.spans]
        }

@dataclass
class LLMMetrics:
    """Metrics for LLM calls"""
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    cost_usd: float
    timestamp: float = field(default_factory=lambda: datetime.utcnow().timestamp())
```

### Step 2: Create the Storage Layer

```python
# observability/storage.py
import sqlite3
import json
from typing import List, Optional
from contextlib import contextmanager
from .models import Trace, Span, SpanStatus, SpanKind, LLMMetrics

class ObservabilityStorage:
    """SQLite storage for observability data"""
    
    def __init__(self, db_path: str = "observability.db"):
        self.db_path = db_path
        self._init_db()
    
    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def _init_db(self):
        with self._get_connection() as conn:
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
                    kind TEXT,
                    start_time REAL,
                    end_time REAL,
                    duration_ms REAL,
                    status TEXT,
                    error_message TEXT,
                    attributes TEXT,
                    events TEXT,
                    FOREIGN KEY (trace_id) REFERENCES traces(trace_id)
                );
                
                CREATE TABLE IF NOT EXISTS llm_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT,
                    span_id TEXT,
                    model TEXT,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    latency_ms REAL,
                    cost_usd REAL,
                    timestamp REAL
                );
                
                CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id);
                CREATE INDEX IF NOT EXISTS idx_traces_created ON traces(created_at);
                CREATE INDEX IF NOT EXISTS idx_metrics_model ON llm_metrics(model);
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON llm_metrics(timestamp);
            """)
    
    def save_trace(self, trace: Trace):
        """Save a complete trace"""
        with self._get_connection() as conn:
            # Save trace
            conn.execute("""
                INSERT OR REPLACE INTO traces (trace_id, created_at, duration_ms, status, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                trace.trace_id,
                trace.root_span.start_time if trace.root_span else 0,
                trace.duration_ms,
                trace.status.value,
                json.dumps(trace.metadata)
            ))
            
            # Save spans
            for span in trace.spans:
                conn.execute("""
                    INSERT OR REPLACE INTO spans
                    (span_id, trace_id, parent_id, name, kind, start_time, end_time,
                     duration_ms, status, error_message, attributes, events)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    span.span_id,
                    span.trace_id,
                    span.parent_id,
                    span.name,
                    span.kind.value,
                    span.start_time,
                    span.end_time,
                    span.duration_ms,
                    span.status.value,
                    span.error_message,
                    json.dumps(span.attributes),
                    json.dumps([asdict(e) for e in span.events])
                ))
    
    def save_metrics(self, metrics: LLMMetrics, trace_id: str = None, span_id: str = None):
        """Save LLM metrics"""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO llm_metrics
                (trace_id, span_id, model, prompt_tokens, completion_tokens,
                 total_tokens, latency_ms, cost_usd, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trace_id,
                span_id,
                metrics.model,
                metrics.prompt_tokens,
                metrics.completion_tokens,
                metrics.total_tokens,
                metrics.latency_ms,
                metrics.cost_usd,
                metrics.timestamp
            ))
    
    def get_traces(
        self,
        status: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[dict]:
        """Get traces with optional filtering"""
        with self._get_connection() as conn:
            query = "SELECT * FROM traces WHERE 1=1"
            params = []
            
            if status:
                query += " AND status = ?"
                params.append(status)
            
            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_trace_with_spans(self, trace_id: str) -> Optional[dict]:
        """Get a trace with all its spans"""
        with self._get_connection() as conn:
            trace = conn.execute(
                "SELECT * FROM traces WHERE trace_id = ?",
                (trace_id,)
            ).fetchone()
            
            if not trace:
                return None
            
            spans = conn.execute(
                "SELECT * FROM spans WHERE trace_id = ? ORDER BY start_time",
                (trace_id,)
            ).fetchall()
            
            return {
                **dict(trace),
                "spans": [dict(s) for s in spans]
            }
    
    def get_metrics_summary(self, hours: int = 24) -> dict:
        """Get metrics summary for the last N hours"""
        import time
        cutoff = time.time() - (hours * 3600)
        
        with self._get_connection() as conn:
            # Overall stats
            overall = conn.execute("""
                SELECT 
                    COUNT(*) as total_calls,
                    SUM(prompt_tokens) as total_prompt_tokens,
                    SUM(completion_tokens) as total_completion_tokens,
                    SUM(total_tokens) as total_tokens,
                    AVG(latency_ms) as avg_latency,
                    SUM(cost_usd) as total_cost
                FROM llm_metrics
                WHERE timestamp >= ?
            """, (cutoff,)).fetchone()
            
            # By model
            by_model = conn.execute("""
                SELECT 
                    model,
                    COUNT(*) as calls,
                    SUM(total_tokens) as tokens,
                    AVG(latency_ms) as avg_latency,
                    SUM(cost_usd) as cost
                FROM llm_metrics
                WHERE timestamp >= ?
                GROUP BY model
            """, (cutoff,)).fetchall()
            
            # Error rate
            traces = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'ERROR' THEN 1 ELSE 0 END) as errors
                FROM traces
                WHERE created_at >= ?
            """, (cutoff,)).fetchone()
            
            return {
                "period_hours": hours,
                "total_calls": overall["total_calls"] or 0,
                "total_tokens": overall["total_tokens"] or 0,
                "avg_latency_ms": overall["avg_latency"] or 0,
                "total_cost_usd": overall["total_cost"] or 0,
                "error_rate": (traces["errors"] / traces["total"]) if traces["total"] else 0,
                "by_model": [dict(m) for m in by_model]
            }
```

### Step 3: Build the Tracer

```python
# observability/tracer.py
import time
import uuid
from contextvars import ContextVar
from contextlib import contextmanager
from typing import Optional, Dict, Any
from .models import Trace, Span, SpanKind, SpanStatus, Event, LLMMetrics
from .storage import ObservabilityStorage

# Context variables
_current_trace: ContextVar[Optional[Trace]] = ContextVar('current_trace', default=None)
_current_span: ContextVar[Optional[Span]] = ContextVar('current_span', default=None)

class LLMTracer:
    """Tracer for LLM applications"""
    
    # Pricing per 1K tokens (input, output)
    MODEL_PRICING = {
        "gpt-4o": (0.005, 0.015),
        "gpt-4o-mini": (0.00015, 0.0006),
        "gpt-4-turbo": (0.01, 0.03),
        "gpt-3.5-turbo": (0.0005, 0.0015),
        "claude-3-opus": (0.015, 0.075),
        "claude-3-sonnet": (0.003, 0.015),
        "claude-3-haiku": (0.00025, 0.00125),
    }
    
    def __init__(self, service_name: str, storage: ObservabilityStorage = None):
        self.service_name = service_name
        self.storage = storage or ObservabilityStorage()
    
    @contextmanager
    def trace(self, name: str = None, metadata: Dict = None):
        """Start a new trace"""
        trace_id = str(uuid.uuid4())
        trace = Trace(
            trace_id=trace_id,
            metadata=metadata or {}
        )
        
        _current_trace.set(trace)
        
        # Create root span
        with self.span(name or f"{self.service_name}.trace", SpanKind.CHAIN):
            try:
                yield trace
            finally:
                _current_trace.set(None)
                self.storage.save_trace(trace)
    
    @contextmanager
    def span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Dict = None
    ):
        """Create a new span"""
        trace = _current_trace.get()
        parent = _current_span.get()
        
        span = Span(
            span_id=str(uuid.uuid4()),
            trace_id=trace.trace_id if trace else str(uuid.uuid4()),
            parent_id=parent.span_id if parent else None,
            name=name,
            kind=kind,
            start_time=time.time(),
            attributes=attributes or {}
        )
        
        if trace:
            trace.spans.append(span)
        
        _current_span.set(span)
        
        try:
            yield span
            span.status = SpanStatus.OK
        except Exception as e:
            span.status = SpanStatus.ERROR
            span.error_message = str(e)
            raise
        finally:
            span.end_time = time.time()
            
            # Restore parent span
            _current_span.set(parent)
    
    def add_event(self, name: str, attributes: Dict = None):
        """Add event to current span"""
        span = _current_span.get()
        if span:
            span.events.append(Event(
                name=name,
                timestamp=time.time(),
                attributes=attributes or {}
            ))
    
    def set_attribute(self, key: str, value: Any):
        """Set attribute on current span"""
        span = _current_span.get()
        if span:
            span.attributes[key] = value
    
    def log_llm_metrics(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float
    ):
        """Log LLM metrics and calculate cost"""
        trace = _current_trace.get()
        span = _current_span.get()
        
        # Calculate cost
        input_price, output_price = self.MODEL_PRICING.get(model, (0.01, 0.03))
        cost = (prompt_tokens / 1000 * input_price) + (completion_tokens / 1000 * output_price)
        
        metrics = LLMMetrics(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
            cost_usd=round(cost, 6)
        )
        
        # Set span attributes
        if span:
            span.attributes.update({
                "llm.model": model,
                "llm.prompt_tokens": prompt_tokens,
                "llm.completion_tokens": completion_tokens,
                "llm.total_tokens": prompt_tokens + completion_tokens,
                "llm.latency_ms": latency_ms,
                "llm.cost_usd": cost
            })
        
        # Save metrics
        self.storage.save_metrics(
            metrics,
            trace_id=trace.trace_id if trace else None,
            span_id=span.span_id if span else None
        )
        
        return metrics
    
    @property
    def current_trace_id(self) -> Optional[str]:
        trace = _current_trace.get()
        return trace.trace_id if trace else None
```

---

## Part 2: Instrumenting OpenAI

### Step 4: Create OpenAI Wrapper

```python
# observability/instrumentation.py
from openai import OpenAI
from functools import wraps
import time
from typing import Any
from .tracer import LLMTracer
from .models import SpanKind

class InstrumentedOpenAI:
    """OpenAI client with built-in observability"""
    
    def __init__(self, tracer: LLMTracer, api_key: str = None):
        self.tracer = tracer
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self._wrap_methods()
    
    def _wrap_methods(self):
        """Wrap OpenAI methods with tracing"""
        original_create = self.client.chat.completions.create
        
        @wraps(original_create)
        def traced_create(*args, **kwargs):
            return self._traced_completion(original_create, *args, **kwargs)
        
        self.client.chat.completions.create = traced_create
    
    def _traced_completion(self, func, *args, **kwargs):
        """Trace a completion call"""
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        
        with self.tracer.span(f"llm.{model}", SpanKind.LLM) as span:
            # Log input info
            self.tracer.set_attribute("llm.num_messages", len(messages))
            self.tracer.set_attribute("llm.model", model)
            self.tracer.add_event("llm_request_start")
            
            start_time = time.time()
            
            try:
                response = func(*args, **kwargs)
                
                latency_ms = (time.time() - start_time) * 1000
                
                # Log metrics
                usage = response.usage
                self.tracer.log_llm_metrics(
                    model=model,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    latency_ms=latency_ms
                )
                
                self.tracer.set_attribute("llm.finish_reason", response.choices[0].finish_reason)
                self.tracer.add_event("llm_request_complete", {
                    "tokens": usage.total_tokens
                })
                
                return response
                
            except Exception as e:
                self.tracer.add_event("llm_request_error", {
                    "error": str(e)
                })
                raise
    
    @property
    def chat(self):
        return self.client.chat

# Usage example
def create_instrumented_client(tracer: LLMTracer) -> InstrumentedOpenAI:
    return InstrumentedOpenAI(tracer)
```

---

## Part 3: Building the Dashboard

### Step 5: Create Streamlit Dashboard

```python
# dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

from observability.storage import ObservabilityStorage
from observability.tracer import LLMTracer

# Initialize storage
storage = ObservabilityStorage()

st.set_page_config(
    page_title="LLM Observability Dashboard",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 LLM Observability Dashboard")

# Sidebar filters
st.sidebar.header("Filters")
time_range = st.sidebar.selectbox(
    "Time Range",
    ["Last Hour", "Last 24 Hours", "Last 7 Days", "Last 30 Days"],
    index=1
)

time_mapping = {
    "Last Hour": 1,
    "Last 24 Hours": 24,
    "Last 7 Days": 168,
    "Last 30 Days": 720
}
hours = time_mapping[time_range]

# Get metrics summary
summary = storage.get_metrics_summary(hours=hours)

# Key Metrics Row
st.header("📊 Key Metrics")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Total Requests",
        f"{summary['total_calls']:,}",
        help="Total LLM API calls"
    )

with col2:
    st.metric(
        "Total Tokens",
        f"{summary['total_tokens']:,}",
        help="Total tokens processed"
    )

with col3:
    st.metric(
        "Avg Latency",
        f"{summary['avg_latency_ms']:.0f}ms",
        help="Average response time"
    )

with col4:
    st.metric(
        "Total Cost",
        f"${summary['total_cost_usd']:.2f}",
        help="Estimated API cost"
    )

with col5:
    error_pct = summary['error_rate'] * 100
    st.metric(
        "Error Rate",
        f"{error_pct:.1f}%",
        help="Percentage of failed requests"
    )

# Charts Row
st.header("📈 Analytics")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Usage by Model")
    if summary['by_model']:
        df_model = pd.DataFrame(summary['by_model'])
        fig = px.pie(
            df_model,
            values='calls',
            names='model',
            title='Requests by Model'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available")

with col2:
    st.subheader("Cost by Model")
    if summary['by_model']:
        df_model = pd.DataFrame(summary['by_model'])
        fig = px.bar(
            df_model,
            x='model',
            y='cost',
            title='Cost by Model (USD)'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available")

# Recent Traces
st.header("🔗 Recent Traces")

status_filter = st.selectbox(
    "Filter by Status",
    ["All", "OK", "ERROR"],
    index=0
)

traces = storage.get_traces(
    status=status_filter if status_filter != "All" else None,
    limit=20
)

if traces:
    # Create DataFrame
    df_traces = pd.DataFrame(traces)
    df_traces['created_at'] = pd.to_datetime(df_traces['created_at'], unit='s')
    df_traces['duration'] = df_traces['duration_ms'].apply(lambda x: f"{x:.0f}ms")
    
    # Color code status
    def status_color(status):
        return "🟢" if status == "OK" else "🔴"
    
    df_traces['status_icon'] = df_traces['status'].apply(status_color)
    
    # Display table
    st.dataframe(
        df_traces[['status_icon', 'trace_id', 'created_at', 'duration', 'status']],
        hide_index=True,
        column_config={
            "status_icon": st.column_config.TextColumn(""),
            "trace_id": st.column_config.TextColumn("Trace ID"),
            "created_at": st.column_config.DatetimeColumn("Time"),
            "duration": st.column_config.TextColumn("Duration"),
            "status": st.column_config.TextColumn("Status")
        }
    )
    
    # Trace details
    selected_trace = st.selectbox(
        "Select trace for details",
        df_traces['trace_id'].tolist(),
        index=0
    )
    
    if selected_trace:
        trace_detail = storage.get_trace_with_spans(selected_trace)
        if trace_detail:
            st.subheader(f"Trace: {selected_trace[:8]}...")
            
            # Show spans as timeline
            spans = trace_detail['spans']
            if spans:
                # Create Gantt-like chart
                span_data = []
                for span in spans:
                    span_data.append({
                        'Span': span['name'][:30],
                        'Start': span['start_time'],
                        'End': span['end_time'] or span['start_time'] + 0.001,
                        'Status': span['status'],
                        'Duration': f"{span['duration_ms']:.0f}ms"
                    })
                
                df_spans = pd.DataFrame(span_data)
                
                # Display span table
                st.dataframe(df_spans, hide_index=True)
                
                # Show attributes for each span
                with st.expander("Span Details"):
                    for span in spans:
                        st.markdown(f"**{span['name']}**")
                        if span['attributes']:
                            attrs = json.loads(span['attributes']) if isinstance(span['attributes'], str) else span['attributes']
                            st.json(attrs)
                        st.divider()

else:
    st.info("No traces found")

# Footer
st.divider()
st.caption("LLM Observability Dashboard | Built with Streamlit")
```

---

## Part 4: Complete Example Application

### Step 6: Build an Instrumented Chatbot

```python
# app.py
from openai import OpenAI
from observability.storage import ObservabilityStorage
from observability.tracer import LLMTracer
from observability.models import SpanKind
import time

# Initialize observability
storage = ObservabilityStorage()
tracer = LLMTracer("chatbot", storage)

# Initialize OpenAI
client = OpenAI()

def retrieve_context(query: str) -> list:
    """Simulate RAG retrieval"""
    with tracer.span("retrieve_context", SpanKind.RETRIEVER):
        time.sleep(0.1)  # Simulate retrieval
        tracer.set_attribute("query_length", len(query))
        tracer.add_event("retrieval_complete")
        
        # Return mock context
        return [
            "Our company was founded in 2015.",
            "We offer premium support for enterprise customers.",
            "Contact us at support@example.com"
        ]

def generate_response(query: str, context: list) -> str:
    """Generate response using LLM"""
    with tracer.span("generate_response", SpanKind.LLM):
        # Build prompt
        context_text = "\n".join(context)
        messages = [
            {"role": "system", "content": f"Use this context:\n{context_text}"},
            {"role": "user", "content": query}
        ]
        
        tracer.set_attribute("num_context_docs", len(context))
        tracer.add_event("llm_request_start")
        
        start_time = time.time()
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=500
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Log metrics
        tracer.log_llm_metrics(
            model="gpt-4o-mini",
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            latency_ms=latency_ms
        )
        
        return response.choices[0].message.content

def chat(user_message: str, user_id: str = "anonymous") -> str:
    """Process a chat message with full observability"""
    
    with tracer.trace("chat_request", metadata={"user_id": user_id}):
        tracer.set_attribute("user_message_length", len(user_message))
        tracer.add_event("request_received")
        
        # Step 1: Retrieve context
        context = retrieve_context(user_message)
        
        # Step 2: Generate response
        response = generate_response(user_message, context)
        
        tracer.set_attribute("response_length", len(response))
        tracer.add_event("request_complete")
        
        return response

# Example usage
if __name__ == "__main__":
    # Run some test queries
    queries = [
        "When was the company founded?",
        "How can I contact support?",
        "What enterprise features do you offer?"
    ]
    
    for query in queries:
        print(f"\nUser: {query}")
        response = chat(query, user_id="test_user")
        print(f"Bot: {response[:200]}...")
    
    # Print summary
    print("\n" + "="*50)
    print("Metrics Summary:")
    summary = storage.get_metrics_summary(hours=1)
    print(f"  Total Calls: {summary['total_calls']}")
    print(f"  Total Tokens: {summary['total_tokens']}")
    print(f"  Total Cost: ${summary['total_cost_usd']:.4f}")
    print(f"  Avg Latency: {summary['avg_latency_ms']:.0f}ms")
```

---

## Running the Lab

### Step 7: Run Everything Together

```bash
# Terminal 1: Run the chatbot
python app.py

# Terminal 2: Run the dashboard
streamlit run dashboard.py
```

The dashboard will be available at `http://localhost:8501`

---

## Challenge Exercises

### Challenge 1: Add Alerting
Implement email alerts when:
- Error rate exceeds 10%
- Average latency exceeds 5 seconds
- Hourly cost exceeds a threshold

### Challenge 2: Add Sampling
Implement trace sampling to reduce storage:
- Sample 10% of successful requests
- Keep all error requests

### Challenge 3: Export to External Systems
Add exporters for:
- Prometheus metrics
- Elasticsearch/Kibana
- Grafana

---

## Key Takeaways

1. **Structured Data**: Use consistent models for traces and metrics
2. **Context Propagation**: Use context variables for trace correlation
3. **Cost Tracking**: Calculate and track LLM API costs
4. **Visualization**: Build dashboards for quick insights
5. **Storage Flexibility**: Design for different storage backends

---

## Next Steps

- [Cost Optimization](/learn/advanced-topics/optimization/semantic-caching) - Use observability data to optimize
- [Production Deployment](/learn/deployment/production-checklist) - Deploy with full observability
