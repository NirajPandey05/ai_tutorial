# Phoenix and Arize: LLM Observability

Learn to use Phoenix (open-source) and Arize for comprehensive LLM monitoring, evaluation, and debugging.

---

## Phoenix Overview

[Phoenix](https://github.com/Arize-ai/phoenix) is an open-source observability library for LLM applications. It runs locally and provides:

- Trace visualization
- Embedding analysis
- Evaluation tools
- No data leaves your machine

---

## Getting Started with Phoenix

### Installation

```bash
pip install arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp
```

### Launch Phoenix

```python
import phoenix as px

# Launch Phoenix in the background
session = px.launch_app()

# Open the UI
print(f"Phoenix UI: {session.url}")  # Usually http://localhost:6006
```

### Auto-Instrumentation

Phoenix can automatically instrument OpenAI and other providers:

```python
import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor

# Launch Phoenix
px.launch_app()

# Set up tracing
tracer_provider = register(project_name="my-llm-app")

# Instrument OpenAI
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

# Now all OpenAI calls are traced
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}]
)

# View traces in Phoenix UI at http://localhost:6006
```

---

## Manual Tracing

### Using OpenTelemetry Spans

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from phoenix.otel import register

# Set up tracing
tracer_provider = register(project_name="my-app")
tracer = trace.get_tracer(__name__)

@tracer.start_as_current_span("process_query")
def process_query(query: str) -> dict:
    """Process a user query with tracing"""
    
    span = trace.get_current_span()
    span.set_attribute("query.text", query)
    span.set_attribute("query.length", len(query))
    
    # Sub-operation
    with tracer.start_as_current_span("retrieve_context") as retrieve_span:
        context = retrieve_documents(query)
        retrieve_span.set_attribute("num_documents", len(context))
    
    # Another sub-operation
    with tracer.start_as_current_span("generate_response") as gen_span:
        response = generate_answer(query, context)
        gen_span.set_attribute("response.length", len(response))
    
    return {"query": query, "response": response}
```

### Logging Inputs and Outputs

```python
from openinference.semconv.trace import (
    SpanAttributes,
    OpenInferenceSpanKindValues,
)

def log_llm_call(span, messages, response):
    """Log LLM inputs and outputs to span"""
    
    span.set_attribute(
        SpanAttributes.OPENINFERENCE_SPAN_KIND,
        OpenInferenceSpanKindValues.LLM.value
    )
    
    # Log input
    span.set_attribute(SpanAttributes.INPUT_VALUE, str(messages))
    
    # Log output
    span.set_attribute(SpanAttributes.OUTPUT_VALUE, response)
    
    # Log token usage
    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, prompt_tokens)
    span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, completion_tokens)
```

---

## Embedding Analysis

Phoenix excels at analyzing embeddings:

```python
import phoenix as px
import pandas as pd
import numpy as np

# Create sample data with embeddings
data = pd.DataFrame({
    "text": ["Document 1", "Document 2", "Document 3"],
    "category": ["tech", "science", "tech"],
    "embedding": [
        np.random.rand(384).tolist(),
        np.random.rand(384).tolist(),
        np.random.rand(384).tolist()
    ]
})

# Launch with embedding analysis
schema = px.Schema(
    feature_column_names=["text", "category"],
    embedding_feature_column_names={
        "embedding": px.EmbeddingColumnNames(
            vector_column_name="embedding"
        )
    }
)

session = px.launch_app(
    primary=px.Inferences(data, schema),
    run_in_thread=True
)

# Visualize embedding clusters at http://localhost:6006
```

### Analyzing Retrieval Quality

```python
import phoenix as px
from phoenix.evals import (
    HallucinationEvaluator,
    QAEvaluator,
    RelevanceEvaluator,
)

# Get traces from Phoenix
px_client = px.Client()
traces_df = px_client.get_spans_dataframe()

# Filter to retrieval spans
retrieval_spans = traces_df[
    traces_df["span_kind"] == "RETRIEVER"
]

# Evaluate relevance of retrieved documents
evaluator = RelevanceEvaluator()

for _, span in retrieval_spans.iterrows():
    query = span["attributes.input.value"]
    documents = span["attributes.output.value"]
    
    relevance_score = evaluator.evaluate(query, documents)
    print(f"Query: {query[:50]}...")
    print(f"Relevance: {relevance_score}")
```

---

## Arize Platform

Arize is the enterprise version with additional features:

### Setup

```python
from arize.pandas.logger import Client
from arize.utils.types import ModelTypes, Environments

# Initialize Arize client
arize_client = Client(
    space_key="YOUR_SPACE_KEY",
    api_key="YOUR_API_KEY"
)

# Log predictions
response = arize_client.log(
    dataframe=predictions_df,
    model_id="my-llm-app",
    model_version="1.0",
    model_type=ModelTypes.GENERATIVE_LLM,
    environment=Environments.PRODUCTION,
    schema=arize_schema
)
```

### LLM-Specific Logging

```python
from arize.utils.types import (
    EmbeddingColumnNames,
    LLMConfigColumnNames,
    PromptTemplateColumnNames,
)

# Define schema for LLM data
schema = Schema(
    prompt_column_names="prompt",
    response_column_names="response",
    llm_config_column_names=LLMConfigColumnNames(
        model_column_name="model_name",
        temperature_column_name="temperature"
    ),
    prompt_template_column_names=PromptTemplateColumnNames(
        template_column_name="prompt_template",
        template_version_column_name="template_version"
    ),
    embedding_feature_column_names={
        "prompt_embedding": EmbeddingColumnNames(
            vector_column_name="prompt_vector"
        )
    }
)
```

---

## Building Custom Observability

### Lightweight Tracing System

```python
import time
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
import uuid

@dataclass
class Span:
    """A single span in a trace"""
    span_id: str
    parent_id: Optional[str]
    name: str
    start_time: float
    end_time: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict] = field(default_factory=list)
    status: str = "OK"
    error: Optional[str] = None

@dataclass
class Trace:
    """Complete trace with multiple spans"""
    trace_id: str
    spans: List[Span] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class SimpleTracer:
    """Lightweight tracing for LLM applications"""
    
    def __init__(self, service_name: str = "llm-app"):
        self.service_name = service_name
        self.traces: Dict[str, Trace] = {}
        self._current_trace_id: Optional[str] = None
        self._current_span_id: Optional[str] = None
        self._span_stack: List[str] = []
    
    def start_trace(self, metadata: Dict = None) -> str:
        """Start a new trace"""
        trace_id = str(uuid.uuid4())
        self.traces[trace_id] = Trace(
            trace_id=trace_id,
            metadata=metadata or {}
        )
        self._current_trace_id = trace_id
        return trace_id
    
    @contextmanager
    def span(self, name: str, attributes: Dict = None):
        """Create a span context"""
        span_id = str(uuid.uuid4())
        parent_id = self._span_stack[-1] if self._span_stack else None
        
        span = Span(
            span_id=span_id,
            parent_id=parent_id,
            name=name,
            start_time=time.time(),
            attributes=attributes or {}
        )
        
        self._span_stack.append(span_id)
        
        try:
            yield span
            span.status = "OK"
        except Exception as e:
            span.status = "ERROR"
            span.error = str(e)
            raise
        finally:
            span.end_time = time.time()
            self._span_stack.pop()
            
            if self._current_trace_id:
                self.traces[self._current_trace_id].spans.append(span)
    
    def add_event(self, span: Span, name: str, attributes: Dict = None):
        """Add an event to a span"""
        span.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {}
        })
    
    def export_trace(self, trace_id: str) -> Dict:
        """Export trace as JSON"""
        trace = self.traces.get(trace_id)
        if not trace:
            return {}
        
        return {
            "trace_id": trace.trace_id,
            "metadata": trace.metadata,
            "spans": [asdict(s) for s in trace.spans],
            "duration_ms": self._calculate_duration(trace)
        }
    
    def _calculate_duration(self, trace: Trace) -> float:
        if not trace.spans:
            return 0
        
        start = min(s.start_time for s in trace.spans)
        end = max(s.end_time or s.start_time for s in trace.spans)
        return (end - start) * 1000

# Usage
tracer = SimpleTracer(service_name="my-chatbot")

def process_chat(message: str) -> str:
    trace_id = tracer.start_trace(metadata={"user_id": "123"})
    
    with tracer.span("process_chat", {"message_length": len(message)}) as root:
        
        with tracer.span("retrieve_context") as retrieve_span:
            context = retrieve_documents(message)
            retrieve_span.attributes["num_docs"] = len(context)
            tracer.add_event(retrieve_span, "retrieval_complete")
        
        with tracer.span("generate_response") as gen_span:
            response = generate(message, context)
            gen_span.attributes["response_length"] = len(response)
            gen_span.attributes["model"] = "gpt-4o"
    
    # Export trace
    trace_data = tracer.export_trace(trace_id)
    print(json.dumps(trace_data, indent=2))
    
    return response
```

---

## Evaluation with Phoenix

### Built-in Evaluators

```python
from phoenix.evals import (
    OpenAIModel,
    HallucinationEvaluator,
    QAEvaluator,
    RelevanceEvaluator,
    ToxicityEvaluator,
)

# Initialize the model for evaluation
eval_model = OpenAIModel(model="gpt-4o-mini")

# Hallucination detection
hallucination_eval = HallucinationEvaluator(eval_model)
result = hallucination_eval.evaluate(
    output="Paris is the capital of Germany",
    reference="Paris is the capital of France"
)
print(f"Hallucination detected: {result}")

# Q&A correctness
qa_eval = QAEvaluator(eval_model)
result = qa_eval.evaluate(
    question="What is the capital of France?",
    answer="Paris",
    reference="Paris is the capital of France"
)
print(f"Answer correct: {result}")

# Relevance
relevance_eval = RelevanceEvaluator(eval_model)
result = relevance_eval.evaluate(
    query="Python programming",
    document="Python is a programming language created by Guido van Rossum"
)
print(f"Relevance score: {result}")

# Toxicity
toxicity_eval = ToxicityEvaluator(eval_model)
result = toxicity_eval.evaluate(
    text="This is a helpful and friendly response"
)
print(f"Toxicity score: {result}")
```

### Custom Evaluators

```python
from phoenix.evals import LLMEvaluator

class CustomEvaluator(LLMEvaluator):
    """Custom evaluator for specific criteria"""
    
    template = """
    Evaluate the following response for helpfulness.
    
    Question: {question}
    Response: {response}
    
    Score from 0-10 where:
    - 0-3: Not helpful, missing information
    - 4-6: Somewhat helpful, partially answers
    - 7-10: Very helpful, complete answer
    
    Provide your score and a brief explanation.
    
    Score:
    """
    
    def parse_output(self, output: str) -> float:
        """Extract numeric score from LLM output"""
        import re
        match = re.search(r'\b(\d+)\b', output)
        if match:
            return float(match.group(1)) / 10
        return 0.5

# Use custom evaluator
custom_eval = CustomEvaluator(eval_model)
score = custom_eval.evaluate(
    question="How do I install Python?",
    response="Download from python.org and run the installer."
)
```

---

## Dashboards and Alerts

### Building a Simple Dashboard

```python
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

def create_observability_dashboard(tracer: SimpleTracer):
    """Simple Streamlit dashboard for LLM observability"""
    
    st.title("LLM Observability Dashboard")
    
    # Trace summary
    st.header("Trace Summary")
    
    traces_data = []
    for trace_id, trace in tracer.traces.items():
        traces_data.append({
            "trace_id": trace_id[:8],
            "spans": len(trace.spans),
            "duration_ms": tracer._calculate_duration(trace),
            "errors": sum(1 for s in trace.spans if s.status == "ERROR")
        })
    
    df = pd.DataFrame(traces_data)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Traces", len(df))
    col2.metric("Avg Duration", f"{df['duration_ms'].mean():.0f}ms")
    col3.metric("Error Rate", f"{(df['errors'] > 0).mean():.1%}")
    
    # Latency chart
    st.subheader("Latency Distribution")
    st.bar_chart(df['duration_ms'])
    
    # Error traces
    st.subheader("Error Traces")
    error_traces = df[df['errors'] > 0]
    st.dataframe(error_traces)

# Run with: streamlit run dashboard.py
```

### Setting Up Alerts

```python
import smtplib
from typing import Callable, List

class AlertManager:
    """Manage alerts based on observability metrics"""
    
    def __init__(self):
        self.rules: List[dict] = []
        self.handlers: List[Callable] = []
    
    def add_rule(
        self,
        name: str,
        condition: Callable,
        severity: str = "warning"
    ):
        """Add an alerting rule"""
        self.rules.append({
            "name": name,
            "condition": condition,
            "severity": severity
        })
    
    def add_handler(self, handler: Callable):
        """Add alert handler (email, Slack, etc.)"""
        self.handlers.append(handler)
    
    def check_metrics(self, metrics: dict):
        """Check metrics against rules"""
        for rule in self.rules:
            if rule["condition"](metrics):
                alert = {
                    "rule": rule["name"],
                    "severity": rule["severity"],
                    "metrics": metrics,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                for handler in self.handlers:
                    handler(alert)

# Set up alerts
alerts = AlertManager()

# Rule: High latency
alerts.add_rule(
    name="high_latency",
    condition=lambda m: m.get("avg_latency_ms", 0) > 5000,
    severity="warning"
)

# Rule: High error rate
alerts.add_rule(
    name="high_error_rate",
    condition=lambda m: m.get("error_rate", 0) > 0.1,
    severity="critical"
)

# Handler: Print (replace with email/Slack)
def print_alert(alert):
    print(f"🚨 ALERT [{alert['severity']}]: {alert['rule']}")
    print(f"   Metrics: {alert['metrics']}")

alerts.add_handler(print_alert)

# Check periodically
metrics = {
    "avg_latency_ms": 6000,
    "error_rate": 0.05,
    "total_requests": 1000
}
alerts.check_metrics(metrics)
```

---

## Best Practices

### 1. Instrument Early

```python
# Add tracing from the start of your project
import phoenix as px
px.launch_app()  # Start Phoenix early

# Then build your application with tracing built-in
```

### 2. Use Semantic Conventions

```python
# Follow OpenTelemetry conventions
span.set_attribute("llm.model", "gpt-4o")
span.set_attribute("llm.prompt_tokens", 100)
span.set_attribute("llm.completion_tokens", 50)
span.set_attribute("llm.temperature", 0.7)
```

### 3. Sample in Production

```python
import random

SAMPLE_RATE = 0.1  # 10% of requests

def should_trace() -> bool:
    return random.random() < SAMPLE_RATE

# Only trace sampled requests
if should_trace():
    with tracer.span("expensive_operation"):
        # ...
```

### 4. Include Business Context

```python
span.set_attribute("user.tier", "premium")
span.set_attribute("feature.name", "document_qa")
span.set_attribute("ab_test.variant", "v2")
```

---

## Summary

| Tool | Type | Best For |
|------|------|----------|
| Phoenix | Open Source | Local development, embeddings |
| Arize | Enterprise | Production monitoring, alerting |
| Custom | DIY | Specific requirements, privacy |

**Key Takeaways:**
1. Start with Phoenix for local development
2. Use auto-instrumentation when possible
3. Add meaningful attributes to spans
4. Analyze embeddings to understand retrieval
5. Set up alerts for production issues

---

## Next Steps

- [Custom Logging](/learn/advanced-topics/observability/custom-logging) - Build detailed logging
- [Lab: Observability](/learn/advanced-topics/observability/observability-lab) - Hands-on implementation
