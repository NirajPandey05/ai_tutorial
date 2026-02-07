# LangSmith: LLM Observability and Tracing

Master LangSmith for comprehensive monitoring, debugging, and optimization of your LLM applications.

---

## What is LangSmith?

LangSmith is a platform for debugging, testing, evaluating, and monitoring LLM applications. It provides:

- **Tracing**: Track every LLM call and its context
- **Debugging**: Understand why your chain produced specific outputs
- **Evaluation**: Score and compare model outputs
- **Monitoring**: Track performance, costs, and errors in production
- **Datasets**: Create and manage test datasets

---

## Getting Started

### Installation and Setup

```bash
pip install langsmith langchain langchain-openai
```

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-api-key"
os.environ["LANGCHAIN_PROJECT"] = "my-llm-project"  # Optional: organize by project
```

### Basic Tracing

Once configured, all LangChain operations are automatically traced:

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o")

# This call is automatically traced
response = llm.invoke([HumanMessage(content="What is LangSmith?")])
print(response.content)

# View the trace in LangSmith dashboard
```

---

## Manual Tracing with `@traceable`

For non-LangChain code, use the `@traceable` decorator:

```python
from langsmith import traceable
from openai import OpenAI

client = OpenAI()

@traceable(name="sentiment_analysis")
def analyze_sentiment(text: str) -> dict:
    """Analyze sentiment with full tracing"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Analyze sentiment. Return: positive, negative, or neutral"},
            {"role": "user", "content": text}
        ]
    )
    
    sentiment = response.choices[0].message.content
    
    return {
        "text": text,
        "sentiment": sentiment,
        "model": "gpt-4o-mini"
    }

# Call the function - automatically traced
result = analyze_sentiment("I love this product!")
```

### Nested Tracing

```python
from langsmith import traceable

@traceable(name="extract_entities")
def extract_entities(text: str) -> list:
    """Extract named entities"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract named entities as JSON array"},
            {"role": "user", "content": text}
        ]
    )
    return json.loads(response.choices[0].message.content)

@traceable(name="classify_document")
def classify_document(text: str) -> str:
    """Classify document type"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Classify as: news, blog, academic, other"},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content

@traceable(name="process_document")
def process_document(text: str) -> dict:
    """Full document processing pipeline"""
    
    # Each sub-function creates a child span
    entities = extract_entities(text)
    doc_type = classify_document(text)
    sentiment = analyze_sentiment(text)
    
    return {
        "entities": entities,
        "type": doc_type,
        "sentiment": sentiment
    }

# The trace shows the full hierarchy
result = process_document("Apple CEO Tim Cook announced...")
```

---

## Adding Metadata and Tags

Enrich traces with custom metadata:

```python
from langsmith import traceable

@traceable(
    name="chat_completion",
    tags=["production", "customer-service"],
    metadata={"version": "2.0", "team": "support"}
)
def chat_completion(
    user_message: str,
    user_id: str,
    session_id: str
) -> str:
    # Add run-specific metadata
    from langsmith import get_current_run_tree
    
    run = get_current_run_tree()
    if run:
        run.metadata["user_id"] = user_id
        run.metadata["session_id"] = session_id
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": user_message}]
    )
    
    return response.choices[0].message.content
```

### Dynamic Tags

```python
@traceable(name="adaptive_chat")
def adaptive_chat(message: str, priority: str = "normal") -> str:
    from langsmith import get_current_run_tree
    
    run = get_current_run_tree()
    if run:
        # Add dynamic tags based on input
        run.tags = run.tags or []
        run.tags.append(f"priority:{priority}")
        
        if len(message) > 1000:
            run.tags.append("long-input")
        
        if "urgent" in message.lower():
            run.tags.append("urgent")
    
    # Process message...
    return "Response"
```

---

## Logging Inputs and Outputs

### Custom Input/Output Processing

```python
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

@traceable(name="rag_query")
def rag_query(query: str, context_docs: list) -> str:
    run = get_current_run_tree()
    
    # Log input details
    if run:
        run.inputs = {
            "query": query,
            "num_context_docs": len(context_docs),
            "context_preview": [doc[:100] for doc in context_docs[:3]]
        }
    
    # Build prompt
    context = "\n".join(context_docs)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"Context:\n{context}"},
            {"role": "user", "content": query}
        ]
    )
    
    answer = response.choices[0].message.content
    
    # Log output details
    if run:
        run.outputs = {
            "answer": answer,
            "answer_length": len(answer),
            "used_context": True
        }
    
    return answer
```

### Logging Errors

```python
from langsmith import traceable
import traceback

@traceable(name="safe_llm_call")
def safe_llm_call(prompt: str) -> str:
    from langsmith import get_current_run_tree
    
    run = get_current_run_tree()
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
        
    except Exception as e:
        if run:
            run.error = str(e)
            run.outputs = {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            }
        raise
```

---

## Feedback and Evaluation

### Collecting User Feedback

```python
from langsmith import Client

client = Client()

def submit_feedback(run_id: str, score: float, comment: str = None):
    """Submit user feedback for a trace"""
    
    client.create_feedback(
        run_id=run_id,
        key="user-rating",
        score=score,  # 0.0 to 1.0
        comment=comment
    )

def submit_thumbs(run_id: str, is_positive: bool):
    """Submit thumbs up/down feedback"""
    
    client.create_feedback(
        run_id=run_id,
        key="thumbs",
        score=1.0 if is_positive else 0.0
    )

# In your application
@traceable(name="chat_with_feedback")
def chat_with_feedback(message: str) -> tuple[str, str]:
    from langsmith import get_current_run_tree
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": message}]
    )
    
    run = get_current_run_tree()
    run_id = run.id if run else None
    
    return response.choices[0].message.content, run_id

# Usage
response, run_id = chat_with_feedback("Explain quantum computing")

# Later, when user provides feedback
submit_feedback(run_id, score=0.8, comment="Good explanation!")
```

### Automated Evaluation

```python
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# Define an evaluator
def accuracy_evaluator(run, example) -> dict:
    """Evaluate if output matches expected answer"""
    
    predicted = run.outputs.get("output", "")
    expected = example.outputs.get("answer", "")
    
    # Simple exact match (use better logic in practice)
    score = 1.0 if predicted.lower().strip() == expected.lower().strip() else 0.0
    
    return {
        "key": "accuracy",
        "score": score,
        "comment": f"Expected: {expected[:50]}, Got: {predicted[:50]}"
    }

def relevance_evaluator(run, example) -> dict:
    """Use LLM to evaluate relevance"""
    
    predicted = run.outputs.get("output", "")
    question = example.inputs.get("question", "")
    
    eval_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Rate how relevant the answer is to the question. Return only a number 0-10."
            },
            {
                "role": "user",
                "content": f"Question: {question}\nAnswer: {predicted}"
            }
        ]
    )
    
    try:
        score = float(eval_response.choices[0].message.content) / 10
    except:
        score = 0.5
    
    return {
        "key": "relevance",
        "score": score
    }

# Run evaluation
results = evaluate(
    target=my_chain,  # Your LangChain runnable or function
    data="my-dataset",  # Dataset name in LangSmith
    evaluators=[accuracy_evaluator, relevance_evaluator],
    experiment_prefix="qa-eval"
)
```

---

## Creating and Using Datasets

### Creating a Dataset

```python
from langsmith import Client

client = Client()

# Create dataset
dataset = client.create_dataset(
    dataset_name="qa-test-cases",
    description="Test cases for Q&A system"
)

# Add examples
examples = [
    {
        "inputs": {"question": "What is the capital of France?"},
        "outputs": {"answer": "Paris"}
    },
    {
        "inputs": {"question": "What is 2+2?"},
        "outputs": {"answer": "4"}
    },
    {
        "inputs": {"question": "Who wrote Romeo and Juliet?"},
        "outputs": {"answer": "William Shakespeare"}
    }
]

for example in examples:
    client.create_example(
        dataset_id=dataset.id,
        inputs=example["inputs"],
        outputs=example["outputs"]
    )
```

### Running Against Datasets

```python
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

def my_qa_function(inputs: dict) -> dict:
    """Your Q&A function to test"""
    question = inputs["question"]
    
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": question}]
    )
    
    return {"output": response.choices[0].message.content}

# Run evaluation against dataset
results = evaluate(
    target=my_qa_function,
    data="qa-test-cases",
    evaluators=[accuracy_evaluator],
    experiment_prefix="qa-v1"
)

# View results
for result in results:
    print(f"Input: {result.example.inputs}")
    print(f"Output: {result.run.outputs}")
    print(f"Score: {result.evaluation_results}")
    print("---")
```

---

## Production Monitoring

### Setting Up Monitoring

```python
from langsmith import Client
import time

client = Client()

def get_project_stats(project_name: str, hours: int = 24) -> dict:
    """Get statistics for a project"""
    
    # Get recent runs
    runs = list(client.list_runs(
        project_name=project_name,
        start_time=time.time() - (hours * 3600)
    ))
    
    if not runs:
        return {"message": "No runs found"}
    
    # Calculate statistics
    total_runs = len(runs)
    errors = sum(1 for r in runs if r.error)
    total_latency = sum(
        (r.end_time - r.start_time).total_seconds() 
        for r in runs if r.end_time
    )
    avg_latency = total_latency / total_runs if total_runs else 0
    
    # Token usage
    total_tokens = sum(
        r.total_tokens or 0 for r in runs
    )
    
    # Cost estimation (rough)
    estimated_cost = total_tokens * 0.00001  # Adjust based on your models
    
    return {
        "total_runs": total_runs,
        "error_rate": errors / total_runs if total_runs else 0,
        "avg_latency_seconds": avg_latency,
        "total_tokens": total_tokens,
        "estimated_cost_usd": estimated_cost
    }

# Get stats
stats = get_project_stats("my-llm-project", hours=24)
print(f"Last 24 hours:")
print(f"  Runs: {stats['total_runs']}")
print(f"  Error Rate: {stats['error_rate']:.1%}")
print(f"  Avg Latency: {stats['avg_latency_seconds']:.2f}s")
print(f"  Est. Cost: ${stats['estimated_cost_usd']:.2f}")
```

### Alerting on Issues

```python
from langsmith import Client
import smtplib
from email.mime.text import MIMEText

client = Client()

def check_and_alert(project_name: str, threshold_error_rate: float = 0.1):
    """Check error rate and send alert if too high"""
    
    stats = get_project_stats(project_name, hours=1)
    
    if stats["error_rate"] > threshold_error_rate:
        send_alert(
            f"High Error Rate Alert",
            f"Error rate for {project_name}: {stats['error_rate']:.1%}\n"
            f"Threshold: {threshold_error_rate:.1%}\n"
            f"Total runs in last hour: {stats['total_runs']}"
        )

def send_alert(subject: str, body: str):
    """Send email alert (implement with your email service)"""
    print(f"ALERT: {subject}\n{body}")
    # Implement actual email/Slack/PagerDuty notification
```

---

## Best Practices

### 1. Organize by Project

```python
import os

# Different projects for different environments
if os.getenv("ENV") == "production":
    os.environ["LANGCHAIN_PROJECT"] = "my-app-production"
else:
    os.environ["LANGCHAIN_PROJECT"] = "my-app-development"
```

### 2. Use Meaningful Names

```python
@traceable(name="customer-support-chat")  # Descriptive
def chat(...): pass

# Not this:
@traceable(name="func1")  # Vague
def chat(...): pass
```

### 3. Tag Important Traces

```python
@traceable(
    name="premium-user-query",
    tags=["premium", "high-priority"]
)
def premium_query(user_id: str, query: str):
    pass
```

### 4. Include Business Context

```python
@traceable(name="product-recommendation")
def recommend_products(user_id: str, category: str):
    from langsmith import get_current_run_tree
    
    run = get_current_run_tree()
    if run:
        run.metadata.update({
            "user_tier": get_user_tier(user_id),
            "category": category,
            "ab_test_variant": "v2"
        })
```

---

## Summary

| Feature | Use Case |
|---------|----------|
| `@traceable` | Manual tracing for any function |
| Metadata | Add context (user_id, version, etc.) |
| Tags | Filter and organize traces |
| Feedback | Collect user ratings |
| Datasets | Test against known inputs |
| Evaluation | Automated quality scoring |
| Monitoring | Track production performance |

**Key Takeaways:**
1. Enable tracing early in development
2. Add meaningful metadata and tags
3. Collect feedback for continuous improvement
4. Create datasets for regression testing
5. Monitor production metrics

---

## Next Steps

- [Phoenix/Arize](/learn/advanced-topics/observability/phoenix) - Alternative observability tools
- [Custom Logging](/learn/advanced-topics/observability/custom-logging) - Build your own tracing
- [Lab: Observability](/learn/advanced-topics/observability/observability-lab) - Hands-on implementation
