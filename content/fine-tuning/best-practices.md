# Fine-tuning Best Practices

Learn how to evaluate fine-tuned models, avoid common pitfalls, and optimize for quality and cost.

## Learning Objectives

By the end of this lesson, you will:
- Evaluate fine-tuned model performance effectively
- Avoid catastrophic forgetting
- Optimize training for cost and quality
- Debug and improve fine-tuning results

---

## Evaluation Framework

### What to Measure

```
┌─────────────────────────────────────────────────────────────┐
│              Fine-tuning Evaluation Metrics                 │
│                                                             │
│  TASK-SPECIFIC METRICS                                      │
│  ─────────────────────                                      │
│  • Accuracy on your test set                               │
│  • Format compliance rate                                   │
│  • Domain-specific correctness                              │
│                                                             │
│  GENERAL CAPABILITIES                                       │
│  ─────────────────────                                      │
│  • Perplexity on held-out data                             │
│  • Performance on standard benchmarks                       │
│  • General instruction-following                            │
│                                                             │
│  QUALITY FACTORS                                            │
│  ───────────────                                            │
│  • Response coherence                                       │
│  • Factual accuracy                                         │
│  • Style consistency                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Creating a Test Set

```python
def create_evaluation_set(training_data, test_ratio=0.1, seed=42):
    """Create held-out test set from training data"""
    import random
    random.seed(seed)
    
    data = training_data.copy()
    random.shuffle(data)
    
    split_idx = int(len(data) * (1 - test_ratio))
    train_set = data[:split_idx]
    test_set = data[split_idx:]
    
    print(f"Training examples: {len(train_set)}")
    print(f"Test examples: {len(test_set)}")
    
    return train_set, test_set
```

### Automated Evaluation

```python
from openai import OpenAI
import json

client = OpenAI()

def evaluate_model(model_name, test_cases):
    """Evaluate fine-tuned model on test cases"""
    
    results = {
        'correct': 0,
        'format_compliant': 0,
        'total': len(test_cases),
        'examples': []
    }
    
    for test in test_cases:
        # Generate response
        response = client.chat.completions.create(
            model=model_name,
            messages=test['messages'][:-1],  # Exclude expected response
            temperature=0  # Deterministic for evaluation
        )
        
        actual = response.choices[0].message.content
        expected = test['messages'][-1]['content']
        
        # Check correctness (customize based on your task)
        is_correct = evaluate_correctness(actual, expected)
        is_format_compliant = check_format(actual, test.get('expected_format'))
        
        results['correct'] += is_correct
        results['format_compliant'] += is_format_compliant
        results['examples'].append({
            'input': test['messages'][-2]['content'],
            'expected': expected,
            'actual': actual,
            'correct': is_correct
        })
    
    results['accuracy'] = results['correct'] / results['total']
    results['format_rate'] = results['format_compliant'] / results['total']
    
    return results

def evaluate_correctness(actual, expected):
    """Custom correctness check - modify for your use case"""
    # Simple similarity check
    actual_lower = actual.lower().strip()
    expected_lower = expected.lower().strip()
    
    # Exact match
    if actual_lower == expected_lower:
        return True
    
    # Key content match
    key_phrases = extract_key_phrases(expected)
    matches = sum(1 for p in key_phrases if p in actual_lower)
    return matches / len(key_phrases) > 0.8 if key_phrases else False

def check_format(response, expected_format):
    """Check if response follows expected format"""
    if not expected_format:
        return True
    
    if expected_format == 'json':
        try:
            json.loads(response)
            return True
        except:
            return False
    
    # Add more format checks as needed
    return True
```

### LLM-as-Judge Evaluation

```python
def llm_judge_evaluation(test_case, model_response, criteria):
    """Use GPT-4 to evaluate response quality"""
    
    judge_prompt = f"""Evaluate this AI assistant response on a scale of 1-5 for each criterion.

USER QUERY:
{test_case['query']}

EXPECTED RESPONSE CHARACTERISTICS:
{test_case.get('expected_characteristics', 'Follow instructions accurately')}

ACTUAL RESPONSE:
{model_response}

EVALUATION CRITERIA:
{json.dumps(criteria, indent=2)}

For each criterion, provide:
1. Score (1-5)
2. Brief justification

Return as JSON with format:
{{"criterion_name": {{"score": N, "justification": "..."}}, ...}}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": judge_prompt}],
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)

# Example criteria
evaluation_criteria = {
    "accuracy": "Response is factually correct and addresses the query",
    "helpfulness": "Response provides useful, actionable information",
    "style": "Response matches expected tone and format",
    "completeness": "Response covers all aspects of the query"
}
```

---

## Avoiding Catastrophic Forgetting

### The Problem

```
┌─────────────────────────────────────────────────────────────┐
│              Catastrophic Forgetting                        │
│                                                             │
│  Before Fine-tuning:                                        │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ General Knowledge ████████████████████████████████  │   │
│  │ Math Ability      ████████████████████████████████  │   │
│  │ Coding Skills     ████████████████████████████████  │   │
│  │ Domain Knowledge  ███                                │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  After Aggressive Fine-tuning:                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ General Knowledge ██████████████                     │   │
│  │ Math Ability      █████████                          │   │
│  │ Coding Skills     ████████                           │   │
│  │ Domain Knowledge  ████████████████████████████████  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Model becomes narrow specialist, loses general abilities  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Prevention Strategies

#### 1. Use LoRA Instead of Full Fine-tuning

```python
# LoRA keeps base model frozen
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    # Base model weights are NOT modified
)
```

#### 2. Mix in General Data

```python
def create_mixed_dataset(domain_data, general_data, domain_ratio=0.7):
    """Mix domain-specific and general data"""
    
    # Calculate sizes
    total_size = len(domain_data) / domain_ratio
    general_size = int(total_size * (1 - domain_ratio))
    
    # Sample general data
    import random
    general_sample = random.sample(general_data, min(general_size, len(general_data)))
    
    # Combine and shuffle
    mixed = domain_data + general_sample
    random.shuffle(mixed)
    
    print(f"Domain examples: {len(domain_data)} ({domain_ratio:.0%})")
    print(f"General examples: {len(general_sample)} ({1-domain_ratio:.0%})")
    
    return mixed

# Usage
mixed_data = create_mixed_dataset(
    domain_data=my_training_data,
    general_data=load_dataset("Open-Orca/OpenOrca")['train'],
    domain_ratio=0.7
)
```

#### 3. Lower Learning Rate

```python
training_args = TrainingArguments(
    # Lower learning rate = gentler updates
    learning_rate=1e-5,  # Instead of 2e-4
    
    # Warmup helps stability
    warmup_ratio=0.1,
    
    # Fewer epochs
    num_train_epochs=2,  # Instead of 4+
)
```

#### 4. Regularization

```python
training_args = TrainingArguments(
    # Weight decay prevents overfitting
    weight_decay=0.1,
    
    # Lower LoRA dropout
    # In LoraConfig:
    # lora_dropout=0.1,
)
```

### Testing for Forgetting

```python
def test_general_capabilities(model, tokenizer):
    """Test model on general tasks to check for forgetting"""
    
    general_tests = [
        {
            "task": "Math",
            "prompt": "What is 15 * 23?",
            "expected_contains": "345"
        },
        {
            "task": "Coding",
            "prompt": "Write a Python function to reverse a string",
            "expected_contains": "def"
        },
        {
            "task": "General Knowledge",
            "prompt": "What is the capital of France?",
            "expected_contains": "Paris"
        },
        {
            "task": "Reasoning",
            "prompt": "If all cats are animals, and Fluffy is a cat, what can we conclude?",
            "expected_contains": "animal"
        }
    ]
    
    results = []
    for test in general_tests:
        response = generate(model, tokenizer, test["prompt"])
        passed = test["expected_contains"].lower() in response.lower()
        results.append({
            "task": test["task"],
            "passed": passed,
            "response": response[:200]
        })
    
    passed_count = sum(r["passed"] for r in results)
    print(f"General capability score: {passed_count}/{len(results)}")
    
    return results
```

---

## Cost Optimization

### Training Cost Factors

```
Cost = Tokens × Epochs × Price_per_token

┌─────────────────────────────────────────────────────────────┐
│              Cost Optimization Levers                       │
│                                                             │
│  1. REDUCE TOKENS                                           │
│     • Use shorter examples                                 │
│     • Remove unnecessary system prompts                    │
│     • Truncate verbose responses                           │
│                                                             │
│  2. REDUCE EPOCHS                                           │
│     • Start with 1-2 epochs                                │
│     • Use early stopping                                   │
│     • Monitor validation loss                              │
│                                                             │
│  3. CHOOSE CHEAPER MODEL                                    │
│     • gpt-4o-mini instead of gpt-4o                       │
│     • Smaller open-source model                            │
│                                                             │
│  4. LOCAL TRAINING (no token costs)                        │
│     • Use QLoRA on consumer GPU                            │
│     • One-time compute cost                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Efficient Data Strategies

```python
def optimize_training_data(data, max_tokens=1000):
    """Optimize data for cost-effective training"""
    
    optimized = []
    for item in data:
        # Remove redundant system prompts (keep one representative)
        messages = item['messages']
        
        # Truncate very long responses
        for msg in messages:
            if msg['role'] == 'assistant' and len(msg['content']) > 2000:
                msg['content'] = truncate_response(msg['content'], max_chars=2000)
        
        # Skip examples that are too long
        total_length = sum(len(m['content']) for m in messages)
        if total_length < max_tokens * 4:  # Rough token estimate
            optimized.append(item)
    
    print(f"Kept {len(optimized)}/{len(data)} examples")
    return optimized
```

### Early Stopping

```python
from transformers import EarlyStoppingCallback

training_args = TrainingArguments(
    # ... other args ...
    evaluation_strategy="steps",
    eval_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

trainer = SFTTrainer(
    # ... other args ...
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)
```

---

## Debugging Poor Results

### Diagnosis Framework

```
┌─────────────────────────────────────────────────────────────┐
│              Debugging Decision Tree                        │
│                                                             │
│  Model not learning?                                        │
│  ├── Learning rate too low → Increase 2-10x                │
│  ├── Learning rate too high → Decrease, watch for NaN      │
│  └── Data format wrong → Validate with provider tools      │
│                                                             │
│  Model overfitting?                                         │
│  ├── Training loss ↓ but val loss ↑                        │
│  ├── → Add regularization (weight decay, dropout)          │
│  ├── → Reduce epochs                                       │
│  └── → Add more training data                              │
│                                                             │
│  Wrong style/format?                                        │
│  ├── Inconsistent training examples → Audit and fix        │
│  ├── Too few examples → Add more examples of target style  │
│  └── Examples don't match use case → Better examples       │
│                                                             │
│  Degraded general ability?                                  │
│  ├── → Mix in general data                                 │
│  ├── → Use LoRA instead of full fine-tune                  │
│  └── → Lower learning rate                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Analyzing Training Logs

```python
def analyze_training_run(log_file):
    """Analyze training logs for issues"""
    import json
    
    with open(log_file) as f:
        logs = [json.loads(line) for line in f]
    
    train_losses = [l['loss'] for l in logs if 'loss' in l]
    eval_losses = [l['eval_loss'] for l in logs if 'eval_loss' in l]
    
    # Check for issues
    issues = []
    
    # Loss not decreasing
    if len(train_losses) > 10:
        recent = train_losses[-5:]
        early = train_losses[:5]
        if sum(recent) >= sum(early):
            issues.append("Training loss not decreasing - check learning rate")
    
    # NaN values
    if any(math.isnan(l) for l in train_losses):
        issues.append("NaN loss detected - reduce learning rate")
    
    # Overfitting
    if eval_losses:
        if eval_losses[-1] > eval_losses[0] and train_losses[-1] < train_losses[0]:
            issues.append("Possible overfitting - validation loss increasing")
    
    return issues
```

### A/B Testing Fine-tuned Models

```python
def ab_test_models(model_a, model_b, test_prompts, judge_model="gpt-4o"):
    """A/B test two models using LLM judge"""
    
    results = {"model_a_wins": 0, "model_b_wins": 0, "ties": 0}
    
    for prompt in test_prompts:
        # Get responses
        response_a = generate(model_a, prompt)
        response_b = generate(model_b, prompt)
        
        # Judge comparison
        judge_prompt = f"""Compare these two AI responses to the same prompt.

PROMPT: {prompt}

RESPONSE A:
{response_a}

RESPONSE B:
{response_b}

Which response is better? Consider accuracy, helpfulness, and clarity.
Reply with just: "A", "B", or "TIE" """

        judgment = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0
        ).choices[0].message.content.strip().upper()
        
        if "A" in judgment:
            results["model_a_wins"] += 1
        elif "B" in judgment:
            results["model_b_wins"] += 1
        else:
            results["ties"] += 1
    
    total = len(test_prompts)
    print(f"Model A wins: {results['model_a_wins']}/{total} ({results['model_a_wins']/total:.1%})")
    print(f"Model B wins: {results['model_b_wins']}/{total} ({results['model_b_wins']/total:.1%})")
    print(f"Ties: {results['ties']}/{total}")
    
    return results
```

---

## Production Checklist

### Before Deployment

```
┌─────────────────────────────────────────────────────────────┐
│              Production Readiness Checklist                 │
│                                                             │
│  □ Evaluated on held-out test set                          │
│  □ Tested for catastrophic forgetting                      │
│  □ Compared to base model performance                      │
│  □ Tested edge cases and failure modes                     │
│  □ Verified format compliance                              │
│  □ Checked for harmful/biased outputs                      │
│  □ Documented training data and parameters                 │
│  □ Set up monitoring for production                        │
│  □ Created rollback plan                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Monitoring in Production

```python
def log_inference(model_name, prompt, response, metadata=None):
    """Log inference for monitoring"""
    import datetime
    
    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model": model_name,
        "prompt_tokens": count_tokens(prompt),
        "response_tokens": count_tokens(response),
        "latency_ms": metadata.get("latency_ms"),
        "prompt_preview": prompt[:100],
        "response_preview": response[:100],
    }
    
    # Log to your monitoring system
    # logging.info(json.dumps(log_entry))
    
    # Check for issues
    if metadata.get("latency_ms", 0) > 5000:
        alert("High latency detected")
    
    if len(response) < 10:
        alert("Unusually short response")
```

---

## Key Takeaways

1. **Evaluate comprehensively** - Test both task-specific and general capabilities

2. **Prevent forgetting** - Use LoRA, mix data, lower learning rates

3. **Optimize costs** - Shorter examples, fewer epochs, early stopping

4. **Debug systematically** - Use the diagnosis framework

5. **A/B test** - Compare fine-tuned vs base, different versions

6. **Monitor production** - Track quality metrics over time

---

## Summary: Fine-tuning Module Complete

You've now learned the complete fine-tuning workflow:

1. **When to fine-tune** - vs RAG and prompting
2. **Types of fine-tuning** - Full, LoRA, QLoRA
3. **Data preparation** - Formats, quality, sizing
4. **Cloud fine-tuning** - OpenAI, Google, Together
5. **Local fine-tuning** - LoRA with PEFT
6. **Best practices** - Evaluation, debugging, optimization

---

## What's Next?

Ready to apply your knowledge? Try the labs:
- [Lab: Fine-tune with OpenAI API](/learn/fine-tuning/cloud-finetuning/openai-finetune)
- [Lab: Local Fine-tuning with QLoRA](/learn/fine-tuning/local-finetuning/qlora)

Or continue to:
- [Self-Hosting LLMs](../self-hosting/introduction.md) - Run models locally
- [RAG](../rag/introduction.md) - When fine-tuning isn't the answer
