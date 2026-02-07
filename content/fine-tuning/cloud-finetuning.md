# Cloud Fine-tuning with Provider APIs

Cloud providers offer managed fine-tuning services that handle infrastructure, training, and hosting. This lesson covers fine-tuning with OpenAI, Google, and other providers.

## Learning Objectives

By the end of this lesson, you will:
- Fine-tune models using OpenAI's API
- Understand pricing and limitations of cloud fine-tuning
- Monitor training progress and evaluate results
- Deploy and use fine-tuned models

---

## Cloud Fine-tuning Overview

### Why Use Cloud Fine-tuning?

```
┌─────────────────────────────────────────────────────────────┐
│              Cloud vs Local Fine-tuning                     │
│                                                             │
│  Cloud (OpenAI, etc.)         Local (Your Hardware)         │
│  ────────────────────         ─────────────────────         │
│  ✓ No GPU required            ✓ Full control                │
│  ✓ Managed infrastructure     ✓ Keep data private           │
│  ✓ Easy to get started        ✓ No usage fees               │
│  ✓ Automatic scaling          ✓ Customize everything        │
│                                                             │
│  ✗ Data leaves your control   ✗ Requires GPU hardware       │
│  ✗ Limited customization      ✗ More technical setup        │
│  ✗ Ongoing usage costs        ✗ You manage infrastructure   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Provider Comparison

| Provider | Models Available | Min Examples | Pricing Model |
|----------|-----------------|--------------|---------------|
| **OpenAI** | GPT-4o-mini, GPT-4o | 10 | Per 1K tokens trained |
| **Google Vertex** | Gemini, PaLM 2 | Varies | Per node-hour |
| **Anthropic** | Not yet available | - | - |
| **Together AI** | LLaMA, Mistral, etc. | 1 | Per token |
| **Anyscale** | Open models | 1 | Per GPU-hour |

---

## OpenAI Fine-tuning

### Supported Models

| Model | Fine-tunable | Context | Best For |
|-------|--------------|---------|----------|
| **gpt-4o-mini-2024-07-18** | ✓ | 128K | Cost-effective, general |
| **gpt-4o-2024-08-06** | ✓ | 128K | Complex tasks, highest quality |
| **gpt-3.5-turbo-0125** | ✓ | 16K | Legacy, budget option |

### Step 1: Prepare Your Data

```python
import json

# Create training data in JSONL format
training_data = [
    {
        "messages": [
            {"role": "system", "content": "You are a helpful customer support agent for TechCo."},
            {"role": "user", "content": "How do I reset my password?"},
            {"role": "assistant", "content": "I'd be happy to help you reset your password! Here's how:\n\n1. Go to techco.com/reset\n2. Enter your email address\n3. Click 'Send Reset Link'\n4. Check your email and click the link\n5. Create a new password\n\nThe link expires in 24 hours. Let me know if you need any other help!"}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are a helpful customer support agent for TechCo."},
            {"role": "user", "content": "What are your pricing plans?"},
            {"role": "assistant", "content": "Great question! Here are our current plans:\n\n**Free Tier**: $0/month\n- Up to 100 API calls\n- Basic features\n\n**Pro**: $29/month\n- Unlimited API calls\n- Priority support\n- Advanced analytics\n\n**Enterprise**: Custom pricing\n- Dedicated infrastructure\n- SLA guarantee\n- Custom integrations\n\nWould you like me to help you choose the right plan for your needs?"}
        ]
    }
    # Add more examples...
]

# Write to JSONL file
with open('training_data.jsonl', 'w') as f:
    for item in training_data:
        f.write(json.dumps(item) + '\n')
```

### Step 2: Upload Training File

```python
from openai import OpenAI

client = OpenAI()

# Upload the file
file = client.files.create(
    file=open("training_data.jsonl", "rb"),
    purpose="fine-tune"
)

print(f"File uploaded: {file.id}")
# Output: file-abc123...
```

### Step 3: Create Fine-tuning Job

```python
# Create fine-tuning job
job = client.fine_tuning.jobs.create(
    training_file=file.id,
    model="gpt-4o-mini-2024-07-18",
    hyperparameters={
        "n_epochs": 3,  # Number of training epochs
        "batch_size": "auto",
        "learning_rate_multiplier": "auto"
    },
    suffix="techco-support"  # Custom name suffix
)

print(f"Job created: {job.id}")
print(f"Status: {job.status}")
```

### Step 4: Monitor Training Progress

```python
import time

def monitor_job(job_id):
    """Monitor fine-tuning job until completion"""
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        print(f"Status: {job.status}")
        
        if job.status == 'succeeded':
            print(f"✅ Training complete!")
            print(f"Fine-tuned model: {job.fine_tuned_model}")
            return job.fine_tuned_model
        
        elif job.status == 'failed':
            print(f"❌ Training failed: {job.error}")
            return None
        
        elif job.status in ['validating_files', 'queued', 'running']:
            # Get training metrics if available
            if hasattr(job, 'result_files') and job.result_files:
                events = client.fine_tuning.jobs.list_events(
                    fine_tuning_job_id=job_id,
                    limit=5
                )
                for event in events.data:
                    print(f"  {event.message}")
            
            time.sleep(60)  # Check every minute
        
        else:
            print(f"Unknown status: {job.status}")
            time.sleep(60)

# Monitor
model_name = monitor_job(job.id)
```

### Step 5: Use Your Fine-tuned Model

```python
# Use the fine-tuned model
response = client.chat.completions.create(
    model="ft:gpt-4o-mini-2024-07-18:your-org::abc123",  # Your fine-tuned model ID
    messages=[
        {"role": "system", "content": "You are a helpful customer support agent for TechCo."},
        {"role": "user", "content": "Can I get a refund?"}
    ]
)

print(response.choices[0].message.content)
```

---

## OpenAI Fine-tuning Pricing

### Training Costs

| Model | Training Cost | Input Cost (Inference) | Output Cost (Inference) |
|-------|--------------|------------------------|------------------------|
| gpt-4o-mini | $3.00 / 1M tokens | $0.30 / 1M | $1.20 / 1M |
| gpt-4o | $25.00 / 1M tokens | $3.75 / 1M | $15.00 / 1M |
| gpt-3.5-turbo | $8.00 / 1M tokens | $3.00 / 1M | $6.00 / 1M |

### Cost Estimation

```python
def estimate_training_cost(filepath, model="gpt-4o-mini"):
    """Estimate fine-tuning cost"""
    import tiktoken
    
    pricing = {
        "gpt-4o-mini": 3.00,  # per 1M tokens
        "gpt-4o": 25.00,
        "gpt-3.5-turbo": 8.00
    }
    
    enc = tiktoken.encoding_for_model("gpt-4o")
    total_tokens = 0
    
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            for msg in data['messages']:
                total_tokens += len(enc.encode(msg['content']))
    
    # Training typically runs for 3-4 epochs
    epochs = 3
    training_tokens = total_tokens * epochs
    
    cost = (training_tokens / 1_000_000) * pricing[model]
    
    print(f"Tokens per example: {total_tokens}")
    print(f"Training tokens (x{epochs} epochs): {training_tokens:,}")
    print(f"Estimated cost: ${cost:.2f}")
    
    return cost

estimate_training_cost('training_data.jsonl')
```

---

## Advanced OpenAI Options

### Hyperparameter Tuning

```python
job = client.fine_tuning.jobs.create(
    training_file=file.id,
    model="gpt-4o-mini-2024-07-18",
    hyperparameters={
        "n_epochs": 4,  # More epochs for smaller datasets
        "batch_size": 4,  # Smaller batch for more updates
        "learning_rate_multiplier": 0.5  # Lower LR for stability
    }
)
```

### Hyperparameter Guidelines

| Parameter | Small Dataset (<100) | Medium (100-1000) | Large (>1000) |
|-----------|---------------------|-------------------|---------------|
| **n_epochs** | 3-6 | 2-4 | 1-2 |
| **batch_size** | 1-4 | 4-8 | 8-16 |
| **learning_rate** | 0.5-1.0 | 1.0-1.5 | 1.0-2.0 |

### Validation Data

```python
# Upload validation file
val_file = client.files.create(
    file=open("validation_data.jsonl", "rb"),
    purpose="fine-tune"
)

# Create job with validation
job = client.fine_tuning.jobs.create(
    training_file=train_file.id,
    validation_file=val_file.id,
    model="gpt-4o-mini-2024-07-18"
)
```

### List and Manage Jobs

```python
# List all fine-tuning jobs
jobs = client.fine_tuning.jobs.list(limit=10)
for job in jobs.data:
    print(f"{job.id}: {job.status} - {job.fine_tuned_model or 'training...'}")

# Cancel a running job
client.fine_tuning.jobs.cancel("ftjob-abc123")

# Delete a fine-tuned model
client.models.delete("ft:gpt-4o-mini:your-org::abc123")
```

---

## Google Vertex AI Fine-tuning

### Setup

```python
from google.cloud import aiplatform

aiplatform.init(
    project="your-project-id",
    location="us-central1"
)
```

### Create Tuning Job

```python
from vertexai.generative_models import GenerativeModel
from vertexai.tuning import sft

# Prepare data in Google's format
# Upload to Google Cloud Storage
training_data_uri = "gs://your-bucket/training_data.jsonl"

# Create supervised fine-tuning job
sft_tuning_job = sft.train(
    source_model="gemini-1.5-flash-002",
    train_dataset=training_data_uri,
    tuned_model_display_name="my-tuned-gemini",
    epochs=3,
    learning_rate_multiplier=1.0,
)

# Monitor progress
print(f"Job: {sft_tuning_job.name}")
print(f"Status: {sft_tuning_job.state}")
```

### Use Tuned Model

```python
# Load and use tuned model
tuned_model = GenerativeModel(sft_tuning_job.tuned_model_endpoint_name)

response = tuned_model.generate_content("Your prompt here")
print(response.text)
```

---

## Together AI Fine-tuning

Together AI offers fine-tuning for open-source models:

### Create Fine-tuning Job

```python
import together

# Upload data
file = together.Files.upload(file="training_data.jsonl")

# Create fine-tuning job
response = together.Finetune.create(
    training_file=file['id'],
    model='meta-llama/Meta-Llama-3.1-8B-Instruct',
    n_epochs=3,
    learning_rate=1e-5,
    wandb_api_key="optional-for-logging"
)

print(f"Job ID: {response['id']}")
```

### Available Models

| Model | Size | Context | Best For |
|-------|------|---------|----------|
| Llama 3.1 8B | 8B | 128K | General, efficient |
| Llama 3.1 70B | 70B | 128K | Complex tasks |
| Mistral 7B | 7B | 32K | Fast, capable |
| Mixtral 8x7B | 46B | 32K | MoE efficiency |
| CodeLlama | 7-34B | 16K | Code generation |

---

## Best Practices

### Before Fine-tuning

```
┌─────────────────────────────────────────────────────────────┐
│                Pre-Training Checklist                       │
│                                                             │
│  □ Validated data format (use provider's validator)        │
│  □ Checked for PII/sensitive data                          │
│  □ Created held-out test set                               │
│  □ Estimated training cost                                  │
│  □ Set success criteria (what makes it "good enough"?)     │
│  □ Tried prompting first (is fine-tuning necessary?)       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### During Fine-tuning

```python
# Set up monitoring and alerts
def check_training_metrics(job_id):
    """Check training loss and validation metrics"""
    events = client.fine_tuning.jobs.list_events(
        fine_tuning_job_id=job_id,
        limit=100
    )
    
    for event in events.data:
        if 'metrics' in event.data:
            metrics = event.data['metrics']
            print(f"Step {metrics.get('step')}: Loss = {metrics.get('train_loss'):.4f}")
```

### After Fine-tuning

```python
def evaluate_fine_tuned_model(model_name, test_cases):
    """Evaluate fine-tuned model against test cases"""
    results = []
    
    for test in test_cases:
        response = client.chat.completions.create(
            model=model_name,
            messages=test['messages']
        )
        
        output = response.choices[0].message.content
        
        # Compare to expected output
        results.append({
            'input': test['messages'][-1]['content'],
            'expected': test['expected'],
            'actual': output,
            'matches': test['expected'].lower() in output.lower()
        })
    
    accuracy = sum(r['matches'] for r in results) / len(results)
    print(f"Accuracy: {accuracy:.1%}")
    
    return results
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "Invalid file format" | Wrong JSONL structure | Validate with provider's tool |
| "Training failed" | Data quality issues | Check for empty/truncated examples |
| "Model not improving" | Too few examples | Add more diverse data |
| "Overfitting" | Too many epochs | Reduce epochs, add validation |
| "Wrong style" | Inconsistent training data | Audit and standardize examples |

### Debugging Poor Results

```python
# Compare base vs fine-tuned outputs
def compare_models(prompt, system_prompt):
    """Compare base model with fine-tuned model"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    # Base model
    base_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    
    # Fine-tuned model
    ft_response = client.chat.completions.create(
        model="ft:gpt-4o-mini:your-org::abc123",
        messages=messages
    )
    
    print("=" * 50)
    print("BASE MODEL:")
    print(base_response.choices[0].message.content)
    print("\nFINE-TUNED MODEL:")
    print(ft_response.choices[0].message.content)
    print("=" * 50)
```

---

## Key Takeaways

1. **Start small** - Use minimum viable dataset to validate approach

2. **Validate data** - Use provider tools to check format before uploading

3. **Monitor training** - Watch loss curves and validation metrics

4. **Test thoroughly** - Compare against base model on diverse inputs

5. **Iterate** - First fine-tune rarely perfect; plan to improve data

6. **Consider costs** - Training + inference costs add up

---

## What's Next?

Want more control over fine-tuning? Learn to **fine-tune locally** with LoRA and QLoRA → [Local Fine-tuning](/learn/fine-tuning/local-finetuning/local-finetuning)

---

## Lab Exercise

Practice what you've learned:
→ [Lab: Fine-tune with OpenAI API](/learn/fine-tuning/cloud-finetuning/openai-finetune)
