# Lab: Fine-tune with OpenAI API

In this hands-on lab, you'll fine-tune a GPT model using OpenAI's API to create a custom customer support assistant.

## Lab Overview

| Aspect | Details |
|--------|---------|
| **Duration** | 45 minutes |
| **Difficulty** | Intermediate |
| **Prerequisites** | OpenAI API key, LLM Fundamentals |
| **Cost** | ~$1-5 (depending on dataset size) |

---

## Learning Objectives

By completing this lab, you will:
- Prepare training data in OpenAI's JSONL format
- Upload data and create a fine-tuning job
- Monitor training progress
- Evaluate and use your fine-tuned model

---

## Scenario

You're building a customer support assistant for "TechFlow", a fictional SaaS company. The assistant should:
- Use a friendly, professional tone
- Follow a specific response format
- Handle common support queries consistently

---

## Exercise 1: Prepare Training Data

### Step 1: Create Training Examples

Create high-quality training examples that demonstrate your desired behavior.

```python
training_examples = [
    {
        "messages": [
            {"role": "system", "content": "You are TechFlow's helpful support assistant. Be friendly, concise, and always offer next steps."},
            {"role": "user", "content": "How do I reset my password?"},
            {"role": "assistant", "content": "I'd be happy to help you reset your password! 🔐\n\n**Quick Steps:**\n1. Go to techflow.com/reset\n2. Enter your email address\n3. Click the reset link in your email\n4. Create your new password\n\n**Need more help?** Let me know if the reset email doesn't arrive within 5 minutes!"}
        ]
    },
    {
        "messages": [
            {"role": "system", "content": "You are TechFlow's helpful support assistant. Be friendly, concise, and always offer next steps."},
            {"role": "user", "content": "What's your pricing?"},
            {"role": "assistant", "content": "Great question! Here's our current pricing: 💰\n\n**Plans:**\n- **Free**: $0/mo - Up to 100 tasks\n- **Pro**: $19/mo - Unlimited tasks + analytics\n- **Team**: $49/mo - Everything in Pro + collaboration\n- **Enterprise**: Custom - Contact sales\n\n**Ready to upgrade?** I can help you choose the right plan for your needs!"}
        ]
    },
    # Add more examples...
]
```

### Step 2: Validate Data Quality

```python
def validate_training_data(examples):
    """Check training data quality"""
    issues = []
    
    for i, ex in enumerate(examples):
        # Check structure
        if 'messages' not in ex:
            issues.append(f"Example {i}: Missing 'messages' field")
            continue
        
        messages = ex['messages']
        roles = [m['role'] for m in messages]
        
        # Must have system, user, and assistant
        if 'assistant' not in roles:
            issues.append(f"Example {i}: No assistant response")
        if 'user' not in roles:
            issues.append(f"Example {i}: No user message")
        
        # Check response quality
        for msg in messages:
            if msg['role'] == 'assistant':
                if len(msg['content']) < 50:
                    issues.append(f"Example {i}: Very short assistant response")
    
    if issues:
        print("❌ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ All examples valid!")
    
    return len(issues) == 0

# Run validation
validate_training_data(training_examples)
```

### Your Task

Create at least 20 training examples covering:
- [ ] Password/account issues (5 examples)
- [ ] Billing questions (5 examples)  
- [ ] Feature questions (5 examples)
- [ ] Bug reports (5 examples)

---

## Exercise 2: Upload and Train

### Step 1: Save to JSONL

```python
import json

def save_to_jsonl(examples, filename):
    """Save examples to JSONL format"""
    with open(filename, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex) + '\n')
    print(f"✅ Saved {len(examples)} examples to {filename}")

save_to_jsonl(training_examples, 'techflow_training.jsonl')
```

### Step 2: Upload to OpenAI

```python
from openai import OpenAI

client = OpenAI()

# Upload file
file = client.files.create(
    file=open("techflow_training.jsonl", "rb"),
    purpose="fine-tune"
)

print(f"✅ File uploaded: {file.id}")
```

### Step 3: Create Fine-tuning Job

```python
# Create fine-tuning job
job = client.fine_tuning.jobs.create(
    training_file=file.id,
    model="gpt-4o-mini-2024-07-18",
    suffix="techflow-support",
    hyperparameters={
        "n_epochs": 3
    }
)

print(f"✅ Job created: {job.id}")
print(f"Status: {job.status}")
```

### Step 4: Monitor Progress

```python
import time

def wait_for_training(job_id):
    """Wait for fine-tuning to complete"""
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        print(f"Status: {job.status}")
        
        if job.status == 'succeeded':
            print(f"✅ Training complete!")
            print(f"Model: {job.fine_tuned_model}")
            return job.fine_tuned_model
        elif job.status == 'failed':
            print(f"❌ Training failed: {job.error}")
            return None
        else:
            time.sleep(30)

model_name = wait_for_training(job.id)
```

---

## Exercise 3: Evaluate Your Model

### Compare Base vs Fine-tuned

```python
def compare_models(prompt, system_prompt):
    """Compare base and fine-tuned model responses"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    # Base model
    base = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    ).choices[0].message.content
    
    # Fine-tuned model
    ft = client.chat.completions.create(
        model=model_name,  # Your fine-tuned model
        messages=messages
    ).choices[0].message.content
    
    print("=" * 50)
    print("PROMPT:", prompt)
    print("\n📦 BASE MODEL:")
    print(base)
    print("\n🎯 FINE-TUNED MODEL:")
    print(ft)
    print("=" * 50)

# Test prompts
test_prompts = [
    "My payment failed, what do I do?",
    "Can I get a refund?",
    "How do I export my data?",
    "Your service is down!"
]

system = "You are TechFlow's helpful support assistant."

for prompt in test_prompts:
    compare_models(prompt, system)
    print()
```

### Evaluation Questions

Rate your fine-tuned model on each test case:

| Criterion | Score (1-5) | Notes |
|-----------|-------------|-------|
| Uses correct tone | | |
| Follows format | | |
| Helpful response | | |
| Offers next steps | | |

---

## Exercise 4: Iterate and Improve

### Identify Issues

Based on your evaluation, what needs improvement?

```python
improvement_areas = []

# Example issues:
# - "Model doesn't use emojis consistently"
# - "Responses are too long"
# - "Missing 'next steps' section in some responses"
```

### Add More Training Data

```python
# Create additional examples targeting weak areas
additional_examples = [
    # Add examples that fix identified issues
]

# Combine with original data
all_examples = training_examples + additional_examples
```

### Retrain with Improved Data

```python
# Save updated data
save_to_jsonl(all_examples, 'techflow_training_v2.jsonl')

# Upload and train again
file_v2 = client.files.create(
    file=open("techflow_training_v2.jsonl", "rb"),
    purpose="fine-tune"
)

job_v2 = client.fine_tuning.jobs.create(
    training_file=file_v2.id,
    model="gpt-4o-mini-2024-07-18",
    suffix="techflow-support-v2"
)
```

---

## Challenge: Production-Ready Assistant

Build a complete support assistant that:

1. **Handles multiple categories**: Account, billing, features, bugs
2. **Escalates appropriately**: Knows when to say "Let me connect you with a human"
3. **Maintains context**: References previous messages
4. **Stays safe**: Doesn't make promises or share sensitive info

### Success Criteria

- [ ] 90%+ format compliance
- [ ] Appropriate escalation on 5 test cases
- [ ] No hallucinated features or policies
- [ ] Consistent tone across 20 test prompts

---

## Cleanup

```python
# Delete fine-tuned models you no longer need
# client.models.delete("ft:gpt-4o-mini:your-org::abc123")

# Delete uploaded files
# client.files.delete(file.id)
```

---

## Key Takeaways

1. **Quality > Quantity** - 20 excellent examples beat 100 mediocre ones
2. **Test thoroughly** - Compare base vs fine-tuned on diverse prompts
3. **Iterate** - First version is rarely perfect
4. **Monitor costs** - Track training and inference expenses

---

## Solutions

See example training data and evaluation scripts in the solutions folder.
