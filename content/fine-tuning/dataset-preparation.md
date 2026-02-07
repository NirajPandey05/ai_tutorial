# Dataset Preparation for Fine-tuning

The quality of your fine-tuning results depends heavily on your training data. This lesson covers how to create, format, and validate datasets for LLM fine-tuning.

## Learning Objectives

By the end of this lesson, you will:
- Understand different training data formats
- Know how to create high-quality training examples
- Apply data validation and cleaning techniques
- Size your dataset appropriately

---

## Training Data Formats

Different fine-tuning scenarios use different data formats:

### 1. Instruction Format

Best for: General instruction-following, task completion

```json
{
  "instruction": "Summarize the following article",
  "input": "The article text goes here...",
  "output": "This is the summary..."
}
```

### 2. Conversation Format

Best for: Chat assistants, multi-turn dialogue

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language..."}
  ]
}
```

### 3. Completion Format

Best for: Text completion, continuation tasks

```json
{
  "prompt": "The capital of France is",
  "completion": " Paris."
}
```

### Format Comparison

| Format | Use Case | Providers |
|--------|----------|-----------|
| **Instruction** | Task-specific fine-tuning | Local, Hugging Face |
| **Conversation** | Chat fine-tuning | OpenAI, Anthropic |
| **Completion** | Simple completion tasks | Legacy APIs |

---

## OpenAI Format (JSONL)

### Chat Format

```jsonl
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What's 2+2?"}, {"role": "assistant", "content": "2+2 equals 4."}]}
```

### Requirements

| Requirement | Details |
|-------------|---------|
| Format | JSONL (one JSON per line) |
| Minimum examples | 10 (recommended: 50-100+) |
| Maximum tokens | Model context limit |
| Roles | system, user, assistant |

### Validation Script

```python
import json

def validate_openai_format(filepath):
    """Validate OpenAI fine-tuning format"""
    errors = []
    
    with open(filepath, 'r') as f:
        for i, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                
                # Check messages field exists
                if 'messages' not in data:
                    errors.append(f"Line {i}: Missing 'messages' field")
                    continue
                
                messages = data['messages']
                
                # Check at least one user and assistant message
                roles = [m['role'] for m in messages]
                if 'user' not in roles:
                    errors.append(f"Line {i}: No 'user' message")
                if 'assistant' not in roles:
                    errors.append(f"Line {i}: No 'assistant' message")
                
                # Check message structure
                for j, msg in enumerate(messages):
                    if 'role' not in msg or 'content' not in msg:
                        errors.append(f"Line {i}, Message {j}: Missing role or content")
                    
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: Invalid JSON - {e}")
    
    return errors

# Usage
errors = validate_openai_format('training_data.jsonl')
if errors:
    for error in errors:
        print(error)
else:
    print("Dataset is valid!")
```

---

## Alpaca/Stanford Format

Popular for open-source model fine-tuning:

### Format

```json
[
  {
    "instruction": "Write a haiku about programming",
    "input": "",
    "output": "Code flows like water\nBugs emerge from the shadows\nCoffee fuels the fix"
  },
  {
    "instruction": "Translate to French",
    "input": "Hello, how are you?",
    "output": "Bonjour, comment allez-vous?"
  }
]
```

### Prompt Template

The data is combined into prompts during training:

```
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

### Conversion Script

```python
def alpaca_to_openai(alpaca_data):
    """Convert Alpaca format to OpenAI chat format"""
    openai_data = []
    
    for item in alpaca_data:
        messages = []
        
        # Combine instruction and input
        user_content = item['instruction']
        if item.get('input'):
            user_content += f"\n\nInput: {item['input']}"
        
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": item['output']})
        
        openai_data.append({"messages": messages})
    
    return openai_data
```

---

## ShareGPT Format

Used for multi-turn conversations:

### Format

```json
{
  "conversations": [
    {"from": "human", "value": "Hi, I need help with Python"},
    {"from": "gpt", "value": "Hello! I'd be happy to help with Python."},
    {"from": "human", "value": "How do I read a file?"},
    {"from": "gpt", "value": "You can use the open() function..."}
  ]
}
```

### Conversion Script

```python
def sharegpt_to_openai(sharegpt_data):
    """Convert ShareGPT format to OpenAI format"""
    role_map = {"human": "user", "gpt": "assistant"}
    
    openai_data = []
    for item in sharegpt_data:
        messages = []
        for turn in item['conversations']:
            role = role_map.get(turn['from'], turn['from'])
            messages.append({"role": role, "content": turn['value']})
        openai_data.append({"messages": messages})
    
    return openai_data
```

---

## Creating Quality Training Data

### Data Quality Checklist

```
┌─────────────────────────────────────────────────────────────┐
│              Training Data Quality Checklist                │
│                                                             │
│  □ Diverse: Covers range of expected inputs                │
│  □ Representative: Matches production distribution          │
│  □ Accurate: Outputs are correct and high-quality          │
│  □ Consistent: Same style/format throughout                 │
│  □ Complete: No truncated or incomplete examples           │
│  □ Balanced: No over-represented categories                │
│  □ Clean: No typos, formatting issues, or errors           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Example: Creating Customer Support Data

```python
# Good example: Specific, helpful, on-brand
good_example = {
    "messages": [
        {
            "role": "system", 
            "content": "You are a helpful customer support agent for TechCo. Be friendly, concise, and solution-oriented."
        },
        {
            "role": "user",
            "content": "I can't log into my account. It says my password is wrong but I know it's correct."
        },
        {
            "role": "assistant",
            "content": "I'm sorry you're having trouble logging in! Let's get this sorted out. Here are a few things to try:\n\n1. **Check Caps Lock** - Passwords are case-sensitive\n2. **Clear browser cache** - Old saved credentials might interfere\n3. **Try password reset** - I can send a reset link to your email\n\nWould you like me to send that reset link? I just need to verify your account email."
        }
    ]
}

# Bad example: Vague, unhelpful, no brand voice
bad_example = {
    "messages": [
        {"role": "user", "content": "cant login"},
        {"role": "assistant", "content": "try again later"}
    ]
}
```

### Example: Creating Code Assistant Data

```python
# High-quality code example
good_code_example = {
    "messages": [
        {
            "role": "system",
            "content": "You are an expert Python developer. Write clean, well-documented code with error handling."
        },
        {
            "role": "user",
            "content": "Write a function to read a CSV file and return it as a list of dictionaries"
        },
        {
            "role": "assistant",
            "content": '''```python
import csv
from typing import List, Dict
from pathlib import Path

def read_csv_as_dicts(filepath: str) -> List[Dict[str, str]]:
    """
    Read a CSV file and return its contents as a list of dictionaries.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        List of dictionaries, where each dict represents a row
        with column headers as keys
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        csv.Error: If the file is not valid CSV
    
    Example:
        >>> data = read_csv_as_dicts('users.csv')
        >>> print(data[0])
        {'name': 'Alice', 'email': 'alice@example.com'}
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    
    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)
```

This function:
- Uses type hints for clarity
- Includes comprehensive docstring
- Has proper error handling
- Uses pathlib for cross-platform compatibility'''
        }
    ]
}
```

---

## Dataset Size Guidelines

### How Much Data Do You Need?

```
┌─────────────────────────────────────────────────────────────┐
│                Dataset Size Guidelines                      │
│                                                             │
│  Task Type                    Recommended Size              │
│  ─────────────────────────────────────────────────          │
│  Style/tone adjustment        50-100 examples               │
│  Format compliance            100-200 examples              │
│  Domain adaptation            500-1000 examples             │
│  New task/skill               1000-5000 examples            │
│  Complex behavior change      5000-10000+ examples          │
│                                                             │
│  Note: Quality > Quantity always                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Diminishing Returns

```
Quality
   ↑
   │        ┌────────────────────────
   │       /
   │      /
   │     /
   │    /
   │   /
   │  /
   │ /
   │/
   └─────────────────────────────────→ Dataset Size
         100   500   1K    5K   10K
         
   Most gains in first 500-1000 examples
   Diminishing returns after ~5000
```

---

## Data Cleaning & Validation

### Automated Checks

```python
import json
import re
from collections import Counter

def analyze_dataset(filepath):
    """Analyze a fine-tuning dataset for potential issues"""
    
    stats = {
        'total_examples': 0,
        'avg_turns': 0,
        'avg_user_length': 0,
        'avg_assistant_length': 0,
        'issues': []
    }
    
    turn_counts = []
    user_lengths = []
    assistant_lengths = []
    
    with open(filepath, 'r') as f:
        for i, line in enumerate(f, 1):
            data = json.loads(line)
            messages = data.get('messages', [])
            
            stats['total_examples'] += 1
            turn_counts.append(len(messages))
            
            for msg in messages:
                content = msg.get('content', '')
                
                if msg['role'] == 'user':
                    user_lengths.append(len(content))
                elif msg['role'] == 'assistant':
                    assistant_lengths.append(len(content))
                    
                    # Check for issues
                    if len(content) < 10:
                        stats['issues'].append(f"Line {i}: Very short assistant response")
                    if content.strip().endswith('...'):
                        stats['issues'].append(f"Line {i}: Possibly truncated response")
    
    stats['avg_turns'] = sum(turn_counts) / len(turn_counts)
    stats['avg_user_length'] = sum(user_lengths) / len(user_lengths)
    stats['avg_assistant_length'] = sum(assistant_lengths) / len(assistant_lengths)
    
    return stats

# Usage
stats = analyze_dataset('training_data.jsonl')
print(f"Total examples: {stats['total_examples']}")
print(f"Average turns per conversation: {stats['avg_turns']:.1f}")
print(f"Average user message length: {stats['avg_user_length']:.0f} chars")
print(f"Average assistant response length: {stats['avg_assistant_length']:.0f} chars")
print(f"\nIssues found: {len(stats['issues'])}")
```

### Common Issues to Fix

| Issue | Detection | Solution |
|-------|-----------|----------|
| Truncated responses | Ends with "..." | Complete the response |
| Too short | < 10 characters | Expand or remove |
| Inconsistent format | Regex patterns | Standardize |
| Duplicate data | Hash comparison | Deduplicate |
| PII/sensitive data | Regex/NER | Anonymize or remove |
| Encoding issues | Unicode errors | Fix encoding |

### Deduplication

```python
import hashlib

def deduplicate_dataset(data):
    """Remove duplicate examples based on content hash"""
    seen_hashes = set()
    unique_data = []
    
    for item in data:
        # Create hash of content
        content = json.dumps(item, sort_keys=True)
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_data.append(item)
    
    print(f"Removed {len(data) - len(unique_data)} duplicates")
    return unique_data
```

---

## Generating Synthetic Data

When you don't have enough real data, generate synthetic examples:

### Using an LLM to Generate Training Data

```python
from openai import OpenAI

client = OpenAI()

def generate_training_examples(task_description, num_examples=10):
    """Generate synthetic training examples using GPT-4"""
    
    prompt = f"""Generate {num_examples} diverse training examples for this task:

Task: {task_description}

For each example, provide:
1. A realistic user input
2. An ideal assistant response

Format as JSON array with 'user' and 'assistant' fields.
Make examples diverse in complexity, length, and edge cases."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)

# Generate examples
examples = generate_training_examples(
    "Customer support for a software company handling billing questions"
)
```

### Data Augmentation

```python
def augment_example(example, variations=3):
    """Create variations of a training example"""
    
    prompt = f"""Create {variations} variations of this training example.
Keep the same meaning but vary:
- Phrasing and word choice
- Level of formality
- Question format (direct question, statement, etc.)

Original user message: {example['user']}
Original assistant response: {example['assistant']}

Return as JSON array with 'user' and 'assistant' fields."""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)
```

---

## Dataset Splitting

### Train/Validation Split

```python
import random

def split_dataset(data, val_ratio=0.1, seed=42):
    """Split dataset into training and validation sets"""
    random.seed(seed)
    
    data_copy = data.copy()
    random.shuffle(data_copy)
    
    split_idx = int(len(data_copy) * (1 - val_ratio))
    
    train_data = data_copy[:split_idx]
    val_data = data_copy[split_idx:]
    
    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(val_data)}")
    
    return train_data, val_data

# Usage
train, val = split_dataset(all_data, val_ratio=0.1)
```

---

## Key Takeaways

1. **Format matters** - Match your provider's expected format exactly

2. **Quality > Quantity** - 100 great examples beat 1000 mediocre ones

3. **Diversity is crucial** - Cover the range of expected inputs

4. **Validate thoroughly** - Check for issues before training

5. **Augment carefully** - Synthetic data can help but verify quality

6. **Keep a validation set** - Always hold out data for evaluation

---

## What's Next?

Now that you know how to prepare data, let's learn about **fine-tuning with cloud providers** → [Cloud Fine-tuning](/learn/fine-tuning/cloud-finetuning/cloud-finetuning)

---

## Quick Reference: Data Formats

| Provider | Format | File Type |
|----------|--------|-----------|
| OpenAI | Chat messages | JSONL |
| Anthropic | Messages | JSONL |
| Together AI | Various | JSONL |
| Local (HF) | Alpaca/ShareGPT | JSON/JSONL |
