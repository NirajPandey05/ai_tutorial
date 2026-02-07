# Local Fine-tuning with LoRA and QLoRA

Fine-tune models on your own hardware using parameter-efficient methods. This lesson covers practical implementation with Hugging Face libraries.

## Learning Objectives

By the end of this lesson, you will:
- Set up a local fine-tuning environment
- Fine-tune models using LoRA with PEFT
- Use QLoRA for consumer GPU training
- Merge and deploy fine-tuned adapters

---

## Environment Setup

### Required Libraries

```bash
# Core libraries
pip install torch transformers datasets
pip install accelerate bitsandbytes
pip install peft trl

# For monitoring
pip install wandb tensorboard
```

### GPU Requirements

| Method | 7B Model | 13B Model | 70B Model |
|--------|----------|-----------|-----------|
| **Full Fine-tune** | ~60GB | ~100GB | ~500GB |
| **LoRA** | ~16GB | ~26GB | ~80GB |
| **QLoRA** | ~6GB | ~10GB | ~40GB |

### Check Your Setup

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

---

## Loading Models with Quantization

### 4-bit Quantization (QLoRA)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Quantization config for QLoRA
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # NormalFloat4 quantization
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # Double quantization
)

# Load model with quantization
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
```

### 8-bit Quantization

```python
# For 8-bit (less memory savings, better quality)
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)
```

---

## Setting Up LoRA

### LoRA Configuration

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Prepare model for training
model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    r=16,                       # Rank (higher = more capacity, more VRAM)
    lora_alpha=32,              # Scaling factor (usually 2x rank)
    lora_dropout=0.05,          # Dropout for regularization
    bias="none",                # Don't train bias terms
    task_type="CAUSAL_LM",      # Task type
    target_modules=[            # Which layers to adapt
        "q_proj", "k_proj", "v_proj",  # Attention layers
        "o_proj",
        "gate_proj", "up_proj", "down_proj"  # MLP layers (optional)
    ],
)

# Create PEFT model
model = get_peft_model(model, lora_config)

# Check trainable parameters
model.print_trainable_parameters()
# Output: trainable params: 13,631,488 || all params: 8,043,317,248 || trainable%: 0.17%
```

### Understanding Target Modules

| Module | Description | When to Include |
|--------|-------------|-----------------|
| **q_proj, k_proj, v_proj** | Attention Q, K, V projections | Always |
| **o_proj** | Attention output projection | Usually |
| **gate_proj, up_proj, down_proj** | MLP layers | For more adaptation |
| **embed_tokens** | Input embeddings | For new vocab |
| **lm_head** | Output head | For new vocab |

### Finding Target Modules

```python
# List all linear layers in the model
def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return list(lora_module_names)

target_modules = find_all_linear_names(model)
print(f"Available modules: {target_modules}")
```

---

## Preparing Training Data

### Dataset Loading

```python
from datasets import load_dataset, Dataset

# Load from Hugging Face
dataset = load_dataset("your-username/your-dataset")

# Or from local JSONL
def load_jsonl(filepath):
    import json
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)

dataset = load_jsonl("training_data.jsonl")
```

### Formatting for Training

```python
def format_instruction(sample):
    """Format data for instruction fine-tuning"""
    
    # For chat format
    if 'messages' in sample:
        text = ""
        for msg in sample['messages']:
            if msg['role'] == 'system':
                text += f"<|system|>\n{msg['content']}</s>\n"
            elif msg['role'] == 'user':
                text += f"<|user|>\n{msg['content']}</s>\n"
            elif msg['role'] == 'assistant':
                text += f"<|assistant|>\n{msg['content']}</s>\n"
        return {"text": text}
    
    # For instruction format
    elif 'instruction' in sample:
        text = f"""### Instruction:
{sample['instruction']}

### Input:
{sample.get('input', '')}

### Response:
{sample['output']}"""
        return {"text": text}

# Apply formatting
formatted_dataset = dataset.map(format_instruction)
```

### Using Chat Templates

```python
def format_with_chat_template(sample):
    """Use model's built-in chat template"""
    messages = sample['messages']
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return {"text": text}

formatted_dataset = dataset.map(format_with_chat_template)
```

---

## Training with SFTTrainer

### Basic Training Setup

```python
from trl import SFTTrainer
from transformers import TrainingArguments

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch = 4 * 4 = 16
    
    # Optimization
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    
    # Memory optimization
    fp16=True,  # or bf16=True for newer GPUs
    optim="paged_adamw_8bit",  # Memory-efficient optimizer
    gradient_checkpointing=True,
    
    # Logging
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    
    # Misc
    report_to="wandb",  # or "tensorboard"
    push_to_hub=False,
)

# Create trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset["train"],
    eval_dataset=formatted_dataset.get("validation"),
    peft_config=lora_config,
    tokenizer=tokenizer,
    args=training_args,
    dataset_text_field="text",
    max_seq_length=2048,
    packing=False,  # Set True to pack short examples together
)

# Train!
trainer.train()
```

### Memory Optimization Tips

```python
# For low-memory GPUs, add these to TrainingArguments:
training_args = TrainingArguments(
    # ... other args ...
    
    # Gradient checkpointing (trade compute for memory)
    gradient_checkpointing=True,
    
    # Use 8-bit optimizer
    optim="paged_adamw_8bit",
    
    # Smaller batch with more accumulation
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    
    # Mixed precision
    fp16=True,
    
    # Limit sequence length
    max_seq_length=1024,
)
```

---

## Saving and Loading Adapters

### Save Adapter

```python
# Save LoRA adapter (small file, ~10-50MB)
trainer.save_model("./my-lora-adapter")

# Or just the adapter
model.save_pretrained("./my-lora-adapter")
tokenizer.save_pretrained("./my-lora-adapter")
```

### Load Adapter

```python
from peft import PeftModel, PeftConfig

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)

# Load adapter on top
model = PeftModel.from_pretrained(base_model, "./my-lora-adapter")

# For inference
model.eval()
```

### Merge Adapter into Base Model

```python
# Merge LoRA weights into base model (creates full-size model)
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("./merged-model")
tokenizer.save_pretrained("./merged-model")

# Now you have a regular model without adapter overhead
```

---

## Complete Training Script

```python
"""
Complete LoRA/QLoRA Fine-tuning Script
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

# Configuration
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DATASET_NAME = "your-dataset"
OUTPUT_DIR = "./fine-tuned-model"

# Quantization config (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load model and tokenizer
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Prepare for training
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load dataset
print("Loading dataset...")
dataset = load_dataset(DATASET_NAME)

def format_data(sample):
    """Format for chat"""
    text = tokenizer.apply_chat_template(
        sample['messages'],
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

dataset = dataset.map(format_data)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    fp16=True,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch" if "validation" in dataset else "no",
    report_to="wandb",
)

# Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset.get("validation"),
    tokenizer=tokenizer,
    args=training_args,
    dataset_text_field="text",
    max_seq_length=2048,
)

# Train
print("Starting training...")
trainer.train()

# Save
print("Saving model...")
trainer.save_model(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")
```

---

## Using Your Fine-tuned Model

### Inference with Adapter

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Load with quantization for inference
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)

# Load adapter
model = PeftModel.from_pretrained(base_model, "./fine-tuned-model")
tokenizer = AutoTokenizer.from_pretrained("./fine-tuned-model")

# Generate
def generate(prompt, max_new_tokens=256):
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("assistant")[-1].strip()

# Test
response = generate("What are your capabilities?")
print(response)
```

### Serve with vLLM

```python
# After merging adapter into base model:
from vllm import LLM, SamplingParams

llm = LLM(
    model="./merged-model",
    quantization="awq",  # Optional further quantization
)

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=256,
)

outputs = llm.generate(["Your prompt here"], sampling_params)
print(outputs[0].outputs[0].text)
```

---

## Troubleshooting

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| CUDA OOM | Not enough VRAM | Reduce batch size, use gradient checkpointing |
| "bitsandbytes not found" | Missing library | `pip install bitsandbytes` |
| "CUDA not available" | GPU not detected | Check CUDA installation |
| "NaN loss" | Learning rate too high | Reduce learning rate |
| "Model not improving" | Too few epochs or bad data | More epochs, check data quality |

### Memory Estimation

```python
def estimate_memory(model_params_billion, method="qlora"):
    """Estimate VRAM needed for training"""
    
    if method == "full":
        # Full precision + gradients + optimizer states
        return model_params_billion * 4 * 4  # ~16 bytes per param
    elif method == "lora":
        # Model in fp16 + small trainable params
        return model_params_billion * 2 + 0.5
    elif method == "qlora":
        # Model in 4-bit + small trainable params
        return model_params_billion * 0.5 + 0.5

print(f"7B model:")
print(f"  Full: {estimate_memory(7, 'full'):.1f} GB")
print(f"  LoRA: {estimate_memory(7, 'lora'):.1f} GB")
print(f"  QLoRA: {estimate_memory(7, 'qlora'):.1f} GB")
```

---

## Key Takeaways

1. **QLoRA enables consumer GPU training** - Fine-tune 7B models on 8GB VRAM

2. **Choose target modules wisely** - Start with attention, add MLP if needed

3. **Rank (r) affects capacity** - Higher rank = more learning, more memory

4. **Always use gradient checkpointing** for memory efficiency

5. **Merge adapters for deployment** - Eliminates runtime overhead

6. **Monitor training metrics** - Watch for overfitting and convergence

---

## What's Next?

Learn how to **evaluate and improve** your fine-tuned models → [Fine-tuning Best Practices](/learn/fine-tuning/finetuning-practices/best-practices)

---

## Quick Reference

| Setting | Low Memory | Balanced | Maximum Quality |
|---------|------------|----------|-----------------|
| **Quantization** | 4-bit | 4-bit | 8-bit or fp16 |
| **LoRA rank** | 8 | 16 | 64 |
| **Target modules** | Attention only | + gate_proj | All linear |
| **Batch size** | 1 | 4 | 8+ |
| **Gradient checkpointing** | Yes | Yes | Optional |
