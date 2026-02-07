# Temperature and Sampling

## Understanding Temperature

**Temperature** is the most important parameter for controlling LLM output. It determines how "creative" or "deterministic" the model's responses will be.

```
Temperature Scale:
0.0 ─────────────────────────────────────────────── 2.0
    │                    │                         │
 Deterministic      Balanced               Creative/Random
 (Focused)         (Default ~0.7)          (Unpredictable)
```

## How Temperature Works

When an LLM generates text, it predicts probabilities for the next token. Temperature affects how these probabilities influence the selection:

### Token Probability Example

For the prompt "The best programming language is", the model might assign:

| Token | Raw Probability | Temp=0.0 | Temp=0.7 | Temp=1.5 |
|-------|----------------|----------|----------|----------|
| Python | 40% | 99% | 52% | 35% |
| JavaScript | 25% | 1% | 24% | 25% |
| depends | 15% | 0% | 12% | 18% |
| Rust | 10% | 0% | 7% | 12% |
| COBOL | 1% | 0% | 0.5% | 5% |

### The Math Behind It

Temperature modifies the softmax function:

$$P(token_i) = \frac{e^{logit_i / T}}{\sum_j e^{logit_j / T}}$$

Where $T$ is the temperature:
- **T → 0**: Highest probability token always wins (deterministic)
- **T = 1**: Natural probability distribution
- **T > 1**: Flattened distribution (more randomness)

## Temperature Values Explained

### Temperature 0.0 - Deterministic

```python
# The model always picks the highest probability token
temperature = 0.0
```

**Characteristics:**
- Same input → Same output (mostly)
- Safest, most predictable
- Can feel repetitive or "robotic"

**Best for:**
- Code generation
- Factual Q&A
- Data extraction
- Classification tasks
- Production systems requiring consistency

**Example Output:**
```
Prompt: "Write a function to add two numbers"

# At temp=0, you'll get essentially the same response every time:
def add(a, b):
    return a + b
```

---

### Temperature 0.3-0.5 - Slightly Creative

```python
temperature = 0.4
```

**Characteristics:**
- Mostly consistent with slight variation
- Good balance for technical content
- Occasionally surprises you

**Best for:**
- Technical documentation
- Code with comments
- Structured creative writing
- Email drafts

---

### Temperature 0.7-0.8 - Balanced (Default)

```python
temperature = 0.7  # Most common default
```

**Characteristics:**
- Good mix of coherence and creativity
- Natural-feeling responses
- Suitable for most general tasks

**Best for:**
- Conversational AI
- General writing assistance
- Explanations and tutorials
- Most chatbot applications

---

### Temperature 1.0+ - Creative

```python
temperature = 1.2
```

**Characteristics:**
- More diverse outputs
- Higher chance of unusual phrasings
- Less predictable

**Best for:**
- Creative writing
- Brainstorming
- Poetry and fiction
- Generating alternatives

---

### Temperature 1.5-2.0 - Very Random

```python
temperature = 1.8
```

**Characteristics:**
- Highly unpredictable
- May produce incoherent text
- Useful for extreme creativity

**Best for:**
- Experimental generation
- Finding unusual ideas
- Breaking out of patterns

⚠️ **Warning:** High temperatures can produce nonsensical output.

## Practical Examples

### Example 1: Code Generation

**Low Temperature (0.0)** - Recommended
```python
# Prompt: "Write a Python function to reverse a string"

# Response is consistent and correct:
def reverse_string(s):
    return s[::-1]
```

**Higher Temperature (1.2)** - Not recommended for code
```python
# Might produce variations like:
def flip_text(text_input):
    """Reverses the wonderful string!"""
    backwards = ""
    for char in text_input:
        backwards = char + backwards
    return backwards  # Magic! ✨
```

---

### Example 2: Creative Writing

**Low Temperature (0.2)**
```
Prompt: "Write an opening line for a mystery novel"

Response: "Detective Sarah Mitchell arrived at the crime scene 
just as the sun was setting."
```

**High Temperature (1.3)**
```
Prompt: "Write an opening line for a mystery novel"

Response: "The grandfather clock hadn't chimed in seventeen years—
until tonight, when it screamed."
```

---

### Example 3: Brainstorming

**Low Temperature (0.0)**
```
Prompt: "Give me 5 startup ideas"

1. Food delivery app
2. Online learning platform
3. Fitness tracking app
4. Social media management tool
5. E-commerce marketplace
```

**High Temperature (1.4)**
```
Prompt: "Give me 5 startup ideas"

1. AI sommelier that pairs wine with your Spotify playlist
2. Micro-nap pods that optimize REM cycles in 12-minute sessions
3. Blockchain-verified authentic restaurant reviews
4. Plant emotions translator using biometric sensors
5. Subscription service for handwritten letters from strangers
```

## Top-P (Nucleus Sampling)

Often used alongside temperature, **top_p** limits token selection to a cumulative probability threshold.

### How Top-P Works

Instead of considering all possible tokens, top_p only considers tokens whose cumulative probability reaches the threshold:

```
If top_p = 0.9:

Token probabilities (sorted):
  Python:     35%  ─┐
  JavaScript: 25%   │
  depends:    15%   ├─ These sum to 90%, so only these are considered
  Rust:       10%   │
  TypeScript: 5%   ─┘
  Go:         4%    ─ Excluded (would exceed 90%)
  Java:       3%    
  C++:        3%    
```

### Top-P Values

| Value | Effect |
|-------|--------|
| 0.1 | Very focused (only top ~10% probability mass) |
| 0.5 | Moderate diversity |
| 0.9 | Default - good balance |
| 1.0 | Consider all tokens (disabled) |

### Temperature vs Top-P

```
┌────────────────────────────────────────────────────────────────┐
│                                                                │
│   Temperature: Changes the probability distribution shape      │
│   Top-P: Cuts off the long tail of low-probability tokens     │
│                                                                │
│   Low Temperature + Top-P:                                    │
│   ▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░ → ▓▓▓▓▓▓▓▓▓▓▓                    │
│   (Peaked)                    (Tail cut)                       │
│                                                                │
│   High Temperature + Top-P:                                   │
│   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░ → ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓                │
│   (Flattened)                 (Some tail cut)                  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Using Both Together

```python
# Recommended combinations:

# Deterministic output
temperature = 0.0
top_p = 1.0  # Doesn't matter at temp=0

# Conservative but natural
temperature = 0.5
top_p = 0.9

# Balanced (typical default)
temperature = 0.7
top_p = 1.0

# Creative but coherent
temperature = 1.0
top_p = 0.95

# Very creative
temperature = 1.2
top_p = 0.8  # Cut tail to prevent nonsense
```

> **Note:** OpenAI recommends adjusting either temperature OR top_p, not both. Google and Anthropic have similar guidance.

## Temperature by Provider

| Provider | Default | Range | Notes |
|----------|---------|-------|-------|
| OpenAI | 1.0 | 0-2 | Also supports top_p |
| Anthropic | 1.0 | 0-1 | Values >1 not recommended |
| Google | 0.7 | 0-2 | Also has top_k parameter |
| xAI | 0.7 | 0-2 | Similar to OpenAI |

## Best Practices

### 1. Match Temperature to Task

| Task | Recommended Temperature |
|------|------------------------|
| Code generation | 0.0 - 0.3 |
| Data extraction | 0.0 |
| Factual Q&A | 0.0 - 0.3 |
| Email/messages | 0.5 - 0.7 |
| General chat | 0.7 - 0.9 |
| Creative writing | 0.9 - 1.3 |
| Brainstorming | 1.0 - 1.5 |

### 2. Test Multiple Values

```python
# Generate same prompt at different temperatures to compare:
temperatures = [0.0, 0.5, 0.7, 1.0, 1.3]

for temp in temperatures:
    response = model.generate(prompt, temperature=temp)
    print(f"Temp {temp}: {response}")
```

### 3. Consider Reproducibility

For production systems:
```python
# Use low temperature for consistent results
temperature = 0.0

# Or set a seed for reproducibility (if supported)
seed = 42
```

### 4. Avoid Extremes Without Purpose

High temperatures (>1.5) often produce:
- Grammatical errors
- Logical inconsistencies  
- Random tangents

## Key Takeaways

1. **Temperature controls randomness** - Lower = focused, higher = creative
2. **Match to your task** - Code needs low temp, stories need higher
3. **Top-P complements temperature** - Use one or the other primarily
4. **Test and iterate** - The "right" value depends on your specific use case
5. **Production = low temperature** - Consistency matters in real applications

## Next Steps

Now that you understand temperature and sampling, let's explore **Token Limits and Max Tokens** to control response length.
