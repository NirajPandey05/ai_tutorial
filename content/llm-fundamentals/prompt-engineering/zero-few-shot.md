# Zero-Shot, One-Shot, and Few-Shot Learning

## Understanding Learning Modes

When working with LLMs, you can provide varying amounts of examples to guide the model's behavior. This spectrum is crucial for getting the outputs you need.

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Zero-Shot  →  One-Shot  →  Few-Shot  →  Many-Shot            │
│   (0 examples)  (1 example)  (2-5 examples) (5+ examples)       │
│                                                                 │
│   Less specific ←─────────────────────────→ More specific       │
│   More flexible ←─────────────────────────→ More constrained    │
│   Fewer tokens  ←─────────────────────────→ More tokens         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Zero-Shot Learning

The model performs a task with **no examples** - relying entirely on its pre-trained knowledge.

### When It Works Well

✅ Simple, well-defined tasks:
```text
Translate "Good morning" to Spanish.
```

✅ Common formats:
```text
Write a haiku about programming.
```

✅ General knowledge:
```text
Explain photosynthesis in simple terms.
```

### When It Struggles

❌ Custom formats the model hasn't seen:
```text
Convert this data to our proprietary XYZ format.
```

❌ Domain-specific jargon or rules:
```text
Generate a CRD for our internal deployment system.
```

❌ Subjective tasks with specific criteria:
```text
Rate this writing on our company's quality scale.
```

### Zero-Shot Example

```text
Classify the following support ticket into one of these categories:
billing, technical, general inquiry, complaint

Ticket: "I can't log into my account. I've tried resetting my password 
three times but the reset email never arrives."

Category:
```

**Output:** `technical`

---

## One-Shot Learning

Provide **exactly one example** to show the model what you want.

### Template

```text
[Task description]

Example:
Input: [sample input]
Output: [expected output]

Now process:
Input: [actual input]
Output:
```

### One-Shot Example

```text
Extract the product name, price, and availability from product listings.

Example:
Input: "Apple MacBook Pro 14-inch, $1999.99, Ships in 2-3 days"
Output: {"product": "Apple MacBook Pro 14-inch", "price": 1999.99, "available": true}

Now process:
Input: "Sony WH-1000XM5 Headphones, $348.00, Out of stock"
Output:
```

**Output:** `{"product": "Sony WH-1000XM5 Headphones", "price": 348.00, "available": false}`

### When to Use One-Shot

- Establishing a specific output format
- Simple pattern matching tasks
- When token budget is limited
- When the pattern is clear from one example

---

## Few-Shot Learning

Provide **2-5 examples** to demonstrate patterns more robustly.

### Template

```text
[Task description]

Examples:

Input: [example 1 input]
Output: [example 1 output]

Input: [example 2 input]
Output: [example 2 output]

Input: [example 3 input]
Output: [example 3 output]

Now process:
Input: [actual input]
Output:
```

### Few-Shot Example

```text
Determine if a code review comment is constructive or unconstructive.
Constructive comments explain the issue and suggest improvements.
Unconstructive comments are vague, dismissive, or unhelpful.

Examples:

Comment: "This is wrong."
Classification: unconstructive

Comment: "This loop could be optimized using a dictionary lookup instead of nested loops, reducing complexity from O(n²) to O(n)."
Classification: constructive

Comment: "Why would you do it this way?"
Classification: unconstructive

Comment: "Consider adding error handling here - if the API returns a 500, this will crash. You could wrap it in a try/except block."
Classification: constructive

Now process:
Comment: "This function is too long. Consider extracting the validation logic into a separate helper function for better readability and testability."
Classification:
```

**Output:** `constructive`

---

## Choosing the Right Number of Examples

### Decision Framework

```
┌─────────────────────────────────────────────────────────────────┐
│                    How Many Examples?                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Is the task straightforward and common?                       │
│    YES → Try Zero-Shot first                                   │
│    NO  → Continue...                                           │
│                                                                 │
│  Is there a specific output format needed?                     │
│    YES → At least One-Shot                                     │
│    NO  → Zero-Shot may work                                    │
│                                                                 │
│  Are there edge cases or exceptions?                           │
│    YES → Few-Shot (show the edge cases)                        │
│    NO  → One-Shot might suffice                                │
│                                                                 │
│  Is the task subjective or nuanced?                            │
│    YES → Few-Shot (show the nuances)                           │
│    NO  → One-Shot usually works                                │
│                                                                 │
│  Is consistency across multiple inputs critical?               │
│    YES → Few-Shot (more examples = more consistency)           │
│    NO  → Fewer examples are fine                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Comparison Table

| Aspect | Zero-Shot | One-Shot | Few-Shot |
|--------|-----------|----------|----------|
| **Token Usage** | Lowest | Low | Higher |
| **Flexibility** | Highest | High | Lower |
| **Consistency** | Variable | Good | Best |
| **Format Control** | Low | High | Highest |
| **Setup Effort** | None | Minimal | More |

---

## Best Practices for Examples

### 1. Use Diverse Examples

Show different scenarios, not repetitions:

❌ **Bad - Repetitive:**
```text
Input: "I love this product!"
Output: positive

Input: "This is amazing!"
Output: positive

Input: "Great purchase!"
Output: positive
```

✅ **Good - Diverse:**
```text
Input: "I love this product!"
Output: positive

Input: "Completely disappointed. Waste of money."
Output: negative

Input: "It works. Nothing special."
Output: neutral
```

### 2. Include Edge Cases

Show how to handle unusual situations:

```text
Extract dates from text. Use ISO format (YYYY-MM-DD).

Examples:

Input: "Meeting on January 15, 2025"
Output: 2025-01-15

Input: "Due next Friday" 
Output: [relative date - cannot determine exact date without reference]

Input: "Sometime in Q3"
Output: [ambiguous - Q3 could be July-September]

Input: "The report covers 2024-2025"
Output: [date range: 2024-01-01 to 2025-12-31]
```

### 3. Order Matters

Place examples strategically:

- **Most typical cases first** - establishes the baseline
- **Edge cases in the middle** - shows exceptions
- **Most similar to actual input last** - freshest in context

### 4. Match Example Complexity

Examples should match the complexity of actual inputs:

❌ **Bad - Mismatched:**
```text
Example:
Input: "Hi"
Output: greeting

Actual:
Input: "Dear Mr. Johnson, I hope this email finds you well. I wanted to follow up on our previous discussion regarding the Q3 budget allocation..."
```

✅ **Good - Matched:**
```text
Example:
Input: "Dear Mr. Johnson, I hope this message finds you well. Following up on our conversation about the project timeline..."
Output: professional_followup
```

---

## Zero-Shot Chain of Thought (Zero-Shot CoT)

A powerful technique combining zero-shot with reasoning:

### Standard Zero-Shot
```text
What is 23 × 17?
```

### Zero-Shot CoT
```text
What is 23 × 17?

Let's think step by step.
```

Adding "Let's think step by step" dramatically improves reasoning performance without needing examples.

### Complex Example

```text
A store offers a 20% discount on all items. After the discount, 
there's an additional 10% off for members. If a jacket originally 
costs $150, how much does a member pay?

Let's work through this step by step.
```

**Output:**
```
Step 1: Calculate the 20% discount
$150 × 0.20 = $30 discount
$150 - $30 = $120 after first discount

Step 2: Calculate the additional 10% member discount
$120 × 0.10 = $12 additional discount
$120 - $12 = $108 final price

A member pays $108 for the jacket.
```

---

## Few-Shot with Chain of Thought

Combine examples with explicit reasoning:

```text
Determine if the argument is logically valid.

Example 1:
Argument: "All cats are mammals. Whiskers is a cat. Therefore, Whiskers is a mammal."
Reasoning: This follows the valid syllogism pattern: All A are B, X is A, therefore X is B.
Valid: Yes

Example 2:
Argument: "Some birds can fly. Penguins are birds. Therefore, penguins can fly."
Reasoning: The premise only states "some" birds can fly, not all. We cannot conclude that any specific bird can fly from this premise.
Valid: No

Now analyze:
Argument: "If it rains, the ground gets wet. The ground is wet. Therefore, it rained."
Reasoning:
```

---

## Real-World Application Examples

### Sentiment Analysis (Few-Shot)

```text
Analyze customer feedback sentiment. Include confidence level.

Feedback: "The app crashes every time I try to upload photos. So frustrating!"
Sentiment: negative
Confidence: high
Reason: explicit frustration, technical issue

Feedback: "Pretty good overall, though the UI could use some polish."
Sentiment: positive
Confidence: medium
Reason: "pretty good" indicates positive, with minor criticism

Feedback: "It does what it's supposed to do."
Sentiment: neutral
Confidence: medium
Reason: functional acknowledgment without positive or negative emotion

Now analyze:
Feedback: "I've been using this for 6 months and honestly can't imagine going back to the old way of doing things."
```

### Code Review (One-Shot)

```text
Review code and provide feedback in this format.

Example:
```python
def add(a, b):
    return a + b
```
Review:
- ✅ Function is simple and focused
- ⚠️ Missing type hints
- ⚠️ Missing docstring
- 💡 Suggestion: `def add(a: int, b: int) -> int:`

Now review:
```python
def process_users(users):
    result = []
    for user in users:
        if user['status'] == 'active':
            result.append(user['email'])
    return result
```
```

---

## Key Takeaways

1. **Start with zero-shot** - Only add examples if needed
2. **Add "think step by step"** - Free reasoning boost for complex problems
3. **Use diverse examples** - Cover different scenarios and edge cases
4. **Match complexity** - Examples should resemble actual inputs
5. **Quality over quantity** - 3 good examples beat 10 poor ones
6. **Test and iterate** - Adjust based on actual model performance

## Next Steps

Ready to practice? Head to the **Prompt Playground Lab** to experiment with different prompting strategies in real-time.
