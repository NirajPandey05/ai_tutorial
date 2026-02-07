# Prompt Patterns & Techniques

## Overview

Prompt patterns are reusable templates that solve common problems when working with LLMs. Just as software design patterns provide proven solutions to recurring challenges, prompt patterns give you reliable approaches to different types of tasks.

## Pattern 1: Persona Pattern

Assign a specific role or expertise to the model.

### Template
```text
Act as a [ROLE] with expertise in [DOMAIN].
Your task is to [SPECIFIC TASK].
[Additional constraints or guidelines]
```

### Example
```text
Act as a senior security engineer with expertise in web application security.
Your task is to review code for vulnerabilities.
Focus on OWASP Top 10 issues and provide remediation steps.
```

### When to Use
- Technical explanations requiring domain expertise
- Creative writing with specific voice/style
- Simulating expert consultation

---

## Pattern 2: Template Pattern

Provide a structured format for the output.

### Template
```text
[Instructions]

Use this exact format for your response:

## [Section 1]
[What goes here]

## [Section 2]
[What goes here]

## [Section 3]
[What goes here]
```

### Example
```text
Analyze this startup idea and provide feedback.

Use this exact format:

## Summary
One paragraph overview of the idea.

## Strengths (3-5 bullet points)
- Strength 1
- Strength 2

## Weaknesses (3-5 bullet points)
- Weakness 1
- Weakness 2

## Recommendation
One paragraph with your verdict.
```

### When to Use
- Consistent output structure needed
- Generating reports or documentation
- Comparing multiple items with same criteria

---

## Pattern 3: Chain of Thought (CoT)

Encourage step-by-step reasoning for complex problems.

### Template
```text
[Problem statement]

Think through this step by step:
1. First, identify [aspect 1]
2. Then, consider [aspect 2]
3. Next, evaluate [aspect 3]
4. Finally, conclude with [final answer]

Show your reasoning at each step.
```

### Example
```text
A company has 150 employees. 40% work in engineering, 25% in sales,
and the rest in operations. If 30% of engineers and 50% of sales staff
work remotely, how many people work in the office?

Think through this step by step:
1. First, calculate the number in each department
2. Then, calculate remote workers per department
3. Calculate total remote workers
4. Subtract from total to get office workers

Show your reasoning at each step.
```

### When to Use
- Math and logic problems
- Multi-step reasoning tasks
- Debugging complex issues
- When you need to verify the reasoning

---

## Pattern 4: Few-Shot Learning

Provide examples to demonstrate the desired behavior.

### Template
```text
[Task description]

Examples:

Input: [Example input 1]
Output: [Expected output 1]

Input: [Example input 2]
Output: [Expected output 2]

Input: [Example input 3]
Output: [Expected output 3]

Now process this:
Input: [Actual input]
Output:
```

### Example
```text
Classify the sentiment of customer reviews as positive, negative, or neutral.

Examples:

Input: "This product exceeded my expectations! Highly recommend."
Output: positive

Input: "The item arrived damaged and customer service was unhelpful."
Output: negative

Input: "The product works as described. Nothing special."
Output: neutral

Now process this:
Input: "Decent quality for the price, but shipping took longer than expected."
Output:
```

### When to Use
- Classification tasks
- Format transformation
- Style matching
- When zero-shot doesn't work well

---

## Pattern 5: Constraint Pattern

Set explicit boundaries on the output.

### Template
```text
[Task description]

Constraints:
- Must be [length constraint]
- Must include [required elements]
- Must NOT include [forbidden elements]
- Tone should be [tone description]
```

### Example
```text
Write a product description for a wireless Bluetooth speaker.

Constraints:
- Must be exactly 3 paragraphs
- Must include: battery life, sound quality, portability
- Must NOT include: competitor comparisons, pricing
- Tone should be enthusiastic but not hyperbolic
```

### When to Use
- Content that must fit specific guidelines
- Avoiding unwanted content
- Maintaining consistency across outputs

---

## Pattern 6: Decomposition Pattern

Break complex tasks into manageable subtasks.

### Template
```text
I need to [complex goal].

Let's break this down:

Step 1: [First subtask]
Step 2: [Second subtask]  
Step 3: [Third subtask]
...

Start with Step 1.
```

### Example
```text
I need to create a complete REST API design for a task management application.

Let's break this down:

Step 1: Define the data models (Task, User, Project)
Step 2: Design the endpoints for each resource
Step 3: Specify authentication requirements
Step 4: Document request/response formats
Step 5: Add error handling specifications

Start with Step 1: Define the data models with their fields and relationships.
```

### When to Use
- Large, complex tasks
- When output might be truncated
- Maintaining quality across long generations

---

## Pattern 7: Critique Pattern

Have the model review and improve its own output.

### Template
```text
[Initial request]

After generating your response:
1. Review it for [criteria]
2. Identify any weaknesses
3. Provide an improved version
```

### Example
```text
Write a function that validates email addresses.

After generating your response:
1. Review it for edge cases and security issues
2. Identify any patterns it might incorrectly accept or reject
3. Provide an improved version addressing these issues
```

### When to Use
- Code generation requiring thoroughness
- Creative content needing refinement
- When quality matters more than speed

---

## Pattern 8: Audience Pattern

Tailor the response to a specific audience.

### Template
```text
Explain [topic] for [specific audience].

Consider:
- Their technical level: [beginner/intermediate/expert]
- Their background: [relevant context]
- Their goal: [what they want to achieve]
```

### Example
```text
Explain how machine learning models make predictions for a marketing manager.

Consider:
- Their technical level: beginner, non-technical
- Their background: understands business metrics and customer data
- Their goal: deciding whether to invest in ML for customer segmentation
```

### When to Use
- Technical communication to non-experts
- Documentation for different skill levels
- Training materials

---

## Pattern 9: Socratic Pattern

Engage in dialogue rather than direct answers.

### Template
```text
I'm learning about [topic]. Instead of giving me the answer directly:
1. Ask me guiding questions to help me discover the answer
2. Provide hints when I'm stuck
3. Confirm when I'm on the right track
```

### Example
```text
I'm learning about recursion in programming. Instead of giving me the answer directly:
1. Ask me guiding questions to help me understand the concept
2. Provide hints when I'm stuck
3. Confirm when I'm on the right track

Start by asking me what I already know about functions calling themselves.
```

### When to Use
- Educational contexts
- Self-directed learning
- Deeper understanding of concepts

---

## Pattern 10: Comparison Pattern

Analyze multiple options systematically.

### Template
```text
Compare [Option A] vs [Option B] for [use case].

Analyze:
- [Criterion 1]
- [Criterion 2]
- [Criterion 3]

Format as a comparison table, then provide a recommendation based on [specific need].
```

### Example
```text
Compare PostgreSQL vs MongoDB for a social media application backend.

Analyze:
- Data structure flexibility
- Query complexity support
- Horizontal scaling capabilities
- Consistency guarantees

Format as a comparison table, then provide a recommendation based on a startup 
expecting rapid growth and semi-structured user content.
```

### When to Use
- Technology decisions
- Product comparisons
- Evaluating trade-offs

---

## Combining Patterns

The real power comes from combining patterns:

```text
[PERSONA]: Act as a senior backend architect.

[DECOMPOSITION]: I need to design a microservices architecture for an e-commerce platform.
Break this into logical steps.

[TEMPLATE]: For each service, document:
## Service Name
- **Responsibility**: 
- **Endpoints**:
- **Data Store**:
- **Dependencies**:

[CONSTRAINTS]:
- Maximum 8 services
- Must include: user, order, inventory, payment services
- Consider eventual consistency

[CHAIN OF THOUGHT]: Explain your reasoning for service boundaries.

Start with identifying the core domain boundaries.
```

## Pattern Selection Guide

| Task Type | Recommended Patterns |
|-----------|---------------------|
| Code generation | Persona + Constraint + Critique |
| Data analysis | Template + Chain of Thought |
| Creative writing | Persona + Constraint + Audience |
| Problem solving | Decomposition + Chain of Thought |
| Learning/Teaching | Socratic + Audience |
| Decision making | Comparison + Chain of Thought |

## Key Takeaways

1. **Patterns are composable** - Combine them for complex tasks
2. **Match pattern to task** - Not every pattern fits every situation
3. **Iterate on patterns** - Adjust based on model responses
4. **Document what works** - Build your own pattern library

## Next Steps

Now let's explore **Zero-Shot and Few-Shot Learning** in depth to understand when and how to use examples effectively.
