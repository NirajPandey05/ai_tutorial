# RAG vs Fine-tuning: Choosing the Right Approach

Two powerful ways to customize LLM behavior—but they solve different problems.

## Quick Comparison

```yaml
rag:
  what_it_does: "Gives the model access to external knowledge"
  best_for: "Facts, documentation, up-to-date information"
  knowledge_update: "Instant - just update your documents"
  cost: "$ - Pay per query, no training costs"

fine_tuning:
  what_it_does: "Changes how the model behaves and responds"
  best_for: "Style, tone, format, domain-specific patterns"
  knowledge_update: "Requires retraining"
  cost: "$$$ - Training + inference costs"
```

## The Decision Framework

```
┌─────────────────────────────────────────────────────────────────┐
│                    What are you trying to do?                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  "I need the model to KNOW specific facts"                      │
│       └──► Use RAG                                               │
│                                                                  │
│  "I need the model to BEHAVE in a specific way"                 │
│       └──► Consider Fine-tuning                                  │
│                                                                  │
│  "I need both"                                                   │
│       └──► Use RAG + Fine-tuning together                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## When to Use RAG

### Perfect RAG Use Cases

```python
rag_ideal_scenarios = {
    "knowledge_base_qa": {
        "example": "Answer questions about company documentation",
        "why_rag": "Knowledge changes frequently, need citations"
    },
    "customer_support": {
        "example": "Chatbot that answers product questions",
        "why_rag": "Product info updates constantly"
    },
    "research_assistant": {
        "example": "Search and synthesize from papers",
        "why_rag": "New papers published daily"
    },
    "legal_compliance": {
        "example": "Query regulations and policies",
        "why_rag": "Need exact citations for compliance"
    },
    "code_documentation": {
        "example": "Answer questions about codebase",
        "why_rag": "Code changes with every commit"
    }
}
```

### RAG Advantages

```yaml
advantages:
  instant_updates:
    description: "Add new document, immediately searchable"
    example: "New policy uploaded at 9am, answering questions by 9:01am"
  
  verifiable_answers:
    description: "Can cite exact sources"
    example: "According to HR_Policy.pdf page 12..."
  
  no_training_needed:
    description: "Works with any base LLM"
    example: "Switch from GPT-4 to Claude instantly"
  
  data_privacy:
    description: "Documents stay in your infrastructure"
    example: "Sensitive data never sent to training servers"
  
  cost_effective:
    description: "Only pay for queries made"
    example: "100 queries/day costs ~$1-5"
```

## When to Use Fine-tuning

### Perfect Fine-tuning Use Cases

```python
finetuning_ideal_scenarios = {
    "consistent_style": {
        "example": "Always respond in brand voice",
        "why_finetune": "Style is about behavior, not facts"
    },
    "specific_format": {
        "example": "Output structured JSON for API",
        "why_finetune": "Format patterns learned through training"
    },
    "domain_language": {
        "example": "Use medical/legal terminology correctly",
        "why_finetune": "Vocabulary and phrasing patterns"
    },
    "reasoning_patterns": {
        "example": "Follow specific diagnostic workflow",
        "why_finetune": "Multi-step reasoning procedures"
    },
    "reduce_tokens": {
        "example": "Shorter prompts, same behavior",
        "why_finetune": "Behavior baked in, no examples needed"
    }
}
```

### Fine-tuning Advantages

```yaml
advantages:
  consistent_behavior:
    description: "Reliable patterns without long prompts"
    example: "Always outputs valid JSON without reminding"
  
  reduced_latency:
    description: "No retrieval step needed"
    example: "Direct response without database lookup"
  
  lower_per_query_cost:
    description: "Shorter prompts after training"
    example: "50% fewer tokens per request"
  
  specialized_reasoning:
    description: "Domain-specific thought patterns"
    example: "Medical diagnosis following protocol"
```

## Side-by-Side Comparison

| Aspect | RAG | Fine-tuning |
|--------|-----|-------------|
| **Setup Cost** | Low ($) | High ($$$) |
| **Setup Time** | Hours | Days to weeks |
| **Knowledge Update** | Instant | Requires retraining |
| **Citation Support** | Native | Not available |
| **Latency** | Higher (retrieval) | Lower |
| **Per-Query Cost** | Medium | Lower |
| **Data Requirements** | Documents | Training examples |
| **Hallucination Risk** | Lower (grounded) | Higher |
| **Specialization** | Knowledge | Behavior |

## Practical Examples

### Example 1: Company FAQ Bot

```python
# RAG is better here
# Why: FAQ changes frequently, need citations

# RAG approach
class FAQBot:
    def __init__(self):
        self.rag = RAGSystem()
        self.rag.index_documents(load_faq_documents())
    
    def answer(self, question: str) -> dict:
        result = self.rag.query(question)
        return {
            "answer": result["answer"],
            "source": result["sources"][0]["document"],
            "confidence": result["score"]
        }

# Fine-tuning would require:
# - Retraining every time FAQ updates
# - No way to cite which FAQ entry was used
# - Risk of outdated answers after changes
```

### Example 2: JSON API Response Generator

```python
# Fine-tuning is better here
# Why: Consistent format is behavioral, not factual

# Without fine-tuning (verbose prompting needed)
prompt = """You are an API that outputs JSON only.
Never include markdown formatting.
Always use this exact structure:
{
  "status": "success" or "error",
  "data": { ... },
  "timestamp": "ISO8601"
}

User request: {request}
"""

# With fine-tuning (simple prompt)
prompt = "{request}"  # Model learned the format
```

### Example 3: Medical Assistant

```python
# Use BOTH - RAG + Fine-tuning
# Why: Need current medical knowledge AND proper clinical reasoning

class MedicalAssistant:
    def __init__(self):
        # Fine-tuned model for clinical reasoning patterns
        self.model = "ft:gpt-4o:medical-org:clinical-v2"
        
        # RAG for current medical literature
        self.rag = RAGSystem(collection="medical_literature")
    
    def assess(self, symptoms: str) -> dict:
        # Retrieve relevant medical information
        context = self.rag.retrieve(symptoms)
        
        # Use fine-tuned model for proper clinical reasoning
        response = self.client.chat.completions.create(
            model=self.model,  # Fine-tuned for clinical patterns
            messages=[{
                "role": "user",
                "content": f"Literature:\n{context}\n\nPatient symptoms: {symptoms}"
            }]
        )
        
        return self.parse_clinical_assessment(response)
```

## Decision Tree

```
START: What problem are you solving?
│
├─► "Model doesn't know X information"
│   │
│   ├─► Information changes frequently? ──► RAG
│   ├─► Need to cite sources? ──► RAG
│   ├─► Information is static domain knowledge? ──► Consider fine-tuning
│   └─► Both? ──► RAG (safer, easier to update)
│
├─► "Model doesn't respond the way I want"
│   │
│   ├─► Need specific format/structure? ──► Fine-tuning
│   ├─► Need specific tone/style? ──► Fine-tuning
│   ├─► Need domain terminology? ──► Fine-tuning
│   └─► Can solve with better prompts? ──► Try prompting first
│
└─► "Both knowledge AND behavior issues"
    │
    └─► Use RAG + Fine-tuning together
```

## Cost Analysis

```python
# RAG costs (example: 1000 queries/day)
rag_costs = {
    "embedding_generation": {
        "one_time": "10,000 docs × $0.00002 = $0.20",
        "per_query": "1 query × $0.00002 = negligible"
    },
    "vector_db": {
        "chromadb_local": "$0 (self-hosted)",
        "pinecone": "$70/month (1M vectors)"
    },
    "llm_inference": {
        "per_query": "~1000 tokens × $0.00015 = $0.15",
        "daily": "1000 queries × $0.15 = $150"
    },
    "monthly_total": "$150-250"
}

# Fine-tuning costs
finetuning_costs = {
    "training": {
        "gpt_4o_mini": "1M tokens × $3 = $3,000",
        "gpt_4o": "1M tokens × $25 = $25,000"
    },
    "inference": {
        "per_query": "~500 tokens × $0.00060 = $0.30",
        "daily": "1000 queries × $0.30 = $300"
    },
    "retraining": "Every update = another $3,000-25,000",
    "monthly_total": "$300+ inference + retraining costs"
}
```

## Hybrid Approach

```python
class HybridSystem:
    """Combine RAG and fine-tuning for best results."""
    
    def __init__(self):
        # Fine-tuned model for consistent behavior
        self.model = "ft:gpt-4o-mini:org:support-bot-v3"
        
        # RAG for up-to-date knowledge
        self.rag = RAGSystem()
    
    def respond(self, query: str) -> str:
        # Get relevant context
        context = self.rag.retrieve(query, n=3)
        
        # Generate with fine-tuned model + context
        response = self.client.chat.completions.create(
            model=self.model,  # Knows HOW to respond
            messages=[{
                "role": "user",
                "content": f"Knowledge:\n{context}\n\nCustomer: {query}"
            }]
        )
        
        return response.choices[0].message.content

# Benefits:
# - Fine-tuned model ensures consistent tone/format
# - RAG provides current, accurate information
# - Can update knowledge without retraining
# - Can update behavior without re-indexing
```

## Summary

```yaml
choose_rag_when:
  - "Knowledge updates frequently"
  - "Need to cite sources"
  - "Want to avoid training costs"
  - "Data privacy is critical"
  - "Testing and iterating quickly"

choose_finetuning_when:
  - "Need consistent style/format"
  - "Domain-specific language patterns"
  - "Reducing prompt complexity"
  - "Specialized reasoning patterns"
  - "High volume, cost-sensitive"

choose_both_when:
  - "Complex domains (medical, legal)"
  - "Need current knowledge + specific behavior"
  - "Production systems with strict requirements"

start_with_rag:
  - "Easier to implement and iterate"
  - "No training data needed"
  - "Can always add fine-tuning later"
```

## Next Steps

1. **RAG Architecture** - Design your RAG system
2. **Document Processing** - Prepare documents for RAG
3. **Fine-tuning Guide** - When you're ready to fine-tune
