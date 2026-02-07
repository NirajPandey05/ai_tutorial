# RAG Prompts: Designing Effective Context-Augmented Prompts

Learn to structure prompts that effectively use retrieved context for high-quality answers.

## The RAG Prompt Challenge

```yaml
goals:
  - "Incorporate context naturally into prompts"
  - "Guide the model to use retrieved information"
  - "Handle conflicting or incomplete context"
  - "Ensure accurate, grounded responses"

common_mistakes:
  - "Dumping context without structure"
  - "Not instructing model to use context"
  - "Missing fallback instructions"
  - "Ignoring source attribution"
```

## Basic RAG Prompt Template

```python
from openai import OpenAI


def basic_rag_prompt(query: str, context: list[dict]) -> str:
    """Simple RAG prompt template."""
    
    # Format context
    context_text = "\n\n".join([
        f"[Source {i+1}]: {doc['content']}"
        for i, doc in enumerate(context)
    ])
    
    prompt = f"""Answer the question based on the provided context.

Context:
{context_text}

Question: {query}

Answer:"""
    
    return prompt


# Usage
client = OpenAI()

context = [
    {"content": "Our return policy allows returns within 30 days of purchase."},
    {"content": "Items must be in original packaging with receipt."},
]

prompt = basic_rag_prompt("What is your return policy?", context)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print(response.choices[0].message.content)
```

## Structured RAG Prompts

```python
class RAGPromptBuilder:
    """Build structured RAG prompts with various options."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model
    
    def build_prompt(
        self,
        query: str,
        context: list[dict],
        system_prompt: str = None,
        require_citations: bool = False,
        allow_uncertainty: bool = True
    ) -> list[dict]:
        """Build a structured RAG prompt."""
        
        messages = []
        
        # System message
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({
                "role": "system",
                "content": self._default_system_prompt(require_citations, allow_uncertainty)
            })
        
        # Format context
        context_block = self._format_context(context)
        
        # User message with context and query
        user_message = f"""Context information:
{context_block}

User question: {query}"""
        
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def _default_system_prompt(
        self,
        require_citations: bool,
        allow_uncertainty: bool
    ) -> str:
        """Generate default system prompt."""
        
        base = """You are a helpful assistant that answers questions based on the provided context.

Guidelines:
- Base your answer ONLY on the information in the context
- Be concise and direct
- If the context doesn't contain relevant information, say so"""
        
        if require_citations:
            base += "\n- Cite your sources using [Source N] notation"
        
        if allow_uncertainty:
            base += "\n- If you're unsure, express uncertainty rather than guessing"
        else:
            base += "\n- Only provide answers you're confident about from the context"
        
        return base
    
    def _format_context(self, context: list[dict]) -> str:
        """Format context documents."""
        
        formatted = []
        for i, doc in enumerate(context):
            source_info = f"[Source {i+1}]"
            
            if "metadata" in doc:
                if "title" in doc["metadata"]:
                    source_info += f" ({doc['metadata']['title']})"
                if "date" in doc["metadata"]:
                    source_info += f" [{doc['metadata']['date']}]"
            
            formatted.append(f"{source_info}\n{doc['content']}")
        
        return "\n\n---\n\n".join(formatted)
    
    def generate(
        self,
        query: str,
        context: list[dict],
        **kwargs
    ) -> str:
        """Build prompt and generate response."""
        
        messages = self.build_prompt(query, context, **kwargs)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0
        )
        
        return response.choices[0].message.content


# Usage
builder = RAGPromptBuilder()

context = [
    {
        "content": "TechCorp was founded in 2015 in San Francisco.",
        "metadata": {"title": "Company Overview", "date": "2024-01"}
    },
    {
        "content": "The company has grown to 500 employees across 3 offices.",
        "metadata": {"title": "Annual Report", "date": "2024-06"}
    }
]

answer = builder.generate(
    query="When was TechCorp founded and how big is it now?",
    context=context,
    require_citations=True
)

print(answer)
# Output: "TechCorp was founded in 2015 in San Francisco [Source 1]. 
# The company has since grown to 500 employees across 3 offices [Source 2]."
```

## Role-Based RAG Prompts

```python
class RoleBasedRAGPrompt:
    """RAG prompts with specific persona/role."""
    
    ROLES = {
        "customer_support": """You are a friendly customer support agent. 
Answer customer questions using the provided knowledge base.
- Be empathetic and helpful
- Use clear, simple language
- Offer to help further if needed
- Never make up policies or information""",
        
        "technical_expert": """You are a technical expert assistant.
Answer questions using the provided documentation.
- Be precise and technically accurate
- Include relevant code examples when helpful
- Reference specific documentation sections
- Acknowledge limitations in the documentation""",
        
        "legal_assistant": """You are a legal information assistant.
Provide information based on the provided legal documents.
- Be extremely precise with language
- Always cite specific sources
- Clearly state this is information, not legal advice
- Note any ambiguities or gaps in the provided information""",
        
        "research_assistant": """You are a research assistant.
Synthesize information from multiple sources to answer questions.
- Compare and contrast different sources
- Note agreements and contradictions
- Identify gaps in available information
- Suggest areas for further research"""
    }
    
    def __init__(self, role: str = "customer_support"):
        self.client = OpenAI()
        self.system_prompt = self.ROLES.get(role, self.ROLES["customer_support"])
    
    def generate(
        self,
        query: str,
        context: list[dict]
    ) -> str:
        """Generate role-appropriate response."""
        
        context_text = "\n\n".join([
            f"[{i+1}] {doc['content']}"
            for i, doc in enumerate(context)
        ])
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"""Reference Information:
{context_text}

Customer Question: {query}"""
            }
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3
        )
        
        return response.choices[0].message.content


# Usage examples
support = RoleBasedRAGPrompt(role="customer_support")
tech = RoleBasedRAGPrompt(role="technical_expert")

context = [
    {"content": "Password reset can be done via Settings > Security > Reset Password"},
    {"content": "Reset links expire after 24 hours for security reasons"}
]

# Customer support response
print("Customer Support:")
print(support.generate("How do I reset my password?", context))

# Technical expert response  
print("\nTechnical Expert:")
print(tech.generate("How do I reset my password?", context))
```

## Handling Edge Cases

```python
class RobustRAGPrompt:
    """Handle edge cases in RAG prompts."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def generate(
        self,
        query: str,
        context: list[dict],
        conversation_history: list[dict] = None
    ) -> dict:
        """Generate with edge case handling."""
        
        # Check for empty context
        if not context:
            return self._handle_no_context(query)
        
        # Check for conflicting information
        conflicts = self._detect_conflicts(context)
        
        # Build prompt with appropriate handling
        if conflicts:
            return self._handle_conflicts(query, context, conflicts)
        
        # Standard generation
        return self._standard_generate(query, context, conversation_history)
    
    def _handle_no_context(self, query: str) -> dict:
        """Handle queries with no relevant context."""
        
        messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant. The user has asked a question,
but no relevant information was found in the knowledge base.
Politely inform the user and offer alternatives."""
            },
            {
                "role": "user",
                "content": f"Question: {query}"
            }
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3
        )
        
        return {
            "answer": response.choices[0].message.content,
            "status": "no_context",
            "sources": []
        }
    
    def _detect_conflicts(self, context: list[dict]) -> list[str]:
        """Detect potential conflicting information."""
        
        if len(context) < 2:
            return []
        
        # Use LLM to detect conflicts
        context_text = "\n".join([
            f"[{i+1}] {doc['content']}"
            for i, doc in enumerate(context)
        ])
        
        messages = [
            {
                "role": "system",
                "content": "Analyze these text passages and identify any contradictions or conflicts. Return 'NONE' if no conflicts, otherwise describe the conflicts."
            },
            {"role": "user", "content": context_text}
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=200,
            temperature=0
        )
        
        result = response.choices[0].message.content.strip()
        
        if result.upper() == "NONE":
            return []
        
        return [result]
    
    def _handle_conflicts(
        self,
        query: str,
        context: list[dict],
        conflicts: list[str]
    ) -> dict:
        """Handle conflicting context."""
        
        context_text = "\n".join([
            f"[Source {i+1}] {doc['content']}"
            for i, doc in enumerate(context)
        ])
        
        messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant. The provided context contains some conflicting information.
Address the user's question while:
- Acknowledging the conflicting information
- Presenting both perspectives fairly
- Recommending they verify with an authoritative source if needed"""
            },
            {
                "role": "user",
                "content": f"""Context (with some conflicts):
{context_text}

Detected conflicts: {conflicts[0]}

Question: {query}"""
            }
        ]
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3
        )
        
        return {
            "answer": response.choices[0].message.content,
            "status": "conflicting_sources",
            "conflicts": conflicts,
            "sources": [i+1 for i in range(len(context))]
        }
    
    def _standard_generate(
        self,
        query: str,
        context: list[dict],
        conversation_history: list[dict] = None
    ) -> dict:
        """Standard RAG generation."""
        
        context_text = "\n".join([
            f"[Source {i+1}] {doc['content']}"
            for i, doc in enumerate(context)
        ])
        
        messages = [
            {
                "role": "system",
                "content": """Answer the user's question based on the provided context.
Cite sources using [Source N] notation. If the context doesn't fully answer the question, say so."""
            }
        ]
        
        # Add conversation history if present
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append({
            "role": "user",
            "content": f"""Context:
{context_text}

Question: {query}"""
        })
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0
        )
        
        return {
            "answer": response.choices[0].message.content,
            "status": "success",
            "sources": [i+1 for i in range(len(context))]
        }


# Usage
robust_rag = RobustRAGPrompt()

# Test with conflicting context
conflicting_context = [
    {"content": "The office opens at 9 AM Monday through Friday."},
    {"content": "Business hours are 8 AM to 5 PM on weekdays."}
]

result = robust_rag.generate(
    "What time does the office open?",
    conflicting_context
)

print(f"Status: {result['status']}")
print(f"Answer: {result['answer']}")
```

## Conversational RAG Prompts

```python
class ConversationalRAGPrompt:
    """RAG prompts that maintain conversation context."""
    
    def __init__(self):
        self.client = OpenAI()
        self.conversation_history = []
    
    def generate(
        self,
        query: str,
        context: list[dict]
    ) -> str:
        """Generate response maintaining conversation context."""
        
        context_text = "\n".join([
            f"[{i+1}] {doc['content']}"
            for i, doc in enumerate(context)
        ])
        
        messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant having a conversation with a user.
Use the provided context to answer questions. Maintain conversation continuity.
When the user refers to previous topics with pronouns (it, that, they), use conversation history to understand what they mean."""
            }
        ]
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Add current turn with context
        current_message = f"""New context for this question:
{context_text}

User: {query}"""
        
        messages.append({"role": "user", "content": current_message})
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3
        )
        
        answer = response.choices[0].message.content
        
        # Update history (simplified - store without context)
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": answer})
        
        # Keep history manageable
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        return answer
    
    def reset(self):
        """Reset conversation history."""
        self.conversation_history = []


# Usage
conv_rag = ConversationalRAGPrompt()

# Turn 1
context1 = [{"content": "Our enterprise plan costs $99/month per user."}]
print("User: How much is the enterprise plan?")
print("Bot:", conv_rag.generate("How much is the enterprise plan?", context1))

# Turn 2 - uses conversation context
context2 = [{"content": "Enterprise plan includes priority support and advanced analytics."}]
print("\nUser: What features does it include?")
print("Bot:", conv_rag.generate("What features does it include?", context2))

# Turn 3 - references previous context
context3 = [{"content": "Volume discounts available for 50+ users."}]
print("\nUser: Are there any discounts?")
print("Bot:", conv_rag.generate("Are there any discounts?", context3))
```

## Summary

```yaml
prompt_design_principles:
  1: "Clearly separate context from instructions"
  2: "Provide explicit guidance on using context"
  3: "Include fallback instructions"
  4: "Request citations when needed"
  5: "Handle edge cases gracefully"

template_types:
  basic: "Simple context + question"
  structured: "System prompt + formatted context + instructions"
  role_based: "Persona-specific prompting"
  conversational: "Multi-turn with history"

best_practices:
  - "Test prompts with edge cases"
  - "Include 'I don't know' instructions"
  - "Format context for readability"
  - "Use system prompts for consistent behavior"
```

## Next Steps

1. **Context Overflow** - Handle large context scenarios
2. **Citation Tracking** - Implement source attribution
3. **Streaming Sources** - Real-time citation display
