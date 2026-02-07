# Managing Context Effectively

## The Challenge

As conversations grow and documents pile up, managing context becomes critical. Poor context management leads to:

- **Context overflow** - Exceeding the model's limit
- **Lost information** - Important details pruned incorrectly
- **Increased costs** - Paying for irrelevant tokens
- **Slower responses** - Larger contexts take longer to process

This guide covers practical strategies for effective context management.

## Strategy 1: Sliding Window

Keep only the most recent N messages in the conversation.

```python
class SlidingWindowContext:
    """Maintain a fixed-size window of recent messages."""
    
    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self.messages = []
        self.system_prompt = None
    
    def set_system(self, prompt: str):
        self.system_prompt = prompt
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        
        # Keep only recent messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_context(self) -> list:
        context = []
        if self.system_prompt:
            context.append({"role": "system", "content": self.system_prompt})
        context.extend(self.messages)
        return context
```

```
┌─────────────────────────────────────────────────────────────────┐
│                    Sliding Window                               │
│                                                                 │
│   Message 1  │  Dropped                                        │
│   Message 2  │  Dropped                                        │
│   Message 3  │  Dropped                                        │
│   ─────────────────────                                        │
│   Message 4  │  ← Window Start                                 │
│   Message 5  │                                                 │
│   Message 6  │  Included                                       │
│   Message 7  │                                                 │
│   Message 8  │  ← Window End (current)                         │
│                                                                 │
│   Window size: 5 messages                                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Pros:** Simple, predictable
**Cons:** Old but important context is lost

---

## Strategy 2: Summarization-Based Compression

Periodically summarize old messages to maintain long-term context.

```python
class SummarizingContext:
    """Compress old messages into summaries."""
    
    def __init__(self, summarize_threshold: int = 10, keep_recent: int = 4):
        self.summarize_threshold = summarize_threshold
        self.keep_recent = keep_recent
        self.summary = ""
        self.messages = []
        self.llm_client = None  # Set your LLM client
    
    async def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        
        # Check if we need to summarize
        if len(self.messages) > self.summarize_threshold:
            await self._compress()
    
    async def _compress(self):
        """Summarize old messages and keep recent ones."""
        
        # Messages to summarize
        old_messages = self.messages[:-self.keep_recent]
        recent_messages = self.messages[-self.keep_recent:]
        
        # Create summary prompt
        conversation_text = "\n".join([
            f"{m['role'].upper()}: {m['content']}" 
            for m in old_messages
        ])
        
        summary_prompt = f"""
        Summarize this conversation, preserving key information, 
        decisions made, and any important context:
        
        Previous Summary: {self.summary}
        
        New Messages:
        {conversation_text}
        
        Provide a concise summary (max 200 words).
        """
        
        # Generate summary
        self.summary = await self.llm_client.generate(summary_prompt)
        
        # Keep only recent messages
        self.messages = recent_messages
    
    def get_context(self) -> list:
        context = []
        
        if self.summary:
            context.append({
                "role": "system",
                "content": f"Previous conversation summary: {self.summary}"
            })
        
        context.extend(self.messages)
        return context
```

```
Before compression:
┌─────────────────────────────────────────────────────────────────┐
│  Msg 1 │ Msg 2 │ Msg 3 │ Msg 4 │ Msg 5 │ Msg 6 │ Msg 7 │ Msg 8 │
└─────────────────────────────────────────────────────────────────┘

After compression:
┌─────────────────────────────────────────────────────────────────┐
│     [Summary of Msgs 1-4]     │ Msg 5 │ Msg 6 │ Msg 7 │ Msg 8 │
└─────────────────────────────────────────────────────────────────┘
```

**Pros:** Maintains long-term memory efficiently
**Cons:** Some nuance lost in summarization, extra API calls

---

## Strategy 3: Token-Budgeted Context

Allocate token budgets to different context components.

```python
import tiktoken

class TokenBudgetedContext:
    """Manage context with explicit token budgets."""
    
    def __init__(
        self,
        max_total_tokens: int = 120000,
        system_budget: int = 500,
        history_budget: int = 10000,
        documents_budget: int = 80000,
        response_reserve: int = 4000
    ):
        self.enc = tiktoken.encoding_for_model("gpt-4")
        
        self.max_total = max_total_tokens
        self.budgets = {
            "system": system_budget,
            "history": history_budget,
            "documents": documents_budget,
            "response": response_reserve
        }
        
        self.system_prompt = ""
        self.messages = []
        self.documents = []
    
    def _count_tokens(self, text: str) -> int:
        return len(self.enc.encode(text))
    
    def set_system_prompt(self, prompt: str):
        tokens = self._count_tokens(prompt)
        if tokens > self.budgets["system"]:
            # Truncate to fit budget
            encoded = self.enc.encode(prompt)[:self.budgets["system"]]
            prompt = self.enc.decode(encoded)
        self.system_prompt = prompt
    
    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self._trim_history()
    
    def add_document(self, doc: str, priority: int = 1):
        """Add document with priority (higher = more important)."""
        tokens = self._count_tokens(doc)
        self.documents.append({
            "content": doc,
            "tokens": tokens,
            "priority": priority
        })
        self._trim_documents()
    
    def _trim_history(self):
        """Remove old messages to fit history budget."""
        total = sum(self._count_tokens(m["content"]) for m in self.messages)
        
        while total > self.budgets["history"] and len(self.messages) > 1:
            # Remove oldest non-system message
            removed = self.messages.pop(0)
            total -= self._count_tokens(removed["content"])
    
    def _trim_documents(self):
        """Remove low-priority documents to fit budget."""
        total = sum(d["tokens"] for d in self.documents)
        
        while total > self.budgets["documents"] and self.documents:
            # Sort by priority and remove lowest
            self.documents.sort(key=lambda x: x["priority"])
            removed = self.documents.pop(0)
            total -= removed["tokens"]
    
    def get_context(self) -> list:
        context = []
        
        # System prompt
        if self.system_prompt:
            context.append({
                "role": "system", 
                "content": self.system_prompt
            })
        
        # Documents (injected as system context)
        if self.documents:
            docs_content = "\n\n---\n\n".join(d["content"] for d in self.documents)
            context.append({
                "role": "system",
                "content": f"Relevant documents:\n{docs_content}"
            })
        
        # Conversation history
        context.extend(self.messages)
        
        return context
    
    def get_remaining_budget(self) -> int:
        """Calculate remaining tokens for new content."""
        used = self._count_tokens(self.system_prompt)
        used += sum(self._count_tokens(m["content"]) for m in self.messages)
        used += sum(d["tokens"] for d in self.documents)
        
        return self.max_total - used - self.budgets["response"]
```

---

## Strategy 4: Importance-Based Retention

Keep messages based on importance rather than recency.

```python
class ImportanceBasedContext:
    """Retain messages based on computed importance scores."""
    
    def __init__(self, max_tokens: int = 50000):
        self.max_tokens = max_tokens
        self.messages = []
        self.enc = tiktoken.encoding_for_model("gpt-4")
    
    def add_message(self, role: str, content: str, importance: float = 0.5):
        """
        Add message with importance score.
        importance: 0.0 (drop first) to 1.0 (never drop)
        """
        tokens = len(self.enc.encode(content))
        
        self.messages.append({
            "role": role,
            "content": content,
            "tokens": tokens,
            "importance": importance,
            "position": len(self.messages)  # Track original position
        })
        
        self._trim()
    
    def _compute_retention_score(self, msg: dict, total_msgs: int) -> float:
        """
        Compute retention score combining importance and recency.
        Higher score = more likely to keep.
        """
        # Recency bonus (newer messages score higher)
        recency = msg["position"] / max(1, total_msgs)
        
        # Combine importance and recency
        return msg["importance"] * 0.7 + recency * 0.3
    
    def _trim(self):
        """Remove messages with lowest retention scores."""
        total_tokens = sum(m["tokens"] for m in self.messages)
        
        while total_tokens > self.max_tokens and len(self.messages) > 2:
            # Compute scores
            total = len(self.messages)
            scores = [
                (i, self._compute_retention_score(m, total))
                for i, m in enumerate(self.messages)
            ]
            
            # Find lowest score (but never remove first or last)
            removable = [(i, s) for i, s in scores if 0 < i < len(self.messages) - 1]
            
            if not removable:
                break
            
            # Remove message with lowest score
            remove_idx = min(removable, key=lambda x: x[1])[0]
            removed = self.messages.pop(remove_idx)
            total_tokens -= removed["tokens"]
    
    def get_context(self) -> list:
        # Return messages in original order
        sorted_msgs = sorted(self.messages, key=lambda x: x["position"])
        return [{"role": m["role"], "content": m["content"]} for m in sorted_msgs]
```

### What Makes a Message Important?

```python
def compute_importance(message: dict, context: str = "") -> float:
    """Heuristics for message importance."""
    
    content = message["content"].lower()
    importance = 0.5  # Base score
    
    # High importance indicators
    if message["role"] == "system":
        importance = 1.0  # Never drop system prompts
    
    if any(word in content for word in ["important", "critical", "must", "required"]):
        importance += 0.2
    
    if "?" in content:  # Questions are often important
        importance += 0.1
    
    if any(word in content for word in ["decision", "agreed", "confirmed"]):
        importance += 0.2  # Decisions should be retained
    
    # Low importance indicators
    if content in ["ok", "thanks", "got it", "sure"]:
        importance -= 0.3
    
    if len(content) < 20:  # Very short messages
        importance -= 0.1
    
    return min(1.0, max(0.0, importance))
```

---

## Strategy 5: Semantic Compression

Use embeddings to keep semantically distinct content.

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SemanticContext:
    """Keep semantically diverse messages."""
    
    def __init__(self, max_messages: int = 20, similarity_threshold: float = 0.85):
        self.max_messages = max_messages
        self.similarity_threshold = similarity_threshold
        self.messages = []
        self.embeddings = []
    
    async def add_message(self, role: str, content: str):
        # Get embedding for new message
        embedding = await self._embed(content)
        
        # Check for redundancy
        if not self._is_redundant(embedding):
            self.messages.append({"role": role, "content": content})
            self.embeddings.append(embedding)
        
        # Trim if needed
        while len(self.messages) > self.max_messages:
            self._remove_most_redundant()
    
    async def _embed(self, text: str) -> np.ndarray:
        """Get embedding for text (implement with your embedding model)."""
        # Using OpenAI embeddings as example
        response = await openai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return np.array(response.data[0].embedding)
    
    def _is_redundant(self, new_embedding: np.ndarray) -> bool:
        """Check if new content is too similar to existing."""
        if not self.embeddings:
            return False
        
        similarities = cosine_similarity([new_embedding], self.embeddings)[0]
        return any(sim > self.similarity_threshold for sim in similarities)
    
    def _remove_most_redundant(self):
        """Remove the message most similar to its neighbors."""
        if len(self.embeddings) < 3:
            # Remove oldest
            self.messages.pop(0)
            self.embeddings.pop(0)
            return
        
        # Compute average similarity to other messages
        avg_similarities = []
        for i, emb in enumerate(self.embeddings):
            others = [e for j, e in enumerate(self.embeddings) if j != i]
            sims = cosine_similarity([emb], others)[0]
            avg_similarities.append(np.mean(sims))
        
        # Remove most redundant (highest average similarity)
        remove_idx = np.argmax(avg_similarities)
        self.messages.pop(remove_idx)
        self.embeddings.pop(remove_idx)
```

---

## Combining Strategies

Real applications often combine multiple strategies:

```python
class HybridContextManager:
    """Combines multiple context management strategies."""
    
    def __init__(self, max_tokens: int = 100000):
        self.max_tokens = max_tokens
        
        # Component managers
        self.system_prompt = ""
        self.summary = ""
        self.recent_messages = []  # Sliding window
        self.important_facts = []  # Key information to preserve
        self.documents = []  # RAG context
        
        self.enc = tiktoken.encoding_for_model("gpt-4")
    
    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt
    
    async def add_message(self, role: str, content: str):
        self.recent_messages.append({"role": role, "content": content})
        
        # Extract important facts
        facts = await self._extract_facts(content)
        self.important_facts.extend(facts)
        
        # Manage window
        if len(self.recent_messages) > 10:
            old_messages = self.recent_messages[:-6]
            self.recent_messages = self.recent_messages[-6:]
            
            # Summarize old messages
            self.summary = await self._update_summary(old_messages)
    
    def add_documents(self, docs: list):
        """Add RAG-retrieved documents."""
        self.documents = docs
    
    async def _extract_facts(self, content: str) -> list:
        """Extract key facts worth preserving."""
        prompt = f"""
        Extract any key facts, decisions, or important information from:
        "{content}"
        
        Return as a JSON array of strings. Return [] if nothing important.
        """
        response = await self.llm.generate(prompt)
        return json.loads(response)
    
    async def _update_summary(self, messages: list) -> str:
        """Update the rolling summary."""
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        prompt = f"""
        Update this conversation summary with new information:
        
        Current summary: {self.summary or 'No previous summary'}
        
        New messages:
        {text}
        
        Provide updated summary (max 150 words):
        """
        return await self.llm.generate(prompt)
    
    def get_context(self) -> list:
        context = []
        
        # 1. System prompt
        if self.system_prompt:
            context.append({"role": "system", "content": self.system_prompt})
        
        # 2. Summary of old conversation
        if self.summary:
            context.append({
                "role": "system",
                "content": f"Previous conversation summary: {self.summary}"
            })
        
        # 3. Important facts
        if self.important_facts:
            facts = "\n".join(f"• {f}" for f in self.important_facts[-20:])
            context.append({
                "role": "system",
                "content": f"Key facts to remember:\n{facts}"
            })
        
        # 4. Retrieved documents
        if self.documents:
            docs = "\n\n---\n\n".join(self.documents[:5])
            context.append({
                "role": "system",
                "content": f"Relevant documents:\n{docs}"
            })
        
        # 5. Recent messages
        context.extend(self.recent_messages)
        
        return context
```

## Best Practices

### 1. Always Monitor Token Usage

```python
def log_context_usage(context: list, max_tokens: int):
    """Log context token usage for monitoring."""
    enc = tiktoken.encoding_for_model("gpt-4")
    total = sum(len(enc.encode(m["content"])) for m in context)
    
    print(f"Context: {total}/{max_tokens} tokens ({total/max_tokens*100:.1f}%)")
    
    if total > max_tokens * 0.9:
        print("⚠️ Warning: Context near limit!")
```

### 2. Preserve System Prompts

```python
# Never drop system prompts - they define behavior
if message["role"] == "system":
    importance = 1.0  # Maximum importance
```

### 3. Keep User's Last Message

```python
# Always include the user's most recent message
# It contains the current query/request
```

### 4. Test with Edge Cases

```python
# Test scenarios:
# - Very long single message
# - Rapid-fire short messages
# - Long conversation (100+ turns)
# - Large document injection
```

## Key Takeaways

1. **No single strategy fits all** - Combine based on your use case
2. **Prioritize recent + important** - Not just one or the other
3. **Monitor continuously** - Token counts can surprise you
4. **Summarize strategically** - Preserve key decisions and facts
5. **Test at scale** - Edge cases reveal management flaws

## Next Steps

Ready to put this into practice? Head to the **Token Counter Lab** to experiment with context management strategies hands-on.
