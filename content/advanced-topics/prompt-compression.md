# Prompt Compression Techniques

Learn strategies to reduce token usage while maintaining output quality, directly impacting LLM costs.

---

## Why Prompt Compression Matters

LLM costs scale linearly with token count. Reducing tokens by 50% cuts costs by 50%.

| Scenario | Original Tokens | Compressed | Savings |
|----------|----------------|------------|---------|
| RAG Context | 4000 tokens | 1500 tokens | 62.5% |
| System Prompts | 800 tokens | 300 tokens | 62.5% |
| Chat History | 2000 tokens | 800 tokens | 60% |

---

## Text Compression Techniques

### 1. LLMLingua - Microsoft's Compression Library

```python
from llmlingua import PromptCompressor

class LLMLinguaCompressor:
    """Compress prompts using LLMLingua"""
    
    def __init__(self, compression_ratio: float = 0.5):
        self.compressor = PromptCompressor(
            model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            use_llmlingua2=True
        )
        self.ratio = compression_ratio
    
    def compress(
        self,
        context: str,
        instruction: str = "",
        question: str = ""
    ) -> dict:
        """
        Compress context while preserving important information.
        
        Returns:
            dict with compressed_prompt and compression stats
        """
        result = self.compressor.compress_prompt(
            context=[context],
            instruction=instruction,
            question=question,
            target_token=int(len(context.split()) * self.ratio),
            condition_compare=True,
            condition_in_question="after",
            reorder_context="sort",
            dynamic_context_compression_ratio=0.4
        )
        
        return {
            "compressed_prompt": result["compressed_prompt"],
            "original_tokens": result["origin_tokens"],
            "compressed_tokens": result["compressed_tokens"],
            "compression_ratio": result["ratio"]
        }

# Usage
compressor = LLMLinguaCompressor(compression_ratio=0.4)

long_context = """
The quarterly financial report for Q3 2024 shows significant growth 
across all business segments. Revenue increased by 15% compared to 
the previous quarter, reaching $2.3 billion. The technology division 
led the growth with a 22% increase, while the services division 
showed steady improvement at 8%. Operating margins improved by 
2 percentage points to 18%, driven by cost optimization initiatives 
and economies of scale. The company's cash position remains strong 
at $5.2 billion, providing flexibility for strategic investments 
and potential acquisitions. Employee headcount grew by 5% to support 
expansion in emerging markets, particularly in Asia-Pacific and 
Latin America regions.
"""

result = compressor.compress(
    context=long_context,
    question="What was the revenue growth?"
)

print(f"Original: {result['original_tokens']} tokens")
print(f"Compressed: {result['compressed_tokens']} tokens")
print(f"Ratio: {result['compression_ratio']:.2%}")
print(f"\nCompressed:\n{result['compressed_prompt']}")
```

### 2. Extractive Summarization

```python
from transformers import pipeline
import spacy
from collections import Counter

class ExtractiveSummarizer:
    """Extract key sentences to compress context"""
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=-1  # CPU
        )
    
    def compress_by_extraction(
        self,
        text: str,
        num_sentences: int = 3,
        query: str = None
    ) -> str:
        """Extract most relevant sentences"""
        doc = self.nlp(text)
        sentences = list(doc.sents)
        
        # Score sentences
        scores = []
        for sent in sentences:
            score = self._score_sentence(sent, query)
            scores.append((sent.text, score))
        
        # Sort by score and take top N
        scores.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in scores[:num_sentences]]
        
        # Maintain original order
        ordered = []
        for sent in sentences:
            if sent.text in top_sentences:
                ordered.append(sent.text)
        
        return " ".join(ordered)
    
    def _score_sentence(self, sentence, query: str = None) -> float:
        """Score sentence importance"""
        score = 0
        
        # Named entities boost
        score += len(sentence.ents) * 2
        
        # Numbers boost (often important)
        score += sum(1 for token in sentence if token.like_num) * 1.5
        
        # Length penalty for very short/long
        words = len([t for t in sentence if not t.is_punct])
        if 10 <= words <= 30:
            score += 1
        
        # Query relevance
        if query:
            query_words = set(query.lower().split())
            sent_words = set(sentence.text.lower().split())
            overlap = len(query_words & sent_words)
            score += overlap * 3
        
        return score
    
    def compress_by_summary(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 50
    ) -> str:
        """Use abstractive summarization"""
        if len(text.split()) < 50:
            return text
        
        result = self.summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )
        
        return result[0]["summary_text"]

# Usage
summarizer = ExtractiveSummarizer()

compressed = summarizer.compress_by_extraction(
    text=long_context,
    num_sentences=2,
    query="revenue growth"
)
print(compressed)
```

### 3. Keyword Extraction

```python
from keybert import KeyBERT
from typing import List, Tuple

class KeywordCompressor:
    """Compress by extracting key information"""
    
    def __init__(self):
        self.kw_model = KeyBERT()
    
    def extract_key_info(
        self,
        text: str,
        top_n: int = 10,
        diversity: float = 0.7
    ) -> str:
        """Extract keywords and key phrases"""
        # Extract keywords
        keywords = self.kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            use_maxsum=True,
            nr_candidates=20,
            top_n=top_n,
            diversity=diversity
        )
        
        # Format as compressed context
        key_phrases = [kw[0] for kw in keywords]
        return "Key points: " + ", ".join(key_phrases)
    
    def compress_with_structure(
        self,
        text: str,
        sections: List[str] = None
    ) -> str:
        """Compress maintaining document structure"""
        if sections is None:
            # Auto-detect paragraphs as sections
            sections = text.split("\n\n")
        
        compressed_sections = []
        for section in sections:
            if len(section.strip()) < 20:
                continue
            
            keywords = self.kw_model.extract_keywords(
                section,
                keyphrase_ngram_range=(1, 2),
                top_n=3
            )
            
            key_phrases = [kw[0] for kw in keywords]
            if key_phrases:
                compressed_sections.append(", ".join(key_phrases))
        
        return " | ".join(compressed_sections)

# Usage
kw_compressor = KeywordCompressor()
compressed = kw_compressor.extract_key_info(long_context, top_n=8)
print(compressed)
# Output: Key points: quarterly financial report, revenue increased, technology division, operating margins, cash position, strategic investments, emerging markets, employee headcount
```

---

## Chat History Compression

### Rolling Summary

```python
from openai import OpenAI

class ChatHistoryCompressor:
    """Compress chat history while preserving context"""
    
    def __init__(self, max_tokens: int = 500):
        self.client = OpenAI()
        self.max_tokens = max_tokens
        self.summary = ""
        self.recent_messages = []
        self.summary_threshold = 10  # Summarize every N messages
    
    def add_message(self, role: str, content: str):
        """Add message and compress if needed"""
        self.recent_messages.append({
            "role": role,
            "content": content
        })
        
        # Check if compression needed
        if len(self.recent_messages) >= self.summary_threshold:
            self._compress_history()
    
    def _compress_history(self):
        """Compress older messages into summary"""
        # Keep last 4 messages uncompressed
        to_summarize = self.recent_messages[:-4]
        self.recent_messages = self.recent_messages[-4:]
        
        if not to_summarize:
            return
        
        # Format messages for summarization
        history_text = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in to_summarize
        ])
        
        # Summarize with LLM
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Summarize this conversation concisely, preserving key facts, decisions, and context."
                },
                {
                    "role": "user",
                    "content": f"Previous summary:\n{self.summary}\n\nNew messages:\n{history_text}"
                }
            ],
            max_tokens=200
        )
        
        self.summary = response.choices[0].message.content
    
    def get_context_messages(self) -> list:
        """Get compressed context for LLM"""
        messages = []
        
        # Add summary if exists
        if self.summary:
            messages.append({
                "role": "system",
                "content": f"Conversation summary: {self.summary}"
            })
        
        # Add recent messages
        messages.extend(self.recent_messages)
        
        return messages
    
    def estimate_tokens(self) -> int:
        """Estimate current token count"""
        total = len(self.summary.split()) * 1.3
        for msg in self.recent_messages:
            total += len(msg["content"].split()) * 1.3
        return int(total)

# Usage
compressor = ChatHistoryCompressor(max_tokens=500)

# Simulate conversation
messages = [
    ("user", "Hi, I need help with my order #12345"),
    ("assistant", "I'd be happy to help with order #12345. What seems to be the issue?"),
    ("user", "It hasn't arrived yet and it's been 2 weeks"),
    ("assistant", "I apologize for the delay. Let me check the shipping status for you."),
    ("user", "Thanks, I ordered it on January 15th"),
    ("assistant", "I can see your order was shipped on January 18th. There seems to be a delay in transit."),
    ("user", "Can you expedite it?"),
    ("assistant", "Yes, I've upgraded your shipping. It should arrive within 2 days."),
    ("user", "Great, will I get a new tracking number?"),
    ("assistant", "Yes, you'll receive the new tracking number via email within the hour."),
    ("user", "Perfect, thank you!"),  # This triggers compression
]

for role, content in messages:
    compressor.add_message(role, content)

context = compressor.get_context_messages()
print(f"Token estimate: {compressor.estimate_tokens()}")
```

### Sliding Window with Importance Scoring

```python
class ImportanceBasedCompressor:
    """Keep important messages, compress less important ones"""
    
    def __init__(self, window_size: int = 6, max_tokens: int = 1000):
        self.window_size = window_size
        self.max_tokens = max_tokens
        self.messages = []
    
    def add_message(self, role: str, content: str, importance: float = None):
        """Add message with optional importance score"""
        if importance is None:
            importance = self._calculate_importance(content)
        
        self.messages.append({
            "role": role,
            "content": content,
            "importance": importance
        })
    
    def _calculate_importance(self, content: str) -> float:
        """Score message importance"""
        score = 0.5  # Base score
        
        # Questions are important
        if "?" in content:
            score += 0.2
        
        # Decisions and confirmations
        decision_words = ["yes", "no", "agree", "confirm", "proceed", "cancel"]
        if any(word in content.lower() for word in decision_words):
            score += 0.2
        
        # Numbers and specific data
        import re
        if re.search(r'\d+', content):
            score += 0.1
        
        # Named entities (simplified)
        if re.search(r'#\d+|@\w+|\$\d+', content):
            score += 0.2
        
        return min(score, 1.0)
    
    def get_compressed_history(self) -> list:
        """Get compressed message history"""
        if len(self.messages) <= self.window_size:
            return [{"role": m["role"], "content": m["content"]} 
                    for m in self.messages]
        
        # Always keep most recent messages
        recent = self.messages[-self.window_size // 2:]
        older = self.messages[:-self.window_size // 2]
        
        # Sort older by importance, keep top ones
        older_sorted = sorted(older, key=lambda x: x["importance"], reverse=True)
        important_older = older_sorted[:self.window_size // 2]
        
        # Combine and sort by original order
        all_kept = important_older + recent
        
        # Convert back to simple format
        return [{"role": m["role"], "content": m["content"]} for m in all_kept]
```

---

## RAG Context Compression

### Selective Context Inclusion

```python
from typing import List, Dict
import numpy as np

class RAGContextCompressor:
    """Compress RAG context based on relevance and redundancy"""
    
    def __init__(self, max_tokens: int = 2000):
        self.max_tokens = max_tokens
    
    def compress_contexts(
        self,
        contexts: List[Dict],
        query: str,
        query_embedding: np.ndarray = None
    ) -> List[str]:
        """
        Select and compress most relevant contexts.
        
        Args:
            contexts: List of {text, score, embedding} dicts
            query: User query
            query_embedding: Optional query embedding for reranking
        """
        # Sort by relevance score
        sorted_contexts = sorted(
            contexts, 
            key=lambda x: x.get("score", 0), 
            reverse=True
        )
        
        # Remove redundant contexts
        unique_contexts = self._remove_redundant(sorted_contexts)
        
        # Fit within token budget
        selected = self._fit_token_budget(unique_contexts)
        
        return [c["text"] for c in selected]
    
    def _remove_redundant(
        self,
        contexts: List[Dict],
        similarity_threshold: float = 0.85
    ) -> List[Dict]:
        """Remove contexts that are too similar to already selected ones"""
        if not contexts:
            return []
        
        selected = [contexts[0]]
        
        for ctx in contexts[1:]:
            is_redundant = False
            
            for selected_ctx in selected:
                # Check text similarity (simplified)
                sim = self._text_similarity(
                    ctx["text"], 
                    selected_ctx["text"]
                )
                
                if sim > similarity_threshold:
                    is_redundant = True
                    break
            
            if not is_redundant:
                selected.append(ctx)
        
        return selected
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple Jaccard similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0
    
    def _fit_token_budget(self, contexts: List[Dict]) -> List[Dict]:
        """Select contexts that fit within token budget"""
        selected = []
        current_tokens = 0
        
        for ctx in contexts:
            ctx_tokens = len(ctx["text"].split()) * 1.3  # Approximate
            
            if current_tokens + ctx_tokens <= self.max_tokens:
                selected.append(ctx)
                current_tokens += ctx_tokens
            else:
                # Try to fit a compressed version
                remaining = self.max_tokens - current_tokens
                if remaining > 100:  # Only if meaningful space left
                    compressed = self._truncate_context(
                        ctx["text"],
                        int(remaining / 1.3)
                    )
                    selected.append({"text": compressed, "score": ctx.get("score", 0)})
                break
        
        return selected
    
    def _truncate_context(self, text: str, max_words: int) -> str:
        """Intelligently truncate context"""
        words = text.split()
        
        if len(words) <= max_words:
            return text
        
        # Keep beginning and end (often most informative)
        half = max_words // 2
        return " ".join(words[:half]) + " ... " + " ".join(words[-half:])
```

---

## System Prompt Optimization

### Template-Based Compression

```python
class SystemPromptOptimizer:
    """Optimize system prompts for minimal tokens"""
    
    # Verbose to concise mappings
    COMPRESSION_RULES = {
        "You are an AI assistant that helps users": "You help users",
        "Please make sure to": "Ensure",
        "It is important that you": "Always",
        "When responding to the user": "When responding",
        "In order to": "To",
        "Make sure that": "Ensure",
        "You should always": "Always",
        "Please remember to": "",
        "Keep in mind that": "",
        "Note that": "",
    }
    
    def optimize(self, prompt: str) -> str:
        """Compress system prompt"""
        optimized = prompt
        
        # Apply compression rules
        for verbose, concise in self.COMPRESSION_RULES.items():
            optimized = optimized.replace(verbose, concise)
        
        # Remove redundant whitespace
        optimized = " ".join(optimized.split())
        
        # Remove filler phrases
        filler_phrases = [
            "basically", "actually", "really", "very",
            "please note", "it should be noted"
        ]
        for filler in filler_phrases:
            optimized = optimized.replace(filler, "")
        
        return optimized.strip()
    
    def restructure_as_bullets(self, prompt: str) -> str:
        """Convert prose to bullet points (often more token-efficient)"""
        # Split into sentences
        import re
        sentences = re.split(r'[.!?]+', prompt)
        
        # Convert to bullet format
        bullets = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 10:  # Skip very short fragments
                bullets.append(f"- {sent}")
        
        return "\n".join(bullets)

# Example
optimizer = SystemPromptOptimizer()

verbose_prompt = """
You are an AI assistant that helps users with their questions. 
Please make sure to always provide accurate and helpful responses.
It is important that you maintain a professional and friendly tone.
When responding to the user, make sure that your answers are clear and concise.
Please remember to cite sources when providing factual information.
"""

optimized = optimizer.optimize(verbose_prompt)
print(f"Original: {len(verbose_prompt.split())} words")
print(f"Optimized: {len(optimized.split())} words")
print(f"\nOptimized prompt:\n{optimized}")

# Output:
# Original: 65 words
# Optimized: 34 words
```

---

## Measuring Compression Quality

```python
from openai import OpenAI

class CompressionEvaluator:
    """Evaluate quality of compression"""
    
    def __init__(self):
        self.client = OpenAI()
    
    def evaluate_semantic_preservation(
        self,
        original: str,
        compressed: str,
        test_questions: List[str]
    ) -> dict:
        """
        Test if compressed text answers questions as well as original.
        """
        results = []
        
        for question in test_questions:
            # Answer with original
            original_answer = self._ask(question, original)
            
            # Answer with compressed
            compressed_answer = self._ask(question, compressed)
            
            # Compare answers
            similarity = self._compare_answers(
                original_answer, 
                compressed_answer,
                question
            )
            
            results.append({
                "question": question,
                "similarity": similarity,
                "original_answer": original_answer[:100],
                "compressed_answer": compressed_answer[:100]
            })
        
        avg_similarity = sum(r["similarity"] for r in results) / len(results)
        
        return {
            "average_similarity": avg_similarity,
            "compression_ratio": len(compressed.split()) / len(original.split()),
            "quality_score": avg_similarity * (1 - len(compressed.split()) / len(original.split())),
            "details": results
        }
    
    def _ask(self, question: str, context: str) -> str:
        """Ask a question given context"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"Context: {context}"},
                {"role": "user", "content": question}
            ],
            max_tokens=100
        )
        return response.choices[0].message.content
    
    def _compare_answers(
        self,
        answer1: str,
        answer2: str,
        question: str
    ) -> float:
        """Compare two answers for semantic similarity"""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Rate how similar these two answers are on a scale of 0-1.
                
Question: {question}
Answer 1: {answer1}
Answer 2: {answer2}

Return only a number between 0 and 1."""
            }],
            max_tokens=10
        )
        
        try:
            return float(response.choices[0].message.content.strip())
        except:
            return 0.5

# Usage
evaluator = CompressionEvaluator()

test_questions = [
    "What was the revenue growth?",
    "Which division performed best?",
    "What is the cash position?"
]

results = evaluator.evaluate_semantic_preservation(
    original=long_context,
    compressed=compressed_context,
    test_questions=test_questions
)

print(f"Quality Score: {results['quality_score']:.2f}")
print(f"Compression Ratio: {results['compression_ratio']:.1%}")
```

---

## Best Practices

1. **Match compression to use case** - FAQ needs less compression than complex analysis
2. **Preserve key facts** - Numbers, names, dates should survive compression
3. **Test output quality** - Compressed context should produce similar outputs
4. **Consider embedding costs** - Some compression methods need embeddings
5. **Cache compressed versions** - Don't re-compress the same content

---

## Key Takeaways

1. **LLMLingua** provides state-of-the-art compression with minimal quality loss
2. **Chat history** benefits from rolling summaries
3. **RAG contexts** should be deduplicated before inclusion
4. **System prompts** can often be reduced 40-60% without impact
5. **Always measure** - compression savings vs quality tradeoff

---

## Next Steps

- [Semantic Caching](/learn/advanced-topics/optimization/semantic-caching) - Avoid repeated LLM calls
- [Model Routing](/learn/advanced-topics/optimization/model-routing) - Use cheaper models when appropriate
