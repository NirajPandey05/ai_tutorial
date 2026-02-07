# What is RAG (Retrieval-Augmented Generation)?

RAG combines the power of LLMs with external knowledge retrieval, enabling accurate, up-to-date, and verifiable AI responses.

## The Knowledge Problem

```yaml
llm_limitations:
  training_cutoff: "Knowledge stops at training date"
  no_private_data: "Can't access your documents"
  hallucinations: "May confidently state false information"
  no_sources: "Can't cite where information came from"

example:
  question: "What's in our Q3 2025 sales report?"
  llm_alone: "I don't have access to your company documents"
  with_rag: "Based on your Q3 report, revenue grew 23%... [Source: Q3_Sales.pdf, page 4]"
```

## RAG: The Solution

RAG augments LLM responses by retrieving relevant context from external knowledge bases before generating answers.

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAG Pipeline - Detailed View                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Phase 1: INDEXING (Offline)                                    │
│  ───────────────────────────────────────────────────────────── │
│  Documents → Chunk & Clean → Embed → Store in Vector DB        │
│                                                                  │
│  Phase 2: RETRIEVAL (At Query Time)                             │
│  ───────────────────────────────────────────────────────────── │
│  User Query → Embed → Search Vector DB → Get Top-K Docs        │
│                                                                  │
│  Phase 3: GENERATION (At Query Time)                            │
│  ───────────────────────────────────────────────────────────── │
│  [Retrieved Docs + Query] → LLM → Response with Citations       │
│                                                                  │
│  Example Flow:                                                  │
│  ──────────────────────────────────────────────────────────── │
│  Input: "What's in our Q3 2025 sales report?"                  │
│                                                                  │
│  1. Embed query                                                 │
│     Query embedding: [0.21, -0.45, 0.78, ...]                  │
│                                                                  │
│  2. Search vector DB for similar docs                           │
│     Found: Q3_Sales_Summary.pdf, Financial_Report.pdf, ...     │
│                                                                  │
│  3. Build augmented prompt:                                     │
│     "Context: [Q3 report content...]\                           │
│      Question: What's in Q3 2025 report?\                       │
│      Answer: Based on the context..."                           │
│                                                                  │
│  4. LLM generates response with sources                         │
│     "Revenue grew 23% to $5.2M. [Source: Q3_Sales.pdf, pg 4]"  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Why RAG Solves These Problems

| LLM Limitation | RAG Solution |
|---|---|
| **Knowledge Cutoff** | Retrieves current documents in real-time |
| **Hallucinations** | Grounds responses in actual source material |
| **No Private Data** | Indexes company-specific documents |
| **No Attribution** | Returns exact sources: [Document, Page X] |
| **Outdated Info** | Uses latest indexed information |

## Core Components

### 1. Knowledge Base

```python
# Your documents become the knowledge source
knowledge_sources = {
    "documents": [
        "Company policies",
        "Product documentation",
        "FAQ databases",
        "Research papers",
        "Code repositories"
    ],
    "formats": [
        "PDF", "HTML", "Markdown", 
        "Word", "Text", "Code files"
    ]
}
```

### 2. Embeddings & Vector Store

```python
from openai import OpenAI
import chromadb

client = OpenAI()
chroma = chromadb.PersistentClient(path="./knowledge_db")

# Create searchable knowledge base
collection = chroma.get_or_create_collection("company_docs")

def index_document(doc_id: str, content: str, metadata: dict):
    """Add document to searchable knowledge base."""
    
    # Generate embedding
    response = client.embeddings.create(
        input=content,
        model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding
    
    # Store in vector database
    collection.add(
        ids=[doc_id],
        documents=[content],
        embeddings=[embedding],
        metadatas=[metadata]
    )
```

### 3. Retriever

```python
def retrieve_context(query: str, n_results: int = 5) -> list[str]:
    """Find relevant documents for a query."""
    
    # Generate query embedding
    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_embedding = response.data[0].embedding
    
    # Search vector database
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas"]
    )
    
    return results["documents"][0]
```

### 4. Generator (LLM with Context)

```python
def generate_with_context(query: str, context: list[str]) -> str:
    """Generate response using retrieved context."""
    
    # Format context
    context_text = "\n\n---\n\n".join(context)
    
    # Build RAG prompt
    prompt = f"""Answer the question based on the following context.
If the context doesn't contain the answer, say "I don't have enough information."

Context:
{context_text}

Question: {query}

Answer:"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1  # Lower for factual accuracy
    )
    
    return response.choices[0].message.content
```

## Complete RAG Example

```python
class SimpleRAG:
    """Basic RAG implementation."""
    
    def __init__(self, collection_name: str = "knowledge"):
        self.client = OpenAI()
        self.chroma = chromadb.PersistentClient(path="./rag_db")
        self.collection = self.chroma.get_or_create_collection(collection_name)
    
    def add_documents(self, documents: list[dict]):
        """Index documents for retrieval."""
        
        for doc in documents:
            embedding = self._embed(doc["content"])
            self.collection.add(
                ids=[doc["id"]],
                documents=[doc["content"]],
                embeddings=[embedding],
                metadatas=[doc.get("metadata", {})]
            )
    
    def query(self, question: str, n_context: int = 3) -> dict:
        """Answer question using RAG."""
        
        # Retrieve relevant context
        query_embedding = self._embed(question)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_context,
            include=["documents", "metadatas"]
        )
        
        context = results["documents"][0]
        sources = results["metadatas"][0]
        
        # Generate answer
        answer = self._generate(question, context)
        
        return {
            "answer": answer,
            "sources": sources,
            "context_used": context
        }
    
    def _embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    
    def _generate(self, question: str, context: list[str]) -> str:
        """Generate answer with context."""
        
        context_text = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(context))
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "system",
                "content": "Answer questions based on the provided context. Cite sources using [1], [2], etc."
            }, {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nQuestion: {question}"
            }],
            temperature=0.1
        )
        
        return response.choices[0].message.content


# Usage
rag = SimpleRAG()

# Add knowledge
rag.add_documents([
    {
        "id": "policy-1",
        "content": "Employees receive 20 days of paid vacation annually. Unused days can be carried over to the next year, up to a maximum of 5 days.",
        "metadata": {"source": "HR_Policy.pdf", "page": 12}
    },
    {
        "id": "policy-2", 
        "content": "Remote work is permitted up to 3 days per week with manager approval. Employees must be available during core hours (10am-3pm).",
        "metadata": {"source": "HR_Policy.pdf", "page": 15}
    },
    {
        "id": "benefits-1",
        "content": "Health insurance covers employees and dependents. Dental and vision are optional add-ons. Coverage begins on the first day of employment.",
        "metadata": {"source": "Benefits_Guide.pdf", "page": 3}
    }
])

# Query
result = rag.query("How many vacation days do I get?")
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
```

## When to Use RAG

```yaml
perfect_for:
  - Question answering over documents
  - Customer support chatbots
  - Internal knowledge bases
  - Research assistants
  - Code documentation Q&A
  - Legal/compliance queries

benefits:
  accuracy: "Answers grounded in actual documents"
  freshness: "Easy to update knowledge without retraining"
  transparency: "Can cite sources for verification"
  privacy: "Keep sensitive data in your infrastructure"
  cost: "Cheaper than fine-tuning for many use cases"

limitations:
  retrieval_quality: "Only as good as what's retrieved"
  latency: "Extra step adds response time"
  context_limits: "Can't use entire knowledge base at once"
  complex_reasoning: "Multi-hop queries can be challenging"
```

## RAG vs Just Prompting

```python
# Without RAG - LLM guesses or refuses
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{
        "role": "user",
        "content": "What's our company's remote work policy?"
    }]
)
# Output: "I don't have access to your company policies..."

# With RAG - LLM answers from your documents
context = retrieve_context("remote work policy")
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{
        "role": "user",
        "content": f"Context: {context}\n\nWhat's our company's remote work policy?"
    }]
)
# Output: "According to your HR policy, remote work is permitted up to 3 days per week..."
```

## The RAG Workflow

```
1. PREPARE (Offline - Done Once)
   ├── Load documents
   ├── Split into chunks
   ├── Generate embeddings
   └── Store in vector database

2. RETRIEVE (Online - Per Query)
   ├── Embed user query
   ├── Search vector database
   ├── Get top-k relevant chunks
   └── (Optional) Rerank results

3. GENERATE (Online - Per Query)
   ├── Format context + query
   ├── Send to LLM
   ├── Generate response
   └── Include source citations
```

## Key Success Factors

```yaml
retrieval_quality:
  - "Right chunking strategy for your content"
  - "Good embedding model for your domain"
  - "Proper metadata for filtering"
  - "Enough context without overwhelming"

prompt_engineering:
  - "Clear instructions on using context"
  - "Guidelines for handling missing info"
  - "Citation format requirements"
  - "Tone and response structure"

evaluation:
  - "Does it retrieve the right documents?"
  - "Is the answer faithful to sources?"
  - "Does it admit when it doesn't know?"
  - "Are citations accurate?"
```

## Summary

```yaml
rag_in_one_sentence: "Retrieve relevant context, then generate with that context"

core_insight: "LLMs are great at language, RAG gives them knowledge"

components:
  1. Knowledge Base: "Your documents, indexed and searchable"
  2. Retriever: "Finds relevant context for each query"  
  3. Generator: "LLM that uses context to answer"

next_steps:
  - "RAG vs Fine-tuning: When to use which"
  - "RAG Architecture: Detailed component design"
  - "Document Processing: Chunking strategies"
```

## Next Steps

1. **RAG vs Fine-tuning** - Understand when to use each approach
2. **RAG Architecture** - Deep dive into system design
3. **Document Processing** - Learn chunking strategies
