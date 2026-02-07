# Basic RAG Lab: Your First RAG Pipeline

Build a complete RAG system from scratch in this hands-on lab.

## Lab Overview

```yaml
objectives:
  - "Create a working RAG pipeline"
  - "Load and chunk documents"
  - "Build a vector store"
  - "Generate answers with context"

difficulty: Beginner
time: 30 minutes
prerequisites:
  - "Understanding of embeddings"
  - "Basic Python knowledge"
  - "OpenAI API key configured"
```

## What You'll Build

A simple question-answering system that:
1. Loads text documents
2. Chunks them into smaller pieces
3. Creates embeddings and stores them
4. Retrieves relevant chunks for questions
5. Generates answers using the context

## Part 1: Setup

```python
"""
Basic RAG Lab - Your First RAG Pipeline
Build a complete RAG system step by step
"""

# Install dependencies (run in terminal first)
# pip install openai chromadb tiktoken

from openai import OpenAI
import chromadb
import tiktoken

# Initialize clients
client = OpenAI()
chroma = chromadb.Client()  # In-memory for this lab

print("✅ Setup complete!")
```

## Part 2: Sample Documents

```python
# Our knowledge base - simple FAQ documents
DOCUMENTS = [
    {
        "id": "return-policy",
        "title": "Return Policy",
        "content": """Our return policy allows customers to return most items within 30 days 
        of purchase for a full refund. Items must be in original condition with tags attached. 
        Electronics have a 15-day return window. Sale items are final sale and cannot be returned. 
        To initiate a return, contact customer service or visit any store location with your receipt."""
    },
    {
        "id": "shipping-info",
        "title": "Shipping Information", 
        "content": """We offer several shipping options: Standard shipping (5-7 business days) is free 
        on orders over $50. Express shipping (2-3 business days) costs $9.99. Next-day shipping is 
        available for $19.99 on orders placed before 2 PM. International shipping is available to 
        select countries with delivery times of 10-14 business days."""
    },
    {
        "id": "payment-methods",
        "title": "Payment Methods",
        "content": """We accept all major credit cards (Visa, MasterCard, American Express, Discover), 
        PayPal, Apple Pay, and Google Pay. For orders over $100, we also offer Pay in 4 installments 
        through Klarna with no interest. Gift cards can be combined with other payment methods."""
    },
    {
        "id": "account-help",
        "title": "Account Help",
        "content": """To create an account, click 'Sign Up' and enter your email address. You'll 
        receive a verification email within 5 minutes. If you forgot your password, click 'Forgot 
        Password' on the login page. Passwords must be at least 8 characters with one number and 
        one special character. Contact support if you need to change your email address."""
    },
    {
        "id": "loyalty-program",
        "title": "Loyalty Program",
        "content": """Our loyalty program rewards you with points for every purchase. Earn 1 point 
        per dollar spent. 100 points equals $5 off your next order. Gold members (500+ points/year) 
        get free express shipping and early access to sales. Points expire after 12 months of 
        account inactivity."""
    }
]

print(f"📚 Loaded {len(DOCUMENTS)} documents")
for doc in DOCUMENTS:
    print(f"  - {doc['title']}")
```

## Part 3: Document Chunking

```python
def chunk_document(doc: dict, chunk_size: int = 200, overlap: int = 50) -> list[dict]:
    """Split a document into overlapping chunks."""
    
    content = doc["content"]
    chunks = []
    
    # Simple character-based chunking
    start = 0
    chunk_num = 0
    
    while start < len(content):
        end = start + chunk_size
        chunk_text = content[start:end]
        
        # Try to break at sentence boundary
        if end < len(content):
            # Look for last period, question mark, or newline
            last_break = max(
                chunk_text.rfind('. '),
                chunk_text.rfind('? '),
                chunk_text.rfind('\n')
            )
            if last_break > chunk_size // 2:
                chunk_text = chunk_text[:last_break + 1]
                end = start + last_break + 1
        
        chunks.append({
            "id": f"{doc['id']}_chunk_{chunk_num}",
            "document_id": doc["id"],
            "title": doc["title"],
            "content": chunk_text.strip(),
            "chunk_index": chunk_num
        })
        
        # Move start position with overlap
        start = end - overlap
        chunk_num += 1
    
    return chunks


# Chunk all documents
all_chunks = []
for doc in DOCUMENTS:
    chunks = chunk_document(doc, chunk_size=300, overlap=50)
    all_chunks.extend(chunks)
    print(f"📄 {doc['title']}: {len(chunks)} chunks")

print(f"\n✅ Total chunks: {len(all_chunks)}")
```

## Part 4: Create Vector Store

```python
def create_embeddings(texts: list[str]) -> list[list[float]]:
    """Create embeddings for a list of texts."""
    
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


# Create collection
collection = chroma.create_collection(
    name="basic_rag_lab",
    metadata={"hnsw:space": "cosine"}
)

# Add chunks to vector store
chunk_texts = [chunk["content"] for chunk in all_chunks]
chunk_ids = [chunk["id"] for chunk in all_chunks]
chunk_metadatas = [
    {"document_id": chunk["document_id"], "title": chunk["title"]}
    for chunk in all_chunks
]

# Create embeddings and add to collection
print("🔄 Creating embeddings...")
embeddings = create_embeddings(chunk_texts)

collection.add(
    ids=chunk_ids,
    documents=chunk_texts,
    metadatas=chunk_metadatas,
    embeddings=embeddings
)

print(f"✅ Added {len(all_chunks)} chunks to vector store")
```

## Part 5: Retrieval Function

```python
def retrieve(query: str, k: int = 3) -> list[dict]:
    """Retrieve the most relevant chunks for a query."""
    
    # Create query embedding
    query_embedding = create_embeddings([query])[0]
    
    # Search vector store
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    
    # Format results
    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "id": results["ids"][0][i],
            "content": results["documents"][0][i],
            "title": results["metadatas"][0][i]["title"],
            "score": 1 - results["distances"][0][i]  # Convert distance to similarity
        })
    
    return retrieved


# Test retrieval
test_query = "How do I return an item?"
results = retrieve(test_query, k=3)

print(f"Query: {test_query}\n")
print("Retrieved chunks:")
for i, result in enumerate(results, 1):
    print(f"\n{i}. [{result['title']}] (score: {result['score']:.3f})")
    print(f"   {result['content'][:100]}...")
```

## Part 6: Generation with Context

```python
def generate_answer(query: str, context: list[dict]) -> str:
    """Generate an answer using retrieved context."""
    
    # Build context string
    context_str = "\n\n".join([
        f"[Source: {c['title']}]\n{c['content']}"
        for c in context
    ])
    
    # Create prompt
    system_prompt = """You are a helpful customer service assistant. 
Answer questions based ONLY on the provided context. 
If the answer isn't in the context, say "I don't have information about that."
Be concise and friendly."""
    
    user_prompt = f"""Context:
{context_str}

Question: {query}

Answer:"""
    
    # Generate response
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    
    return response.choices[0].message.content


# Test generation
query = "How do I return an item?"
context = retrieve(query, k=3)
answer = generate_answer(query, context)

print(f"Question: {query}")
print(f"\nAnswer: {answer}")
```

## Part 7: Complete RAG Pipeline

```python
def rag_query(question: str) -> dict:
    """Complete RAG pipeline: Retrieve and Generate."""
    
    # Step 1: Retrieve relevant context
    retrieved = retrieve(question, k=3)
    
    # Step 2: Generate answer
    answer = generate_answer(question, retrieved)
    
    return {
        "question": question,
        "answer": answer,
        "sources": [
            {"title": r["title"], "score": r["score"]}
            for r in retrieved
        ]
    }


# Test the complete pipeline
test_questions = [
    "What is your return policy for electronics?",
    "How much does express shipping cost?",
    "What payment methods do you accept?",
    "How do I reset my password?",
    "How do I earn loyalty points?"
]

print("=" * 60)
print("TESTING RAG PIPELINE")
print("=" * 60)

for question in test_questions:
    result = rag_query(question)
    print(f"\n❓ {result['question']}")
    print(f"💬 {result['answer']}")
    print(f"📚 Sources: {', '.join(s['title'] for s in result['sources'])}")
    print("-" * 40)
```

## Part 8: Interactive Demo

```python
def interactive_rag():
    """Interactive question-answering session."""
    
    print("\n" + "=" * 50)
    print("🤖 RAG Customer Service Assistant")
    print("=" * 50)
    print("Ask questions about returns, shipping, payments, etc.")
    print("Type 'quit' to exit.\n")
    
    while True:
        question = input("You: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! 👋")
            break
        
        if not question:
            continue
        
        result = rag_query(question)
        print(f"\n🤖 Assistant: {result['answer']}")
        print(f"   (Sources: {', '.join(s['title'] for s in result['sources'])})\n")


# Uncomment to run interactive mode:
# interactive_rag()
```

## Exercises

### Exercise 1: Add More Documents
Add at least 2 new documents to the knowledge base (e.g., warranty information, contact hours).

### Exercise 2: Improve Chunking
Modify the chunking function to split on sentence boundaries more precisely using regex or nltk.

### Exercise 3: Add Source Citations
Modify `generate_answer()` to include citations like [1], [2] that reference the sources.

### Exercise 4: Filter by Score
Add a minimum similarity score threshold (e.g., 0.5) to filter out irrelevant results.

## Challenges

### Challenge 1: Conversation Memory
Extend the RAG pipeline to handle follow-up questions by maintaining conversation history.

### Challenge 2: Hybrid Search
Implement keyword matching alongside semantic search to improve retrieval.

### Challenge 3: Streaming Response
Modify the pipeline to stream the response token by token.

## Summary

```yaml
completed:
  - "Document loading and chunking"
  - "Vector store with ChromaDB"
  - "Semantic retrieval"
  - "Answer generation with context"
  - "Complete RAG pipeline"

key_concepts:
  - "RAG = Retrieve + Generate"
  - "Chunking affects retrieval quality"
  - "Embedding similarity finds relevant context"
  - "Context grounds LLM responses"

next_steps:
  - "Document Processing module for advanced chunking"
  - "Retrieval Strategies for better search"
  - "Advanced RAG patterns (HyDE, RAG Fusion)"
```

## What's Next?

Now that you've built a basic RAG pipeline, explore:
- **Document Processing** - Handle PDFs, HTML, and complex documents
- **Retrieval Strategies** - Hybrid search, reranking, multi-query
- **RAG Chatbot Lab** - Build a full conversational system
