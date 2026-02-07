# What Are Embeddings?

Embeddings are numerical representations of data (text, images, audio) that capture semantic meaning in a format computers can process efficiently. They are the foundation of modern AI applications like search, recommendation systems, and semantic understanding.

## The Core Concept

```yaml
embedding_basics:
  definition: "Dense vector representations of data"
  purpose: "Convert meaning into numbers for computation"
  
  key_insight: |
    Similar concepts have similar embeddings.
    "cat" and "kitten" are closer in vector space
    than "cat" and "automobile"
  
  use_cases:
    - "Semantic search (finding similar documents)"
    - "RAG systems (retrieving relevant context)"
    - "Recommendation engines"
    - "Clustering and classification"
    - "Similarity comparisons"
```

## From Text to Numbers

```
Traditional Approach (One-Hot Encoding):
┌─────────────────────────────────────────────────────────┐
│  cat    = [1, 0, 0, 0, 0, 0, ...]  (10,000+ dimensions) │
│  kitten = [0, 1, 0, 0, 0, 0, ...]  (sparse, no meaning) │
│  dog    = [0, 0, 1, 0, 0, 0, ...]                        │
└─────────────────────────────────────────────────────────┘
Problem: All words are equally distant from each other!

Embeddings Approach:
┌─────────────────────────────────────────────────────────┐
│  cat    = [0.2, 0.8, -0.3, 0.5, ...]  (256-3072 dims)  │
│  kitten = [0.25, 0.75, -0.28, 0.52, ...] (similar!)    │
│  dog    = [0.3, 0.7, -0.2, 0.4, ...]   (related)       │
│  car    = [-0.5, 0.1, 0.8, -0.3, ...]  (different)     │
└─────────────────────────────────────────────────────────┘
Benefit: Captures semantic relationships!
```

## How Embeddings Work

```python
from openai import OpenAI

client = OpenAI()

def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """Convert text to an embedding vector."""
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

# Generate embeddings
cat_embedding = get_embedding("cat")
kitten_embedding = get_embedding("kitten")
car_embedding = get_embedding("car")

print(f"Embedding dimensions: {len(cat_embedding)}")  # 1536 for text-embedding-3-small
print(f"First 5 values: {cat_embedding[:5]}")
```

## Measuring Similarity

The magic of embeddings: similar meanings → similar vectors!

```python
import numpy as np

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.array(vec1)
    b = np.array(vec2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Compare semantic similarity
words = {
    "cat": get_embedding("cat"),
    "kitten": get_embedding("kitten"),
    "dog": get_embedding("dog"),
    "car": get_embedding("car"),
    "automobile": get_embedding("automobile")
}

# Similarity matrix
print("Cosine Similarities:")
print("-" * 40)
print(f"cat ↔ kitten:     {cosine_similarity(words['cat'], words['kitten']):.4f}")
print(f"cat ↔ dog:        {cosine_similarity(words['cat'], words['dog']):.4f}")
print(f"cat ↔ car:        {cosine_similarity(words['cat'], words['car']):.4f}")
print(f"car ↔ automobile: {cosine_similarity(words['car'], words['automobile']):.4f}")

# Expected output:
# cat ↔ kitten:     0.8934  (very similar - same animal)
# cat ↔ dog:        0.8012  (related - both pets)
# cat ↔ car:        0.4521  (unrelated)
# car ↔ automobile: 0.9456  (synonyms - very similar!)
```

## Visual Representation

```
3D Projection of Embedding Space:

                    Animals
                       ↑
                       │    • kitten
                       │  • cat
                       │      • dog
                       │        • puppy
                       │
  Vehicles ──────────────────────────── Abstract
                       │
             • car     │
           • truck     │
         • automobile  │
                       │
                       ↓
                    Objects
```

## What Embeddings Capture

```yaml
semantic_relationships:
  synonyms:
    - "car" ≈ "automobile"
    - "happy" ≈ "joyful"
    - "big" ≈ "large"
  
  analogies:
    - "king - man + woman ≈ queen"
    - "Paris - France + Italy ≈ Rome"
    - "doctor - hospital + school ≈ teacher"
  
  categories:
    - Animals cluster together
    - Professions cluster together
    - Emotions cluster together
  
  context:
    - "bank" (financial) vs "bank" (river) - different embeddings!
    - Modern embeddings handle polysemy
```

## Embedding Dimensions

```yaml
dimension_tradeoffs:
  smaller_dimensions:
    examples: "256, 384, 512"
    pros:
      - "Faster computation"
      - "Less storage"
      - "Lower cost"
    cons:
      - "Less nuance"
      - "May miss subtle meanings"
    use_case: "Simple search, resource-constrained"
  
  larger_dimensions:
    examples: "1536, 3072"
    pros:
      - "More nuance"
      - "Better for complex tasks"
      - "Captures subtle relationships"
    cons:
      - "Slower computation"
      - "More storage"
      - "Higher cost"
    use_case: "RAG, semantic analysis, high-quality search"
```

## Types of Embeddings

### 1. Word Embeddings (Legacy)
```python
# Word2Vec style - one vector per word
# Problem: "bank" has same embedding regardless of context

# Historical but foundational
word_vectors = {
    "bank": [0.1, 0.2, ...],  # Same for river bank AND financial bank
    "account": [0.3, 0.4, ...],
}
```

### 2. Sentence/Document Embeddings (Modern)
```python
# Context-aware embeddings
# Same word, different meanings → different embeddings

bank_financial = get_embedding("I need to visit the bank to deposit money")
bank_river = get_embedding("We had a picnic on the river bank")

print(f"Same word, different context similarity: {cosine_similarity(bank_financial, bank_river):.4f}")
# Output: ~0.65 (lower because different meanings!)
```

### 3. Multi-Modal Embeddings
```python
# CLIP-style: images and text in same space
# "A photo of a cat" embedding ≈ actual cat image embedding

from openai import OpenAI

# Both modalities map to same vector space
text_embedding = get_embedding("a fluffy orange cat sleeping")
# image_embedding = get_image_embedding(cat_photo)  # Same API
# cosine_similarity(text_embedding, image_embedding) → high!
```

## Practical Example: Document Search

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

# Sample documents
documents = [
    "Python is a versatile programming language used for web development, data science, and AI.",
    "Machine learning models require large datasets for training.",
    "The Eiffel Tower is located in Paris, France.",
    "Neural networks are inspired by the human brain's structure.",
    "The best pizza in New York is found in Brooklyn.",
]

# Generate embeddings for all documents
def embed_documents(docs: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        input=docs,
        model="text-embedding-3-small"
    )
    return [item.embedding for item in response.data]

doc_embeddings = embed_documents(documents)

# Search function
def search(query: str, top_k: int = 3) -> list[tuple[str, float]]:
    """Find most similar documents to query."""
    query_embedding = get_embedding(query)
    
    similarities = []
    for doc, emb in zip(documents, doc_embeddings):
        sim = cosine_similarity(query_embedding, emb)
        similarities.append((doc, sim))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# Test search
results = search("How do AI systems learn?")
print("Query: 'How do AI systems learn?'")
print("-" * 50)
for doc, score in results:
    print(f"[{score:.4f}] {doc[:60]}...")
```

## Key Takeaways

```yaml
summary:
  what: "Dense vector representations capturing semantic meaning"
  why:
    - "Enable semantic search (find by meaning, not keywords)"
    - "Power RAG systems"
    - "Enable similarity comparisons"
    - "Foundation for AI applications"
  
  how:
    - "Text → Embedding model → Vector"
    - "Similar meaning → Similar vectors"
    - "Compare with cosine similarity"
  
  key_properties:
    - "Dimensionality: 256 to 3072+"
    - "Similar items cluster together"
    - "Context-aware (modern models)"
    - "Cross-modal possible (text ↔ image)"
```

## Next Steps

1. **Embedding Models** - Compare different embedding providers
2. **Vector Databases** - Store and query embeddings efficiently
3. **RAG** - Use embeddings for retrieval-augmented generation
