# Semantic Meaning in Vector Space

Understanding how embeddings capture and represent meaning geometrically is key to building effective AI applications.

## The Geometry of Meaning

```yaml
core_concept:
  principle: "Semantic relationships become geometric relationships"
  implication: "We can use math to reason about meaning"
  
  mappings:
    - "Similar meaning → Close vectors"
    - "Different meaning → Distant vectors"
    - "Analogies → Vector arithmetic"
    - "Categories → Clusters"
```

## Vector Space Visualization

```
2D Projection of Semantic Space:

     ┌─────────────────────────────────────────────┐
     │                  • happy                     │
     │               • joyful                       │
     │            • excited    POSITIVE             │
     │                          EMOTIONS            │
     │                                              │
     │  • calm                                      │
     │     • peaceful              NEUTRAL          │
     │        • serene                              │
     │                                              │
     │                                              │
     │              • anxious                       │
     │           • worried    NEGATIVE              │
     │        • sad           EMOTIONS              │
     │     • depressed                              │
     └─────────────────────────────────────────────┘
```

## Distance Metrics

### Cosine Similarity (Most Common)

```python
import numpy as np

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Measures the angle between vectors.
    Range: -1 (opposite) to 1 (identical)
    Ignores magnitude, focuses on direction.
    """
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Example
vec1 = [1, 2, 3]
vec2 = [2, 4, 6]  # Same direction, different magnitude
vec3 = [-1, -2, -3]  # Opposite direction

print(f"Same direction: {cosine_similarity(vec1, vec2):.4f}")  # 1.0
print(f"Opposite: {cosine_similarity(vec1, vec3):.4f}")  # -1.0
```

### Euclidean Distance

```python
def euclidean_distance(a: list[float], b: list[float]) -> float:
    """
    Straight-line distance between points.
    Range: 0 (identical) to infinity
    Considers both direction and magnitude.
    """
    a, b = np.array(a), np.array(b)
    return np.linalg.norm(a - b)

# For normalized vectors, Euclidean and Cosine are related:
# euclidean² = 2 * (1 - cosine_similarity)
```

### Dot Product

```python
def dot_product(a: list[float], b: list[float]) -> float:
    """
    Simple and fast, but affected by magnitude.
    Often used with normalized vectors.
    """
    return np.dot(a, b)
```

### Comparison Chart

```yaml
distance_metrics:
  cosine_similarity:
    best_for: "Semantic similarity, text comparison"
    range: "-1 to 1"
    normalized: "Yes (ignores magnitude)"
    use_when: "Comparing meaning regardless of text length"
  
  euclidean_distance:
    best_for: "Absolute positioning, clustering"
    range: "0 to ∞"
    normalized: "No"
    use_when: "Magnitude matters"
  
  dot_product:
    best_for: "Fast similarity with normalized vectors"
    range: "-∞ to ∞"
    normalized: "No (but often used with normalized vectors)"
    use_when: "Speed is critical, vectors are normalized"
```

## Vector Arithmetic: Analogies

One of the most fascinating properties of embeddings:

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

def get_embedding(text: str) -> np.ndarray:
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return np.array(response.data[0].embedding)

# Famous analogy: King - Man + Woman ≈ Queen
king = get_embedding("king")
man = get_embedding("man")
woman = get_embedding("woman")
queen = get_embedding("queen")

# Vector arithmetic
result = king - man + woman

# Check similarity to "queen"
similarity = np.dot(result, queen) / (np.linalg.norm(result) * np.linalg.norm(queen))
print(f"king - man + woman ≈ queen: {similarity:.4f}")  # High similarity!
```

### More Analogies

```python
analogies = [
    # Country : Capital
    ("France", "Paris", "Japan", "Tokyo"),
    # Verb tenses
    ("walk", "walked", "run", "ran"),
    # Comparatives
    ("big", "bigger", "small", "smaller"),
    # Professions
    ("doctor", "hospital", "teacher", "school"),
]

for a, b, c, expected in analogies:
    vec_a = get_embedding(a)
    vec_b = get_embedding(b)
    vec_c = get_embedding(c)
    vec_expected = get_embedding(expected)
    
    # b - a + c should ≈ expected
    result = vec_b - vec_a + vec_c
    similarity = cosine_similarity(result.tolist(), vec_expected.tolist())
    print(f"{a}:{b} :: {c}:? → {expected} (sim: {similarity:.3f})")
```

## Clustering and Categories

Similar concepts naturally cluster together:

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Sample words from different categories
words = {
    "animals": ["cat", "dog", "elephant", "tiger", "bird"],
    "fruits": ["apple", "banana", "orange", "grape", "mango"],
    "vehicles": ["car", "truck", "bicycle", "airplane", "boat"],
    "emotions": ["happy", "sad", "angry", "excited", "calm"]
}

# Get embeddings
all_words = []
all_embeddings = []
all_labels = []

for category, word_list in words.items():
    for word in word_list:
        all_words.append(word)
        all_embeddings.append(get_embedding(word))
        all_labels.append(category)

# Cluster using K-means
embeddings_array = np.array(all_embeddings)
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(embeddings_array)

# Check if clusters match categories
from collections import Counter
for i in range(4):
    cluster_words = [w for w, c in zip(all_words, clusters) if c == i]
    cluster_labels = [l for l, c in zip(all_labels, clusters) if c == i]
    print(f"Cluster {i}: {cluster_words}")
    print(f"  Categories: {Counter(cluster_labels)}")
```

## Dimensionality Reduction for Visualization

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Reduce to 2D for visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
embeddings_2d = tsne.fit_transform(embeddings_array)

# Plot
plt.figure(figsize=(12, 8))
colors = {'animals': 'red', 'fruits': 'green', 'vehicles': 'blue', 'emotions': 'purple'}

for i, (word, label) in enumerate(zip(all_words, all_labels)):
    plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], 
                c=colors[label], s=100, alpha=0.7)
    plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                fontsize=9, alpha=0.8)

plt.title("Word Embeddings in 2D Space")
plt.legend(handles=[plt.scatter([], [], c=c, label=l) for l, c in colors.items()])
plt.show()
```

## Semantic Interpolation

Navigate between concepts smoothly:

```python
def interpolate_embeddings(
    start: str, 
    end: str, 
    steps: int = 5
) -> list[np.ndarray]:
    """Generate embeddings between two concepts."""
    start_emb = get_embedding(start)
    end_emb = get_embedding(end)
    
    interpolations = []
    for i in range(steps + 1):
        t = i / steps
        interp = (1 - t) * start_emb + t * end_emb
        interpolations.append(interp)
    
    return interpolations

# Example: Interpolate between "happy" and "sad"
steps = interpolate_embeddings("happy", "sad", steps=5)

# Find nearest word to each interpolation point
reference_words = ["happy", "joyful", "content", "neutral", "melancholy", "sad", "depressed"]
reference_embs = [get_embedding(w) for w in reference_words]

for i, step_emb in enumerate(steps):
    # Find closest reference word
    similarities = [cosine_similarity(step_emb.tolist(), ref.tolist()) 
                   for ref in reference_embs]
    closest_idx = np.argmax(similarities)
    print(f"Step {i}: closest to '{reference_words[closest_idx]}' ({similarities[closest_idx]:.3f})")
```

## Semantic Neighborhoods

Find words in the same semantic neighborhood:

```python
def find_neighbors(
    query: str,
    vocabulary: list[str],
    top_k: int = 5
) -> list[tuple[str, float]]:
    """Find semantically similar words."""
    query_emb = get_embedding(query)
    
    similarities = []
    for word in vocabulary:
        word_emb = get_embedding(word)
        sim = cosine_similarity(query_emb.tolist(), word_emb.tolist())
        similarities.append((word, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# Example vocabulary
vocab = [
    "programming", "coding", "software", "developer",
    "cooking", "baking", "chef", "kitchen",
    "painting", "drawing", "artist", "canvas",
    "running", "jogging", "marathon", "athlete"
]

# Find neighbors of "programming"
neighbors = find_neighbors("programming", vocab)
print("Neighbors of 'programming':")
for word, sim in neighbors:
    print(f"  {word}: {sim:.4f}")
```

## Practical Applications

```yaml
applications:
  semantic_search:
    description: "Find documents by meaning, not keywords"
    example: "Query: 'affordable housing' matches 'low-cost apartments'"
  
  recommendation:
    description: "Recommend similar items"
    example: "User likes 'Inception' → recommend 'Interstellar'"
  
  clustering:
    description: "Group similar items automatically"
    example: "Organize support tickets by topic"
  
  anomaly_detection:
    description: "Find outliers in semantic space"
    example: "Detect spam or off-topic content"
  
  deduplication:
    description: "Find near-duplicates"
    example: "Identify similar product listings"
```

## Key Insights

```yaml
takeaways:
  geometry_is_meaning:
    - "Distance = semantic difference"
    - "Direction = type of relationship"
    - "Clusters = categories"
  
  operations:
    - "Similarity: cosine is king"
    - "Arithmetic: analogies work!"
    - "Interpolation: smooth transitions"
  
  visualization:
    - "Use t-SNE or UMAP for 2D/3D"
    - "Clusters emerge naturally"
    - "Outliers indicate unusual concepts"
```

## Next Steps

1. **Embedding Models** - Compare providers and models
2. **Vector Databases** - Store and query at scale
3. **RAG** - Use semantic search for retrieval
