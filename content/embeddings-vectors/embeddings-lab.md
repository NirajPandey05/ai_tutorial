# Lab: Generate and Visualize Embeddings

Build an interactive embedding explorer to understand how semantic meaning maps to vector space.

## Learning Objectives

By completing this lab, you will:
- Generate embeddings using multiple providers
- Calculate similarity between texts
- Visualize embeddings in 2D/3D space
- Build a semantic search demo

## Prerequisites

```yaml
requirements:
  knowledge:
    - "Basic Python"
    - "Understanding of what embeddings are"
  
  packages:
    - "openai"
    - "numpy"
    - "scikit-learn"
    - "matplotlib"
    - "plotly (optional, for 3D)"
```

---

## Part 1: Basic Embedding Generation

### Exercise 1.1: Generate Your First Embedding

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """Generate an embedding for the given text."""
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

# Generate your first embedding
text = "The quick brown fox jumps over the lazy dog"
embedding = get_embedding(text)

print(f"Text: {text}")
print(f"Embedding dimensions: {len(embedding)}")
print(f"First 10 values: {embedding[:10]}")
print(f"Value range: [{min(embedding):.4f}, {max(embedding):.4f}]")
```

### Exercise 1.2: Batch Embeddings

```python
def get_embeddings(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """Generate embeddings for multiple texts efficiently."""
    response = client.embeddings.create(
        input=texts,
        model=model
    )
    return [item.embedding for item in response.data]

# Batch processing
sentences = [
    "I love programming in Python",
    "Python is my favorite language",
    "JavaScript is used for web development",
    "The weather is sunny today",
    "Machine learning is fascinating"
]

embeddings = get_embeddings(sentences)
print(f"Generated {len(embeddings)} embeddings")
print(f"Each has {len(embeddings[0])} dimensions")
```

---

## Part 2: Similarity Calculations

### Exercise 2.1: Implement Cosine Similarity

```python
def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.array(vec1)
    b = np.array(vec2)
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    return dot_product / (norm_a * norm_b)

# Test with known similar/dissimilar pairs
similar_texts = [
    ("I love dogs", "Dogs are my favorite pets"),
    ("The cat sat on the mat", "A feline rested on the rug"),
]

dissimilar_texts = [
    ("I love programming", "The weather is nice"),
    ("Machine learning models", "Cooking pasta recipes"),
]

print("Similar texts:")
for text1, text2 in similar_texts:
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    sim = cosine_similarity(emb1, emb2)
    print(f"  '{text1[:30]}...' ↔ '{text2[:30]}...': {sim:.4f}")

print("\nDissimilar texts:")
for text1, text2 in dissimilar_texts:
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)
    sim = cosine_similarity(emb1, emb2)
    print(f"  '{text1[:30]}...' ↔ '{text2[:30]}...': {sim:.4f}")
```

### Exercise 2.2: Build a Similarity Matrix

```python
def similarity_matrix(texts: list[str]) -> np.ndarray:
    """Create a similarity matrix for all text pairs."""
    embeddings = get_embeddings(texts)
    n = len(texts)
    matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            matrix[i][j] = cosine_similarity(embeddings[i], embeddings[j])
    
    return matrix

# Create matrix
topics = [
    "Python programming basics",
    "Learning to code in Python",
    "JavaScript web frameworks",
    "React and Vue comparison",
    "Cooking Italian pasta",
    "Making homemade pizza"
]

matrix = similarity_matrix(topics)

# Display as heatmap
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(
    matrix, 
    annot=True, 
    fmt='.2f',
    xticklabels=[t[:20] + "..." for t in topics],
    yticklabels=[t[:20] + "..." for t in topics],
    cmap='RdYlGn'
)
plt.title("Semantic Similarity Matrix")
plt.tight_layout()
plt.savefig("similarity_matrix.png")
plt.show()
```

---

## Part 3: 2D Visualization with t-SNE

### Exercise 3.1: Reduce Dimensions and Visualize

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Sample data with categories
data = {
    "programming": [
        "Python is great for data science",
        "JavaScript powers the web",
        "Learning to code takes practice",
        "Debugging is an important skill",
        "Software engineers write code"
    ],
    "food": [
        "Pizza is my favorite food",
        "Cooking pasta is relaxing",
        "Fresh vegetables are healthy",
        "Baking bread smells amazing",
        "Coffee helps me wake up"
    ],
    "sports": [
        "Soccer is played worldwide",
        "Basketball requires teamwork",
        "Running improves fitness",
        "Swimming is low impact exercise",
        "Tennis needs good reflexes"
    ]
}

# Flatten and track categories
texts = []
categories = []
for category, items in data.items():
    texts.extend(items)
    categories.extend([category] * len(items))

# Get embeddings
embeddings = get_embeddings(texts)
embeddings_array = np.array(embeddings)

# Reduce to 2D with t-SNE
tsne = TSNE(
    n_components=2, 
    random_state=42, 
    perplexity=5,
    n_iter=1000
)
embeddings_2d = tsne.fit_transform(embeddings_array)

# Plot
plt.figure(figsize=(12, 8))
colors = {'programming': 'blue', 'food': 'green', 'sports': 'red'}

for i, (text, category) in enumerate(zip(texts, categories)):
    plt.scatter(
        embeddings_2d[i, 0], 
        embeddings_2d[i, 1],
        c=colors[category],
        s=100,
        alpha=0.7
    )
    plt.annotate(
        text[:25] + "...",
        (embeddings_2d[i, 0], embeddings_2d[i, 1]),
        fontsize=8,
        alpha=0.8
    )

plt.title("Embeddings Visualized in 2D (t-SNE)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")

# Add legend
for category, color in colors.items():
    plt.scatter([], [], c=color, label=category, s=100)
plt.legend()

plt.tight_layout()
plt.savefig("tsne_visualization.png", dpi=150)
plt.show()
```

### Exercise 3.2: Interactive 3D Visualization with Plotly

```python
from sklearn.manifold import TSNE
import plotly.express as px
import pandas as pd

# Reduce to 3D
tsne_3d = TSNE(
    n_components=3, 
    random_state=42, 
    perplexity=5,
    n_iter=1000
)
embeddings_3d = tsne_3d.fit_transform(embeddings_array)

# Create DataFrame for Plotly
df = pd.DataFrame({
    'x': embeddings_3d[:, 0],
    'y': embeddings_3d[:, 1],
    'z': embeddings_3d[:, 2],
    'text': texts,
    'category': categories
})

# Create interactive 3D plot
fig = px.scatter_3d(
    df, 
    x='x', 
    y='y', 
    z='z',
    color='category',
    hover_data=['text'],
    title='3D Embedding Visualization'
)

fig.update_layout(
    scene=dict(
        xaxis_title='Dim 1',
        yaxis_title='Dim 2',
        zaxis_title='Dim 3'
    )
)

fig.write_html("embeddings_3d.html")
fig.show()
```

---

## Part 4: Semantic Search

### Exercise 4.1: Build a Simple Search Engine

```python
class SemanticSearch:
    """Simple semantic search engine."""
    
    def __init__(self):
        self.documents = []
        self.embeddings = []
    
    def add_documents(self, documents: list[str]):
        """Add documents to the search index."""
        self.documents.extend(documents)
        new_embeddings = get_embeddings(documents)
        self.embeddings.extend(new_embeddings)
        print(f"Added {len(documents)} documents. Total: {len(self.documents)}")
    
    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Search for documents similar to query."""
        query_embedding = get_embedding(query)
        
        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            sim = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((self.documents[i], sim, i))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return [(doc, score) for doc, score, _ in similarities[:top_k]]

# Create search engine
search_engine = SemanticSearch()

# Add sample documents
documents = [
    "Python is a versatile programming language",
    "Machine learning requires lots of data",
    "Neural networks are inspired by the brain",
    "Web development uses HTML, CSS, and JavaScript",
    "Data science combines statistics and programming",
    "The quick brown fox jumps over the lazy dog",
    "Natural language processing analyzes text",
    "Deep learning uses multiple neural network layers",
    "APIs allow applications to communicate",
    "Cloud computing provides scalable infrastructure"
]

search_engine.add_documents(documents)

# Test searches
queries = [
    "How do I learn to code?",
    "What is AI?",
    "Building websites"
]

for query in queries:
    print(f"\nQuery: '{query}'")
    print("-" * 50)
    results = search_engine.search(query, top_k=3)
    for doc, score in results:
        print(f"  [{score:.4f}] {doc}")
```

### Exercise 4.2: Search with Highlighting

```python
def search_with_context(
    query: str,
    search_engine: SemanticSearch,
    top_k: int = 5
) -> None:
    """Search and display results with visual similarity indicator."""
    
    results = search_engine.search(query, top_k)
    
    print(f"\n🔍 Query: '{query}'")
    print("=" * 60)
    
    for rank, (doc, score) in enumerate(results, 1):
        # Create visual similarity bar
        bar_length = int(score * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        
        # Color coding based on score
        if score > 0.8:
            indicator = "🟢"
        elif score > 0.6:
            indicator = "🟡"
        else:
            indicator = "🔴"
        
        print(f"\n{indicator} Result {rank} (similarity: {score:.4f})")
        print(f"   [{bar}]")
        print(f"   {doc}")
    
    print("\n" + "=" * 60)

# Interactive search
while True:
    query = input("\nEnter search query (or 'quit' to exit): ")
    if query.lower() == 'quit':
        break
    search_with_context(query, search_engine)
```

---

## Part 5: Embedding Exploration

### Exercise 5.1: Analogies with Embeddings

```python
def find_analogy(
    a: str, 
    b: str, 
    c: str, 
    vocabulary: list[str]
) -> list[tuple[str, float]]:
    """
    Find d such that a:b :: c:d
    (a is to b as c is to d)
    """
    # Get embeddings
    emb_a = np.array(get_embedding(a))
    emb_b = np.array(get_embedding(b))
    emb_c = np.array(get_embedding(c))
    
    # Calculate target vector: b - a + c
    target = emb_b - emb_a + emb_c
    
    # Find closest word in vocabulary
    vocab_embeddings = get_embeddings(vocabulary)
    
    similarities = []
    for word, emb in zip(vocabulary, vocab_embeddings):
        sim = cosine_similarity(target.tolist(), emb)
        similarities.append((word, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:5]

# Test analogies
vocabulary = [
    "king", "queen", "man", "woman", "boy", "girl",
    "prince", "princess", "father", "mother",
    "husband", "wife", "son", "daughter"
]

print("Analogy: man:king :: woman:?")
results = find_analogy("man", "king", "woman", vocabulary)
for word, score in results:
    print(f"  {word}: {score:.4f}")

print("\nAnalogy: father:mother :: son:?")
results = find_analogy("father", "mother", "son", vocabulary)
for word, score in results:
    print(f"  {word}: {score:.4f}")
```

### Exercise 5.2: Clustering Similar Items

```python
from sklearn.cluster import KMeans

def cluster_texts(
    texts: list[str], 
    n_clusters: int = 3
) -> dict[int, list[str]]:
    """Cluster texts based on their embeddings."""
    
    embeddings = get_embeddings(texts)
    embeddings_array = np.array(embeddings)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings_array)
    
    # Group texts by cluster
    clusters = {}
    for text, label in zip(texts, labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(text)
    
    return clusters

# Test clustering
mixed_texts = [
    # Programming
    "Python programming tutorials",
    "JavaScript web development",
    "Learning to code online",
    # Cooking
    "Delicious pasta recipes",
    "How to bake bread",
    "Italian cooking techniques",
    # Sports
    "Soccer world cup highlights",
    "Basketball training drills",
    "Olympic swimming records"
]

clusters = cluster_texts(mixed_texts, n_clusters=3)

print("Automatically discovered clusters:")
for cluster_id, texts in clusters.items():
    print(f"\nCluster {cluster_id + 1}:")
    for text in texts:
        print(f"  • {text}")
```

---

## Challenge: Build an Embedding Dashboard

Combine everything into a complete application:

```python
class EmbeddingDashboard:
    """Complete embedding exploration dashboard."""
    
    def __init__(self):
        self.texts = []
        self.embeddings = []
        self.categories = []
    
    def add_texts(self, texts: list[str], category: str = "default"):
        """Add texts with category label."""
        self.texts.extend(texts)
        self.categories.extend([category] * len(texts))
        new_embeddings = get_embeddings(texts)
        self.embeddings.extend(new_embeddings)
    
    def search(self, query: str, top_k: int = 5):
        """Semantic search."""
        query_emb = get_embedding(query)
        similarities = [
            (text, cosine_similarity(query_emb, emb))
            for text, emb in zip(self.texts, self.embeddings)
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def visualize_2d(self, output_file: str = "dashboard.png"):
        """Create 2D visualization."""
        if len(self.embeddings) < 2:
            print("Need at least 2 texts to visualize")
            return
        
        embeddings_array = np.array(self.embeddings)
        
        # t-SNE reduction
        perplexity = min(5, len(self.texts) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        reduced = tsne.fit_transform(embeddings_array)
        
        # Plot
        plt.figure(figsize=(14, 10))
        
        unique_categories = list(set(self.categories))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_categories)))
        color_map = dict(zip(unique_categories, colors))
        
        for i, (text, category) in enumerate(zip(self.texts, self.categories)):
            plt.scatter(
                reduced[i, 0], reduced[i, 1],
                c=[color_map[category]],
                s=100, alpha=0.7
            )
            plt.annotate(
                text[:30] + "..." if len(text) > 30 else text,
                (reduced[i, 0], reduced[i, 1]),
                fontsize=8
            )
        
        for category, color in color_map.items():
            plt.scatter([], [], c=[color], label=category, s=100)
        
        plt.legend()
        plt.title("Embedding Space Visualization")
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.show()
    
    def similarity_report(self):
        """Generate similarity report."""
        n = len(self.texts)
        if n < 2:
            return
        
        print("\n📊 Similarity Report")
        print("=" * 60)
        
        # Find most similar pairs
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = cosine_similarity(self.embeddings[i], self.embeddings[j])
                pairs.append((self.texts[i], self.texts[j], sim))
        
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        print("\n🔗 Most Similar Pairs:")
        for t1, t2, sim in pairs[:5]:
            print(f"  [{sim:.4f}] '{t1[:25]}...' ↔ '{t2[:25]}...'")
        
        print("\n🔀 Least Similar Pairs:")
        for t1, t2, sim in pairs[-3:]:
            print(f"  [{sim:.4f}] '{t1[:25]}...' ↔ '{t2[:25]}...'")

# Usage
dashboard = EmbeddingDashboard()

# Add categorized data
dashboard.add_texts([
    "Machine learning algorithms",
    "Deep neural networks",
    "Training AI models"
], category="AI")

dashboard.add_texts([
    "Cooking Italian pasta",
    "Baking chocolate cake",
    "Making fresh bread"
], category="Cooking")

dashboard.add_texts([
    "Running marathons",
    "Playing basketball",
    "Swimming techniques"
], category="Sports")

# Generate reports
dashboard.similarity_report()

# Search
print("\n🔍 Search Results for 'artificial intelligence':")
for text, score in dashboard.search("artificial intelligence"):
    print(f"  [{score:.4f}] {text}")

# Visualize
dashboard.visualize_2d()
```

---

## Summary

You've learned to:

✅ Generate embeddings from text
✅ Calculate cosine similarity
✅ Build similarity matrices
✅ Visualize embeddings in 2D/3D
✅ Implement semantic search
✅ Explore analogies and clustering

## Next Steps

1. **Vector Databases** - Store embeddings at scale
2. **RAG** - Use embeddings for retrieval
3. **Fine-tuning Embeddings** - Customize for your domain
