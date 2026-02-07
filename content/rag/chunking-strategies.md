# Chunking Strategies for RAG

How you split documents dramatically affects retrieval quality. Learn the strategies that work.

## Why Chunking Matters

```yaml
the_challenge:
  too_small: "Chunks lose context, retrieval finds fragments"
  too_large: "Chunks contain irrelevant content, waste context"
  wrong_boundaries: "Split mid-sentence, break logical units"

the_goal:
  - "Each chunk is semantically complete"
  - "Related information stays together"
  - "Chunks fit well in LLM context"
```

## Chunking Strategies Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Chunking Strategy Spectrum                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Simple ◄─────────────────────────────────────────► Complex     │
│                                                                  │
│  Fixed Size    Recursive    Sentence    Semantic    Document    │
│     │             │            │           │         Structure  │
│     ▼             ▼            ▼           ▼            ▼       │
│  Fast,        Balance      Natural     Meaning     Best for    │
│  Naive        of both      breaks      based       structured   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 1. Fixed-Size Chunking

```python
def fixed_size_chunking(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
) -> list[str]:
    """Split text into fixed-size chunks with overlap."""
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
    
    return chunks


# Example
text = "..." * 1000  # Long document
chunks = fixed_size_chunking(text, chunk_size=500, overlap=50)

# Pros:
# - Simple and fast
# - Predictable chunk sizes
# - Easy to implement

# Cons:
# - May split mid-sentence or mid-word
# - No semantic awareness
# - Can break important context
```

## 2. Recursive Character Chunking

```python
def recursive_chunking(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    separators: list[str] = None
) -> list[str]:
    """Recursively split using natural boundaries."""
    
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]
    
    chunks = []
    
    def split_text(text: str, seps: list[str]) -> list[str]:
        if not seps:
            # Base case: just split by size
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]
        
        sep = seps[0]
        remaining_seps = seps[1:]
        
        if sep:
            splits = text.split(sep)
        else:
            splits = list(text)
        
        result = []
        current = ""
        
        for split in splits:
            test_chunk = current + (sep if current else "") + split
            
            if len(test_chunk) <= chunk_size:
                current = test_chunk
            else:
                if current:
                    if len(current) <= chunk_size:
                        result.append(current)
                    else:
                        # Recursively split with next separator
                        result.extend(split_text(current, remaining_seps))
                current = split
        
        if current:
            if len(current) <= chunk_size:
                result.append(current)
            else:
                result.extend(split_text(current, remaining_seps))
        
        return result
    
    return [c.strip() for c in split_text(text, separators) if c.strip()]


# Pros:
# - Respects natural boundaries
# - More semantic than fixed-size
# - Configurable separator hierarchy

# Cons:
# - More complex logic
# - Chunk sizes can vary significantly
```

## 3. Sentence-Based Chunking

```python
import re
from typing import Iterator


def sentence_chunking(
    text: str,
    max_sentences: int = 5,
    overlap_sentences: int = 1
) -> list[str]:
    """Chunk by grouping sentences together."""
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    start = 0
    
    while start < len(sentences):
        end = min(start + max_sentences, len(sentences))
        chunk = " ".join(sentences[start:end])
        chunks.append(chunk)
        start = end - overlap_sentences
    
    return chunks


def smart_sentence_chunking(
    text: str,
    target_size: int = 500,
    overlap_sentences: int = 1
) -> list[str]:
    """Group sentences to reach target size."""
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        
        if current_size + sentence_size > target_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            # Keep overlap
            current_chunk = current_chunk[-overlap_sentences:] if overlap_sentences else []
            current_size = sum(len(s) for s in current_chunk)
        
        current_chunk.append(sentence)
        current_size += sentence_size
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


# Pros:
# - Never splits mid-sentence
# - Natural reading units
# - Good for conversational content

# Cons:
# - Sentence detection can fail
# - Chunk sizes vary a lot
# - Doesn't respect document structure
```

## 4. Semantic Chunking

```python
from openai import OpenAI
import numpy as np


class SemanticChunker:
    """Chunk based on semantic similarity."""
    
    def __init__(self, breakpoint_threshold: float = 0.3):
        self.client = OpenAI()
        self.threshold = breakpoint_threshold
    
    def chunk(self, text: str) -> list[str]:
        """Split at semantic breakpoints."""
        
        # First split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 1:
            return [text]
        
        # Get embeddings for each sentence
        embeddings = self._get_embeddings(sentences)
        
        # Find breakpoints where similarity drops
        breakpoints = self._find_breakpoints(embeddings)
        
        # Group sentences between breakpoints
        chunks = []
        start = 0
        
        for bp in breakpoints:
            chunk = " ".join(sentences[start:bp])
            if chunk:
                chunks.append(chunk)
            start = bp
        
        # Don't forget the last chunk
        if start < len(sentences):
            chunks.append(" ".join(sentences[start:]))
        
        return chunks
    
    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for texts."""
        response = self.client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        return [d.embedding for d in sorted(response.data, key=lambda x: x.index)]
    
    def _find_breakpoints(self, embeddings: list[list[float]]) -> list[int]:
        """Find indices where semantic similarity drops."""
        breakpoints = []
        
        for i in range(1, len(embeddings)):
            similarity = self._cosine_similarity(embeddings[i-1], embeddings[i])
            if similarity < self.threshold:
                breakpoints.append(i)
        
        return breakpoints
    
    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity."""
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# Usage
chunker = SemanticChunker(breakpoint_threshold=0.5)
chunks = chunker.chunk(document_text)

# Pros:
# - Semantically coherent chunks
# - Adapts to content changes
# - Best for varied content

# Cons:
# - Expensive (embedding API calls)
# - Slower than rule-based
# - Threshold tuning needed
```

## 5. Document Structure Chunking

```python
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class StructuredChunk:
    content: str
    heading: Optional[str]
    level: int
    parent_headings: list[str]


class DocumentStructureChunker:
    """Chunk based on document structure (headers, sections)."""
    
    def __init__(
        self,
        max_chunk_size: int = 1000,
        include_parent_context: bool = True
    ):
        self.max_size = max_chunk_size
        self.include_parent = include_parent_context
    
    def chunk_markdown(self, text: str) -> list[StructuredChunk]:
        """Chunk markdown by headers."""
        
        # Parse headers and content
        sections = []
        current_section = {"heading": None, "level": 0, "content": "", "parents": []}
        heading_stack = []  # Track parent headings
        
        for line in text.split("\n"):
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            
            if header_match:
                # Save current section
                if current_section["content"].strip():
                    sections.append(current_section.copy())
                
                level = len(header_match.group(1))
                heading = header_match.group(2)
                
                # Update heading stack
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                
                parent_headings = [h[1] for h in heading_stack]
                heading_stack.append((level, heading))
                
                current_section = {
                    "heading": heading,
                    "level": level,
                    "content": "",
                    "parents": parent_headings.copy()
                }
            else:
                current_section["content"] += line + "\n"
        
        # Don't forget last section
        if current_section["content"].strip():
            sections.append(current_section)
        
        # Convert to chunks, splitting large sections
        chunks = []
        for section in sections:
            content = section["content"].strip()
            
            if self.include_parent and section["parents"]:
                context = " > ".join(section["parents"])
                if section["heading"]:
                    context += f" > {section['heading']}"
            else:
                context = section["heading"] or ""
            
            if len(content) <= self.max_size:
                chunks.append(StructuredChunk(
                    content=content,
                    heading=section["heading"],
                    level=section["level"],
                    parent_headings=section["parents"]
                ))
            else:
                # Split large sections
                sub_chunks = self._split_content(content)
                for i, sub in enumerate(sub_chunks):
                    chunks.append(StructuredChunk(
                        content=sub,
                        heading=f"{section['heading']} (part {i+1})" if section["heading"] else None,
                        level=section["level"],
                        parent_headings=section["parents"]
                    ))
        
        return chunks
    
    def _split_content(self, content: str) -> list[str]:
        """Split large content into smaller parts."""
        return recursive_chunking(
            content,
            chunk_size=self.max_size,
            overlap=50
        )


# Usage
chunker = DocumentStructureChunker(
    max_chunk_size=800,
    include_parent_context=True
)
chunks = chunker.chunk_markdown(markdown_text)

for chunk in chunks:
    print(f"[{chunk.level}] {chunk.heading}")
    print(f"  Parents: {' > '.join(chunk.parent_headings)}")
    print(f"  Content: {chunk.content[:100]}...")
```

## 6. Code-Aware Chunking

```python
import ast
from dataclasses import dataclass


@dataclass
class CodeChunk:
    content: str
    chunk_type: str  # function, class, module
    name: str
    docstring: Optional[str]
    dependencies: list[str]


class CodeChunker:
    """Chunk code files by logical units."""
    
    def chunk_python(self, code: str) -> list[CodeChunk]:
        """Chunk Python code by functions and classes."""
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Fall back to simple chunking
            return [CodeChunk(
                content=code,
                chunk_type="module",
                name="module",
                docstring=None,
                dependencies=[]
            )]
        
        chunks = []
        lines = code.split("\n")
        
        # Extract imports as first chunk
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(ast.unparse(node))
        
        if imports:
            chunks.append(CodeChunk(
                content="\n".join(imports),
                chunk_type="imports",
                name="imports",
                docstring=None,
                dependencies=[]
            ))
        
        # Extract classes and functions
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                start = node.lineno - 1
                end = node.end_lineno
                content = "\n".join(lines[start:end])
                
                chunks.append(CodeChunk(
                    content=content,
                    chunk_type="class",
                    name=node.name,
                    docstring=ast.get_docstring(node),
                    dependencies=self._find_dependencies(node)
                ))
            
            elif isinstance(node, ast.FunctionDef):
                start = node.lineno - 1
                end = node.end_lineno
                content = "\n".join(lines[start:end])
                
                chunks.append(CodeChunk(
                    content=content,
                    chunk_type="function",
                    name=node.name,
                    docstring=ast.get_docstring(node),
                    dependencies=self._find_dependencies(node)
                ))
        
        return chunks
    
    def _find_dependencies(self, node: ast.AST) -> list[str]:
        """Find function/class dependencies."""
        deps = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    deps.add(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    deps.add(child.func.attr)
        return list(deps)


# Usage
code_chunker = CodeChunker()
code_chunks = code_chunker.chunk_python(python_code)

for chunk in code_chunks:
    print(f"[{chunk.chunk_type}] {chunk.name}")
    if chunk.docstring:
        print(f"  Doc: {chunk.docstring[:50]}...")
```

## Choosing the Right Strategy

```yaml
decision_guide:
  use_fixed_size:
    when:
      - "Simple text without structure"
      - "Speed is critical"
      - "Content is homogeneous"
    parameters:
      chunk_size: "400-600 characters"
      overlap: "50-100 characters (10-20%)"
  
  use_recursive:
    when:
      - "General documents"
      - "Mixed content types"
      - "Good balance needed"
    parameters:
      separators: "['\n\n', '\n', '. ', ' ']"
      chunk_size: "400-800 characters"
  
  use_sentence:
    when:
      - "Conversational content"
      - "QA pairs or FAQs"
      - "Social media content"
    parameters:
      sentences_per_chunk: "3-5"
      overlap: "1 sentence"
  
  use_semantic:
    when:
      - "Topic shifts matter"
      - "High quality needed"
      - "Cost is acceptable"
    parameters:
      threshold: "0.3-0.7 (tune for content)"
  
  use_document_structure:
    when:
      - "Markdown/structured docs"
      - "Technical documentation"
      - "Books with chapters"
    parameters:
      respect_headers: true
      include_breadcrumbs: true
  
  use_code_chunking:
    when:
      - "Source code files"
      - "Technical repos"
      - "API documentation"
```

## Hybrid Chunking

```python
class HybridChunker:
    """Combine multiple strategies based on content type."""
    
    def __init__(self):
        self.code_chunker = CodeChunker()
        self.doc_chunker = DocumentStructureChunker()
    
    def chunk(self, content: str, content_type: str) -> list[dict]:
        """Route to appropriate chunker."""
        
        if content_type in ["python", "javascript", "typescript"]:
            chunks = self.code_chunker.chunk_python(content)
            return [{"content": c.content, "type": c.chunk_type} for c in chunks]
        
        elif content_type == "markdown":
            chunks = self.doc_chunker.chunk_markdown(content)
            return [{"content": c.content, "heading": c.heading} for c in chunks]
        
        else:
            # Default to recursive
            chunks = recursive_chunking(content)
            return [{"content": c, "type": "text"} for c in chunks]
```

## Summary

```yaml
chunking_principles:
  1: "Match strategy to content type"
  2: "Preserve semantic completeness"
  3: "Include enough context"
  4: "Use overlap to maintain continuity"
  5: "Test with your actual queries"

common_mistakes:
  - "Chunks too small (lose context)"
  - "Chunks too large (irrelevant content)"
  - "Ignoring document structure"
  - "No overlap (context discontinuity)"
  - "One-size-fits-all approach"
```

## Next Steps

1. **Chunk Sizing Guide** - Optimal sizes for different use cases
2. **Document Parsing** - Advanced parsing for complex formats
3. **Chunking Lab** - Hands-on practice
