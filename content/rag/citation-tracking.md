# Citation Tracking: Source Attribution in RAG

Implement accurate source attribution to ground responses and build trust.

## Why Citation Tracking Matters

```yaml
importance:
  trust: "Users can verify information"
  accuracy: "Reduces hallucination risk"
  compliance: "Required in many domains"
  debugging: "Trace errors to source"

challenges:
  - "Mapping generated text to sources"
  - "Handling paraphrased content"
  - "Managing multiple sources per claim"
  - "Formatting citations clearly"
```

## Basic Citation Implementation

```python
from openai import OpenAI
import re


class BasicCitationRAG:
    """RAG with simple citation tracking."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def generate_with_citations(
        self,
        query: str,
        documents: list[dict]
    ) -> dict:
        """Generate answer with source citations."""
        
        # Format documents with IDs
        context_parts = []
        source_map = {}
        
        for i, doc in enumerate(documents):
            source_id = f"[{i+1}]"
            context_parts.append(f"{source_id} {doc['content']}")
            source_map[source_id] = {
                "id": doc.get("id", f"doc_{i}"),
                "content": doc["content"],
                "metadata": doc.get("metadata", {})
            }
        
        context = "\n\n".join(context_parts)
        
        # Generate with citation instructions
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """Answer questions using ONLY the provided sources.
Cite sources using [N] notation after each fact or claim.
If information comes from multiple sources, cite all: [1][2]
If you can't answer from sources, say so."""
                },
                {
                    "role": "user",
                    "content": f"""Sources:
{context}

Question: {query}"""
                }
            ],
            temperature=0
        )
        
        answer = response.choices[0].message.content
        
        # Extract cited sources
        cited_refs = set(re.findall(r'\[(\d+)\]', answer))
        
        cited_sources = [
            source_map[f"[{ref}]"]
            for ref in cited_refs
            if f"[{ref}]" in source_map
        ]
        
        return {
            "answer": answer,
            "sources": cited_sources,
            "all_sources": list(source_map.values()),
            "citation_count": len(cited_refs)
        }


# Usage
rag = BasicCitationRAG()

documents = [
    {
        "id": "policy_doc",
        "content": "Returns are accepted within 30 days of purchase with original receipt.",
        "metadata": {"title": "Return Policy", "updated": "2024-01"}
    },
    {
        "id": "faq_doc",
        "content": "Items must be in original, unopened packaging for full refund.",
        "metadata": {"title": "FAQ", "updated": "2024-03"}
    }
]

result = rag.generate_with_citations(
    "What do I need to return an item?",
    documents
)

print(f"Answer: {result['answer']}")
print(f"\nCited sources:")
for src in result['sources']:
    print(f"  - {src['metadata'].get('title', 'Unknown')}")
```

## Inline Citation Formatting

```python
class InlineCitationRAG:
    """Generate citations with multiple formats."""
    
    FORMATS = {
        "bracketed": {"prefix": "[", "suffix": "]"},
        "superscript": {"prefix": "^", "suffix": ""},
        "parenthetical": {"prefix": "(", "suffix": ")"},
        "footnote": {"prefix": "[^", "suffix": "]"}
    }
    
    def __init__(self, format: str = "bracketed"):
        self.client = OpenAI()
        self.format = self.FORMATS.get(format, self.FORMATS["bracketed"])
    
    def generate(
        self,
        query: str,
        documents: list[dict],
        include_footnotes: bool = True
    ) -> dict:
        """Generate with formatted citations."""
        
        # Prepare sources
        sources = []
        for i, doc in enumerate(documents):
            source_num = i + 1
            source_ref = f"{self.format['prefix']}{source_num}{self.format['suffix']}"
            sources.append({
                "num": source_num,
                "ref": source_ref,
                "doc": doc
            })
        
        # Build context
        context = "\n\n".join([
            f"Source {s['num']}: {s['doc']['content']}"
            for s in sources
        ])
        
        # Generate
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"""Answer using the provided sources.
Cite each fact with {self.format['prefix']}N{self.format['suffix']} where N is the source number.
Place citations immediately after the relevant claim."""
                },
                {
                    "role": "user",
                    "content": f"Sources:\n{context}\n\nQuestion: {query}"
                }
            ],
            temperature=0
        )
        
        answer = response.choices[0].message.content
        
        # Build footnotes if requested
        footnotes = ""
        if include_footnotes:
            pattern = re.escape(self.format['prefix']) + r'(\d+)' + re.escape(self.format['suffix'])
            cited_nums = set(re.findall(pattern, answer))
            
            footnote_parts = []
            for num in sorted(cited_nums, key=int):
                source = next((s for s in sources if str(s['num']) == num), None)
                if source:
                    title = source['doc'].get('metadata', {}).get('title', f'Source {num}')
                    footnote_parts.append(f"{self.format['prefix']}{num}{self.format['suffix']} {title}")
            
            if footnote_parts:
                footnotes = "\n\n---\nSources:\n" + "\n".join(footnote_parts)
        
        return {
            "answer": answer,
            "answer_with_footnotes": answer + footnotes,
            "sources_used": [s['doc'] for s in sources if str(s['num']) in cited_nums]
        }


# Usage
rag = InlineCitationRAG(format="bracketed")

result = rag.generate(
    "What is the company's vacation policy?",
    documents,
    include_footnotes=True
)

print(result['answer_with_footnotes'])
```

## Sentence-Level Citation Mapping

```python
class SentenceLevelCitation:
    """Map each sentence to its source."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def generate_with_mapping(
        self,
        query: str,
        documents: list[dict]
    ) -> dict:
        """Generate with sentence-level source mapping."""
        
        # Prepare sources with IDs
        sources = {
            f"S{i+1}": doc for i, doc in enumerate(documents)
        }
        
        context = "\n".join([
            f"{sid}: {doc['content']}"
            for sid, doc in sources.items()
        ])
        
        # Generate answer with structured citations
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """Answer the question using the provided sources.
Format your response as a JSON object with this structure:
{
  "sentences": [
    {"text": "sentence text", "sources": ["S1", "S2"]}
  ]
}
Each sentence must cite its source(s)."""
                },
                {
                    "role": "user",
                    "content": f"Sources:\n{context}\n\nQuestion: {query}"
                }
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        
        import json
        result = json.loads(response.choices[0].message.content)
        
        # Enrich with source details
        for sentence in result.get("sentences", []):
            sentence["source_details"] = [
                {
                    "id": sid,
                    "content": sources[sid]["content"][:100] + "...",
                    "metadata": sources[sid].get("metadata", {})
                }
                for sid in sentence.get("sources", [])
                if sid in sources
            ]
        
        # Create plain text version
        plain_text = " ".join(
            f"{s['text']} [{','.join(s['sources'])}]"
            for s in result.get("sentences", [])
        )
        
        return {
            "structured": result,
            "plain_text": plain_text,
            "source_coverage": self._calculate_coverage(result, sources)
        }
    
    def _calculate_coverage(self, result: dict, sources: dict) -> dict:
        """Calculate which sources were used and how much."""
        
        used_sources = set()
        for sentence in result.get("sentences", []):
            used_sources.update(sentence.get("sources", []))
        
        return {
            "total_sources": len(sources),
            "sources_used": len(used_sources),
            "unused_sources": list(set(sources.keys()) - used_sources),
            "coverage_ratio": len(used_sources) / len(sources) if sources else 0
        }


# Usage
rag = SentenceLevelCitation()

result = rag.generate_with_mapping(
    "What are the working hours and remote work policy?",
    [
        {"content": "Office hours are 9 AM to 5 PM, Monday through Friday.", "metadata": {"title": "Office Hours"}},
        {"content": "Remote work is available for all employees with manager approval.", "metadata": {"title": "Remote Policy"}},
        {"content": "Core collaboration hours are 10 AM to 3 PM.", "metadata": {"title": "Collaboration Guidelines"}}
    ]
)

print("Structured output:")
for sentence in result['structured']['sentences']:
    print(f"  '{sentence['text']}' - Sources: {sentence['sources']}")

print(f"\nCoverage: {result['source_coverage']}")
```

## Citation Verification

```python
class CitationVerifier:
    """Verify that citations are accurate."""
    
    def __init__(self):
        self.client = OpenAI()
    
    def verify_citations(
        self,
        answer: str,
        documents: list[dict]
    ) -> dict:
        """Verify each citation is supported by source."""
        
        # Extract claims with citations
        claims = self._extract_claims(answer)
        
        # Verify each claim
        verifications = []
        
        for claim in claims:
            verification = self._verify_claim(claim, documents)
            verifications.append(verification)
        
        # Calculate accuracy
        verified_count = sum(1 for v in verifications if v["verified"])
        
        return {
            "total_claims": len(claims),
            "verified_claims": verified_count,
            "accuracy": verified_count / len(claims) if claims else 1.0,
            "details": verifications
        }
    
    def _extract_claims(self, answer: str) -> list[dict]:
        """Extract claims with their citations."""
        
        # Pattern: text followed by citation(s)
        pattern = r'([^.!?]+[.!?])\s*(\[\d+\](?:\[\d+\])*)'
        matches = re.findall(pattern, answer)
        
        claims = []
        for text, citations in matches:
            citation_nums = re.findall(r'\[(\d+)\]', citations)
            claims.append({
                "text": text.strip(),
                "citations": [int(n) for n in citation_nums]
            })
        
        return claims
    
    def _verify_claim(self, claim: dict, documents: list[dict]) -> dict:
        """Verify a single claim against its cited sources."""
        
        # Get cited documents
        cited_docs = []
        for num in claim["citations"]:
            if 0 < num <= len(documents):
                cited_docs.append(documents[num - 1])
        
        if not cited_docs:
            return {
                "claim": claim["text"],
                "verified": False,
                "reason": "No valid citations"
            }
        
        # Verify with LLM
        source_text = "\n".join([d["content"] for d in cited_docs])
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """Determine if the claim is supported by the source text.
Return JSON: {"supported": boolean, "reason": "explanation"}"""
                },
                {
                    "role": "user",
                    "content": f"Claim: {claim['text']}\n\nSource: {source_text}"
                }
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        
        import json
        result = json.loads(response.choices[0].message.content)
        
        return {
            "claim": claim["text"],
            "citations": claim["citations"],
            "verified": result.get("supported", False),
            "reason": result.get("reason", "")
        }


# Usage
verifier = CitationVerifier()

# Example answer with citations
answer = """
Returns are accepted within 30 days of purchase. [1]
Items must be in original packaging to receive a full refund. [2]
Gift cards cannot be returned. [3]
"""

documents = [
    {"content": "Our return policy allows returns within 30 days of purchase."},
    {"content": "Full refunds require items in original, unopened packaging."},
    {"content": "Gift cards and final sale items are non-returnable."}
]

result = verifier.verify_citations(answer, documents)

print(f"Citation accuracy: {result['accuracy']:.0%}")
for detail in result['details']:
    status = "✓" if detail['verified'] else "✗"
    print(f"  {status} '{detail['claim'][:50]}...'")
    if not detail['verified']:
        print(f"    Reason: {detail['reason']}")
```

## Rich Citation Display

```python
class RichCitationFormatter:
    """Format citations for rich display."""
    
    def format_for_display(
        self,
        answer: str,
        sources: list[dict]
    ) -> dict:
        """Format answer with rich citation display."""
        
        # Parse citations
        parsed = self._parse_citations(answer)
        
        # Build HTML version
        html = self._to_html(parsed, sources)
        
        # Build Markdown version
        markdown = self._to_markdown(parsed, sources)
        
        # Build JSON structure
        structured = self._to_structured(parsed, sources)
        
        return {
            "html": html,
            "markdown": markdown,
            "structured": structured,
            "plain": answer
        }
    
    def _parse_citations(self, text: str) -> list[dict]:
        """Parse text into segments with citations."""
        
        segments = []
        last_end = 0
        
        for match in re.finditer(r'(\[\d+\])', text):
            # Text before citation
            if match.start() > last_end:
                segments.append({
                    "type": "text",
                    "content": text[last_end:match.start()]
                })
            
            # Citation
            segments.append({
                "type": "citation",
                "content": match.group(1),
                "num": int(re.search(r'\d+', match.group(1)).group())
            })
            
            last_end = match.end()
        
        # Remaining text
        if last_end < len(text):
            segments.append({
                "type": "text",
                "content": text[last_end:]
            })
        
        return segments
    
    def _to_html(self, segments: list[dict], sources: list[dict]) -> str:
        """Convert to HTML with hoverable citations."""
        
        html_parts = []
        
        for seg in segments:
            if seg["type"] == "text":
                html_parts.append(seg["content"])
            else:
                num = seg["num"]
                if 0 < num <= len(sources):
                    source = sources[num - 1]
                    title = source.get("metadata", {}).get("title", f"Source {num}")
                    preview = source["content"][:100]
                    
                    html_parts.append(
                        f'<sup class="citation" data-source="{num}" '
                        f'title="{title}: {preview}...">[{num}]</sup>'
                    )
                else:
                    html_parts.append(seg["content"])
        
        return "".join(html_parts)
    
    def _to_markdown(self, segments: list[dict], sources: list[dict]) -> str:
        """Convert to Markdown with footnotes."""
        
        md_parts = []
        used_sources = set()
        
        for seg in segments:
            if seg["type"] == "text":
                md_parts.append(seg["content"])
            else:
                num = seg["num"]
                md_parts.append(f"[^{num}]")
                used_sources.add(num)
        
        # Add footnotes
        md_parts.append("\n\n---\n")
        for num in sorted(used_sources):
            if 0 < num <= len(sources):
                source = sources[num - 1]
                title = source.get("metadata", {}).get("title", f"Source {num}")
                md_parts.append(f"[^{num}]: {title}\n")
        
        return "".join(md_parts)
    
    def _to_structured(self, segments: list[dict], sources: list[dict]) -> dict:
        """Convert to structured JSON."""
        
        return {
            "segments": segments,
            "sources": {
                i + 1: {
                    "content": s["content"],
                    "metadata": s.get("metadata", {})
                }
                for i, s in enumerate(sources)
            }
        }


# Usage
formatter = RichCitationFormatter()

answer = "The policy allows 30-day returns [1] with original packaging [2]."
sources = [
    {"content": "Returns accepted within 30 days.", "metadata": {"title": "Return Policy"}},
    {"content": "Items must be in original packaging.", "metadata": {"title": "Return Guidelines"}}
]

result = formatter.format_for_display(answer, sources)

print("HTML version:")
print(result['html'])
print("\nMarkdown version:")
print(result['markdown'])
```

## Summary

```yaml
citation_best_practices:
  1: "Use consistent citation format"
  2: "Cite at sentence level for accuracy"
  3: "Verify citations automatically"
  4: "Provide source previews/links"
  5: "Track citation coverage"

implementation_tips:
  - "Start with simple bracketed citations"
  - "Add verification for critical applications"
  - "Consider UI/UX for citation display"
  - "Log citation patterns for quality monitoring"

common_pitfalls:
  - "Not validating citation accuracy"
  - "Inconsistent citation formatting"
  - "Missing source metadata"
  - "Overwhelming users with too many citations"
```

## Next Steps

1. **Streaming Sources** - Real-time citation display
2. **RAG Chatbot Lab** - Build complete cited chatbot
3. **Production RAG** - Scale citation systems
