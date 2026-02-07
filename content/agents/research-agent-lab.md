# Lab: Building a Research Agent

Build a comprehensive research agent that can search the web, analyze information, and produce structured reports.

## Lab Objectives

By the end of this lab, you will:
- Build a research agent with multiple web tools
- Implement source tracking and citation
- Create structured output formats
- Handle research failures gracefully

## Prerequisites

```bash
pip install openai httpx beautifulsoup4 python-dotenv rich pydantic
```

## Project Structure

```
research_agent/
├── agent.py          # Main research agent
├── tools.py          # Web tools
├── output.py         # Output formatting
├── main.py           # Entry point
└── reports/          # Generated reports
```

## Part 1: Web Tools

### Step 1: Create Search and Read Tools (tools.py)

```python
# tools.py
import httpx
from bs4 import BeautifulSoup
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio
import re


@dataclass
class Source:
    """A source of information."""
    url: str
    title: str
    content: str
    fetched_at: datetime = field(default_factory=datetime.now)
    
    def to_citation(self, index: int) -> str:
        return f"[{index}] {self.title} - {self.url}"


@dataclass
class SearchResult:
    """A search result."""
    title: str
    url: str
    snippet: str


class WebTools:
    """Collection of web research tools."""
    
    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.sources: List[Source] = []  # Track all sources
    
    async def search(
        self,
        query: str,
        num_results: int = 5
    ) -> List[SearchResult]:
        """Search the web using DuckDuckGo."""
        
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": query},
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    },
                    timeout=self.timeout
                )
                
                soup = BeautifulSoup(response.text, "html.parser")
                results = []
                
                for result in soup.select(".result")[:num_results]:
                    title_elem = result.select_one(".result__title")
                    url_elem = result.select_one(".result__url")
                    snippet_elem = result.select_one(".result__snippet")
                    
                    if title_elem and url_elem:
                        # Get actual URL
                        link = result.select_one("a.result__a")
                        url = link.get("href") if link else url_elem.get_text(strip=True)
                        
                        results.append(SearchResult(
                            title=title_elem.get_text(strip=True),
                            url=url,
                            snippet=snippet_elem.get_text(strip=True) if snippet_elem else ""
                        ))
                
                return results
                
            except Exception as e:
                print(f"Search error: {e}")
                return []
    
    async def read_page(
        self,
        url: str,
        max_length: int = 5000
    ) -> Optional[Source]:
        """Read and extract content from a webpage."""
        
        async with httpx.AsyncClient(follow_redirects=True) as client:
            try:
                response = await client.get(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (compatible; ResearchAgent/1.0)"
                    },
                    timeout=self.timeout
                )
                
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, "html.parser")
                
                # Extract title
                title = soup.find("title")
                title_text = title.get_text(strip=True) if title else url
                
                # Remove unwanted elements
                for elem in soup(["script", "style", "nav", "header", 
                                 "footer", "aside", "form", "iframe"]):
                    elem.decompose()
                
                # Find main content
                main = (
                    soup.find("main") or
                    soup.find("article") or
                    soup.find(id=re.compile(r"content|main|article", re.I)) or
                    soup.find(class_=re.compile(r"content|main|article", re.I)) or
                    soup.body
                )
                
                if main:
                    text = main.get_text(separator="\n", strip=True)
                    # Clean up whitespace
                    text = re.sub(r'\n{3,}', '\n\n', text)
                    text = text[:max_length]
                else:
                    text = "Could not extract content"
                
                source = Source(
                    url=str(response.url),
                    title=title_text,
                    content=text
                )
                
                self.sources.append(source)
                return source
                
            except Exception as e:
                print(f"Read error for {url}: {e}")
                return None
    
    async def search_and_read(
        self,
        query: str,
        num_sources: int = 3
    ) -> List[Source]:
        """Search and read top results."""
        
        results = await self.search(query, num_results=num_sources + 2)
        
        sources = []
        for result in results:
            if len(sources) >= num_sources:
                break
            
            source = await self.read_page(result.url)
            if source and len(source.content) > 100:
                sources.append(source)
        
        return sources
    
    def get_all_sources(self) -> List[Source]:
        """Get all collected sources."""
        return self.sources
    
    def clear_sources(self):
        """Clear collected sources."""
        self.sources = []
```

## Part 2: Output Formatting

### Step 2: Create Output Formatters (output.py)

```python
# output.py
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from tools import Source


@dataclass
class ResearchReport:
    """A structured research report."""
    topic: str
    summary: str
    sections: List[Dict[str, str]]
    sources: List[Source]
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def to_markdown(self) -> str:
        """Convert report to Markdown format."""
        
        lines = [
            f"# Research Report: {self.topic}",
            f"*Generated: {self.created_at.strftime('%Y-%m-%d %H:%M')}*",
            "",
            "## Summary",
            self.summary,
            ""
        ]
        
        for section in self.sections:
            lines.append(f"## {section['title']}")
            lines.append(section['content'])
            lines.append("")
        
        lines.append("## Sources")
        for i, source in enumerate(self.sources, 1):
            lines.append(f"{i}. [{source.title}]({source.url})")
        
        return "\n".join(lines)
    
    def to_json(self) -> str:
        """Convert report to JSON format."""
        
        return json.dumps({
            "topic": self.topic,
            "summary": self.summary,
            "sections": self.sections,
            "sources": [
                {"title": s.title, "url": s.url}
                for s in self.sources
            ],
            "created_at": self.created_at.isoformat()
        }, indent=2)
    
    def to_html(self) -> str:
        """Convert report to HTML format."""
        
        sections_html = "".join([
            f"<section><h2>{s['title']}</h2><p>{s['content']}</p></section>"
            for s in self.sections
        ])
        
        sources_html = "".join([
            f'<li><a href="{s.url}">{s.title}</a></li>'
            for s in self.sources
        ])
        
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Research: {self.topic}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; border-bottom: 1px solid #ddd; }}
        .meta {{ color: #888; font-size: 0.9em; }}
        .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>Research Report: {self.topic}</h1>
    <p class="meta">Generated: {self.created_at.strftime('%Y-%m-%d %H:%M')}</p>
    
    <div class="summary">
        <h2>Summary</h2>
        <p>{self.summary}</p>
    </div>
    
    {sections_html}
    
    <section>
        <h2>Sources</h2>
        <ol>{sources_html}</ol>
    </section>
</body>
</html>
"""


class OutputFormatter:
    """Format research output in various formats."""
    
    @staticmethod
    def bullet_list(items: List[str]) -> str:
        """Format as bullet list."""
        return "\n".join([f"• {item}" for item in items])
    
    @staticmethod
    def numbered_list(items: List[str]) -> str:
        """Format as numbered list."""
        return "\n".join([f"{i}. {item}" for i, item in enumerate(items, 1)])
    
    @staticmethod
    def table(headers: List[str], rows: List[List[str]]) -> str:
        """Format as Markdown table."""
        
        # Header row
        header = "| " + " | ".join(headers) + " |"
        separator = "| " + " | ".join(["---"] * len(headers)) + " |"
        
        # Data rows
        data_rows = []
        for row in rows:
            data_rows.append("| " + " | ".join(row) + " |")
        
        return "\n".join([header, separator] + data_rows)
```

## Part 3: The Research Agent

### Step 3: Implement the Agent (agent.py)

```python
# agent.py
import asyncio
import json
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from tools import WebTools, Source, SearchResult
from output import ResearchReport, OutputFormatter


class ResearchAgent:
    """Agent specialized in conducting research."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        max_iterations: int = 10
    ):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.max_iterations = max_iterations
        self.web_tools = WebTools()
        self.console = Console()
        
        self.system_prompt = """You are a research agent that conducts thorough research on topics.

Available Tools:
- search(query, num_results): Search the web
- read_page(url): Read content from a URL
- search_and_read(query, num_sources): Search and read top results

Research Strategy:
1. Start with broad searches to understand the topic
2. Read multiple sources for different perspectives
3. Follow up with specific searches for details
4. Cross-reference information across sources
5. Synthesize findings into structured insights

Always:
- Use multiple sources (minimum 3)
- Note conflicting information
- Cite sources for key facts
- Distinguish facts from opinions"""
    
    async def research(
        self,
        topic: str,
        depth: str = "standard",
        output_format: str = "markdown"
    ) -> ResearchReport:
        """Conduct research on a topic."""
        
        self.console.print(Panel(
            f"[bold]Researching:[/bold] {topic}\n[dim]Depth: {depth}[/dim]",
            title="🔬 Research Agent"
        ))
        
        # Clear previous sources
        self.web_tools.clear_sources()
        
        # Determine research scope
        num_sources = {"quick": 2, "standard": 4, "deep": 6}.get(depth, 4)
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._build_research_prompt(topic, num_sources)}
        ]
        
        tools = self._get_tool_definitions()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Researching...", total=None)
            
            for i in range(self.max_iterations):
                progress.update(task, description=f"Research iteration {i+1}...")
                
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto"
                )
                
                message = response.choices[0].message
                
                if not message.tool_calls:
                    # Research complete, parse the report
                    progress.update(task, description="Generating report...")
                    return self._parse_report(
                        topic,
                        message.content,
                        self.web_tools.get_all_sources()
                    )
                
                messages.append(message)
                
                # Execute tool calls
                for tool_call in message.tool_calls:
                    result = await self._execute_tool(
                        tool_call.function.name,
                        json.loads(tool_call.function.arguments),
                        progress
                    )
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, default=str)
                    })
        
        # Max iterations - generate best effort report
        return self._parse_report(
            topic,
            "Research incomplete - max iterations reached",
            self.web_tools.get_all_sources()
        )
    
    def _build_research_prompt(self, topic: str, num_sources: int) -> str:
        """Build the research task prompt."""
        
        return f"""Research Topic: {topic}

Requirements:
1. Find and read at least {num_sources} authoritative sources
2. Cover different aspects/perspectives of the topic
3. Identify key facts, statistics, and insights
4. Note any controversies or conflicting information

After gathering information, provide a structured report with:
1. Executive Summary (2-3 sentences)
2. Key Findings (bullet points)
3. Detailed Analysis (organized by subtopic)
4. Conclusions

Begin your research now."""
    
    async def _execute_tool(
        self,
        name: str,
        args: Dict[str, Any],
        progress: Progress
    ) -> Any:
        """Execute a research tool."""
        
        if name == "search":
            progress.update(
                progress.task_ids[0],
                description=f"Searching: {args.get('query', '')[:30]}..."
            )
            results = await self.web_tools.search(**args)
            return [
                {"title": r.title, "url": r.url, "snippet": r.snippet}
                for r in results
            ]
        
        elif name == "read_page":
            progress.update(
                progress.task_ids[0],
                description=f"Reading: {args.get('url', '')[:40]}..."
            )
            source = await self.web_tools.read_page(**args)
            if source:
                return {
                    "title": source.title,
                    "url": source.url,
                    "content": source.content[:3000]  # Limit for context
                }
            return {"error": "Failed to read page"}
        
        elif name == "search_and_read":
            progress.update(
                progress.task_ids[0],
                description=f"Searching & reading: {args.get('query', '')[:30]}..."
            )
            sources = await self.web_tools.search_and_read(**args)
            return [
                {
                    "title": s.title,
                    "url": s.url,
                    "content": s.content[:2000]
                }
                for s in sources
            ]
        
        return {"error": f"Unknown tool: {name}"}
    
    def _parse_report(
        self,
        topic: str,
        content: str,
        sources: List[Source]
    ) -> ResearchReport:
        """Parse LLM output into structured report."""
        
        # Extract sections from content
        sections = []
        
        # Try to find markdown headers
        import re
        header_pattern = r'^##\s+(.+)$'
        parts = re.split(header_pattern, content, flags=re.MULTILINE)
        
        if len(parts) > 1:
            # Skip the first part (before any headers or summary)
            i = 1
            while i < len(parts) - 1:
                title = parts[i].strip()
                body = parts[i + 1].strip()
                
                if title.lower() not in ['sources', 'references']:
                    sections.append({
                        "title": title,
                        "content": body
                    })
                i += 2
        
        # Extract summary
        summary_match = re.search(
            r'(?:summary|executive summary)[:\s]*(.+?)(?=##|\Z)',
            content,
            re.IGNORECASE | re.DOTALL
        )
        summary = summary_match.group(1).strip() if summary_match else content[:500]
        
        return ResearchReport(
            topic=topic,
            summary=summary,
            sections=sections if sections else [{"title": "Research Findings", "content": content}],
            sources=sources
        )
    
    def _get_tool_definitions(self) -> List[dict]:
        """Get OpenAI tool definitions."""
        
        return [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web for information on a topic",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "num_results": {
                                "type": "integer",
                                "description": "Number of results (1-10)",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_page",
                    "description": "Read and extract content from a webpage URL",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "URL to read"
                            }
                        },
                        "required": ["url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_and_read",
                    "description": "Search for a topic and read the top results",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "num_sources": {
                                "type": "integer",
                                "description": "Number of sources to read",
                                "default": 3
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
```

## Part 4: Main Application

### Step 4: Create Entry Point (main.py)

```python
# main.py
import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel

from agent import ResearchAgent


async def interactive_research():
    """Run interactive research session."""
    
    load_dotenv()
    console = Console()
    
    console.print(Panel(
        "[bold cyan]🔬 Research Agent[/bold cyan]\n"
        "Conduct thorough research on any topic with AI assistance.",
        title="Welcome"
    ))
    
    agent = ResearchAgent(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4"
    )
    
    while True:
        console.print()
        topic = Prompt.ask("[bold]Enter research topic[/bold] (or 'quit' to exit)")
        
        if topic.lower() == 'quit':
            break
        
        depth = Prompt.ask(
            "Research depth",
            choices=["quick", "standard", "deep"],
            default="standard"
        )
        
        try:
            report = await agent.research(topic, depth=depth)
            
            console.print()
            console.print(Panel(
                report.to_markdown(),
                title=f"📋 Research Report: {topic}",
                border_style="green"
            ))
            
            # Save option
            if Confirm.ask("Save report to file?"):
                filename = f"reports/{topic.replace(' ', '_')}_{datetime.now():%Y%m%d_%H%M}.md"
                os.makedirs("reports", exist_ok=True)
                
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(report.to_markdown())
                
                console.print(f"[green]Saved to {filename}[/green]")
                
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    console.print("[yellow]Thank you for using Research Agent![/yellow]")


async def batch_research(topics: list[str], output_dir: str = "reports"):
    """Research multiple topics in batch."""
    
    load_dotenv()
    console = Console()
    
    os.makedirs(output_dir, exist_ok=True)
    
    agent = ResearchAgent(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4"
    )
    
    results = []
    
    for topic in topics:
        console.print(f"\n[bold]Researching: {topic}[/bold]")
        
        try:
            report = await agent.research(topic, depth="standard")
            
            filename = f"{output_dir}/{topic.replace(' ', '_')}.md"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(report.to_markdown())
            
            results.append({"topic": topic, "status": "success", "file": filename})
            console.print(f"[green]✓ Completed: {topic}[/green]")
            
        except Exception as e:
            results.append({"topic": topic, "status": "error", "error": str(e)})
            console.print(f"[red]✗ Failed: {topic} - {e}[/red]")
    
    # Summary
    console.print("\n[bold]Batch Research Complete[/bold]")
    success = sum(1 for r in results if r["status"] == "success")
    console.print(f"✓ Successful: {success}/{len(topics)}")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Batch mode with topics from command line
        topics = sys.argv[1:]
        asyncio.run(batch_research(topics))
    else:
        # Interactive mode
        asyncio.run(interactive_research())
```

## Running the Lab

```bash
# Interactive mode
python main.py

# Batch research
python main.py "AI trends 2024" "Quantum computing applications" "Climate tech solutions"
```

## Example Output

```markdown
# Research Report: AI trends 2024

*Generated: 2024-01-15 14:30*

## Summary
AI in 2024 is characterized by the rise of multimodal models, 
increased focus on AI safety, and widespread enterprise adoption 
of generative AI tools.

## Key Findings
- Multimodal AI models (text, image, audio) becoming mainstream
- AI regulation advancing in EU, US, and China
- Open-source AI models challenging proprietary solutions
- AI agents moving from research to production

## Market Trends
The AI market is expected to reach $500B by 2024...

## Technical Advances
Transformer architectures continue to dominate...

## Sources
1. [State of AI Report 2024](https://example.com/ai-report)
2. [McKinsey AI Survey](https://example.com/mckinsey)
3. [Nature AI Review](https://example.com/nature)
```

## Exercises

1. **Add Source Quality Scoring**: Implement a system to rate source credibility
2. **Add Fact Verification**: Cross-reference facts across multiple sources
3. **Add Follow-up Questions**: Generate and research follow-up questions
4. **Add Image Support**: Include relevant images in reports

## Summary

You've built a research agent that can:
- ✅ Search the web for information
- ✅ Read and extract page content
- ✅ Track and cite sources
- ✅ Generate structured reports
- ✅ Export in multiple formats

Next: [Code Assistant Lab](/learn/agents/agent-capabilities/code-assistant-lab) →
