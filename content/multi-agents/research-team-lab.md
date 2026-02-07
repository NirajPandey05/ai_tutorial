# Lab: Build a Research Team

In this lab, you'll build a multi-agent research team that can investigate topics from multiple angles and synthesize findings.

## Learning Objectives

By the end of this lab, you will:
- Create specialized research agents with different roles
- Implement parallel research execution
- Build a synthesis agent that combines findings
- Handle coordination between team members

## Prerequisites

- Python 3.9+
- OpenAI API key
- Completed Writer-Reviewer lab concepts

## Part 1: Project Setup

```
research_team/
├── agents/
│   ├── __init__.py
│   ├── base.py
│   ├── researcher.py
│   ├── analyst.py
│   └── synthesizer.py
├── core/
│   ├── __init__.py
│   └── team.py
├── main.py
└── requirements.txt
```

**requirements.txt:**
```
openai>=1.0.0
python-dotenv
```

## Part 2: Base Agent

**agents/base.py**:

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from openai import AsyncOpenAI

class ResearchAgent(ABC):
    """Base class for research team agents."""
    
    def __init__(
        self,
        name: str,
        llm_client: AsyncOpenAI,
        model: str = "gpt-4"
    ):
        self.name = name
        self.llm = llm_client
        self.model = model
        self.findings: List[Dict[str, Any]] = []
    
    @property
    @abstractmethod
    def role_description(self) -> str:
        """Describe this agent's role."""
        pass
    
    @property
    @abstractmethod
    def research_focus(self) -> str:
        """What this agent focuses on in research."""
        pass
    
    async def research(self, topic: str, context: str = "") -> Dict[str, Any]:
        """Conduct research on a topic."""
        
        prompt = f"""You are a {self.role_description}.

Research Topic: {topic}

{f"Additional Context: {context}" if context else ""}

Your research focus: {self.research_focus}

Conduct thorough research and provide:
1. Key findings (5-10 bullet points)
2. Important details and nuances
3. Gaps or uncertainties
4. Confidence level (low/medium/high)

Be specific and factual."""
        
        response = await self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"You are a {self.role_description}"},
                {"role": "user", "content": prompt}
            ]
        )
        
        content = response.choices[0].message.content
        
        finding = {
            "agent": self.name,
            "role": self.role_description,
            "focus": self.research_focus,
            "topic": topic,
            "findings": content
        }
        
        self.findings.append(finding)
        return finding
    
    async def follow_up(self, question: str, previous_findings: str) -> str:
        """Answer a follow-up question based on findings."""
        
        prompt = f"""Based on your previous research:

{previous_findings}

Answer this follow-up question:
{question}

Provide additional details or clarification."""
        
        response = await self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
```

## Part 3: Specialized Researchers

**agents/researcher.py**:

```python
from agents.base import ResearchAgent

class FactFinder(ResearchAgent):
    """Finds and verifies facts."""
    
    @property
    def role_description(self) -> str:
        return "Fact-Finding Research Specialist"
    
    @property
    def research_focus(self) -> str:
        return """
        - Verifiable facts and statistics
        - Dates, numbers, and concrete data
        - Primary sources when possible
        - Cross-referencing information
        """


class TrendAnalyst(ResearchAgent):
    """Analyzes trends and patterns."""
    
    @property
    def role_description(self) -> str:
        return "Trend and Pattern Analyst"
    
    @property
    def research_focus(self) -> str:
        return """
        - Historical trends and evolution
        - Current patterns and momentum
        - Future projections and forecasts
        - Comparative analysis over time
        """


class ExpertPerspective(ResearchAgent):
    """Provides expert viewpoints."""
    
    @property
    def role_description(self) -> str:
        return "Expert Opinion Researcher"
    
    @property
    def research_focus(self) -> str:
        return """
        - Expert opinions and viewpoints
        - Different schools of thought
        - Debates and controversies
        - Consensus vs. disagreement areas
        """


class PracticalApplications(ResearchAgent):
    """Focuses on practical applications."""
    
    @property
    def role_description(self) -> str:
        return "Practical Applications Researcher"
    
    @property
    def research_focus(self) -> str:
        return """
        - Real-world use cases
        - Implementation examples
        - Best practices
        - Common challenges and solutions
        """
```

**agents/analyst.py**:

```python
from agents.base import ResearchAgent
from typing import List, Dict, Any

class CriticalAnalyst(ResearchAgent):
    """Critically analyzes findings for gaps and issues."""
    
    @property
    def role_description(self) -> str:
        return "Critical Analysis Specialist"
    
    @property
    def research_focus(self) -> str:
        return """
        - Logical consistency
        - Potential biases
        - Missing perspectives
        - Methodological concerns
        """
    
    async def analyze_findings(
        self,
        findings: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Critically analyze collected findings."""
        
        findings_text = "\n\n".join(
            f"[{f['agent']} - {f['focus'][:50]}...]\n{f['findings']}"
            for f in findings
        )
        
        prompt = f"""Critically analyze these research findings:

{findings_text}

Provide:
1. Strengths of the research
2. Gaps or missing information
3. Potential biases or limitations
4. Contradictions between sources
5. Areas needing more investigation
6. Overall assessment of reliability"""
        
        response = await self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return {
            "agent": self.name,
            "analysis_type": "critical",
            "analysis": response.choices[0].message.content
        }
```

**agents/synthesizer.py**:

```python
from agents.base import ResearchAgent
from typing import List, Dict, Any

class Synthesizer(ResearchAgent):
    """Synthesizes findings into coherent output."""
    
    @property
    def role_description(self) -> str:
        return "Research Synthesis Specialist"
    
    @property
    def research_focus(self) -> str:
        return """
        - Integrating diverse findings
        - Identifying key themes
        - Creating coherent narratives
        - Highlighting important conclusions
        """
    
    async def synthesize(
        self,
        topic: str,
        findings: List[Dict[str, Any]],
        analysis: Dict[str, Any],
        output_format: str = "report"
    ) -> str:
        """Synthesize all research into final output."""
        
        findings_text = "\n\n".join(
            f"### {f['agent']} ({f['role']})\n{f['findings']}"
            for f in findings
        )
        
        format_instructions = {
            "report": "Create a comprehensive research report with sections and subsections.",
            "summary": "Create a concise executive summary (500 words max).",
            "briefing": "Create a bullet-point briefing document.",
            "article": "Create an engaging article for a general audience."
        }
        
        prompt = f"""Synthesize this research into a {output_format}:

Topic: {topic}

Research Findings:
{findings_text}

Critical Analysis:
{analysis.get('analysis', 'No analysis available')}

Instructions: {format_instructions.get(output_format, format_instructions['report'])}

Create a well-organized, coherent {output_format} that:
1. Integrates all key findings
2. Addresses noted limitations
3. Provides clear conclusions
4. Is properly structured"""
        
        response = await self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
```

## Part 4: Research Team Coordinator

**core/team.py**:

```python
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from openai import AsyncOpenAI

from agents.researcher import (
    FactFinder,
    TrendAnalyst,
    ExpertPerspective,
    PracticalApplications
)
from agents.analyst import CriticalAnalyst
from agents.synthesizer import Synthesizer

@dataclass
class ResearchResult:
    """Complete research result."""
    topic: str
    findings: List[Dict[str, Any]]
    analysis: Dict[str, Any]
    synthesis: str
    metadata: Dict[str, Any]


class ResearchTeam:
    """Coordinates a team of research agents."""
    
    def __init__(self, llm_client: AsyncOpenAI, model: str = "gpt-4"):
        self.llm = llm_client
        self.model = model
        
        # Create team members
        self.researchers = [
            FactFinder("Fact Finder", llm_client, model),
            TrendAnalyst("Trend Analyst", llm_client, model),
            ExpertPerspective("Expert Researcher", llm_client, model),
            PracticalApplications("Applications Researcher", llm_client, model)
        ]
        
        self.analyst = CriticalAnalyst("Critical Analyst", llm_client, model)
        self.synthesizer = Synthesizer("Synthesizer", llm_client, model)
    
    async def research(
        self,
        topic: str,
        context: str = "",
        output_format: str = "report",
        parallel: bool = True
    ) -> ResearchResult:
        """Conduct comprehensive research on a topic."""
        
        print(f"Starting research on: {topic}")
        print("=" * 50)
        
        # Phase 1: Parallel research
        print("\n📚 Phase 1: Gathering research...")
        if parallel:
            findings = await self._parallel_research(topic, context)
        else:
            findings = await self._sequential_research(topic, context)
        
        print(f"   Collected {len(findings)} research perspectives")
        
        # Phase 2: Critical analysis
        print("\n🔍 Phase 2: Critical analysis...")
        analysis = await self.analyst.analyze_findings(findings)
        print("   Analysis complete")
        
        # Phase 3: Synthesis
        print(f"\n📝 Phase 3: Synthesizing into {output_format}...")
        synthesis = await self.synthesizer.synthesize(
            topic, findings, analysis, output_format
        )
        print("   Synthesis complete")
        
        return ResearchResult(
            topic=topic,
            findings=findings,
            analysis=analysis,
            synthesis=synthesis,
            metadata={
                "researchers": [r.name for r in self.researchers],
                "output_format": output_format,
                "parallel": parallel
            }
        )
    
    async def _parallel_research(
        self,
        topic: str,
        context: str
    ) -> List[Dict[str, Any]]:
        """Research in parallel with all researchers."""
        
        tasks = [
            researcher.research(topic, context)
            for researcher in self.researchers
        ]
        
        findings = await asyncio.gather(*tasks)
        return list(findings)
    
    async def _sequential_research(
        self,
        topic: str,
        context: str
    ) -> List[Dict[str, Any]]:
        """Research sequentially, building on each other."""
        
        findings = []
        accumulated_context = context
        
        for researcher in self.researchers:
            # Include previous findings as context
            finding = await researcher.research(topic, accumulated_context)
            findings.append(finding)
            
            # Add to accumulated context
            accumulated_context += f"\n\nPrevious research ({researcher.name}):\n"
            accumulated_context += finding["findings"][:500]  # Truncate to avoid token limits
        
        return findings
    
    async def deep_dive(
        self,
        topic: str,
        aspect: str,
        initial_result: ResearchResult
    ) -> Dict[str, Any]:
        """Do a deep dive on a specific aspect."""
        
        # Find most relevant researcher
        prompt = f"""Which research perspective is most relevant for a deep dive on:
Aspect: {aspect}

Available researchers:
{chr(10).join(f"- {r.name}: {r.research_focus[:100]}" for r in self.researchers)}

Return only the researcher name."""
        
        response = await self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        researcher_name = response.choices[0].message.content.strip()
        
        # Find researcher
        researcher = next(
            (r for r in self.researchers if r.name.lower() in researcher_name.lower()),
            self.researchers[0]
        )
        
        # Deep dive with context from initial research
        context = f"Building on initial research:\n{initial_result.synthesis[:1000]}"
        
        return await researcher.research(f"{topic} - Deep dive: {aspect}", context)
```

## Part 5: Main Application

**main.py**:

```python
import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

from core.team import ResearchTeam

load_dotenv()

async def main():
    """Run the research team."""
    
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    team = ResearchTeam(client)
    
    # Define research topic
    topic = "The impact of AI on software development practices"
    
    print("🔬 RESEARCH TEAM")
    print("=" * 60)
    print(f"Topic: {topic}")
    print("=" * 60)
    
    # Conduct research
    result = await team.research(
        topic=topic,
        context="Focus on practical impacts for professional developers",
        output_format="report",
        parallel=True
    )
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 RESEARCH COMPLETE")
    print("=" * 60)
    
    print("\n📋 Individual Findings:")
    print("-" * 40)
    for finding in result.findings:
        print(f"\n[{finding['agent']}]")
        print(finding['findings'][:300] + "...")
    
    print("\n\n🔍 Critical Analysis:")
    print("-" * 40)
    print(result.analysis['analysis'][:500] + "...")
    
    print("\n\n📄 Final Report:")
    print("-" * 40)
    print(result.synthesis)
    
    # Optional: Deep dive
    print("\n" + "=" * 60)
    deep_dive = await team.deep_dive(
        topic=topic,
        aspect="AI code assistants like Copilot",
        initial_result=result
    )
    
    print("\n🔎 Deep Dive: AI Code Assistants")
    print("-" * 40)
    print(deep_dive['findings'])


async def interactive_mode():
    """Run in interactive mode."""
    
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    team = ResearchTeam(client)
    
    print("🔬 Research Team Interactive Mode")
    print("Commands: 'research <topic>', 'deep <aspect>', 'quit'")
    print("-" * 40)
    
    current_result = None
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if user_input.lower() == 'quit':
                break
            
            if user_input.lower().startswith('research '):
                topic = user_input[9:]
                current_result = await team.research(topic, output_format="summary")
                print("\n" + current_result.synthesis)
            
            elif user_input.lower().startswith('deep ') and current_result:
                aspect = user_input[5:]
                deep_dive = await team.deep_dive(
                    current_result.topic,
                    aspect,
                    current_result
                )
                print("\n" + deep_dive['findings'])
            
            else:
                print("Unknown command. Try 'research <topic>' or 'deep <aspect>'")
        
        except KeyboardInterrupt:
            break
    
    print("\nGoodbye!")


if __name__ == "__main__":
    import sys
    
    if "--interactive" in sys.argv:
        asyncio.run(interactive_mode())
    else:
        asyncio.run(main())
```

## Part 6: Run the Lab

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY=your-key-here

# Run standard mode
python main.py

# Or run interactive mode
python main.py --interactive
```

## Expected Output

```
🔬 RESEARCH TEAM
============================================================
Topic: The impact of AI on software development practices
============================================================

📚 Phase 1: Gathering research...
   Collected 4 research perspectives

🔍 Phase 2: Critical analysis...
   Analysis complete

📝 Phase 3: Synthesizing into report...
   Synthesis complete

============================================================
📊 RESEARCH COMPLETE
============================================================

📋 Individual Findings:
----------------------------------------

[Fact Finder]
Key findings on AI in software development:
• 97% of developers have used AI coding tools (2024 Stack Overflow survey)
• GitHub Copilot has 1.3 million paying subscribers
...

[Trend Analyst]
Historical and emerging trends:
• 2020-2021: Initial AI code completion tools emerge
• 2022-2023: Rapid adoption of AI assistants
...

📄 Final Report:
----------------------------------------
# The Impact of AI on Software Development Practices

## Executive Summary
...
```

## Challenges

1. **Add a Fact Checker**: Create an agent that verifies claims made by other researchers.

2. **Implement Debate**: Have researchers debate conflicting findings before synthesis.

3. **Add Sources**: Track and cite sources for all research findings.

4. **Quality Scoring**: Implement a scoring system for research quality.

## Summary

You've built a complete research team with:
- ✅ Specialized research agents (4 different focuses)
- ✅ Critical analysis agent
- ✅ Synthesis agent
- ✅ Parallel execution
- ✅ Deep dive capability
- ✅ Interactive mode

This pattern extends to many domains: market research, competitive analysis, technical investigation, and more.

Next: [Coding Team Lab](/learn/multi-agents/advanced-multi-agent/coding-team) →
