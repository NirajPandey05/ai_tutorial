# Sequential Workflows

Learn how to build multi-agent workflows where agents work in sequence, each building on the previous agent's output.

## What are Sequential Workflows?

Sequential workflows pass work through a chain of specialized agents, each performing a specific task.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Sequential Workflow                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Input                                                                  │
│     │                                                                    │
│     ▼                                                                    │
│   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐            │
│   │   Agent A   │ ───► │   Agent B   │ ───► │   Agent C   │            │
│   │  (Research) │      │   (Write)   │      │   (Edit)    │            │
│   └─────────────┘      └─────────────┘      └─────────────┘            │
│                                                     │                    │
│                                                     ▼                    │
│                                                  Output                  │
│                                                                          │
│   Each agent:                                                            │
│   • Receives input from previous stage                                  │
│   • Performs specialized task                                           │
│   • Passes result to next stage                                         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Basic Sequential Pipeline

```python
from typing import Any, List, Callable, TypeVar
from dataclasses import dataclass
from abc import ABC, abstractmethod

T = TypeVar('T')

@dataclass
class PipelineResult:
    """Result from a pipeline stage."""
    stage: str
    input: Any
    output: Any
    success: bool
    error: str = None

class PipelineStage(ABC):
    """Abstract base class for pipeline stages."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """Process input and return output."""
        pass

class SequentialPipeline:
    """Execute agents in sequence."""
    
    def __init__(self, stages: List[PipelineStage] = None):
        self.stages = stages or []
        self.results: List[PipelineResult] = []
    
    def add_stage(self, stage: PipelineStage):
        """Add a stage to the pipeline."""
        self.stages.append(stage)
        return self  # Enable chaining
    
    async def run(self, initial_input: Any) -> Any:
        """Run the pipeline with given input."""
        
        current_data = initial_input
        self.results = []
        
        for stage in self.stages:
            try:
                output = await stage.process(current_data)
                
                self.results.append(PipelineResult(
                    stage=stage.name,
                    input=current_data,
                    output=output,
                    success=True
                ))
                
                current_data = output
                
            except Exception as e:
                self.results.append(PipelineResult(
                    stage=stage.name,
                    input=current_data,
                    output=None,
                    success=False,
                    error=str(e)
                ))
                raise
        
        return current_data
    
    def get_history(self) -> List[PipelineResult]:
        """Get execution history."""
        return self.results


# Example stages
class ResearchStage(PipelineStage):
    def __init__(self, llm_client):
        super().__init__("research")
        self.llm = llm_client
    
    async def process(self, topic: str) -> dict:
        prompt = f"""Research the following topic and provide key facts:
Topic: {topic}

Provide:
1. Overview (2-3 sentences)
2. Key facts (5 bullet points)
3. Important considerations
"""
        response = await self.llm.chat(prompt)
        return {"topic": topic, "research": response}


class WritingStage(PipelineStage):
    def __init__(self, llm_client):
        super().__init__("writing")
        self.llm = llm_client
    
    async def process(self, research: dict) -> dict:
        prompt = f"""Write an article based on this research:

Topic: {research['topic']}
Research: {research['research']}

Write a well-structured article with:
- Engaging introduction
- Clear body with sections
- Conclusion
"""
        response = await self.llm.chat(prompt)
        return {"topic": research["topic"], "draft": response}


class EditingStage(PipelineStage):
    def __init__(self, llm_client):
        super().__init__("editing")
        self.llm = llm_client
    
    async def process(self, draft: dict) -> str:
        prompt = f"""Edit this article for clarity and style:

{draft['draft']}

Focus on:
- Clear, concise language
- Grammar and punctuation
- Flow and readability
- Consistent tone
"""
        response = await self.llm.chat(prompt)
        return response


# Usage
async def content_pipeline(topic: str) -> str:
    pipeline = SequentialPipeline()
    
    pipeline.add_stage(ResearchStage(llm_client))
    pipeline.add_stage(WritingStage(llm_client))
    pipeline.add_stage(EditingStage(llm_client))
    
    return await pipeline.run(topic)
```

## Advanced Sequential Patterns

### Pattern 1: Conditional Branching

```python
from typing import Callable, Dict

class ConditionalPipeline:
    """Pipeline with conditional branching."""
    
    def __init__(self):
        self.stages: List[PipelineStage] = []
        self.branches: Dict[str, List[PipelineStage]] = {}
        self.condition_checks: Dict[str, Callable] = {}
    
    def add_stage(self, stage: PipelineStage):
        """Add a stage to main path."""
        self.stages.append(stage)
        return self
    
    def add_branch(
        self,
        name: str,
        condition: Callable[[Any], bool],
        stages: List[PipelineStage]
    ):
        """Add a conditional branch."""
        self.branches[name] = stages
        self.condition_checks[name] = condition
        return self
    
    async def run(self, initial_input: Any) -> Any:
        """Run with conditional branching."""
        
        current_data = initial_input
        
        for stage in self.stages:
            current_data = await stage.process(current_data)
            
            # Check for branches
            for branch_name, condition in self.condition_checks.items():
                if condition(current_data):
                    # Execute branch
                    for branch_stage in self.branches[branch_name]:
                        current_data = await branch_stage.process(current_data)
                    break
        
        return current_data


# Example: Route based on content type
pipeline = ConditionalPipeline()

pipeline.add_stage(ClassifierStage(llm))  # Classifies content type

pipeline.add_branch(
    "technical",
    condition=lambda x: x.get("type") == "technical",
    stages=[TechnicalReviewStage(llm), CodeFormatterStage()]
)

pipeline.add_branch(
    "creative",
    condition=lambda x: x.get("type") == "creative",
    stages=[CreativeReviewStage(llm), StyleEnhancerStage(llm)]
)
```

### Pattern 2: Pipeline with Checkpoints

```python
import json
from pathlib import Path
from datetime import datetime

class CheckpointedPipeline(SequentialPipeline):
    """Pipeline that saves checkpoints for recovery."""
    
    def __init__(
        self,
        stages: List[PipelineStage] = None,
        checkpoint_dir: str = "./checkpoints"
    ):
        super().__init__(stages)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.run_id: str = None
    
    def _save_checkpoint(self, stage_index: int, data: Any):
        """Save checkpoint after a stage."""
        checkpoint = {
            "run_id": self.run_id,
            "stage_index": stage_index,
            "stage_name": self.stages[stage_index].name,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        path = self.checkpoint_dir / f"{self.run_id}_stage_{stage_index}.json"
        with open(path, "w") as f:
            json.dump(checkpoint, f, default=str)
    
    def _load_checkpoint(self, run_id: str) -> tuple[int, Any]:
        """Load the latest checkpoint for a run."""
        checkpoints = list(self.checkpoint_dir.glob(f"{run_id}_stage_*.json"))
        
        if not checkpoints:
            return -1, None
        
        # Get latest checkpoint
        latest = max(checkpoints, key=lambda p: int(p.stem.split("_")[-1]))
        
        with open(latest) as f:
            checkpoint = json.load(f)
        
        return checkpoint["stage_index"], checkpoint["data"]
    
    async def run(
        self,
        initial_input: Any,
        run_id: str = None,
        resume: bool = False
    ) -> Any:
        """Run pipeline with checkpointing."""
        
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        start_index = 0
        current_data = initial_input
        
        if resume and run_id:
            checkpoint_index, checkpoint_data = self._load_checkpoint(run_id)
            if checkpoint_index >= 0:
                start_index = checkpoint_index + 1
                current_data = checkpoint_data
                print(f"Resuming from stage {start_index}")
        
        for i, stage in enumerate(self.stages[start_index:], start=start_index):
            current_data = await stage.process(current_data)
            self._save_checkpoint(i, current_data)
        
        return current_data


# Usage with recovery
async def resilient_workflow():
    pipeline = CheckpointedPipeline([
        ResearchStage(llm),
        WritingStage(llm),
        EditingStage(llm),
    ])
    
    run_id = "article_001"
    
    try:
        result = await pipeline.run("AI in healthcare", run_id=run_id)
    except Exception as e:
        print(f"Failed at stage, will resume: {e}")
        # Resume from last checkpoint
        result = await pipeline.run(None, run_id=run_id, resume=True)
    
    return result
```

### Pattern 3: Pipeline with Feedback Loops

```python
class FeedbackPipeline:
    """Pipeline that can loop back for revisions."""
    
    def __init__(self, max_iterations: int = 3):
        self.stages: List[PipelineStage] = []
        self.validators: Dict[str, Callable] = {}
        self.max_iterations = max_iterations
    
    def add_stage(
        self,
        stage: PipelineStage,
        validator: Callable[[Any], tuple[bool, str]] = None
    ):
        """Add stage with optional validator."""
        self.stages.append(stage)
        if validator:
            self.validators[stage.name] = validator
        return self
    
    async def run(self, initial_input: Any) -> Any:
        """Run with feedback loops."""
        
        current_data = initial_input
        
        for stage in self.stages:
            iterations = 0
            
            while iterations < self.max_iterations:
                # Process
                output = await stage.process(current_data)
                
                # Validate if validator exists
                if stage.name in self.validators:
                    is_valid, feedback = self.validators[stage.name](output)
                    
                    if is_valid:
                        current_data = output
                        break
                    else:
                        # Add feedback for next iteration
                        current_data = {
                            "original": current_data,
                            "attempt": output,
                            "feedback": feedback
                        }
                        iterations += 1
                else:
                    current_data = output
                    break
            
            if iterations >= self.max_iterations:
                print(f"Warning: Max iterations reached at {stage.name}")
                current_data = output
        
        return current_data


# Example: Writing with quality checks
def quality_validator(output: str) -> tuple[bool, str]:
    """Check if writing meets quality standards."""
    issues = []
    
    if len(output) < 500:
        issues.append("Content too short, need more detail")
    
    if "conclusion" not in output.lower():
        issues.append("Missing conclusion section")
    
    if issues:
        return False, "; ".join(issues)
    return True, ""


pipeline = FeedbackPipeline(max_iterations=3)

pipeline.add_stage(WritingStage(llm), validator=quality_validator)
pipeline.add_stage(EditingStage(llm))
```

## Complete Sequential Workflow Example

```python
from openai import AsyncOpenAI
from dataclasses import dataclass
from typing import List, Dict, Any
import asyncio

@dataclass
class ContentPlan:
    topic: str
    outline: List[str]
    target_audience: str
    tone: str

@dataclass
class Research:
    facts: List[str]
    sources: List[str]
    key_points: List[str]

@dataclass
class Draft:
    title: str
    sections: List[Dict[str, str]]
    word_count: int

@dataclass
class FinalArticle:
    title: str
    content: str
    metadata: Dict[str, Any]


class ContentCreationPipeline:
    """Complete content creation pipeline."""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
    
    async def _call_llm(self, prompt: str, system: str = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        return response.choices[0].message.content
    
    async def plan(self, topic: str, requirements: str = "") -> ContentPlan:
        """Stage 1: Plan the content."""
        
        prompt = f"""Create a content plan for: {topic}
{f"Requirements: {requirements}" if requirements else ""}

Provide:
1. A detailed outline (5-7 main points)
2. Target audience
3. Recommended tone

Format as JSON:
{{"outline": [...], "target_audience": "...", "tone": "..."}}
"""
        
        response = await self._call_llm(
            prompt,
            system="You are a content strategist. Output valid JSON only."
        )
        
        import json
        data = json.loads(response)
        
        return ContentPlan(
            topic=topic,
            outline=data["outline"],
            target_audience=data["target_audience"],
            tone=data["tone"]
        )
    
    async def research(self, plan: ContentPlan) -> Research:
        """Stage 2: Research the topic."""
        
        prompt = f"""Research the following topic thoroughly:

Topic: {plan.topic}
Outline points to cover:
{chr(10).join(f"- {point}" for point in plan.outline)}

Provide:
1. Key facts (at least 10)
2. Important statistics or data points
3. Main takeaways

Format as JSON:
{{"facts": [...], "sources": [...], "key_points": [...]}}
"""
        
        response = await self._call_llm(
            prompt,
            system="You are a research specialist. Provide accurate, factual information."
        )
        
        import json
        data = json.loads(response)
        
        return Research(
            facts=data["facts"],
            sources=data.get("sources", []),
            key_points=data["key_points"]
        )
    
    async def write(self, plan: ContentPlan, research: Research) -> Draft:
        """Stage 3: Write the draft."""
        
        prompt = f"""Write an article based on this plan and research:

Topic: {plan.topic}
Target Audience: {plan.target_audience}
Tone: {plan.tone}

Outline:
{chr(10).join(f"- {point}" for point in plan.outline)}

Research Facts:
{chr(10).join(f"- {fact}" for fact in research.facts[:10])}

Key Points:
{chr(10).join(f"- {point}" for point in research.key_points)}

Write a complete article with:
- Engaging title
- Introduction
- Body sections following the outline
- Conclusion

Format as JSON:
{{"title": "...", "sections": [{{"heading": "...", "content": "..."}}]}}
"""
        
        response = await self._call_llm(
            prompt,
            system=f"You are a skilled writer. Write in a {plan.tone} tone for {plan.target_audience}."
        )
        
        import json
        data = json.loads(response)
        
        word_count = sum(
            len(s["content"].split())
            for s in data["sections"]
        )
        
        return Draft(
            title=data["title"],
            sections=data["sections"],
            word_count=word_count
        )
    
    async def edit(self, draft: Draft) -> Draft:
        """Stage 4: Edit and improve."""
        
        content = "\n\n".join(
            f"## {s['heading']}\n{s['content']}"
            for s in draft.sections
        )
        
        prompt = f"""Edit and improve this article:

Title: {draft.title}

{content}

Improve:
1. Clarity and readability
2. Grammar and punctuation
3. Flow between sections
4. Engagement and interest

Return the improved version in the same JSON format:
{{"title": "...", "sections": [{{"heading": "...", "content": "..."}}]}}
"""
        
        response = await self._call_llm(
            prompt,
            system="You are a professional editor. Improve the writing while maintaining the voice."
        )
        
        import json
        data = json.loads(response)
        
        return Draft(
            title=data["title"],
            sections=data["sections"],
            word_count=sum(len(s["content"].split()) for s in data["sections"])
        )
    
    async def finalize(self, draft: Draft, plan: ContentPlan) -> FinalArticle:
        """Stage 5: Create final output."""
        
        content_parts = [f"# {draft.title}\n"]
        
        for section in draft.sections:
            content_parts.append(f"## {section['heading']}\n\n{section['content']}\n")
        
        return FinalArticle(
            title=draft.title,
            content="\n".join(content_parts),
            metadata={
                "word_count": draft.word_count,
                "sections": len(draft.sections),
                "target_audience": plan.target_audience,
                "tone": plan.tone
            }
        )
    
    async def run(self, topic: str, requirements: str = "") -> FinalArticle:
        """Run the complete pipeline."""
        
        print("📋 Stage 1: Planning...")
        plan = await self.plan(topic, requirements)
        
        print("🔍 Stage 2: Researching...")
        research = await self.research(plan)
        
        print("✍️ Stage 3: Writing...")
        draft = await self.write(plan, research)
        
        print("📝 Stage 4: Editing...")
        edited = await self.edit(draft)
        
        print("✅ Stage 5: Finalizing...")
        final = await self.finalize(edited, plan)
        
        print(f"Done! Created article: {final.title} ({final.metadata['word_count']} words)")
        
        return final


# Usage
async def main():
    pipeline = ContentCreationPipeline(api_key="...")
    
    article = await pipeline.run(
        topic="The Future of Remote Work",
        requirements="Focus on technology and productivity"
    )
    
    print(article.content)
```

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                Sequential Workflows - Summary                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Core Pattern:                                                           │
│    Input → Stage A → Stage B → Stage C → Output                         │
│                                                                          │
│  Advanced Patterns:                                                      │
│    • Conditional Branching - Route based on conditions                  │
│    • Checkpointing - Save progress for recovery                         │
│    • Feedback Loops - Iterate until quality met                         │
│                                                                          │
│  Benefits:                                                               │
│    • Clear flow of data                                                 │
│    • Easy to understand and debug                                       │
│    • Each stage is specialized                                          │
│    • Natural for linear processes                                       │
│                                                                          │
│  Considerations:                                                         │
│    • Latency accumulates across stages                                  │
│    • Error in early stage affects all downstream                        │
│    • No parallelism (use parallel workflows for that)                   │
│                                                                          │
│  Best For:                                                               │
│    • Content creation pipelines                                         │
│    • Document processing                                                │
│    • Quality assurance workflows                                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

Next: [Parallel Workflows](/learn/multi-agents/orchestration-patterns/parallel-workflows) →
