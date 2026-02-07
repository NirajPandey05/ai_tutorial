# Parallel Workflows

Learn how to build multi-agent workflows where agents work simultaneously on different tasks.

## What are Parallel Workflows?

Parallel workflows distribute work across multiple agents that execute concurrently, improving throughput and reducing total execution time.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Parallel Workflow                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Input                                                                  │
│     │                                                                    │
│     ├──────────────────┬──────────────────┐                             │
│     │                  │                  │                             │
│     ▼                  ▼                  ▼                             │
│   ┌─────────┐      ┌─────────┐      ┌─────────┐                        │
│   │ Agent A │      │ Agent B │      │ Agent C │    ← Parallel          │
│   │(Task 1) │      │(Task 2) │      │(Task 3) │      Execution         │
│   └────┬────┘      └────┬────┘      └────┬────┘                        │
│        │                │                │                              │
│        └────────────────┼────────────────┘                              │
│                         │                                                │
│                         ▼                                                │
│                   ┌───────────┐                                          │
│                   │ Aggregator│  ← Combine Results                      │
│                   └─────┬─────┘                                          │
│                         │                                                │
│                         ▼                                                │
│                      Output                                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Basic Parallel Execution

```python
import asyncio
from typing import List, Any, Callable, Dict
from dataclasses import dataclass

@dataclass
class ParallelTask:
    """A task to be executed in parallel."""
    id: str
    input: Any
    processor: Callable

@dataclass 
class ParallelResult:
    """Result from a parallel task."""
    task_id: str
    output: Any
    success: bool
    error: str = None
    execution_time: float = 0.0


class ParallelExecutor:
    """Execute multiple tasks in parallel."""
    
    def __init__(self, max_concurrency: int = 10):
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)
    
    async def _execute_task(self, task: ParallelTask) -> ParallelResult:
        """Execute a single task with semaphore."""
        async with self.semaphore:
            start = asyncio.get_event_loop().time()
            
            try:
                if asyncio.iscoroutinefunction(task.processor):
                    output = await task.processor(task.input)
                else:
                    output = task.processor(task.input)
                
                return ParallelResult(
                    task_id=task.id,
                    output=output,
                    success=True,
                    execution_time=asyncio.get_event_loop().time() - start
                )
            except Exception as e:
                return ParallelResult(
                    task_id=task.id,
                    output=None,
                    success=False,
                    error=str(e),
                    execution_time=asyncio.get_event_loop().time() - start
                )
    
    async def execute(self, tasks: List[ParallelTask]) -> List[ParallelResult]:
        """Execute all tasks in parallel."""
        
        coroutines = [self._execute_task(task) for task in tasks]
        results = await asyncio.gather(*coroutines)
        
        return results


# Example usage
async def parallel_research(topics: List[str]) -> Dict[str, str]:
    """Research multiple topics in parallel."""
    
    async def research_topic(topic: str) -> str:
        # Simulated research
        await asyncio.sleep(1)  # Simulate API call
        return f"Research results for: {topic}"
    
    executor = ParallelExecutor(max_concurrency=5)
    
    tasks = [
        ParallelTask(
            id=f"research_{i}",
            input=topic,
            processor=research_topic
        )
        for i, topic in enumerate(topics)
    ]
    
    results = await executor.execute(tasks)
    
    return {
        r.task_id: r.output
        for r in results
        if r.success
    }
```

## Parallel Agent Patterns

### Pattern 1: Fan-Out / Fan-In

Split work across agents, then aggregate results.

```python
from typing import TypeVar, Generic, Callable
from abc import ABC, abstractmethod

T = TypeVar('T')
R = TypeVar('R')

class FanOutFanIn(Generic[T, R]):
    """Fan-out work to agents, fan-in results."""
    
    def __init__(
        self,
        splitter: Callable[[T], List[Any]],
        aggregator: Callable[[List[Any]], R]
    ):
        self.splitter = splitter
        self.aggregator = aggregator
        self.agents: List[Callable] = []
    
    def add_agent(self, agent: Callable):
        """Add an agent to the pool."""
        self.agents.append(agent)
    
    async def execute(self, input_data: T) -> R:
        """Execute fan-out/fan-in pattern."""
        
        # Fan-out: split input into tasks
        tasks = self.splitter(input_data)
        
        # Distribute tasks to agents
        agent_tasks = []
        for i, task in enumerate(tasks):
            agent = self.agents[i % len(self.agents)]
            if asyncio.iscoroutinefunction(agent):
                agent_tasks.append(agent(task))
            else:
                agent_tasks.append(asyncio.to_thread(agent, task))
        
        # Execute in parallel
        results = await asyncio.gather(*agent_tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [
            r for r in results
            if not isinstance(r, Exception)
        ]
        
        # Fan-in: aggregate results
        return self.aggregator(valid_results)


# Example: Parallel document analysis
class DocumentAnalysisPipeline:
    """Analyze documents in parallel."""
    
    def __init__(self, llm_client):
        self.llm = llm_client
        
        # Define specialized agents
        self.sentiment_agent = self._create_agent("sentiment")
        self.entity_agent = self._create_agent("entities")
        self.summary_agent = self._create_agent("summary")
        self.keywords_agent = self._create_agent("keywords")
    
    def _create_agent(self, task_type: str) -> Callable:
        """Create a specialized analysis agent."""
        
        prompts = {
            "sentiment": "Analyze the sentiment of this text. Return: positive, negative, or neutral with confidence score.",
            "entities": "Extract all named entities (people, places, organizations) from this text.",
            "summary": "Provide a concise 2-sentence summary of this text.",
            "keywords": "Extract the top 5 keywords from this text."
        }
        
        async def agent(text: str) -> dict:
            response = await self.llm.chat(
                f"{prompts[task_type]}\n\nText: {text}"
            )
            return {task_type: response}
        
        return agent
    
    async def analyze(self, document: str) -> dict:
        """Run all analysis in parallel."""
        
        # Execute all agents in parallel
        results = await asyncio.gather(
            self.sentiment_agent(document),
            self.entity_agent(document),
            self.summary_agent(document),
            self.keywords_agent(document)
        )
        
        # Merge results
        analysis = {}
        for result in results:
            analysis.update(result)
        
        return analysis
```

### Pattern 2: Map-Reduce with Agents

```python
class MapReduceAgents:
    """Map-Reduce pattern with AI agents."""
    
    def __init__(self, llm_client, num_mappers: int = 4):
        self.llm = llm_client
        self.num_mappers = num_mappers
    
    async def map_phase(
        self,
        chunks: List[str],
        map_prompt: str
    ) -> List[str]:
        """Map phase: process chunks in parallel."""
        
        async def process_chunk(chunk: str) -> str:
            prompt = f"{map_prompt}\n\nContent:\n{chunk}"
            return await self.llm.chat(prompt)
        
        # Process all chunks in parallel
        results = await asyncio.gather(
            *[process_chunk(chunk) for chunk in chunks]
        )
        
        return results
    
    async def reduce_phase(
        self,
        mapped_results: List[str],
        reduce_prompt: str
    ) -> str:
        """Reduce phase: combine results."""
        
        combined = "\n\n---\n\n".join(mapped_results)
        
        prompt = f"""{reduce_prompt}

Individual Results:
{combined}
"""
        
        return await self.llm.chat(prompt)
    
    async def execute(
        self,
        data: str,
        chunk_size: int,
        map_prompt: str,
        reduce_prompt: str
    ) -> str:
        """Execute map-reduce."""
        
        # Split into chunks
        words = data.split()
        chunks = [
            " ".join(words[i:i + chunk_size])
            for i in range(0, len(words), chunk_size)
        ]
        
        print(f"Processing {len(chunks)} chunks in parallel...")
        
        # Map phase
        mapped = await self.map_phase(chunks, map_prompt)
        
        # Reduce phase
        result = await self.reduce_phase(mapped, reduce_prompt)
        
        return result


# Example: Summarize a long document
async def summarize_long_document(document: str) -> str:
    mr = MapReduceAgents(llm_client, num_mappers=4)
    
    summary = await mr.execute(
        data=document,
        chunk_size=500,
        map_prompt="Summarize the key points from this section:",
        reduce_prompt="Combine these summaries into a coherent overall summary:"
    )
    
    return summary
```

### Pattern 3: Competing Agents

Multiple agents work on the same task, best result wins.

```python
from dataclasses import dataclass
from typing import Callable, Optional

@dataclass
class CompetitionResult:
    winner: str
    winning_output: Any
    all_outputs: Dict[str, Any]
    scores: Dict[str, float]

class AgentCompetition:
    """Multiple agents compete on the same task."""
    
    def __init__(
        self,
        judge: Callable[[List[Any]], tuple[int, Dict[str, float]]]
    ):
        self.agents: Dict[str, Callable] = {}
        self.judge = judge
    
    def add_competitor(self, name: str, agent: Callable):
        """Add a competing agent."""
        self.agents[name] = agent
    
    async def compete(self, task: Any) -> CompetitionResult:
        """Run competition on a task."""
        
        # All agents work in parallel
        async def run_agent(name: str, agent: Callable):
            try:
                if asyncio.iscoroutinefunction(agent):
                    return name, await agent(task)
                else:
                    return name, agent(task)
            except Exception as e:
                return name, None
        
        results = await asyncio.gather(
            *[run_agent(name, agent) for name, agent in self.agents.items()]
        )
        
        # Collect valid outputs
        outputs = {name: output for name, output in results if output is not None}
        
        # Judge selects winner
        output_list = list(outputs.values())
        winner_idx, scores = self.judge(output_list)
        
        agent_names = list(outputs.keys())
        winner_name = agent_names[winner_idx]
        
        return CompetitionResult(
            winner=winner_name,
            winning_output=outputs[winner_name],
            all_outputs=outputs,
            scores=dict(zip(agent_names, [scores.get(i, 0) for i in range(len(agent_names))]))
        )


# Example: Best code solution
async def code_competition():
    
    async def judge_code(solutions: List[str]) -> tuple[int, Dict[int, float]]:
        """LLM judges the solutions."""
        
        prompt = f"""Rate these code solutions from 0-10 on:
- Correctness
- Readability
- Efficiency

Solutions:
{chr(10).join(f"Solution {i+1}:{chr(10)}{sol}" for i, sol in enumerate(solutions))}

Return JSON: {{"ratings": [score1, score2, ...]}}
"""
        response = await llm.chat(prompt)
        import json
        data = json.loads(response)
        
        scores = {i: s for i, s in enumerate(data["ratings"])}
        winner = max(scores, key=scores.get)
        
        return winner, scores
    
    competition = AgentCompetition(judge=judge_code)
    
    # Add different coding styles
    competition.add_competitor("python_expert", python_coder_agent)
    competition.add_competitor("clean_coder", clean_code_agent)
    competition.add_competitor("fast_coder", performance_agent)
    
    result = await competition.compete("Write a function to find prime numbers")
    
    print(f"Winner: {result.winner}")
    print(f"Scores: {result.scores}")
    
    return result.winning_output
```

## Complete Parallel Workflow System

```python
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time

class TaskPriority(Enum):
    HIGH = 1
    MEDIUM = 2
    LOW = 3

@dataclass
class WorkItem:
    id: str
    data: Any
    priority: TaskPriority = TaskPriority.MEDIUM
    dependencies: List[str] = field(default_factory=list)
    result: Any = None
    status: str = "pending"
    
class ParallelWorkflow:
    """Advanced parallel workflow with dependencies."""
    
    def __init__(self, max_concurrency: int = 10):
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.processors: Dict[str, Callable] = {}
        self.items: Dict[str, WorkItem] = {}
        self.results: Dict[str, Any] = {}
    
    def register_processor(self, name: str, processor: Callable):
        """Register a processor for a type of work."""
        self.processors[name] = processor
    
    def add_item(
        self,
        item_id: str,
        data: Any,
        processor_name: str,
        dependencies: List[str] = None,
        priority: TaskPriority = TaskPriority.MEDIUM
    ):
        """Add a work item."""
        self.items[item_id] = WorkItem(
            id=item_id,
            data={"input": data, "processor": processor_name},
            priority=priority,
            dependencies=dependencies or []
        )
    
    def _get_ready_items(self) -> List[WorkItem]:
        """Get items ready to process (dependencies met)."""
        ready = []
        
        for item in self.items.values():
            if item.status != "pending":
                continue
            
            deps_met = all(
                self.items[dep].status == "completed"
                for dep in item.dependencies
                if dep in self.items
            )
            
            if deps_met:
                ready.append(item)
        
        # Sort by priority
        ready.sort(key=lambda x: x.priority.value)
        
        return ready
    
    async def _process_item(self, item: WorkItem):
        """Process a single item."""
        async with self.semaphore:
            item.status = "running"
            
            processor_name = item.data["processor"]
            processor = self.processors.get(processor_name)
            
            if not processor:
                item.status = "failed"
                item.result = f"Unknown processor: {processor_name}"
                return
            
            try:
                # Get dependency results if needed
                dep_results = {
                    dep: self.items[dep].result
                    for dep in item.dependencies
                }
                
                input_data = {
                    "input": item.data["input"],
                    "dependencies": dep_results
                }
                
                if asyncio.iscoroutinefunction(processor):
                    result = await processor(input_data)
                else:
                    result = processor(input_data)
                
                item.result = result
                item.status = "completed"
                self.results[item.id] = result
                
            except Exception as e:
                item.status = "failed"
                item.result = str(e)
    
    async def run(self) -> Dict[str, Any]:
        """Run the workflow."""
        
        while True:
            ready = self._get_ready_items()
            
            if not ready:
                # Check if done
                all_done = all(
                    item.status in ("completed", "failed")
                    for item in self.items.values()
                )
                
                if all_done:
                    break
                
                # Wait for running items
                await asyncio.sleep(0.1)
                continue
            
            # Process ready items in parallel
            await asyncio.gather(
                *[self._process_item(item) for item in ready]
            )
        
        return self.results


# Example: Parallel research with aggregation
async def parallel_research_workflow():
    workflow = ParallelWorkflow(max_concurrency=5)
    
    # Register processors
    async def research_processor(data: dict) -> str:
        topic = data["input"]
        # Simulate research
        await asyncio.sleep(1)
        return f"Research on {topic}: [findings...]"
    
    async def aggregate_processor(data: dict) -> str:
        deps = data["dependencies"]
        all_research = "\n".join(deps.values())
        return f"Aggregated Report:\n{all_research}"
    
    workflow.register_processor("research", research_processor)
    workflow.register_processor("aggregate", aggregate_processor)
    
    # Add parallel research tasks
    topics = ["AI", "Machine Learning", "Neural Networks", "NLP"]
    
    for i, topic in enumerate(topics):
        workflow.add_item(
            f"research_{i}",
            topic,
            "research",
            priority=TaskPriority.HIGH
        )
    
    # Add aggregation that depends on all research
    workflow.add_item(
        "final_report",
        "Combine all research",
        "aggregate",
        dependencies=[f"research_{i}" for i in range(len(topics))],
        priority=TaskPriority.LOW
    )
    
    results = await workflow.run()
    return results["final_report"]
```

## Parallel vs Sequential: When to Use

```
┌─────────────────────────────────────────────────────────────────────────┐
│                Parallel vs Sequential                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Use PARALLEL when:                                                      │
│    ✓ Tasks are independent                                              │
│    ✓ Speed is important                                                 │
│    ✓ Tasks don't need each other's output                              │
│    ✓ Different expertise needed simultaneously                          │
│                                                                          │
│  Use SEQUENTIAL when:                                                    │
│    ✓ Output of one is input to next                                    │
│    ✓ Order matters                                                      │
│    ✓ Need to validate before proceeding                                │
│    ✓ Building on previous work                                         │
│                                                                          │
│  Use HYBRID when:                                                        │
│    ✓ Some tasks parallel, some sequential                              │
│    ✓ Complex workflows with branches                                   │
│    ✓ Need both speed and quality checks                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                Parallel Workflows - Summary                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Patterns:                                                               │
│    • Fan-Out/Fan-In - Split, process, aggregate                        │
│    • Map-Reduce - Transform chunks, combine                            │
│    • Competition - Multiple attempts, best wins                        │
│                                                                          │
│  Benefits:                                                               │
│    • Faster execution (parallel > sequential)                          │
│    • Better resource utilization                                       │
│    • Handle independent tasks efficiently                              │
│                                                                          │
│  Considerations:                                                         │
│    • Manage concurrency limits                                         │
│    • Handle partial failures                                           │
│    • Aggregate results correctly                                       │
│    • Higher cost (more simultaneous API calls)                         │
│                                                                          │
│  Best For:                                                               │
│    • Document analysis (multiple aspects)                              │
│    • Research (multiple topics)                                        │
│    • Content translation (multiple languages)                          │
│    • Batch processing                                                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

Next: [Hierarchical Agents](/learn/multi-agents/orchestration-patterns/hierarchical-agents) →
