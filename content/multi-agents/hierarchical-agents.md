# Hierarchical Agents

Learn how to build hierarchical agent systems with supervisors managing teams of worker agents.

## What are Hierarchical Agents?

Hierarchical systems organize agents in a tree structure where supervisor agents manage and coordinate worker agents.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Hierarchical Agent Structure                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                         ┌──────────────────┐                            │
│                         │   Top Supervisor │                            │
│                         │   (Orchestrator) │                            │
│                         └────────┬─────────┘                            │
│                                  │                                       │
│              ┌───────────────────┼───────────────────┐                  │
│              │                   │                   │                  │
│              ▼                   ▼                   ▼                  │
│     ┌────────────────┐  ┌────────────────┐  ┌────────────────┐         │
│     │ Team Lead A    │  │ Team Lead B    │  │ Team Lead C    │         │
│     │ (Research)     │  │ (Writing)      │  │ (Review)       │         │
│     └───────┬────────┘  └───────┬────────┘  └───────┬────────┘         │
│             │                   │                   │                  │
│        ┌────┴────┐         ┌────┴────┐         ┌────┴────┐            │
│        │         │         │         │         │         │            │
│        ▼         ▼         ▼         ▼         ▼         ▼            │
│   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐  │
│   │Worker 1│ │Worker 2│ │Worker 3│ │Worker 4│ │Worker 5│ │Worker 6│  │
│   └────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Supervisor-Worker Pattern

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from enum import Enum
import asyncio

class WorkerStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"
    FAILED = "failed"

@dataclass
class Task:
    id: str
    description: str
    assigned_to: Optional[str] = None
    result: Any = None
    status: str = "pending"

@dataclass
class Worker:
    """A worker agent managed by a supervisor."""
    
    name: str
    capabilities: List[str]
    status: WorkerStatus = WorkerStatus.IDLE
    current_task: Optional[Task] = None
    
    async def execute(self, task: Task) -> Any:
        """Execute a task."""
        self.status = WorkerStatus.BUSY
        self.current_task = task
        
        try:
            result = await self._do_work(task)
            return result
        finally:
            self.status = WorkerStatus.IDLE
            self.current_task = None
    
    async def _do_work(self, task: Task) -> Any:
        """Override in subclass."""
        raise NotImplementedError


class Supervisor:
    """Supervisor that manages worker agents."""
    
    def __init__(self, name: str, llm_client):
        self.name = name
        self.llm = llm_client
        self.workers: Dict[str, Worker] = {}
        self.task_queue: List[Task] = []
        self.completed_tasks: List[Task] = []
    
    def add_worker(self, worker: Worker):
        """Add a worker to manage."""
        self.workers[worker.name] = worker
    
    def get_available_workers(self) -> List[Worker]:
        """Get workers that are not busy."""
        return [
            w for w in self.workers.values()
            if w.status == WorkerStatus.IDLE
        ]
    
    async def delegate(self, task: Task) -> str:
        """Decide which worker should handle a task."""
        
        available = self.get_available_workers()
        if not available:
            return None
        
        # Use LLM to match task to worker capabilities
        worker_info = "\n".join(
            f"- {w.name}: {', '.join(w.capabilities)}"
            for w in available
        )
        
        prompt = f"""Match this task to the best worker:

Task: {task.description}

Available Workers:
{worker_info}

Return only the worker name that is best suited for this task."""
        
        response = await self.llm.chat(prompt)
        worker_name = response.strip()
        
        if worker_name in self.workers:
            return worker_name
        
        # Fallback to first available
        return available[0].name
    
    async def assign_and_execute(self, task: Task) -> Any:
        """Assign task to worker and wait for completion."""
        
        worker_name = await self.delegate(task)
        
        if not worker_name:
            raise Exception("No workers available")
        
        worker = self.workers[worker_name]
        task.assigned_to = worker_name
        task.status = "in_progress"
        
        print(f"[{self.name}] Assigned '{task.id}' to {worker_name}")
        
        result = await worker.execute(task)
        
        task.result = result
        task.status = "completed"
        self.completed_tasks.append(task)
        
        return result
    
    async def run_workflow(self, tasks: List[Task]) -> Dict[str, Any]:
        """Run multiple tasks, parallelizing where possible."""
        
        results = {}
        
        # Process tasks in batches based on available workers
        pending = tasks.copy()
        
        while pending:
            available_count = len(self.get_available_workers())
            
            if available_count == 0:
                await asyncio.sleep(0.1)
                continue
            
            # Take batch of tasks
            batch = pending[:available_count]
            pending = pending[available_count:]
            
            # Execute batch in parallel
            batch_results = await asyncio.gather(
                *[self.assign_and_execute(task) for task in batch],
                return_exceptions=True
            )
            
            for task, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    results[task.id] = {"error": str(result)}
                else:
                    results[task.id] = result
        
        return results
```

## Multi-Level Hierarchy

```python
from typing import Union

class HierarchicalAgent:
    """Agent that can be both supervisor and worker."""
    
    def __init__(self, name: str, llm_client, parent: "HierarchicalAgent" = None):
        self.name = name
        self.llm = llm_client
        self.parent = parent
        self.subordinates: List["HierarchicalAgent"] = []
        self.capabilities: List[str] = []
    
    def add_subordinate(self, agent: "HierarchicalAgent"):
        """Add a subordinate agent."""
        agent.parent = self
        self.subordinates.append(agent)
    
    @property
    def is_supervisor(self) -> bool:
        return len(self.subordinates) > 0
    
    @property
    def is_worker(self) -> bool:
        return len(self.subordinates) == 0
    
    async def handle_task(self, task: Task) -> Any:
        """Handle a task - delegate or execute."""
        
        if self.is_worker:
            # Execute directly
            return await self._execute_task(task)
        else:
            # Delegate to subordinates
            return await self._delegate_task(task)
    
    async def _execute_task(self, task: Task) -> Any:
        """Execute task as a worker."""
        prompt = f"""Complete this task:
{task.description}

You are: {self.name}
Capabilities: {', '.join(self.capabilities)}
"""
        return await self.llm.chat(prompt)
    
    async def _delegate_task(self, task: Task) -> Any:
        """Delegate task to subordinates."""
        
        # Decompose task if complex
        subtasks = await self._decompose_task(task)
        
        if len(subtasks) == 1:
            # Simple task - delegate to best subordinate
            best = await self._select_subordinate(subtasks[0])
            return await best.handle_task(subtasks[0])
        else:
            # Multiple subtasks - distribute and aggregate
            results = await self._distribute_tasks(subtasks)
            return await self._aggregate_results(task, results)
    
    async def _decompose_task(self, task: Task) -> List[Task]:
        """Break down a complex task into subtasks."""
        
        subordinate_info = "\n".join(
            f"- {s.name}: {', '.join(s.capabilities) if s.capabilities else 'general'}"
            for s in self.subordinates
        )
        
        prompt = f"""Analyze this task and decide how to handle it:

Task: {task.description}

Available team members:
{subordinate_info}

If the task should be broken into subtasks, list them as JSON:
{{"subtasks": ["subtask 1 description", "subtask 2 description"]}}

If it can be handled by one person, respond:
{{"subtasks": ["{task.description}"]}}
"""
        
        response = await self.llm.chat(prompt)
        
        import json
        data = json.loads(response)
        
        return [
            Task(id=f"{task.id}_sub_{i}", description=desc)
            for i, desc in enumerate(data["subtasks"])
        ]
    
    async def _select_subordinate(self, task: Task) -> "HierarchicalAgent":
        """Select best subordinate for a task."""
        
        if len(self.subordinates) == 1:
            return self.subordinates[0]
        
        # Score subordinates based on capability match
        best = self.subordinates[0]
        best_score = 0
        
        for sub in self.subordinates:
            score = sum(
                1 for cap in sub.capabilities
                if cap.lower() in task.description.lower()
            )
            if score > best_score:
                best = sub
                best_score = score
        
        return best
    
    async def _distribute_tasks(self, tasks: List[Task]) -> Dict[str, Any]:
        """Distribute tasks to subordinates."""
        
        results = {}
        
        async def assign_task(task: Task):
            sub = await self._select_subordinate(task)
            result = await sub.handle_task(task)
            return task.id, result
        
        task_results = await asyncio.gather(
            *[assign_task(t) for t in tasks]
        )
        
        for task_id, result in task_results:
            results[task_id] = result
        
        return results
    
    async def _aggregate_results(
        self,
        original_task: Task,
        results: Dict[str, Any]
    ) -> Any:
        """Aggregate results from subtasks."""
        
        results_text = "\n".join(
            f"- {task_id}: {result}"
            for task_id, result in results.items()
        )
        
        prompt = f"""Combine these subtask results into a final answer:

Original Task: {original_task.description}

Subtask Results:
{results_text}

Provide a coherent combined response.
"""
        
        return await self.llm.chat(prompt)


# Example: Build a hierarchy
def create_content_team(llm_client) -> HierarchicalAgent:
    """Create a hierarchical content team."""
    
    # Top-level manager
    manager = HierarchicalAgent("Content Manager", llm_client)
    
    # Research team
    research_lead = HierarchicalAgent("Research Lead", llm_client, manager)
    research_lead.capabilities = ["research", "analysis"]
    
    researcher1 = HierarchicalAgent("Researcher 1", llm_client)
    researcher1.capabilities = ["web research", "fact checking"]
    
    researcher2 = HierarchicalAgent("Researcher 2", llm_client)
    researcher2.capabilities = ["data analysis", "statistics"]
    
    research_lead.add_subordinate(researcher1)
    research_lead.add_subordinate(researcher2)
    
    # Writing team
    writing_lead = HierarchicalAgent("Writing Lead", llm_client, manager)
    writing_lead.capabilities = ["writing", "editing"]
    
    writer1 = HierarchicalAgent("Writer 1", llm_client)
    writer1.capabilities = ["technical writing", "documentation"]
    
    writer2 = HierarchicalAgent("Writer 2", llm_client)
    writer2.capabilities = ["creative writing", "storytelling"]
    
    writing_lead.add_subordinate(writer1)
    writing_lead.add_subordinate(writer2)
    
    # Add teams to manager
    manager.add_subordinate(research_lead)
    manager.add_subordinate(writing_lead)
    
    return manager


# Usage
async def run_hierarchical_workflow():
    team = create_content_team(llm_client)
    
    task = Task(
        id="article_001",
        description="Create a comprehensive article about quantum computing including research, technical explanation, and engaging narrative"
    )
    
    result = await team.handle_task(task)
    return result
```

## Supervisor Patterns

### Pattern 1: Escalation

Workers escalate issues they can't handle.

```python
class EscalatingWorker(Worker):
    """Worker that can escalate to supervisor."""
    
    def __init__(self, name: str, capabilities: List[str], supervisor: "EscalatingSupervisor"):
        super().__init__(name, capabilities)
        self.supervisor = supervisor
        self.confidence_threshold = 0.7
    
    async def _do_work(self, task: Task) -> Any:
        """Execute with escalation support."""
        
        # Assess confidence
        confidence = await self._assess_confidence(task)
        
        if confidence < self.confidence_threshold:
            # Escalate to supervisor
            print(f"[{self.name}] Escalating task '{task.id}' (confidence: {confidence:.0%})")
            return await self.supervisor.handle_escalation(task, self, confidence)
        
        # Handle normally
        return await self._execute(task)
    
    async def _assess_confidence(self, task: Task) -> float:
        """Assess confidence in handling task."""
        
        matching_caps = sum(
            1 for cap in self.capabilities
            if cap.lower() in task.description.lower()
        )
        
        return min(1.0, matching_caps * 0.3 + 0.4)


class EscalatingSupervisor(Supervisor):
    """Supervisor that handles escalations."""
    
    async def handle_escalation(
        self,
        task: Task,
        escalating_worker: Worker,
        confidence: float
    ) -> Any:
        """Handle an escalated task."""
        
        print(f"[{self.name}] Handling escalation from {escalating_worker.name}")
        
        # Options: 
        # 1. Handle directly
        # 2. Reassign to another worker
        # 3. Decompose and distribute
        
        # Find better suited worker
        other_workers = [
            w for w in self.workers.values()
            if w.name != escalating_worker.name and w.status == WorkerStatus.IDLE
        ]
        
        if other_workers:
            best_worker = max(
                other_workers,
                key=lambda w: sum(
                    1 for cap in w.capabilities
                    if cap.lower() in task.description.lower()
                )
            )
            
            return await self.assign_and_execute_to(task, best_worker)
        
        # Handle directly if no better option
        return await self._handle_directly(task)
```

### Pattern 2: Approval Chain

Work requires approval up the chain.

```python
class ApprovalHierarchy:
    """Hierarchy with approval requirements."""
    
    def __init__(self):
        self.levels: List[List["ApprovalAgent"]] = []
    
    def add_level(self, agents: List["ApprovalAgent"]):
        """Add a level to the hierarchy."""
        self.levels.append(agents)
    
    async def submit_for_approval(
        self,
        content: Any,
        starting_level: int = 0
    ) -> tuple[bool, Any]:
        """Submit content through approval chain."""
        
        current_content = content
        
        for level_idx in range(starting_level, len(self.levels)):
            level = self.levels[level_idx]
            
            # Get approval from any agent at this level
            for agent in level:
                approved, feedback = await agent.review(current_content)
                
                if approved:
                    print(f"[{agent.name}] Approved at level {level_idx}")
                    break
                else:
                    print(f"[{agent.name}] Rejected: {feedback}")
                    
                    # Revision required
                    current_content = await self._revise(current_content, feedback)
            else:
                # No approval at this level
                return False, current_content
        
        return True, current_content


class ApprovalAgent:
    """Agent that can approve or reject work."""
    
    def __init__(self, name: str, llm_client, approval_criteria: str):
        self.name = name
        self.llm = llm_client
        self.criteria = approval_criteria
    
    async def review(self, content: Any) -> tuple[bool, str]:
        """Review content for approval."""
        
        prompt = f"""Review this content for approval:

Content:
{content}

Approval Criteria:
{self.criteria}

Respond with JSON:
{{"approved": true/false, "feedback": "explanation"}}
"""
        
        response = await self.llm.chat(prompt)
        
        import json
        data = json.loads(response)
        
        return data["approved"], data.get("feedback", "")
```

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│               Hierarchical Agents - Summary                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Structure:                                                              │
│    • Supervisors manage workers                                         │
│    • Multi-level hierarchies possible                                   │
│    • Clear chain of command                                             │
│                                                                          │
│  Supervisor Responsibilities:                                            │
│    • Task decomposition                                                 │
│    • Worker selection                                                   │
│    • Result aggregation                                                 │
│    • Escalation handling                                                │
│                                                                          │
│  Patterns:                                                               │
│    • Supervisor-Worker - Basic delegation                               │
│    • Escalation - Workers can escalate issues                          │
│    • Approval Chain - Work flows up for approval                       │
│                                                                          │
│  Benefits:                                                               │
│    • Clear accountability                                               │
│    • Scalable team structure                                           │
│    • Specialization at each level                                      │
│                                                                          │
│  Best For:                                                               │
│    • Complex projects needing coordination                             │
│    • Teams with different specializations                              │
│    • Work requiring oversight/approval                                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

Next: [Debate Patterns](/learn/multi-agents/orchestration-patterns/debate-patterns) →
