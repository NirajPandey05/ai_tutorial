# State Sharing and Handoffs

Learn how agents share state and hand off work to each other in multi-agent systems.

## Why State Sharing?

Agents need to share context and state to collaborate effectively, but must do so safely without conflicts.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    State Sharing Patterns                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Isolated State:           Shared State:                               │
│                                                                          │
│   ┌─────────┐               ┌─────────┐                                 │
│   │ Agent A │               │ Agent A │                                 │
│   │ [state] │               └────┬────┘                                 │
│   └─────────┘                    │                                      │
│                                  ▼                                      │
│   ┌─────────┐              ┌───────────┐                                │
│   │ Agent B │              │  Shared   │ ◄── Single source              │
│   │ [state] │              │   State   │     of truth                   │
│   └─────────┘              └─────┬─────┘                                │
│                                  │                                      │
│   Problems:                      ▼                                      │
│   - Inconsistent            ┌─────────┐                                 │
│   - Duplicated              │ Agent B │                                 │
│   - Hard to sync            └─────────┘                                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Shared State Store

```python
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import asyncio
import json
import copy

@dataclass
class StateChange:
    """Record of a state change."""
    timestamp: datetime
    agent_id: str
    key: str
    old_value: Any
    new_value: Any
    operation: str  # set, update, delete


class SharedStateStore:
    """Thread-safe shared state store for agents."""
    
    def __init__(self):
        self._state: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._history: List[StateChange] = []
        self._subscribers: Dict[str, List[Callable]] = {}
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get a value from state."""
        async with self._lock:
            return copy.deepcopy(self._state.get(key, default))
    
    async def set(self, key: str, value: Any, agent_id: str = "system") -> None:
        """Set a value in state."""
        async with self._lock:
            old_value = self._state.get(key)
            self._state[key] = value
            
            # Record change
            self._history.append(StateChange(
                timestamp=datetime.now(),
                agent_id=agent_id,
                key=key,
                old_value=old_value,
                new_value=value,
                operation="set"
            ))
            
            # Notify subscribers
            await self._notify_subscribers(key, value)
    
    async def update(
        self,
        key: str,
        update_fn: Callable[[Any], Any],
        agent_id: str = "system"
    ) -> Any:
        """Update a value using a function."""
        async with self._lock:
            old_value = self._state.get(key)
            new_value = update_fn(copy.deepcopy(old_value))
            self._state[key] = new_value
            
            self._history.append(StateChange(
                timestamp=datetime.now(),
                agent_id=agent_id,
                key=key,
                old_value=old_value,
                new_value=new_value,
                operation="update"
            ))
            
            await self._notify_subscribers(key, new_value)
            return new_value
    
    async def delete(self, key: str, agent_id: str = "system") -> None:
        """Delete a key from state."""
        async with self._lock:
            old_value = self._state.pop(key, None)
            
            self._history.append(StateChange(
                timestamp=datetime.now(),
                agent_id=agent_id,
                key=key,
                old_value=old_value,
                new_value=None,
                operation="delete"
            ))
    
    def subscribe(self, key: str, callback: Callable):
        """Subscribe to changes on a key."""
        if key not in self._subscribers:
            self._subscribers[key] = []
        self._subscribers[key].append(callback)
    
    async def _notify_subscribers(self, key: str, value: Any):
        """Notify subscribers of a change."""
        if key in self._subscribers:
            for callback in self._subscribers[key]:
                if asyncio.iscoroutinefunction(callback):
                    await callback(key, value)
                else:
                    callback(key, value)
    
    def get_history(self, key: str = None) -> List[StateChange]:
        """Get change history."""
        if key:
            return [c for c in self._history if c.key == key]
        return self._history.copy()
    
    async def snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of current state."""
        async with self._lock:
            return copy.deepcopy(self._state)


# Namespaced state for agent isolation
class NamespacedState:
    """State store with namespace isolation."""
    
    def __init__(self, store: SharedStateStore, namespace: str):
        self.store = store
        self.namespace = namespace
    
    def _key(self, key: str) -> str:
        return f"{self.namespace}:{key}"
    
    async def get(self, key: str, default: Any = None) -> Any:
        return await self.store.get(self._key(key), default)
    
    async def set(self, key: str, value: Any, agent_id: str = "system") -> None:
        await self.store.set(self._key(key), value, agent_id)
    
    async def update(
        self,
        key: str,
        update_fn: Callable[[Any], Any],
        agent_id: str = "system"
    ) -> Any:
        return await self.store.update(self._key(key), update_fn, agent_id)
```

## Work Handoff Patterns

### Pattern 1: Explicit Handoff

```python
@dataclass
class HandoffContext:
    """Context passed during a handoff."""
    from_agent: str
    to_agent: str
    task_id: str
    status: str
    work_completed: Dict[str, Any]
    remaining_work: str
    notes: str
    timestamp: datetime = field(default_factory=datetime.now)


class HandoffManager:
    """Manage handoffs between agents."""
    
    def __init__(self, state_store: SharedStateStore):
        self.state = state_store
        self.pending_handoffs: Dict[str, HandoffContext] = {}
    
    async def initiate_handoff(
        self,
        from_agent: str,
        to_agent: str,
        task_id: str,
        work_completed: Dict[str, Any],
        remaining_work: str,
        notes: str = ""
    ) -> HandoffContext:
        """Initiate a handoff to another agent."""
        
        context = HandoffContext(
            from_agent=from_agent,
            to_agent=to_agent,
            task_id=task_id,
            status="pending",
            work_completed=work_completed,
            remaining_work=remaining_work,
            notes=notes
        )
        
        # Store handoff
        self.pending_handoffs[task_id] = context
        
        # Update shared state
        await self.state.set(
            f"handoff:{task_id}",
            {
                "context": context.__dict__,
                "status": "pending"
            },
            from_agent
        )
        
        return context
    
    async def accept_handoff(
        self,
        agent_id: str,
        task_id: str
    ) -> Optional[HandoffContext]:
        """Accept a pending handoff."""
        
        handoff = self.pending_handoffs.get(task_id)
        
        if not handoff or handoff.to_agent != agent_id:
            return None
        
        handoff.status = "accepted"
        
        await self.state.update(
            f"handoff:{task_id}",
            lambda x: {**x, "status": "accepted", "accepted_at": datetime.now().isoformat()},
            agent_id
        )
        
        return handoff
    
    async def complete_handoff(
        self,
        agent_id: str,
        task_id: str,
        result: Any
    ) -> None:
        """Mark a handoff as complete."""
        
        await self.state.update(
            f"handoff:{task_id}",
            lambda x: {
                **x,
                "status": "completed",
                "result": result,
                "completed_at": datetime.now().isoformat()
            },
            agent_id
        )
        
        del self.pending_handoffs[task_id]


class HandoffAgent:
    """Agent that supports handoffs."""
    
    def __init__(
        self,
        agent_id: str,
        handoff_manager: HandoffManager,
        state_store: SharedStateStore,
        llm_client: Any
    ):
        self.id = agent_id
        self.handoffs = handoff_manager
        self.state = state_store
        self.llm = llm_client
    
    async def work_on_task(
        self,
        task_id: str,
        description: str,
        handoff_context: HandoffContext = None
    ) -> Dict[str, Any]:
        """Work on a task, possibly from handoff."""
        
        # Get context from handoff if available
        if handoff_context:
            prior_work = handoff_context.work_completed
            task_description = handoff_context.remaining_work
            notes = handoff_context.notes
        else:
            prior_work = {}
            task_description = description
            notes = ""
        
        # Do work
        result = await self._process(task_description, prior_work, notes)
        
        # Update shared state with progress
        await self.state.set(
            f"task:{task_id}:progress",
            {
                "agent": self.id,
                "status": "in_progress",
                "last_update": datetime.now().isoformat()
            },
            self.id
        )
        
        return result
    
    async def handoff_to(
        self,
        next_agent: str,
        task_id: str,
        work_completed: Dict[str, Any],
        remaining_work: str,
        notes: str = ""
    ) -> HandoffContext:
        """Hand off work to another agent."""
        
        return await self.handoffs.initiate_handoff(
            from_agent=self.id,
            to_agent=next_agent,
            task_id=task_id,
            work_completed=work_completed,
            remaining_work=remaining_work,
            notes=notes
        )
    
    async def _process(
        self,
        task: str,
        prior_work: Dict,
        notes: str
    ) -> Dict[str, Any]:
        """Process task (implement in subclass)."""
        pass
```

### Pattern 2: Checkpoint-Based Handoff

```python
@dataclass
class Checkpoint:
    """A checkpoint in work progress."""
    id: str
    task_id: str
    agent_id: str
    timestamp: datetime
    state: Dict[str, Any]
    description: str


class CheckpointManager:
    """Manage checkpoints for task handoffs."""
    
    def __init__(self, state_store: SharedStateStore):
        self.state = state_store
    
    async def create_checkpoint(
        self,
        task_id: str,
        agent_id: str,
        checkpoint_state: Dict[str, Any],
        description: str
    ) -> Checkpoint:
        """Create a checkpoint."""
        
        checkpoint = Checkpoint(
            id=f"{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            task_id=task_id,
            agent_id=agent_id,
            timestamp=datetime.now(),
            state=checkpoint_state,
            description=description
        )
        
        # Store checkpoint
        await self.state.update(
            f"checkpoints:{task_id}",
            lambda x: (x or []) + [checkpoint.__dict__],
            agent_id
        )
        
        return checkpoint
    
    async def get_latest_checkpoint(self, task_id: str) -> Optional[Checkpoint]:
        """Get the latest checkpoint for a task."""
        
        checkpoints = await self.state.get(f"checkpoints:{task_id}", [])
        
        if not checkpoints:
            return None
        
        latest = max(checkpoints, key=lambda x: x["timestamp"])
        return Checkpoint(**latest)
    
    async def restore_from_checkpoint(
        self,
        task_id: str,
        checkpoint_id: str = None
    ) -> Optional[Dict[str, Any]]:
        """Restore state from a checkpoint."""
        
        checkpoints = await self.state.get(f"checkpoints:{task_id}", [])
        
        if not checkpoints:
            return None
        
        if checkpoint_id:
            checkpoint = next(
                (c for c in checkpoints if c["id"] == checkpoint_id),
                None
            )
        else:
            checkpoint = max(checkpoints, key=lambda x: x["timestamp"])
        
        if checkpoint:
            return checkpoint["state"]
        return None
```

### Pattern 3: Pipeline State

```python
@dataclass
class PipelineStage:
    name: str
    agent_id: str
    status: str = "pending"
    input: Any = None
    output: Any = None
    started_at: datetime = None
    completed_at: datetime = None


class PipelineStateManager:
    """Manage state across a pipeline of agents."""
    
    def __init__(self, state_store: SharedStateStore):
        self.state = state_store
    
    async def initialize_pipeline(
        self,
        pipeline_id: str,
        stages: List[Dict[str, str]]
    ) -> None:
        """Initialize pipeline state."""
        
        pipeline_state = {
            "id": pipeline_id,
            "stages": [
                PipelineStage(
                    name=s["name"],
                    agent_id=s["agent_id"]
                ).__dict__
                for s in stages
            ],
            "current_stage": 0,
            "status": "initialized"
        }
        
        await self.state.set(f"pipeline:{pipeline_id}", pipeline_state, "system")
    
    async def start_stage(
        self,
        pipeline_id: str,
        stage_name: str,
        input_data: Any
    ) -> None:
        """Mark a stage as started."""
        
        async def update_stage(pipeline):
            for stage in pipeline["stages"]:
                if stage["name"] == stage_name:
                    stage["status"] = "running"
                    stage["input"] = input_data
                    stage["started_at"] = datetime.now().isoformat()
                    break
            pipeline["status"] = "running"
            return pipeline
        
        await self.state.update(f"pipeline:{pipeline_id}", update_stage, stage_name)
    
    async def complete_stage(
        self,
        pipeline_id: str,
        stage_name: str,
        output_data: Any
    ) -> Optional[str]:
        """Mark a stage as complete, return next stage."""
        
        next_stage = None
        
        async def update_stage(pipeline):
            nonlocal next_stage
            
            for i, stage in enumerate(pipeline["stages"]):
                if stage["name"] == stage_name:
                    stage["status"] = "completed"
                    stage["output"] = output_data
                    stage["completed_at"] = datetime.now().isoformat()
                    
                    # Check for next stage
                    if i + 1 < len(pipeline["stages"]):
                        next_stage = pipeline["stages"][i + 1]["name"]
                        pipeline["current_stage"] = i + 1
                    else:
                        pipeline["status"] = "completed"
                    break
            
            return pipeline
        
        await self.state.update(f"pipeline:{pipeline_id}", update_stage, stage_name)
        return next_stage
    
    async def get_stage_output(
        self,
        pipeline_id: str,
        stage_name: str
    ) -> Any:
        """Get output from a completed stage."""
        
        pipeline = await self.state.get(f"pipeline:{pipeline_id}")
        
        if not pipeline:
            return None
        
        for stage in pipeline["stages"]:
            if stage["name"] == stage_name:
                return stage.get("output")
        
        return None


# Usage example
async def pipeline_workflow():
    state = SharedStateStore()
    pipeline = PipelineStateManager(state)
    
    # Initialize
    await pipeline.initialize_pipeline(
        "content_pipeline_001",
        [
            {"name": "research", "agent_id": "researcher"},
            {"name": "write", "agent_id": "writer"},
            {"name": "review", "agent_id": "reviewer"}
        ]
    )
    
    # Research stage
    await pipeline.start_stage("content_pipeline_001", "research", {"topic": "AI"})
    research_output = {"facts": ["fact1", "fact2"], "sources": ["src1"]}
    next_stage = await pipeline.complete_stage(
        "content_pipeline_001", "research", research_output
    )
    
    # Write stage uses research output
    research_data = await pipeline.get_stage_output("content_pipeline_001", "research")
    await pipeline.start_stage("content_pipeline_001", "write", research_data)
    # ... continue pipeline
```

## Agent with State Awareness

```python
class StateAwareAgent:
    """Agent that maintains awareness of shared state."""
    
    def __init__(
        self,
        agent_id: str,
        state_store: SharedStateStore,
        llm_client: Any
    ):
        self.id = agent_id
        self.state = NamespacedState(state_store, agent_id)
        self.global_state = state_store
        self.llm = llm_client
        
        # Subscribe to relevant state changes
        state_store.subscribe("task:*", self._on_task_change)
    
    async def _on_task_change(self, key: str, value: Any):
        """React to task state changes."""
        # Check if this affects our work
        pass
    
    async def execute_with_context(self, task: str) -> str:
        """Execute task with full state awareness."""
        
        # Gather relevant state
        context = await self._gather_context(task)
        
        prompt = f"""Complete this task with awareness of the current context:

Task: {task}

Current Context:
{json.dumps(context, indent=2)}

Consider:
1. What has already been done
2. What other agents are working on
3. What state needs to be updated after completion"""
        
        response = await self.llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        result = response.choices[0].message.content
        
        # Update state with results
        await self.state.set("last_task_result", {
            "task": task,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }, self.id)
        
        return result
    
    async def _gather_context(self, task: str) -> Dict[str, Any]:
        """Gather relevant context from state."""
        
        snapshot = await self.global_state.snapshot()
        
        # Filter to relevant keys
        relevant = {}
        for key, value in snapshot.items():
            if self.id in key or "global" in key or "task" in key:
                relevant[key] = value
        
        return relevant
```

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│              State Sharing & Handoffs - Summary                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  State Store Features:                                                   │
│    • Thread-safe access                                                 │
│    • Change history tracking                                            │
│    • Subscriptions/notifications                                        │
│    • Namespaced isolation                                               │
│                                                                          │
│  Handoff Patterns:                                                       │
│    • Explicit Handoff: Clear context transfer                          │
│    • Checkpoint-Based: Save/restore points                             │
│    • Pipeline State: Stage-by-stage progression                        │
│                                                                          │
│  Best Practices:                                                         │
│    • Use locks for concurrent access                                   │
│    • Track change history                                              │
│    • Include context in handoffs                                       │
│    • Subscribe to relevant changes                                     │
│                                                                          │
│  Use Cases:                                                              │
│    • Multi-stage content creation                                      │
│    • Collaborative research                                            │
│    • Review and approval workflows                                     │
│    • Long-running task coordination                                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

Next: [Writer-Reviewer Lab](/learn/multi-agents/building-multi-agent/writer-reviewer) →
