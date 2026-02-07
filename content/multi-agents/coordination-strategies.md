# Coordination Strategies

Learn how to coordinate multiple agents to work together effectively and avoid conflicts.

## The Coordination Challenge

When multiple agents work together, they need coordination to:
- Avoid duplicating work
- Maintain consistency
- Handle dependencies
- Resolve conflicts

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   Coordination Challenges                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Without Coordination:                                                  │
│                                                                          │
│   Agent A: "I'll research topic X"                                      │
│   Agent B: "I'll research topic X"    ← Duplicate work                  │
│   Agent C: "I need X but it's not ready" ← Dependency issue             │
│                                                                          │
│   Agent A writes: "The answer is 42"                                    │
│   Agent B writes: "The answer is 37"  ← Conflict!                       │
│                                                                          │
│   With Coordination:                                                     │
│                                                                          │
│   Coordinator: "A does X, B does Y, C waits for A"                      │
│   Agent A: "X complete"                                                 │
│   Agent C: "Proceeding with X results"  ← Dependencies met             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Strategy 1: Centralized Coordinator

A single coordinator agent manages all other agents.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   Centralized Coordinator                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                        ┌─────────────────┐                              │
│                        │   Coordinator   │                              │
│                        │    (Manager)    │                              │
│                        └────────┬────────┘                              │
│                                 │                                        │
│               ┌─────────────────┼─────────────────┐                     │
│               │                 │                 │                     │
│               ▼                 ▼                 ▼                     │
│        ┌───────────┐     ┌───────────┐     ┌───────────┐              │
│        │  Worker A │     │  Worker B │     │  Worker C │              │
│        └───────────┘     └───────────┘     └───────────┘              │
│                                                                          │
│   Coordinator responsibilities:                                          │
│   • Assign tasks to workers                                             │
│   • Track progress                                                      │
│   • Handle dependencies                                                 │
│   • Aggregate results                                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum
import asyncio

class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    id: str
    description: str
    assigned_to: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = None
    result: Any = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

class CentralizedCoordinator:
    """Coordinator that manages worker agents."""
    
    def __init__(self, name: str = "coordinator"):
        self.name = name
        self.workers: Dict[str, "WorkerAgent"] = {}
        self.tasks: Dict[str, Task] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
    
    def register_worker(self, worker: "WorkerAgent"):
        """Register a worker agent."""
        self.workers[worker.name] = worker
        worker.coordinator = self
    
    def add_task(self, task: Task):
        """Add a task to be coordinated."""
        self.tasks[task.id] = task
    
    def get_ready_tasks(self) -> List[Task]:
        """Get tasks whose dependencies are met."""
        ready = []
        
        for task in self.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            
            # Check dependencies
            deps_met = all(
                self.tasks[dep_id].status == TaskStatus.COMPLETED
                for dep_id in task.dependencies
                if dep_id in self.tasks
            )
            
            if deps_met:
                ready.append(task)
        
        return ready
    
    def get_available_workers(self) -> List["WorkerAgent"]:
        """Get workers that are not busy."""
        return [w for w in self.workers.values() if not w.is_busy]
    
    async def assign_task(self, task: Task, worker: "WorkerAgent"):
        """Assign a task to a worker."""
        task.assigned_to = worker.name
        task.status = TaskStatus.ASSIGNED
        
        print(f"[Coordinator] Assigned '{task.id}' to {worker.name}")
        
        # Execute the task
        try:
            result = await worker.execute(task)
            task.result = result
            task.status = TaskStatus.COMPLETED
            print(f"[Coordinator] Task '{task.id}' completed")
        except Exception as e:
            task.status = TaskStatus.FAILED
            print(f"[Coordinator] Task '{task.id}' failed: {e}")
    
    async def run(self) -> Dict[str, Any]:
        """Run coordination until all tasks complete."""
        
        while True:
            # Get tasks ready to run
            ready_tasks = self.get_ready_tasks()
            available_workers = self.get_available_workers()
            
            if not ready_tasks:
                # Check if all done
                all_completed = all(
                    t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
                    for t in self.tasks.values()
                )
                
                if all_completed:
                    break
                else:
                    # Wait for running tasks
                    await asyncio.sleep(0.1)
                    continue
            
            # Assign tasks to available workers
            assignments = []
            for task in ready_tasks:
                if not available_workers:
                    break
                
                worker = available_workers.pop(0)
                assignments.append(self.assign_task(task, worker))
            
            if assignments:
                await asyncio.gather(*assignments)
        
        # Return all results
        return {
            task_id: task.result
            for task_id, task in self.tasks.items()
            if task.status == TaskStatus.COMPLETED
        }


class WorkerAgent:
    """Worker agent managed by coordinator."""
    
    def __init__(self, name: str, capabilities: List[str] = None):
        self.name = name
        self.capabilities = capabilities or []
        self.coordinator: Optional[CentralizedCoordinator] = None
        self.is_busy = False
    
    async def execute(self, task: Task) -> Any:
        """Execute a task."""
        self.is_busy = True
        task.status = TaskStatus.IN_PROGRESS
        
        try:
            # Simulate work
            result = await self._do_work(task)
            return result
        finally:
            self.is_busy = False
    
    async def _do_work(self, task: Task) -> Any:
        """Override in subclass to do actual work."""
        raise NotImplementedError


# Example usage
async def coordinated_workflow():
    coordinator = CentralizedCoordinator()
    
    # Create workers
    researcher = ResearchWorker("researcher")
    writer = WriterWorker("writer")
    editor = EditorWorker("editor")
    
    coordinator.register_worker(researcher)
    coordinator.register_worker(writer)
    coordinator.register_worker(editor)
    
    # Define tasks with dependencies
    coordinator.add_task(Task(
        id="research",
        description="Research the topic"
    ))
    
    coordinator.add_task(Task(
        id="write",
        description="Write the article",
        dependencies=["research"]  # Depends on research
    ))
    
    coordinator.add_task(Task(
        id="edit",
        description="Edit the article",
        dependencies=["write"]  # Depends on writing
    ))
    
    # Run workflow
    results = await coordinator.run()
    return results
```

## Strategy 2: Consensus-Based Coordination

Agents reach consensus through voting or agreement protocols.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   Consensus-Based Coordination                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Proposal Phase:                                                        │
│   ┌───────────┐                                                         │
│   │  Agent A  │ ─── "I propose we do X" ───────────►  All Agents       │
│   └───────────┘                                                         │
│                                                                          │
│   Voting Phase:                                                          │
│   ┌───────────┐                                                         │
│   │  Agent A  │ ─── Vote: Yes ───┐                                     │
│   └───────────┘                  │                                      │
│   ┌───────────┐                  ├──► Collect Votes                    │
│   │  Agent B  │ ─── Vote: Yes ───┤                                     │
│   └───────────┘                  │                                      │
│   ┌───────────┐                  │                                      │
│   │  Agent C  │ ─── Vote: No ────┘                                     │
│   └───────────┘                                                         │
│                                                                          │
│   Decision: 2/3 majority → Proceed with X                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

class Vote(Enum):
    YES = "yes"
    NO = "no"
    ABSTAIN = "abstain"

@dataclass
class Proposal:
    id: str
    content: str
    proposer: str
    votes: Dict[str, Vote] = None
    
    def __post_init__(self):
        if self.votes is None:
            self.votes = {}
    
    def count_votes(self) -> Dict[Vote, int]:
        counts = {Vote.YES: 0, Vote.NO: 0, Vote.ABSTAIN: 0}
        for vote in self.votes.values():
            counts[vote] += 1
        return counts
    
    def is_approved(self, threshold: float = 0.5) -> bool:
        counts = self.count_votes()
        total_votes = counts[Vote.YES] + counts[Vote.NO]
        if total_votes == 0:
            return False
        return counts[Vote.YES] / total_votes > threshold

class ConsensusAgent:
    """Agent that participates in consensus decisions."""
    
    def __init__(self, name: str, llm_client):
        self.name = name
        self.llm = llm_client
        self.peers: List["ConsensusAgent"] = []
    
    def add_peer(self, peer: "ConsensusAgent"):
        self.peers.append(peer)
    
    async def propose(self, content: str) -> Proposal:
        """Make a proposal to all peers."""
        proposal = Proposal(
            id=f"prop_{self.name}_{len(self.peers)}",
            content=content,
            proposer=self.name
        )
        
        # Collect votes from all peers
        for peer in self.peers:
            vote = await peer.vote_on(proposal)
            proposal.votes[peer.name] = vote
        
        # Add own vote
        proposal.votes[self.name] = Vote.YES
        
        return proposal
    
    async def vote_on(self, proposal: Proposal) -> Vote:
        """Vote on a proposal using LLM reasoning."""
        
        prompt = f"""
You are {self.name}. You need to vote on a proposal.

Proposal: {proposal.content}
Proposed by: {proposal.proposer}

Consider:
1. Is this a good idea?
2. Are there any risks?
3. Does it align with our goals?

Respond with exactly one of: YES, NO, or ABSTAIN
"""
        
        response = await self.llm.chat(prompt)
        vote_text = response.strip().upper()
        
        if "YES" in vote_text:
            return Vote.YES
        elif "NO" in vote_text:
            return Vote.NO
        else:
            return Vote.ABSTAIN


class ConsensusCoordinator:
    """Coordinate consensus-based decisions."""
    
    def __init__(self, agents: List[ConsensusAgent], threshold: float = 0.5):
        self.agents = agents
        self.threshold = threshold
        self.decisions: List[Proposal] = []
        
        # Connect all agents as peers
        for agent in agents:
            for other in agents:
                if other != agent:
                    agent.add_peer(other)
    
    async def decide(self, question: str, proposer_name: str) -> bool:
        """Make a consensus decision on a question."""
        
        proposer = next(a for a in self.agents if a.name == proposer_name)
        proposal = await proposer.propose(question)
        
        self.decisions.append(proposal)
        
        approved = proposal.is_approved(self.threshold)
        counts = proposal.count_votes()
        
        print(f"Proposal: {question}")
        print(f"Votes: Yes={counts[Vote.YES]}, No={counts[Vote.NO]}, Abstain={counts[Vote.ABSTAIN]}")
        print(f"Decision: {'APPROVED' if approved else 'REJECTED'}")
        
        return approved
```

## Strategy 3: Market-Based Coordination

Agents bid for tasks based on capability and availability.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   Market-Based Coordination                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Task Announcement:                                                     │
│   ┌──────────────────────────────────────────────────────────────┐     │
│   │  Task: "Write Python code"  Reward: 100 points               │     │
│   └──────────────────────────────────────────────────────────────┘     │
│                                                                          │
│   Bidding:                                                               │
│   ┌───────────┐  Bid: 80 pts, Quality: High, Time: 5min                │
│   │  Agent A  │ ────────────────────────────────────────►              │
│   └───────────┘                                                         │
│   ┌───────────┐  Bid: 60 pts, Quality: Medium, Time: 3min              │
│   │  Agent B  │ ────────────────────────────────────────►              │
│   └───────────┘                                                         │
│   ┌───────────┐  Bid: 90 pts, Quality: High, Time: 4min                │
│   │  Agent C  │ ────────────────────────────────────────►              │
│   └───────────┘                                                         │
│                                                                          │
│   Award: Agent B wins (best value: low cost, acceptable quality)        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
from dataclasses import dataclass
from typing import List, Optional
import asyncio

@dataclass
class Bid:
    agent_name: str
    price: float  # How much agent wants for the task
    estimated_time: float  # Minutes
    confidence: float  # 0-1 scale
    
    def score(self, weights: dict = None) -> float:
        """Calculate bid score (lower is better)."""
        weights = weights or {"price": 0.4, "time": 0.3, "confidence": 0.3}
        
        # Normalize and combine (confidence is inverted - higher is better)
        return (
            weights["price"] * self.price / 100 +
            weights["time"] * self.estimated_time / 60 -
            weights["confidence"] * self.confidence
        )

@dataclass
class TaskAuction:
    task_id: str
    description: str
    min_confidence: float = 0.5
    bids: List[Bid] = None
    winner: Optional[str] = None
    
    def __post_init__(self):
        if self.bids is None:
            self.bids = []
    
    def select_winner(self) -> Optional[Bid]:
        """Select the winning bid."""
        valid_bids = [b for b in self.bids if b.confidence >= self.min_confidence]
        
        if not valid_bids:
            return None
        
        # Sort by score (lower is better)
        valid_bids.sort(key=lambda b: b.score())
        
        winner = valid_bids[0]
        self.winner = winner.agent_name
        
        return winner

class MarketAgent:
    """Agent that bids on tasks in a market."""
    
    def __init__(
        self,
        name: str,
        capabilities: List[str],
        base_price: float = 50.0
    ):
        self.name = name
        self.capabilities = capabilities
        self.base_price = base_price
        self.current_load = 0  # Number of active tasks
    
    async def evaluate_task(self, task: TaskAuction) -> Optional[Bid]:
        """Evaluate a task and submit a bid."""
        
        # Check if we can do this task
        can_do = any(
            cap in task.description.lower()
            for cap in self.capabilities
        )
        
        if not can_do:
            return None
        
        # Calculate bid based on load and capability
        load_multiplier = 1 + (self.current_load * 0.2)
        
        # Estimate confidence based on capability match
        capability_matches = sum(
            1 for cap in self.capabilities
            if cap in task.description.lower()
        )
        confidence = min(0.9, 0.5 + (capability_matches * 0.2))
        
        bid = Bid(
            agent_name=self.name,
            price=self.base_price * load_multiplier,
            estimated_time=10 / confidence,  # Better capability = faster
            confidence=confidence
        )
        
        return bid
    
    async def execute_task(self, task: TaskAuction) -> str:
        """Execute a won task."""
        self.current_load += 1
        
        try:
            # Simulate work
            result = f"Result from {self.name} for {task.task_id}"
            return result
        finally:
            self.current_load -= 1

class TaskMarket:
    """Market for coordinating task assignment via bidding."""
    
    def __init__(self):
        self.agents: List[MarketAgent] = []
        self.completed_auctions: List[TaskAuction] = []
    
    def register_agent(self, agent: MarketAgent):
        """Register an agent in the market."""
        self.agents.append(agent)
    
    async def auction_task(self, task_id: str, description: str) -> Optional[str]:
        """Run an auction for a task."""
        
        auction = TaskAuction(
            task_id=task_id,
            description=description
        )
        
        # Collect bids
        for agent in self.agents:
            bid = await agent.evaluate_task(auction)
            if bid:
                auction.bids.append(bid)
                print(f"  {agent.name} bids: {bid.price:.0f} pts, {bid.confidence:.0%} confidence")
        
        # Select winner
        winner_bid = auction.select_winner()
        
        if winner_bid:
            print(f"  Winner: {winner_bid.agent_name}")
            
            # Execute task
            winner_agent = next(a for a in self.agents if a.name == winner_bid.agent_name)
            result = await winner_agent.execute_task(auction)
            
            self.completed_auctions.append(auction)
            return result
        
        print("  No valid bids received")
        return None


# Example usage
async def market_coordination():
    market = TaskMarket()
    
    # Register agents with different capabilities
    market.register_agent(MarketAgent(
        "python_expert",
        capabilities=["python", "code", "algorithm"],
        base_price=60
    ))
    
    market.register_agent(MarketAgent(
        "writer",
        capabilities=["write", "content", "documentation"],
        base_price=40
    ))
    
    market.register_agent(MarketAgent(
        "generalist",
        capabilities=["python", "write", "research"],
        base_price=50
    ))
    
    # Auction tasks
    tasks = [
        ("task_1", "Write Python code for sorting"),
        ("task_2", "Write documentation for the API"),
        ("task_3", "Research best practices for testing"),
    ]
    
    for task_id, description in tasks:
        print(f"\nAuctioning: {description}")
        result = await market.auction_task(task_id, description)
```

## Strategy 4: Role-Based Coordination

Agents have defined roles with clear responsibilities and handoff protocols.

```python
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

class AgentRole(Enum):
    PLANNER = "planner"       # Plans the overall approach
    EXECUTOR = "executor"     # Executes individual tasks
    REVIEWER = "reviewer"     # Reviews and validates work
    INTEGRATOR = "integrator" # Combines results

@dataclass
class WorkItem:
    id: str
    content: Any
    stage: str
    created_by: str
    approved_by: List[str] = None
    
    def __post_init__(self):
        if self.approved_by is None:
            self.approved_by = []

class RoleBasedCoordinator:
    """Coordinate agents based on their roles."""
    
    def __init__(self):
        self.agents: Dict[AgentRole, List["RoleAgent"]] = {
            role: [] for role in AgentRole
        }
        self.workflow: List[WorkItem] = []
    
    def register_agent(self, agent: "RoleAgent"):
        """Register an agent with its role."""
        self.agents[agent.role].append(agent)
    
    async def execute_workflow(self, task: str) -> Any:
        """Execute a role-based workflow."""
        
        # Step 1: Planning
        planners = self.agents[AgentRole.PLANNER]
        if not planners:
            raise ValueError("No planner available")
        
        plan = await planners[0].create_plan(task)
        self.workflow.append(WorkItem(
            id="plan",
            content=plan,
            stage="planning",
            created_by=planners[0].name
        ))
        
        # Step 2: Execution
        executors = self.agents[AgentRole.EXECUTOR]
        results = []
        
        for i, subtask in enumerate(plan["subtasks"]):
            executor = executors[i % len(executors)]
            result = await executor.execute(subtask)
            
            self.workflow.append(WorkItem(
                id=f"result_{i}",
                content=result,
                stage="execution",
                created_by=executor.name
            ))
            results.append(result)
        
        # Step 3: Review
        reviewers = self.agents[AgentRole.REVIEWER]
        if reviewers:
            for item in self.workflow:
                if item.stage == "execution":
                    approval = await reviewers[0].review(item.content)
                    if approval:
                        item.approved_by.append(reviewers[0].name)
        
        # Step 4: Integration
        integrators = self.agents[AgentRole.INTEGRATOR]
        if integrators:
            approved_results = [
                item.content for item in self.workflow
                if item.stage == "execution" and item.approved_by
            ]
            
            final = await integrators[0].integrate(approved_results)
            return final
        
        return results


class RoleAgent:
    """Agent with a specific role."""
    
    def __init__(self, name: str, role: AgentRole, llm_client):
        self.name = name
        self.role = role
        self.llm = llm_client
    
    async def create_plan(self, task: str) -> Dict:
        """Planner: Create a plan."""
        # Implementation
        pass
    
    async def execute(self, subtask: str) -> Any:
        """Executor: Execute a subtask."""
        # Implementation
        pass
    
    async def review(self, result: Any) -> bool:
        """Reviewer: Review and approve/reject."""
        # Implementation
        pass
    
    async def integrate(self, results: List[Any]) -> Any:
        """Integrator: Combine results."""
        # Implementation
        pass
```

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│              Coordination Strategies - Summary                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Centralized Coordinator:                                                │
│    • Single point of control                                            │
│    • Clear task assignment                                              │
│    • Best for: Structured workflows with dependencies                   │
│                                                                          │
│  Consensus-Based:                                                        │
│    • Democratic decision making                                         │
│    • Voting on proposals                                                │
│    • Best for: Important decisions needing agreement                    │
│                                                                          │
│  Market-Based:                                                           │
│    • Agents bid for tasks                                               │
│    • Self-organizing                                                    │
│    • Best for: Dynamic task allocation                                  │
│                                                                          │
│  Role-Based:                                                             │
│    • Clear responsibilities                                             │
│    • Defined handoff protocols                                          │
│    • Best for: Specialized team workflows                               │
│                                                                          │
│  Choose Based On:                                                        │
│    • Task structure and dependencies                                    │
│    • Need for flexibility vs control                                    │
│    • Agent specialization                                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

Next: [Sequential Workflows](/learn/multi-agents/orchestration-patterns/sequential-workflows) →
