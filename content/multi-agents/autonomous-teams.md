# Autonomous Teams

Learn how to build self-organizing teams of agents that can plan, execute, and adapt without constant supervision.

## What are Autonomous Teams?

Autonomous teams are groups of agents that can independently plan their work, coordinate among themselves, and adapt to changing requirements.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   Autonomous Team Structure                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Traditional:                    Autonomous:                           │
│                                                                          │
│   Human constantly                Human provides                        │
│   directing agents                high-level goal                       │
│        │                              │                                  │
│        ▼                              ▼                                  │
│   ┌─────────┐                  ┌─────────────┐                          │
│   │ Agent 1 │◄─────────────►   │    Team     │                          │
│   └─────────┘                  │  ┌───┬───┐  │                          │
│        │                       │  │ A │ B │  │ Self-organizing          │
│   ┌────▼────┐                  │  ├───┼───┤  │                          │
│   │ Agent 2 │                  │  │ C │ D │  │                          │
│   └─────────┘                  │  └───┴───┘  │                          │
│                                └──────┬──────┘                          │
│   Step-by-step                       │                                  │
│   instructions                       ▼                                  │
│                                   Result                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Team Planning System

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import asyncio
from datetime import datetime

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class TeamTask:
    """A task for the team."""
    id: str
    description: str
    assigned_to: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    priority: int = 1
    estimated_effort: int = 1  # 1-5 scale

@dataclass
class TeamPlan:
    """A plan created by the team."""
    goal: str
    tasks: List[TeamTask]
    created_at: datetime = field(default_factory=datetime.now)
    estimated_completion: Optional[datetime] = None


class TeamPlanner:
    """Plans work for an autonomous team."""
    
    def __init__(self, llm_client: Any):
        self.llm = llm_client
    
    async def create_plan(
        self,
        goal: str,
        team_capabilities: Dict[str, List[str]],
        constraints: List[str] = None
    ) -> TeamPlan:
        """Create a plan to achieve a goal."""
        
        capabilities_text = "\n".join(
            f"- {name}: {', '.join(caps)}"
            for name, caps in team_capabilities.items()
        )
        
        constraints_text = "\n".join(f"- {c}" for c in (constraints or []))
        
        prompt = f"""Create a detailed plan to achieve this goal:

Goal: {goal}

Team Capabilities:
{capabilities_text}

{f"Constraints:{chr(10)}{constraints_text}" if constraints else ""}

Create a plan with specific tasks. For each task provide:
1. A clear description
2. Which team member should do it
3. Dependencies on other tasks
4. Estimated effort (1-5)
5. Priority (1-5, 1 is highest)

Return as JSON:
{{
    "tasks": [
        {{
            "id": "task_1",
            "description": "...",
            "assigned_to": "team_member_name",
            "dependencies": [],
            "estimated_effort": 2,
            "priority": 1
        }}
    ]
}}"""
        
        response = await self.llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        import json
        try:
            data = json.loads(response.choices[0].message.content)
            tasks = [
                TeamTask(
                    id=t["id"],
                    description=t["description"],
                    assigned_to=t.get("assigned_to"),
                    dependencies=t.get("dependencies", []),
                    priority=t.get("priority", 1),
                    estimated_effort=t.get("estimated_effort", 1)
                )
                for t in data["tasks"]
            ]
        except json.JSONDecodeError:
            # Fallback to single task
            tasks = [TeamTask(id="task_1", description=goal)]
        
        return TeamPlan(goal=goal, tasks=tasks)
    
    async def replan(
        self,
        original_plan: TeamPlan,
        completed_tasks: List[str],
        issues: List[str],
        team_capabilities: Dict[str, List[str]]
    ) -> TeamPlan:
        """Replan based on progress and issues."""
        
        completed_text = "\n".join(f"- {t}" for t in completed_tasks)
        issues_text = "\n".join(f"- {i}" for i in issues)
        remaining_tasks = [
            t for t in original_plan.tasks
            if t.id not in completed_tasks
        ]
        remaining_text = "\n".join(
            f"- {t.id}: {t.description} (status: {t.status.value})"
            for t in remaining_tasks
        )
        
        prompt = f"""Replan based on current progress:

Original Goal: {original_plan.goal}

Completed Tasks:
{completed_text or "None"}

Remaining Tasks:
{remaining_text}

Issues Encountered:
{issues_text or "None"}

Create an updated plan that:
1. Addresses the issues
2. Adjusts priorities if needed
3. Adds new tasks if required
4. Removes tasks that are no longer needed

Return as JSON with the same format as before."""
        
        response = await self.llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        import json
        try:
            data = json.loads(response.choices[0].message.content)
            tasks = [
                TeamTask(
                    id=t["id"],
                    description=t["description"],
                    assigned_to=t.get("assigned_to"),
                    dependencies=t.get("dependencies", []),
                    priority=t.get("priority", 1),
                    estimated_effort=t.get("estimated_effort", 1)
                )
                for t in data["tasks"]
            ]
        except json.JSONDecodeError:
            return original_plan
        
        return TeamPlan(goal=original_plan.goal, tasks=tasks)
```

## Autonomous Team Implementation

```python
class AutonomousTeamMember:
    """A self-directed team member."""
    
    def __init__(
        self,
        name: str,
        capabilities: List[str],
        llm_client: Any
    ):
        self.name = name
        self.capabilities = capabilities
        self.llm = llm_client
        self.current_task: Optional[TeamTask] = None
        self.completed_tasks: List[str] = []
    
    async def can_handle(self, task: TeamTask) -> bool:
        """Check if this member can handle a task."""
        
        prompt = f"""Can you handle this task given your capabilities?

Task: {task.description}

Your capabilities: {', '.join(self.capabilities)}

Answer only YES or NO."""
        
        response = await self.llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return "YES" in response.choices[0].message.content.upper()
    
    async def execute_task(self, task: TeamTask, context: Dict[str, Any] = None) -> Any:
        """Execute a task."""
        
        self.current_task = task
        task.status = TaskStatus.IN_PROGRESS
        
        # Get context from dependencies
        deps_context = ""
        if context and task.dependencies:
            deps_context = "\n".join(
                f"- {dep}: {context.get(dep, 'N/A')}"
                for dep in task.dependencies
            )
        
        prompt = f"""Complete this task:

Task: {task.description}

Your role: {self.name}
Your capabilities: {', '.join(self.capabilities)}

{f"Context from previous tasks:{chr(10)}{deps_context}" if deps_context else ""}

Provide a complete, high-quality result."""
        
        response = await self.llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        result = response.choices[0].message.content
        
        task.result = result
        task.status = TaskStatus.COMPLETED
        self.completed_tasks.append(task.id)
        self.current_task = None
        
        return result
    
    async def assess_blockers(self, task: TeamTask) -> List[str]:
        """Assess what's blocking a task."""
        
        prompt = f"""What might block this task?

Task: {task.description}
Dependencies: {task.dependencies}

List any potential blockers or missing information."""
        
        response = await self.llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return [response.choices[0].message.content]


class AutonomousTeam:
    """A self-organizing team of agents."""
    
    def __init__(self, llm_client: Any):
        self.llm = llm_client
        self.members: Dict[str, AutonomousTeamMember] = {}
        self.planner = TeamPlanner(llm_client)
        self.current_plan: Optional[TeamPlan] = None
        self.task_results: Dict[str, Any] = {}
        self.issues: List[str] = []
    
    def add_member(self, name: str, capabilities: List[str]):
        """Add a team member."""
        self.members[name] = AutonomousTeamMember(name, capabilities, self.llm)
    
    async def work_on_goal(
        self,
        goal: str,
        constraints: List[str] = None,
        max_iterations: int = 10
    ) -> Dict[str, Any]:
        """Work autonomously on a goal."""
        
        # Create initial plan
        capabilities = {
            name: member.capabilities
            for name, member in self.members.items()
        }
        
        self.current_plan = await self.planner.create_plan(
            goal, capabilities, constraints
        )
        
        print(f"Created plan with {len(self.current_plan.tasks)} tasks")
        
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Get ready tasks
            ready_tasks = self._get_ready_tasks()
            
            if not ready_tasks:
                # Check if done
                if self._all_complete():
                    break
                
                # Check for deadlock
                if self._is_deadlocked():
                    await self._handle_deadlock()
                    continue
                
                await asyncio.sleep(0.1)
                continue
            
            # Execute ready tasks in parallel
            await self._execute_tasks(ready_tasks)
            
            # Check for issues and replan if needed
            if len(self.issues) > 2:
                await self._replan()
        
        return {
            "goal": goal,
            "status": "completed" if self._all_complete() else "incomplete",
            "iterations": iteration,
            "results": self.task_results,
            "issues": self.issues
        }
    
    def _get_ready_tasks(self) -> List[TeamTask]:
        """Get tasks ready to execute."""
        
        ready = []
        
        for task in self.current_plan.tasks:
            if task.status != TaskStatus.PENDING:
                continue
            
            # Check dependencies
            deps_met = all(
                self.current_plan.tasks[
                    next(i for i, t in enumerate(self.current_plan.tasks) if t.id == dep)
                ].status == TaskStatus.COMPLETED
                for dep in task.dependencies
                if any(t.id == dep for t in self.current_plan.tasks)
            )
            
            if deps_met:
                ready.append(task)
        
        # Sort by priority
        ready.sort(key=lambda t: t.priority)
        
        return ready
    
    async def _execute_tasks(self, tasks: List[TeamTask]):
        """Execute tasks, assigning to appropriate members."""
        
        for task in tasks:
            # Find member
            member = None
            
            if task.assigned_to and task.assigned_to in self.members:
                member = self.members[task.assigned_to]
            else:
                # Find capable member
                for m in self.members.values():
                    if m.current_task is None:
                        if await m.can_handle(task):
                            member = m
                            break
            
            if member:
                try:
                    result = await member.execute_task(task, self.task_results)
                    self.task_results[task.id] = result
                    print(f"[{member.name}] Completed: {task.id}")
                except Exception as e:
                    task.status = TaskStatus.FAILED
                    self.issues.append(f"Task {task.id} failed: {str(e)}")
            else:
                task.status = TaskStatus.BLOCKED
                self.issues.append(f"No member available for: {task.id}")
    
    def _all_complete(self) -> bool:
        """Check if all tasks are complete."""
        return all(
            t.status == TaskStatus.COMPLETED
            for t in self.current_plan.tasks
        )
    
    def _is_deadlocked(self) -> bool:
        """Check for deadlock."""
        pending = [t for t in self.current_plan.tasks if t.status == TaskStatus.PENDING]
        blocked = [t for t in self.current_plan.tasks if t.status == TaskStatus.BLOCKED]
        
        return len(pending) == 0 and len(blocked) > 0
    
    async def _handle_deadlock(self):
        """Handle a deadlocked situation."""
        
        blocked_tasks = [
            t for t in self.current_plan.tasks
            if t.status == TaskStatus.BLOCKED
        ]
        
        self.issues.append(f"Deadlock detected with {len(blocked_tasks)} blocked tasks")
        
        # Try to unblock
        for task in blocked_tasks:
            task.status = TaskStatus.PENDING
            task.dependencies = []  # Remove dependencies as last resort
    
    async def _replan(self):
        """Replan based on issues."""
        
        completed = [t.id for t in self.current_plan.tasks if t.status == TaskStatus.COMPLETED]
        
        capabilities = {
            name: member.capabilities
            for name, member in self.members.items()
        }
        
        self.current_plan = await self.planner.replan(
            self.current_plan,
            completed,
            self.issues[-5:],  # Last 5 issues
            capabilities
        )
        
        self.issues = []  # Clear issues after replan
        print("Replanned based on issues")
```

## Team Collaboration Patterns

### Pattern 1: Consensus Building

```python
class ConsensusTeam(AutonomousTeam):
    """Team that builds consensus on decisions."""
    
    async def make_decision(
        self,
        question: str,
        options: List[str]
    ) -> Dict[str, Any]:
        """Make a team decision through consensus."""
        
        # Get initial opinions
        opinions = {}
        for name, member in self.members.items():
            opinion = await self._get_opinion(member, question, options)
            opinions[name] = opinion
        
        # Discussion rounds
        rounds = 0
        max_rounds = 3
        
        while rounds < max_rounds:
            rounds += 1
            
            # Check for consensus
            votes = list(opinions.values())
            if len(set(votes)) == 1:
                return {
                    "decision": votes[0],
                    "unanimous": True,
                    "rounds": rounds
                }
            
            # Share opinions and update
            new_opinions = {}
            for name, member in self.members.items():
                other_opinions = {k: v for k, v in opinions.items() if k != name}
                new_opinion = await self._update_opinion(
                    member, question, options, other_opinions
                )
                new_opinions[name] = new_opinion
            
            opinions = new_opinions
        
        # No consensus - majority vote
        from collections import Counter
        vote_counts = Counter(opinions.values())
        winner = vote_counts.most_common(1)[0][0]
        
        return {
            "decision": winner,
            "unanimous": False,
            "rounds": rounds,
            "votes": dict(vote_counts)
        }
    
    async def _get_opinion(
        self,
        member: AutonomousTeamMember,
        question: str,
        options: List[str]
    ) -> str:
        """Get a member's opinion."""
        
        prompt = f"""As {member.name} with expertise in {', '.join(member.capabilities)}:

Question: {question}

Options:
{chr(10).join(f"- {opt}" for opt in options)}

Which option do you recommend and why? State your choice clearly."""
        
        response = await self.llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Extract choice
        text = response.choices[0].message.content
        for opt in options:
            if opt.lower() in text.lower():
                return opt
        
        return options[0]
    
    async def _update_opinion(
        self,
        member: AutonomousTeamMember,
        question: str,
        options: List[str],
        other_opinions: Dict[str, str]
    ) -> str:
        """Update opinion based on team discussion."""
        
        opinions_text = "\n".join(
            f"- {name}: chose {choice}"
            for name, choice in other_opinions.items()
        )
        
        prompt = f"""Reconsider your position after hearing from teammates:

Question: {question}

Team opinions:
{opinions_text}

Options:
{chr(10).join(f"- {opt}" for opt in options)}

Do you want to change your answer? State your final choice."""
        
        response = await self.llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        text = response.choices[0].message.content
        for opt in options:
            if opt.lower() in text.lower():
                return opt
        
        return options[0]
```

### Pattern 2: Self-Improving Team

```python
class SelfImprovingTeam(AutonomousTeam):
    """Team that learns and improves over time."""
    
    def __init__(self, llm_client: Any):
        super().__init__(llm_client)
        self.retrospectives: List[Dict[str, Any]] = []
        self.learned_patterns: List[str] = []
    
    async def work_on_goal(
        self,
        goal: str,
        constraints: List[str] = None,
        max_iterations: int = 10
    ) -> Dict[str, Any]:
        """Work on goal with retrospective."""
        
        result = await super().work_on_goal(goal, constraints, max_iterations)
        
        # Conduct retrospective
        retro = await self._retrospective(result)
        self.retrospectives.append(retro)
        
        # Extract learnings
        await self._extract_learnings(retro)
        
        return {**result, "retrospective": retro}
    
    async def _retrospective(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct a team retrospective."""
        
        prompt = f"""Conduct a retrospective on this completed work:

Goal: {result['goal']}
Status: {result['status']}
Iterations: {result['iterations']}
Issues Encountered: {result['issues']}

Analyze:
1. What went well?
2. What could be improved?
3. What patterns should we remember?
4. What should we do differently next time?

Return as JSON:
{{
    "went_well": ["..."],
    "improvements": ["..."],
    "patterns": ["..."],
    "action_items": ["..."]
}}"""
        
        response = await self.llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        import json
        try:
            return json.loads(response.choices[0].message.content)
        except:
            return {"raw": response.choices[0].message.content}
    
    async def _extract_learnings(self, retro: Dict[str, Any]):
        """Extract and store learnings."""
        
        patterns = retro.get("patterns", [])
        self.learned_patterns.extend(patterns)
        
        # Keep only recent patterns
        self.learned_patterns = self.learned_patterns[-20:]
```

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│               Autonomous Teams - Summary                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Key Components:                                                         │
│    • Planner - Creates and adjusts plans                                │
│    • Team Members - Execute tasks autonomously                          │
│    • Coordinator - Manages workflow                                     │
│                                                                          │
│  Capabilities:                                                           │
│    • Self-planning from high-level goals                               │
│    • Task assignment based on capabilities                             │
│    • Dependency management                                             │
│    • Replanning when issues arise                                      │
│    • Consensus building                                                │
│                                                                          │
│  Benefits:                                                               │
│    • Less human oversight needed                                       │
│    • Adapts to changes                                                 │
│    • Handles complex multi-step work                                   │
│    • Can learn and improve                                             │
│                                                                          │
│  Best For:                                                               │
│    • Complex projects with many parts                                  │
│    • Work requiring multiple specialties                               │
│    • Situations needing adaptation                                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

Next: [Human-in-the-Loop](/learn/multi-agents/advanced-multi-agent/human-in-the-loop) →
