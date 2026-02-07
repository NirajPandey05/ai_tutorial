# Agent Swarms

Learn how to build swarm-based multi-agent systems where many agents work together like a collective.

## What are Agent Swarms?

Agent swarms use many simple agents working together to accomplish complex tasks through emergent behavior.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Agent Swarm Overview                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Traditional (Hierarchical):        Swarm (Decentralized):             │
│                                                                          │
│         ┌─────────┐                    ○ ─── ○ ─── ○                    │
│         │ Manager │                   /│\   /│\   /│\                   │
│         └────┬────┘                  ○ ○ ○ ○ ○ ○ ○ ○ ○                  │
│        ┌────┼────┐                    \│/   \│/   \│/                   │
│        ▼    ▼    ▼                     ○ ─── ○ ─── ○                    │
│       [A]  [B]  [C]                                                      │
│                                      All agents equal                    │
│   Top-down control                   Self-organizing                     │
│   Single point of failure            Resilient                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Swarm Principles

```python
"""
Swarm Intelligence Principles:

1. DECENTRALIZATION - No single agent controls the swarm
2. SIMPLE RULES - Each agent follows simple local rules
3. EMERGENCE - Complex behavior emerges from simple interactions
4. REDUNDANCY - Many agents doing similar work provides resilience
5. STIGMERGY - Agents communicate through the environment
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum
import asyncio
import random

class SwarmState(Enum):
    EXPLORING = "exploring"
    WORKING = "working"
    SHARING = "sharing"
    IDLE = "idle"

@dataclass
class SwarmAgent:
    """A simple agent in a swarm."""
    id: str
    state: SwarmState = SwarmState.IDLE
    energy: float = 1.0  # Agents can "tire"
    knowledge: Dict[str, Any] = field(default_factory=dict)
    neighbors: Set[str] = field(default_factory=set)
```

## Basic Swarm Implementation

```python
from typing import Callable, Awaitable
import uuid

class SwarmEnvironment:
    """Shared environment for swarm agents."""
    
    def __init__(self):
        self.pheromones: Dict[str, float] = {}  # Topic -> strength
        self.shared_knowledge: Dict[str, List[Any]] = {}
        self.tasks: List[Dict[str, Any]] = []
        self.completed: List[Dict[str, Any]] = []
    
    def deposit_pheromone(self, topic: str, strength: float = 1.0):
        """Leave a signal about a topic."""
        current = self.pheromones.get(topic, 0)
        self.pheromones[topic] = min(current + strength, 10.0)
    
    def sense_pheromones(self, threshold: float = 0.5) -> List[str]:
        """Find topics with strong signals."""
        return [
            topic for topic, strength in self.pheromones.items()
            if strength >= threshold
        ]
    
    def decay_pheromones(self, rate: float = 0.1):
        """Pheromones decay over time."""
        for topic in list(self.pheromones.keys()):
            self.pheromones[topic] *= (1 - rate)
            if self.pheromones[topic] < 0.1:
                del self.pheromones[topic]
    
    def share_knowledge(self, topic: str, knowledge: Any):
        """Share knowledge to the environment."""
        if topic not in self.shared_knowledge:
            self.shared_knowledge[topic] = []
        self.shared_knowledge[topic].append(knowledge)
    
    def get_knowledge(self, topic: str) -> List[Any]:
        """Get shared knowledge about a topic."""
        return self.shared_knowledge.get(topic, [])


class SwarmMember:
    """An individual swarm agent."""
    
    def __init__(
        self,
        swarm_id: str,
        environment: SwarmEnvironment,
        llm_client: Any,
        specialty: str = None
    ):
        self.id = f"agent_{uuid.uuid4().hex[:8]}"
        self.swarm_id = swarm_id
        self.env = environment
        self.llm = llm_client
        self.specialty = specialty
        self.state = SwarmState.IDLE
        self.local_knowledge: Dict[str, Any] = {}
        self.task_count = 0
    
    async def act(self) -> Optional[Dict[str, Any]]:
        """Take an action based on current state and environment."""
        
        if self.state == SwarmState.IDLE:
            return await self._decide_action()
        elif self.state == SwarmState.EXPLORING:
            return await self._explore()
        elif self.state == SwarmState.WORKING:
            return await self._work()
        elif self.state == SwarmState.SHARING:
            return await self._share()
        
        return None
    
    async def _decide_action(self) -> Dict[str, Any]:
        """Decide what to do next."""
        
        # Check for strong pheromone signals
        hot_topics = self.env.sense_pheromones(threshold=0.5)
        
        if hot_topics and random.random() < 0.7:
            # Follow pheromone trail
            topic = random.choice(hot_topics)
            self.local_knowledge["current_topic"] = topic
            self.state = SwarmState.WORKING
            return {"action": "follow_signal", "topic": topic}
        
        # Check for available tasks
        if self.env.tasks:
            task = self.env.tasks.pop(0)
            self.local_knowledge["current_task"] = task
            self.state = SwarmState.WORKING
            return {"action": "take_task", "task": task}
        
        # Explore
        self.state = SwarmState.EXPLORING
        return {"action": "explore"}
    
    async def _explore(self) -> Dict[str, Any]:
        """Explore for new information."""
        
        # Simulate exploration - in real implementation, could search, browse, etc.
        prompt = f"""You are an explorer agent with specialty: {self.specialty or 'general'}.
        
Suggest one interesting topic or question to investigate that would be valuable for a team.
Be specific and actionable. Return just the topic/question."""
        
        response = await self.llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        discovery = response.choices[0].message.content.strip()
        
        # Leave pheromone and share
        self.env.deposit_pheromone(discovery, strength=0.5)
        self.local_knowledge["discovery"] = discovery
        self.state = SwarmState.SHARING
        
        return {"action": "discovered", "topic": discovery}
    
    async def _work(self) -> Dict[str, Any]:
        """Do work on current task/topic."""
        
        task = self.local_knowledge.get("current_task")
        topic = self.local_knowledge.get("current_topic")
        
        work_target = task.get("description") if task else topic
        
        if not work_target:
            self.state = SwarmState.IDLE
            return {"action": "no_work"}
        
        # Get existing knowledge
        existing = self.env.get_knowledge(work_target)
        existing_summary = "\n".join(str(k)[:100] for k in existing[:3])
        
        prompt = f"""You are a worker agent with specialty: {self.specialty or 'general'}.

Task/Topic: {work_target}

Existing knowledge from swarm:
{existing_summary if existing_summary else "None yet"}

Contribute new, unique knowledge or complete a portion of this task.
Build on what exists, don't repeat it.
Be concise and specific."""
        
        response = await self.llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        result = response.choices[0].message.content
        
        # Share result
        self.env.share_knowledge(work_target, {
            "agent": self.id,
            "contribution": result,
            "specialty": self.specialty
        })
        
        # Strengthen pheromone if task is important
        self.env.deposit_pheromone(work_target, strength=0.3)
        
        self.task_count += 1
        self.local_knowledge["last_work"] = result
        self.state = SwarmState.SHARING
        
        if task:
            self.env.completed.append({
                "task": task,
                "result": result,
                "agent": self.id
            })
        
        return {"action": "worked", "result": result[:200]}
    
    async def _share(self) -> Dict[str, Any]:
        """Share knowledge with neighbors."""
        
        # In swarm, sharing happens through environment
        # Reset state
        self.state = SwarmState.IDLE
        return {"action": "shared"}


class Swarm:
    """A swarm of agents working together."""
    
    def __init__(self, llm_client: Any, specialties: List[str] = None):
        self.llm = llm_client
        self.environment = SwarmEnvironment()
        self.agents: List[SwarmMember] = []
        self.specialties = specialties or ["general"]
    
    def spawn_agents(self, count: int):
        """Create swarm agents."""
        for i in range(count):
            specialty = self.specialties[i % len(self.specialties)]
            agent = SwarmMember(
                swarm_id=f"swarm_{id(self)}",
                environment=self.environment,
                llm_client=self.llm,
                specialty=specialty
            )
            self.agents.append(agent)
    
    def add_task(self, description: str, priority: int = 1):
        """Add a task for the swarm."""
        self.environment.tasks.append({
            "id": str(uuid.uuid4()),
            "description": description,
            "priority": priority
        })
    
    async def run_cycle(self) -> List[Dict[str, Any]]:
        """Run one cycle of swarm activity."""
        
        # All agents act in parallel
        results = await asyncio.gather(
            *[agent.act() for agent in self.agents]
        )
        
        # Decay pheromones
        self.environment.decay_pheromones()
        
        return results
    
    async def run(self, cycles: int = 10) -> Dict[str, Any]:
        """Run the swarm for multiple cycles."""
        
        all_results = []
        
        for cycle in range(cycles):
            results = await self.run_cycle()
            all_results.append({
                "cycle": cycle,
                "results": results,
                "active_topics": list(self.environment.pheromones.keys()),
                "completed_tasks": len(self.environment.completed)
            })
        
        return {
            "cycles_run": cycles,
            "total_completed": len(self.environment.completed),
            "knowledge_base": self.environment.shared_knowledge,
            "cycle_history": all_results
        }
    
    def aggregate_knowledge(self, topic: str) -> str:
        """Aggregate all knowledge about a topic."""
        
        knowledge = self.environment.get_knowledge(topic)
        
        if not knowledge:
            return "No knowledge gathered on this topic."
        
        contributions = "\n\n".join(
            f"[{k['agent']} - {k['specialty']}]: {k['contribution']}"
            for k in knowledge
        )
        
        return contributions
```

## Specialized Swarm Patterns

### Pattern 1: Research Swarm

```python
class ResearchSwarm(Swarm):
    """Swarm specialized for research tasks."""
    
    def __init__(self, llm_client: Any):
        super().__init__(
            llm_client,
            specialties=[
                "fact_finding",
                "analysis",
                "synthesis",
                "verification",
                "perspective"
            ]
        )
    
    async def research(self, question: str, depth: int = 3) -> Dict[str, Any]:
        """Research a question with the swarm."""
        
        # Spawn agents
        self.spawn_agents(count=5)
        
        # Add initial task
        self.add_task(question)
        
        # Run swarm
        await self.run(cycles=depth * 2)
        
        # Synthesize results
        synthesis = await self._synthesize(question)
        
        return {
            "question": question,
            "contributions": self.environment.get_knowledge(question),
            "synthesis": synthesis,
            "confidence": self._calculate_confidence(question)
        }
    
    async def _synthesize(self, question: str) -> str:
        """Synthesize all research into a coherent answer."""
        
        contributions = self.aggregate_knowledge(question)
        
        prompt = f"""Synthesize these research contributions into a coherent answer:

Question: {question}

Contributions from research swarm:
{contributions}

Provide a well-organized synthesis that:
1. Combines all relevant findings
2. Notes any conflicting information
3. Highlights key insights
4. Acknowledges limitations"""
        
        response = await self.llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
    
    def _calculate_confidence(self, topic: str) -> float:
        """Calculate confidence based on contribution count and diversity."""
        
        knowledge = self.environment.get_knowledge(topic)
        
        if not knowledge:
            return 0.0
        
        # More contributions = higher confidence
        count_score = min(len(knowledge) / 5, 1.0)
        
        # More diverse specialties = higher confidence
        specialties = set(k.get("specialty") for k in knowledge)
        diversity_score = len(specialties) / len(self.specialties)
        
        return (count_score + diversity_score) / 2


# Usage
async def swarm_research_example():
    swarm = ResearchSwarm(llm_client)
    
    result = await swarm.research(
        "What are the key challenges in deploying LLMs to production?",
        depth=3
    )
    
    print(f"Question: {result['question']}")
    print(f"Confidence: {result['confidence']:.0%}")
    print(f"\nSynthesis:\n{result['synthesis']}")
```

### Pattern 2: Creative Swarm

```python
class CreativeSwarm(Swarm):
    """Swarm for creative brainstorming."""
    
    def __init__(self, llm_client: Any):
        super().__init__(
            llm_client,
            specialties=[
                "wild_ideas",
                "practical_ideas",
                "combination",
                "critique",
                "refinement"
            ]
        )
    
    async def brainstorm(
        self,
        challenge: str,
        num_ideas: int = 10
    ) -> List[Dict[str, Any]]:
        """Generate ideas for a challenge."""
        
        self.spawn_agents(count=5)
        self.add_task(challenge)
        
        # More cycles = more ideas
        await self.run(cycles=num_ideas // 2)
        
        # Collect and rank ideas
        ideas = self._collect_ideas(challenge)
        ranked = await self._rank_ideas(ideas, challenge)
        
        return ranked[:num_ideas]
    
    def _collect_ideas(self, challenge: str) -> List[str]:
        """Collect all generated ideas."""
        
        knowledge = self.environment.get_knowledge(challenge)
        
        ideas = []
        for k in knowledge:
            # Extract ideas from contributions
            contribution = k.get("contribution", "")
            ideas.append({
                "idea": contribution,
                "specialty": k.get("specialty"),
                "agent": k.get("agent")
            })
        
        return ideas
    
    async def _rank_ideas(
        self,
        ideas: List[Dict],
        challenge: str
    ) -> List[Dict[str, Any]]:
        """Rank ideas by quality."""
        
        ideas_text = "\n".join(
            f"{i+1}. {idea['idea'][:200]}"
            for i, idea in enumerate(ideas)
        )
        
        prompt = f"""Rank these ideas for the challenge: {challenge}

Ideas:
{ideas_text}

Rate each idea 1-10 on:
- Originality
- Feasibility  
- Impact

Return JSON: [{{"idea_num": 1, "score": 8.5, "reason": "..."}}, ...]"""
        
        response = await self.llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        import json
        try:
            rankings = json.loads(response.choices[0].message.content)
            
            for ranking in rankings:
                idx = ranking["idea_num"] - 1
                if 0 <= idx < len(ideas):
                    ideas[idx]["score"] = ranking.get("score", 0)
                    ideas[idx]["reason"] = ranking.get("reason", "")
            
            return sorted(ideas, key=lambda x: x.get("score", 0), reverse=True)
        except:
            return ideas
```

## Swarm Coordination Strategies

```python
class CoordinatedSwarm(Swarm):
    """Swarm with coordination mechanisms."""
    
    def __init__(self, llm_client: Any, coordination_strategy: str = "democratic"):
        super().__init__(llm_client)
        self.strategy = coordination_strategy
    
    async def vote(self, question: str, options: List[str]) -> str:
        """Democratic voting among agents."""
        
        votes = {}
        
        for agent in self.agents:
            vote = await self._get_vote(agent, question, options)
            votes[vote] = votes.get(vote, 0) + 1
        
        winner = max(votes, key=votes.get)
        return winner
    
    async def _get_vote(
        self,
        agent: SwarmMember,
        question: str,
        options: List[str]
    ) -> str:
        """Get an agent's vote."""
        
        options_text = "\n".join(f"- {opt}" for opt in options)
        
        prompt = f"""Vote on this question:
{question}

Options:
{options_text}

Your specialty: {agent.specialty}
Consider from your perspective. Return only the option text."""
        
        response = await self.llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        vote = response.choices[0].message.content.strip()
        
        # Match to closest option
        for opt in options:
            if opt.lower() in vote.lower() or vote.lower() in opt.lower():
                return opt
        
        return options[0]  # Default
    
    async def reach_consensus(self, topic: str, max_rounds: int = 3) -> str:
        """Reach consensus through discussion."""
        
        positions = {}
        
        # Initial positions
        for agent in self.agents:
            position = await self._get_position(agent, topic)
            positions[agent.id] = position
        
        # Discussion rounds
        for round in range(max_rounds):
            new_positions = {}
            
            for agent in self.agents:
                other_positions = {
                    k: v for k, v in positions.items()
                    if k != agent.id
                }
                
                new_pos = await self._update_position(agent, topic, other_positions)
                new_positions[agent.id] = new_pos
            
            positions = new_positions
            
            # Check for consensus
            unique_positions = set(positions.values())
            if len(unique_positions) == 1:
                return list(unique_positions)[0]
        
        # No consensus - synthesize
        return await self._synthesize_positions(topic, positions)
```

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   Agent Swarms - Summary                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Principles:                                                             │
│    • Decentralization - No single controller                            │
│    • Simple rules - Local decisions                                     │
│    • Emergence - Complex behavior from simplicity                       │
│    • Stigmergy - Environmental communication                            │
│                                                                          │
│  Patterns:                                                               │
│    • Research Swarm - Parallel investigation                           │
│    • Creative Swarm - Brainstorming and ideation                       │
│    • Coordinated Swarm - Voting and consensus                          │
│                                                                          │
│  Benefits:                                                               │
│    • Resilient to failures                                             │
│    • Scales well                                                        │
│    • Diverse perspectives                                              │
│    • Self-organizing                                                   │
│                                                                          │
│  Best For:                                                               │
│    • Research and exploration                                          │
│    • Creative ideation                                                 │
│    • Tasks with uncertain scope                                        │
│    • Problems benefiting from diversity                                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

Next: [Autonomous Teams](/learn/multi-agents/advanced-multi-agent/autonomous-teams) →
