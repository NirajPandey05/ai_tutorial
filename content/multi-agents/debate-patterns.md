# Debate and Collaboration Patterns

Learn how to build multi-agent systems where agents debate, critique, and improve each other's work.

## Why Debate Patterns?

Debate patterns leverage multiple perspectives to improve output quality through constructive criticism and iteration.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Debate Pattern Overview                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Single Agent:          Debate Pattern:                                │
│                                                                          │
│   ┌─────────┐           ┌─────────┐     ┌─────────┐                    │
│   │ Agent A │           │ Agent A │ ◄─► │ Agent B │                    │
│   │         │           │(Proposer)│     │(Critic) │                    │
│   └────┬────┘           └────┬────┘     └────┬────┘                    │
│        │                     │  Iterate     │                          │
│        ▼                     └──────┬───────┘                          │
│   First Draft                       ▼                                   │
│   (No Review)               Improved Output                             │
│                            (Multiple Reviews)                           │
│                                                                          │
│   Quality: Medium          Quality: Higher                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Pattern 1: Proposer-Critic

One agent proposes, another critiques, iterate until good.

```python
from dataclasses import dataclass
from typing import List, Optional
import asyncio

@dataclass
class DebateRound:
    """A single round of debate."""
    round_number: int
    proposal: str
    critique: str
    is_approved: bool

@dataclass
class DebateResult:
    """Final result of a debate."""
    final_output: str
    rounds: List[DebateRound]
    total_rounds: int
    approved: bool


class ProposerAgent:
    """Agent that proposes and revises solutions."""
    
    def __init__(self, name: str, llm_client, expertise: str):
        self.name = name
        self.llm = llm_client
        self.expertise = expertise
    
    async def propose(self, task: str) -> str:
        """Create initial proposal."""
        
        prompt = f"""As an expert in {self.expertise}, create a solution for:

Task: {task}

Provide a complete, well-thought-out response."""
        
        return await self.llm.chat(prompt)
    
    async def revise(self, proposal: str, critique: str) -> str:
        """Revise proposal based on critique."""
        
        prompt = f"""Revise your proposal based on this feedback:

Original Proposal:
{proposal}

Critique:
{critique}

Address all valid concerns while maintaining the strengths of the original.
Provide the complete revised version."""
        
        return await self.llm.chat(prompt)


class CriticAgent:
    """Agent that critiques proposals."""
    
    def __init__(self, name: str, llm_client, criteria: List[str]):
        self.name = name
        self.llm = llm_client
        self.criteria = criteria
    
    async def critique(self, proposal: str) -> tuple[bool, str]:
        """Critique a proposal. Returns (approved, feedback)."""
        
        criteria_text = "\n".join(f"- {c}" for c in self.criteria)
        
        prompt = f"""Critically evaluate this proposal:

{proposal}

Evaluation Criteria:
{criteria_text}

Provide your critique covering:
1. Strengths
2. Weaknesses
3. Specific suggestions for improvement

End with APPROVED or NEEDS_REVISION"""
        
        response = await self.llm.chat(prompt)
        
        approved = "APPROVED" in response.upper() and "NEEDS_REVISION" not in response.upper()
        
        return approved, response


class ProposerCriticDebate:
    """Debate between proposer and critic."""
    
    def __init__(
        self,
        proposer: ProposerAgent,
        critic: CriticAgent,
        max_rounds: int = 5
    ):
        self.proposer = proposer
        self.critic = critic
        self.max_rounds = max_rounds
    
    async def debate(self, task: str) -> DebateResult:
        """Run the debate process."""
        
        rounds = []
        
        # Initial proposal
        proposal = await self.proposer.propose(task)
        
        for round_num in range(1, self.max_rounds + 1):
            # Critique
            approved, critique = await self.critic.critique(proposal)
            
            rounds.append(DebateRound(
                round_number=round_num,
                proposal=proposal,
                critique=critique,
                is_approved=approved
            ))
            
            if approved:
                return DebateResult(
                    final_output=proposal,
                    rounds=rounds,
                    total_rounds=round_num,
                    approved=True
                )
            
            # Revise
            proposal = await self.proposer.revise(proposal, critique)
        
        # Max rounds reached
        return DebateResult(
            final_output=proposal,
            rounds=rounds,
            total_rounds=self.max_rounds,
            approved=False
        )


# Example usage
async def run_debate():
    proposer = ProposerAgent(
        "Technical Writer",
        llm_client,
        expertise="software documentation"
    )
    
    critic = CriticAgent(
        "Quality Reviewer",
        llm_client,
        criteria=[
            "Technical accuracy",
            "Clarity and readability",
            "Completeness",
            "Proper examples"
        ]
    )
    
    debate = ProposerCriticDebate(proposer, critic, max_rounds=3)
    
    result = await debate.debate(
        "Write documentation for a REST API authentication endpoint"
    )
    
    print(f"Completed in {result.total_rounds} rounds")
    print(f"Approved: {result.approved}")
    print(f"\nFinal Output:\n{result.final_output}")
```

## Pattern 2: Multi-Agent Debate

Multiple agents debate from different perspectives.

```python
from typing import Dict

class DebatingAgent:
    """Agent with a specific perspective in debates."""
    
    def __init__(self, name: str, llm_client, perspective: str, priorities: List[str]):
        self.name = name
        self.llm = llm_client
        self.perspective = perspective
        self.priorities = priorities
    
    async def respond_to_proposal(
        self,
        proposal: str,
        context: str = ""
    ) -> str:
        """Respond to a proposal from this agent's perspective."""
        
        prompt = f"""You are debating from the perspective of: {self.perspective}

Your priorities are:
{chr(10).join(f"- {p}" for p in self.priorities)}

{f"Context: {context}" if context else ""}

Respond to this proposal:
{proposal}

Provide:
1. Points you agree with
2. Concerns from your perspective
3. Suggestions for improvement"""
        
        return await self.llm.chat(prompt)
    
    async def synthesize(
        self,
        original: str,
        responses: Dict[str, str]
    ) -> str:
        """Synthesize a new proposal from all responses."""
        
        responses_text = "\n\n".join(
            f"[{name}]:\n{response}"
            for name, response in responses.items()
        )
        
        prompt = f"""Create an improved proposal that addresses all perspectives:

Original Proposal:
{original}

Feedback from Different Perspectives:
{responses_text}

Create a revised proposal that:
1. Maintains the core idea
2. Addresses valid concerns from all perspectives
3. Balances competing priorities"""
        
        return await self.llm.chat(prompt)


class MultiAgentDebate:
    """Debate with multiple agents from different perspectives."""
    
    def __init__(
        self,
        agents: List[DebatingAgent],
        moderator: Optional["ModeratorAgent"] = None,
        max_rounds: int = 3
    ):
        self.agents = agents
        self.moderator = moderator
        self.max_rounds = max_rounds
    
    async def debate(self, initial_proposal: str) -> str:
        """Run multi-agent debate."""
        
        current_proposal = initial_proposal
        
        for round_num in range(self.max_rounds):
            print(f"\n=== Debate Round {round_num + 1} ===")
            
            # Collect responses from all agents
            responses = {}
            for agent in self.agents:
                response = await agent.respond_to_proposal(current_proposal)
                responses[agent.name] = response
                print(f"\n[{agent.name}]: {response[:200]}...")
            
            # Check for consensus (optional moderator)
            if self.moderator:
                has_consensus, summary = await self.moderator.check_consensus(responses)
                
                if has_consensus:
                    print(f"\nConsensus reached: {summary}")
                    break
            
            # Synthesize improved proposal
            # Rotate which agent synthesizes
            synthesizer = self.agents[round_num % len(self.agents)]
            current_proposal = await synthesizer.synthesize(
                current_proposal,
                responses
            )
        
        return current_proposal


class ModeratorAgent:
    """Moderator that guides debate and checks for consensus."""
    
    def __init__(self, name: str, llm_client):
        self.name = name
        self.llm = llm_client
    
    async def check_consensus(
        self,
        responses: Dict[str, str]
    ) -> tuple[bool, str]:
        """Check if agents have reached consensus."""
        
        responses_text = "\n\n".join(
            f"[{name}]: {response}"
            for name, response in responses.items()
        )
        
        prompt = f"""Analyze these debate responses for consensus:

{responses_text}

Determine:
1. Are there major disagreements remaining?
2. What points do all parties agree on?
3. Is there enough agreement to proceed?

Respond with:
CONSENSUS: [yes/no]
SUMMARY: [brief summary of current state]"""
        
        response = await self.llm.chat(prompt)
        
        has_consensus = "CONSENSUS: yes" in response.lower()
        
        # Extract summary
        summary = response.split("SUMMARY:")[-1].strip() if "SUMMARY:" in response else response
        
        return has_consensus, summary


# Example: Product feature debate
async def product_debate():
    agents = [
        DebatingAgent(
            "Engineer",
            llm_client,
            perspective="Technical Implementation",
            priorities=["Feasibility", "Performance", "Maintainability"]
        ),
        DebatingAgent(
            "Designer",
            llm_client,
            perspective="User Experience",
            priorities=["Usability", "Accessibility", "Visual Appeal"]
        ),
        DebatingAgent(
            "Business",
            llm_client,
            perspective="Business Value",
            priorities=["Revenue Impact", "Time to Market", "Cost"]
        )
    ]
    
    moderator = ModeratorAgent("Product Manager", llm_client)
    
    debate = MultiAgentDebate(agents, moderator, max_rounds=3)
    
    result = await debate.debate(
        "We should add a dark mode feature to our application"
    )
    
    return result
```

## Pattern 3: Devil's Advocate

One agent specifically challenges assumptions.

```python
class DevilsAdvocate:
    """Agent that challenges and finds flaws."""
    
    def __init__(self, name: str, llm_client):
        self.name = name
        self.llm = llm_client
    
    async def challenge(self, proposal: str) -> str:
        """Find every possible flaw and challenge."""
        
        prompt = f"""You are playing devil's advocate. Your job is to find every possible flaw, 
weakness, and potential problem with this proposal.

Proposal:
{proposal}

Challenge the proposal by:
1. Questioning assumptions
2. Identifying risks and failure modes
3. Finding logical flaws
4. Considering edge cases
5. Playing out worst-case scenarios

Be thorough but constructive - the goal is to make the proposal stronger."""
        
        return await self.llm.chat(prompt)


class RobustProposalBuilder:
    """Build robust proposals through devil's advocate challenges."""
    
    def __init__(self, proposer: ProposerAgent, devils_advocate: DevilsAdvocate):
        self.proposer = proposer
        self.devils_advocate = devils_advocate
    
    async def build(self, task: str, iterations: int = 2) -> str:
        """Build a robust proposal through challenges."""
        
        # Initial proposal
        proposal = await self.proposer.propose(task)
        
        for i in range(iterations):
            print(f"\n--- Challenge Round {i+1} ---")
            
            # Challenge
            challenges = await self.devils_advocate.challenge(proposal)
            print(f"Challenges: {challenges[:300]}...")
            
            # Strengthen proposal
            prompt = f"""Strengthen your proposal to address these challenges:

Current Proposal:
{proposal}

Challenges Raised:
{challenges}

Create an improved version that:
1. Addresses each valid challenge
2. Acknowledges limitations where they exist
3. Adds mitigations for identified risks"""
            
            proposal = await self.proposer.llm.chat(prompt)
        
        return proposal
```

## Pattern 4: Collaborative Building

Agents build on each other's work constructively.

```python
class CollaborativeAgent:
    """Agent that contributes to collaborative building."""
    
    def __init__(self, name: str, llm_client, specialty: str):
        self.name = name
        self.llm = llm_client
        self.specialty = specialty
    
    async def contribute(
        self,
        current_work: str,
        contribution_history: List[Dict[str, str]]
    ) -> str:
        """Add contribution to collaborative work."""
        
        history_text = "\n".join(
            f"[{h['agent']}]: {h['contribution'][:200]}..."
            for h in contribution_history
        )
        
        prompt = f"""You are contributing to a collaborative project.

Your specialty: {self.specialty}

Previous contributions:
{history_text if history_text else "None yet - you're starting!"}

Current state of work:
{current_work}

Add your contribution:
1. Build on what others have done
2. Apply your specialty expertise
3. Improve and extend the work
4. Don't repeat what's already there

Provide your addition to the work."""
        
        return await self.llm.chat(prompt)


class CollaborativeBuilder:
    """Coordinate collaborative building."""
    
    def __init__(self, agents: List[CollaborativeAgent], rounds: int = 2):
        self.agents = agents
        self.rounds = rounds
    
    async def build(self, initial_prompt: str) -> str:
        """Build collaboratively through multiple rounds."""
        
        current_work = initial_prompt
        history = []
        
        for round_num in range(self.rounds):
            print(f"\n=== Building Round {round_num + 1} ===")
            
            for agent in self.agents:
                contribution = await agent.contribute(current_work, history)
                
                history.append({
                    "agent": agent.name,
                    "round": round_num,
                    "contribution": contribution
                })
                
                # Integrate contribution
                current_work = await self._integrate(current_work, contribution)
                
                print(f"[{agent.name}] contributed")
        
        return current_work
    
    async def _integrate(self, current: str, addition: str) -> str:
        """Integrate new contribution into current work."""
        
        # For simple cases, just append
        # For complex cases, could use another LLM call to merge
        return f"{current}\n\n{addition}"


# Example: Collaborative story writing
async def collaborative_story():
    agents = [
        CollaborativeAgent("Plot Developer", llm_client, "story structure and plot"),
        CollaborativeAgent("Character Writer", llm_client, "character development and dialogue"),
        CollaborativeAgent("Setting Designer", llm_client, "world-building and atmosphere"),
    ]
    
    builder = CollaborativeBuilder(agents, rounds=3)
    
    story = await builder.build(
        "Write a short story about a robot discovering emotions."
    )
    
    return story
```

## Pattern 5: Socratic Dialogue

One agent teaches/guides another through questions.

```python
class SocraticTeacher:
    """Agent that teaches through questioning."""
    
    def __init__(self, name: str, llm_client):
        self.name = name
        self.llm = llm_client
    
    async def ask_question(
        self,
        topic: str,
        student_understanding: str,
        previous_exchanges: List[Dict[str, str]]
    ) -> str:
        """Ask a guiding question."""
        
        history = "\n".join(
            f"Q: {e['question']}\nA: {e['answer']}"
            for e in previous_exchanges
        )
        
        prompt = f"""You are a Socratic teacher guiding a student's understanding.

Topic: {topic}

Student's current understanding:
{student_understanding}

Previous exchanges:
{history if history else "None yet"}

Ask a thought-provoking question that:
1. Challenges assumptions
2. Reveals gaps in understanding
3. Guides toward deeper insight
4. Doesn't give away the answer directly"""
        
        return await self.llm.chat(prompt)


class SocraticStudent:
    """Agent that learns through questioning."""
    
    def __init__(self, name: str, llm_client):
        self.name = name
        self.llm = llm_client
    
    async def answer(self, question: str, context: str) -> str:
        """Answer a Socratic question."""
        
        prompt = f"""Answer this question thoughtfully:

Context: {context}

Question: {question}

Think through your answer:
1. What do you know about this?
2. What assumptions are you making?
3. What might you be missing?

Provide your answer."""
        
        return await self.llm.chat(prompt)


class SocraticDialogue:
    """Run a Socratic dialogue for learning."""
    
    def __init__(self, teacher: SocraticTeacher, student: SocraticStudent, exchanges: int = 5):
        self.teacher = teacher
        self.student = student
        self.exchanges = exchanges
    
    async def dialogue(self, topic: str, initial_understanding: str) -> List[Dict[str, str]]:
        """Run Socratic dialogue."""
        
        history = []
        current_understanding = initial_understanding
        
        for i in range(self.exchanges):
            # Teacher asks
            question = await self.teacher.ask_question(
                topic,
                current_understanding,
                history
            )
            
            # Student answers
            answer = await self.student.answer(question, topic)
            
            history.append({
                "question": question,
                "answer": answer
            })
            
            # Update understanding
            current_understanding = answer
            
            print(f"\nQ: {question}")
            print(f"A: {answer}")
        
        return history
```

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│              Debate Patterns - Summary                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Patterns:                                                               │
│    • Proposer-Critic: Propose, critique, iterate                       │
│    • Multi-Agent Debate: Multiple perspectives debate                  │
│    • Devil's Advocate: Challenge all assumptions                       │
│    • Collaborative Building: Build on each other's work               │
│    • Socratic Dialogue: Teach through questioning                      │
│                                                                          │
│  Benefits:                                                               │
│    • Higher quality outputs through iteration                          │
│    • Multiple perspectives considered                                  │
│    • Self-correcting through feedback                                  │
│    • Robust against blind spots                                        │
│                                                                          │
│  When to Use:                                                            │
│    • Quality is critical                                               │
│    • Task benefits from multiple viewpoints                            │
│    • Time/cost for iteration is acceptable                             │
│    • Complex decisions with trade-offs                                 │
│                                                                          │
│  Considerations:                                                         │
│    • More iterations = higher cost                                     │
│    • Need clear termination criteria                                   │
│    • Can get stuck in loops without good moderation                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

Next: [Agent Roles](/learn/multi-agents/building-multi-agent/agent-roles) →
