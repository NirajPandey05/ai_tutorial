# Lab: Build a Writer-Reviewer Multi-Agent System

In this lab, you'll build a complete multi-agent system where a writer agent creates content and a reviewer agent provides feedback through multiple iterations.

## Learning Objectives

By the end of this lab, you will:
- Implement specialized agent roles
- Build a message passing system between agents
- Create an iterative improvement loop
- Handle state sharing during handoffs

## Prerequisites

- Python 3.9+
- OpenAI API key
- Understanding of async/await in Python

## Part 1: Project Setup

Create a new project with the following structure:

```
writer_reviewer/
├── agents/
│   ├── __init__.py
│   ├── base.py
│   ├── writer.py
│   └── reviewer.py
├── core/
│   ├── __init__.py
│   ├── messages.py
│   └── state.py
├── main.py
└── requirements.txt
```

**requirements.txt:**
```
openai>=1.0.0
python-dotenv
```

## Part 2: Core Infrastructure

**core/messages.py** - Message types and queue:

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
from datetime import datetime
import uuid
import asyncio

class MessageType(Enum):
    TASK = "task"
    DRAFT = "draft"
    REVIEW = "review"
    REVISION = "revision"
    APPROVED = "approved"

@dataclass
class Message:
    """Message between agents."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.TASK
    sender: str = ""
    content: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    iteration: int = 0

class MessageBus:
    """Simple message bus for agent communication."""
    
    def __init__(self):
        self._queues: Dict[str, asyncio.Queue] = {}
        self._history: List[Message] = []
    
    def register(self, agent_id: str):
        """Register an agent's message queue."""
        self._queues[agent_id] = asyncio.Queue()
    
    async def send(self, recipient: str, message: Message):
        """Send a message to an agent."""
        self._history.append(message)
        if recipient in self._queues:
            await self._queues[recipient].put(message)
    
    async def receive(self, agent_id: str, timeout: float = 30.0) -> Optional[Message]:
        """Receive a message for an agent."""
        try:
            return await asyncio.wait_for(
                self._queues[agent_id].get(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return None
    
    def get_history(self) -> List[Message]:
        """Get all message history."""
        return self._history.copy()
```

**core/state.py** - Shared state:

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List
from datetime import datetime
import asyncio

@dataclass
class ContentState:
    """State of content through the review process."""
    task: str = ""
    current_draft: str = ""
    draft_history: List[Dict[str, Any]] = field(default_factory=list)
    reviews: List[Dict[str, Any]] = field(default_factory=list)
    iteration: int = 0
    status: str = "not_started"
    
    def add_draft(self, draft: str, agent_id: str):
        """Add a new draft."""
        self.draft_history.append({
            "iteration": self.iteration,
            "draft": draft,
            "agent": agent_id,
            "timestamp": datetime.now().isoformat()
        })
        self.current_draft = draft
    
    def add_review(self, review: Dict[str, Any], agent_id: str):
        """Add a review."""
        self.reviews.append({
            "iteration": self.iteration,
            "review": review,
            "agent": agent_id,
            "timestamp": datetime.now().isoformat()
        })

class StateManager:
    """Manage shared state."""
    
    def __init__(self):
        self._state: Dict[str, ContentState] = {}
        self._lock = asyncio.Lock()
    
    async def create_session(self, session_id: str, task: str) -> ContentState:
        """Create a new content session."""
        async with self._lock:
            state = ContentState(task=task, status="in_progress")
            self._state[session_id] = state
            return state
    
    async def get_state(self, session_id: str) -> Optional[ContentState]:
        """Get state for a session."""
        return self._state.get(session_id)
    
    async def update_state(self, session_id: str, **updates):
        """Update state fields."""
        async with self._lock:
            state = self._state.get(session_id)
            if state:
                for key, value in updates.items():
                    if hasattr(state, key):
                        setattr(state, key, value)
```

## Part 3: Base Agent

**agents/base.py**:

```python
from abc import ABC, abstractmethod
from typing import Any, Dict
from openai import AsyncOpenAI
from core.messages import MessageBus, Message, MessageType

class BaseAgent(ABC):
    """Base class for all agents."""
    
    def __init__(
        self,
        agent_id: str,
        message_bus: MessageBus,
        llm_client: AsyncOpenAI,
        model: str = "gpt-4"
    ):
        self.id = agent_id
        self.bus = message_bus
        self.llm = llm_client
        self.model = model
        
        # Register with message bus
        self.bus.register(self.id)
    
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the agent's system prompt."""
        pass
    
    async def call_llm(self, user_prompt: str) -> str:
        """Call the LLM with the agent's context."""
        response = await self.llm.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content
    
    async def send(self, recipient: str, msg_type: MessageType, content: Any, **metadata):
        """Send a message to another agent."""
        message = Message(
            type=msg_type,
            sender=self.id,
            content=content,
            metadata=metadata
        )
        await self.bus.send(recipient, message)
    
    async def receive(self, timeout: float = 30.0) -> Message:
        """Receive a message."""
        return await self.bus.receive(self.id, timeout)
    
    @abstractmethod
    async def process(self, message: Message) -> Any:
        """Process a received message."""
        pass
```

## Part 4: Writer Agent

**agents/writer.py**:

```python
from agents.base import BaseAgent
from core.messages import Message, MessageType
from core.state import StateManager

class WriterAgent(BaseAgent):
    """Agent that writes content."""
    
    def __init__(self, agent_id: str, message_bus, llm_client, state_manager: StateManager):
        super().__init__(agent_id, message_bus, llm_client)
        self.state_manager = state_manager
    
    @property
    def system_prompt(self) -> str:
        return """You are an expert content writer.

Your responsibilities:
- Write clear, engaging content
- Follow the given task requirements
- Incorporate feedback from reviewers
- Maintain consistent quality and tone

When revising:
- Address each point of feedback specifically
- Explain what changes you made
- Preserve what was working well"""
    
    async def process(self, message: Message):
        """Process incoming messages."""
        
        if message.type == MessageType.TASK:
            return await self._handle_task(message)
        elif message.type == MessageType.REVIEW:
            return await self._handle_review(message)
        
        return None
    
    async def _handle_task(self, message: Message) -> str:
        """Handle a new writing task."""
        
        session_id = message.metadata.get("session_id")
        task = message.content
        
        prompt = f"""Write content for the following task:

Task: {task}

Requirements:
- Be thorough and comprehensive
- Use clear, engaging language
- Structure the content logically
- Include relevant examples where appropriate"""
        
        draft = await self.call_llm(prompt)
        
        # Update state
        state = await self.state_manager.get_state(session_id)
        if state:
            state.add_draft(draft, self.id)
        
        # Send draft to reviewer
        await self.send(
            "reviewer",
            MessageType.DRAFT,
            draft,
            session_id=session_id,
            iteration=0
        )
        
        return draft
    
    async def _handle_review(self, message: Message) -> str:
        """Handle review feedback and revise."""
        
        session_id = message.metadata.get("session_id")
        iteration = message.metadata.get("iteration", 0) + 1
        review = message.content
        
        # Get current state
        state = await self.state_manager.get_state(session_id)
        current_draft = state.current_draft if state else ""
        
        prompt = f"""Revise your content based on the following feedback:

Current Draft:
{current_draft}

Reviewer Feedback:
Approved: {review.get('approved', False)}
Strengths: {review.get('strengths', 'N/A')}
Issues: {review.get('issues', 'N/A')}
Suggestions: {review.get('suggestions', 'N/A')}

Instructions:
1. Address each issue raised
2. Incorporate the suggestions
3. Build on the identified strengths
4. Maintain overall quality and coherence

Provide the complete revised content."""
        
        revised = await self.call_llm(prompt)
        
        # Update state
        if state:
            state.iteration = iteration
            state.add_draft(revised, self.id)
        
        # Send revised draft to reviewer
        await self.send(
            "reviewer",
            MessageType.REVISION,
            revised,
            session_id=session_id,
            iteration=iteration
        )
        
        return revised
```

## Part 5: Reviewer Agent

**agents/reviewer.py**:

```python
import json
from agents.base import BaseAgent
from core.messages import Message, MessageType
from core.state import StateManager

class ReviewerAgent(BaseAgent):
    """Agent that reviews and provides feedback."""
    
    def __init__(
        self,
        agent_id: str,
        message_bus,
        llm_client,
        state_manager: StateManager,
        max_iterations: int = 3
    ):
        super().__init__(agent_id, message_bus, llm_client)
        self.state_manager = state_manager
        self.max_iterations = max_iterations
    
    @property
    def system_prompt(self) -> str:
        return """You are an expert content reviewer.

Your responsibilities:
- Provide thorough, constructive feedback
- Identify both strengths and weaknesses
- Be specific with suggestions for improvement
- Know when content meets quality standards

Review Criteria:
- Clarity and readability
- Accuracy and completeness
- Structure and organization
- Engagement and tone
- Grammar and style

Be rigorous but fair. Approve only when quality is genuinely good."""
    
    async def process(self, message: Message):
        """Process incoming messages."""
        
        if message.type in [MessageType.DRAFT, MessageType.REVISION]:
            return await self._handle_draft(message)
        
        return None
    
    async def _handle_draft(self, message: Message) -> dict:
        """Review a draft."""
        
        session_id = message.metadata.get("session_id")
        iteration = message.metadata.get("iteration", 0)
        draft = message.content
        
        # Get task for context
        state = await self.state_manager.get_state(session_id)
        task = state.task if state else "Unknown task"
        
        prompt = f"""Review the following content:

Original Task: {task}

Content to Review (Iteration {iteration}):
{draft}

Previous Iterations: {iteration}
Max Iterations: {self.max_iterations}

Provide a thorough review in JSON format:
{{
    "approved": true/false,
    "overall_quality": "excellent/good/adequate/needs_improvement/poor",
    "strengths": ["strength1", "strength2"],
    "issues": ["issue1", "issue2"],
    "suggestions": ["suggestion1", "suggestion2"],
    "priority_fixes": ["most important fix"],
    "reasoning": "explanation of your decision"
}}

Be rigorous but fair. Only approve if the content truly meets quality standards.
Consider that we have {self.max_iterations - iteration} iterations remaining."""
        
        response = await self.call_llm(prompt)
        
        # Parse review
        try:
            # Find JSON in response
            start = response.find('{')
            end = response.rfind('}') + 1
            review = json.loads(response[start:end])
        except (json.JSONDecodeError, ValueError):
            review = {
                "approved": False,
                "overall_quality": "needs_improvement",
                "strengths": [],
                "issues": ["Could not parse review"],
                "suggestions": ["Please try again"],
                "priority_fixes": [],
                "reasoning": response
            }
        
        # Update state
        if state:
            state.add_review(review, self.id)
        
        # Decide next action
        if review.get("approved", False):
            # Content approved
            if state:
                state.status = "approved"
            
            await self.send(
                "coordinator",
                MessageType.APPROVED,
                {
                    "final_content": draft,
                    "review": review,
                    "iterations": iteration
                },
                session_id=session_id
            )
        elif iteration >= self.max_iterations:
            # Max iterations reached
            if state:
                state.status = "max_iterations"
            
            await self.send(
                "coordinator",
                MessageType.APPROVED,
                {
                    "final_content": draft,
                    "review": review,
                    "iterations": iteration,
                    "note": "Max iterations reached, best effort"
                },
                session_id=session_id
            )
        else:
            # Send back for revision
            await self.send(
                "writer",
                MessageType.REVIEW,
                review,
                session_id=session_id,
                iteration=iteration
            )
        
        return review
```

## Part 6: Coordinator

**main.py**:

```python
import asyncio
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI

from core.messages import MessageBus, Message, MessageType
from core.state import StateManager, ContentState
from agents.writer import WriterAgent
from agents.reviewer import ReviewerAgent

load_dotenv()

class Coordinator:
    """Coordinates the writer-reviewer workflow."""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.bus = MessageBus()
        self.state_manager = StateManager()
        
        # Register coordinator
        self.bus.register("coordinator")
        
        # Create agents
        self.writer = WriterAgent(
            "writer",
            self.bus,
            self.client,
            self.state_manager
        )
        self.reviewer = ReviewerAgent(
            "reviewer",
            self.bus,
            self.client,
            self.state_manager,
            max_iterations=3
        )
    
    async def run_workflow(self, task: str) -> dict:
        """Run the complete writer-reviewer workflow."""
        
        session_id = f"session_{id(task)}"
        
        # Initialize state
        await self.state_manager.create_session(session_id, task)
        
        # Start writer and reviewer processing loops
        writer_task = asyncio.create_task(self._agent_loop(self.writer))
        reviewer_task = asyncio.create_task(self._agent_loop(self.reviewer))
        
        # Send initial task to writer
        await self.bus.send(
            "writer",
            Message(
                type=MessageType.TASK,
                sender="coordinator",
                content=task,
                metadata={"session_id": session_id}
            )
        )
        
        # Wait for completion
        result = await self._wait_for_completion(session_id)
        
        # Cancel agent loops
        writer_task.cancel()
        reviewer_task.cancel()
        
        return result
    
    async def _agent_loop(self, agent):
        """Run an agent's message processing loop."""
        while True:
            try:
                message = await agent.receive(timeout=60.0)
                if message:
                    await agent.process(message)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error in {agent.id}: {e}")
    
    async def _wait_for_completion(self, session_id: str, timeout: float = 300.0) -> dict:
        """Wait for workflow completion."""
        
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            message = await self.bus.receive("coordinator", timeout=5.0)
            
            if message and message.type == MessageType.APPROVED:
                state = await self.state_manager.get_state(session_id)
                
                return {
                    "status": "completed",
                    "final_content": message.content.get("final_content"),
                    "iterations": message.content.get("iterations"),
                    "approved": message.content.get("review", {}).get("approved", False),
                    "history": {
                        "drafts": state.draft_history if state else [],
                        "reviews": state.reviews if state else []
                    }
                }
        
        return {"status": "timeout", "error": "Workflow timed out"}


async def main():
    """Run the writer-reviewer workflow."""
    
    coordinator = Coordinator()
    
    task = """Write a comprehensive guide about getting started with Python for beginners.
    
    Include:
    - Why Python is a good first language
    - How to install Python
    - Basic concepts (variables, data types, functions)
    - A simple example project
    - Next steps for learning
    
    Target audience: Complete beginners with no programming experience.
    Tone: Friendly, encouraging, clear.
    Length: Approximately 1000-1500 words."""
    
    print("Starting Writer-Reviewer Workflow")
    print("=" * 50)
    print(f"Task: {task[:100]}...")
    print("=" * 50)
    
    result = await coordinator.run_workflow(task)
    
    print("\n" + "=" * 50)
    print("WORKFLOW COMPLETE")
    print("=" * 50)
    print(f"Status: {result['status']}")
    print(f"Iterations: {result.get('iterations', 'N/A')}")
    print(f"Approved: {result.get('approved', 'N/A')}")
    
    print("\n" + "-" * 50)
    print("FINAL CONTENT:")
    print("-" * 50)
    print(result.get("final_content", "No content"))
    
    # Show iteration history
    print("\n" + "-" * 50)
    print("ITERATION HISTORY:")
    print("-" * 50)
    
    history = result.get("history", {})
    for i, review in enumerate(history.get("reviews", [])):
        print(f"\nReview {i + 1}:")
        print(f"  Quality: {review['review'].get('overall_quality', 'N/A')}")
        print(f"  Approved: {review['review'].get('approved', False)}")
        if review['review'].get('issues'):
            print(f"  Issues: {', '.join(review['review']['issues'][:2])}")


if __name__ == "__main__":
    asyncio.run(main())
```

## Part 7: Run the Lab

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set your API key:**
```bash
# Create .env file
echo "OPENAI_API_KEY=your-key-here" > .env
```

3. **Run the workflow:**
```bash
python main.py
```

## Expected Output

```
Starting Writer-Reviewer Workflow
==================================================
Task: Write a comprehensive guide about getting started with Python...
==================================================

[Writer] Creating initial draft...
[Reviewer] Reviewing draft (iteration 0)...
[Writer] Revising based on feedback (iteration 1)...
[Reviewer] Reviewing revision (iteration 1)...
[Writer] Revising based on feedback (iteration 2)...
[Reviewer] Reviewing revision (iteration 2)...
[Reviewer] Content APPROVED

==================================================
WORKFLOW COMPLETE
==================================================
Status: completed
Iterations: 2
Approved: True
```

## Challenges

1. **Add a third agent**: Create an Editor agent that does a final polish pass after reviewer approval.

2. **Implement quality metrics**: Add word count, readability score, and other metrics to the review process.

3. **Add persistence**: Save drafts and reviews to files so you can resume interrupted workflows.

4. **Parallel review**: Have multiple reviewers and require consensus before approval.

## Summary

You've built a complete writer-reviewer multi-agent system with:
- ✅ Specialized agent roles (writer, reviewer)
- ✅ Message passing communication
- ✅ Shared state management
- ✅ Iterative improvement loop
- ✅ Coordination and workflow management

This pattern can be extended for many use cases: code review, document approval, creative collaboration, and more.

Next: [Agent Swarms](/learn/multi-agents/advanced-multi-agent/agent-swarms) →
