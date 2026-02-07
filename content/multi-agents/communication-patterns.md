# Agent Communication Patterns

Learn the different ways agents can communicate and share information in multi-agent systems.

## Overview of Communication Patterns

Agents need to exchange information effectively. The pattern you choose affects system behavior, reliability, and complexity.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  Communication Pattern Spectrum                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Direct Messages          Shared State          Event-Based            │
│   ◄─────────────────────────────────────────────────────────────►       │
│                                                                          │
│   • Point-to-point         • Central store       • Pub/Sub              │
│   • Request/Response       • All agents read     • Async                │
│   • Synchronous            • Concurrent access   • Decoupled            │
│   • Tight coupling         • Medium coupling     • Loose coupling       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Pattern 1: Direct Messaging

Agents communicate directly with each other through explicit messages.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Direct Messaging Pattern                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│       ┌───────────┐         Message          ┌───────────┐             │
│       │  Agent A  │ ───────────────────────► │  Agent B  │             │
│       └───────────┘                          └─────┬─────┘             │
│                                                    │                    │
│                            Response                │                    │
│       ┌───────────┐ ◄──────────────────────────────┘                   │
│       │  Agent A  │                                                     │
│       └───────────┘                                                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime
import uuid

@dataclass
class Message:
    """A message between agents."""
    sender: str
    receiver: str
    content: Any
    message_type: str = "default"
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    reply_to: Optional[str] = None

class DirectMessagingAgent:
    """Agent that communicates via direct messages."""
    
    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.system_prompt = system_prompt
        self.inbox: list[Message] = []
        self.peers: dict[str, "DirectMessagingAgent"] = {}
    
    def register_peer(self, peer: "DirectMessagingAgent"):
        """Register another agent as a peer."""
        self.peers[peer.name] = peer
    
    def send(self, to: str, content: Any, msg_type: str = "default") -> Message:
        """Send a message to another agent."""
        if to not in self.peers:
            raise ValueError(f"Unknown peer: {to}")
        
        message = Message(
            sender=self.name,
            receiver=to,
            content=content,
            message_type=msg_type
        )
        
        self.peers[to].receive(message)
        return message
    
    def receive(self, message: Message):
        """Receive a message from another agent."""
        self.inbox.append(message)
    
    def reply(self, original: Message, content: Any) -> Message:
        """Reply to a message."""
        reply = Message(
            sender=self.name,
            receiver=original.sender,
            content=content,
            message_type="reply",
            reply_to=original.id
        )
        
        self.peers[original.sender].receive(reply)
        return reply
    
    async def process_inbox(self) -> list[Message]:
        """Process all messages in inbox."""
        responses = []
        
        while self.inbox:
            message = self.inbox.pop(0)
            response = await self._handle_message(message)
            if response:
                responses.append(response)
        
        return responses
    
    async def _handle_message(self, message: Message) -> Optional[Message]:
        """Handle a single message (override in subclass)."""
        raise NotImplementedError


# Example: Writer-Reviewer communication
class WriterAgent(DirectMessagingAgent):
    async def _handle_message(self, message: Message) -> Optional[Message]:
        if message.message_type == "review_feedback":
            # Revise based on feedback
            revised = await self._revise(message.content)
            return self.reply(message, revised)
        return None

class ReviewerAgent(DirectMessagingAgent):
    async def _handle_message(self, message: Message) -> Optional[Message]:
        if message.message_type == "review_request":
            # Review the content
            feedback = await self._review(message.content)
            return self.reply(message, feedback)
        return None
```

### Use Cases for Direct Messaging

- **Request/Response workflows**: One agent asks, another answers
- **Chain of responsibility**: Pass work from agent to agent
- **Explicit handoffs**: Clear transfer of control

## Pattern 2: Shared State (Blackboard)

All agents read from and write to a central shared state.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Shared State (Blackboard) Pattern                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│       ┌───────────┐                          ┌───────────┐             │
│       │  Agent A  │ ────── read/write ─────► │           │             │
│       └───────────┘                          │           │             │
│                                              │  SHARED   │             │
│       ┌───────────┐                          │   STATE   │             │
│       │  Agent B  │ ────── read/write ─────► │           │             │
│       └───────────┘                          │           │             │
│                                              │(Blackboard)│             │
│       ┌───────────┐                          │           │             │
│       │  Agent C  │ ────── read/write ─────► │           │             │
│       └───────────┘                          └───────────┘             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
import asyncio
import json

@dataclass
class StateEntry:
    """An entry in the shared state."""
    key: str
    value: Any
    author: str
    timestamp: datetime = field(default_factory=datetime.now)
    version: int = 1

class SharedState:
    """Thread-safe shared state for multi-agent systems."""
    
    def __init__(self):
        self._state: Dict[str, StateEntry] = {}
        self._history: List[StateEntry] = []
        self._lock = asyncio.Lock()
        self._subscribers: Dict[str, List[callable]] = {}
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from shared state."""
        async with self._lock:
            entry = self._state.get(key)
            return entry.value if entry else None
    
    async def set(self, key: str, value: Any, author: str):
        """Set a value in shared state."""
        async with self._lock:
            existing = self._state.get(key)
            version = (existing.version + 1) if existing else 1
            
            entry = StateEntry(
                key=key,
                value=value,
                author=author,
                version=version
            )
            
            self._state[key] = entry
            self._history.append(entry)
        
        # Notify subscribers
        await self._notify(key, entry)
    
    async def append(self, key: str, value: Any, author: str):
        """Append to a list in shared state."""
        async with self._lock:
            existing = self._state.get(key)
            
            if existing and isinstance(existing.value, list):
                new_value = existing.value + [value]
            else:
                new_value = [value]
            
            entry = StateEntry(
                key=key,
                value=new_value,
                author=author,
                version=(existing.version + 1) if existing else 1
            )
            
            self._state[key] = entry
            self._history.append(entry)
        
        await self._notify(key, entry)
    
    def subscribe(self, key: str, callback: callable):
        """Subscribe to changes on a key."""
        if key not in self._subscribers:
            self._subscribers[key] = []
        self._subscribers[key].append(callback)
    
    async def _notify(self, key: str, entry: StateEntry):
        """Notify subscribers of a change."""
        if key in self._subscribers:
            for callback in self._subscribers[key]:
                if asyncio.iscoroutinefunction(callback):
                    await callback(entry)
                else:
                    callback(entry)
    
    async def get_all(self) -> Dict[str, Any]:
        """Get all current state."""
        async with self._lock:
            return {k: v.value for k, v in self._state.items()}
    
    def get_history(self, key: str = None) -> List[StateEntry]:
        """Get history of changes."""
        if key:
            return [e for e in self._history if e.key == key]
        return self._history.copy()


class BlackboardAgent:
    """Agent that uses shared state for communication."""
    
    def __init__(self, name: str, state: SharedState):
        self.name = name
        self.state = state
    
    async def read(self, key: str) -> Any:
        """Read from shared state."""
        return await self.state.get(key)
    
    async def write(self, key: str, value: Any):
        """Write to shared state."""
        await self.state.set(key, value, self.name)
    
    async def contribute(self, key: str, value: Any):
        """Add to a shared list."""
        await self.state.append(key, value, self.name)


# Example: Research collaboration
async def collaborative_research():
    state = SharedState()
    
    researcher = BlackboardAgent("researcher", state)
    analyst = BlackboardAgent("analyst", state)
    writer = BlackboardAgent("writer", state)
    
    # Researcher adds findings
    await researcher.write("topic", "AI in Healthcare")
    await researcher.contribute("findings", {
        "source": "paper_1",
        "summary": "AI improves diagnosis accuracy by 30%"
    })
    
    # Analyst reads and adds analysis
    findings = await analyst.read("findings")
    await analyst.write("analysis", {
        "key_insight": "Healthcare AI market growing",
        "based_on": len(findings)
    })
    
    # Writer reads everything and produces output
    topic = await writer.read("topic")
    analysis = await writer.read("analysis")
    await writer.write("draft", f"Article about {topic}...")
```

### Use Cases for Shared State

- **Collaborative editing**: Multiple agents contribute to same document
- **Knowledge accumulation**: Build up shared knowledge base
- **Workflow tracking**: All agents see current progress

## Pattern 3: Event-Based (Pub/Sub)

Agents publish events and subscribe to topics of interest.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Event-Based (Pub/Sub) Pattern                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Publishers                   Event Bus                Subscribers     │
│                                                                          │
│   ┌───────────┐              ┌─────────┐              ┌───────────┐    │
│   │  Agent A  │ ─publish──►  │         │ ─deliver──► │  Agent C  │    │
│   └───────────┘              │         │              └───────────┘    │
│                              │  Topic: │                               │
│   ┌───────────┐              │ "draft" │              ┌───────────┐    │
│   │  Agent B  │ ─publish──►  │         │ ─deliver──► │  Agent D  │    │
│   └───────────┘              └─────────┘              └───────────┘    │
│                                                                          │
│   Events flow: Publisher → Topic → All Subscribers                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Callable
from datetime import datetime
import asyncio

@dataclass
class Event:
    """An event in the system."""
    topic: str
    data: Any
    publisher: str
    timestamp: datetime = field(default_factory=datetime.now)

class EventBus:
    """Pub/Sub event bus for agent communication."""
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._event_history: List[Event] = []
    
    def subscribe(self, topic: str, handler: Callable):
        """Subscribe to a topic."""
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(handler)
    
    def unsubscribe(self, topic: str, handler: Callable):
        """Unsubscribe from a topic."""
        if topic in self._subscribers:
            self._subscribers[topic].remove(handler)
    
    async def publish(self, topic: str, data: Any, publisher: str):
        """Publish an event to a topic."""
        event = Event(
            topic=topic,
            data=data,
            publisher=publisher
        )
        
        self._event_history.append(event)
        
        # Notify all subscribers
        if topic in self._subscribers:
            tasks = []
            for handler in self._subscribers[topic]:
                if asyncio.iscoroutinefunction(handler):
                    tasks.append(handler(event))
                else:
                    handler(event)
            
            if tasks:
                await asyncio.gather(*tasks)
    
    async def publish_and_wait(
        self, 
        topic: str, 
        data: Any, 
        publisher: str,
        response_topic: str,
        timeout: float = 30.0
    ) -> Event:
        """Publish and wait for a response on another topic."""
        
        response_event = asyncio.Event()
        response_data = {}
        
        async def response_handler(event: Event):
            response_data["event"] = event
            response_event.set()
        
        self.subscribe(response_topic, response_handler)
        
        try:
            await self.publish(topic, data, publisher)
            
            await asyncio.wait_for(
                response_event.wait(),
                timeout=timeout
            )
            
            return response_data["event"]
        finally:
            self.unsubscribe(response_topic, response_handler)


class EventDrivenAgent:
    """Agent that communicates via events."""
    
    def __init__(self, name: str, bus: EventBus):
        self.name = name
        self.bus = bus
        self._handlers: Dict[str, Callable] = {}
    
    def on(self, topic: str, handler: Callable):
        """Register handler for a topic."""
        self._handlers[topic] = handler
        self.bus.subscribe(topic, self._create_wrapper(handler))
    
    def _create_wrapper(self, handler: Callable):
        """Create wrapper that adds agent context."""
        async def wrapper(event: Event):
            # Don't process own events
            if event.publisher == self.name:
                return
            await handler(event)
        return wrapper
    
    async def emit(self, topic: str, data: Any):
        """Emit an event."""
        await self.bus.publish(topic, data, self.name)


# Example: Event-driven workflow
async def event_driven_workflow():
    bus = EventBus()
    
    # Create agents
    writer = EventDrivenAgent("writer", bus)
    reviewer = EventDrivenAgent("reviewer", bus)
    publisher_agent = EventDrivenAgent("publisher", bus)
    
    # Writer listens for feedback
    @writer.on("review.complete")
    async def handle_review(event):
        if event.data["approved"]:
            await writer.emit("draft.final", event.data["content"])
        else:
            revised = await revise_draft(event.data["feedback"])
            await writer.emit("draft.ready", revised)
    
    # Reviewer listens for drafts
    @reviewer.on("draft.ready")
    async def handle_draft(event):
        review = await review_content(event.data)
        await reviewer.emit("review.complete", review)
    
    # Publisher listens for final drafts
    @publisher_agent.on("draft.final")
    async def handle_final(event):
        await publish_content(event.data)
        await publisher_agent.emit("content.published", {"url": "..."})
    
    # Start the workflow
    initial_draft = "This is my article..."
    await writer.emit("draft.ready", initial_draft)
```

### Use Cases for Event-Based Communication

- **Decoupled systems**: Agents don't need to know about each other
- **Reactive workflows**: Respond to events as they occur
- **Scalable architectures**: Easy to add new agents

## Pattern Comparison

| Pattern | Coupling | Complexity | Best For |
|---------|----------|------------|----------|
| Direct Messaging | Tight | Low | Simple request/response |
| Shared State | Medium | Medium | Collaborative work |
| Event-Based | Loose | Higher | Scalable, reactive systems |

## Hybrid Approaches

Real systems often combine patterns:

```python
class HybridAgent:
    """Agent that uses multiple communication patterns."""
    
    def __init__(self, name: str, state: SharedState, bus: EventBus):
        self.name = name
        self.state = state
        self.bus = bus
        self.peers: Dict[str, "HybridAgent"] = {}
    
    # Direct messaging for urgent/specific communication
    async def send_direct(self, to: str, message: Any):
        if to in self.peers:
            await self.peers[to].receive_direct(message)
    
    # Shared state for collaborative data
    async def update_shared(self, key: str, value: Any):
        await self.state.set(key, value, self.name)
    
    # Events for broadcast notifications
    async def broadcast(self, topic: str, data: Any):
        await self.bus.publish(topic, data, self.name)
```

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│              Communication Patterns - Summary                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Direct Messaging:                                                       │
│    • Explicit point-to-point communication                              │
│    • Clear request/response patterns                                    │
│    • Best for: Chain workflows, explicit handoffs                       │
│                                                                          │
│  Shared State (Blackboard):                                              │
│    • Central store all agents access                                    │
│    • Good for collaborative building                                    │
│    • Best for: Knowledge accumulation, shared context                   │
│                                                                          │
│  Event-Based (Pub/Sub):                                                  │
│    • Loosely coupled, reactive                                          │
│    • Agents subscribe to topics                                         │
│    • Best for: Scalable, decoupled systems                             │
│                                                                          │
│  Choose Based On:                                                        │
│    • How tightly agents need to coordinate                              │
│    • Whether communication is 1:1 or 1:many                             │
│    • Synchronous vs asynchronous needs                                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

Next: [Coordination Strategies](/learn/multi-agents/multi-agent-concepts/coordination-strategies) →
