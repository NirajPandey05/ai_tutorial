# Message Passing Between Agents

Learn how to implement communication between agents through message passing.

## Why Message Passing?

Message passing provides structured communication between agents while maintaining loose coupling.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Message Passing Overview                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Direct Coupling:           Message Passing:                           │
│                                                                          │
│   ┌─────────┐               ┌─────────┐                                 │
│   │ Agent A ├───────────►   │ Agent A │                                 │
│   └────┬────┘               └────┬────┘                                 │
│        │ Tight                   │                                      │
│        │ Coupling                ▼                                      │
│        │                  ┌─────────────┐                               │
│   ┌────▼────┐             │   Message   │ ◄── Loose Coupling            │
│   │ Agent B │             │    Queue    │                               │
│   └─────────┘             └──────┬──────┘                               │
│                                  │                                      │
│                                  ▼                                      │
│                           ┌─────────┐                                   │
│                           │ Agent B │                                   │
│                           └─────────┘                                   │
│                                                                          │
│   Problems:                Benefits:                                    │
│   - Hard to scale         - Easy to scale                              │
│   - Tight dependencies    - Async communication                        │
│   - Hard to debug         - Message history                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Message Types and Structures

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
from datetime import datetime
import uuid

class MessageType(Enum):
    """Types of messages agents can send."""
    REQUEST = "request"           # Asking for something
    RESPONSE = "response"         # Answering a request
    NOTIFICATION = "notification" # Informing about something
    TASK = "task"                # Assigning work
    RESULT = "result"            # Returning work output
    ERROR = "error"              # Reporting an error
    STATUS = "status"            # Reporting status update

class MessagePriority(Enum):
    """Priority levels for messages."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass
class Message:
    """A message between agents."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""
    recipient: str = ""
    type: MessageType = MessageType.NOTIFICATION
    priority: MessagePriority = MessagePriority.NORMAL
    content: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    reply_to: Optional[str] = None  # ID of message this replies to
    conversation_id: Optional[str] = None  # Group related messages
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "sender": self.sender,
            "recipient": self.recipient,
            "type": self.type.value,
            "priority": self.priority.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "reply_to": self.reply_to,
            "conversation_id": self.conversation_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            sender=data["sender"],
            recipient=data["recipient"],
            type=MessageType(data["type"]),
            priority=MessagePriority(data["priority"]),
            content=data["content"],
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            reply_to=data.get("reply_to"),
            conversation_id=data.get("conversation_id")
        )


# Helper functions for creating messages
def create_task_message(
    sender: str,
    recipient: str,
    task_description: str,
    context: Dict[str, Any] = None,
    priority: MessagePriority = MessagePriority.NORMAL
) -> Message:
    """Create a task assignment message."""
    return Message(
        sender=sender,
        recipient=recipient,
        type=MessageType.TASK,
        priority=priority,
        content={
            "description": task_description,
            "context": context or {}
        },
        conversation_id=str(uuid.uuid4())
    )


def create_result_message(
    sender: str,
    recipient: str,
    result: Any,
    original_message: Message
) -> Message:
    """Create a result message."""
    return Message(
        sender=sender,
        recipient=recipient,
        type=MessageType.RESULT,
        content=result,
        reply_to=original_message.id,
        conversation_id=original_message.conversation_id
    )
```

## Message Queue Implementation

```python
from collections import defaultdict
from queue import PriorityQueue
from asyncio import Queue as AsyncQueue
import asyncio

class MessageQueue:
    """Async message queue for agent communication."""
    
    def __init__(self):
        self._queues: Dict[str, AsyncQueue] = defaultdict(AsyncQueue)
        self._history: List[Message] = []
        self._subscribers: Dict[str, List[callable]] = defaultdict(list)
    
    async def send(self, message: Message) -> None:
        """Send a message to a recipient."""
        
        # Store in history
        self._history.append(message)
        
        # Add to recipient's queue
        await self._queues[message.recipient].put(message)
        
        # Notify subscribers
        for callback in self._subscribers[message.recipient]:
            await callback(message)
    
    async def receive(
        self,
        agent_id: str,
        timeout: float = None
    ) -> Optional[Message]:
        """Receive a message for an agent."""
        
        try:
            if timeout:
                message = await asyncio.wait_for(
                    self._queues[agent_id].get(),
                    timeout=timeout
                )
            else:
                message = await self._queues[agent_id].get()
            
            return message
        except asyncio.TimeoutError:
            return None
    
    async def receive_all(self, agent_id: str) -> List[Message]:
        """Receive all pending messages for an agent."""
        
        messages = []
        while not self._queues[agent_id].empty():
            message = await self._queues[agent_id].get()
            messages.append(message)
        
        return messages
    
    def subscribe(self, agent_id: str, callback: callable):
        """Subscribe to messages for an agent."""
        self._subscribers[agent_id].append(callback)
    
    def get_conversation(self, conversation_id: str) -> List[Message]:
        """Get all messages in a conversation."""
        return [
            m for m in self._history
            if m.conversation_id == conversation_id
        ]
    
    def get_agent_history(self, agent_id: str) -> List[Message]:
        """Get message history for an agent."""
        return [
            m for m in self._history
            if m.sender == agent_id or m.recipient == agent_id
        ]


# Priority queue version
class PriorityMessageQueue:
    """Message queue with priority handling."""
    
    def __init__(self):
        self._queues: Dict[str, List[Message]] = defaultdict(list)
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
    
    async def send(self, message: Message):
        """Send a message, sorted by priority."""
        
        async with self._locks[message.recipient]:
            # Insert in priority order
            queue = self._queues[message.recipient]
            
            # Find insertion point (higher priority = lower number = front)
            insert_idx = len(queue)
            for i, m in enumerate(queue):
                if message.priority.value > m.priority.value:
                    insert_idx = i
                    break
            
            queue.insert(insert_idx, message)
    
    async def receive(self, agent_id: str) -> Optional[Message]:
        """Receive highest priority message."""
        
        async with self._locks[agent_id]:
            if self._queues[agent_id]:
                return self._queues[agent_id].pop(0)
            return None
```

## Message-Passing Agent

```python
from typing import Callable, Awaitable

class MessagePassingAgent:
    """Agent that communicates through message passing."""
    
    def __init__(
        self,
        agent_id: str,
        message_queue: MessageQueue,
        llm_client: Any
    ):
        self.id = agent_id
        self.queue = message_queue
        self.llm = llm_client
        self.handlers: Dict[MessageType, Callable] = {}
        self.running = False
        
        # Set up default handlers
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Set up message type handlers."""
        
        self.handlers[MessageType.TASK] = self._handle_task
        self.handlers[MessageType.REQUEST] = self._handle_request
        self.handlers[MessageType.NOTIFICATION] = self._handle_notification
    
    async def _handle_task(self, message: Message) -> Optional[Message]:
        """Handle a task message."""
        
        task = message.content["description"]
        context = message.content.get("context", {})
        
        # Process task
        result = await self._process_task(task, context)
        
        # Send result back
        return create_result_message(
            sender=self.id,
            recipient=message.sender,
            result=result,
            original_message=message
        )
    
    async def _handle_request(self, message: Message) -> Optional[Message]:
        """Handle a request message."""
        
        # Process request and respond
        response = await self._process_request(message.content)
        
        return Message(
            sender=self.id,
            recipient=message.sender,
            type=MessageType.RESPONSE,
            content=response,
            reply_to=message.id,
            conversation_id=message.conversation_id
        )
    
    async def _handle_notification(self, message: Message) -> None:
        """Handle a notification (no response needed)."""
        
        # Just log or process notification
        print(f"[{self.id}] Notification: {message.content}")
        return None
    
    async def _process_task(self, task: str, context: Dict) -> Any:
        """Process a task using LLM."""
        
        context_str = "\n".join(f"{k}: {v}" for k, v in context.items())
        
        prompt = f"""Complete this task:
{task}

Context:
{context_str if context_str else "None"}"""
        
        response = await self.llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.choices[0].message.content
    
    async def _process_request(self, request: Any) -> Any:
        """Process a request."""
        # Override in subclass for custom handling
        return f"Received request: {request}"
    
    async def send_message(
        self,
        recipient: str,
        content: Any,
        msg_type: MessageType = MessageType.NOTIFICATION,
        priority: MessagePriority = MessagePriority.NORMAL
    ) -> Message:
        """Send a message to another agent."""
        
        message = Message(
            sender=self.id,
            recipient=recipient,
            type=msg_type,
            priority=priority,
            content=content
        )
        
        await self.queue.send(message)
        return message
    
    async def send_task(
        self,
        recipient: str,
        task_description: str,
        context: Dict = None,
        wait_for_result: bool = True,
        timeout: float = 30.0
    ) -> Optional[Any]:
        """Send a task and optionally wait for result."""
        
        message = create_task_message(
            sender=self.id,
            recipient=recipient,
            task_description=task_description,
            context=context
        )
        
        await self.queue.send(message)
        
        if not wait_for_result:
            return None
        
        # Wait for result
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            # Check for result message
            response = await self.queue.receive(self.id, timeout=1.0)
            
            if response and response.reply_to == message.id:
                return response.content
        
        return None
    
    async def start(self):
        """Start processing messages."""
        
        self.running = True
        
        while self.running:
            message = await self.queue.receive(self.id, timeout=1.0)
            
            if message:
                handler = self.handlers.get(message.type)
                
                if handler:
                    response = await handler(message)
                    
                    if response:
                        await self.queue.send(response)
    
    def stop(self):
        """Stop processing messages."""
        self.running = False
```

## Message Routing

```python
class MessageRouter:
    """Route messages between agents."""
    
    def __init__(self):
        self.routes: Dict[str, List[str]] = {}
        self.filters: Dict[str, Callable[[Message], bool]] = {}
        self.transforms: Dict[str, Callable[[Message], Message]] = {}
    
    def add_route(self, pattern: str, destinations: List[str]):
        """Add a routing rule."""
        self.routes[pattern] = destinations
    
    def add_filter(
        self,
        route_pattern: str,
        filter_func: Callable[[Message], bool]
    ):
        """Add a filter for a route."""
        self.filters[route_pattern] = filter_func
    
    def add_transform(
        self,
        route_pattern: str,
        transform_func: Callable[[Message], Message]
    ):
        """Add a transformation for messages on a route."""
        self.transforms[route_pattern] = transform_func
    
    def get_destinations(self, message: Message) -> List[str]:
        """Get destinations for a message."""
        
        destinations = []
        
        for pattern, dests in self.routes.items():
            if self._matches_pattern(message, pattern):
                # Check filter
                if pattern in self.filters:
                    if not self.filters[pattern](message):
                        continue
                
                destinations.extend(dests)
        
        return list(set(destinations))  # Remove duplicates
    
    def _matches_pattern(self, message: Message, pattern: str) -> bool:
        """Check if message matches a routing pattern."""
        
        # Simple pattern matching
        # "type:task" - matches task messages
        # "sender:agent_*" - matches agents starting with "agent_"
        # "*" - matches all
        
        if pattern == "*":
            return True
        
        if ":" in pattern:
            key, value = pattern.split(":", 1)
            
            if key == "type":
                return message.type.value == value
            elif key == "sender":
                if value.endswith("*"):
                    return message.sender.startswith(value[:-1])
                return message.sender == value
            elif key == "priority":
                return str(message.priority.value) == value
        
        return False


class RoutingMessageQueue(MessageQueue):
    """Message queue with routing support."""
    
    def __init__(self, router: MessageRouter):
        super().__init__()
        self.router = router
    
    async def send(self, message: Message) -> None:
        """Send message through router."""
        
        # Get all destinations
        if message.recipient:
            destinations = [message.recipient]
        else:
            destinations = self.router.get_destinations(message)
        
        # Send to all destinations
        for dest in destinations:
            routed_message = Message(
                id=message.id,
                sender=message.sender,
                recipient=dest,
                type=message.type,
                priority=message.priority,
                content=message.content,
                metadata=message.metadata,
                timestamp=message.timestamp,
                reply_to=message.reply_to,
                conversation_id=message.conversation_id
            )
            
            self._history.append(routed_message)
            await self._queues[dest].put(routed_message)
```

## Complete Example

```python
async def message_passing_demo():
    """Demo of message passing between agents."""
    
    # Create message queue
    queue = MessageQueue()
    
    # Create agents
    class ResearchAgent(MessagePassingAgent):
        async def _process_task(self, task: str, context: Dict) -> Any:
            return f"Research completed on: {task}"
    
    class WriterAgent(MessagePassingAgent):
        async def _process_task(self, task: str, context: Dict) -> Any:
            research = context.get("research", "")
            return f"Article written based on: {research}"
    
    researcher = ResearchAgent("researcher", queue, llm_client)
    writer = WriterAgent("writer", queue, llm_client)
    
    # Start agents in background
    async def run_agents():
        await asyncio.gather(
            researcher.start(),
            writer.start()
        )
    
    agent_task = asyncio.create_task(run_agents())
    
    # Coordinator sends tasks
    # Step 1: Send research task
    research_result = await researcher.send_task(
        recipient="researcher",
        task_description="Research AI trends for 2024",
        wait_for_result=True
    )
    
    # Step 2: Send writing task with research context
    article = await writer.send_task(
        recipient="writer",
        task_description="Write an article about AI trends",
        context={"research": research_result},
        wait_for_result=True
    )
    
    # Stop agents
    researcher.stop()
    writer.stop()
    
    return article
```

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│               Message Passing - Summary                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Message Components:                                                     │
│    • ID and timestamps                                                  │
│    • Sender and recipient                                               │
│    • Type (task, result, notification, etc.)                           │
│    • Priority level                                                     │
│    • Content and metadata                                               │
│    • Conversation tracking                                              │
│                                                                          │
│  Queue Features:                                                         │
│    • Async send/receive                                                 │
│    • Priority handling                                                  │
│    • Message history                                                    │
│    • Subscriptions                                                      │
│    • Routing                                                            │
│                                                                          │
│  Benefits:                                                               │
│    • Loose coupling between agents                                     │
│    • Async communication                                               │
│    • Traceable message history                                         │
│    • Scalable architecture                                             │
│                                                                          │
│  Best For:                                                               │
│    • Complex multi-agent systems                                       │
│    • Systems needing audit trails                                      │
│    • Async workflows                                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

Next: [State Sharing](/learn/multi-agents/building-multi-agent/state-sharing) →
