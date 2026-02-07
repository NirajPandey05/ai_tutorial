# Memory and State Management

Learn how to give your agents persistent memory and maintain state across interactions.

## Why Agents Need Memory

Without memory, agents are limited to single interactions. Memory enables:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Why Memory Matters                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   WITHOUT MEMORY                      WITH MEMORY                       │
│   ──────────────                      ───────────                       │
│                                                                          │
│   User: "My name is Alice"            User: "My name is Alice"          │
│   Agent: "Nice to meet you, Alice"    Agent: "Nice to meet you, Alice"  │
│                                        [Stores: user_name = "Alice"]    │
│   ... later ...                                                         │
│                                       ... later ...                     │
│   User: "What's my name?"                                               │
│   Agent: "I don't know your name"     User: "What's my name?"           │
│                                       Agent: "Your name is Alice"       │
│                                                                          │
│   ✗ Can't maintain context           ✓ Remembers past interactions     │
│   ✗ Repeats questions                ✓ Personalizes responses          │
│   ✗ No learning from experience      ✓ Improves over time              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Memory Architecture

### Memory Types

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Agent Memory Architecture                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                      WORKING MEMORY                              │   │
│   │                    (Short-term Buffer)                           │   │
│   │                                                                  │   │
│   │  • Current conversation messages                                │   │
│   │  • Active task state                                            │   │
│   │  • Recent tool results                                          │   │
│   │  • Temporary calculations                                       │   │
│   │                                                                  │   │
│   │  Duration: Session only | Size: Limited (context window)        │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                               │                                          │
│                               ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                     EPISODIC MEMORY                              │   │
│   │                   (Experience Storage)                           │   │
│   │                                                                  │   │
│   │  • Past conversation summaries                                  │   │
│   │  • Completed task records                                       │   │
│   │  • User interactions history                                    │   │
│   │  • Mistakes and corrections                                     │   │
│   │                                                                  │   │
│   │  Duration: Persistent | Storage: Vector database                │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                               │                                          │
│                               ▼                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                     SEMANTIC MEMORY                              │   │
│   │                    (Knowledge Base)                              │   │
│   │                                                                  │   │
│   │  • Learned facts                                                │   │
│   │  • User preferences                                             │   │
│   │  • Domain knowledge                                             │   │
│   │  • Entity relationships                                         │   │
│   │                                                                  │   │
│   │  Duration: Persistent | Storage: Database / Knowledge graph     │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Implementing Working Memory

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

@dataclass
class Message:
    """A single message in the conversation."""
    role: str  # "user", "assistant", "system", "tool"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

class WorkingMemory:
    """Short-term memory for the current session."""
    
    def __init__(self, max_messages: int = 50, max_tokens: int = 8000):
        self.messages: List[Message] = []
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.variables: Dict[str, Any] = {}  # Scratchpad
    
    def add_message(self, role: str, content: str, **metadata):
        """Add a message to working memory."""
        message = Message(
            role=role,
            content=content,
            metadata=metadata
        )
        self.messages.append(message)
        self._enforce_limits()
    
    def get_messages(self, last_n: Optional[int] = None) -> List[Message]:
        """Get messages, optionally limiting to last N."""
        if last_n:
            return self.messages[-last_n:]
        return self.messages
    
    def to_openai_format(self) -> List[Dict[str, str]]:
        """Convert messages to OpenAI API format."""
        return [
            {"role": m.role, "content": m.content}
            for m in self.messages
        ]
    
    def set_variable(self, key: str, value: Any):
        """Set a variable in the scratchpad."""
        self.variables[key] = value
    
    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get a variable from the scratchpad."""
        return self.variables.get(key, default)
    
    def clear(self):
        """Clear all working memory."""
        self.messages = []
        self.variables = {}
    
    def _enforce_limits(self):
        """Remove old messages to stay within limits."""
        # Remove oldest messages if over count limit
        while len(self.messages) > self.max_messages:
            self.messages.pop(0)
        
        # Remove oldest messages if over token limit
        total_tokens = sum(len(m.content.split()) * 1.3 for m in self.messages)
        while total_tokens > self.max_tokens and len(self.messages) > 1:
            self.messages.pop(0)
            total_tokens = sum(len(m.content.split()) * 1.3 for m in self.messages)
    
    def get_context_summary(self) -> str:
        """Generate a summary of current context."""
        return f"""
Working Memory Status:
- Messages: {len(self.messages)}
- Variables: {list(self.variables.keys())}
- First message: {self.messages[0].timestamp if self.messages else 'N/A'}
- Last message: {self.messages[-1].timestamp if self.messages else 'N/A'}
"""
```

## Implementing Episodic Memory

```python
import chromadb
from chromadb.utils import embedding_functions
import json

class EpisodicMemory:
    """Long-term storage of past experiences."""
    
    def __init__(self, agent_id: str, persist_directory: str = "./memory"):
        self.agent_id = agent_id
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Create or get collection for episodes
        self.collection = self.client.get_or_create_collection(
            name=f"{agent_id}_episodes",
            embedding_function=embedding_functions.DefaultEmbeddingFunction()
        )
    
    def store_episode(
        self,
        summary: str,
        full_content: Dict[str, Any],
        episode_type: str = "conversation",
        tags: List[str] = None
    ):
        """Store an episode in long-term memory."""
        
        episode_id = f"{self.agent_id}_{datetime.now().timestamp()}"
        
        self.collection.add(
            documents=[summary],
            metadatas=[{
                "episode_type": episode_type,
                "tags": json.dumps(tags or []),
                "full_content": json.dumps(full_content),
                "timestamp": datetime.now().isoformat(),
                "agent_id": self.agent_id
            }],
            ids=[episode_id]
        )
        
        return episode_id
    
    def recall(
        self,
        query: str,
        n_results: int = 5,
        episode_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Recall similar episodes from memory."""
        
        where_filter = {"agent_id": self.agent_id}
        if episode_type:
            where_filter["episode_type"] = episode_type
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter
        )
        
        episodes = []
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i]
            episodes.append({
                "summary": doc,
                "type": metadata["episode_type"],
                "tags": json.loads(metadata["tags"]),
                "content": json.loads(metadata["full_content"]),
                "timestamp": metadata["timestamp"],
                "distance": results["distances"][0][i] if results["distances"] else None
            })
        
        return episodes
    
    def summarize_conversation(
        self,
        messages: List[Message],
        llm_client
    ) -> str:
        """Generate a summary of a conversation for storage."""
        
        conversation_text = "\n".join([
            f"{m.role}: {m.content}" for m in messages
        ])
        
        response = llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Summarize this conversation in 2-3 sentences, 
capturing the key topic, any decisions made, and important information learned:

{conversation_text}"""
            }],
            max_tokens=200
        )
        
        return response.choices[0].message.content
```

## Implementing Semantic Memory

```python
class SemanticMemory:
    """Knowledge base for facts and relationships."""
    
    def __init__(self, agent_id: str, persist_directory: str = "./memory"):
        self.agent_id = agent_id
        
        # Initialize ChromaDB for semantic storage
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Separate collections for different knowledge types
        self.facts = self.client.get_or_create_collection(
            name=f"{agent_id}_facts"
        )
        self.preferences = self.client.get_or_create_collection(
            name=f"{agent_id}_preferences"
        )
        self.entities = self.client.get_or_create_collection(
            name=f"{agent_id}_entities"
        )
    
    def store_fact(
        self,
        fact: str,
        category: str,
        confidence: float = 1.0,
        source: str = "user"
    ):
        """Store a learned fact."""
        
        fact_id = f"fact_{hash(fact)}"
        
        self.facts.upsert(
            documents=[fact],
            metadatas=[{
                "category": category,
                "confidence": confidence,
                "source": source,
                "timestamp": datetime.now().isoformat()
            }],
            ids=[fact_id]
        )
    
    def store_preference(
        self,
        user_id: str,
        preference_key: str,
        preference_value: str,
        context: str = ""
    ):
        """Store a user preference."""
        
        pref_id = f"{user_id}_{preference_key}"
        
        self.preferences.upsert(
            documents=[f"{preference_key}: {preference_value}. {context}"],
            metadatas=[{
                "user_id": user_id,
                "key": preference_key,
                "value": preference_value,
                "context": context,
                "timestamp": datetime.now().isoformat()
            }],
            ids=[pref_id]
        )
    
    def store_entity(
        self,
        entity_name: str,
        entity_type: str,
        attributes: Dict[str, Any]
    ):
        """Store information about an entity."""
        
        entity_id = f"{entity_type}_{entity_name.lower().replace(' ', '_')}"
        
        # Create a text representation for embedding
        attr_text = ", ".join([f"{k}: {v}" for k, v in attributes.items()])
        doc = f"{entity_type} '{entity_name}' - {attr_text}"
        
        self.entities.upsert(
            documents=[doc],
            metadatas=[{
                "name": entity_name,
                "type": entity_type,
                "attributes": json.dumps(attributes),
                "timestamp": datetime.now().isoformat()
            }],
            ids=[entity_id]
        )
    
    def query_facts(
        self,
        query: str,
        category: Optional[str] = None,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Query facts by semantic similarity."""
        
        where_filter = {}
        if category:
            where_filter["category"] = category
        
        results = self.facts.query(
            query_texts=[query],
            n_results=n_results,
            where=where_filter if where_filter else None
        )
        
        return self._format_results(results)
    
    def get_user_preferences(
        self,
        user_id: str,
        query: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get preferences for a user."""
        
        if query:
            results = self.preferences.query(
                query_texts=[query],
                n_results=10,
                where={"user_id": user_id}
            )
        else:
            results = self.preferences.get(
                where={"user_id": user_id}
            )
        
        return self._format_results(results)
    
    def _format_results(self, results: Dict) -> List[Dict[str, Any]]:
        """Format ChromaDB results into a standard structure."""
        formatted = []
        
        if "documents" in results and results["documents"]:
            docs = results["documents"][0] if isinstance(results["documents"][0], list) else results["documents"]
            metas = results["metadatas"][0] if results.get("metadatas") and isinstance(results["metadatas"][0], list) else results.get("metadatas", [])
            
            for i, doc in enumerate(docs):
                formatted.append({
                    "content": doc,
                    "metadata": metas[i] if i < len(metas) else {}
                })
        
        return formatted
```

## Unified Memory Manager

```python
class MemoryManager:
    """Unified interface for all memory types."""
    
    def __init__(
        self,
        agent_id: str,
        persist_directory: str = "./memory",
        llm_client = None
    ):
        self.agent_id = agent_id
        self.llm_client = llm_client
        
        self.working = WorkingMemory()
        self.episodic = EpisodicMemory(agent_id, persist_directory)
        self.semantic = SemanticMemory(agent_id, persist_directory)
    
    def add_message(self, role: str, content: str, **metadata):
        """Add a message to working memory."""
        self.working.add_message(role, content, **metadata)
    
    def get_context(
        self,
        include_episodic: bool = True,
        include_semantic: bool = True,
        query: str = None
    ) -> str:
        """Build context from all memory types."""
        
        context_parts = []
        
        # Working memory (recent messages)
        recent_messages = self.working.get_messages(last_n=10)
        if recent_messages:
            context_parts.append("## Recent Conversation")
            for msg in recent_messages:
                context_parts.append(f"{msg.role}: {msg.content}")
        
        # Episodic memory (similar past experiences)
        if include_episodic and query:
            episodes = self.episodic.recall(query, n_results=3)
            if episodes:
                context_parts.append("\n## Relevant Past Experiences")
                for ep in episodes:
                    context_parts.append(f"- {ep['summary']}")
        
        # Semantic memory (relevant knowledge)
        if include_semantic and query:
            facts = self.semantic.query_facts(query, n_results=5)
            if facts:
                context_parts.append("\n## Relevant Knowledge")
                for fact in facts:
                    context_parts.append(f"- {fact['content']}")
        
        return "\n".join(context_parts)
    
    def end_session(self, save_episode: bool = True):
        """End the current session and optionally save to episodic memory."""
        
        if save_episode and self.llm_client and len(self.working.messages) > 2:
            # Summarize the conversation
            summary = self.episodic.summarize_conversation(
                self.working.messages,
                self.llm_client
            )
            
            # Store the episode
            self.episodic.store_episode(
                summary=summary,
                full_content={
                    "messages": [
                        {"role": m.role, "content": m.content}
                        for m in self.working.messages
                    ]
                },
                episode_type="conversation"
            )
        
        # Clear working memory
        self.working.clear()
    
    def learn_from_interaction(
        self,
        user_id: str,
        interaction: Dict[str, Any]
    ):
        """Extract and store knowledge from an interaction."""
        
        # Example: Extract user preferences
        if "preference" in interaction:
            self.semantic.store_preference(
                user_id=user_id,
                preference_key=interaction["preference"]["key"],
                preference_value=interaction["preference"]["value"]
            )
        
        # Example: Extract facts
        if "learned_fact" in interaction:
            self.semantic.store_fact(
                fact=interaction["learned_fact"],
                category="user_provided",
                source=user_id
            )
```

## Using Memory in Agents

```python
class MemoryEnabledAgent:
    """Agent with full memory capabilities."""
    
    def __init__(
        self,
        agent_id: str,
        llm_client,
        tools: Dict[str, callable]
    ):
        self.agent_id = agent_id
        self.client = llm_client
        self.tools = tools
        
        # Initialize memory
        self.memory = MemoryManager(
            agent_id=agent_id,
            llm_client=llm_client
        )
    
    async def chat(self, user_id: str, user_message: str) -> str:
        """Process a user message with memory context."""
        
        # Add user message to working memory
        self.memory.add_message("user", user_message, user_id=user_id)
        
        # Build context from memory
        context = self.memory.get_context(
            include_episodic=True,
            include_semantic=True,
            query=user_message
        )
        
        # Get user preferences
        preferences = self.memory.semantic.get_user_preferences(user_id)
        preference_context = "\n".join([
            f"- {p['metadata'].get('key', '')}: {p['metadata'].get('value', '')}"
            for p in preferences
        ]) if preferences else "No stored preferences"
        
        # Build the prompt
        system_prompt = f"""You are a helpful AI assistant with memory.

User Preferences:
{preference_context}

Memory Context:
{context}

Use this context to provide personalized, consistent responses.
Remember information the user shares for future interactions."""

        # Generate response
        response = await self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                *self.memory.working.to_openai_format()
            ]
        )
        
        assistant_message = response.choices[0].message.content
        
        # Store response in working memory
        self.memory.add_message("assistant", assistant_message)
        
        # Extract any learnable information
        await self._extract_knowledge(user_id, user_message, assistant_message)
        
        return assistant_message
    
    async def _extract_knowledge(
        self,
        user_id: str,
        user_message: str,
        assistant_response: str
    ):
        """Extract knowledge from the interaction."""
        
        # Use LLM to identify extractable information
        extraction_prompt = f"""Analyze this conversation turn and extract any:
1. User preferences (e.g., "likes", "prefers", "wants")
2. Facts about the user (e.g., name, location, job)
3. Important information to remember

User: {user_message}
Assistant: {assistant_response}

Respond in JSON format:
{{
    "preferences": [{{"key": "...", "value": "..."}}],
    "facts": ["..."],
    "entities": [{{"name": "...", "type": "...", "attributes": {{}}}}]
}}

If nothing to extract, return empty arrays."""

        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": extraction_prompt}],
            response_format={"type": "json_object"}
        )
        
        try:
            extracted = json.loads(response.choices[0].message.content)
            
            # Store extracted preferences
            for pref in extracted.get("preferences", []):
                self.memory.semantic.store_preference(
                    user_id=user_id,
                    preference_key=pref["key"],
                    preference_value=pref["value"]
                )
            
            # Store extracted facts
            for fact in extracted.get("facts", []):
                self.memory.semantic.store_fact(
                    fact=fact,
                    category="user_info",
                    source=user_id
                )
            
            # Store extracted entities
            for entity in extracted.get("entities", []):
                self.memory.semantic.store_entity(
                    entity_name=entity["name"],
                    entity_type=entity["type"],
                    attributes=entity.get("attributes", {})
                )
                
        except json.JSONDecodeError:
            pass  # Skip if extraction fails
```

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│              Memory and State Management - Summary                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Memory Types:                                                           │
│    • Working: Current session, limited size                             │
│    • Episodic: Past experiences, searchable                             │
│    • Semantic: Facts, preferences, knowledge                            │
│                                                                          │
│  Key Capabilities:                                                       │
│    • Maintain context across turns                                      │
│    • Recall relevant past experiences                                   │
│    • Store and retrieve user preferences                                │
│    • Learn from interactions                                            │
│                                                                          │
│  Implementation Tips:                                                    │
│    • Use vector DBs for semantic search                                 │
│    • Summarize long conversations                                       │
│    • Extract knowledge automatically                                    │
│    • Balance memory size with relevance                                 │
│                                                                          │
│  Storage Options:                                                        │
│    • ChromaDB, Pinecone, Weaviate (vectors)                             │
│    • Redis, SQLite (key-value, structured)                              │
│    • Knowledge graphs (relationships)                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

Next: [ReAct Agent Lab](/learn/agents/agent-loop/react-agent-lab) →
