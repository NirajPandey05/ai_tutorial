# Agent Components

Understand the essential building blocks that make up an AI agent system.

## The Agent Component Stack

Every agent, regardless of architecture, is built from common components:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Agent Component Stack                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                        USER INTERFACE                            │   │
│   │                   (Chat, API, Triggers)                          │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                   │                                      │
│                                   ▼                                      │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                        ORCHESTRATOR                              │   │
│   │              (Agent Loop, State Management)                      │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│           │              │               │              │                │
│           ▼              ▼               ▼              ▼                │
│   ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐           │
│   │   LLM     │  │  MEMORY   │  │  TOOLS    │  │ PLANNING  │           │
│   │  (Brain)  │  │ (Context) │  │ (Actions) │  │ (Strategy)│           │
│   └───────────┘  └───────────┘  └───────────┘  └───────────┘           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## 1. The LLM (Brain)

The LLM is the reasoning engine of the agent.

### Role of the LLM

```yaml
llm_responsibilities:
  reasoning: "Think through problems step-by-step"
  decision_making: "Choose which action to take next"
  natural_language: "Understand and generate text"
  tool_selection: "Decide which tool to use and how"
  synthesis: "Combine information into coherent outputs"
```

### LLM Configuration for Agents

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class AgentLLMConfig:
    """Configuration for agent's LLM."""
    
    # Model selection
    model: str = "gpt-4"
    
    # Generation parameters
    temperature: float = 0.1      # Low for consistency
    max_tokens: int = 2000        # Enough for reasoning
    top_p: float = 0.95
    
    # Agent-specific settings
    system_prompt: str = ""       # Agent personality/instructions
    stop_sequences: list = None   # Stop before tool output
    
    # Reliability settings
    retry_count: int = 3
    timeout: int = 60


class AgentLLM:
    """LLM wrapper optimized for agent use."""
    
    def __init__(self, config: AgentLLMConfig, client):
        self.config = config
        self.client = client
    
    async def generate(
        self, 
        prompt: str,
        stop: list[str] = None
    ) -> str:
        """Generate a response with agent-optimized settings."""
        
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stop=stop or self.config.stop_sequences
        )
        
        return response.choices[0].message.content
    
    async def generate_with_tools(
        self,
        messages: list,
        tools: list[dict]
    ) -> dict:
        """Generate with function calling."""
        
        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=self.config.temperature
        )
        
        return response.choices[0]
```

## 2. Memory System

Memory allows agents to maintain context and learn over time.

### Types of Agent Memory

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Agent Memory Types                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────────────┐                                               │
│   │   WORKING MEMORY    │  Current conversation, immediate context      │
│   │   (Short-term)      │  Stored in: Message history                   │
│   │                     │  Duration: Single session                     │
│   └─────────────────────┘                                               │
│             │                                                            │
│             ▼                                                            │
│   ┌─────────────────────┐                                               │
│   │  EPISODIC MEMORY    │  Past interactions, task histories            │
│   │   (Experiences)     │  Stored in: Vector database                   │
│   │                     │  Duration: Persistent                         │
│   └─────────────────────┘                                               │
│             │                                                            │
│             ▼                                                            │
│   ┌─────────────────────┐                                               │
│   │  SEMANTIC MEMORY    │  Facts, knowledge, learned information        │
│   │   (Knowledge)       │  Stored in: Knowledge graph / Vector DB       │
│   │                     │  Duration: Persistent                         │
│   └─────────────────────┘                                               │
│             │                                                            │
│             ▼                                                            │
│   ┌─────────────────────┐                                               │
│   │ PROCEDURAL MEMORY   │  How to do things, skills, workflows          │
│   │   (Skills)          │  Stored in: Code / Prompts                    │
│   │                     │  Duration: Persistent                         │
│   └─────────────────────┘                                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Memory Implementation

```python
from datetime import datetime
from typing import List, Dict, Any, Optional
import chromadb

class AgentMemory:
    """Comprehensive memory system for agents."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        
        # Working memory (current session)
        self.working_memory: List[Dict] = []
        
        # Long-term storage
        self.chroma_client = chromadb.Client()
        
        # Episodic memory collection
        self.episodes = self.chroma_client.get_or_create_collection(
            name=f"{agent_id}_episodes"
        )
        
        # Semantic memory collection  
        self.knowledge = self.chroma_client.get_or_create_collection(
            name=f"{agent_id}_knowledge"
        )
    
    # Working Memory Methods
    def add_to_working_memory(self, role: str, content: str):
        """Add a message to working memory."""
        self.working_memory.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_working_memory(self, last_n: int = None) -> List[Dict]:
        """Get recent working memory."""
        if last_n:
            return self.working_memory[-last_n:]
        return self.working_memory
    
    def clear_working_memory(self):
        """Clear working memory (e.g., new session)."""
        self.working_memory = []
    
    # Episodic Memory Methods
    def store_episode(self, episode: Dict[str, Any]):
        """Store a completed task/episode."""
        episode_id = f"ep_{datetime.now().timestamp()}"
        
        # Create a summary for embedding
        summary = f"""
        Task: {episode.get('task', '')}
        Outcome: {episode.get('outcome', '')}
        Steps taken: {len(episode.get('steps', []))}
        Success: {episode.get('success', False)}
        """
        
        self.episodes.add(
            documents=[summary],
            metadatas=[{
                "task": episode.get("task", ""),
                "success": episode.get("success", False),
                "timestamp": datetime.now().isoformat(),
                "full_episode": str(episode)
            }],
            ids=[episode_id]
        )
    
    def recall_similar_episodes(
        self, 
        query: str, 
        n_results: int = 3
    ) -> List[Dict]:
        """Recall similar past experiences."""
        results = self.episodes.query(
            query_texts=[query],
            n_results=n_results
        )
        
        return [
            {
                "summary": doc,
                "metadata": meta
            }
            for doc, meta in zip(
                results["documents"][0],
                results["metadatas"][0]
            )
        ]
    
    # Semantic Memory Methods
    def store_knowledge(self, fact: str, metadata: Dict = None):
        """Store a fact or piece of knowledge."""
        fact_id = f"fact_{hash(fact)}"
        
        self.knowledge.add(
            documents=[fact],
            metadatas=[metadata or {}],
            ids=[fact_id]
        )
    
    def query_knowledge(self, query: str, n_results: int = 5) -> List[str]:
        """Query the knowledge base."""
        results = self.knowledge.query(
            query_texts=[query],
            n_results=n_results
        )
        return results["documents"][0]
    
    # Context Building
    def build_context(self, current_task: str) -> str:
        """Build context from all memory types for the current task."""
        
        # Recent working memory
        recent = self.get_working_memory(last_n=10)
        working_context = "\n".join([
            f"{m['role']}: {m['content']}" for m in recent
        ])
        
        # Similar past episodes
        episodes = self.recall_similar_episodes(current_task, n_results=2)
        episode_context = "\n".join([
            f"- Past task: {e['metadata']['task']} (Success: {e['metadata']['success']})"
            for e in episodes
        ])
        
        # Relevant knowledge
        knowledge = self.query_knowledge(current_task, n_results=3)
        knowledge_context = "\n".join([f"- {k}" for k in knowledge])
        
        return f"""
## Recent Conversation:
{working_context}

## Similar Past Experiences:
{episode_context}

## Relevant Knowledge:
{knowledge_context}
"""
```

## 3. Tool System

Tools give agents the ability to take actions in the world.

### Tool Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Tool System Architecture                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                      TOOL REGISTRY                               │   │
│   │                                                                  │   │
│   │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │   │
│   │  │  Search  │ │Calculator│ │  Code    │ │  File    │           │   │
│   │  │   Tool   │ │   Tool   │ │ Executor │ │  System  │           │   │
│   │  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │   │
│   │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │   │
│   │  │   API    │ │ Database │ │  Email   │ │ Browser  │           │   │
│   │  │  Client  │ │  Query   │ │  Sender  │ │  Control │           │   │
│   │  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                   │                                      │
│                                   ▼                                      │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                      TOOL EXECUTOR                               │   │
│   │                                                                  │   │
│   │  • Parse tool call from LLM                                     │   │
│   │  • Validate arguments                                           │   │
│   │  • Execute tool with timeout                                    │   │
│   │  • Format result for LLM                                        │   │
│   │  • Handle errors gracefully                                     │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Tool Implementation

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional
import asyncio
import json

@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    output: str
    error: Optional[str] = None

class Tool(ABC):
    """Base class for all tools."""
    
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given arguments."""
        pass
    
    def to_openai_schema(self) -> Dict:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


class WebSearchTool(Tool):
    """Tool for searching the web."""
    
    name = "web_search"
    description = "Search the web for information. Returns top results."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query"
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return",
                "default": 5
            }
        },
        "required": ["query"]
    }
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    async def execute(self, query: str, num_results: int = 5) -> ToolResult:
        try:
            # Simulate web search (replace with real API)
            results = await self._search(query, num_results)
            
            output = f"Search results for '{query}':\n\n"
            for i, r in enumerate(results, 1):
                output += f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}\n\n"
            
            return ToolResult(success=True, output=output)
            
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class CalculatorTool(Tool):
    """Tool for mathematical calculations."""
    
    name = "calculator"
    description = "Perform mathematical calculations. Supports basic arithmetic and common functions."
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)')"
            }
        },
        "required": ["expression"]
    }
    
    async def execute(self, expression: str) -> ToolResult:
        import math
        
        # Safe evaluation with limited functions
        allowed_names = {
            "abs": abs, "round": round,
            "sqrt": math.sqrt, "pow": pow,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "log": math.log, "log10": math.log10,
            "pi": math.pi, "e": math.e
        }
        
        try:
            # Sanitize expression
            for char in expression:
                if char not in "0123456789+-*/.() " and not char.isalpha():
                    raise ValueError(f"Invalid character: {char}")
            
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return ToolResult(success=True, output=f"{expression} = {result}")
            
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class ToolRegistry:
    """Registry and executor for tools."""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool):
        """Register a tool."""
        self.tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def get_all_schemas(self) -> list:
        """Get OpenAI schemas for all tools."""
        return [tool.to_openai_schema() for tool in self.tools.values()]
    
    async def execute(
        self, 
        tool_name: str, 
        arguments: Dict[str, Any],
        timeout: int = 30
    ) -> ToolResult:
        """Execute a tool with timeout."""
        
        tool = self.get_tool(tool_name)
        if not tool:
            return ToolResult(
                success=False, 
                output="", 
                error=f"Unknown tool: {tool_name}"
            )
        
        try:
            result = await asyncio.wait_for(
                tool.execute(**arguments),
                timeout=timeout
            )
            return result
            
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                output="",
                error=f"Tool execution timed out after {timeout}s"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Tool execution failed: {str(e)}"
            )
```

## 4. Planning System

Planning enables agents to think ahead and strategize.

### Planning Approaches

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Planning Approaches                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   NO PLANNING                     WITH PLANNING                         │
│   ───────────                     ─────────────                         │
│                                                                          │
│   Task: Write report              Task: Write report                    │
│          │                               │                              │
│          ▼                               ▼                              │
│   Immediate action:               1. Research topic                     │
│   Start writing...                2. Create outline                     │
│          │                        3. Write introduction                 │
│          ▼                        4. Write sections                     │
│   Get stuck, backtrack            5. Write conclusion                   │
│          │                        6. Review and edit                    │
│          ▼                               │                              │
│   Try different approach          Execute step by step                  │
│          │                               │                              │
│          ▼                               ▼                              │
│   Eventually finish               Systematic completion                 │
│   (inefficient)                   (efficient)                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Planning Implementation

```python
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

class StepStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class PlanStep:
    """A single step in a plan."""
    id: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None

@dataclass
class Plan:
    """A complete plan for accomplishing a task."""
    task: str
    steps: List[PlanStep]
    current_step_index: int = 0
    
    @property
    def current_step(self) -> Optional[PlanStep]:
        if self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None
    
    @property
    def is_complete(self) -> bool:
        return all(s.status == StepStatus.COMPLETED for s in self.steps)
    
    @property
    def progress(self) -> float:
        completed = sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)
        return completed / len(self.steps) if self.steps else 0


class Planner:
    """Creates and manages plans for tasks."""
    
    def __init__(self, llm):
        self.llm = llm
    
    async def create_plan(self, task: str, context: str = "") -> Plan:
        """Create a plan for a task."""
        
        prompt = f"""Create a detailed plan to accomplish this task:

Task: {task}

Context: {context}

Break this into clear, actionable steps. Consider:
- What information is needed
- What actions must be taken
- What order makes sense
- What might depend on what

Format your response as a numbered list:
1. [Step description]
2. [Step description]
...

Only include necessary steps. Be specific."""

        response = await self.llm.generate(prompt)
        steps = self._parse_steps(response)
        
        return Plan(task=task, steps=steps)
    
    def _parse_steps(self, response: str) -> List[PlanStep]:
        """Parse LLM response into plan steps."""
        steps = []
        lines = response.strip().split("\n")
        
        for i, line in enumerate(lines):
            # Extract step text
            line = line.strip()
            if line and line[0].isdigit():
                # Remove numbering
                text = line.split(".", 1)[-1].strip()
                
                steps.append(PlanStep(
                    id=f"step_{i+1}",
                    description=text,
                    dependencies=[f"step_{i}"] if i > 0 else []
                ))
        
        return steps
    
    async def should_replan(
        self, 
        plan: Plan, 
        observation: str
    ) -> tuple[bool, str]:
        """Determine if we should replan based on an observation."""
        
        prompt = f"""Given this plan and recent observation, should we replan?

Original Task: {plan.task}

Current Plan:
{self._format_plan(plan)}

Recent Observation: {observation}

Should we:
A) Continue with current plan
B) Modify the plan
C) Start over with new plan

Respond with A, B, or C and explain why."""

        response = await self.llm.generate(prompt)
        
        if response.startswith("A"):
            return False, "Continue with current plan"
        else:
            return True, response
    
    def _format_plan(self, plan: Plan) -> str:
        """Format plan for display."""
        lines = []
        for step in plan.steps:
            status_icon = {
                StepStatus.PENDING: "⏳",
                StepStatus.IN_PROGRESS: "🔄",
                StepStatus.COMPLETED: "✅",
                StepStatus.FAILED: "❌",
                StepStatus.SKIPPED: "⏭️"
            }[step.status]
            
            lines.append(f"{status_icon} {step.id}: {step.description}")
            
        return "\n".join(lines)
```

## 5. Orchestrator

The orchestrator ties all components together.

```python
class AgentOrchestrator:
    """Main orchestrator that coordinates all agent components."""
    
    def __init__(
        self,
        llm: AgentLLM,
        memory: AgentMemory,
        tools: ToolRegistry,
        planner: Planner
    ):
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.planner = planner
        
        self.max_iterations = 20
        self.current_plan: Optional[Plan] = None
    
    async def run(self, task: str) -> str:
        """Run the agent on a task."""
        
        # Store task in memory
        self.memory.add_to_working_memory("user", task)
        
        # Build context from memory
        context = self.memory.build_context(task)
        
        # Create a plan
        self.current_plan = await self.planner.create_plan(task, context)
        
        # Execute the plan
        for iteration in range(self.max_iterations):
            if self.current_plan.is_complete:
                break
            
            step = self.current_plan.current_step
            if not step:
                break
            
            # Execute current step
            step.status = StepStatus.IN_PROGRESS
            result = await self._execute_step(step)
            
            if result.success:
                step.status = StepStatus.COMPLETED
                step.result = result.output
                self.current_plan.current_step_index += 1
            else:
                step.status = StepStatus.FAILED
                step.error = result.error
                
                # Check if we should replan
                should_replan, reason = await self.planner.should_replan(
                    self.current_plan, result.error
                )
                if should_replan:
                    self.current_plan = await self.planner.create_plan(
                        task, 
                        f"Previous attempt failed: {result.error}"
                    )
        
        # Generate final response
        final_response = await self._synthesize_response(task)
        
        # Store episode in memory
        self.memory.store_episode({
            "task": task,
            "plan": self.current_plan,
            "success": self.current_plan.is_complete,
            "outcome": final_response
        })
        
        return final_response
    
    async def _execute_step(self, step: PlanStep) -> ToolResult:
        """Execute a single plan step."""
        
        # Determine if tools are needed
        prompt = f"""Execute this step: {step.description}

Available tools: {[t.name for t in self.tools.tools.values()]}

If you need a tool, respond with:
TOOL: <tool_name>
INPUT: <json_input>

If no tool needed, just provide the result."""

        response = await self.llm.generate(prompt)
        
        if "TOOL:" in response:
            # Parse and execute tool
            tool_name, tool_input = self._parse_tool_call(response)
            return await self.tools.execute(tool_name, tool_input)
        else:
            return ToolResult(success=True, output=response)
```

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   Agent Components - Summary                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  LLM (Brain):     Reasoning, decision-making, language understanding    │
│                                                                          │
│  Memory:          Working (short-term), Episodic (experiences),         │
│                   Semantic (knowledge), Procedural (skills)             │
│                                                                          │
│  Tools:           Actions the agent can take in the world               │
│                   (search, calculate, code, API calls, etc.)            │
│                                                                          │
│  Planning:        Breaking tasks into steps, strategizing               │
│                                                                          │
│  Orchestrator:    Coordinates all components, manages the agent loop    │
│                                                                          │
│  Key Insight:     An agent is more than just an LLM - it's a system    │
│                   of interacting components working together            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

Next: [Function Calling Basics](/learn/agents/tool-use/function-calling) →
