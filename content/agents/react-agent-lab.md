# Lab: Building a ReAct Agent

Build a complete ReAct agent with memory, multiple tools, and streaming output.

## Lab Objectives

By the end of this lab, you will:
- Implement a full ReAct agent from scratch
- Add working and episodic memory
- Create multiple useful tools
- Build a streaming interface
- Handle edge cases and errors

## Prerequisites

```bash
pip install openai chromadb python-dotenv httpx rich
```

```python
# .env file
OPENAI_API_KEY=your-api-key-here
```

## Project Structure

```
react_agent/
├── agent.py          # Main agent implementation
├── memory.py         # Memory systems
├── tools.py          # Tool implementations
├── parser.py         # Response parsing
├── main.py           # Entry point
└── tests.py          # Test cases
```

## Part 1: Core Data Structures

### Step 1: Define Core Types (parser.py)

```python
# parser.py
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum
import json
import re


class ActionType(Enum):
    TOOL_USE = "tool_use"
    FINAL_ANSWER = "final_answer"


@dataclass
class Thought:
    """Agent's reasoning."""
    content: str
    
    def __str__(self) -> str:
        return f"💭 THOUGHT: {self.content}"


@dataclass  
class Action:
    """Agent's decided action."""
    type: ActionType
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    answer: Optional[str] = None
    
    def __str__(self) -> str:
        if self.type == ActionType.FINAL_ANSWER:
            return f"✅ FINAL ANSWER: {self.answer}"
        return f"🔧 ACTION: {self.tool_name}({json.dumps(self.tool_input)})"


@dataclass
class Observation:
    """Result from environment/tool."""
    content: str
    source: str
    success: bool = True
    
    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"👁 OBSERVATION [{self.source}] {status}: {self.content}"


@dataclass
class Step:
    """A single step in the agent's trajectory."""
    thought: Thought
    action: Action
    observation: Optional[Observation] = None


class ResponseParser:
    """Parse LLM responses into structured components."""
    
    @staticmethod
    def parse(response: str) -> tuple[Thought, Action]:
        """Parse a ReAct response into thought and action."""
        
        # Extract thought
        thought_content = ""
        thought_match = re.search(
            r'THOUGHT:\s*(.+?)(?=ACTION:|FINAL ANSWER:|$)', 
            response, 
            re.DOTALL | re.IGNORECASE
        )
        if thought_match:
            thought_content = thought_match.group(1).strip()
        
        thought = Thought(content=thought_content)
        
        # Check for final answer
        final_match = re.search(
            r'FINAL ANSWER:\s*(.+?)$',
            response,
            re.DOTALL | re.IGNORECASE
        )
        if final_match:
            return thought, Action(
                type=ActionType.FINAL_ANSWER,
                answer=final_match.group(1).strip()
            )
        
        # Extract action and input
        action_match = re.search(
            r'ACTION:\s*(\w+)',
            response,
            re.IGNORECASE
        )
        
        input_match = re.search(
            r'ACTION INPUT:\s*(.+?)(?=THOUGHT:|ACTION:|OBSERVATION:|$)',
            response,
            re.DOTALL | re.IGNORECASE
        )
        
        if action_match:
            tool_name = action_match.group(1).strip()
            tool_input = {}
            
            if input_match:
                input_str = input_match.group(1).strip()
                try:
                    tool_input = json.loads(input_str)
                except json.JSONDecodeError:
                    # Try to extract as simple string
                    tool_input = {"input": input_str}
            
            return thought, Action(
                type=ActionType.TOOL_USE,
                tool_name=tool_name,
                tool_input=tool_input
            )
        
        # Fallback: treat entire response as final answer
        return thought, Action(
            type=ActionType.FINAL_ANSWER,
            answer=response
        )
```

## Part 2: Memory System

### Step 2: Implement Memory (memory.py)

```python
# memory.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import chromadb


@dataclass
class Message:
    """A conversation message."""
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


class WorkingMemory:
    """Short-term memory for current session."""
    
    def __init__(self, max_messages: int = 50):
        self.messages: List[Message] = []
        self.max_messages = max_messages
        self.scratchpad: Dict[str, Any] = {}
    
    def add(self, role: str, content: str):
        """Add a message."""
        self.messages.append(Message(role=role, content=content))
        self._enforce_limit()
    
    def get_messages(self, last_n: Optional[int] = None) -> List[Message]:
        """Get messages."""
        if last_n:
            return self.messages[-last_n:]
        return self.messages
    
    def to_prompt(self) -> str:
        """Convert to prompt format."""
        lines = []
        for msg in self.messages:
            lines.append(f"{msg.role.upper()}: {msg.content}")
        return "\n".join(lines)
    
    def set(self, key: str, value: Any):
        """Set scratchpad value."""
        self.scratchpad[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get scratchpad value."""
        return self.scratchpad.get(key, default)
    
    def clear(self):
        """Clear all memory."""
        self.messages = []
        self.scratchpad = {}
    
    def _enforce_limit(self):
        """Keep messages under limit."""
        while len(self.messages) > self.max_messages:
            self.messages.pop(0)


class EpisodicMemory:
    """Long-term memory for past experiences."""
    
    def __init__(self, agent_id: str, persist_path: str = "./agent_memory"):
        self.agent_id = agent_id
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(
            name=f"{agent_id}_episodes"
        )
    
    def store(
        self,
        summary: str,
        details: Dict[str, Any],
        tags: List[str] = None
    ) -> str:
        """Store an episode."""
        
        episode_id = f"ep_{datetime.now().timestamp()}"
        
        self.collection.add(
            documents=[summary],
            metadatas=[{
                "details": json.dumps(details),
                "tags": json.dumps(tags or []),
                "timestamp": datetime.now().isoformat()
            }],
            ids=[episode_id]
        )
        
        return episode_id
    
    def recall(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Recall relevant episodes."""
        
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        episodes = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i]
                episodes.append({
                    "summary": doc,
                    "details": json.loads(meta.get("details", "{}")),
                    "tags": json.loads(meta.get("tags", "[]")),
                    "timestamp": meta.get("timestamp"),
                    "relevance": 1 - results["distances"][0][i] if results["distances"] else None
                })
        
        return episodes
    
    def get_recent(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get most recent episodes."""
        
        all_results = self.collection.get()
        
        if not all_results["documents"]:
            return []
        
        # Sort by timestamp
        episodes = []
        for i, doc in enumerate(all_results["documents"]):
            meta = all_results["metadatas"][i]
            episodes.append({
                "summary": doc,
                "timestamp": meta.get("timestamp", "")
            })
        
        episodes.sort(key=lambda x: x["timestamp"], reverse=True)
        return episodes[:n]


class AgentMemory:
    """Combined memory system."""
    
    def __init__(self, agent_id: str):
        self.working = WorkingMemory()
        self.episodic = EpisodicMemory(agent_id)
    
    def build_context(self, current_query: str) -> str:
        """Build full context from all memories."""
        
        parts = []
        
        # Add relevant past episodes
        episodes = self.episodic.recall(current_query, n_results=3)
        if episodes:
            parts.append("## Relevant Past Experiences:")
            for ep in episodes:
                parts.append(f"- {ep['summary']}")
        
        # Add recent conversation
        recent = self.working.get_messages(last_n=10)
        if recent:
            parts.append("\n## Recent Conversation:")
            for msg in recent:
                parts.append(f"{msg.role}: {msg.content}")
        
        return "\n".join(parts)
```

## Part 3: Tool System

### Step 3: Implement Tools (tools.py)

```python
# tools.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import httpx
import math
import json
from datetime import datetime


@dataclass
class ToolResult:
    """Result of tool execution."""
    success: bool
    output: str
    error: Optional[str] = None


class Tool(ABC):
    """Base tool class."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    async def run(self, **kwargs) -> ToolResult:
        pass
    
    def format_for_prompt(self) -> str:
        """Format tool description for prompt."""
        params = ", ".join([
            f"{k}: {v.get('description', v.get('type', 'any'))}"
            for k, v in self.parameters.get("properties", {}).items()
        ])
        return f"- {self.name}({params}): {self.description}"


class CalculatorTool(Tool):
    @property
    def name(self) -> str:
        return "calculator"
    
    @property
    def description(self) -> str:
        return "Evaluate mathematical expressions. Supports +, -, *, /, **, sqrt, sin, cos, tan, log."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate"
                }
            },
            "required": ["expression"]
        }
    
    async def run(self, expression: str) -> ToolResult:
        safe_dict = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sqrt": math.sqrt, "pow": pow,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "log": math.log, "log10": math.log10,
            "pi": math.pi, "e": math.e
        }
        
        try:
            expression = expression.replace("^", "**")
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            return ToolResult(success=True, output=f"{expression} = {result}")
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class WikipediaTool(Tool):
    @property
    def name(self) -> str:
        return "wikipedia"
    
    @property
    def description(self) -> str:
        return "Search Wikipedia for information. Returns a summary of the topic."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Topic to search for"
                }
            },
            "required": ["query"]
        }
    
    async def run(self, query: str) -> ToolResult:
        try:
            async with httpx.AsyncClient() as client:
                # Search for page
                search_url = "https://en.wikipedia.org/w/api.php"
                search_params = {
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "format": "json",
                    "srlimit": 1
                }
                
                search_resp = await client.get(search_url, params=search_params, timeout=10.0)
                search_data = search_resp.json()
                
                if not search_data.get("query", {}).get("search"):
                    return ToolResult(success=True, output=f"No Wikipedia article found for '{query}'")
                
                title = search_data["query"]["search"][0]["title"]
                
                # Get summary
                summary_params = {
                    "action": "query",
                    "prop": "extracts",
                    "exintro": True,
                    "explaintext": True,
                    "titles": title,
                    "format": "json"
                }
                
                summary_resp = await client.get(search_url, params=summary_params, timeout=10.0)
                summary_data = summary_resp.json()
                
                pages = summary_data.get("query", {}).get("pages", {})
                if pages:
                    page = list(pages.values())[0]
                    extract = page.get("extract", "")[:1000]  # Limit length
                    return ToolResult(
                        success=True,
                        output=f"Wikipedia: {title}\n\n{extract}"
                    )
                
                return ToolResult(success=True, output=f"Could not retrieve summary for '{title}'")
                
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class DateTimeTool(Tool):
    @property
    def name(self) -> str:
        return "datetime"
    
    @property
    def description(self) -> str:
        return "Get current date/time or calculate days between dates."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["now", "date", "weekday", "days_until", "days_between"],
                    "description": "Operation to perform"
                },
                "date1": {
                    "type": "string",
                    "description": "First date (YYYY-MM-DD format)"
                },
                "date2": {
                    "type": "string",
                    "description": "Second date (YYYY-MM-DD format)"
                }
            },
            "required": ["operation"]
        }
    
    async def run(
        self, 
        operation: str, 
        date1: str = None, 
        date2: str = None
    ) -> ToolResult:
        try:
            now = datetime.now()
            
            if operation == "now":
                return ToolResult(
                    success=True,
                    output=f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
                )
            
            elif operation == "date":
                return ToolResult(
                    success=True,
                    output=f"Today's date: {now.strftime('%Y-%m-%d')}"
                )
            
            elif operation == "weekday":
                return ToolResult(
                    success=True,
                    output=f"Today is {now.strftime('%A')}"
                )
            
            elif operation == "days_until":
                if not date1:
                    return ToolResult(success=False, output="", error="date1 required")
                target = datetime.strptime(date1, "%Y-%m-%d")
                days = (target - now).days
                return ToolResult(
                    success=True,
                    output=f"{days} days until {date1}"
                )
            
            elif operation == "days_between":
                if not date1 or not date2:
                    return ToolResult(success=False, output="", error="date1 and date2 required")
                d1 = datetime.strptime(date1, "%Y-%m-%d")
                d2 = datetime.strptime(date2, "%Y-%m-%d")
                days = abs((d2 - d1).days)
                return ToolResult(
                    success=True,
                    output=f"{days} days between {date1} and {date2}"
                )
            
            return ToolResult(success=False, output="", error=f"Unknown operation: {operation}")
            
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))


class WeatherTool(Tool):
    @property
    def name(self) -> str:
        return "weather"
    
    @property
    def description(self) -> str:
        return "Get current weather for a city (simulated data for demo)."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["city"]
        }
    
    async def run(self, city: str) -> ToolResult:
        # Simulated weather data
        import random
        
        conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Rainy", "Stormy"]
        temp = random.randint(10, 35)
        humidity = random.randint(30, 90)
        condition = random.choice(conditions)
        
        return ToolResult(
            success=True,
            output=f"Weather in {city}: {temp}°C, {condition}, {humidity}% humidity"
        )


class ToolRegistry:
    """Registry of available tools."""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool):
        """Register a tool."""
        self.tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> List[str]:
        """List all tool names."""
        return list(self.tools.keys())
    
    def format_all_for_prompt(self) -> str:
        """Format all tools for prompt."""
        return "\n".join([tool.format_for_prompt() for tool in self.tools.values()])
    
    async def execute(self, name: str, **kwargs) -> ToolResult:
        """Execute a tool."""
        tool = self.get(name)
        if not tool:
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown tool: {name}. Available: {self.list_tools()}"
            )
        
        try:
            return await tool.run(**kwargs)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))
```

## Part 4: The ReAct Agent

### Step 4: Implement the Agent (agent.py)

```python
# agent.py
import asyncio
from typing import List, Optional, AsyncGenerator
from openai import AsyncOpenAI
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from parser import ResponseParser, Thought, Action, Observation, Step, ActionType
from memory import AgentMemory
from tools import ToolRegistry, ToolResult


class ReActAgent:
    """Complete ReAct agent implementation."""
    
    def __init__(
        self,
        agent_id: str,
        api_key: str,
        model: str = "gpt-4",
        max_steps: int = 10
    ):
        self.agent_id = agent_id
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.max_steps = max_steps
        
        self.memory = AgentMemory(agent_id)
        self.tools = ToolRegistry()
        self.parser = ResponseParser()
        self.console = Console()
        
        self._build_system_prompt()
    
    def _build_system_prompt(self):
        """Build the system prompt."""
        self.system_prompt_template = """You are a ReAct agent that solves problems step-by-step using tools.

## Available Tools
{tool_descriptions}

## Response Format
Always respond in this EXACT format:

THOUGHT: [Your detailed reasoning about what to do next]
ACTION: [tool_name]
ACTION INPUT: {{"param": "value"}}

OR when you have the final answer:

THOUGHT: [Your reasoning for the conclusion]
FINAL ANSWER: [Your complete answer to the user]

## Rules
1. Always start with THOUGHT to explain your reasoning
2. Only use tools from the list above
3. Provide valid JSON for ACTION INPUT
4. When you have enough information, provide FINAL ANSWER
5. If a tool fails, try a different approach
6. Be thorough but efficient

## Memory Context
{memory_context}
"""
    
    def register_tool(self, tool):
        """Register a tool with the agent."""
        self.tools.register(tool)
        self.console.print(f"[green]✓ Registered tool: {tool.name}[/green]")
    
    async def run(self, query: str, stream: bool = False) -> str:
        """Run the agent on a query."""
        
        # Store query in memory
        self.memory.working.add("user", query)
        
        # Build context
        memory_context = self.memory.build_context(query)
        tool_descriptions = self.tools.format_all_for_prompt()
        
        system_prompt = self.system_prompt_template.format(
            tool_descriptions=tool_descriptions,
            memory_context=memory_context
        )
        
        # Initialize trajectory
        trajectory: List[Step] = []
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Task: {query}"}
        ]
        
        for step_num in range(self.max_steps):
            self.console.print(f"\n[bold blue]═══ Step {step_num + 1} ═══[/bold blue]")
            
            # Get LLM response
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=1000
            )
            
            response_text = response.choices[0].message.content
            
            # Parse response
            thought, action = self.parser.parse(response_text)
            
            # Display thought
            self.console.print(Panel(
                thought.content,
                title="💭 Thought",
                border_style="cyan"
            ))
            
            # Check for final answer
            if action.type == ActionType.FINAL_ANSWER:
                self.console.print(Panel(
                    action.answer,
                    title="✅ Final Answer",
                    border_style="green"
                ))
                
                # Store in memory
                self.memory.working.add("assistant", action.answer)
                
                # Save episode
                self._save_episode(query, trajectory, action.answer)
                
                return action.answer
            
            # Display action
            self.console.print(Panel(
                f"Tool: {action.tool_name}\nInput: {action.tool_input}",
                title="🔧 Action",
                border_style="yellow"
            ))
            
            # Execute tool
            result = await self.tools.execute(
                action.tool_name,
                **(action.tool_input or {})
            )
            
            observation = Observation(
                content=result.output if result.success else f"Error: {result.error}",
                source=action.tool_name,
                success=result.success
            )
            
            # Display observation
            style = "green" if observation.success else "red"
            self.console.print(Panel(
                observation.content[:500] + ("..." if len(observation.content) > 500 else ""),
                title=f"👁 Observation ({observation.source})",
                border_style=style
            ))
            
            # Record step
            step = Step(thought=thought, action=action, observation=observation)
            trajectory.append(step)
            
            # Update messages for next iteration
            messages.append({"role": "assistant", "content": response_text})
            messages.append({
                "role": "user",
                "content": f"OBSERVATION: {observation.content}\n\nContinue with your next thought and action."
            })
        
        # Max steps reached
        final = "I've reached the maximum number of steps. Based on my investigation:\n"
        final += "\n".join([
            f"- {step.observation.content[:100]}..."
            for step in trajectory if step.observation
        ])
        
        self.memory.working.add("assistant", final)
        return final
    
    def _save_episode(
        self,
        query: str,
        trajectory: List[Step],
        answer: str
    ):
        """Save the episode to long-term memory."""
        
        summary = f"Task: {query[:100]}... | Answer: {answer[:100]}..."
        details = {
            "query": query,
            "answer": answer,
            "steps": len(trajectory),
            "tools_used": [
                step.action.tool_name
                for step in trajectory
                if step.action.type == ActionType.TOOL_USE
            ]
        }
        
        self.memory.episodic.store(
            summary=summary,
            details=details,
            tags=["completed_task"]
        )
    
    async def chat(self, message: str) -> str:
        """Simple chat interface."""
        return await self.run(message)
```

## Part 5: Main Application

### Step 5: Create Entry Point (main.py)

```python
# main.py
import asyncio
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt

from agent import ReActAgent
from tools import (
    CalculatorTool,
    WikipediaTool,
    DateTimeTool,
    WeatherTool
)


async def demo():
    """Run demo queries."""
    
    load_dotenv()
    console = Console()
    
    # Create agent
    agent = ReActAgent(
        agent_id="demo_agent",
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4",
        max_steps=10
    )
    
    # Register tools
    agent.register_tool(CalculatorTool())
    agent.register_tool(WikipediaTool())
    agent.register_tool(DateTimeTool())
    agent.register_tool(WeatherTool())
    
    # Demo queries
    queries = [
        "What is the square root of 144 plus 25% of 80?",
        "Who invented the telephone and in what year?",
        "What day of the week is it today, and what's the weather like in Paris?",
    ]
    
    for query in queries:
        console.print(f"\n[bold magenta]{'='*60}[/bold magenta]")
        console.print(f"[bold]Query: {query}[/bold]")
        console.print(f"[bold magenta]{'='*60}[/bold magenta]")
        
        answer = await agent.run(query)
        
        console.print(f"\n[bold green]Final Result:[/bold green]")
        console.print(answer)
        console.print()


async def interactive():
    """Run interactive chat."""
    
    load_dotenv()
    console = Console()
    
    console.print("[bold cyan]🤖 ReAct Agent Interactive Mode[/bold cyan]")
    console.print("Type 'quit' to exit, 'history' to see past queries\n")
    
    # Create agent
    agent = ReActAgent(
        agent_id="interactive_agent",
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4",
        max_steps=10
    )
    
    # Register tools
    agent.register_tool(CalculatorTool())
    agent.register_tool(WikipediaTool())
    agent.register_tool(DateTimeTool())
    agent.register_tool(WeatherTool())
    
    while True:
        try:
            query = Prompt.ask("\n[bold cyan]You[/bold cyan]")
            
            if query.lower() == 'quit':
                console.print("[yellow]Goodbye![/yellow]")
                break
            
            if query.lower() == 'history':
                episodes = agent.memory.episodic.get_recent(5)
                console.print("\n[bold]Recent Interactions:[/bold]")
                for ep in episodes:
                    console.print(f"- {ep['summary']}")
                continue
            
            if not query.strip():
                continue
            
            answer = await agent.run(query)
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type 'quit' to exit.[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        asyncio.run(demo())
    else:
        asyncio.run(interactive())
```

## Part 6: Testing

### Step 6: Create Tests (tests.py)

```python
# tests.py
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from parser import ResponseParser, ActionType
from tools import CalculatorTool, DateTimeTool, ToolRegistry
from memory import WorkingMemory, AgentMemory


class TestResponseParser:
    def test_parse_thought_and_action(self):
        response = """THOUGHT: I need to calculate this
ACTION: calculator
ACTION INPUT: {"expression": "2 + 2"}"""
        
        thought, action = ResponseParser.parse(response)
        
        assert "calculate" in thought.content.lower()
        assert action.type == ActionType.TOOL_USE
        assert action.tool_name == "calculator"
        assert action.tool_input == {"expression": "2 + 2"}
    
    def test_parse_final_answer(self):
        response = """THOUGHT: I have all the information
FINAL ANSWER: The answer is 42."""
        
        thought, action = ResponseParser.parse(response)
        
        assert action.type == ActionType.FINAL_ANSWER
        assert action.answer == "The answer is 42."


class TestTools:
    @pytest.mark.asyncio
    async def test_calculator(self):
        tool = CalculatorTool()
        result = await tool.run(expression="2 + 2")
        
        assert result.success
        assert "4" in result.output
    
    @pytest.mark.asyncio
    async def test_calculator_error(self):
        tool = CalculatorTool()
        result = await tool.run(expression="1/0")
        
        assert not result.success
        assert result.error is not None
    
    @pytest.mark.asyncio
    async def test_datetime(self):
        tool = DateTimeTool()
        result = await tool.run(operation="weekday")
        
        assert result.success
        assert any(day in result.output for day in 
                   ["Monday", "Tuesday", "Wednesday", "Thursday", 
                    "Friday", "Saturday", "Sunday"])


class TestToolRegistry:
    @pytest.mark.asyncio
    async def test_registry(self):
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        
        assert "calculator" in registry.list_tools()
        
        result = await registry.execute("calculator", expression="3 * 3")
        assert result.success
        assert "9" in result.output
    
    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        registry = ToolRegistry()
        result = await registry.execute("unknown_tool")
        
        assert not result.success
        assert "Unknown tool" in result.error


class TestWorkingMemory:
    def test_add_and_get(self):
        memory = WorkingMemory()
        memory.add("user", "Hello")
        memory.add("assistant", "Hi there!")
        
        messages = memory.get_messages()
        assert len(messages) == 2
        assert messages[0].content == "Hello"
    
    def test_max_limit(self):
        memory = WorkingMemory(max_messages=3)
        
        for i in range(5):
            memory.add("user", f"Message {i}")
        
        messages = memory.get_messages()
        assert len(messages) == 3
        assert messages[0].content == "Message 2"  # First 2 removed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## Running the Lab

```bash
# Run demo mode
python main.py demo

# Run interactive mode
python main.py

# Run tests
pytest tests.py -v
```

## Exercises

1. **Add a new tool**: Create a `NewsSearchTool` that searches for recent news
2. **Implement streaming**: Modify the agent to stream thoughts as they're generated
3. **Add tool history**: Track which tools were used most frequently
4. **Implement retry logic**: Add automatic retry when tools fail

## Summary

You've built a complete ReAct agent with:
- ✅ Structured response parsing
- ✅ Working and episodic memory
- ✅ Multiple useful tools
- ✅ Rich console output
- ✅ Error handling
- ✅ Test coverage

Next: [Web Agents](/learn/agents/agent-capabilities/web-agents) →
