# Lab: Building a Function-Calling Agent

Build a complete agent that uses multiple tools to answer user questions and complete tasks.

## Lab Objectives

By the end of this lab, you will:
- Create a multi-tool agent from scratch
- Define proper tool schemas
- Implement robust tool execution
- Handle errors gracefully
- Process multiple tool calls

## Prerequisites

```bash
pip install openai python-dotenv httpx
```

```python
# .env file
OPENAI_API_KEY=your-api-key-here
```

## Part 1: Setting Up the Foundation

### Step 1: Create the Base Tool Class

```python
# tools.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional
import json

@dataclass
class ToolResult:
    """Result from executing a tool."""
    success: bool
    output: str
    error: Optional[str] = None

class BaseTool(ABC):
    """Abstract base class for all tools."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for the LLM."""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """JSON Schema for parameters."""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool."""
        pass
    
    def to_openai_schema(self) -> Dict:
        """Convert to OpenAI function format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }
```

### Step 2: Implement Concrete Tools

```python
# tools.py (continued)
import httpx
import math
from datetime import datetime

class CalculatorTool(BaseTool):
    """Tool for mathematical calculations."""
    
    @property
    def name(self) -> str:
        return "calculator"
    
    @property
    def description(self) -> str:
        return """Perform mathematical calculations. 
Supports basic arithmetic (+, -, *, /), exponents (**), 
and functions like sqrt, sin, cos, tan, log.

Examples:
- "2 + 2" → 4
- "sqrt(16)" → 4.0
- "sin(3.14159/2)" → 1.0
- "10 ** 2" → 100"""
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }
    
    async def execute(self, expression: str) -> ToolResult:
        """Evaluate a mathematical expression safely."""
        
        # Allowed functions
        safe_dict = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sqrt": math.sqrt, "pow": pow,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "log": math.log, "log10": math.log10, "log2": math.log2,
            "exp": math.exp, "floor": math.floor, "ceil": math.ceil,
            "pi": math.pi, "e": math.e
        }
        
        try:
            # Basic sanitization
            expression = expression.replace("^", "**")
            
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            return ToolResult(
                success=True,
                output=f"Result: {result}"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Calculation error: {str(e)}"
            )


class WebSearchTool(BaseTool):
    """Tool for searching the web (using DuckDuckGo API)."""
    
    @property
    def name(self) -> str:
        return "web_search"
    
    @property
    def description(self) -> str:
        return """Search the web for current information.
Use this when you need up-to-date information, facts, 
news, or data that might not be in your training data.

Returns summaries and URLs of relevant results."""
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results (1-5)",
                    "default": 3
                }
            },
            "required": ["query"]
        }
    
    async def execute(
        self, 
        query: str, 
        max_results: int = 3
    ) -> ToolResult:
        """Search using DuckDuckGo instant answers."""
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.duckduckgo.com/",
                    params={
                        "q": query,
                        "format": "json",
                        "no_html": 1
                    },
                    timeout=10.0
                )
                
                data = response.json()
                
                results = []
                
                # Abstract (main answer)
                if data.get("Abstract"):
                    results.append(f"Summary: {data['Abstract']}")
                    if data.get("AbstractURL"):
                        results.append(f"Source: {data['AbstractURL']}")
                
                # Related topics
                for topic in data.get("RelatedTopics", [])[:max_results]:
                    if isinstance(topic, dict) and topic.get("Text"):
                        results.append(f"- {topic['Text']}")
                
                if not results:
                    return ToolResult(
                        success=True,
                        output=f"No results found for '{query}'. Try a different search query."
                    )
                
                return ToolResult(
                    success=True,
                    output="\n".join(results)
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Search failed: {str(e)}"
            )


class DateTimeTool(BaseTool):
    """Tool for getting current date/time information."""
    
    @property
    def name(self) -> str:
        return "datetime"
    
    @property
    def description(self) -> str:
        return """Get current date, time, or perform date calculations.

Operations:
- "now": Get current date and time
- "today": Get today's date
- "weekday": Get current day of the week
- "days_until:YYYY-MM-DD": Days until a specific date"""
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "The operation to perform",
                    "enum": ["now", "today", "weekday", "days_until"]
                },
                "target_date": {
                    "type": "string",
                    "description": "Target date for calculations (YYYY-MM-DD format)"
                }
            },
            "required": ["operation"]
        }
    
    async def execute(
        self, 
        operation: str, 
        target_date: str = None
    ) -> ToolResult:
        """Execute datetime operation."""
        
        now = datetime.now()
        
        try:
            if operation == "now":
                return ToolResult(
                    success=True,
                    output=f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}"
                )
            
            elif operation == "today":
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
                if not target_date:
                    return ToolResult(
                        success=False,
                        output="",
                        error="target_date is required for days_until operation"
                    )
                
                target = datetime.strptime(target_date, "%Y-%m-%d")
                delta = (target - now).days
                
                if delta < 0:
                    return ToolResult(
                        success=True,
                        output=f"{target_date} was {abs(delta)} days ago"
                    )
                else:
                    return ToolResult(
                        success=True,
                        output=f"{delta} days until {target_date}"
                    )
            
            else:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Unknown operation: {operation}"
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"DateTime error: {str(e)}"
            )


class UnitConverterTool(BaseTool):
    """Tool for converting between units."""
    
    @property
    def name(self) -> str:
        return "unit_converter"
    
    @property
    def description(self) -> str:
        return """Convert between different units of measurement.

Supported conversions:
- Length: meters, feet, inches, miles, kilometers
- Weight: kilograms, pounds, ounces, grams
- Temperature: celsius, fahrenheit, kelvin
- Data: bytes, kilobytes, megabytes, gigabytes"""
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "value": {
                    "type": "number",
                    "description": "The value to convert"
                },
                "from_unit": {
                    "type": "string",
                    "description": "Source unit"
                },
                "to_unit": {
                    "type": "string",
                    "description": "Target unit"
                }
            },
            "required": ["value", "from_unit", "to_unit"]
        }
    
    async def execute(
        self, 
        value: float, 
        from_unit: str, 
        to_unit: str
    ) -> ToolResult:
        """Convert between units."""
        
        # Conversion factors to base units
        conversions = {
            # Length (base: meters)
            "meters": 1, "m": 1,
            "feet": 0.3048, "ft": 0.3048,
            "inches": 0.0254, "in": 0.0254,
            "miles": 1609.34, "mi": 1609.34,
            "kilometers": 1000, "km": 1000,
            
            # Weight (base: kilograms)
            "kilograms": 1, "kg": 1,
            "pounds": 0.453592, "lbs": 0.453592, "lb": 0.453592,
            "ounces": 0.0283495, "oz": 0.0283495,
            "grams": 0.001, "g": 0.001,
            
            # Data (base: bytes)
            "bytes": 1, "b": 1,
            "kilobytes": 1024, "kb": 1024,
            "megabytes": 1024**2, "mb": 1024**2,
            "gigabytes": 1024**3, "gb": 1024**3,
        }
        
        from_unit = from_unit.lower()
        to_unit = to_unit.lower()
        
        # Temperature requires special handling
        if from_unit in ["celsius", "c", "fahrenheit", "f", "kelvin", "k"]:
            return self._convert_temperature(value, from_unit, to_unit)
        
        try:
            if from_unit not in conversions or to_unit not in conversions:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Unknown unit. Supported: {list(conversions.keys())}"
                )
            
            # Convert to base unit, then to target
            base_value = value * conversions[from_unit]
            result = base_value / conversions[to_unit]
            
            return ToolResult(
                success=True,
                output=f"{value} {from_unit} = {result:.4f} {to_unit}"
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Conversion error: {str(e)}"
            )
    
    def _convert_temperature(
        self, 
        value: float, 
        from_unit: str, 
        to_unit: str
    ) -> ToolResult:
        """Convert temperature units."""
        
        # Normalize unit names
        unit_map = {
            "celsius": "c", "c": "c",
            "fahrenheit": "f", "f": "f",
            "kelvin": "k", "k": "k"
        }
        
        from_u = unit_map.get(from_unit)
        to_u = unit_map.get(to_unit)
        
        if not from_u or not to_u:
            return ToolResult(
                success=False,
                output="",
                error="Invalid temperature unit"
            )
        
        # Convert to Celsius first
        if from_u == "c":
            celsius = value
        elif from_u == "f":
            celsius = (value - 32) * 5/9
        else:  # kelvin
            celsius = value - 273.15
        
        # Convert from Celsius to target
        if to_u == "c":
            result = celsius
        elif to_u == "f":
            result = celsius * 9/5 + 32
        else:  # kelvin
            result = celsius + 273.15
        
        return ToolResult(
            success=True,
            output=f"{value}° {from_unit} = {result:.2f}° {to_unit}"
        )
```

## Part 2: Building the Agent

### Step 3: Create the Tool Executor

```python
# agent.py
import asyncio
import json
from typing import Dict, List, Any, Optional
from openai import AsyncOpenAI
from dotenv import load_dotenv

from tools import BaseTool, ToolResult

load_dotenv()


class ToolExecutor:
    """Manages and executes tools."""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
    
    def register(self, tool: BaseTool):
        """Register a tool."""
        self.tools[tool.name] = tool
        print(f"✓ Registered tool: {tool.name}")
    
    def get_schemas(self) -> List[Dict]:
        """Get OpenAI schemas for all tools."""
        return [tool.to_openai_schema() for tool in self.tools.values()]
    
    async def execute(
        self, 
        tool_name: str, 
        arguments: Dict[str, Any]
    ) -> ToolResult:
        """Execute a tool by name."""
        
        if tool_name not in self.tools:
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown tool: {tool_name}"
            )
        
        tool = self.tools[tool_name]
        
        try:
            return await asyncio.wait_for(
                tool.execute(**arguments),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            return ToolResult(
                success=False,
                output="",
                error=f"Tool {tool_name} timed out"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output="",
                error=f"Tool execution error: {str(e)}"
            )
```

### Step 4: Create the Agent Class

```python
# agent.py (continued)

class FunctionCallingAgent:
    """Agent that uses function calling to complete tasks."""
    
    def __init__(
        self,
        model: str = "gpt-4",
        max_iterations: int = 10
    ):
        self.client = AsyncOpenAI()
        self.model = model
        self.max_iterations = max_iterations
        self.executor = ToolExecutor()
        
        self.system_prompt = """You are a helpful AI assistant with access to various tools.

When a user asks a question or requests a task:
1. Think about what information or actions you need
2. Use the appropriate tools to gather information or take action
3. Provide a clear, helpful response based on the results

Always be accurate and cite your sources when using search results.
If a tool fails, try an alternative approach or explain the limitation."""
    
    def register_tool(self, tool: BaseTool):
        """Register a tool with the agent."""
        self.executor.register(tool)
    
    async def run(self, user_message: str) -> str:
        """Process a user message and return a response."""
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        tools = self.executor.get_schemas()
        
        for iteration in range(self.max_iterations):
            # Get response from LLM
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools if tools else None,
                tool_choice="auto" if tools else None
            )
            
            message = response.choices[0].message
            
            # If no tool calls, we're done
            if not message.tool_calls:
                return message.content
            
            # Process tool calls
            messages.append(message)
            
            print(f"\n--- Iteration {iteration + 1} ---")
            
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                
                print(f"🔧 Calling: {function_name}")
                print(f"   Args: {arguments}")
                
                # Execute the tool
                result = await self.executor.execute(function_name, arguments)
                
                if result.success:
                    print(f"   ✓ Success: {result.output[:100]}...")
                    content = result.output
                else:
                    print(f"   ✗ Error: {result.error}")
                    content = f"Error: {result.error}"
                
                # Add result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": content
                })
        
        return "I've reached the maximum number of steps. Here's what I found so far based on the conversation."
    
    async def run_interactive(self):
        """Run an interactive chat session."""
        
        print("\n🤖 Function Calling Agent")
        print("=" * 50)
        print("Available tools:", list(self.executor.tools.keys()))
        print("Type 'quit' to exit\n")
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            try:
                response = await self.run(user_input)
                print(f"\nAssistant: {response}")
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
```

## Part 3: Putting It All Together

### Step 5: Main Application

```python
# main.py
import asyncio
from agent import FunctionCallingAgent
from tools import (
    CalculatorTool,
    WebSearchTool,
    DateTimeTool,
    UnitConverterTool
)


async def main():
    """Run the function calling agent."""
    
    # Create the agent
    agent = FunctionCallingAgent(model="gpt-4")
    
    # Register tools
    agent.register_tool(CalculatorTool())
    agent.register_tool(WebSearchTool())
    agent.register_tool(DateTimeTool())
    agent.register_tool(UnitConverterTool())
    
    # Run interactive mode
    await agent.run_interactive()


async def demo():
    """Run demonstration queries."""
    
    agent = FunctionCallingAgent(model="gpt-4")
    
    # Register tools
    agent.register_tool(CalculatorTool())
    agent.register_tool(WebSearchTool())
    agent.register_tool(DateTimeTool())
    agent.register_tool(UnitConverterTool())
    
    # Test queries
    queries = [
        "What is 15% of 847?",
        "Convert 72 degrees Fahrenheit to Celsius",
        "What day of the week is today?",
        "Search for the capital of Australia and calculate its distance from Sydney (approximately 248km) in miles"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("="*60)
        
        response = await agent.run(query)
        print(f"\nFinal Answer: {response}")


if __name__ == "__main__":
    # Choose one:
    asyncio.run(demo())      # Run demo queries
    # asyncio.run(main())    # Run interactive mode
```

## Part 4: Testing Your Agent

### Test Cases

```python
# test_agent.py
import asyncio
import pytest
from agent import FunctionCallingAgent
from tools import CalculatorTool, DateTimeTool, UnitConverterTool

@pytest.fixture
def agent():
    """Create a test agent."""
    agent = FunctionCallingAgent(model="gpt-4")
    agent.register_tool(CalculatorTool())
    agent.register_tool(DateTimeTool())
    agent.register_tool(UnitConverterTool())
    return agent


@pytest.mark.asyncio
async def test_calculator(agent):
    """Test calculator functionality."""
    response = await agent.run("What is 25 * 4?")
    assert "100" in response


@pytest.mark.asyncio
async def test_unit_conversion(agent):
    """Test unit conversion."""
    response = await agent.run("Convert 100 kilometers to miles")
    assert "62" in response  # ~62.14 miles


@pytest.mark.asyncio
async def test_multi_tool(agent):
    """Test using multiple tools."""
    response = await agent.run(
        "What is today's date and how many days until December 31, 2025?"
    )
    # Should use datetime tool twice
    assert "days" in response.lower()


@pytest.mark.asyncio  
async def test_error_handling(agent):
    """Test error handling."""
    response = await agent.run("Calculate the square root of -1")
    # Should handle math domain error gracefully
    assert "error" in response.lower() or "cannot" in response.lower()
```

## Exercises

### Exercise 1: Add a Weather Tool

Create a new tool that fetches weather information:

```python
class WeatherTool(BaseTool):
    """Tool for getting weather information."""
    
    @property
    def name(self) -> str:
        return "weather"
    
    # Implement the rest...
```

### Exercise 2: Add Tool Result Caching

Implement caching to avoid redundant tool calls:

```python
class CachedToolExecutor(ToolExecutor):
    def __init__(self, cache_ttl: int = 300):
        super().__init__()
        self.cache = {}
        self.cache_ttl = cache_ttl
    
    # Implement caching logic...
```

### Exercise 3: Add Conversation Memory

Extend the agent to remember previous conversations:

```python
class ConversationalAgent(FunctionCallingAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conversation_history = []
    
    # Implement conversation persistence...
```

## Summary

In this lab, you built:

1. **Base Tool Class**: Abstract class for creating tools
2. **Multiple Tools**: Calculator, web search, datetime, unit converter
3. **Tool Executor**: Manages and executes tools safely
4. **Function Calling Agent**: Integrates LLM with tools
5. **Interactive Mode**: Chat interface for testing

Key takeaways:
- Tools need clear schemas for LLM understanding
- Error handling is crucial for robustness
- The agent loop handles multiple tool calls iteratively
- Parallel tool calls can improve efficiency

Next: [Agent Loop Implementation](/learn/agents/agent-loop/loop-basics) →
