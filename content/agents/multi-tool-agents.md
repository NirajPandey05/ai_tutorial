# Multi-Tool Agents

Learn how to build agents that intelligently combine multiple tools to solve complex tasks.

## Why Multi-Tool Agents?

Real-world tasks often require combining different capabilities. Multi-tool agents can orchestrate various tools together.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Multi-Tool Agent Architecture                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   User Task: "Analyze my company's competitors and create a report"     │
│                                                                          │
│                         ┌───────────────┐                               │
│                         │   LLM Brain   │                               │
│                         │   (Planner)   │                               │
│                         └───────┬───────┘                               │
│                                 │                                        │
│   ┌─────────────────────────────┼─────────────────────────────┐         │
│   │                             │                             │         │
│   ▼                             ▼                             ▼         │
│ ┌─────────┐             ┌─────────────┐              ┌─────────────┐   │
│ │ Web     │             │  Database   │              │   Code      │   │
│ │ Search  │             │   Query     │              │  Executor   │   │
│ └────┬────┘             └──────┬──────┘              └──────┬──────┘   │
│      │                         │                            │          │
│      │    ┌────────────────────┼────────────────────┐      │          │
│      │    │                    │                    │      │          │
│      ▼    ▼                    ▼                    ▼      ▼          │
│ ┌──────────────┐      ┌──────────────┐      ┌──────────────┐         │
│ │ Page Reader  │      │  Calculator  │      │ File Writer  │         │
│ └──────────────┘      └──────────────┘      └──────────────┘         │
│                                                                        │
│   Result: Comprehensive competitive analysis with charts and data     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Tool Integration Patterns

### Pattern 1: Sequential Tool Chain

```python
async def sequential_analysis(agent, topic: str):
    """Tools used in sequence, each building on the previous."""
    
    # Step 1: Search for information
    search_results = await agent.use_tool(
        "web_search",
        query=f"{topic} overview statistics"
    )
    
    # Step 2: Read top results
    content = []
    for result in search_results[:3]:
        page = await agent.use_tool(
            "read_webpage",
            url=result["url"]
        )
        content.append(page)
    
    # Step 3: Extract data and analyze
    analysis = await agent.use_tool(
        "execute_python",
        code=f"""
import json
data = {json.dumps(content)}
# Analyze and summarize
summary = analyze_content(data)
print(json.dumps(summary))
"""
    )
    
    return analysis
```

### Pattern 2: Parallel Tool Execution

```python
import asyncio

async def parallel_research(agent, topics: list[str]):
    """Execute multiple tool calls in parallel."""
    
    async def research_topic(topic: str):
        results = await agent.use_tool(
            "web_search",
            query=topic
        )
        
        if results:
            content = await agent.use_tool(
                "read_webpage",
                url=results[0]["url"]
            )
            return {"topic": topic, "content": content}
        return {"topic": topic, "content": None}
    
    # Run all research in parallel
    tasks = [research_topic(t) for t in topics]
    results = await asyncio.gather(*tasks)
    
    return results
```

### Pattern 3: Conditional Tool Selection

```python
class ConditionalToolAgent:
    """Agent that selects tools based on task type."""
    
    def __init__(self, tools: dict):
        self.tools = tools
        self.tool_selector = {
            "calculation": ["calculator", "python_executor"],
            "research": ["web_search", "read_webpage", "wikipedia"],
            "data": ["database", "python_executor", "file_reader"],
            "communication": ["email", "slack", "calendar"],
            "code": ["python_executor", "file_writer", "git"]
        }
    
    async def solve(self, task: str, task_type: str) -> Any:
        """Solve task using appropriate tools."""
        
        available_tools = self.tool_selector.get(task_type, [])
        
        # Filter to only available tools
        tools_to_use = {
            name: self.tools[name]
            for name in available_tools
            if name in self.tools
        }
        
        return await self._execute_with_tools(task, tools_to_use)
```

## Building a Comprehensive Multi-Tool Agent

```python
from openai import AsyncOpenAI
from typing import Dict, Any, List, Optional
import json
import asyncio

class MultiToolAgent:
    """Agent that can use multiple tools to solve complex tasks."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        max_iterations: int = 15
    ):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.max_iterations = max_iterations
        self.tools: Dict[str, Any] = {}
        self.tool_schemas: List[dict] = []
    
    def register_tool(
        self,
        name: str,
        description: str,
        parameters: dict,
        handler: callable
    ):
        """Register a tool with the agent."""
        
        self.tools[name] = {
            "handler": handler,
            "description": description
        }
        
        self.tool_schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters
            }
        })
    
    def build_system_prompt(self) -> str:
        """Build system prompt with all tool descriptions."""
        
        tool_list = "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.tools.items()
        ])
        
        return f"""You are a capable AI agent with access to multiple tools.

Available Tools:
{tool_list}

Strategy for Complex Tasks:
1. Break down the task into subtasks
2. Identify which tools are needed for each subtask
3. Execute tools in logical order
4. Combine results to form final answer
5. Handle errors by trying alternative approaches

You can call multiple tools in a single response when they are independent.
Always verify your results before providing a final answer."""
    
    async def run(self, task: str, context: str = "") -> dict:
        """Run the agent on a task."""
        
        system_prompt = self.build_system_prompt()
        
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        if context:
            messages.append({
                "role": "user",
                "content": f"Context:\n{context}"
            })
        
        messages.append({
            "role": "user",
            "content": f"Task: {task}"
        })
        
        tool_usage_log = []
        
        for iteration in range(self.max_iterations):
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tool_schemas if self.tool_schemas else None,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            
            # Check if done
            if not message.tool_calls:
                return {
                    "success": True,
                    "answer": message.content,
                    "tool_usage": tool_usage_log,
                    "iterations": iteration + 1
                }
            
            messages.append(message)
            
            # Execute all tool calls (potentially in parallel)
            tool_results = await self._execute_tools_parallel(
                message.tool_calls
            )
            
            for tool_call, result in tool_results:
                tool_usage_log.append({
                    "tool": tool_call.function.name,
                    "args": json.loads(tool_call.function.arguments),
                    "result": result
                })
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })
        
        return {
            "success": False,
            "answer": "Max iterations reached",
            "tool_usage": tool_usage_log,
            "iterations": self.max_iterations
        }
    
    async def _execute_tools_parallel(
        self,
        tool_calls: list
    ) -> list[tuple]:
        """Execute multiple tool calls in parallel."""
        
        async def execute_one(tool_call):
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            
            if name not in self.tools:
                result = {"error": f"Unknown tool: {name}"}
            else:
                try:
                    handler = self.tools[name]["handler"]
                    if asyncio.iscoroutinefunction(handler):
                        result = await handler(**args)
                    else:
                        result = handler(**args)
                except Exception as e:
                    result = {"error": str(e)}
            
            return tool_call, result
        
        tasks = [execute_one(tc) for tc in tool_calls]
        return await asyncio.gather(*tasks)
```

## Example: Research and Analysis Agent

```python
# Register diverse tools for research and analysis

agent = MultiToolAgent(api_key="...", model="gpt-4")

# Web tools
agent.register_tool(
    name="web_search",
    description="Search the web for information",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "num_results": {"type": "integer", "default": 5}
        },
        "required": ["query"]
    },
    handler=web_search_function
)

agent.register_tool(
    name="read_webpage",
    description="Read content from a URL",
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string"}
        },
        "required": ["url"]
    },
    handler=read_webpage_function
)

# Data tools
agent.register_tool(
    name="execute_python",
    description="Execute Python code for data analysis",
    parameters={
        "type": "object",
        "properties": {
            "code": {"type": "string"}
        },
        "required": ["code"]
    },
    handler=python_executor
)

agent.register_tool(
    name="create_chart",
    description="Create a chart from data",
    parameters={
        "type": "object",
        "properties": {
            "data": {"type": "object"},
            "chart_type": {"type": "string", "enum": ["bar", "line", "pie"]},
            "title": {"type": "string"}
        },
        "required": ["data", "chart_type"]
    },
    handler=chart_creator
)

# File tools
agent.register_tool(
    name="write_file",
    description="Write content to a file",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"}
        },
        "required": ["path", "content"]
    },
    handler=write_file_function
)

agent.register_tool(
    name="read_file",
    description="Read content from a file",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string"}
        },
        "required": ["path"]
    },
    handler=read_file_function
)

# Example usage
result = await agent.run("""
Research the top 5 electric vehicle companies by market share in 2024.
For each company:
1. Find their current market share
2. Find their stock price trend
3. Find recent news

Create a summary report with:
- A comparison table
- A bar chart of market shares
- Key insights

Save the report as 'ev_analysis.md'
""")
```

## Tool Composition Strategies

### 1. Pipeline Pattern

```python
class PipelineAgent:
    """Execute tools in a defined pipeline."""
    
    async def run_pipeline(
        self,
        input_data: Any,
        pipeline: List[dict]
    ) -> Any:
        """Run a series of tools in sequence."""
        
        current_data = input_data
        
        for step in pipeline:
            tool_name = step["tool"]
            arg_mapping = step.get("args", {})
            
            # Map current data to tool arguments
            args = {}
            for arg_name, source in arg_mapping.items():
                if source == "$input":
                    args[arg_name] = current_data
                elif source.startswith("$result."):
                    path = source[8:]  # Remove "$result."
                    args[arg_name] = self._get_nested(current_data, path)
                else:
                    args[arg_name] = source
            
            current_data = await self.execute_tool(tool_name, args)
        
        return current_data


# Example pipeline
pipeline = [
    {
        "tool": "web_search",
        "args": {"query": "$input"}
    },
    {
        "tool": "read_webpage",
        "args": {"url": "$result.0.url"}
    },
    {
        "tool": "summarize",
        "args": {"text": "$result.content"}
    }
]
```

### 2. Router Pattern

```python
class RouterAgent:
    """Route tasks to specialized sub-agents."""
    
    def __init__(self):
        self.routes = {}
    
    def add_route(
        self,
        pattern: str,
        handler: callable,
        tools: List[str]
    ):
        """Add a route for a task pattern."""
        self.routes[pattern] = {
            "handler": handler,
            "tools": tools
        }
    
    async def route(self, task: str) -> Any:
        """Route task to appropriate handler."""
        
        # Use LLM to classify task
        classification = await self.classify_task(task)
        
        if classification in self.routes:
            route = self.routes[classification]
            return await route["handler"](task, route["tools"])
        
        # Default handling
        return await self.default_handler(task)


# Example routes
router = RouterAgent()

router.add_route(
    pattern="research",
    handler=research_handler,
    tools=["web_search", "read_webpage", "summarize"]
)

router.add_route(
    pattern="code",
    handler=code_handler,
    tools=["execute_python", "read_file", "write_file"]
)

router.add_route(
    pattern="data_analysis",
    handler=analysis_handler,
    tools=["database", "execute_python", "create_chart"]
)
```

### 3. Fallback Pattern

```python
class FallbackAgent:
    """Try tools in order until one succeeds."""
    
    async def execute_with_fallbacks(
        self,
        task: str,
        tool_sequence: List[str]
    ) -> Any:
        """Try tools in sequence until success."""
        
        last_error = None
        
        for tool_name in tool_sequence:
            try:
                result = await self.execute_tool(tool_name, {"task": task})
                
                if result.get("success", True):
                    return result
                
                last_error = result.get("error")
            except Exception as e:
                last_error = str(e)
        
        return {
            "success": False,
            "error": f"All tools failed. Last error: {last_error}"
        }


# Example: Try multiple search providers
result = await agent.execute_with_fallbacks(
    task="Find information about quantum computing",
    tool_sequence=["google_search", "bing_search", "duckduckgo_search"]
)
```

## Handling Tool Dependencies

```python
class DependencyAwareAgent:
    """Agent that understands tool dependencies."""
    
    def __init__(self):
        self.dependencies = {
            "analyze_data": ["load_data"],
            "create_chart": ["analyze_data"],
            "send_report": ["create_chart", "write_report"],
        }
    
    async def execute_with_dependencies(
        self,
        target_tool: str,
        args: dict
    ) -> Any:
        """Execute tool and all its dependencies."""
        
        execution_order = self._resolve_dependencies(target_tool)
        results = {}
        
        for tool in execution_order:
            tool_args = self._prepare_args(tool, args, results)
            results[tool] = await self.execute_tool(tool, tool_args)
        
        return results[target_tool]
    
    def _resolve_dependencies(self, tool: str) -> List[str]:
        """Resolve dependencies in topological order."""
        visited = set()
        order = []
        
        def visit(t):
            if t in visited:
                return
            visited.add(t)
            
            for dep in self.dependencies.get(t, []):
                visit(dep)
            
            order.append(t)
        
        visit(tool)
        return order
```

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   Multi-Tool Agents - Summary                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Integration Patterns:                                                   │
│    • Sequential: Tools in series, each builds on previous              │
│    • Parallel: Independent tools run concurrently                      │
│    • Conditional: Select tools based on task type                      │
│                                                                          │
│  Composition Strategies:                                                 │
│    • Pipeline: Predefined tool sequences                                │
│    • Router: Task classification to specialized handlers               │
│    • Fallback: Try alternatives on failure                             │
│                                                                          │
│  Best Practices:                                                         │
│    • Register tools with clear descriptions                            │
│    • Handle tool failures gracefully                                   │
│    • Use parallel execution when possible                              │
│    • Log tool usage for debugging                                      │
│    • Manage tool dependencies                                          │
│                                                                          │
│  Key Consideration:                                                      │
│    The LLM orchestrates tools - good prompting is crucial              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

Next: [Research Agent Lab](/learn/agents/agent-capabilities/research-agent-lab) →
